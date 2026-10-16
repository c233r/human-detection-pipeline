"""
EdgeCrafter API 客户端示例
展示如何调用 EdgeCrafter 推理服务

使用方法:
    python client_example.py

或者在 Python 中导入使用:
    from client_example import EdgeCrafterClient
    client = EdgeCrafterClient("http://localhost:8000")
    result = client.upload_video("test.mp4")
"""

import os
import sys
import json
import time
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse


@dataclass
class Detection:
    """检测结果数据类"""
    bbox: List[int]
    score: float
    category_id: int
    category_name: str
    
    def __str__(self):
        return f"[{self.category_name}] {self.bbox} (置信度: {self.score:.2f})"


@dataclass
class TaskInfo:
    """任务信息数据类"""
    task_id: str
    filename: str
    status: str
    progress: int
    message: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    processing_time: Optional[float] = None
    total_frames: Optional[int] = None
    processed_frames: Optional[int] = None
    total_detections: Optional[int] = None
    output_dir: Optional[str] = None


class EdgeCrafterClient:
    """
    EdgeCrafter API 客户端
    
    提供简洁的接口来调用 EdgeCrafter 推理服务
    服务通过外部脚本执行推理和抽帧（异步队列处理）
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 300):
        """
        初始化客户端
        
        Args:
            base_url: API 服务地址
            timeout: 请求超时时间（秒）
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'EdgeCrafter-Client/2.0'
        })
    
    def _request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """发送请求的辅助方法"""
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.request(method, url, timeout=self.timeout, **kwargs)
            response.raise_for_status()
            return response
        except requests.exceptions.ConnectionError:
            print("❌ 无法连接到服务器，请确认服务已启动")
            raise
        except requests.exceptions.Timeout:
            print("⏰ 请求超时")
            raise
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 503:
                raise  # 队列满，交给上层 retry 逻辑处理
            # 其他 HTTP 错误直接显示
            status = e.response.status_code if e.response is not None else "?"
            print(f"❌ 服务器返回错误 (HTTP {status}): {e.response.reason if e.response is not None else 'Unknown'}")
            raise
    
    # ==================== 健康检查 ====================
    def health_check(self) -> Dict[str, Any]:
        """
        健康检查
        
        Returns:
            服务状态信息
        """
        response = self._request('GET', '/health')
        return response.json()
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取服务状态信息
        
        Returns:
            服务状态和配置信息
        """
        response = self._request('GET', '/')
        return response.json()
    
    # ==================== 视频推理（异步）====================
    def upload_video(
        self, 
        video_path: str, 
        conf_threshold: float = 0.5,
        queue_retry: bool = True,
        retry_interval: float = 600.0
    ) -> Dict[str, Any]:
        """
        上传视频到队列（异步处理）
        
        视频会被添加到队列中等待处理，返回任务ID，
        可以通过 get_task_status() 查询处理进度。
        队列满时自动等待重试（不限制重试次数）。
        
        Args:
            video_path: 视频文件路径
            conf_threshold: 置信度阈值
            queue_retry: 队列满时是否自动重试
            retry_interval: 重试间隔（秒），默认10分钟
        
        Returns:
            任务信息，包含 task_id 和队列位置
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        attempt = 0
        while True:
            attempt += 1
            try:
                with open(video_path, 'rb') as f:
                    files = {'file': f}
                    params = {'conf_threshold': conf_threshold}
                    response = self._request('POST', '/infer/video', files=files, params=params)
                return response.json()
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 503 and queue_retry:
                    detail = e.response.json()
                    queue_info = detail.get("detail", detail)
                    wait_time = retry_interval
                    print(f"\n  ⏳ 队列已满 ({queue_info.get('queue_size', '?')}/{queue_info.get('max_queue_size', '?')})")
                    print(f"    活跃任务: {queue_info.get('active_tasks', '?')}, 等待: {queue_info.get('pending_tasks', '?')}")
                    print(f"    等待 {wait_time:.0f} 秒后重试 (第{attempt}次)...")
                    time.sleep(wait_time)
                else:
                    raise
    
    def get_task_status(self, task_id: str) -> TaskInfo:
        """
        查询任务状态
        
        Args:
            task_id: 任务ID
        
        Returns:
            任务状态信息
        """
        response = self._request('GET', f'/tasks/{task_id}')
        data = response.json()
        
        return TaskInfo(
            task_id=data['task_id'],
            filename=data['filename'],
            status=data['status'],
            progress=data['progress'],
            message=data['message'],
            created_at=data['created_at'],
            started_at=data.get('started_at'),
            completed_at=data.get('completed_at'),
            processing_time=data.get('processing_time'),
            total_frames=data.get('total_frames'),
            processed_frames=data.get('processed_frames'),
            total_detections=data.get('total_detections'),
            output_dir=data.get('output_dir')
        )
    
    def wait_for_completion(
        self, 
        task_id: str, 
        poll_interval: float = 2.0,
        timeout: float = 300.0
    ) -> TaskInfo:
        """
        等待任务完成（阻塞直到任务完成或超时）
        
        Args:
            task_id: 任务ID
            poll_interval: 轮询间隔（秒）
            timeout: 超时时间（秒）
        
        Returns:
            最终的任务状态信息
        
        Raises:
            TimeoutError: 任务超时
        """
        start_time = time.time()
        
        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"任务超时，已等待 {timeout:.1f} 秒")
            
            task = self.get_task_status(task_id)
            
            print(f"⏳ [{task.progress}%] {task.message}")
            
            if task.status in ['completed', 'failed']:
                return task
            
            time.sleep(poll_interval)
    
    def list_tasks(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        列出所有任务
        
        Args:
            status: 按状态过滤 (pending/processing/completed/failed)
            limit: 返回数量
            offset: 偏移量
        
        Returns:
            任务列表
        """
        params = {
            'limit': limit,
            'offset': offset
        }
        if status:
            params['status'] = status
        
        response = self._request('GET', '/tasks', params=params)
        return response.json()
    
    def delete_task(self, task_id: str) -> Dict[str, Any]:
        """
        删除任务记录
        
        Args:
            task_id: 任务ID
        
        Returns:
            删除结果
        """
        response = self._request('DELETE', f'/tasks/{task_id}')
        return response.json()
    
    # ==================== 队列状态 ====================
    def get_queue_status(self) -> Dict[str, Any]:
        """
        获取队列状态
        
        Returns:
            队列状态信息
        """
        response = self._request('GET', '/queue/status')
        return response.json()
    
    # ==================== 帧目录推理（同步）====================
    def infer_frame_directory(
        self,
        frame_dir: str,
        conf_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        帧目录推理 - 对服务器端的帧目录进行批量推理
        
        兼容命令行模式：
        CUDA_VISIBLE_DEVICES=5 python tools/inference/torch_inf.py -c configs/ecdet/ecdet_l.yml -r /path/to/best.pth -i /path/to/frames -t 0.5
        
        Args:
            frame_dir: 服务器端的帧目录绝对路径
            conf_threshold: 置信度阈值
        
        Returns:
            帧目录推理结果
        """
        # 注意：原有的帧目录推理接口已移除，因为现在主要使用异步视频上传
        # 如果需要帧目录推理，可以使用 /infer/frame-dir 端点
        # 这里保留这个方法以便向后兼容
        data = {
            'frame_dir': frame_dir,
            'conf_threshold': conf_threshold
        }
        try:
            response = self._request('POST', '/infer/frame-dir', data=data)
            return response.json()
        except requests.exceptions.HTTPError as e:
            print(f"帧目录推理接口不可用: {e}")
            return {"error": "帧目录推理接口已移除，请使用视频上传接口"}


# ==================== 使用示例 ====================
def main():
    """主函数 - 演示各种用法"""
    
    # 创建客户端
    client = EdgeCrafterClient("http://localhost:8000")
    
    print("=" * 60)
    print("   EdgeCrafter API 客户端示例 v2.0")
    print("=" * 60)
    
    # 1. 健康检查
    print("\n【1. 健康检查】")
    try:
        health = client.health_check()
        print(f"   服务状态: {'✅ 正常' if health.get('status') == 'healthy' else '⚠️ 异常'}")
        print(f"   推理脚本: {'✅ 存在' if health.get('inference_script_exists') else '❌ 缺失'}")
        print(f"   抽帧脚本: {'✅ 存在' if health.get('extract_frames_script_exists') else '❌ 缺失'}")
        print(f"   配置文件: {'✅ 存在' if health.get('config_exists') else '❌ 缺失'}")
        print(f"   模型权重: {'✅ 存在' if health.get('checkpoint_exists') else '❌ 缺失'}")
    except Exception as e:
        print(f"   ❌ 服务不可用: {e}")
        print("   请确保 API 服务已启动: python api_service.py")
        return
    
    # 2. 获取服务状态
    print("\n【2. 服务状态】")
    try:
        status = client.get_status()
        print(f"   服务名称: {status.get('service')}")
        print(f"   版本: {status.get('version')}")
        print(f"   类别: {status.get('config', {}).get('classes')}")
        print(f"   置信度阈值: {status.get('config', {}).get('conf_threshold')}")
        print(f"   最大队列大小: {status.get('config', {}).get('max_queue_size')}")
    except Exception as e:
        print(f"   获取状态失败: {e}")
    
    # 3. 队列状态
    print("\n【3. 队列状态】")
    try:
        queue_status = client.get_queue_status()
        print(f"   队列大小: {queue_status.get('queue_size', 0)}/{queue_status.get('max_queue_size', 0)}")
        print(f"   活跃任务: {queue_status.get('active_tasks', 0)}")
        print(f"   等待任务: {queue_status.get('pending_tasks', 0)}")
    except Exception as e:
        print(f"   获取队列状态失败: {e}")
    
    # 4. 视频上传示例（异步）
    print("\n【4. 视频上传（异步）】")
    test_video = "/data2/caimoyao/test_video.mp4"
    
    if os.path.exists(test_video):
        try:
            print(f"   正在上传视频: {test_video}")
            result = client.upload_video(test_video, conf_threshold=0.5)
            
            task_id = result['task_id']
            print(f"   ✅ 上传成功")
            print(f"   任务ID: {task_id}")
            print(f"   队列位置: {result.get('position_in_queue', 0)}")
            
            # 等待任务完成
            print("\n   等待处理完成...")
            final_status = client.wait_for_completion(task_id, poll_interval=3.0)
            
            print(f"\n   处理结果:")
            print(f"   状态: {final_status.status}")
            print(f"   总帧数: {final_status.total_frames}")
            print(f"   处理帧数: {final_status.processed_frames}")
            print(f"   总检测数: {final_status.total_detections}")
            print(f"   处理时间: {final_status.processing_time:.2f}s")
            print(f"   输出目录: {final_status.output_dir}")
            
        except TimeoutError:
            print(f"   ⏰ 任务超时")
        except Exception as e:
            print(f"   ❌ 上传失败: {e}")
    else:
        print(f"   ⚠️ 测试视频不存在: {test_video}")
    
    # 5. 列出最近任务
    print("\n【5. 最近任务】")
    try:
        tasks = client.list_tasks(limit=5)
        print(f"   最近 {len(tasks.get('results', []))} 个任务:")
        for r in tasks.get('results', []):
            status_icon = {
                'completed': '✅',
                'failed': '❌',
                'processing': '🔄',
                'pending': '⏳'
            }.get(r.get('status'), '❓')
            print(f"     {status_icon} {r.get('filename', 'unknown')}: "
                  f"{r.get('total_detections', 0)} 个检测")
    except Exception as e:
        print(f"   ❌ 获取任务列表失败: {e}")
    
    print("\n" + "=" * 60)
    print("   示例完成")
    print("=" * 60)


# ==================== 命令行接口 ====================
def cli():
    """命令行接口"""
    parser = argparse.ArgumentParser(
        description="EdgeCrafter API 客户端 v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 健康检查
  python client_example.py --health

  # 获取服务状态
  python client_example.py --status

  # 获取队列状态
  python client_example.py --queue-status

  # 上传视频（异步）
  python client_example.py --upload /path/to/video.mp4

  # 上传视频并等待完成
  python client_example.py --upload /path/to/video.mp4 --wait

  # 上传文件夹中所有视频
  python client_example.py --upload-dir /path/to/videos

  # 上传文件夹中前5个视频
  python client_example.py --upload-dir /path/to/videos --limit 5

  # 查询任务状态
  python client_example.py --task-status <task_id>

  # 列出任务
  python client_example.py --list-tasks

  # 删除任务
  python client_example.py --delete-task <task_id>
        """
    )
    
    parser.add_argument('--url', default='http://localhost:8000', help='API 服务地址')
    parser.add_argument('--health', action='store_true', help='健康检查')
    parser.add_argument('--status', action='store_true', help='获取服务状态')
    parser.add_argument('--queue-status', action='store_true', help='获取队列状态')
    parser.add_argument('--upload', type=str, help='上传视频文件（异步）')
    parser.add_argument('--upload-dir', type=str, help='上传文件夹中所有视频')
    parser.add_argument('--ext', type=str, default='.mp4,.avi,.mov,.mkv,.webm',
                        help='视频文件扩展名（逗号分隔，默认: .mp4,.avi,.mov,.mkv,.webm）')
    parser.add_argument('--parallel', type=int, default=1,
                        help='并行上传数（默认: 1）')
    parser.add_argument('--limit', type=int, default=0,
                        help='限制上传视频数量（默认: 0，不限制）')
    parser.add_argument('--wait-timeout', type=float, default=3600,
                        help='等待超时时间（秒，默认: 3600）')
    parser.add_argument('--wait', action='store_true', help='上传后等待处理完成')
    parser.add_argument('--delete-original', action='store_true', help='任务完成后删除原视频文件')
    parser.add_argument('--task-status', type=str, help='查询任务状态')
    parser.add_argument('--list-tasks', action='store_true', help='列出所有任务')
    parser.add_argument('--delete-task', type=str, help='删除任务记录')
    parser.add_argument('--conf-threshold', type=float, default=0.5, help='置信度阈值')
    parser.add_argument('--poll-interval', type=float, default=2.0, help='轮询间隔（秒）')
    parser.add_argument('--no-queue-retry', action='store_true',
                        help='队列满时不等待重试，直接报错')
    parser.add_argument('--queue-retry-interval', type=float, default=600.0,
                        help='队列满的重试间隔（秒，默认: 600）')
    
    args = parser.parse_args()
    
    client = EdgeCrafterClient(args.url)
    
    if args.health:
        print(json.dumps(client.health_check(), indent=2))
    
    elif args.status:
        print(json.dumps(client.get_status(), indent=2))
    
    elif args.queue_status:
        print(json.dumps(client.get_queue_status(), indent=2))
    
    elif args.upload:
        result = client.upload_video(
            args.upload,
            conf_threshold=args.conf_threshold,
            queue_retry=not args.no_queue_retry,
            retry_interval=args.queue_retry_interval
        )
        print(json.dumps(result, indent=2))
        
        if args.wait:
            print("\n等待处理完成...")
            task_id = result['task_id']
            try:
                final_status = client.wait_for_completion(task_id, poll_interval=args.poll_interval)
                print("\n处理完成:")
                print(json.dumps({
                    "task_id": final_status.task_id,
                    "status": final_status.status,
                    "total_frames": final_status.total_frames,
                    "total_detections": final_status.total_detections,
                    "processing_time": final_status.processing_time,
                    "output_dir": final_status.output_dir
                }, indent=2))
            except TimeoutError:
                print("任务超时")
    
    elif args.upload_dir:
        if not os.path.isdir(args.upload_dir):
            print(f"❌ 文件夹不存在: {args.upload_dir}")
            return
        
        exts = [e.strip() for e in args.ext.split(',')]
        video_files = sorted([
            f for f in Path(args.upload_dir).iterdir()
            if f.is_file() and f.suffix.lower() in exts
        ])
        
        if not video_files:
            print(f"❌ 文件夹中没有找到视频文件（支持的扩展名: {args.ext}）")
            return
        
        if args.limit > 0 and len(video_files) > args.limit:
            video_files = video_files[:args.limit]
            print(f"找到 {len(video_files)} 个视频文件，限制上传前 {args.limit} 个...")
        else:
            print(f"找到 {len(video_files)} 个视频文件，开始上传...")
        
        def upload_one(video_path):
            try:
                result = client.upload_video(
                    str(video_path),
                    conf_threshold=args.conf_threshold,
                    queue_retry=not args.no_queue_retry,
                    retry_interval=args.queue_retry_interval
                )
                if args.delete_original:
                    try:
                        Path(str(video_path)).unlink()
                        print(f"     🗑️ 已删除原视频: {video_path.name}")
                    except Exception as de:
                        print(f"     ⚠️ 删除原视频失败: {video_path.name} | {de}")
                return (video_path.name, result)
            except Exception as e:
                return (video_path.name, {"error": str(e)})
        
        task_ids = []
        if args.parallel > 1:
            with ThreadPoolExecutor(max_workers=args.parallel) as pool:
                futures = {pool.submit(upload_one, v): v for v in video_files}
                for future in as_completed(futures):
                    name, result = future.result()
                    if "error" in result:
                        print(f"  ❌ {name}: {result['error']}")
                    else:
                        print(f"  ✅ {name}: task_id={result['task_id']}")
                        if args.wait:
                            task_ids.append((name, result['task_id']))
        else:
            for vf in video_files:
                print(f"\n[{video_files.index(vf)+1}/{len(video_files)}] 上传: {vf.name}")
                try:
                    result = client.upload_video(
                        str(vf),
                        conf_threshold=args.conf_threshold,
                        queue_retry=not args.no_queue_retry,
                        retry_interval=args.queue_retry_interval
                    )
                    print(f"  ✅ task_id={result['task_id']}")
                    if args.delete_original:
                        try:
                            Path(str(vf)).unlink()
                            print(f"     🗑️ 已删除原视频: {vf.name}")
                        except Exception as de:
                            print(f"     ⚠️ 删除原视频失败: {vf.name} | {de}")
                    if args.wait:
                        task_ids.append((vf.name, result['task_id']))
                except Exception as e:
                    print(f"  ❌ {e}")
        
        if args.wait and task_ids:
            print(f"\n等待 {len(task_ids)} 个任务完成...")
            for name, tid in task_ids:
                try:
                    final = client.wait_for_completion(tid, poll_interval=args.poll_interval,
                                                       timeout=args.wait_timeout)
                    print(f"  ✅ {name}: {final.status}, {final.total_detections} 个检测, "
                          f"{final.processing_time:.1f}s")
                except TimeoutError:
                    print(f"  ⏰ {name}: 超时")
                except Exception as e:
                    print(f"  ❌ {name}: {e}")
    
    elif args.task_status:
        task_info = client.get_task_status(args.task_status)
        print(json.dumps({
            "task_id": task_info.task_id,
            "filename": task_info.filename,
            "status": task_info.status,
            "progress": task_info.progress,
            "message": task_info.message,
            "total_frames": task_info.total_frames,
            "total_detections": task_info.total_detections,
            "output_dir": task_info.output_dir
        }, indent=2))
    
    elif args.list_tasks:
        results = client.list_tasks()
        print(json.dumps(results, indent=2))
    
    elif args.delete_task:
        result = client.delete_task(args.delete_task)
        print(result)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        cli()
    else:
        main()