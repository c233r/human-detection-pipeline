"""
数据预处理流水线示例
展示如何构建完整的自动化数据处理流程

功能:
1. 从数据源下载/读取图片
2. 调用 EdgeCrafter API 进行推理（通过外部脚本，异步队列处理）
3. 过滤和处理检测结果
4. 生成标注文件（YOLO/COCO 格式）
5. 数据集统计分析

使用方法:
    python data_pipeline.py --input-dir /path/to/images --output-dir /path/to/output
"""

import os
import sys
import json
import time
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import logging

# 导入客户端
from client_example import EdgeCrafterClient, TaskInfo

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== 数据模型 ====================
@dataclass
class BoundingBox:
    """边界框"""
    x1: int
    y1: int
    x2: int
    y2: int
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1
    
    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2
    
    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2
    
    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass
class DetectionResult:
    """检测结果"""
    filename: str
    image_width: int
    image_height: int
    detections: List[Dict[str, Any]]
    timestamp: str
    
    @property
    def count(self) -> int:
        return len(self.detections)
    
    def filter_by_category(self, categories: List[str]) -> 'DetectionResult':
        """按类别过滤"""
        filtered = [
            d for d in self.detections 
            if d.get('category_name') in categories
        ]
        return DetectionResult(
            filename=self.filename,
            image_width=self.image_width,
            image_height=self.image_height,
            detections=filtered,
            timestamp=self.timestamp
        )
    
    def filter_by_score(self, min_score: float) -> 'DetectionResult':
        """按置信度过滤"""
        filtered = [
            d for d in self.detections 
            if d.get('score', 0) >= min_score
        ]
        return DetectionResult(
            filename=self.filename,
            image_width=self.image_width,
            image_height=self.image_height,
            detections=filtered,
            timestamp=self.timestamp
        )


# ==================== 数据流水线 ====================
class DataPipeline:
    """
    数据预处理流水线
    
    通过调用外部脚本执行推理和抽帧（异步队列处理）
    """
    
    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        input_dir: str = "./input",
        output_dir: str = "./output",
        categories: Optional[List[str]] = None,
        min_score: float = 0.5
    ):
        """
        初始化流水线
        
        Args:
            api_url: API 服务地址
            input_dir: 输入图片目录
            output_dir: 输出目录
            categories: 保留的类别列表（None 表示保留所有）
            min_score: 最小置信度阈值
        """
        self.client = EdgeCrafterClient(api_url)
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.categories = categories
        self.min_score = min_score
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "labels").mkdir(exist_ok=True)
        (self.output_dir / "results").mkdir(exist_ok=True)
        
        # 统计信息
        self.stats = {
            "total_images": 0,
            "processed_images": 0,
            "failed_images": 0,
            "total_detections": 0,
            "category_counts": {},
            "processing_time": 0
        }
    
    def check_api_health(self) -> bool:
        """检查 API 是否可用"""
        try:
            health = self.client.health_check()
            if health.get('status') == 'healthy':
                logger.info(f"✅ API 服务正常: {health}")
                return True
            else:
                logger.warning(f"⚠️ API 服务状态异常: {health}")
                return False
        except Exception as e:
            logger.error(f"❌ API 服务不可用: {e}")
            return False
    
    # ==================== 视频处理（异步队列）====================
    def process_video_async(
        self,
        video_path: str,
        conf_threshold: float = 0.5,
        wait_for_completion: bool = True,
        poll_interval: float = 2.0,
        timeout: float = 300.0
    ) -> Optional[TaskInfo]:
        """
        异步处理视频文件（上传到队列）
        
        Args:
            video_path: 视频文件路径
            conf_threshold: 置信度阈值
            wait_for_completion: 是否等待处理完成
            poll_interval: 轮询间隔（秒）
            timeout: 超时时间（秒）
        
        Returns:
            任务信息（如果 wait_for_completion=True，返回最终状态）
        """
        try:
            # 上传视频到队列
            logger.info(f"📤 上传视频到队列: {video_path}")
            result = self.client.upload_video(video_path, conf_threshold=conf_threshold)
            
            task_id = result['task_id']
            logger.info(f"✅ 视频已添加到队列，任务ID: {task_id}")
            
            if wait_for_completion:
                # 等待处理完成
                logger.info(f"⏳ 等待任务处理完成...")
                final_status = self.client.wait_for_completion(
                    task_id, 
                    poll_interval=poll_interval,
                    timeout=timeout
                )
                
                logger.info(f"✅ 视频处理完成: {result['filename']}")
                logger.info(f"   总帧数: {final_status.total_frames}")
                logger.info(f"   总检测: {final_status.total_detections}")
                
                return final_status
            else:
                # 返回任务ID供后续查询
                return TaskInfo(
                    task_id=task_id,
                    filename=result['filename'],
                    status='pending',
                    progress=0,
                    message='已添加到队列',
                    created_at=datetime.now().isoformat()
                )
            
        except TimeoutError:
            logger.error(f"⏰ 视频处理超时")
            return None
        except Exception as e:
            logger.error(f"❌ 视频处理失败: {e}")
            return None
    
    def get_video_task_status(self, task_id: str) -> Optional[TaskInfo]:
        """
        查询视频任务状态
        
        Args:
            task_id: 任务ID
        
        Returns:
            任务状态信息
        """
        try:
            return self.client.get_task_status(task_id)
        except Exception as e:
            logger.error(f"❌ 获取任务状态失败: {e}")
            return None
    
    # ==================== 图片处理 ====================
    def process_image(self, image_path: Path) -> Optional[DetectionResult]:
        """
        处理单张图片（通过帧目录模式）
        
        Args:
            image_path: 图片路径
        
        Returns:
            处理结果
        """
        try:
            # 调用 API 推理（单图推理通过帧目录模式实现）
            result = self.client.infer_frame_directory(
                str(image_path.parent),
                conf_threshold=self.min_score
            )
            
            # 创建检测结果对象
            frame_results = result.get('frame_results', {})
            frame_data = frame_results.get(image_path.name, {})
            
            detection_result = DetectionResult(
                filename=image_path.name,
                image_width=0,
                image_height=0,
                detections=frame_data.get('detections', []),
                timestamp=datetime.now().isoformat()
            )
            
            # 获取图片尺寸
            try:
                from PIL import Image
                with Image.open(image_path) as img:
                    detection_result.image_width = img.width
                    detection_result.image_height = img.height
            except:
                pass
            
            # 应用过滤
            if self.categories:
                detection_result = detection_result.filter_by_category(self.categories)
            
            if self.min_score > 0:
                detection_result = detection_result.filter_by_score(self.min_score)
            
            return detection_result
            
        except Exception as e:
            logger.error(f"处理失败 [{image_path.name}]: {e}")
            return None
    
    def process_batch(
        self, 
        max_workers: int = 4,
        extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp')
    ) -> Dict[str, DetectionResult]:
        """
        批量处理图片
        
        Args:
            max_workers: 最大并发数
            extensions: 支持的图片格式
        
        Returns:
            处理结果字典
        """
        start_time = time.time()
        
        # 收集所有图片
        image_files = []
        for ext in extensions:
            image_files.extend(self.input_dir.glob(f"*{ext}"))
            image_files.extend(self.input_dir.glob(f"*{ext.upper()}"))
        
        self.stats['total_images'] = len(image_files)
        logger.info(f"📂 找到 {len(image_files)} 张图片")
        
        if not image_files:
            logger.warning("没有找到任何图片")
            return {}
        
        results = {}
        
        # 使用线程池并发处理
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(self.process_image, path): path
                for path in image_files
            }
            
            for i, future in enumerate(as_completed(future_to_path), 1):
                path = future_to_path[future]
                try:
                    result = future.result()
                    if result:
                        results[path.name] = result
                        self.stats['processed_images'] += 1
                        self.stats['total_detections'] += result.count
                        
                        # 统计类别
                        for det in result.detections:
                            cat = det.get('category_name', 'unknown')
                            self.stats['category_counts'][cat] = \
                                self.stats['category_counts'].get(cat, 0) + 1
                        
                        logger.info(
                            f"[{i}/{len(image_files)}] ✅ {path.name}: "
                            f"{result.count} 个检测"
                        )
                    else:
                        self.stats['failed_images'] += 1
                        logger.warning(f"[{i}/{len(image_files)}] ⚠️ {path.name}: 处理失败")
                        
                except Exception as e:
                    self.stats['failed_images'] += 1
                    logger.error(f"[{i}/{len(image_files)}] ❌ {path.name}: {e}")
        
        self.stats['processing_time'] = time.time() - start_time
        
        return results
    
    # ==================== 导出和统计 ====================
    def export_yolo_labels(self, results: Dict[str, DetectionResult]):
        """导出为 YOLO 格式标签"""
        logger.info("📤 导出 YOLO 格式标签...")
        
        # 类别映射
        class_names = list(set(
            cat 
            for r in results.values() 
            for det in r.detections 
            for cat in [det.get('category_name', 'unknown')]
        ))
        class_names.sort()
        class_map = {name: i for i, name in enumerate(class_names)}
        
        # 保存类别映射
        with open(self.output_dir / "labels" / "classes.txt", 'w') as f:
            for name in class_names:
                f.write(f"{name}\n")
        
        # 保存每个图片的标签
        for filename, result in results.items():
            label_file = self.output_dir / "labels" / f"{Path(filename).stem}.txt"
            
            with open(label_file, 'w') as f:
                for det in result.detections:
                    bbox = det.get('bbox', [])
                    if len(bbox) != 4:
                        continue
                    
                    x1, y1, x2, y2 = bbox
                    
                    center_x = ((x1 + x2) / 2) / result.image_width
                    center_y = ((y1 + y2) / 2) / result.image_height
                    width = (x2 - x1) / result.image_width
                    height = (y2 - y1) / result.image_height
                    
                    center_x = max(0, min(1, center_x))
                    center_y = max(0, min(1, center_y))
                    width = max(0, min(1, width))
                    height = max(0, min(1, height))
                    
                    class_id = class_map.get(det.get('category_name', 'unknown'), 0)
                    
                    f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
        
        logger.info(f"✅ YOLO 标签已导出: {self.output_dir / 'labels'}")
    
    def export_coco_annotations(self, results: Dict[str, DetectionResult]):
        """导出为 COCO 格式标注"""
        logger.info("📤 导出 COCO 格式标注...")
        
        class_names = list(set(
            cat 
            for r in results.values() 
            for det in r.detections 
            for cat in [det.get('category_name', 'unknown')]
        ))
        class_names.sort()
        
        categories = [
            {"id": i, "name": name, "supercategory": "object"}
            for i, name in enumerate(class_names)
        ]
        class_map = {name: i for i, name in enumerate(class_names)}
        
        coco = {
            "info": {
                "description": "EdgeCrafter Detection Results",
                "version": "1.0",
                "date_created": datetime.now().isoformat()
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": categories
        }
        
        annotation_id = 1
        
        for img_id, (filename, result) in enumerate(results.items(), 1):
            coco["images"].append({
                "id": img_id,
                "file_name": filename,
                "width": result.image_width,
                "height": result.image_height
            })
            
            for det in result.detections:
                bbox = det.get('bbox', [])
                if len(bbox) != 4:
                    continue
                
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1
                
                coco["annotations"].append({
                    "id": annotation_id,
                    "image_id": img_id,
                    "category_id": class_map.get(det.get('category_name', 'unknown'), 0),
                    "bbox": [x1, y1, width, height],
                    "area": width * height,
                    "score": det.get('score', 1.0),
                    "iscrowd": 0
                })
                
                annotation_id += 1
        
        output_file = self.output_dir / "annotations.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(coco, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ COCO 标注已导出: {output_file}")
    
    def copy_images(self, results: Dict[str, DetectionResult]):
        """复制有检测结果的图片到输出目录"""
        logger.info("📁 复制图片到输出目录...")
        
        copied = 0
        for filename in results.keys():
            src = self.input_dir / filename
            dst = self.output_dir / "images" / filename
            
            if src.exists():
                shutil.copy2(src, dst)
                copied += 1
        
        logger.info(f"✅ 已复制 {copied} 张图片")
    
    def generate_statistics(self, results: Dict[str, DetectionResult]) -> Dict:
        """生成统计分析报告"""
        logger.info("📊 生成统计报告...")
        
        stats = {
            "summary": {
                "total_images": len(results),
                "total_detections": sum(r.count for r in results.values()),
                "avg_detections_per_image": 0,
                "images_with_detections": sum(1 for r in results.values() if r.count > 0),
                "images_without_detections": sum(1 for r in results.values() if r.count == 0)
            },
            "category_distribution": {},
            "size_distribution": {
                "small": 0,
                "medium": 0,
                "large": 0
            },
            "score_distribution": {
                "high": 0,
                "medium": 0,
                "low": 0
            }
        }
        
        if stats["summary"]["total_images"] > 0:
            stats["summary"]["avg_detections_per_image"] = \
                stats["summary"]["total_detections"] / stats["summary"]["total_images"]
        
        for result in results.values():
            for det in result.detections:
                cat = det.get('category_name', 'unknown')
                stats["category_distribution"][cat] = \
                    stats["category_distribution"].get(cat, 0) + 1
                
                bbox = det.get('bbox', [])
                if len(bbox) == 4:
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    if area < 32 * 32:
                        stats["size_distribution"]["small"] += 1
                    elif area < 96 * 96:
                        stats["size_distribution"]["medium"] += 1
                    else:
                        stats["size_distribution"]["large"] += 1
                
                score = det.get('score', 0)
                if score > 0.8:
                    stats["score_distribution"]["high"] += 1
                elif score > 0.5:
                    stats["score_distribution"]["medium"] += 1
                else:
                    stats["score_distribution"]["low"] += 1
        
        report_file = self.output_dir / "statistics.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ 统计报告已生成: {report_file}")
        
        print("\n" + "=" * 60)
        print("📊 数据流水线统计报告")
        print("=" * 60)
        print(f"总图片数: {stats['summary']['total_images']}")
        print(f"总检测数: {stats['summary']['total_detections']}")
        print(f"平均检测数/图: {stats['summary']['avg_detections_per_image']:.2f}")
        print(f"有检测的图片: {stats['summary']['images_with_detections']}")
        print(f"无检测的图片: {stats['summary']['images_without_detections']}")
        print("\n类别分布:")
        for cat, count in sorted(stats['category_distribution'].items()):
            print(f"  {cat}: {count}")
        print("=" * 60)
        
        return stats
    
    def run(
        self,
        max_workers: int = 4,
        export_format: str = "both",
        copy_images: bool = True
    ):
        """
        运行完整的数据处理流水线（图片模式）
        
        Args:
            max_workers: 最大并发数
            export_format: 导出格式 (yolo/coco/both)
            copy_images: 是否复制图片
        """
        logger.info("=" * 60)
        logger.info("🚀 启动数据预处理流水线")
        logger.info("=" * 60)
        logger.info(f"输入目录: {self.input_dir}")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info(f"类别过滤: {self.categories or '无'}")
        logger.info(f"置信度阈值: {self.min_score}")
        
        # 检查 API
        if not self.check_api_health():
            logger.error("API 服务不可用，请先启动: python api_service.py")
            return None
        
        # 批量处理
        results = self.process_batch(max_workers=max_workers)
        
        if not results:
            logger.warning("没有处理任何图片")
            return None
        
        # 导出标签
        if export_format in ['yolo', 'both']:
            self.export_yolo_labels(results)
        
        if export_format in ['coco', 'both']:
            self.export_coco_annotations(results)
        
        # 复制图片
        if copy_images:
            self.copy_images(results)
        
        # 生成统计
        stats = self.generate_statistics(results)
        
        # 保存完整结果
        output_file = self.output_dir / "results" / "all_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(
                {k: asdict(v) for k, v in results.items()},
                f, indent=2, ensure_ascii=False
            )
        
        logger.info("=" * 60)
        logger.info("✅ 数据处理流水线完成！")
        logger.info(f"📂 输出目录: {self.output_dir}")
        logger.info("=" * 60)
        
        return results


# ==================== 命令行接口 ====================
def main():
    parser = argparse.ArgumentParser(
        description="EdgeCrafter 数据预处理流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # 模式选择
    parser.add_argument(
        '--mode',
        choices=['image', 'video', 'video-async'],
        default='image',
        help='处理模式: image(图片目录) / video(视频文件同步) / video-async(视频文件异步队列)'
    )
    
    # 通用参数
    parser.add_argument(
        '--input-dir', '-i',
        default='./input',
        help='输入图片目录（image模式）'
    )
    parser.add_argument(
        '--video-path', '-v',
        help='视频文件路径（video模式）'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='./pipeline_output',
        help='输出目录'
    )
    parser.add_argument(
        '--api-url',
        default='http://localhost:8000',
        help='API 服务地址'
    )
    parser.add_argument(
        '--categories',
        nargs='+',
        help='保留的类别列表'
    )
    parser.add_argument(
        '--min-score',
        type=float,
        default=0.5,
        help='最小置信度阈值'
    )
    parser.add_argument(
        '--export-format',
        choices=['yolo', 'coco', 'both'],
        default='both',
        help='导出格式'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=4,
        help='最大并发数'
    )
    
    # 视频处理参数
    parser.add_argument(
        '--wait',
        action='store_true',
        help='视频模式下等待处理完成'
    )
    parser.add_argument(
        '--poll-interval',
        type=float,
        default=2.0,
        help='轮询间隔（秒）'
    )
    
    args = parser.parse_args()
    
    # 创建流水线
    pipeline = DataPipeline(
        api_url=args.api_url,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        categories=args.categories,
        min_score=args.min_score
    )
    
    # 根据模式执行不同处理
    if args.mode == 'image':
        # 图片目录模式
        pipeline.run(
            max_workers=args.max_workers,
            export_format=args.export_format
        )
    elif args.mode == 'video' or args.mode == 'video-async':
        # 视频文件模式（异步队列）
        if not args.video_path:
            parser.error("video模式需要指定 --video-path 参数")
        
        # 检查API服务
        if not pipeline.check_api_health():
            print("API 服务不可用，请先启动: python api_service.py")
            return
        
        result = pipeline.process_video_async(
            video_path=args.video_path,
            conf_threshold=args.min_score,
            wait_for_completion=args.wait,
            poll_interval=args.poll_interval
        )
        
        if result:
            print("\n视频处理任务信息:")
            print(f"任务ID: {result.task_id}")
            print(f"文件名: {result.filename}")
            print(f"状态: {result.status}")
            if result.total_frames:
                print(f"总帧数: {result.total_frames}")
            if result.total_detections:
                print(f"总检测数: {result.total_detections}")
            if result.output_dir:
                print(f"输出目录: {result.output_dir}")


if __name__ == "__main__":
    main()