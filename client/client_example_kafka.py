"""
EdgeCrafter Kafka 客户端
使用 Kafka 进行任务分配，发送视频路径而非上传视频文件

使用方法:
    python client_example_kafka.py --send /path/to/video.mp4

或者在 Python 中导入使用:
    from client_example_kafka import EdgeCrafterKafkaClient
    client = EdgeCrafterKafkaClient(bootstrap_servers="localhost:9094")
    result = client.send_task("/path/to/video.mp4")
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

try:
    from kafka import KafkaProducer, KafkaConsumer
    from kafka.errors import KafkaError
    from kafka.structs import TopicPartition
except ImportError:
    print("❌ 需要安装 kafka-python: pip install kafka-python")
    sys.exit(1)


@dataclass
class TaskInfo:
    """任务信息数据类"""
    task_id: str
    video_path: str
    conf_threshold: float
    status: str = "pending"
    message: str = ""
    created_at: Optional[str] = None


class EdgeCrafterKafkaClient:
    """
    EdgeCrafter Kafka 客户端
    
    通过 Kafka 发送任务消息，消息内容为视频路径，消费者负责读取路径并处理视频
    自动检测消费者组积压情况，避免发送过快导致消费者处理不过来
    """
    
    def __init__(self, bootstrap_servers: str = "localhost:9094", 
                 topic: str = "video_tasks", consumer_group: str = "video_task_consumers",
                 max_pending: int = 100, poll_interval: int = 600, timeout: int = 30):
        """
        初始化客户端
        
        Args:
            bootstrap_servers: Kafka broker 地址，多个用逗号分隔
            topic: 任务消息发送的 topic 名称
            consumer_group: 消费者组名称，用于查询积压数量
            max_pending: 最大允许的积压任务数，超过则等待
            poll_interval: 查询积压的轮询间隔（秒）
            timeout: Kafka 连接超时时间（秒）
        """
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.consumer_group = consumer_group
        self.max_pending = max_pending
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.producer = None
    
    def _get_producer(self):
        """获取或创建 Kafka 生产者"""
        if self.producer is None:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8'),
                max_block_ms=self.timeout * 1000,
                retries=3,
                retry_backoff_ms=1000
            )
        return self.producer
    
    def get_pending_count(self) -> int:
        """
        查询消费者组中积压的任务数量
        
        通过比较 topic 每个分区的 end_offset 和 committed_offset 来计算 lag
        
        Returns:
            积压的任务总数
        """
        consumer = KafkaConsumer(
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.consumer_group,
            enable_auto_commit=False,
            max_poll_interval_ms=5000
        )
        
        try:
            partitions = consumer.partitions_for_topic(self.topic)
            if not partitions:
                return 0
            
            tps = [TopicPartition(self.topic, p) for p in partitions]
            
            end_offsets = consumer.end_offsets(tps)
            
            total_pending = 0
            for tp in tps:
                committed_offset = consumer.committed(tp)
                end_offset = end_offsets.get(tp, 0)
                if committed_offset is not None:
                    pending = end_offset - committed_offset
                else:
                    pending = end_offset
                if pending > 0:
                    total_pending += pending
            
            return total_pending
        finally:
            consumer.close()
    
    def send_task(
        self, 
        video_path: str, 
        conf_threshold: float = 0.5,
        max_pending: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        发送任务到 Kafka（自动检测积压，超过阈值则等待）
        
        Args:
            video_path: 视频文件的绝对路径
            conf_threshold: 置信度阈值
            max_pending: 最大允许的积压数，覆盖初始化时的设置
        
        Returns:
            任务信息，包含 task_id
        """
        video_path = os.path.abspath(video_path)
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        max_pending = max_pending if max_pending is not None else self.max_pending
        
        # 等待积压任务减少到阈值以下
        while True:
            try:
                pending = self.get_pending_count()
            except Exception as e:
                print(f"  ⚠️ 查询积压数量失败: {e}，直接发送")
                break
            
            if pending < max_pending:
                break
            
            print(f"  ⏳ 积压任务: {pending}/{max_pending}，等待 {self.poll_interval}s...")
            time.sleep(self.poll_interval)
        
        task_id = os.path.splitext(os.path.basename(video_path))[0]
        message = {
            "task_id": task_id,
            "video_path": video_path,
            "filename": os.path.basename(video_path),
            "conf_threshold": conf_threshold,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        producer = self._get_producer()
        
        try:
            future = producer.send(
                self.topic,
                key=task_id,
                value=message
            )
            future.get(timeout=self.timeout)
            print(f"  ✅ 任务已发送: task_id={task_id}, video={os.path.basename(video_path)}")
            
            return {
                "task_id": task_id,
                "video_path": video_path,
                "status": "sent",
                "message": "任务已发送到 Kafka"
            }
        except KafkaError as e:
            print(f"  ❌ Kafka 发送失败: {e}")
            raise
        except Exception as e:
            print(f"  ❌ 发送失败: {e}")
            raise
    
    def close(self):
        """关闭 Kafka 生产者连接"""
        if self.producer:
            self.producer.flush()
            self.producer.close()
            self.producer = None


# ==================== 使用示例 ====================
def main():
    """主函数 - 演示各种用法"""
    
    client = EdgeCrafterKafkaClient(bootstrap_servers="localhost:9094")
    
    print("=" * 60)
    print("   EdgeCrafter Kafka 客户端")
    print("=" * 60)
    
    test_video = "/data2/caimoyao/Jump_rope_dataset/videos/2026-04-14_15-41-06-722.mp4"
    
    if os.path.exists(test_video):
        print(f"\n发送测试任务: {test_video}")
        try:
            result = client.send_task(test_video, conf_threshold=0.5)
            print(f"任务ID: {result['task_id']}")
            print(f"状态: {result['status']}")
        except Exception as e:
            print(f"发送失败: {e}")
    else:
        print(f"\n⚠️ 测试视频不存在: {test_video}")
    
    client.close()
    print("\n" + "=" * 60)
    print("   示例完成")
    print("=" * 60)


# ==================== 命令行接口 ====================
def cli():
    """命令行接口"""
    parser = argparse.ArgumentParser(
        description="EdgeCrafter Kafka 客户端",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 发送单个视频任务
  python client_example_kafka.py --send /path/to/video.mp4

  # 发送文件夹中所有视频
  python client_example_kafka.py --send-dir /path/to/videos

  # 发送文件夹中前10个视频
  python client_example_kafka.py --send-dir /path/to/videos --limit 10

  # 指定 Kafka broker 地址
  python client_example_kafka.py --send /path/to/video.mp4 --broker localhost:9094

  # 指定自定义 topic
  python client_example_kafka.py --send /path/to/video.mp4 --topic my_video_tasks

  # 设置积压阈值并发送（积压超过10个则等待）
  python client_example_kafka.py --send-dir /path/to/videos --max-pending 10

  # 不限制积压，直接发送
  python client_example_kafka.py --send-dir /path/to/videos --max-pending 0

  # 查询当前积压数量
  python client_example_kafka.py --check-pending
        """
    )
    
    parser.add_argument('--broker', default='localhost:9094', 
                        help='Kafka broker 地址（默认: localhost:9094）')
    parser.add_argument('--topic', default='video_tasks', 
                        help='任务消息 topic 名称（默认: video_tasks）')
    parser.add_argument('--consumer-group', default='video_task_consumers',
                        help='消费者组名称（默认: video_task_consumers）')
    parser.add_argument('--send', type=str, help='发送单个视频任务')
    parser.add_argument('--send-dir', type=str, help='发送文件夹中所有视频')
    parser.add_argument('--check-pending', action='store_true', help='查询当前积压任务数量')
    parser.add_argument('--ext', type=str, default='.mp4,.avi,.mov,.mkv,.webm',
                        help='视频文件扩展名（逗号分隔，默认: .mp4,.avi,.mov,.mkv,.webm）')
    parser.add_argument('--parallel', type=int, default=4,
                        help='并行发送数（默认: 4）')
    parser.add_argument('--limit', type=int, default=0,
                        help='限制发送视频数量（默认: 0，不限制）')
    parser.add_argument('--conf-threshold', type=float, default=0.5, 
                        help='置信度阈值（默认: 0.5）')
    parser.add_argument('--max-pending', type=int, default=1000000,
                        help='最大积压任务数，超过则等待（默认: 20，0 表示不限制）')
    parser.add_argument('--poll-interval', type=int, default=600,
                        help='查询积压的轮询间隔（秒，默认: 10）')
    
    args = parser.parse_args()
    
    client = EdgeCrafterKafkaClient(
        bootstrap_servers=args.broker,
        topic=args.topic,
        consumer_group=args.consumer_group,
        max_pending=args.max_pending if args.max_pending > 0 else 999999,
        poll_interval=args.poll_interval
    )
    
    if args.check_pending:
        try:
            pending = client.get_pending_count()
            print(f"当前积压任务数量: {pending}")
        except Exception as e:
            print(f"❌ 查询失败: {e}")
    
    elif args.send:
        try:
            result = client.send_task(args.send, conf_threshold=args.conf_threshold)
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"❌ 发送失败: {e}")
    
    elif args.send_dir:
        if not os.path.isdir(args.send_dir):
            print(f"❌ 文件夹不存在: {args.send_dir}")
            return
        
        exts = [e.strip().lower() for e in args.ext.split(',')]
        video_files = sorted([
            f for f in Path(args.send_dir).iterdir()
            if f.is_file() and f.suffix.lower() in exts
        ])
        
        if not video_files:
            print(f"❌ 文件夹中没有找到视频文件（支持的扩展名: {args.ext}）")
            return
        
        if args.limit > 0 and len(video_files) > args.limit:
            video_files = video_files[:args.limit]
            print(f"找到 {len(video_files)} 个视频文件，限制发送前 {args.limit} 个...")
        else:
            print(f"找到 {len(video_files)} 个视频文件，开始发送...")
        
        def send_one(video_path):
            try:
                result = client.send_task(
                    str(video_path),
                    conf_threshold=args.conf_threshold
                )
                return (video_path.name, result)
            except Exception as e:
                return (video_path.name, {"error": str(e)})
        
        if args.parallel > 1:
            with ThreadPoolExecutor(max_workers=args.parallel) as pool:
                futures = {pool.submit(send_one, v): v for v in video_files}
                for future in as_completed(futures):
                    name, result = future.result()
                    if "error" in result:
                        print(f"  ❌ {name}: {result['error']}")
                    else:
                        print(f"  ✅ {name}: task_id={result['task_id']}")
        else:
            for i, vf in enumerate(video_files, 1):
                print(f"\n[{i}/{len(video_files)}] 发送: {vf.name}")
                try:
                    result = client.send_task(str(vf), conf_threshold=args.conf_threshold)
                    print(f"  ✅ task_id={result['task_id']}")
                except Exception as e:
                    print(f"  ❌ {e}")
    
    else:
        parser.print_help()
    
    client.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        cli()
    else:
        main()