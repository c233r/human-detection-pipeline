#!/usr/bin/env python3
"""
查看 Kafka topic 分区未消费消息数量及消费组分配情况
"""
import argparse
import subprocess
from confluent_kafka import Consumer, TopicPartition
from confluent_kafka.admin import AdminClient


def check_partitions(topic, group_id=None, bootstrap_servers="localhost:9094"):
    consumer = Consumer({
        "bootstrap.servers": bootstrap_servers,
        "group.id": group_id or "check_partitions_temp",
        "enable.auto.commit": False,
        "auto.offset.reset": "earliest",
        "broker.address.family": "v4",
    })

    try:
        metadata = consumer.list_topics(topic, timeout=5)
    except Exception as e:
        print(f"❌ 未找到 topic: {topic}, 错误: {e}")
        consumer.close()
        return

    topic_metadata = metadata.topics.get(topic)
    if not topic_metadata:
        print(f"❌ 未找到 topic: {topic}")
        consumer.close()
        return

    partitions = sorted(topic_metadata.partitions.keys())
    total_pending = 0
    max_pending = 0
    min_pending = float("inf")

    print(f"\n📊 {topic} 分区未消费消息统计")
    print("-" * 70)
    print(f"{'分区':<8} {'已提交':<10} {'最新':<10} {'未消费':<10}")
    print("-" * 70)

    for p_id in partitions:
        tp = TopicPartition(topic, p_id)
        committed = consumer.committed([tp])
        # 从未消费过的 group，committed 返回 OFFSET_INVALID (-1001)，不能直接当真实 offset
        committed_offset = committed[0].offset if committed else -1001

        low, high = consumer.get_watermark_offsets(tp)
        end = high

        if committed_offset < 0:  # 无已提交 offset，按 auto.offset.reset=earliest 从日志头部算
            committed_offset = low
        pending = max(0, end - committed_offset)
        total_pending += pending
        max_pending = max(max_pending, pending)
        min_pending = min(min_pending, pending)

        status = "🔴" if pending > 0 else "🟢"
        print(f"{status} {p_id:<6} {committed_offset:<10} {end:<10} {pending:<10}")

    print("-" * 70)
    print(f"总计: {total_pending} 条 (最大: {max_pending}, 最小: {min_pending})")

    if group_id:
        print(f"\n📝 消费组: {group_id}")

    consumer.close()


def check_group_assignment(group_id, topic=None, bootstrap_servers="localhost:9094"):
    kafka_bin = "/data2/caimoyao/Kafka/kafka/bin/kafka-consumer-groups.sh"
    
    cmd = [
        kafka_bin,
        "--bootstrap-server", bootstrap_servers,
        "--describe",
        "--group", group_id,
    ]
    
    print(f"\n👥 消费组: {group_id}")
    print("-" * 70)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            print(f"❌ 命令执行失败: {result.stderr}")
            return
        
        lines = result.stdout.strip().split("\n")
        
        if not lines:
            print("   暂无成员")
            return
        
        print(result.stdout)
        
    except FileNotFoundError:
        print(f"❌ 未找到 kafka-consumer-groups.sh: {kafka_bin}")
        print("   请设置正确的 KAFKA_BIN 路径")
    except subprocess.TimeoutExpired:
        print("❌ 命令执行超时")
    except Exception as e:
        print(f"❌ 执行命令时出错: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="查看 Kafka 分区和消费组信息")
    parser.add_argument("--topic", "-t", help="要检查的 topic 名称")
    parser.add_argument("--group", "-g", help="消费组 ID")
    parser.add_argument("--broker", "-b", default="localhost:9094", help="Kafka broker 地址")
    parser.add_argument("--list-group", "-l", action="store_true", help="只查看消费组分配情况")
    args = parser.parse_args()

    if args.list_group and args.group:
        check_group_assignment(args.group, args.topic, args.broker)
    elif args.topic:
        check_partitions(args.topic, args.group, args.broker)
        if args.group:
            check_group_assignment(args.group, args.topic, args.broker)
    else:
        parser.print_help()
