#!/bin/bash
# Kafka Test Topics 清理脚本
# 清理所有 *_test topic 中的消息

set -e

KAFKA_HOME="/data2/caimoyao/Kafka/kafka"
BOOTSTRAP_SERVER="localhost:9094"

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Kafka Test Topics 清理脚本${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 获取所有 test topics
echo -e "${YELLOW}[1] 获取 test topics 列表...${NC}"
TOPICS=$($KAFKA_HOME/bin/kafka-topics.sh --bootstrap-server $BOOTSTRAP_SERVER --list 2>/dev/null | grep "_test$")

if [ -z "$TOPICS" ]; then
    echo "没有找到 *_test 的 topics"
    exit 0
fi

echo "找到以下 test topics:"
echo "$TOPICS"
echo ""

# 查看每个 topic 的消息数量
echo -e "${YELLOW}[2] 查看各 topic 消息数量...${NC}"
for TOPIC in $TOPICS; do
    # 获取每个分区的 offset (最新位置)
    OFFSETS=$($KAFKA_HOME/bin/kafka-get-offsets.sh --bootstrap-server $BOOTSTRAP_SERVER --topic $TOPIC --time -1 2>/dev/null)
    TOTAL=0
    for LINE in $OFFSETS; do
        COUNT=$(echo $LINE | awk '{print $3}')
        if [ -n "$COUNT" ] && [ "$COUNT" -gt 0 ]; then
            TOTAL=$((TOTAL + COUNT))
        fi
    done
    echo "  $TOPIC: $TOTAL 条消息"
done
echo ""

# 确认操作
read -p "确定要清理所有 test topics 吗？(y/n): " CONFIRM
if [ "$CONFIRM" != "y" ]; then
    echo "已取消"
    exit 0
fi

echo ""
echo -e "${YELLOW}[3] 开始清理...${NC}"

# 方法1: 重置消费者组 offset（让消费者跳过所有历史消息）
echo "  重置消费者组 offset..."
for TOPIC in $TOPICS; do
    CONSUMER_GROUP="${TOPIC%_test}_consumers_test"
    # 获取 topic 的分区数
    PARTITIONS=$($KAFKA_HOME/bin/kafka-topics.sh --bootstrap-server $BOOTSTRAP_SERVER --describe --topic $TOPIC 2>/dev/null | grep -c "Partition:")
    
    # 重置 offset 到最新位置 (-1)
    $KAFKA_HOME/bin/kafka-consumer-groups.sh --bootstrap-server $BOOTSTRAP_SERVER \
        --reset-offsets --group $CONSUMER_GROUP --topic $TOPIC --to-latest --execute 2>/dev/null || true
    
    echo "    消费者组 $CONSUMER_GROUP offset 已重置"
done

# 方法2: 对没有消费者组的 topic，直接删除并重建
echo ""
echo "  删除并重建 topics..."
for TOPIC in $TOPICS; do
    echo "    处理 $TOPIC..."
    
    # 获取原始分区数
    PARTITIONS=$($KAFKA_HOME/bin/kafka-topics.sh --bootstrap-server $BOOTSTRAP_SERVER --describe --topic $TOPIC 2>/dev/null | grep -c "Partition:")
    if [ "$PARTITIONS" -eq 0 ]; then
        PARTITIONS=1
    fi
    
    # 删除 topic
    $KAFKA_HOME/bin/kafka-topics.sh --bootstrap-server $BOOTSTRAP_SERVER --delete --topic $TOPIC 2>/dev/null || true
    sleep 1
    
    # 重建 topic（保持原有配置）
    $KAFKA_HOME/bin/kafka-topics.sh --bootstrap-server $BOOTSTRAP_SERVER \
        --create --topic $TOPIC --partitions $PARTITIONS --replication-factor 1 2>/dev/null || true
    
    echo -e "    ${GREEN}$TOPIC 已重建${NC}"
done

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  清理完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 验证
echo "验证 topics:"
$KAFKA_HOME/bin/kafka-topics.sh --bootstrap-server $BOOTSTRAP_SERVER --list 2>/dev/null | grep "_test$" | while read T; do
    echo "  $T"
done

echo ""
echo "注意: 如果服务正在运行，可能需要重启服务才能生效"
