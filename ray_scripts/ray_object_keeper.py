"""
Ray Object Keeper - 保活 Actor 服务

功能：
1. 保持 Ray ObjectRef 的引用计数，防止数据被过早回收
2. 记录各服务阶段的完成情况
3. 提供清理接口，在数据不再需要时释放引用

使用方法：
    # 启动保活服务（在 Ray 集群的任意节点上）
    python ray_object_keeper.py

    # 在其他服务中使用
    import ray
    keeper = ray.get_actor("ObjectKeeper")
    
    # 注册需要保活的 ObjectRef（直接传 ObjectRef 对象，Ray 自动维护引用计数）
    keeper.register_ref.remote("task_xxx", frames_ref, ttl=3600)
    
    # 标记阶段完成
    keeper.mark_stage.remote("task_xxx", "monitor_completed")
    
    # 释放引用
    keeper.release_ref.remote("task_xxx")
"""

import time
import logging
from typing import Dict, Optional

import ray

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/data2/caimoyao/logs/object_keeper_ray.log', encoding='utf-8')
    ]
)
logger = logging.getLogger("ObjectKeeper")


@ray.remote
class ObjectKeeper:
    """
    Ray ObjectRef 保活 Actor
    
    特性：
    - 持有 ObjectRef 引用，防止垃圾回收
    - 记录各阶段完成情况
    - 支持 TTL 自动过期
    - 支持主动释放
    """
    
    def __init__(self):
        self._refs: Dict[str, dict] = {}
        self._stages: Dict[str, Dict[str, bool]] = {}
        self._created_at = time.time()
        logger.info(f"ObjectKeeper 已启动")
    
    def register_ref(self, task_id: str, ref: ray.ObjectRef, ttl: int = 1800) -> bool:
        """
        注册 ObjectRef 进行保活
        
        直接接收 ray.ObjectRef 对象作为参数。Ray 会在后台自动维护
        分布式引用计数。
        
        Args:
            task_id: 任务 ID
            ref: 要保活的 ray.ObjectRef
            ttl: 生存时间（秒），默认 1800 秒
        
        Returns:
            是否注册成功
        """
        try:
            # 幂等：同一 task_id 重复注册时，先释放旧引用，再替换为新注册的
            # （避免 preprocess 重试/重复执行导致旧 ObjectRef 残留内存）
            if task_id in self._refs:
                old_ref = self._refs.pop(task_id)
                del old_ref  # 显式释放旧 ObjectRef 的 Python 引用，Ray 引用计数立即 -1
                logger.warning(f"重复注册: task={task_id}, 已释放旧引用并替换为新数据")

            self._refs[task_id] = {
                "ref": ref,
                "created_at": time.time(),
                "expires_at": time.time() + ttl,
                "size": 0,
            }
            self._stages[task_id] = self._stages.get(task_id, {})
            logger.info(f"注册保活: task={task_id}, ttl={ttl}s")
            return True
        except Exception as e:
            logger.error(f"注册失败: task={task_id}, error={e}")
            return False

    def mark_stage(self, task_id: str, stage: str) -> bool:
        """
        标记阶段完成
        
        Args:
            task_id: 任务 ID
            stage: 阶段名称（如 'preprocess', 'monitor', 'hand_crop'）
        """
        if task_id not in self._stages:
            self._stages[task_id] = {}
        
        self._stages[task_id][stage] = True
        logger.info(f"阶段完成: task={task_id}, stage={stage}")
        return True
    
    def get_status(self, task_id: str) -> dict:
        """获取任务状态"""
        ref_info = self._refs.get(task_id)
        stages = self._stages.get(task_id, {})
        
        status = {
            "task_id": task_id,
            "registered": ref_info is not None,
            "stages": stages,
            "alive": False,
        }
        
        if ref_info:
            now = time.time()
            status["alive"] = now < ref_info["expires_at"]
            status["expires_in"] = max(0, int(ref_info["expires_at"] - now))
            status["size"] = ref_info["size"]
            status["created_at"] = ref_info["created_at"]
        
        return status
    
    def get_all_status(self) -> list:
        """获取所有任务状态"""
        return [self.get_status(task_id) for task_id in self._refs]
    
    def get_ref(self, task_id: str):
        """
        获取保活的 ObjectRef
        
        Args:
            task_id: 任务 ID
        
        Returns:
            ObjectRef 或 None
        """
        ref_info = self._refs.get(task_id)
        if ref_info:
            now = time.time()
            if now < ref_info["expires_at"]:
                return ref_info["ref"]
            else:
                logger.warning(f"get_ref: task={task_id} 已过期 (expires_at={ref_info['expires_at']:.0f}, now={now:.0f})")
                return None
        else:
            # 未注册或已被 cleanup_expired 清理，打日志方便排查
            logger.warning(f"get_ref: task={task_id} 不存在（未注册或已过期清理）")
        return None
    
    def release_ref(self, task_id: str) -> bool:
        """
        主动释放 ObjectRef 引用

        当所有阶段完成后调用，释放内存
        """
        removed = False
        if task_id in self._refs:
            del self._refs[task_id]
            removed = True
        if task_id in self._stages:
            del self._stages[task_id]
            removed = True
        if removed:
            logger.info(f"释放引用: task={task_id}")
        return removed

    def cleanup_expired(self) -> int:
        """
        清理过期的引用
        
        Returns:
            清理的数量
        """
        now = time.time()
        expired = [tid for tid, info in self._refs.items() if now > info["expires_at"]]
        
        for tid in expired:
            del self._refs[tid]
            if tid in self._stages:
                del self._stages[tid]
            logger.info(f"过期清理: task={tid}")
        
        # 同时清理 _stages 中残留但 _refs 已不存在的条目（孤儿数据）
        orphan_stages = [tid for tid in self._stages if tid not in self._refs]
        for tid in orphan_stages:
            del self._stages[tid]
            logger.info(f"清理孤儿 stage: task={tid}")
        
        return len(expired) + len(orphan_stages)
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        now = time.time()
        active_refs = sum(1 for info in self._refs.values() if now < info["expires_at"])
        total_size = sum(info["size"] for info in self._refs.values())
        completed_stages = sum(len(stages) for stages in self._stages.values())
        
        return {
            "total_refs": len(self._refs),
            "active_refs": active_refs,
            "completed_stages": completed_stages,
            "total_size_bytes": total_size,
            "uptime_seconds": int(now - self._created_at),
        }


def start_keeper():
    """启动 ObjectKeeper 服务"""
    ray.init(
        address="auto",
        namespace="production",
    )
    
    # 检查是否已存在
    try:
        existing = ray.get_actor("ObjectKeeper", namespace="production")
        logger.warning("ObjectKeeper 已存在，使用现有实例")
        return existing
    except ValueError:
        pass
    
    # 创建新的 Actor
    keeper = ObjectKeeper.options(
        name="ObjectKeeper",
        namespace="production",
        num_cpus=0,  # 不需要 CPU
    ).remote()
    
    logger.info("ObjectKeeper 已创建")
    
    # 启动定期清理任务
    @ray.remote
    def periodic_cleanup():
        while True:
            time.sleep(60)  # 每分钟清理一次
            try:
                keeper.cleanup_expired.remote()
                stats = ray.get(keeper.get_stats.remote())
                if stats["active_refs"] > 0:
                    logger.info(f"统计: total_refs={stats['total_refs']}, active_refs={stats['active_refs']}, stages={stats['completed_stages']}, size={stats['total_size_bytes']}B")
            except Exception as e:
                logger.error(f"清理任务异常: {e}")
    
    # 启动后台清理
    periodic_cleanup.remote()
    
    return keeper


if __name__ == "__main__":
    logger.info("正在启动 ObjectKeeper 服务...")
    keeper = start_keeper()
    
    # 保持运行，只在状态变化时打印（排除 uptime_seconds 字段）
    last_stats = None
    try:
        while True:
            time.sleep(10)
            stats = ray.get(keeper.get_stats.remote())
            # 排除 uptime_seconds，只关注实际数据变化
            stats_snapshot = {k: v for k, v in stats.items() if k != "uptime_seconds"}
            if stats["active_refs"] > 0 and stats_snapshot != last_stats:
                logger.info(f"运行中: {stats}")
                last_stats = stats_snapshot
    except KeyboardInterrupt:
        logger.info("正在关闭 ObjectKeeper...")
    except Exception as e:
        logger.error(f"异常: {e}")
