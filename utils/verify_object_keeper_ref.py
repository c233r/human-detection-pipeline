"""
ObjectKeeper 引用验证工具

随机选取 2 个任务，验证其 ObjectRef 是否可正常获取和访问。
用于调试零拷贝链路。

使用方法:
    python /data2/caimoyao/utils/verify_object_keeper_ref.py
"""

import random
import ray


def verify_single_ref(task_id: str, keeper, timeout: int = 10) -> dict:
    """验证单个任务的数据访问（ray.get 会自动解包返回数据）"""
    result = {
        "task_id": task_id,
        "ref_found": False,
        "data_accessible": False,
        "data_shape": None,
        "error": None,
    }

    try:
        # ray.get(get_ref.remote()) 会自动递归解包，直接返回数据
        data = ray.get(keeper.get_ref.remote(task_id), timeout=timeout)

        if data is None:
            result["error"] = "ObjectKeeper 返回 None（任务不存在或已过期）"
            return result

        result["ref_found"] = True
        result["data_accessible"] = True

        if hasattr(data, 'shape'):
            result["data_shape"] = str(data.shape)
        elif hasattr(data, '__len__'):
            result["data_shape"] = f"len={len(data)}"

    except Exception as e:
        result["error"] = f"获取数据失败: {e}"

    return result


if __name__ == "__main__":
    ray.init(address="auto", namespace="production")

    # 查找 ObjectKeeper
    try:
        keeper = ray.get_actor("ObjectKeeper", namespace="production")
        print(f"✅ ObjectKeeper 已找到")
    except Exception as e:
        print(f"❌ ObjectKeeper 未找到: {e}")
        ray.shutdown()
        exit(1)

    # 获取所有任务状态，随机选 2 个
    try:
        all_status = ray.get(keeper.get_all_status.remote())
        alive_tasks = [s for s in all_status if s.get("alive")]

        if len(alive_tasks) == 0:
            print(f"⚠️  无活跃任务 (共 {len(all_status)} 个，均已过期)")
            ray.shutdown()
            exit(0)

        sample_size = min(2, len(alive_tasks))
        sample = random.sample(alive_tasks, sample_size)

        print(f"📋 活跃任务: {len(alive_tasks)}/{len(all_status)} 个")
        print(f"🔍 随机选取 {sample_size} 个验证:")
        print()

        for task in sample:
            tid = task["task_id"]
            result = verify_single_ref(tid, keeper)
            status_icon = "✅" if (result["ref_found"] and result["data_accessible"]) else "❌"
            print(f"  {status_icon} task_id={tid}")
            print(f"     ref_found: {result['ref_found']}")
            print(f"     data_accessible: {result['data_accessible']}")
            print(f"     data_shape: {result.get('data_shape', 'N/A')}")
            print(result)
            if result["error"]:
                print(f"     error: {result['error']}")
            print()

    except Exception as e:
        print(f"❌ 验证过程异常: {e}")

    ray.shutdown()
