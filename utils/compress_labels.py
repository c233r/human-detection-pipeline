"""
压缩 labels JSON 文件，减小存储压力。
- 浮点数四舍五入到 2 位小数 + 紧凑 JSON
- 使用 asyncio + aiofiles 协程并发
"""

import json
import os
import argparse
import time
import datetime
import asyncio
import aiofiles
from pathlib import Path


def round_floats(obj):
    """递归将所有 float 四舍五入到 2 位小数。"""
    if isinstance(obj, float):
        return round(obj, 2)
    elif isinstance(obj, list):
        return [round_floats(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: round_floats(v) for k, v in obj.items()}
    return obj


async def compress_one(fp: Path) -> dict:
    """压缩单个文件。"""
    path = str(fp)
    orig_size = fp.stat().st_size
    try:
        async with aiofiles.open(path, 'r', encoding='utf-8') as f:
            content = await f.read()

        data = json.loads(content)
        compressed = json.dumps(
            round_floats(data),
            ensure_ascii=False,
            separators=(',', ':'),
        )

        async with aiofiles.open(path, 'w', encoding='utf-8') as f:
            await f.write(compressed)
            await f.write('\n')

        new_size = os.path.getsize(path)
        return {"file": str(fp), "orig": orig_size, "new": new_size,
                "saved": orig_size - new_size, "status": "ok"}
    except Exception as e:
        return {"file": str(fp), "orig": orig_size, "new": orig_size,
                "saved": 0, "status": f"error: {e}"}


async def estimate_one(fp: Path) -> dict:
    """dry-run 估算：只读 + 计算大小，不写回。"""
    try:
        async with aiofiles.open(str(fp), 'r', encoding='utf-8') as f:
            content = await f.read()
        data = json.loads(content)
        compressed = json.dumps(
            round_floats(data),
            ensure_ascii=False,
            separators=(',', ':'),
        )
        return {"file": str(fp), "orig": fp.stat().st_size,
                "new": len(compressed) + 1, "status": "ok"}
    except Exception as e:
        return {"file": str(fp), "orig": fp.stat().st_size,
                "new": fp.stat().st_size, "status": f"error: {e}"}


async def worker(worker_id: int, queue: asyncio.Queue, dry_run: bool,
                 stats: dict):
    """固定 worker：从队列取文件路径，处理，更新统计。"""
    while True:
        fp = await queue.get()
        if fp is None:  # 结束信号
            queue.task_done()
            break
        try:
            result = await (estimate_one(fp) if dry_run else compress_one(fp))
            stats["total_orig"] += result["orig"]
            stats["total_new"] += result["new"]
            if result["status"] == "ok":
                stats["ok_count"] += 1
            else:
                stats["err_count"] += 1
                stats["errors"].append(f"{result['file']} | {result['status']}")
            stats["done_count"] += 1
        except Exception as e:
            stats["err_count"] += 1
            stats["errors"].append(f"{fp} | {e}")
            stats["done_count"] += 1
        finally:
            queue.task_done()


async def reporter(print_interval: int, stats: dict, total: int, t0: float):
    """定期打印进度（每 print_interval 个文件打印一次）。"""
    while stats["done_count"] < total:
        await asyncio.sleep(0.5)
        done = stats["done_count"]
        if done > 0 and done % print_interval == 0:
            elapsed = time.time() - t0
            saved = stats["total_orig"] - stats["total_new"]
            print(f"  进度: {done}/{total}, 节省 {saved / 1024**3:.2f} GB, "
                  f"耗时 {elapsed:.0f}s")


async def main():
    parser = argparse.ArgumentParser(description="压缩 labels JSON 文件")
    parser.add_argument(
        "root",
        nargs="?",
        default="/data2/caimoyao/api_outputs/temp",
        help="labels 根目录（默认: /data2/caimoyao/api_outputs/temp）",
    )
    parser.add_argument(
        "--concurrency", type=int, default=10,
        help="协程并发数（默认: 20）",
    )
    parser.add_argument(
        "--print-interval", type=int, default=1000,
        help="每处理多少个文件打印一次进度（默认: 1000）",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅统计，不写入",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"目录不存在: {root}")
        return

    # 先统计文件总数（rglob 迭代器遍历一遍）
    print("正在扫描文件...")
    t_scan = time.time()
    all_files = list(root.rglob("labels/*.json"))
    if not all_files:
        print(f"未找到任何 labels/*.json 文件")
        return
    scan_time = time.time() - t_scan
    total = len(all_files)
    print(f"找到 {total} 个 label 文件 (扫描耗时 {scan_time:.0f}s)")
    print(f"开始{'统计' if args.dry_run else '压缩'} (并发 {args.concurrency})...")
    t0 = time.time()

    # 共享统计
    stats = {
        "total_orig": 0,
        "total_new": 0,
        "ok_count": 0,
        "err_count": 0,
        "done_count": 0,
        "errors": [],
    }

    # 生产者-消费者：固定 worker 从队列取文件
    queue = asyncio.Queue(maxsize=args.concurrency * 4)

    # 启动 worker
    workers = [
        asyncio.create_task(worker(i, queue, args.dry_run, stats))
        for i in range(args.concurrency)
    ]

    # 启动进度 reporter
    rpt = asyncio.create_task(reporter(args.print_interval, stats, total, t0))

    # 生产者：懒加载文件路径入队
    for fp in all_files:
        await queue.put(fp)

    # 发送结束信号
    for _ in range(args.concurrency):
        await queue.put(None)

    # 等待所有 worker 完成
    await asyncio.gather(*workers)
    rpt.cancel()

    elapsed = time.time() - t0
    saved = stats["total_orig"] - stats["total_new"]
    ratio = (stats["total_new"] / stats["total_orig"] * 100) if stats["total_orig"] else 0

    print(f"\n{'='*60}")
    print(f"处理完成: {stats['ok_count']} 成功, {stats['err_count']} 失败")
    print(f"原始大小: {stats['total_orig'] / 1024**3:.2f} GB")
    print(f"压缩后:   {stats['total_new'] / 1024**3:.2f} GB")
    print(f"节省空间: {saved / 1024**3:.2f} GB ({saved / stats['total_orig'] * 100:.1f}%)")
    print(f"压缩率:   {ratio:.1f}%")
    print(f"耗时:     {elapsed:.0f}s")
    if args.dry_run:
        print(f"\n这是 dry-run 模式，未写入任何文件。去掉 --dry-run 执行实际压缩。")

    if stats["errors"]:
        log_path = "/data2/caimoyao/utils/compress_errors.log"
        if args.dry_run:
            log_path = "/data2/caimoyao/utils/compress_errors_dryrun.log"
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"压缩错误日志 - {datetime.datetime.now()}\n")
            f.write(f"共 {len(stats['errors'])} 个错误\n\n")
            for err in stats["errors"]:
                f.write(err + '\n')
        print(f"\n错误日志已保存: {log_path} ({len(stats['errors'])} 条)")


if __name__ == "__main__":
    asyncio.run(main())
