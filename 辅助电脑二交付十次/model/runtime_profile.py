"""运行期资源实测（第六轮交付要求：`CPU/内存实测`）。

任务书要求交付里附「总耗时、CPU/内存实测」。此前只报了 `time.perf_counter()`
的墙钟时间，没有任何 CPU / 内存数字，所以本模块把这两项补上。

**不引入新依赖**：项目 `requirements.txt` 里没有 `psutil`，本机也没有；
Windows 上 `resource` 模块不存在。因此：

  CPU 时间 ：`time.process_time()`（stdlib，进程内 user+system，跨平台）
  峰值内存 ：Windows 走 `ctypes` → `psapi.GetProcessMemoryInfo` 的
             `PeakWorkingSetSize`；POSIX 走 `resource.getrusage(RUSAGE_SELF)`
             的 `ru_maxrss`。两者都取不到时返回 `None` —— **不猜、不填 0**。

`snapshot()` 返回的字典整体并入 run manifest / permutation summary，使「实测」
有据可查，而不是文档里的一句声明。
"""

from __future__ import annotations

import platform
import sys
import time
from pathlib import Path

__all__ = ["cpu_seconds", "peak_rss_bytes", "snapshot"]


def cpu_seconds() -> float:
    """进程累计 CPU 时间（user + system），秒。"""
    return float(time.process_time())


def _peak_rss_windows() -> int | None:
    import ctypes
    from ctypes import wintypes

    class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = _PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(counters)
    # Windows 7+ ：K32GetProcessMemoryInfo 在 kernel32，psapi 为兼容别名
    for dll_name, func_name in (("kernel32", "K32GetProcessMemoryInfo"),
                                ("psapi", "GetProcessMemoryInfo")):
        try:
            dll = ctypes.WinDLL(dll_name)
            func = getattr(dll, func_name)
        except (OSError, AttributeError):
            continue
        func.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PROCESS_MEMORY_COUNTERS),
                         wintypes.DWORD]
        func.restype = wintypes.BOOL
        if func(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters),
                counters.cb):
            return int(counters.PeakWorkingSetSize)
    return None


def _peak_rss_posix() -> int | None:
    try:
        import resource
    except ImportError:            # pragma: no cover - Windows 上没有
        return None
    # Linux 报 KiB，macOS 报字节 —— 按单位换算，不假设
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def peak_rss_bytes() -> int | None:
    """进程峰值常驻内存（字节）。取不到返回 None。"""
    if sys.platform == "win32":
        return _peak_rss_windows()
    return _peak_rss_posix()


def snapshot(*, wall_seconds: float, cpu_before: float, rss_before: int | None = None) -> dict:
    """一次运行的资源实测快照。

    `cpu_before` 由调用方在计时开始时取（`cpu_seconds()`），使 CPU 时间是
    **本次运行**的增量而非进程生命周期累计。
    """
    peak = peak_rss_bytes()
    result = {
        "wall_seconds": round(float(wall_seconds), 3),
        "cpu_seconds": round(cpu_seconds() - float(cpu_before), 3),
        "peak_rss_bytes": peak,
        "peak_rss_mib": None if peak is None else round(peak / (1024 * 1024), 1),
        "cpu_count": None,
        "platform": f"{platform.system()} {platform.release()}",
        "python": sys.version.split()[0],
    }
    try:
        import os
        result["cpu_count"] = os.cpu_count()
    except Exception:              # noqa: BLE001 - 资源信息缺失不阻断运行
        pass
    return result


def script_entry_snapshot(started: float, cpu_before: float) -> dict:
    """给入口用的便捷包装：`started` 为 `time.perf_counter()` 的起点。"""
    return snapshot(wall_seconds=time.perf_counter() - started, cpu_before=cpu_before)


def _cli() -> int:
    """自检：跑一小段负载，打印实测快照（供 RUNBOOK / 测试核对）。"""
    cpu0 = cpu_seconds()
    t0 = time.perf_counter()
    acc = 0.0
    for i in range(1, 2_000_001):
        acc += i ** 0.5
    snap = script_entry_snapshot(t0, cpu0)
    print(f"self-check acc={acc:.3f}")
    for key, value in snap.items():
        print(f"  {key}: {value}")
    if snap["peak_rss_bytes"] is None:
        print("  ! peak_rss_bytes 取不到（本平台无可用接口）", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
