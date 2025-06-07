import datetime
import os
import shutil
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Self

from joblib import Parallel, delayed
from scipy import stats


@dataclass
class TestDesign:
    design_dir: Path

    def copy_to_dir(self, target_dir: Path) -> Self:
        if not target_dir.exists():
            raise FileNotFoundError(f"Target directory {target_dir} does not exist.")
        design_dir_in_target = target_dir / self.design_dir.name
        if design_dir_in_target.exists():
            raise FileExistsError(
                f"Design directory {design_dir_in_target} already exists."
            )
        shutil.copytree(self.design_dir, design_dir_in_target)
        return TestDesign(design_dir=design_dir_in_target)


def run_vitis_hls(
    tcl_fp: Path, vitis_hls_bin: Path, cwd: Path, env: dict[str, str] | None = None
) -> float:
    args = [
        str(vitis_hls_bin),
        str(tcl_fp),
    ]

    env_for_process = os.environ.copy()
    if env is not None and len(env) > 0:
        for k, v in env.items():
            env_for_process[k] = v

    t0 = time.time()
    p = subprocess.run(
        args,
        cwd=cwd,
        env=env_for_process,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    t1 = time.time()

    if p.returncode != 0:
        raise RuntimeError(f"Vitis HLS failed with return code {p.returncode}")

    dt = t1 - t0
    return dt


def run_on_disk(test_design: TestDesign, work_dir: Path) -> float:
    if not work_dir.exists():
        raise FileNotFoundError(f"Work directory {work_dir} does not exist.")
    iso_timestamp: str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with TemporaryDirectory(
        dir=work_dir, delete=True, prefix=f"{iso_timestamp}__"
    ) as tmpdir:
        temp_fp = Path(tmpdir)
        test_design_in_temp = test_design.copy_to_dir(temp_fp)
        dt = run_vitis_hls(
            tcl_fp=test_design_in_temp.design_dir / "hls_run.tcl",
            vitis_hls_bin="vitis_hls",
            cwd=test_design_in_temp.design_dir,
        )
    return dt


def run_in_memory(
    test_design: TestDesign,
) -> float:
    iso_timestamp: str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with TemporaryDirectory(
        dir="/dev/shm", delete=True, prefix=f"{iso_timestamp}__"
    ) as tmpdir:
        temp_fp = Path(tmpdir)
        test_design_in_temp = test_design.copy_to_dir(temp_fp)
        dt = run_vitis_hls(
            tcl_fp=test_design_in_temp.design_dir / "hls_run.tcl",
            vitis_hls_bin="vitis_hls",
            cwd=test_design_in_temp.design_dir,
        )
    return dt


def run_in_memory_and_use_mimalloc(
    test_design: TestDesign,
    mimalloc_so_fp: Path,
) -> float:
    iso_timestamp: str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with TemporaryDirectory(
        dir="/dev/shm", delete=True, prefix=f"{iso_timestamp}__"
    ) as tmpdir:
        temp_fp = Path(tmpdir)
        test_design_in_temp = test_design.copy_to_dir(temp_fp)
        dt = run_vitis_hls(
            tcl_fp=test_design_in_temp.design_dir / "hls_run.tcl",
            vitis_hls_bin="vitis_hls",
            cwd=test_design_in_temp.design_dir,
            env={
                "LD_PRELOAD": str(mimalloc_so_fp),
                "MIMALLOC_ARENA_EAGER_COMMIT": "2",
                "MIMALLOC_PURGE_DELAY": "200",
            },
        )
    return dt


@dataclass
class Stats:
    mean: float
    std: float
    median: float
    p25: float
    p75: float
    min_time: float
    max_time: float


def compute_stats(times: list[float]) -> Stats:
    mean = statistics.mean(times)
    std = statistics.stdev(times)
    p25 = statistics.quantiles(times, n=4)[0]  # 25th percentile
    p75 = statistics.quantiles(times, n=4)[2]  # 75th percentile
    median = statistics.median(times)

    min_time = min(times)
    max_time = max(times)

    return Stats(
        mean=mean,
        std=std,
        median=median,
        p25=p25,
        p75=p75,
        min_time=min_time,
        max_time=max_time,
    )


if __name__ == "__main__":
    MIMALLOC_SO_FP = Path("/usr/scratch/common/minialloc/libmimalloc.so")

    DIR_CURRENT = Path(__file__).parent

    WORK_DIR = DIR_CURRENT / "work_dir"
    if not WORK_DIR.exists():
        WORK_DIR.mkdir(parents=True)

    TEST_DESIGN_DIR = DIR_CURRENT / "test_design__simple"
    test_design = TestDesign(design_dir=TEST_DESIGN_DIR)

    N_SAMPLES = 32
    N_PARALLEL = 32

    times_mem = Parallel(
        n_jobs=N_PARALLEL,
        backend="multiprocessing",
        verbose=10,
    )(
        delayed(run_in_memory)(d)
        for d in [deepcopy(test_design) for _ in range(N_SAMPLES)]
    )
    stats_mem = compute_stats(times_mem)
    print(f"In-Memory Stats:\n{stats_mem}")

    times_mem_mimalloc = Parallel(
        n_jobs=N_PARALLEL,
        backend="multiprocessing",
        verbose=10,
    )(
        delayed(run_in_memory_and_use_mimalloc)(d, MIMALLOC_SO_FP)
        for d in [deepcopy(test_design) for _ in range(N_SAMPLES)]
    )
    stats_mem_mimalloc = compute_stats(times_mem_mimalloc)
    print(f"In-Memory with mimalloc Stats:\n{stats_mem_mimalloc}")

    # times_disk = Parallel(
    #     n_jobs=N_PARALLEL,
    #     backend="multiprocessing",
    #     verbose=10,
    # )(
    #     delayed(run_on_disk)(d, WORK_DIR)
    #     for d in [deepcopy(test_design) for _ in range(N_SAMPLES)]
    # )
    # stats_disk = compute_stats(times_disk)
    # print(f"On-Disk Stats:\n{stats_disk}")

    # u_statistic, p_value = stats.mannwhitneyu(
    #     times_mem, times_disk, alternative="two-sided"
    # )
    # print(f"Mann-Whitney U test statistic: {u_statistic}, p-value: {p_value}")

    # _, p_value_normality_mem = stats.shapiro(times_mem)
    # _, p_value_normality_disk = stats.shapiro(times_disk)
    # print(f"Shapiro-Wilk test p-value for in-memory: {p_value_normality_mem}")
    # print(f"Shapiro-Wilk test p-value for on-disk: {p_value_normality_disk}")

    # t_statistic, p_value_t = stats.ttest_ind(times_mem, times_disk, equal_var=False)
    # print(f"T-test statistic: {t_statistic}, p-value: {p_value_t}")
