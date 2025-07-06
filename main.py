import datetime
import json
import os
import shlex
import shutil
import statistics
import subprocess
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pp
from tempfile import TemporaryDirectory
from typing import Any, Callable, Optional, Protocol, Self, Union

from joblib import Parallel, delayed


# unwrap function for Optional types
def unwrap[T](value: Optional[T]) -> T:
    if value is None:
        raise ValueError("Value is None")
    return value


def auto_find_bin(bin_name: str) -> Path:
    bin_match = shutil.which(bin_name)
    bin_str = unwrap(bin_match)
    bin_path = Path(bin_str)
    if not bin_path.exists():
        raise FileNotFoundError(f"Binary {bin_name} not found at {bin_path}")
    if not bin_path.is_file():
        raise FileNotFoundError(f"Binary {bin_name} is not a file at {bin_path}")
    return bin_path


def check_return_code(
    p: subprocess.CompletedProcess, user_message: str | None = None
) -> None:
    if p.returncode != 0:
        raise RuntimeError(
            f"Process failed with return code {p.returncode}\n"
            f"COMMAND: {p.args}\n"
            f"STDOUT:\n{p.stdout.decode('utf-8') if p.stdout else 'None'}\n"
            f"STDERR:\n{p.stderr.decode('utf-8') if p.stderr else 'None'}\n"
            f"USERMSG:\n{user_message if user_message else ''}"
        )


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
    tcl_fp = cwd / tcl_fp
    if not tcl_fp.exists():
        raise FileNotFoundError(f"TCL file {tcl_fp} does not exist.")

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

    check_return_code(p)

    dt = t1 - t0
    return dt


def run_vivado(
    tcl_fp: Path, vivado_bin: Path, cwd: Path, env: dict[str, str] | None = None
) -> float:
    tcl_fp = cwd / tcl_fp
    if not tcl_fp.exists():
        raise FileNotFoundError(f"TCL file {tcl_fp} does not exist.")

    args = [
        str(vivado_bin),
        "-mode",
        "batch",
        "-source",
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

    check_return_code(p)

    dt = t1 - t0
    return dt


def run_vtr(
    vtr_bin: Path | None,
    vtr_run_fp: Path,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> float:
    vtr_run_fp = cwd / vtr_run_fp
    if not vtr_run_fp.exists():
        raise FileNotFoundError(f"VTR run file {vtr_run_fp} does not exist.")

    vtr_run = vtr_run_fp.read_text().strip()
    args = shlex.split(vtr_run)
    assert args[0] == "$VPR_BIN", "First argument must be $VPR_BIN"
    args[0] = str(vtr_bin)

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

    check_return_code(p, user_message=f"cwd: {cwd.resolve()}")

    dt = t1 - t0
    return dt


def run_yosys(
    yosys_script_fp: Path,
    yosys_bin: Path,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> float:
    yosys_script_fp = cwd / yosys_script_fp
    if not yosys_script_fp.exists():
        raise FileNotFoundError(f"Yosys script {yosys_script_fp} does not exist.")

    args = [
        str(yosys_bin),
        "-s",
        str(yosys_script_fp),
        "--logfile",
        "yosys.log",
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

    check_return_code(p)

    dt = t1 - t0
    return dt


class TToolRun(Protocol):
    def __call__(self, cwd: Path, env: Optional[dict[str, str]] = None) -> float: ...


def run_on_disk(tool_fn: TToolRun, test_design: TestDesign, work_dir: Path) -> float:
    if not work_dir.exists():
        raise FileNotFoundError(f"Work directory {work_dir} does not exist.")
    iso_timestamp: str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with TemporaryDirectory(
        dir=work_dir, delete=True, prefix=f"{iso_timestamp}__"
    ) as tmpdir:
        temp_fp = Path(tmpdir)
        test_design_in_temp = test_design.copy_to_dir(temp_fp)
        dt = tool_fn(cwd=test_design_in_temp.design_dir, env=None)
    return dt


def run_in_memory(
    tool_fn: TToolRun,
    test_design: TestDesign,
) -> float:
    iso_timestamp: str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with TemporaryDirectory(
        dir="/dev/shm", delete=True, prefix=f"{iso_timestamp}__"
    ) as tmpdir:
        temp_fp = Path(tmpdir)
        test_design_in_temp = test_design.copy_to_dir(temp_fp)
        dt = tool_fn(cwd=test_design_in_temp.design_dir, env=None)

    return dt


def run_in_memory_and_use_mimalloc(
    tool_fn: TToolRun,
    test_design: TestDesign,
    mimalloc_so_fp: Path,
) -> float:
    iso_timestamp: str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with TemporaryDirectory(
        dir="/dev/shm", delete=True, prefix=f"{iso_timestamp}__"
    ) as tmpdir:
        temp_fp = Path(tmpdir)
        test_design_in_temp = test_design.copy_to_dir(temp_fp)
        dt = tool_fn(
            cwd=test_design_in_temp.design_dir,
            env={
                "LD_PRELOAD": str(mimalloc_so_fp),
                "MIMALLOC_ARENA_EAGER_COMMIT": "2",
                "MIMALLOC_PURGE_DELAY": "200",
            },
        )
    return dt


def run_use_mimalloc(
    tool_fn: TToolRun,
    test_design: TestDesign,
    work_dir: Path,
    mimalloc_so_fp: Path,
) -> float:
    if not work_dir.exists():
        raise FileNotFoundError(f"Work directory {work_dir} does not exist.")
    iso_timestamp: str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with TemporaryDirectory(
        dir=work_dir, delete=True, prefix=f"{iso_timestamp}__"
    ) as tmpdir:
        temp_fp = Path(tmpdir)
        test_design_in_temp = test_design.copy_to_dir(temp_fp)
        dt = tool_fn(
            cwd=test_design_in_temp.design_dir,
            env={
                "LD_PRELOAD": str(mimalloc_so_fp),
                "MIMALLOC_ARENA_EAGER_COMMIT": "2",
                "MIMALLOC_PURGE_DELAY": "200",
            },
        )
    return dt


def run_use_mimalloc_and_lcall(
    tool_fn: TToolRun,
    test_design: TestDesign,
    work_dir: Path,
    mimalloc_so_fp: Path,
) -> float:
    if not work_dir.exists():
        raise FileNotFoundError(f"Work directory {work_dir} does not exist.")
    iso_timestamp: str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with TemporaryDirectory(
        dir=work_dir, delete=True, prefix=f"{iso_timestamp}__"
    ) as tmpdir:
        temp_fp = Path(tmpdir)
        test_design_in_temp = test_design.copy_to_dir(temp_fp)
        dt = tool_fn(
            cwd=test_design_in_temp.design_dir,
            env={
                "LD_PRELOAD": str(mimalloc_so_fp),
                "MIMALLOC_ARENA_EAGER_COMMIT": "2",
                "MIMALLOC_PURGE_DELAY": "200",
                "LC_ALL": "C",
            },
        )
    return dt


def create_vtr_tool_fn(
    vtr_bin: Path, vtr_run_fp: Path, use_mimalloc_binary: bool = False
) -> TToolRun:
    def tool_fn(cwd: Path, env: Optional[dict[str, str]] = None) -> float:
        return run_vtr(
            vtr_bin=vtr_bin,
            vtr_run_fp=vtr_run_fp,
            cwd=cwd,
            env=env,
        )

    return tool_fn


def create_tool_runner(
    tool: str, design: TestDesign, mimalloc_enabled: bool = False
) -> TToolRun:
    match tool:
        case "vitis_hls":
            tcl_fp = Path("run.tcl")
            return lambda cwd, env=None: run_vitis_hls(
                tcl_fp=tcl_fp,
                vitis_hls_bin=BIN_VITIS_HLS,
                cwd=cwd,
                env=env,
            )
        case "vivado":
            tcl_fp = Path("run.tcl")
            return lambda cwd, env=None: run_vivado(
                tcl_fp=tcl_fp,
                vivado_bin=BIN_VIVADO,
                cwd=cwd,
                env=env,
            )
        case "vtr":
            vtr_run_fp = Path("run.txt")
            vtr_bin = BIN_VTR_MIMALLOC if mimalloc_enabled else BIN_VTR
            return create_vtr_tool_fn(vtr_bin, vtr_run_fp)
        case "yosys":
            yosys_script_fp = design.design_dir / "run.ys"
            return lambda cwd, env=None: run_yosys(
                yosys_script_fp=yosys_script_fp,
                yosys_bin=DIR_CURRENT / "yosys",
                cwd=cwd,
                env=env,
            )
        case _:
            raise ValueError(f"Unknown tool {tool}")


def create_run_modes(
    tool: str, design: TestDesign, work_dir: Path, mimalloc_so_fp: Path
) -> list[tuple[str, Callable[[], float], list[str]]]:
    if tool == "vtr":
        return [
            (
                "disk",
                lambda: run_on_disk(
                    create_tool_runner(tool, design, False), design, work_dir
                ),
                ["disk"],
            ),
            (
                "shm",
                lambda: run_in_memory(create_tool_runner(tool, design, False), design),
                ["shm"],
            ),
            (
                "shm+mimalloc",
                lambda: run_in_memory(create_tool_runner(tool, design, True), design),
                ["shm", "mimalloc"],
            ),
            (
                "disk+mimalloc",
                lambda: run_on_disk(
                    create_tool_runner(tool, design, True), design, work_dir
                ),
                ["disk", "mimalloc"],
            ),
            (
                "disk+mimalloc+lcall",
                lambda: run_use_mimalloc_and_lcall(
                    create_tool_runner(tool, design, True),
                    design,
                    work_dir,
                    mimalloc_so_fp,
                ),
                ["disk", "mimalloc", "lcall"],
            ),
        ]
    else:
        tool_fn = create_tool_runner(tool, design, False)
        return [
            ("disk", lambda: run_on_disk(tool_fn, design, work_dir), ["disk"]),
            ("shm", lambda: run_in_memory(tool_fn, design), ["shm"]),
            (
                "shm+mimalloc",
                lambda: run_in_memory_and_use_mimalloc(tool_fn, design, mimalloc_so_fp),
                ["shm", "mimalloc"],
            ),
            (
                "disk+mimalloc",
                lambda: run_use_mimalloc(tool_fn, design, work_dir, mimalloc_so_fp),
                ["disk", "mimalloc"],
            ),
            (
                "disk+mimalloc+lcall",
                lambda: run_use_mimalloc_and_lcall(
                    tool_fn, design, work_dir, mimalloc_so_fp
                ),
                ["disk", "mimalloc", "lcall"],
            ),
        ]


@dataclass
class Stats:
    mean: float
    std: float
    var: float
    median: float
    p25: float
    p75: float
    p05: float
    p95: float
    min_time: float
    max_time: float


def compute_stats(times: list[float]) -> Stats:
    mean = statistics.mean(times)
    std = statistics.stdev(times)
    var = statistics.variance(times)
    p25 = statistics.quantiles(times, n=4)[0]  # 25th percentile
    p75 = statistics.quantiles(times, n=4)[2]  # 75th percentile
    p05 = statistics.quantiles(times, n=20)[1]  # 5th percentile
    p95 = statistics.quantiles(times, n=20)[18]  # 95th percentile
    median = statistics.median(times)

    min_time = min(times)
    max_time = max(times)

    return Stats(
        mean=mean,
        std=std,
        var=var,
        median=median,
        p25=p25,
        p75=p75,
        p05=p05,
        p95=p95,
        min_time=min_time,
        max_time=max_time,
    )


@dataclass
class ExpData:
    name: str
    times: list[float]
    stats: Stats
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, float]:
        return {
            "name": self.name,
            "tags": self.tags,
            "times": self.times,
            "mean": self.stats.mean,
            "std": self.stats.std,
            "var": self.stats.var,
            "median": self.stats.median,
            "p25": self.stats.p25,
            "p75": self.stats.p75,
            "p05": self.stats.p05,
            "p95": self.stats.p95,
            "min_time": self.stats.min_time,
            "max_time": self.stats.max_time,
        }

    def to_json_str(self) -> str:
        return json.dumps(self.to_dict(), indent=4)

    def to_json_file(self, fp: Path) -> None:
        fp.write_text(self.to_json_str())

    @classmethod
    def from_json_str(cls, json_str: str) -> Self:
        data = json.loads(json_str)
        stats = Stats(
            mean=data["mean"],
            std=data["std"],
            var=data["var"],
            median=data["median"],
            p25=data["p25"],
            p75=data["p75"],
            p05=data["p05"],
            p95=data["p95"],
            min_time=data["min_time"],
            max_time=data["max_time"],
        )
        return cls(
            name=data["name"],
            times=data["times"],
            stats=stats,
            tags=data.get("tags", []),
        )

    @classmethod
    def from_json_file(cls, fp: Path) -> Self:
        if not fp.exists():
            raise FileNotFoundError(f"File {fp} does not exist.")
        json_str = fp.read_text()
        return cls.from_json_str(json_str)


if __name__ == "__main__":
    MIMALLOC_SO_FP = Path("/usr/scratch/common/mimalloc/libmimalloc.so")

    DIR_CURRENT = Path(__file__).parent

    TEST_DESIGNS_DIR = DIR_CURRENT / "test_designs"

    # test_projects/vitis_hls__nesting
    # test_projects/vitis_hls__simple
    # test_projects/vivado__simple
    # test_projects/vtr__mcnc_big
    # test_projects/vtr__mcnc_big_search
    # test_projects/vtr__mcnc_simple
    # test_projects/yosys__complex
    # test_projects/yosys__simple

    test_designs: dict[str, list[TestDesign]] = {}
    for design_dir in TEST_DESIGNS_DIR.iterdir():
        if design_dir.is_dir():
            prefix = design_dir.name.split("__")[0]
            if prefix not in test_designs:
                test_designs[prefix] = []
            test_designs[prefix].append(TestDesign(design_dir=design_dir))
    print("Available test designs:")
    pp(test_designs)

    BIN_VTR = DIR_CURRENT / "tools" / "vpr"
    BIN_VTR_MIMALLOC = DIR_CURRENT / "tools" / "vpr-mimalloc"
    BIN_VITIS_HLS = auto_find_bin("vitis_hls")
    BIN_VIVADO = auto_find_bin("vivado")
    BIN_YOSYS = auto_find_bin("yosys")

    N_SAMPLES = 48
    N_PARALLEL = 48

    WORK_DIR = DIR_CURRENT / "work_dir"
    if not WORK_DIR.exists():
        WORK_DIR.mkdir(parents=True)

    RESULTS_DIR = DIR_CURRENT / "results"
    if not RESULTS_DIR.exists():
        RESULTS_DIR.mkdir(parents=True)

    # TOOLS_TO_TEST = ["vitis_hls"]
    # DESIGNS_TO_FILTER = [
    #     "vitis_hls__nesting",
    # ]

    TOOLS_TO_TEST = [
        "vtr",
        # "vitis_hls",
    ]
    DESIGNS_TO_FILTER = [
        "vtr__mcnc_big",
        "vtr__mcnc_big_search",
        "vtr__mcnc_simple",
        # "vitis_hls__nesting",
        # "vitis_hls__simple",
    ]
    RUN_MODES_TO_TEST = [
        # "disk",
        # "shm",
        # "shm+mimalloc",
        # "disk+mimalloc",
        "disk+mimalloc+lcall",
    ]

    for tool in TOOLS_TO_TEST:
        designs = test_designs[tool]
        if len(designs) == 0:
            raise ValueError(f"No test designs found for tool {tool}")

        designs_to_test = designs
        if DESIGNS_TO_FILTER is not None and len(DESIGNS_TO_FILTER) > 0:
            designs_to_test = [
                design
                for design in designs
                if design.design_dir.name in DESIGNS_TO_FILTER
            ]

        for design in designs_to_test:
            run_modes = create_run_modes(tool, design, WORK_DIR, MIMALLOC_SO_FP)
            for mode_name, runner, mode_tags in run_modes:
                if mode_name not in RUN_MODES_TO_TEST:
                    continue
                print(f"Running: {tool} / {design.design_dir.name} / {mode_name}")
                times = Parallel(n_jobs=N_PARALLEL)(
                    delayed(runner)() for _ in range(N_SAMPLES)
                )
                run_stats = compute_stats(times)
                exp_name = f"{tool}__{design.design_dir.name}__{mode_name}"
                exp_data = ExpData(
                    name=exp_name,
                    times=times,
                    stats=run_stats,
                    tags=[tool, design.design_dir.name] + mode_tags,
                )
                result_fp = RESULTS_DIR / f"{exp_name}.json"
                exp_data.to_json_file(result_fp)
                print(f"Saved results to {result_fp}")
