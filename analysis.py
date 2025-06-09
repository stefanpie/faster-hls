from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from main import ExpData, unwrap

DIR_CURRENT = Path(__file__).parent


DIR_FIGURES = DIR_CURRENT / "figures"
if not DIR_FIGURES.exists():
    DIR_FIGURES.mkdir(parents=True)


DIR_RESULTS = DIR_CURRENT / "results"


exp_results: list[ExpData] = []
for file in sorted(DIR_RESULTS.glob("*.json")):
    exp_result = ExpData.from_json_file(file)
    exp_results.append(exp_result)


# pp(exp_results)
for r in exp_results:
    print(f"Experiment: {r.name}")
    print(f"  - tags: {', '.join(r.tags)}")


def plot_results_single_design(
    exp_results: list[ExpData],
    design_name: str,
) -> tuple[Figure, Axes]:
    # filter results with the deisgn name in the tag

    filtered_results = [r for r in exp_results if design_name in r.tags]
    assert len(filtered_results) == 3

    disk_result = next(
        (
            r
            for r in filtered_results
            if "disk" in r.tags and "shm" not in r.tags and "mimalloc" not in r.tags
        ),
        None,
    )
    shm_result = next(
        (
            r
            for r in filtered_results
            if "disk" not in r.tags and "shm" in r.tags and "mimalloc" not in r.tags
        ),
        None,
    )
    shm_mimalloc_result = next(
        (
            r
            for r in filtered_results
            if "disk" not in r.tags and "shm" in r.tags and "mimalloc" in r.tags
        ),
        None,
    )

    disk_result = unwrap(disk_result)
    shm_result = unwrap(shm_result)
    shm_mimalloc_result = unwrap(shm_mimalloc_result)

    data = []
    for r, label in zip(
        [disk_result, shm_result, shm_mimalloc_result],
        ["disk", "shm", "shm+mimalloc"],
    ):
        for t in r.times:
            data.append((t, label))

    df_plot = pd.DataFrame(data, columns=["time", "label"])

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.kdeplot(
        data=df_plot,
        x="time",
        hue="label",
        fill=True,
        palette=["blue", "orange", "green"],
        ax=ax,
        alpha=0.2,
    )

    ax.set_title(f"Execution Times for {design_name}")
    ax.set_xlabel("Execution Time (seconds)")
    ax.set_ylabel("Frequency")

    return fig, ax


designs_to_plot = ["vtr__mcnc_simple", "vtr__mcnc_big", "vtr__mcnc_big_search"] + [
    "vitis_hls__simple"
]


for design_name in designs_to_plot:
    fig, ax = plot_results_single_design(exp_results, design_name)
    fig.tight_layout()
    fig.savefig(DIR_FIGURES / f"{design_name}_execution_times.png", dpi=300)
    plt.close(fig)
