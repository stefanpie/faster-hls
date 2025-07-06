from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.stats import mannwhitneyu, shapiro

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


for r in exp_results:
    print(f"Experiment: {r.name}")
    print(f"  - tags: {', '.join(r.tags)}")


TOOLS = ["vtr", "vitis_hls"]

MAP_COLORS = {
    "disk": "red",
    "shm": "orange",
    "shm+mimalloc": "green",
    "disk+mimalloc": "blue",
    "disk+mimalloc+lcall": "purple",
}

MAP_LABELS = {
    "disk": "on-disk (base)",
    "shm": "in-memory",
    "shm+mimalloc": "in-memory + mimalloc",
    "disk+mimalloc": "on-disk + mimalloc",
    "disk+mimalloc+lcall": "on-disk + mimalloc + LC_ALL=C",
}

MAP_ZORDER = {
    "disk": 1,
    "shm": 2,
    "shm+mimalloc": 3,
    "disk+mimalloc": 4,
    "disk+mimalloc+lcall": 5,
}


def gather_data(
    exp_results: list[ExpData],
    design_name: str,
) -> pd.DataFrame:
    # filter results with the deisgn name in the tag

    filtered_results = [r for r in exp_results if design_name in r.tags]

    tag_sets = [set(r.tags) for r in filtered_results]
    for ts in tag_sets:
        ts.remove(design_name)
        for tool in TOOLS:
            if tool in ts:
                ts.remove(tool)

    disk_result = next(
        (r for r, tag_set in zip(filtered_results, tag_sets) if tag_set == {"disk"}),
        None,
    )
    shm_result = next(
        (r for r, tag_set in zip(filtered_results, tag_sets) if tag_set == {"shm"}),
        None,
    )
    shm_mimalloc_result = next(
        (
            r
            for r, tag_set in zip(filtered_results, tag_sets)
            if tag_set == {"shm", "mimalloc"}
        ),
        None,
    )
    disk_mimalloc_result = next(
        (
            r
            for r, tag_set in zip(filtered_results, tag_sets)
            if tag_set == {"disk", "mimalloc"}
        ),
        None,
    )
    disk_mimalloc_lcall_result = next(
        (
            r
            for r, tag_set in zip(filtered_results, tag_sets)
            if tag_set == {"disk", "mimalloc", "lcall"}
        ),
        None,
    )

    disk_result = unwrap(disk_result)
    shm_result = unwrap(shm_result)
    shm_mimalloc_result = unwrap(shm_mimalloc_result)
    disk_mimalloc_result = unwrap(disk_mimalloc_result)
    disk_mimalloc_lcall_result = unwrap(disk_mimalloc_lcall_result)

    # gather all results into a single dataframe
    data = []
    for r, label in zip(
        [
            disk_result,
            shm_result,
            shm_mimalloc_result,
            disk_mimalloc_result,
            disk_mimalloc_lcall_result,
        ],
        ["disk", "shm", "shm+mimalloc", "disk+mimalloc", "disk+mimalloc+lcall"],
    ):
        for t in r.times:
            data.append((t, label))

    df_plot = pd.DataFrame(data, columns=["time", "label"])
    return df_plot


def plot_results_single_design(
    exp_results: list[ExpData],
    design_name: str,
) -> tuple[Figure, Axes]:
    df_plot = gather_data(exp_results, design_name)

    fig, ax = plt.subplots(figsize=(6, 4))

    palette, hue_order = (
        MAP_COLORS,
        ["disk", "shm", "shm+mimalloc", "disk+mimalloc", "disk+mimalloc+lcall"],
    )

    zorder_map = MAP_ZORDER

    for label in hue_order:
        sns.kdeplot(
            data=df_plot[df_plot["label"] == label],
            x="time",
            hue="label",
            fill=True,
            palette=palette,
            ax=ax,
            alpha=0.2,
            label=MAP_LABELS[label],
            zorder=zorder_map[label],
        )

    leg_handles = [
        plt.Rectangle((0, 0), 1, 1, color=MAP_COLORS[case], label=MAP_LABELS[case])
        for case in hue_order
    ]

    ax.legend(
        # ["on-disk (base)", "in-memory", "in-memory + mimalloc"],
        handles=leg_handles,
        loc="upper left",
        title="Runtime Acceleration Method",
        fontsize=8,
        title_fontsize=10,
    )

    ax.set_title(f'Execution Runtimes for "{design_name}" Case')
    ax.set_xlabel("Runtime (seconds)")
    ax.set_ylabel("Frequency")

    return fig, ax


def build_report_single_design(
    exp_results: list[ExpData],
    design_name: str,
) -> str:
    df_plot = gather_data(exp_results, design_name)

    # for each label, compute the mean, median, std, and 5% and 95% quantiles
    stats = (
        df_plot.groupby("label")["time"]
        .agg(["mean", "median", lambda x: x.quantile(0.05), lambda x: x.quantile(0.95)])
        .reset_index()
    )
    stats.columns = ["label", "mean", "median", "5%", "95%"]

    # print(f"## Statistics for {design_name}")
    # print(stats.to_markdown())

    # now computed normalized results normalized to the disk result fore all stats
    stats_norm = stats.copy()
    stats_norm["mean"] = (
        stats_norm["mean"] / stats_norm[stats_norm["label"] == "disk"]["mean"].values[0]
    )
    stats_norm["median"] = (
        stats_norm["median"]
        / stats_norm[stats_norm["label"] == "disk"]["median"].values[0]
    )
    stats_norm["5%"] = (
        stats_norm["5%"] / stats_norm[stats_norm["label"] == "disk"]["5%"].values[0]
    )
    stats_norm["95%"] = (
        stats_norm["95%"] / stats_norm[stats_norm["label"] == "disk"]["95%"].values[0]
    )

    # clean up this table and make it pretty as a markdown table
    stats_norm = stats_norm.drop(columns=["label"])
    stats_norm = stats_norm.round(2)
    stats_norm.index = [
        "disk",
        "shm",
        "shm+mimalloc",
        "disk+mimalloc",
        "disk+mimalloc+lcall",
    ]

    # add an index label
    stats_norm.index.name = "Method"

    stats_norm.columns = ["Mean", "Median", "5% Quantile", "95% Quantile"]

    print(f"**Results for {design_name}**")
    print(stats_norm.to_markdown())
    print()

    return f"**Results for {design_name}**\n\n{stats_norm.to_markdown()}\n"


def build_stats_tests_single_design(
    exp_results: list[ExpData],
    design_name: str,
) -> str:
    df_plot = gather_data(exp_results, design_name)

    report = f"# Statistical Analysis Report: {design_name}\n"

    # test each set of valyues for normality using the Shapiro-Wilk test

    # report += f"## Normality Tests for {design_name}\n"
    # report += "### Shapiro-Wilk Test Results\n"

    # for label in df_plot["label"].unique():
    #     times = df_plot[df_plot["label"] == label]["time"]
    #     stat, p_value = shapiro(times)
    #     report += f"- {label}: W-statistic = {stat:.2f}, p-value = {p_value:.4f}\n"

    # perform a non-parametric test to compare the medians of the shm and shm+mimalloc results against the disk result
    # our null hypothesis is that the medians are equal
    # our alternative hypothesis is that the medians of the shm and shm+mimalloc results are less than the disk result

    disk_times = df_plot[df_plot["label"] == "disk"]["time"]
    shm_times = df_plot[df_plot["label"] == "shm"]["time"]
    shm_mimalloc_times = df_plot[df_plot["label"] == "shm+mimalloc"]["time"]
    disk_mimalloc_times = df_plot[df_plot["label"] == "disk+mimalloc"]["time"]
    disk_mimalloc_lcall_times = df_plot[df_plot["label"] == "disk+mimalloc+lcall"][
        "time"
    ]

    stat_shm, p_shm = mannwhitneyu(shm_times, disk_times, alternative="less")
    stat_shm_mimalloc, p_shm_mimalloc = mannwhitneyu(
        shm_mimalloc_times, disk_times, alternative="less"
    )
    stat_disk_mimalloc, p_disk_mimalloc = mannwhitneyu(
        disk_mimalloc_times, disk_times, alternative="less"
    )
    stat_disk_mimalloc_lcall, p_disk_mimalloc_lcall = mannwhitneyu(
        disk_mimalloc_lcall_times, disk_times, alternative="less"
    )

    report += "## Mann-Whitney U Test Results\n"
    report += f"- SHM < Disk: U-statistic = {stat_shm:.2f}, p-value = {p_shm:.4f}\n"
    report += f"- SHM + Mimalloc < Disk: U-statistic = {stat_shm_mimalloc:.2f}, p-value = {p_shm_mimalloc:.4f}\n"
    report += f"- Disk + Mimalloc < Disk: U-statistic = {stat_disk_mimalloc:.2f}, p-value = {p_disk_mimalloc:.4f}\n"
    report += f"- Disk + Mimalloc + LC_ALL=C < Disk: U-statistic = {stat_disk_mimalloc_lcall:.2f}, p-value = {p_disk_mimalloc_lcall:.4f}\n"

    return report


designs_to_plot = (
    ["vtr__mcnc_simple", "vtr__mcnc_big", "vtr__mcnc_big_search"]
    # + ["vitis_hls__simple"]
    # + ["vitis_hls__nesting"]
)

for design_name in designs_to_plot:
    fig, ax = plot_results_single_design(exp_results, design_name)
    fig.tight_layout()
    fig.savefig(DIR_FIGURES / f"{design_name}_execution_times.png", dpi=300)
    plt.close(fig)

    build_report_single_design(exp_results, design_name)

    stats = build_stats_tests_single_design(exp_results, design_name)
    print(stats)
