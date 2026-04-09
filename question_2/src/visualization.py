"""可视化模块：论文风格图表输出。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PALETTE = {
    "navy": "#1F3A5F",
    "blue": "#4F6D8A",
    "slate": "#7A8FA6",
    "sand": "#D9C7A3",
    "red": "#B45F4A",
    "gold": "#C8A951",
    "green": "#6F8F72",
    "ink": "#2E3440",
    "grid": "#D7DEE7",
    "bg": "#F7F8FA",
}

QUAD_COLOR = {"HH": PALETTE["navy"], "HL": PALETTE["red"], "LH": PALETTE["gold"], "LL": PALETTE["slate"]}
EFFECT_LABEL = {"direct": "直接效应", "indirect": "间接效应", "total": "总效应"}
REGION_ORDER = ["东部", "中部", "西部", "东北"]


def _save(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.2)
    fig.savefig(out_path, dpi=500, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _mark(ax, tag: str) -> None:
    ax.text(0.0, 1.03, f"({tag})", transform=ax.transAxes, ha="left", va="bottom", fontsize=11, fontweight="bold", color=PALETTE["ink"])


def _style(ax, grid_axis: str = "y") -> None:
    ax.set_facecolor(PALETTE["bg"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9AA5B1")
    ax.spines["bottom"].set_color("#9AA5B1")
    ax.tick_params(colors=PALETTE["ink"], labelsize=10)
    ax.grid(axis=grid_axis, linestyle=(0, (4, 4)), linewidth=0.8, color=PALETTE["grid"], alpha=0.85)


def plot_global_moran_trend(moran_df: pd.DataFrame, out_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.6, 7.2), gridspec_kw={"height_ratios": [3.2, 1]}, sharex=True)
    years = moran_df["year"]

    ax1.plot(years, moran_df["moran_low"], marker="o", markersize=6, linewidth=2.4, color=PALETTE["navy"], label="低空经济")
    ax1.plot(years, moran_df["moran_green"], marker="s", markersize=5.5, linewidth=2.4, color=PALETTE["red"], label="绿色交通")
    ax1.axhline(0, color="#808080", linestyle="--", linewidth=1)
    ax1.set_ylabel("全局莫兰指数 I", fontsize=11)
    ax1.set_title("图1 2016—2023年双系统全局莫兰指数变化", fontsize=14, fontweight="bold")
    ax1.legend(frameon=False, loc="upper left", ncol=2)
    _style(ax1)
    _mark(ax1, "a")

    low_sig = (moran_df["p_low"] < 0.05).astype(int)
    green_sig = (moran_df["p_green"] < 0.05).astype(int)
    ax2.fill_between(years, 0, low_sig, step="mid", alpha=0.35, color=PALETTE["blue"], label="低空经济显著")
    ax2.fill_between(years, 0, green_sig * 0.8, step="mid", alpha=0.35, color=PALETTE["sand"], label="绿色交通显著")
    ax2.set_ylim(0, 1.1)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["不显著", "显著"])
    ax2.set_xlabel("年份", fontsize=11)
    ax2.set_ylabel("显著性", fontsize=11)
    ax2.legend(frameon=False, loc="upper left", ncol=2, fontsize=9)
    _style(ax2, grid_axis="x")
    _mark(ax2, "b")
    _save(fig, out_dir / "global_moran_trend.png")


def plot_local_moran_scatter(panel_df: pd.DataFrame, w_df: pd.DataFrame, out_dir: Path) -> None:
    year = int(panel_df["年份"].max())
    ydf = panel_df.loc[panel_df["年份"] == year].set_index("省份").reindex(w_df.index)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8))
    specs = [("low_score", "低空经济局部莫兰散点图", "a"), ("green_score", "绿色交通局部莫兰散点图", "b")]

    for ax, (col, title, tag) in zip(axes, specs):
        x = ydf[col].to_numpy(dtype=float)
        y = w_df.values @ x
        xz = x - x.mean()
        yz = y - y.mean()
        quad = np.where((xz >= 0) & (yz >= 0), "HH", "LL")
        quad = np.where((xz >= 0) & (yz < 0), "HL", quad)
        quad = np.where((xz < 0) & (yz >= 0), "LH", quad)
        for q in ["HH", "HL", "LH", "LL"]:
            mask = quad == q
            ax.scatter(xz[mask], yz[mask], s=70, c=QUAD_COLOR[q], edgecolors="white", linewidth=0.7, alpha=0.95, label=q)
        coef = np.polyfit(xz, yz, 1)
        xs = np.linspace(xz.min() - 0.02, xz.max() + 0.02, 100)
        ax.plot(xs, coef[0] * xs + coef[1], color=PALETTE["ink"], linewidth=1.6)
        ax.axhline(0, color="#7F7F7F", linestyle="--", linewidth=1)
        ax.axvline(0, color="#7F7F7F", linestyle="--", linewidth=1)
        ax.set_xlabel("标准化值", fontsize=11)
        ax.set_ylabel("空间滞后值", fontsize=11)
        ax.set_title(f"图2{tag} {year}年{title}", fontsize=12.5, fontweight="bold")
        ax.legend(title="象限", frameon=False, loc="lower right", ncol=2, fontsize=9)
        _style(ax, grid_axis="both")
        _mark(ax, tag)
    _save(fig, out_dir / "local_moran_scatter_2023.png")


def plot_cluster_distribution(lisa_df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.6))
    specs = [("quad_low", "低空经济空间集聚类型分布", "a"), ("quad_green", "绿色交通空间集聚类型分布", "b")]
    for ax, (col, title, tag) in zip(axes, specs):
        cnt = lisa_df[col].value_counts().reindex(["HH", "HL", "LH", "LL"]).fillna(0).astype(int)
        df = pd.DataFrame({"类型": cnt.index, "数量": cnt.values})
        bars = ax.bar(df["类型"], df["数量"], color=[QUAD_COLOR[k] for k in df["类型"]], edgecolor="white", linewidth=0.9, width=0.62)
        total = max(df["数量"].sum(), 1)
        for bar, value in zip(bars, df["数量"]):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.18, f"{value}\n{value/total:.1%}", ha="center", va="bottom", fontsize=8.8)
        ax.set_title(f"图3{tag} {title}", fontsize=12.5, fontweight="bold")
        ax.set_xlabel("集聚类型", fontsize=11)
        ax.set_ylabel("省份数量", fontsize=11)
        _style(ax)
        _mark(ax, tag)
    _save(fig, out_dir / "cluster_distribution_2023.png")


def plot_sdm_effects(sdm_effects: pd.DataFrame, out_dir: Path) -> None:
    show = sdm_effects.copy()
    show["效应类型"] = show["effect"].map(EFFECT_LABEL)
    show = show.set_index("效应类型").reindex(["直接效应", "间接效应", "总效应"]).reset_index()
    colors = [PALETTE["navy"], PALETTE["red"], PALETTE["gold"]]

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    bars = ax.bar(show["效应类型"], show["coef"], color=colors, edgecolor="white", linewidth=1.0, width=0.58)
    pvals = show["p_value"].tolist()
    for i, (bar, value) in enumerate(zip(bars, show["coef"])):
        star = "***" if pd.notna(pvals[i]) and pvals[i] < 0.01 else "**" if pd.notna(pvals[i]) and pvals[i] < 0.05 else "*" if pd.notna(pvals[i]) and pvals[i] < 0.1 else ""
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}{star}", ha="center", va="bottom", fontsize=10)
    ax.axhline(0, color="#777777", linestyle="--", linewidth=1)
    ax.set_ylabel("效应系数", fontsize=11)
    ax.set_title("图4 空间杜宾模型效应分解结果", fontsize=13, fontweight="bold")
    _style(ax)
    _save(fig, out_dir / "sdm_effects_bar.png")


def plot_mediation_path(mediation_df: pd.DataFrame, out_dir: Path) -> None:
    show = mediation_df.copy().sort_values("contribution", ascending=True)
    fig, ax = plt.subplots(figsize=(10.4, 5.8))
    colors = [PALETTE["slate"], PALETTE["green"], PALETTE["gold"]]
    bars = ax.barh(show["path"], show["contribution"], color=colors, edgecolor="white", linewidth=1)
    err_left = np.maximum(show["indirect_ab"] - show["ci_low"], 0)
    err_right = np.maximum(show["ci_high"] - show["indirect_ab"], 0)
    ax.errorbar(show["contribution"], show["path"], xerr=[err_left, err_right], fmt="none", ecolor=PALETTE["ink"], elinewidth=1.2, capsize=4)
    for bar, ind, low, high in zip(bars, show["indirect_ab"], show["ci_low"], show["ci_high"]):
        ax.text(bar.get_width() + 0.008, bar.get_y() + bar.get_height() / 2, f"{ind:.3f} [{low:.3f}, {high:.3f}]", va="center", fontsize=8.7)
    ax.set_xlabel("贡献度", fontsize=11)
    ax.set_ylabel("中介路径", fontsize=11)
    ax.set_title("图5 多路径并行中介效应贡献图", fontsize=13, fontweight="bold")
    ax.set_xlim(0, max(show["contribution"].max() * 1.45, 0.1))
    _style(ax)
    _save(fig, out_dir / "mediation_contribution.png")


def plot_region_radar(regional_df: pd.DataFrame, out_dir: Path) -> None:
    if regional_df.empty:
        return
    cols = ["技术创新传导", "产业结构升级", "交通结构优化"]
    work = regional_df.copy()
    work["region"] = pd.Categorical(work["region"], categories=REGION_ORDER, ordered=True)
    work = work.sort_values("region")
    angles = np.linspace(0, 2 * np.pi, len(cols), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(8.3, 7.6))
    ax = plt.subplot(111, polar=True)
    colors = [PALETTE["navy"], PALETTE["red"], PALETTE["green"], PALETTE["gold"]]
    for i, (_, row) in enumerate(work.iterrows()):
        vals = [float(row[c]) for c in cols]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2.2, color=colors[i % 4], label=row["region"])
        ax.fill(angles, vals, color=colors[i % 4], alpha=0.10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cols, fontsize=10)
    ax.set_title("图6 四大区域中介路径贡献差异雷达图", fontsize=13, fontweight="bold", pad=24)
    ax.legend(loc="upper right", bbox_to_anchor=(1.24, 1.12), frameon=False)
    ax.grid(color=PALETTE["grid"], linestyle=(0, (4, 4)), linewidth=0.8)
    _save(fig, out_dir / "regional_radar.png")


def plot_region_group_bar(regional_df: pd.DataFrame, out_dir: Path) -> None:
    if regional_df.empty:
        return
    plot_df = regional_df[["region", "direct", "indirect"]].copy()
    plot_df["region"] = pd.Categorical(plot_df["region"], categories=REGION_ORDER, ordered=True)
    plot_df = plot_df.sort_values("region")
    plot_df = plot_df.melt(id_vars=["region"], value_vars=["direct", "indirect"], var_name="effect", value_name="coef")
    plot_df["effect_cn"] = plot_df["effect"].map({"direct": "直接效应", "indirect": "间接效应"})

    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    sns.barplot(data=plot_df, x="region", y="coef", hue="effect_cn", palette=[PALETTE["navy"], PALETTE["red"]], edgecolor="white", ax=ax)
    for p in ax.patches:
        h = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2, h + (0.012 if h >= 0 else -0.03), f"{h:.3f}", ha="center", va="bottom", fontsize=8.5)
    ax.axhline(0, color="#777777", linestyle="--", linewidth=1)
    ax.set_xlabel("区域", fontsize=11)
    ax.set_ylabel("效应系数", fontsize=11)
    ax.set_title("图7 四大区域直接效应与间接效应对比", fontsize=13, fontweight="bold")
    ax.legend(title="", frameon=False)
    _style(ax)
    _save(fig, out_dir / "regional_direct_indirect_grouped.png")


def plot_robustness_compare(base_effects: pd.DataFrame, robust_effects: pd.DataFrame, out_dir: Path) -> None:
    merged = base_effects[["effect", "coef"]].merge(robust_effects[["effect", "coef_econ_w"]], on="effect", how="inner")
    merged["effect_cn"] = merged["effect"].map(EFFECT_LABEL)
    x = np.arange(len(merged))
    width = 0.32

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.bar(x - width / 2, merged["coef"], width=width, color=PALETTE["navy"], edgecolor="white", linewidth=0.9, label="邻接矩阵W")
    ax.bar(x + width / 2, merged["coef_econ_w"], width=width, color=PALETTE["sand"], edgecolor="white", linewidth=0.9, label="经济距离矩阵W")
    for i, row in merged.iterrows():
        ax.text(i - width / 2, row["coef"] + 0.01, f"{row['coef']:.3f}", ha="center", va="bottom", fontsize=8.7)
        ax.text(i + width / 2, row["coef_econ_w"] + 0.01, f"{row['coef_econ_w']:.3f}", ha="center", va="bottom", fontsize=8.7)
    ax.axhline(0, color="#777777", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(merged["effect_cn"])
    ax.set_ylabel("效应系数", fontsize=11)
    ax.set_title("图8 稳健性检验：不同空间权重矩阵下的效应对比", fontsize=13, fontweight="bold")
    ax.legend(frameon=False)
    _style(ax)
    _save(fig, out_dir / "robustness_effects_compare.png")


def export_all_figures(
    panel_df: pd.DataFrame,
    w_df: pd.DataFrame,
    moran_df: pd.DataFrame,
    lisa_df: pd.DataFrame,
    sdm_effects: pd.DataFrame,
    mediation_df: pd.DataFrame,
    regional_df: pd.DataFrame,
    robust_effects: pd.DataFrame,
    region_map: Dict[str, str],
    out_dir: Path,
) -> None:
    sns.set_theme(style="ticks", font="Microsoft YaHei")
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelcolor"] = PALETTE["ink"]

    plot_global_moran_trend(moran_df, out_dir)
    plot_local_moran_scatter(panel_df, w_df, out_dir)
    plot_cluster_distribution(lisa_df, out_dir)
    plot_sdm_effects(sdm_effects, out_dir)
    plot_mediation_path(mediation_df, out_dir)
    plot_region_radar(regional_df, out_dir)
    plot_region_group_bar(regional_df, out_dir)
    plot_robustness_compare(sdm_effects, robust_effects, out_dir)
