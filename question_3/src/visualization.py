"""可视化模块：绘制精度对比、趋势预测与优化结果图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from question_3.config import FIG_DIR


class Question3Visualizer:
    """负责输出问题三所需图表。"""

    def __init__(self, output_dir: Path = FIG_DIR.parent) -> None:
        """初始化绘图器并创建输出目录。"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.palette = {
            "dusty_blue": "#7C93A6",
            "sage": "#9CAF88",
            "muted_clay": "#C08F80",
            "stone": "#C6B9A9",
            "mist": "#D8D4CC",
            "deep_olive": "#6F7D60",
            "ink": "#5C5A57",
            "soft_gold": "#C6A77A",
            "cloud": "#F7F4EF",
            "paper": "#FBF8F3",
            "frame": "#EEE6DC",
            "accent": "#B4836E",
        }
        self.scenario_colors = {
            "基准情景": self.palette["dusty_blue"],
            "加大投入情景": self.palette["muted_clay"],
            "重点投入情景": self.palette["sage"],
        }
        self.region_colors = [
            self.palette["dusty_blue"],
            self.palette["sage"],
            self.palette["muted_clay"],
            self.palette["stone"],
        ]
        self._setup_theme()

    def _setup_theme(self) -> None:
        """统一设置竞赛论文终稿级绘图主题。"""
        sns.set_theme(style="white")
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
        plt.rcParams["axes.unicode_minus"] = False
        plt.rcParams["figure.facecolor"] = self.palette["cloud"]
        plt.rcParams["axes.facecolor"] = self.palette["paper"]
        plt.rcParams["axes.edgecolor"] = self.palette["frame"]
        plt.rcParams["axes.labelcolor"] = self.palette["ink"]
        plt.rcParams["xtick.color"] = self.palette["ink"]
        plt.rcParams["ytick.color"] = self.palette["ink"]
        plt.rcParams["grid.color"] = "#E6DED4"
        plt.rcParams["grid.linestyle"] = "--"
        plt.rcParams["grid.linewidth"] = 0.8
        plt.rcParams["axes.titleweight"] = "bold"
        plt.rcParams["axes.titlesize"] = 13
        plt.rcParams["axes.labelsize"] = 11
        plt.rcParams["xtick.labelsize"] = 10
        plt.rcParams["ytick.labelsize"] = 10
        plt.rcParams["legend.frameon"] = True
        plt.rcParams["legend.facecolor"] = "#FCFAF7"
        plt.rcParams["legend.edgecolor"] = self.palette["mist"]
        plt.rcParams["legend.fontsize"] = 10
        plt.rcParams["savefig.facecolor"] = self.palette["cloud"]
        plt.rcParams["savefig.bbox"] = "tight"

    def _beautify_axes(self, ax: plt.Axes, show_grid_y: bool = True) -> None:
        """统一坐标轴风格。"""
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(self.palette["mist"])
        ax.spines["bottom"].set_color(self.palette["mist"])
        ax.tick_params(axis="both", labelsize=10, length=0)
        if show_grid_y:
            ax.grid(axis="y", alpha=0.72)
        else:
            ax.grid(False)

    def _add_caption(self, fig: plt.Figure, text: str) -> None:
        """在图底部加入论文风格图注。"""
        fig.text(0.5, 0.015, text, ha="center", va="bottom", fontsize=10, color=self.palette["ink"])

    def _add_corner_tag(self, ax: plt.Axes, text: str) -> None:
        """添加终稿风格角标。"""
        ax.text(
            0.985,
            0.965,
            text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color=self.palette["accent"],
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#FFF9F2", "edgecolor": self.palette["frame"]},
        )

    def plot_model_comparison(self, comparison_df: pd.DataFrame) -> None:
        """绘制不同模型预测精度对比柱状图。"""
        fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.9))
        metric_names = ["MAE", "RMSE", "R2"]
        for ax, metric in zip(axes, metric_names):
            plot_df = comparison_df.copy()
            sns.barplot(
                data=plot_df,
                x="模型",
                y=metric,
                hue="模型",
                palette=self.region_colors,
                dodge=False,
                legend=False,
                ax=ax,
                width=0.60,
            )
            self._beautify_axes(ax)
            ax.set_title(f"{metric} 指标", pad=10)
            ax.set_xlabel("")
            ax.set_ylabel(metric)
            ax.tick_params(axis="x", rotation=10)
            for container in ax.containers:
                ax.bar_label(container, fmt="%.3f", padding=3, fontsize=9, color=self.palette["ink"])
            if metric == "R2":
                best_idx = plot_df[metric].idxmax()
                self._add_corner_tag(ax, "越大越优")
            else:
                best_idx = plot_df[metric].idxmin()
                self._add_corner_tag(ax, "越小越优")
            best_x = plot_df.index.get_loc(best_idx)
            best_y = plot_df.loc[best_idx, metric]
            ax.scatter(best_x, best_y, s=58, color=self.palette["soft_gold"], zorder=5, edgecolors="white", linewidths=0.8)
        fig.suptitle("不同预测模型拟合精度对比", fontsize=16, fontweight="bold", y=0.98, color=self.palette["ink"])
        self._add_caption(fig, "图3-1 不同模型在测试集上的 MAE、RMSE 与 R² 对比")
        fig.subplots_adjust(top=0.80, bottom=0.18, wspace=0.25)
        fig.savefig(self.output_dir / "模型拟合精度对比.png", dpi=360)
        plt.close(fig)

    def plot_region_forecast(self, region_summary_df: pd.DataFrame) -> None:
        """绘制四大区域三种情景下的发展趋势图。"""
        fig, axes = plt.subplots(2, 2, figsize=(15.8, 10.8), sharex=True, sharey=True)
        region_order = ["东部", "中部", "西部", "东北"]
        axes = axes.flatten()
        for ax, region in zip(axes, region_order):
            sub_df = region_summary_df[region_summary_df["区域"] == region].copy()
            sns.lineplot(
                data=sub_df,
                x="年份",
                y="预测值",
                hue="情景",
                palette=self.scenario_colors,
                marker="o",
                markersize=6.5,
                linewidth=2.4,
                dashes=False,
                ax=ax,
                legend=False,
            )
            self._beautify_axes(ax)
            ax.set_title(f"{region}地区", pad=8)
            ax.set_ylabel("预测指数")
            ax.set_xlabel("年份")
            ax.set_xlim(sub_df["年份"].min() - 0.2, sub_df["年份"].max() + 0.4)
            region_handles = [plt.Line2D([0], [0], color=color, marker="o", lw=2.4) for color in self.scenario_colors.values()]
            region_labels = list(self.scenario_colors.keys())
            ax.legend(region_handles, region_labels, loc="upper left", fontsize=9, title="情景")
            for line in ax.lines:
                x_data = line.get_xdata()
                y_data = line.get_ydata()
                if len(x_data) > 0 and len(y_data) > 0:
                    ax.text(x_data[-1] + 0.05, y_data[-1], f"{y_data[-1]:.2f}", fontsize=8.8, color=line.get_color(), va="center")
        fig.suptitle("2024—2030年四大区域绿色交通发展趋势预测", fontsize=16, fontweight="bold", y=0.985, color=self.palette["ink"])
        self._add_caption(fig, "图3-2 四大区域在三种情景设定下的绿色交通发展趋势")
        fig.subplots_adjust(top=0.90, bottom=0.12, hspace=0.26, wspace=0.14)
        fig.savefig(self.output_dir / "四大区域情景预测趋势.png", dpi=360)
        plt.close(fig)

    def plot_province_heatmap(self, heatmap_df: pd.DataFrame) -> None:
        """绘制重点投入情景下各省份预测值热力图。"""
        fig, ax = plt.subplots(figsize=(12.6, 10.8))
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            "morandi_heat",
            ["#F5EFE6", "#D6C7B4", "#AAB59A", "#7C93A6", "#6E7B68"],
        )
        sns.heatmap(
            heatmap_df,
            cmap=cmap,
            linewidths=0.50,
            linecolor="#F3EEE7",
            cbar_kws={"shrink": 0.80, "label": "预测指数"},
            ax=ax,
        )
        ax.set_title("重点投入情景下各省份绿色交通预测热力图", pad=12)
        ax.set_xlabel("年份")
        ax.set_ylabel("省份")
        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="y", rotation=0)
        self._add_caption(fig, "图3-3 重点投入情景下各省份绿色交通预测值空间分布热力图")
        fig.subplots_adjust(top=0.90, bottom=0.08)
        fig.savefig(self.output_dir / "重点投入情景_省域热力图.png", dpi=360)
        plt.close(fig)

    def plot_region_ranking(self, ranking_df: pd.DataFrame) -> None:
        """绘制2030年各区域预测水平横向排名图。"""
        fig, ax = plt.subplots(figsize=(10.8, 6.4))
        palette = list(reversed(self.region_colors))
        sns.barplot(
            data=ranking_df,
            y="区域",
            x="预测值",
            hue="区域",
            palette=palette,
            dodge=False,
            legend=False,
            ax=ax,
            width=0.56,
        )
        self._beautify_axes(ax)
        ax.set_title("2030年重点投入情景下区域发展水平排名", pad=12)
        ax.set_xlabel("预测指数")
        ax.set_ylabel("")
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", padding=6, fontsize=10, color=self.palette["ink"])
        self._add_corner_tag(ax, "2030年")
        self._add_caption(fig, "图3-4 2030年重点投入情景下四大区域预测值排序")
        fig.subplots_adjust(top=0.86, bottom=0.16)
        fig.savefig(self.output_dir / "2030区域预测排名.png", dpi=360)
        plt.close(fig)

    def plot_shandong_comparison(self, shandong_df: pd.DataFrame) -> None:
        """绘制山东省在三种情景下的趋势对比图。"""
        fig, ax = plt.subplots(figsize=(11.2, 6.5))
        sns.lineplot(
            data=shandong_df,
            x="年份",
            y="预测值",
            hue="情景",
            palette=self.scenario_colors,
            marker="o",
            markersize=7.5,
            linewidth=2.7,
            dashes=False,
            ax=ax,
        )
        self._beautify_axes(ax)
        ax.set_title("山东省绿色交通发展趋势情景对比", pad=12)
        ax.set_xlabel("年份")
        ax.set_ylabel("预测指数")
        ax.legend(title="情景设定", ncol=3, loc="upper left", bbox_to_anchor=(0.0, 1.02))
        self._add_corner_tag(ax, "山东省")
        for line in ax.lines:
            x_data = line.get_xdata()
            y_data = line.get_ydata()
            if len(x_data) > 0 and len(y_data) > 0:
                ax.text(x_data[-1] + 0.08, y_data[-1], f"{y_data[-1]:.2f}", fontsize=9, color=line.get_color(), va="center")
        self._add_caption(fig, "图3-5 山东省在不同投入情景下的绿色交通发展趋势")
        fig.subplots_adjust(top=0.83, bottom=0.18)
        fig.savefig(self.output_dir / "山东省趋势对比.png", dpi=360)
        plt.close(fig)

    def plot_scenario_gap(self, comparison_df: pd.DataFrame, province: str = "山东") -> None:
        """绘制指定省份相对基准情景的提升面积图。"""
        fig, ax = plt.subplots(figsize=(11.2, 6.2))
        enhanced_df = comparison_df[comparison_df["情景"] != "基准情景"].copy()
        for scenario, sub_df in enhanced_df.groupby("情景"):
            ordered_df = sub_df.sort_values("年份")
            color = self.scenario_colors.get(scenario, self.palette["stone"])
            ax.plot(ordered_df["年份"], ordered_df["相对基准提升"], color=color, linewidth=2.4, marker="o", label=scenario)
            ax.fill_between(ordered_df["年份"], 0, ordered_df["相对基准提升"], color=color, alpha=0.22)
        self._beautify_axes(ax)
        ax.axhline(0, color=self.palette["mist"], linewidth=1.0)
        ax.set_title(f"{province}省相对基准情景的增益变化", pad=12)
        ax.set_xlabel("年份")
        ax.set_ylabel("相对基准提升")
        ax.legend(title="情景设定", loc="upper left")
        self._add_corner_tag(ax, "增量比较")
        self._add_caption(fig, f"图3-6 {province}省在强化投入情景下相对基准情景的增益变化")
        fig.subplots_adjust(top=0.86, bottom=0.18)
        fig.savefig(self.output_dir / f"{province}省情景增益对比.png", dpi=360)
        plt.close(fig)
