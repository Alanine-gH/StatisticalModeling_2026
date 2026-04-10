"""可视化模块：绘制精度对比、趋势预测与优化结果图。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

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
        sns.set_theme(style="whitegrid")
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
        plt.rcParams["axes.unicode_minus"] = False

    def plot_model_comparison(self, comparison_df: pd.DataFrame) -> None:
        """绘制不同模型预测精度对比柱状图。"""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        metric_names = ["MAE", "RMSE", "R2"]
        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
        for ax, metric in zip(axes, metric_names):
            sns.barplot(data=comparison_df, x="模型", y=metric, palette=colors, ax=ax)
            ax.set_title(f"{metric} 指标对比")
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=15)
        fig.suptitle("不同预测模型拟合精度对比", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(self.output_dir / "模型拟合精度对比.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_region_forecast(self, region_summary_df: pd.DataFrame) -> None:
        """绘制四大区域三种情景下的发展趋势图。"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
        region_order = sorted(region_summary_df["区域"].unique().tolist())
        axes = axes.flatten()
        for ax, region in zip(axes, region_order):
            sub_df = region_summary_df[region_summary_df["区域"] == region]
            sns.lineplot(data=sub_df, x="年份", y="预测值", hue="情景", marker="o", linewidth=2.2, ax=ax)
            ax.set_title(f"{region}地区绿色交通发展预测")
            ax.set_ylabel("预测指数")
            ax.legend(title="情景")
        fig.suptitle("2024-2030年四大区域三种情景预测趋势", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(self.output_dir / "四大区域情景预测趋势.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_convergence_curves(self, convergence_results: Dict[str, List[float]]) -> None:
        """绘制改进粒子群算法收敛曲线。"""
        fig, ax = plt.subplots(figsize=(10, 6))
        color_map = {"baseline": "#4C72B0", "stronger": "#DD8452", "targeted": "#55A868"}
        label_map = {"baseline": "基准投入情景", "stronger": "加大投入情景", "targeted": "重点投入情景"}
        for scenario_name, curve in convergence_results.items():
            ax.plot(curve, label=label_map.get(scenario_name, scenario_name), linewidth=2.0, color=color_map.get(scenario_name))
        ax.set_title("改进自适应粒子群算法收敛曲线")
        ax.set_xlabel("迭代次数")
        ax.set_ylabel("目标函数值")
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.output_dir / "改进粒子群收敛曲线.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_optimization_structure(self, optimized_df: pd.DataFrame, scenario_name: str) -> None:
        """绘制某情景下四大区域最优投入结构图。"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        regions = optimized_df["区域"].tolist()
        labels = ["产业投入", "基础设施投入", "创新投入"]
        for ax, region in zip(axes, regions):
            row = optimized_df[optimized_df["区域"] == region].iloc[0]
            values = [row["产业投入比例"], row["基础设施投入比例"], row["创新投入比例"]]
            ax.pie(values, labels=labels, autopct="%.1f%%", startangle=90, colors=["#4C72B0", "#55A868", "#C44E52"])
            ax.set_title(f"{region}地区投入结构")
        title_map = {"baseline": "基准情景", "stronger": "加大投入情景", "targeted": "重点投入情景"}
        fig.suptitle(f"{title_map.get(scenario_name, scenario_name)}下四大区域最优投入结构", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(self.output_dir / f"{scenario_name}_最优投入结构.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_shandong_comparison(self, shandong_df: pd.DataFrame) -> None:
        """绘制山东省在三种情景下的趋势对比图。"""
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=shandong_df, x="年份", y="预测值", hue="情景", marker="o", linewidth=2.2, ax=ax)
        ax.set_title("山东省优化前后绿色交通发展趋势对比")
        ax.set_ylabel("预测指数")
        fig.tight_layout()
        fig.savefig(self.output_dir / "山东省趋势对比.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
