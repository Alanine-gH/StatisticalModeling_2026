"""可视化输出模块。"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from question_1.config import FIG_DIR


warnings.filterwarnings("ignore", message=r"Glyph .* missing from font")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid")


def ensure_figure_dir() -> None:
    """创建图片输出目录。"""
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(file_name: str) -> Path:
    """保存当前图像到 output/figures 目录并返回路径。"""
    ensure_figure_dir()
    output_path = FIG_DIR / file_name
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def plot_weight_comparison(weight_df: pd.DataFrame) -> Path:
    """绘制双系统指标的 CRITIC、熵权、时间衰减权重和综合权重对比柱状图。"""
    available_columns = [column for column in ["CRITIC权重", "熵权", "时间衰减权重", "综合权重"] if column in weight_df.columns]
    plot_df = weight_df.melt(
        id_vars=["系统", "指标"],
        value_vars=available_columns,
        var_name="权重类型",
        value_name="权重值",
    )

    plt.figure(figsize=(18, 8))
    sns.barplot(data=plot_df, x="指标", y="权重值", hue="权重类型", palette="Set2")
    plt.xticks(rotation=75, ha="right")
    plt.title("双系统指标权重对比柱状图")
    plt.xlabel("二级指标")
    plt.ylabel("权重")
    return save_figure("图1_双系统指标权重对比柱状图.png")


def plot_region_trend(region_df: pd.DataFrame) -> Path:
    """绘制四大区域年度平均综合发展水平折线图。"""
    plt.figure(figsize=(12, 7))
    sns.lineplot(
        data=region_df,
        x="年份",
        y="区域平均综合发展水平",
        hue="区域",
        style="区域",
        markers=True,
        dashes=False,
        linewidth=2.4,
    )
    plt.title("四大区域双系统综合发展水平年度变化")
    plt.xlabel("年份")
    plt.ylabel("区域平均综合发展水平")
    return save_figure("图3_四大区域年度变化折线图.png")


def plot_province_ranking(ranking_df: pd.DataFrame) -> Path:
    """绘制2023年各省份双系统综合发展水平排名条形图。"""
    plt.figure(figsize=(14, 10))
    sns.barplot(
        data=ranking_df,
        x="双系统综合发展指数",
        y="省份",
        hue="区域",
        dodge=False,
        palette="viridis",
    )
    plt.title("2023年各省份双系统综合发展水平排名")
    plt.xlabel("双系统综合发展指数")
    plt.ylabel("省份")
    plt.legend(title="区域")
    return save_figure("图4_2023年各省份排名条形图.png")


def plot_correlation_heatmap(corr_df: pd.DataFrame) -> Path:
    """绘制改进法与对照方法的相关系数矩阵热力图。"""
    plt.figure(figsize=(7, 6))
    sns.heatmap(corr_df, annot=True, cmap="YlGnBu", fmt=".4f", square=True)
    plt.title("权重合理性检验：相关系数矩阵热力图")
    return save_figure("检验1_相关系数矩阵热力图.png")


def plot_robustness_lines(robustness_df: pd.DataFrame) -> Path:
    """绘制不同时间衰减因子下全国平均综合指数对比折线图。"""
    plot_df = (
        robustness_df.groupby(["年份", "时间衰减因子"], as_index=False)["双系统综合发展指数"]
        .mean()
        .rename(columns={"双系统综合发展指数": "全国平均综合发展指数"})
    )

    plt.figure(figsize=(11, 7))
    sns.lineplot(
        data=plot_df,
        x="年份",
        y="全国平均综合发展指数",
        hue="时间衰减因子",
        marker="o",
        linewidth=2.2,
        palette="tab10",
    )
    plt.title("稳健性检验：不同时间衰减因子下全国平均综合指数对比")
    plt.xlabel("年份")
    plt.ylabel("全国平均综合发展指数")
    return save_figure("检验2_稳健性检验对比折线图.png")


def plot_model_comparison_trend(comparison_df: pd.DataFrame) -> Path:
    """绘制改进模型与传统模型全国平均指数对比折线图。"""
    plot_df = comparison_df.melt(
        id_vars=["年份"],
        value_vars=["改进模型全国平均指数", "传统模型全国平均指数"],
        var_name="模型类型",
        value_name="全国平均综合指数",
    )
    plt.figure(figsize=(11, 7))
    sns.lineplot(
        data=plot_df,
        x="年份",
        y="全国平均综合指数",
        hue="模型类型",
        style="模型类型",
        markers=True,
        dashes=False,
        linewidth=2.4,
        palette="Set1",
    )
    plt.title("改进模型与传统模型全国平均指数对比")
    plt.xlabel("年份")
    plt.ylabel("全国平均综合指数")
    return save_figure("图5_改进模型与传统模型全国平均指数对比.png")


def plot_model_improvement_bar(comparison_df: pd.DataFrame) -> Path:
    """绘制改进模型相对传统模型年度提升幅度柱状图。"""
    plt.figure(figsize=(11, 7))
    sns.barplot(data=comparison_df, x="年份", y="提升幅度_%", color="#4C72B0")
    plt.axhline(0, color="black", linewidth=1)
    plt.title("图6 改进模型相对传统模型年度提升幅度")
    plt.xlabel("年份")
    plt.ylabel("提升幅度（%）")
    return save_figure("图6_改进模型相对传统模型年度提升幅度.png")
