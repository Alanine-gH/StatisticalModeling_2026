"""传统赋权模型模块：作为改进时间衰减模型的对照组。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class BaselineComparisonMetrics:
    """存储改进模型与传统模型的对比指标。"""

    proposed_mean: float
    baseline_mean: float
    mean_absolute_gap: float
    relative_improvement_pct: float
    rank_consistency: float


class TraditionalCriticEntropyModel:
    """传统 CRITIC-熵权组合赋权模型，不引入时间衰减机制。"""

    def __init__(self, critic_ratio: float = 0.5, entropy_ratio: float = 0.5) -> None:
        self.critic_ratio = critic_ratio
        self.entropy_ratio = entropy_ratio

    @staticmethod
    def _normalize_series(weights: pd.Series) -> pd.Series:
        total = weights.sum()
        if np.isclose(total, 0):
            return pd.Series(np.full(len(weights), 1 / len(weights)), index=weights.index)
        return weights / total

    @staticmethod
    def calculate_critic_weights(df: pd.DataFrame, indicators: List[str]) -> pd.Series:
        data = df[indicators].copy()
        std = data.std(ddof=0)
        corr = data.corr().fillna(0)
        conflict = (1 - corr).sum(axis=1)
        information = std * conflict
        return information / information.sum()

    @staticmethod
    def calculate_entropy_weights(df: pd.DataFrame, indicators: List[str], epsilon: float = 1e-12) -> pd.Series:
        data = df[indicators].copy() + epsilon
        proportion = data.div(data.sum(axis=0), axis=1)
        n = len(data)
        k = 1 / np.log(n)
        entropy = -k * (proportion * np.log(proportion)).sum(axis=0)
        divergence = 1 - entropy
        return divergence / divergence.sum()

    def combine_weights(self, critic_weights: pd.Series, entropy_weights: pd.Series, indicators: List[str]) -> pd.Series:
        combined = self.critic_ratio * critic_weights + self.entropy_ratio * entropy_weights
        return self._normalize_series(combined.reindex(indicators))

    @staticmethod
    def calculate_scores(df: pd.DataFrame, indicators: List[str], weights: pd.Series) -> pd.Series:
        return df[indicators].mul(weights.reindex(indicators).values).sum(axis=1)

    def evaluate_system(self, df: pd.DataFrame, indicators: List[str]) -> Dict[str, pd.DataFrame | pd.Series]:
        critic_weights = self.calculate_critic_weights(df, indicators)
        entropy_weights = self.calculate_entropy_weights(df, indicators)
        combined_weights = self.combine_weights(critic_weights, entropy_weights, indicators)
        scores = self.calculate_scores(df, indicators, combined_weights)

        weight_table = pd.DataFrame(
            {
                "指标": indicators,
                "CRITIC权重": critic_weights.reindex(indicators).values,
                "熵权": entropy_weights.reindex(indicators).values,
                "综合权重": combined_weights.reindex(indicators).values,
            }
        )
        return {
            "critic_weights": critic_weights,
            "entropy_weights": entropy_weights,
            "combined_weights": combined_weights,
            "scores": scores,
            "weight_table": weight_table,
        }

    def evaluate_dual_system(
        self,
        df: pd.DataFrame,
        low_indicators: List[str],
        green_indicators: List[str],
    ) -> Dict[str, pd.DataFrame | pd.Series | Dict[str, pd.DataFrame]]:
        low_result = self.evaluate_system(df, low_indicators)
        green_result = self.evaluate_system(df, green_indicators)

        result_df = df[["年份", "省份", "区域"]].copy()
        result_df["低空经济指数"] = low_result["scores"]
        result_df["绿色交通指数"] = green_result["scores"]
        result_df["双系统综合发展指数"] = 0.5 * result_df["低空经济指数"] + 0.5 * result_df["绿色交通指数"]
        result_df["全国排名"] = result_df.groupby("年份")["双系统综合发展指数"].rank(ascending=False, method="min")

        return {
            "result_df": result_df,
            "low": low_result,
            "green": green_result,
        }

    @staticmethod
    def build_comparison_metrics(proposed_df: pd.DataFrame, baseline_df: pd.DataFrame) -> BaselineComparisonMetrics:
        merged = proposed_df[["年份", "省份", "双系统综合发展指数"]].merge(
            baseline_df[["年份", "省份", "双系统综合发展指数"]],
            on=["年份", "省份"],
            suffixes=("_改进模型", "_传统模型"),
        )

        proposed_mean = float(merged["双系统综合发展指数_改进模型"].mean())
        baseline_mean = float(merged["双系统综合发展指数_传统模型"].mean())
        mean_absolute_gap = float(
            (merged["双系统综合发展指数_改进模型"] - merged["双系统综合发展指数_传统模型"]).abs().mean()
        )

        relative_improvement = 0.0
        if not np.isclose(baseline_mean, 0):
            relative_improvement = (proposed_mean - baseline_mean) / baseline_mean * 100

        rank_consistency = float(
            merged["双系统综合发展指数_改进模型"].corr(
                merged["双系统综合发展指数_传统模型"], method="spearman"
            )
        )

        return BaselineComparisonMetrics(
            proposed_mean=proposed_mean,
            baseline_mean=baseline_mean,
            mean_absolute_gap=mean_absolute_gap,
            relative_improvement_pct=relative_improvement,
            rank_consistency=rank_consistency,
        )


def build_model_comparison_table(proposed_df: pd.DataFrame, baseline_df: pd.DataFrame) -> pd.DataFrame:
    """生成改进模型与传统模型的逐样本对比表。"""
    merged = proposed_df[["年份", "省份", "区域", "双系统综合发展指数"]].merge(
        baseline_df[["年份", "省份", "双系统综合发展指数", "全国排名"]],
        on=["年份", "省份"],
        suffixes=("_改进模型", "_传统模型"),
    )
    merged["指数差值_改进减传统"] = (
        merged["双系统综合发展指数_改进模型"] - merged["双系统综合发展指数_传统模型"]
    )
    merged["排名差值_改进减传统"] = merged["全国排名_改进模型"] - merged["全国排名_传统模型"]
    return merged.sort_values(["年份", "双系统综合发展指数_改进模型"], ascending=[True, False]).reset_index(drop=True)
