"""时间衰减 CRITIC-熵权组合赋权模型模块。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class ModelFitMetrics:
    """存储模型合理性与稳健性检验结果。"""

    correlation_with_traditional: float
    correlation_with_entropy: float
    traditional_vs_entropy: float
    average_correlation: float
    improvement_vs_traditional_pct: float
    improvement_vs_entropy_pct: float


class TimeDecayCriticEntropyModel:
    """实现改进的时间衰减 CRITIC-熵权组合赋权法。"""

    def __init__(
        self,
        decay_base: float = 0.95,
        start_boost_year: int = 2021,
        critic_ratio: float = 0.5,
        entropy_ratio: float = 0.5,
    ) -> None:
        """初始化模型参数。"""
        self.decay_base = decay_base
        self.start_boost_year = start_boost_year
        self.critic_ratio = critic_ratio
        self.entropy_ratio = entropy_ratio

    @staticmethod
    def _normalize_series(weights: pd.Series) -> pd.Series:
        """对权重序列进行归一化，使其和为1。"""
        total = weights.sum()
        if np.isclose(total, 0):
            return pd.Series(np.full(len(weights), 1 / len(weights)), index=weights.index)
        return weights / total

    @staticmethod
    def calculate_critic_weights(df: pd.DataFrame, indicators: List[str]) -> pd.Series:
        """根据标准差与冲突性计算 CRITIC 权重。"""
        data = df[indicators].copy()
        std = data.std(ddof=0)
        corr = data.corr().fillna(0)
        conflict = (1 - corr).sum(axis=1)
        information = std * conflict
        weights = information / information.sum()
        return weights.reindex(indicators)

    @staticmethod
    def calculate_entropy_weights(df: pd.DataFrame, indicators: List[str], epsilon: float = 1e-12) -> pd.Series:
        """根据指标信息熵计算熵权。"""
        data = df[indicators].copy() + epsilon
        proportion = data.div(data.sum(axis=0), axis=1)
        n = len(data)
        k = 1 / np.log(n)
        entropy = -k * (proportion * np.log(proportion)).sum(axis=0)
        divergence = 1 - entropy
        weights = divergence / divergence.sum()
        return weights.reindex(indicators)

    def calculate_time_decay_factors(self, years: pd.Series, indicators: List[str], system_name: str) -> pd.DataFrame:
        """为指标构造时间衰减因子，其中仅对2021年后的低空经济指标增强权重。"""
        factors = pd.DataFrame(1.0, index=years.index, columns=indicators)
        if system_name != "low":
            return factors

        enhanced = years.apply(
            lambda y: self.decay_base ** max(0, 2023 - int(y)) if int(y) >= self.start_boost_year else 1.0
        )
        for indicator in indicators:
            factors[indicator] = enhanced.values
        return factors

    def combine_weights(
        self,
        critic_weights: pd.Series,
        entropy_weights: pd.Series,
        years: pd.Series,
        indicators: List[str],
        system_name: str,
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """融合 CRITIC、熵权与时间衰减因子，生成最终综合权重。"""
        base_weight = self.critic_ratio * critic_weights + self.entropy_ratio * entropy_weights
        base_weight = self._normalize_series(base_weight.reindex(indicators))

        time_factors = self.calculate_time_decay_factors(years, indicators, system_name)
        adjusted = time_factors.mul(base_weight, axis=1)
        adjusted = adjusted.div(adjusted.sum(axis=1), axis=0)
        return base_weight, adjusted

    @staticmethod
    def calculate_scores(df: pd.DataFrame, indicators: List[str], row_weights: pd.DataFrame) -> pd.Series:
        """按照逐行动态权重计算综合评价得分。"""
        weighted_values = df[indicators].mul(row_weights[indicators].values)
        return weighted_values.sum(axis=1)

    def evaluate_system(self, df: pd.DataFrame, indicators: List[str], system_name: str) -> Dict[str, pd.DataFrame | pd.Series]:
        """完成单一系统的权重计算、动态赋权与得分测度。"""
        critic_weights = self.calculate_critic_weights(df, indicators)
        entropy_weights = self.calculate_entropy_weights(df, indicators)
        combined_weight, dynamic_weights = self.combine_weights(
            critic_weights=critic_weights,
            entropy_weights=entropy_weights,
            years=df["年份"],
            indicators=indicators,
            system_name=system_name,
        )
        score = self.calculate_scores(df, indicators, dynamic_weights)

        weight_table = pd.DataFrame(
            {
                "指标": indicators,
                "CRITIC权重": critic_weights.reindex(indicators).values,
                "熵权": entropy_weights.reindex(indicators).values,
                "综合权重": combined_weight.reindex(indicators).values,
            }
        )
        return {
            "critic_weights": critic_weights,
            "entropy_weights": entropy_weights,
            "combined_weights": combined_weight,
            "dynamic_weights": dynamic_weights,
            "scores": score,
            "weight_table": weight_table,
        }

    def evaluate_dual_system(
        self,
        df: pd.DataFrame,
        low_indicators: List[str],
        green_indicators: List[str],
    ) -> Dict[str, pd.DataFrame | pd.Series | Dict[str, pd.DataFrame]]:
        """计算双系统的综合发展指数，并输出联合评价结果。"""
        low_result = self.evaluate_system(df, low_indicators, "low")
        green_result = self.evaluate_system(df, green_indicators, "green")

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
    def calculate_traditional_combined_score(
        df: pd.DataFrame,
        indicators: List[str],
        critic_ratio: float = 0.5,
        entropy_ratio: float = 0.5,
    ) -> pd.Series:
        """计算传统不含时间衰减的 CRITIC-熵权综合得分。"""
        critic_weights = TimeDecayCriticEntropyModel.calculate_critic_weights(df, indicators)
        entropy_weights = TimeDecayCriticEntropyModel.calculate_entropy_weights(df, indicators)
        combined = critic_ratio * critic_weights + entropy_ratio * entropy_weights
        combined = combined / combined.sum()
        return df[indicators].mul(combined.reindex(indicators).values).sum(axis=1)

    @staticmethod
    def calculate_entropy_only_score(df: pd.DataFrame, indicators: List[str]) -> pd.Series:
        """计算仅使用熵权法得到的综合得分。"""
        entropy_weights = TimeDecayCriticEntropyModel.calculate_entropy_weights(df, indicators)
        return df[indicators].mul(entropy_weights.reindex(indicators).values).sum(axis=1)

    def build_validation_scores(
        self,
        df: pd.DataFrame,
        low_indicators: List[str],
        green_indicators: List[str],
    ) -> pd.DataFrame:
        """构建改进法、传统组合赋权法和单纯熵权法的综合指数对比表。"""
        improved = self.evaluate_dual_system(df, low_indicators, green_indicators)["result_df"].copy()

        traditional_low = self.calculate_traditional_combined_score(df, low_indicators)
        traditional_green = self.calculate_traditional_combined_score(df, green_indicators)
        entropy_low = self.calculate_entropy_only_score(df, low_indicators)
        entropy_green = self.calculate_entropy_only_score(df, green_indicators)

        validation_df = improved[["年份", "省份", "区域", "双系统综合发展指数"]].rename(
            columns={"双系统综合发展指数": "改进时间衰减法"}
        )
        validation_df["传统CRITIC熵权法"] = 0.5 * traditional_low + 0.5 * traditional_green
        validation_df["单纯熵权法"] = 0.5 * entropy_low + 0.5 * entropy_green
        return validation_df

    @staticmethod
    def calculate_fit_metrics(validation_df: pd.DataFrame) -> Tuple[pd.DataFrame, ModelFitMetrics]:
        """计算相关系数矩阵与拟合度指标，并给出准确性提升百分比。"""
        corr_matrix = validation_df[["改进时间衰减法", "传统CRITIC熵权法", "单纯熵权法"]].corr(method="pearson")

        corr_traditional = float(corr_matrix.loc["改进时间衰减法", "传统CRITIC熵权法"])
        corr_entropy = float(corr_matrix.loc["改进时间衰减法", "单纯熵权法"])
        corr_trad_entropy = float(corr_matrix.loc["传统CRITIC熵权法", "单纯熵权法"])
        avg_corr = np.mean([corr_traditional, corr_entropy])

        improvement_vs_traditional = max(0.0, (1 - abs(1 - corr_traditional)) * 100 - corr_traditional * 100)
        improvement_vs_entropy = max(0.0, (1 - abs(1 - corr_entropy)) * 100 - corr_entropy * 100)

        metrics = ModelFitMetrics(
            correlation_with_traditional=corr_traditional,
            correlation_with_entropy=corr_entropy,
            traditional_vs_entropy=corr_trad_entropy,
            average_correlation=avg_corr,
            improvement_vs_traditional_pct=improvement_vs_traditional,
            improvement_vs_entropy_pct=improvement_vs_entropy,
        )
        return corr_matrix, metrics

    def robustness_test(
        self,
        df: pd.DataFrame,
        low_indicators: List[str],
        green_indicators: List[str],
        decay_values: List[float],
    ) -> pd.DataFrame:
        """对不同时间衰减因子进行稳健性检验。"""
        outputs = []
        for decay in decay_values:
            temp_model = TimeDecayCriticEntropyModel(
                decay_base=decay,
                start_boost_year=self.start_boost_year,
                critic_ratio=self.critic_ratio,
                entropy_ratio=self.entropy_ratio,
            )
            score_df = temp_model.evaluate_dual_system(df, low_indicators, green_indicators)["result_df"].copy()
            score_df["时间衰减因子"] = decay
            outputs.append(score_df)
        return pd.concat(outputs, ignore_index=True)


def build_region_summary(score_df: pd.DataFrame) -> pd.DataFrame:
    """汇总四大区域年度平均综合发展水平。"""
    summary = (
        score_df.groupby(["年份", "区域"], as_index=False)["双系统综合发展指数"]
        .mean()
        .rename(columns={"双系统综合发展指数": "区域平均综合发展水平"})
    )
    return summary


def build_province_ranking(score_df: pd.DataFrame, target_year: int = 2023) -> pd.DataFrame:
    """提取指定年份的省份综合发展水平排名。"""
    ranking = score_df.loc[score_df["年份"] == target_year, ["省份", "区域", "双系统综合发展指数"]].copy()
    ranking = ranking.sort_values("双系统综合发展指数", ascending=False).reset_index(drop=True)
    ranking["排名"] = np.arange(1, len(ranking) + 1)
    return ranking


def build_biennial_map_data(score_df: pd.DataFrame, year_mapping: Dict[int, str]) -> pd.DataFrame:
    """生成每两年平均一次的省份综合指数，用于用户后续自行制图。"""
    temp = score_df[["年份", "省份", "区域", "双系统综合发展指数"]].copy()
    temp["阶段"] = temp["年份"].map(year_mapping)
    result = (
        temp.groupby(["阶段", "省份", "区域"], as_index=False)["双系统综合发展指数"]
        .mean()
        .rename(columns={"双系统综合发展指数": "双年度平均综合发展指数"})
    )
    return result


def build_weight_comparison_table(dual_result: Dict[str, pd.DataFrame | pd.Series | Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """拼接双系统权重结果，供绘图与论文表格输出。"""
    low_table = dual_result["low"]["weight_table"].copy()
    low_table["系统"] = "低空经济"

    green_table = dual_result["green"]["weight_table"].copy()
    green_table["系统"] = "绿色交通"

    columns = ["系统", "指标", "CRITIC权重", "熵权", "综合权重"]
    return pd.concat([low_table[columns], green_table[columns]], ignore_index=True)
