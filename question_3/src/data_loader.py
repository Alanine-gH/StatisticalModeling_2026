"""数据读取与特征构造模块。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from question_3.config import PROCESSED_DIR, RANDOM_SEED, REGION_MAP, YEARS


@dataclass
class Question3DataBundle:
    """封装问题三建模所需的数据对象。"""

    panel_df: pd.DataFrame
    feature_cols: List[str]
    scenario_feature_cols: List[str]
    target_col: str
    train_df: pd.DataFrame
    valid_df: pd.DataFrame
    test_df: pd.DataFrame
    future_df: pd.DataFrame
    baseline_df: pd.DataFrame
    stronger_df: pd.DataFrame
    targeted_df: pd.DataFrame
    region_map: Dict[str, str]


class Question3DataLoader:
    """负责读取面板数据、构造特征并生成预测情景。"""

    def __init__(self, processed_dir: Path = PROCESSED_DIR) -> None:
        """初始化数据加载器。

        参数:
            processed_dir: 预处理数据所在目录。
        """
        self.processed_dir = Path(processed_dir)
        self.rng = np.random.default_rng(RANDOM_SEED)
        self.target_col = "green_score"
        self.base_feature_cols = [
            "low_score",
            "low_低空经济产业规模",
            "low_航空运输业就业人员数",
            "low_低空经济企业个数",
            "low_低空交通起降基础设施数量",
            "low_互联网宽带接入端口",
            "low_民用通用机场数量",
            "low_规模以上工业企业R&D经费",
            "low_高技术产业有效发明专利数",
            "low_低空经济政策数量",
            "green_道路网密度",
            "green_新能源充电桩数量",
            "green_绿色交通出行分担率",
            "green_新能源车辆占有率",
            "green_人均GDP",
            "green_城市化率",
        ]

    def load_data_bundle(self) -> Question3DataBundle:
        """读取完整数据并返回问题三所需的数据包。"""
        panel_df = self._read_panel_data()
        panel_df = self._add_region_info(panel_df)
        panel_df, feature_cols, scenario_feature_cols = self._build_features(panel_df)
        train_df, valid_df, test_df = self._split_dataset(panel_df)
        future_df = self._build_future_base(panel_df)
        baseline_df = self._build_future_scenario(panel_df, future_df, "baseline")
        stronger_df = self._build_future_scenario(panel_df, future_df, "stronger")
        targeted_df = self._build_future_scenario(panel_df, future_df, "targeted")
        return Question3DataBundle(
            panel_df=panel_df,
            feature_cols=feature_cols,
            scenario_feature_cols=scenario_feature_cols,
            target_col=self.target_col,
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            future_df=future_df,
            baseline_df=baseline_df,
            stronger_df=stronger_df,
            targeted_df=targeted_df,
            region_map=REGION_MAP,
        )

    def _read_panel_data(self) -> pd.DataFrame:
        """读取问题三处理后的省际面板数据。"""
        panel_path = self.processed_dir / "processed_panel.csv"
        if not panel_path.exists():
            raise FileNotFoundError(f"未找到数据文件: {panel_path}")
        panel_df = pd.read_csv(panel_path, encoding="utf-8-sig")
        panel_df["年份"] = panel_df["年份"].astype(int)
        panel_df = panel_df.sort_values(["省份", "年份"]).reset_index(drop=True)
        return panel_df

    def _add_region_info(self, panel_df: pd.DataFrame) -> pd.DataFrame:
        """为面板数据补充区域信息。"""
        output_df = panel_df.copy()
        output_df["区域"] = output_df["省份"].map(REGION_MAP).fillna("其他")
        return output_df

    def _build_features(self, panel_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """构造模型训练和未来预测的特征。"""
        feature_df = panel_df.copy()
        for col in self.base_feature_cols + [self.target_col]:
            feature_df[f"{col}_lag1"] = feature_df.groupby("省份")[col].shift(1)
        feature_df["low_policy_intensity"] = (
            0.35 * feature_df["low_低空经济产业规模"]
            + 0.25 * feature_df["low_低空交通起降基础设施数量"]
            + 0.20 * feature_df["low_规模以上工业企业R&D经费"]
            + 0.20 * feature_df["low_低空经济政策数量"]
        )
        feature_df["green_support_base"] = (
            0.40 * feature_df["green_新能源充电桩数量"]
            + 0.35 * feature_df["green_新能源车辆占有率"]
            + 0.25 * feature_df["green_绿色交通出行分担率"]
        )
        feature_df["policy_trend"] = feature_df.groupby("省份")["low_policy_intensity"].diff().fillna(0.0)
        feature_df["gdp_trend"] = feature_df.groupby("省份")["green_人均GDP"].diff().fillna(0.0)
        feature_df["target_trend"] = feature_df.groupby("省份")[self.target_col].diff().fillna(0.0)
        feature_df["year_index"] = feature_df["年份"] - min(YEARS)
        feature_df["region_code"] = feature_df["区域"].map({"东部": 0, "中部": 1, "西部": 2, "东北": 3}).fillna(4)

        feature_cols = [
            "low_score",
            "low_policy_intensity",
            "green_support_base",
            "low_低空交通起降基础设施数量",
            "low_互联网宽带接入端口",
            "low_规模以上工业企业R&D经费",
            "low_高技术产业有效发明专利数",
            "low_低空经济政策数量",
            "green_道路网密度",
            "green_新能源充电桩数量",
            "green_绿色交通出行分担率",
            "green_新能源车辆占有率",
            "green_人均GDP",
            "green_城市化率",
            f"{self.target_col}_lag1",
            "policy_trend",
            "gdp_trend",
            "target_trend",
            "year_index",
            "region_code",
        ]
        scenario_feature_cols = [
            "low_score",
            "low_policy_intensity",
            "green_support_base",
            "low_低空交通起降基础设施数量",
            "low_互联网宽带接入端口",
            "low_规模以上工业企业R&D经费",
            "low_高技术产业有效发明专利数",
            "low_低空经济政策数量",
            "green_道路网密度",
            "green_新能源充电桩数量",
            "green_绿色交通出行分担率",
            "green_新能源车辆占有率",
            "green_人均GDP",
            "green_城市化率",
            f"{self.target_col}_lag1",
            "policy_trend",
            "gdp_trend",
            "target_trend",
            "year_index",
            "region_code",
        ]
        feature_df = feature_df.dropna(subset=feature_cols).reset_index(drop=True)
        return feature_df, feature_cols, scenario_feature_cols

    def _split_dataset(self, panel_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """按年份划分训练集、验证集和测试集。"""
        train_df = panel_df[panel_df["年份"].between(2017, 2020)].copy()
        valid_df = panel_df[panel_df["年份"].between(2021, 2022)].copy()
        test_df = panel_df[panel_df["年份"] == 2023].copy()
        return train_df, valid_df, test_df

    def _build_future_base(self, panel_df: pd.DataFrame) -> pd.DataFrame:
        """基于 2023 年状态构造 2024-2030 年预测底表。"""
        latest_df = panel_df[panel_df["年份"] == panel_df["年份"].max()].copy()
        future_records: List[pd.Series] = []
        for _, row in latest_df.iterrows():
            previous_target = float(row[self.target_col])
            for year in range(2024, 2031):
                current_row = row.copy()
                current_row["年份"] = year
                current_row[f"{self.target_col}_lag1"] = previous_target
                current_row["year_index"] = year - min(YEARS)
                future_records.append(current_row)
        future_df = pd.DataFrame(future_records).reset_index(drop=True)
        return future_df

    def _build_future_scenario(self, history_df: pd.DataFrame, future_df: pd.DataFrame, scenario: str) -> pd.DataFrame:
        """生成不同投入情景下的未来特征数据。"""
        scenario_df = future_df.copy()
        province_growth = self._estimate_growth_rate(history_df)
        for idx, row in scenario_df.iterrows():
            province = row["省份"]
            region = row["区域"]
            years_ahead = int(row["年份"] - 2023)
            growth_factor = (1 + province_growth.get(province, 0.03)) ** years_ahead
            if scenario == "baseline":
                policy_multiplier = 1.00
                infrastructure_multiplier = 1.00
                innovation_multiplier = 1.00
            elif scenario == "stronger":
                policy_multiplier = 1.20
                infrastructure_multiplier = 1.20
                innovation_multiplier = 1.18
            elif scenario == "targeted":
                if region in {"中部", "西部"}:
                    policy_multiplier = 1.28
                    infrastructure_multiplier = 1.30
                    innovation_multiplier = 1.22
                else:
                    policy_multiplier = 0.95
                    infrastructure_multiplier = 0.95
                    innovation_multiplier = 0.96
            else:
                raise ValueError(f"未知情景: {scenario}")

            scenario_df.at[idx, "low_低空经济产业规模"] = min(row["low_低空经济产业规模"] * growth_factor * policy_multiplier, 1.4)
            scenario_df.at[idx, "low_低空交通起降基础设施数量"] = min(row["low_低空交通起降基础设施数量"] * growth_factor * infrastructure_multiplier, 1.4)
            scenario_df.at[idx, "low_互联网宽带接入端口"] = min(row["low_互联网宽带接入端口"] * (1 + 0.02 * years_ahead), 1.4)
            scenario_df.at[idx, "low_规模以上工业企业R&D经费"] = min(row["low_规模以上工业企业R&D经费"] * growth_factor * innovation_multiplier, 1.4)
            scenario_df.at[idx, "low_高技术产业有效发明专利数"] = min(row["low_高技术产业有效发明专利数"] * growth_factor * innovation_multiplier, 1.4)
            scenario_df.at[idx, "low_低空经济政策数量"] = min(row["low_低空经济政策数量"] + 0.03 * years_ahead * policy_multiplier, 1.5)
            scenario_df.at[idx, "low_score"] = np.mean(
                [
                    scenario_df.at[idx, "low_低空经济产业规模"],
                    scenario_df.at[idx, "low_低空交通起降基础设施数量"],
                    scenario_df.at[idx, "low_规模以上工业企业R&D经费"],
                    scenario_df.at[idx, "low_高技术产业有效发明专利数"],
                    scenario_df.at[idx, "low_低空经济政策数量"],
                ]
            )
            scenario_df.at[idx, "green_新能源充电桩数量"] = min(row["green_新能源充电桩数量"] * (1 + 0.03 * years_ahead), 1.5)
            scenario_df.at[idx, "green_新能源车辆占有率"] = min(row["green_新能源车辆占有率"] * (1 + 0.025 * years_ahead), 1.5)
            scenario_df.at[idx, "green_绿色交通出行分担率"] = min(row["green_绿色交通出行分担率"] * (1 + 0.015 * years_ahead), 1.5)
            scenario_df.at[idx, "green_人均GDP"] = min(row["green_人均GDP"] * (1 + 0.018 * years_ahead), 1.5)
            scenario_df.at[idx, "green_城市化率"] = min(row["green_城市化率"] * (1 + 0.008 * years_ahead), 1.2)
            scenario_df.at[idx, "low_policy_intensity"] = (
                0.35 * scenario_df.at[idx, "low_低空经济产业规模"]
                + 0.25 * scenario_df.at[idx, "low_低空交通起降基础设施数量"]
                + 0.20 * scenario_df.at[idx, "low_规模以上工业企业R&D经费"]
                + 0.20 * scenario_df.at[idx, "low_低空经济政策数量"]
            )
            scenario_df.at[idx, "green_support_base"] = (
                0.40 * scenario_df.at[idx, "green_新能源充电桩数量"]
                + 0.35 * scenario_df.at[idx, "green_新能源车辆占有率"]
                + 0.25 * scenario_df.at[idx, "green_绿色交通出行分担率"]
            )
            scenario_df.at[idx, "policy_trend"] = scenario_df.at[idx, "low_policy_intensity"] - row["low_policy_intensity"]
            scenario_df.at[idx, "gdp_trend"] = scenario_df.at[idx, "green_人均GDP"] - row["green_人均GDP"]
            scenario_df.at[idx, "target_trend"] = scenario_df.at[idx, f"{self.target_col}_lag1"] - row[f"{self.target_col}_lag1"]
        return scenario_df

    def _estimate_growth_rate(self, panel_df: pd.DataFrame) -> Dict[str, float]:
        """估计各省份低空经济综合投入的历史平均增速。"""
        growth_rate: Dict[str, float] = {}
        for province, sub_df in panel_df.groupby("省份"):
            ordered = sub_df.sort_values("年份")
            growth = ordered["low_score"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
            growth_rate[province] = float(np.clip(growth.mean() if not growth.empty else 0.03, 0.01, 0.12))
        return growth_rate


def standardize_features(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """对训练集、验证集和测试集特征进行标准化。"""
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_df[feature_cols])
    valid_x = scaler.transform(valid_df[feature_cols])
    test_x = scaler.transform(test_df[feature_cols])
    return train_x, valid_x, test_x, scaler


def standardize_future_features(
    future_df: pd.DataFrame,
    feature_cols: List[str],
    scaler: StandardScaler,
) -> np.ndarray:
    """使用训练集标准化器处理未来情景特征。"""
    return scaler.transform(future_df[feature_cols])
