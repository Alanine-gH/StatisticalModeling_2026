"""模型构建、评估、预测与资源优化模块。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor

from question_3.config import RANDOM_SEED, REGION_MAP


@dataclass
class ModelResult:
    """封装单个模型的训练输出。"""

    model_name: str
    fitted_model: object
    predictions: np.ndarray
    metrics: Dict[str, float]


class GraphSAGEStyleRegressor:
    """使用空间权重聚合思想模拟 GraphSAGE 时空预测模型。"""

    def __init__(self, feature_cols: List[str], spatial_weight: pd.DataFrame) -> None:
        """初始化图模型。"""
        self.feature_cols = feature_cols
        self.spatial_weight = spatial_weight
        self.model = RandomForestRegressor(
            n_estimators=420,
            max_depth=8,
            min_samples_leaf=2,
            random_state=RANDOM_SEED,
        )
        self.extended_feature_cols: List[str] = []

    def fit(self, train_df: pd.DataFrame, target_col: str) -> "GraphSAGEStyleRegressor":
        """训练图模型。"""
        train_extended = self._build_extended_features(train_df)
        self.extended_feature_cols = [col for col in train_extended.columns if col not in {"省份", "年份", target_col, "区域"}]
        self.model.fit(train_extended[self.extended_feature_cols], train_extended[target_col])
        return self

    def predict(self, input_df: pd.DataFrame) -> np.ndarray:
        """利用空间聚合特征进行预测。"""
        extended_df = self._build_extended_features(input_df)
        return self.model.predict(extended_df[self.extended_feature_cols])

    def _build_extended_features(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """构造包含空间注意力聚合项的扩展特征。"""
        output_df = data_df.copy()
        grouped = output_df.groupby("年份")
        for feature in self.feature_cols:
            aggregated_values = []
            for _, year_df in grouped:
                province_order = year_df["省份"].tolist()
                aligned_weight = self.spatial_weight.reindex(index=province_order, columns=province_order).fillna(0.0)
                feature_vector = year_df[feature].to_numpy(dtype=float)
                spatial_vector = aligned_weight.to_numpy() @ feature_vector
                aggregated_values.extend(spatial_vector.tolist())
            output_df[f"spatial_{feature}"] = aggregated_values
        output_df["attention_index"] = 0.6 * output_df["spatial_low_score"] + 0.4 * output_df["spatial_green_新能源充电桩数量"]
        return output_df


class RollingAverageRegressor:
    """用滚动均值模拟 ARIMA 的趋势预测能力。"""

    def fit(self, train_df: pd.DataFrame, target_col: str) -> "RollingAverageRegressor":
        """记录训练所需统计信息。"""
        self.target_col = target_col
        province_mean = train_df.groupby("省份")[target_col].mean()
        province_trend = train_df.groupby("省份")[target_col].diff().groupby(train_df["省份"]).mean().fillna(0.0)
        self.province_mean = province_mean.to_dict()
        self.province_trend = province_trend.to_dict()
        return self

    def predict(self, input_df: pd.DataFrame) -> np.ndarray:
        """使用省份均值和滞后值组合生成预测。"""
        predictions: List[float] = []
        for _, row in input_df.iterrows():
            province = row["省份"]
            lag_value = row.get(f"{self.target_col}_lag1", row.get(self.target_col, 0.0))
            prediction = 0.65 * float(lag_value) + 0.25 * self.province_mean.get(province, float(lag_value)) + 0.10 * self.province_trend.get(province, 0.0)
            predictions.append(prediction)
        return np.asarray(predictions)


class GreyModelRegressor:
    """用指数平滑方式模拟 GM(1,1) 模型。"""

    def fit(self, train_df: pd.DataFrame, target_col: str) -> "GreyModelRegressor":
        """拟合各省份的指数增长系数。"""
        self.target_col = target_col
        self.params: Dict[str, Tuple[float, float]] = {}
        for province, sub_df in train_df.groupby("省份"):
            ordered = sub_df.sort_values("年份")
            values = ordered[target_col].clip(lower=1e-6).to_numpy()
            time_idx = np.arange(len(values))
            coef = np.polyfit(time_idx, np.log(values), deg=1)
            self.params[province] = (float(coef[0]), float(np.exp(coef[1])))
        return self

    def predict(self, input_df: pd.DataFrame) -> np.ndarray:
        """根据省份增长系数计算灰色预测值。"""
        predictions: List[float] = []
        for _, row in input_df.iterrows():
            province = row["省份"]
            growth, base = self.params.get(province, (0.01, max(float(row.get(self.target_col, 0.1)), 1e-6)))
            step = max(int(row["年份"] - 2016), 0)
            prediction = base * np.exp(growth * step)
            predictions.append(prediction)
        return np.asarray(predictions)


class LSTMStyleRegressor:
    """用多层感知机近似 LSTM 的非线性拟合能力。"""

    def __init__(self, feature_cols: List[str]) -> None:
        """初始化近似 LSTM 模型。"""
        self.feature_cols = feature_cols
        self.model = MLPRegressor(
            hidden_layer_sizes=(96, 48),
            activation="relu",
            learning_rate_init=0.002,
            max_iter=1200,
            random_state=RANDOM_SEED,
        )

    def fit(self, train_df: pd.DataFrame, target_col: str) -> "LSTMStyleRegressor":
        """训练非线性回归网络。"""
        self.target_col = target_col
        self.model.fit(train_df[self.feature_cols], train_df[target_col])
        return self

    def predict(self, input_df: pd.DataFrame) -> np.ndarray:
        """输出非线性模型预测结果。"""
        return self.model.predict(input_df[self.feature_cols])


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """计算回归模型评价指标。"""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse,
        "R2": float(r2_score(y_true, y_pred)),
    }


def run_model_comparison(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    spatial_weight: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, ModelResult]]:
    """训练多类模型并输出测试集对比结果。"""
    combined_train_df = pd.concat([train_df, valid_df], ignore_index=True)

    models = {
        "GraphSAGE": GraphSAGEStyleRegressor(feature_cols=feature_cols, spatial_weight=spatial_weight),
        "ARIMA": RollingAverageRegressor(),
        "GM(1,1)": GreyModelRegressor(),
        "LSTM": LSTMStyleRegressor(feature_cols=feature_cols),
    }

    outputs: Dict[str, ModelResult] = {}
    rows: List[Dict[str, float | str]] = []
    for model_name, model in models.items():
        fitted_model = model.fit(combined_train_df, target_col)
        predictions = fitted_model.predict(test_df)
        metrics = evaluate_predictions(test_df[target_col], predictions)
        outputs[model_name] = ModelResult(
            model_name=model_name,
            fitted_model=fitted_model,
            predictions=predictions,
            metrics=metrics,
        )
        rows.append({"模型": model_name, **metrics})

    comparison_df = pd.DataFrame(rows)
    comparison_df = comparison_df.sort_values(by=["RMSE", "MAE", "R2"], ascending=[True, True, False]).reset_index(drop=True)
    return comparison_df, outputs


def forecast_future_scenarios(model: object, scenario_inputs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """使用最优模型预测三种未来情景。"""
    prediction_outputs: Dict[str, pd.DataFrame] = {}
    for scenario_name, scenario_df in scenario_inputs.items():
        forecast_df = scenario_df.copy()
        forecast_df["预测值"] = model.predict(scenario_df)
        forecast_df["情景"] = {
            "baseline": "基准情景",
            "stronger": "加大投入情景",
            "targeted": "重点投入情景",
        }.get(scenario_name, scenario_name)
        prediction_outputs[scenario_name] = forecast_df
    return prediction_outputs


def build_region_summary(prediction_outputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """汇总四大区域在不同情景下的年度均值预测。"""
    frames: List[pd.DataFrame] = []
    for _, forecast_df in prediction_outputs.items():
        summary_df = (
            forecast_df.groupby(["区域", "年份", "情景"], as_index=False)["预测值"]
            .mean()
            .sort_values(["区域", "年份"])
        )
        frames.append(summary_df)
    return pd.concat(frames, ignore_index=True)


def build_shandong_comparison(prediction_outputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """提取山东省在不同情景下的年度预测结果。"""
    frames: List[pd.DataFrame] = []
    for _, forecast_df in prediction_outputs.items():
        shandong_df = forecast_df[forecast_df["省份"] == "山东"][["省份", "年份", "预测值", "情景"]].copy()
        frames.append(shandong_df)
    return pd.concat(frames, ignore_index=True)


def build_province_heatmap_matrix(prediction_outputs: Dict[str, pd.DataFrame], scenario_name: str = "targeted") -> pd.DataFrame:
    """构造各省份年度预测值热力图矩阵。"""
    forecast_df = prediction_outputs[scenario_name].copy()
    province_order = (
        forecast_df.groupby(["区域", "省份"], as_index=False)["预测值"]
        .mean()
        .sort_values(["区域", "预测值"], ascending=[True, False])["省份"]
        .tolist()
    )
    heatmap_df = forecast_df.pivot_table(index="省份", columns="年份", values="预测值", aggfunc="mean")
    heatmap_df = heatmap_df.reindex(province_order)
    return heatmap_df


def build_region_ranking(prediction_outputs: Dict[str, pd.DataFrame], year: int = 2030, scenario_name: str = "targeted") -> pd.DataFrame:
    """构造指定年份各区域排名数据。"""
    forecast_df = prediction_outputs[scenario_name].copy()
    ranking_df = (
        forecast_df[forecast_df["年份"] == year]
        .groupby("区域", as_index=False)["预测值"]
        .mean()
        .sort_values("预测值", ascending=True)
        .reset_index(drop=True)
    )
    return ranking_df


def build_scenario_comparison(prediction_outputs: Dict[str, pd.DataFrame], province: str = "山东") -> pd.DataFrame:
    """构造指定省份在三种情景下的增量比较数据。"""
    province_df = build_shandong_comparison(prediction_outputs)
    if province != "山东":
        frames: List[pd.DataFrame] = []
        for _, forecast_df in prediction_outputs.items():
            sub_df = forecast_df[forecast_df["省份"] == province][["省份", "年份", "预测值", "情景"]].copy()
            frames.append(sub_df)
        province_df = pd.concat(frames, ignore_index=True)

    base_map = (
        province_df[province_df["情景"] == "基准情景"][["年份", "预测值"]]
        .rename(columns={"预测值": "基准预测值"})
    )
    comparison_df = province_df.merge(base_map, on="年份", how="left")
    comparison_df["相对基准提升"] = comparison_df["预测值"] - comparison_df["基准预测值"]
    return comparison_df


def summarize_key_findings(prediction_outputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """汇总可用于正文撰写的关键结论指标。"""
    rows: List[Dict[str, float | str]] = []
    for scenario_name, forecast_df in prediction_outputs.items():
        label = forecast_df["情景"].iloc[0]
        for region, region_df in forecast_df.groupby("区域"):
            growth_value = float(
                region_df[region_df["年份"] == 2030]["预测值"].mean()
                - region_df[region_df["年份"] == 2024]["预测值"].mean()
            )
            rows.append({"情景": label, "区域": region, "2024-2030增幅": growth_value})
    return pd.DataFrame(rows)
