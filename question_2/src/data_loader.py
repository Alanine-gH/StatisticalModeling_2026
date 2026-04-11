"""数据读取与预处理模块。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

LOW_INDICATORS = [
    "低空经济产业规模",
    "航空运输业就业人员数",
    "低空经济企业个数",
    "低空交通起降基础设施数量",
    "互联网宽带接入端口",
    "民用通用机场数量",
    "航空邮路总长度",
    "民用机场旅客吞吐量",
    "货邮吞吐量",
    "移动电话基站密度",
    "公共安全财政支出",
    "科学研究和技术服务业城镇单位就业人员",
    "规模以上工业企业R&D经费",
    "高技术产业有效发明专利数",
    "低空经济政策数量",
]

GREEN_INDICATORS = [
    "道路网密度",
    "万人公交车标台数",
    "公交站点500m覆盖率",
    "新能源充电桩数量",
    "人均道路面积",
    "绿色交通出行分担率",
    "私家车拥有量",
    "城市化率",
    "人口密度",
    "人均GDP",
    "全年空气质量优良天数",
    "可吸入颗粒物年均浓度",
    "二氧化碳年排放量",
    "高峰期平均车速",
    "高峰拥堵延时指数",
    "建成区绿化覆盖率",
    "环保财政支出占比",
    "道路运输财政支出占比",
    "新能源车辆占有率",
]


@dataclass
class DataBundle:
    panel_df: pd.DataFrame
    weight_matrix: pd.DataFrame
    geo_weight_matrix: pd.DataFrame
    econ_weight_matrix: pd.DataFrame
    low_altitude_cols: List[str]
    green_transport_cols: List[str]
    low_standardized: pd.DataFrame
    green_standardized: pd.DataFrame


def min_max_scale(series: pd.Series, positive: bool = True) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    s_min, s_max = s.min(), s.max()
    if pd.isna(s_min) or pd.isna(s_max) or np.isclose(s_max, s_min):
        scaled = pd.Series(np.zeros(len(s)), index=s.index)
    else:
        scaled = (s - s_min) / (s_max - s_min)
    return scaled if positive else 1.0 - scaled


def entropy_weight(df: pd.DataFrame) -> pd.Series:
    eps = 1e-12
    x = df.astype(float).copy() + eps
    p = x.div(x.sum(axis=0), axis=1)
    n = len(p)
    k = 1.0 / np.log(max(n, 2))
    e = -(k * (p * np.log(p)).sum(axis=0))
    d = 1 - e
    if np.isclose(d.sum(), 0):
        return pd.Series(np.repeat(1.0 / len(d), len(d)), index=df.columns)
    return d / d.sum()


def build_system_index(df: pd.DataFrame, positive_map: Dict[str, bool]) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    scaled = pd.DataFrame(index=df.index)
    for col in df.columns:
        scaled[col] = min_max_scale(df[col], positive=positive_map.get(col, True))
    weights = entropy_weight(scaled)
    score = scaled.mul(weights, axis=1).sum(axis=1)
    return scaled, score, weights


def _read_excel_panel(raw_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    low_file = raw_dir / "低空经济指标数据.xlsx"
    green_file = raw_dir / "绿色交通指标数据.xlsx"
    unified_file = raw_dir / "data.xlsx"
    if unified_file.exists():
        return pd.read_excel(unified_file, sheet_name=0), pd.read_excel(unified_file, sheet_name=1)
    if not low_file.exists() or not green_file.exists():
        raise FileNotFoundError(f"未找到可用输入文件：{low_file.name} / {green_file.name}")
    return pd.read_excel(low_file), pd.read_excel(green_file)


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().replace("\n", "").replace("  ", " ") for c in out.columns]
    return out


def _ensure_required_columns(df: pd.DataFrame, expected_cols: List[str], dataset_name: str) -> None:
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} 缺少字段：{missing}")


def build_panel_dataframe(raw_dir: Path, processed_dir: Path) -> Tuple[pd.DataFrame, List[str], List[str], pd.DataFrame, pd.DataFrame]:
    """分别基于两张指标表构建低空经济系统指数与绿色交通系统指数。"""
    low_df, green_df = _read_excel_panel(raw_dir)
    low_df = normalize_column_names(low_df)
    green_df = normalize_column_names(green_df)

    key_cols = ["年份", "省份"]
    _ensure_required_columns(low_df, key_cols + LOW_INDICATORS, "低空经济数据")
    _ensure_required_columns(green_df, key_cols + GREEN_INDICATORS, "绿色交通数据")

    low_positive = {name: True for name in LOW_INDICATORS}
    green_positive = {name: True for name in GREEN_INDICATORS}
    for col in ["私家车拥有量", "可吸入颗粒物年均浓度", "二氧化碳年排放量", "高峰拥堵延时指数"]:
        green_positive[col] = False

    low_scaled, low_score, low_weights = build_system_index(low_df[LOW_INDICATORS], low_positive)
    green_scaled, green_score, green_weights = build_system_index(green_df[GREEN_INDICATORS], green_positive)

    low_out = pd.concat([low_df[key_cols], low_scaled.add_prefix("low_")], axis=1)
    low_out["low_score"] = low_score
    green_out = pd.concat([green_df[key_cols], green_scaled.add_prefix("green_")], axis=1)
    green_out["green_score"] = green_score

    panel = pd.merge(low_out, green_out, on=key_cols, how="inner")
    panel["年份"] = panel["年份"].astype(int)
    panel = panel.sort_values(["年份", "省份"]).reset_index(drop=True)

    # 第二问只将两套系统指数按“年份-省份”对齐到同一面板中，
    # 供“低空经济 -> 绿色交通”影响机制分析使用，不再额外合成为单一综合指数。

    processed_dir.mkdir(parents=True, exist_ok=True)
    low_standardized = pd.concat([low_df[key_cols], low_scaled], axis=1)
    green_standardized = pd.concat([green_df[key_cols], green_scaled], axis=1)
    low_standardized.to_csv(processed_dir / "低空经济_标准化结果.csv", index=False, encoding="utf-8-sig")
    green_standardized.to_csv(processed_dir / "绿色交通_标准化结果.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"指标": LOW_INDICATORS, "权重": low_weights.values, "系统指数": low_score.values}).to_csv(processed_dir / "低空经济_指标权重.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"指标": GREEN_INDICATORS, "权重": green_weights.values, "系统指数": green_score.values}).to_csv(processed_dir / "绿色交通_指标权重.csv", index=False, encoding="utf-8-sig")
    panel.to_csv(processed_dir / "processed_panel.csv", index=False, encoding="utf-8-sig")
    return panel, LOW_INDICATORS.copy(), GREEN_INDICATORS.copy(), low_standardized, green_standardized


def build_adjacency_matrix(provinces: List[str]) -> pd.DataFrame:
    adjacency: Dict[str, List[str]] = {
        "北京": ["天津", "河北"], "天津": ["北京", "河北"], "河北": ["北京", "天津", "山西", "内蒙古", "辽宁", "山东", "河南"],
        "山西": ["河北", "内蒙古", "陕西", "河南"], "内蒙古": ["黑龙江", "吉林", "辽宁", "河北", "山西", "陕西", "宁夏", "甘肃"],
        "辽宁": ["河北", "内蒙古", "吉林"], "吉林": ["黑龙江", "辽宁", "内蒙古"], "黑龙江": ["吉林", "内蒙古"],
        "上海": ["江苏", "浙江"], "江苏": ["上海", "浙江", "安徽", "山东"], "浙江": ["上海", "江苏", "安徽", "江西", "福建"],
        "安徽": ["江苏", "浙江", "江西", "湖北", "河南", "山东"], "福建": ["浙江", "江西", "广东"],
        "江西": ["浙江", "安徽", "湖北", "湖南", "广东", "福建"], "山东": ["河北", "河南", "安徽", "江苏"],
        "河南": ["河北", "山西", "陕西", "湖北", "安徽", "山东", "江苏"], "湖北": ["河南", "安徽", "江西", "湖南", "重庆", "陕西"],
        "湖南": ["湖北", "江西", "广东", "广西", "贵州", "重庆"], "广东": ["福建", "江西", "湖南", "广西", "海南"],
        "广西": ["广东", "湖南", "贵州", "云南", "海南"], "海南": ["广东", "广西"], "重庆": ["湖北", "湖南", "贵州", "四川", "陕西"],
        "四川": ["重庆", "贵州", "云南", "青海", "甘肃", "陕西"], "贵州": ["重庆", "四川", "云南", "广西", "湖南"],
        "云南": ["广西", "贵州", "四川", "青海"], "陕西": ["内蒙古", "山西", "河南", "湖北", "重庆", "四川", "甘肃", "宁夏", "青海"],
        "甘肃": ["新疆", "青海", "四川", "陕西", "宁夏", "内蒙古"], "青海": ["新疆", "四川", "甘肃", "陕西", "云南", "宁夏"],
        "宁夏": ["内蒙古", "陕西", "甘肃", "青海"], "新疆": ["青海", "甘肃"],
    }
    w = pd.DataFrame(0.0, index=provinces, columns=provinces)
    for prov in provinces:
        for nb in adjacency.get(prov, []):
            if nb in w.columns:
                w.loc[prov, nb] = 1.0
                w.loc[nb, prov] = 1.0
    return w.div(w.sum(axis=1).replace(0.0, 1.0), axis=0)


def build_economic_distance_matrix(panel_df: pd.DataFrame) -> pd.DataFrame:
    latest_year = panel_df["年份"].max()
    gdp = panel_df.loc[panel_df["年份"] == latest_year, ["省份", "green_人均GDP"]].set_index("省份")
    vals = gdp["green_人均GDP"].values.astype(float)
    dist = np.abs(vals.reshape(-1, 1) - vals.reshape(1, -1))
    econ_w = 1.0 / (dist + 1e-6)
    np.fill_diagonal(econ_w, 0.0)
    econ_w = pd.DataFrame(econ_w, index=gdp.index, columns=gdp.index)
    return econ_w.div(econ_w.sum(axis=1).replace(0.0, 1.0), axis=0)


def build_hybrid_weight_matrix(geo_w: pd.DataFrame, econ_w: pd.DataFrame, alpha: float = 0.7) -> pd.DataFrame:
    econ_align = econ_w.reindex(index=geo_w.index, columns=geo_w.columns).fillna(0.0)
    hybrid = (alpha * geo_w + (1.0 - alpha) * econ_align).copy()
    for idx in hybrid.index:
        hybrid.loc[idx, idx] = 0.0
    return hybrid.div(hybrid.sum(axis=1).replace(0.0, 1.0), axis=0)


def load_all_data(base_dir: Path) -> DataBundle:
    raw_dir = base_dir / "data" / "raw"
    processed_dir = base_dir / "data" / "processed"
    panel_df, low_cols, green_cols, low_standardized, green_standardized = build_panel_dataframe(raw_dir, processed_dir)
    provinces = sorted(panel_df["省份"].unique().tolist())
    geo_w = build_adjacency_matrix(provinces)
    econ_w = build_economic_distance_matrix(panel_df)
    hybrid_w = build_hybrid_weight_matrix(geo_w, econ_w, alpha=0.7)
    (base_dir / "output" / "results").mkdir(parents=True, exist_ok=True)
    geo_w.to_csv(base_dir / "output" / "results" / "weight_adjacency.csv", encoding="utf-8-sig")
    econ_w.to_csv(base_dir / "output" / "results" / "weight_economic.csv", encoding="utf-8-sig")
    hybrid_w.to_csv(base_dir / "output" / "results" / "weight_hybrid.csv", encoding="utf-8-sig")
    return DataBundle(panel_df, hybrid_w, geo_w, econ_w, low_cols, green_cols, low_standardized, green_standardized)
