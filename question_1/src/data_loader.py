"""数据读取与预处理模块。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from question_1.config import PROCESSED_DIR, RAW_DIR, REGION_MAP, RESULT_DIR, YEARS


# 按主题框架中的二级指标顺序定义两个系统的指标列表
LOW_ALTITUDE_INDICATORS: List[str] = [
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

GREEN_TRANSPORT_INDICATORS: List[str] = [
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


def ensure_output_dirs() -> None:
    """创建结果输出目录，避免后续保存文件时报错。"""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (Path(RESULT_DIR).parent / "figures").mkdir(parents=True, exist_ok=True)


def load_standardized_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """读取已经标准化完成的低空经济与绿色交通数据。"""
    low_path = PROCESSED_DIR / "低空经济_标准化结果.csv"
    green_path = PROCESSED_DIR / "绿色交通_标准化结果.csv"

    low_df = pd.read_csv(low_path, encoding="utf-8-sig")
    green_df = pd.read_csv(green_path, encoding="utf-8-sig")
    return low_df, green_df


def validate_indicator_columns(df: pd.DataFrame, indicators: List[str], system_name: str) -> None:
    """检查数据中是否包含指定系统全部指标，避免因列名不一致导致模型失败。"""
    missing = [col for col in indicators if col not in df.columns]
    if missing:
        raise ValueError(f"{system_name}缺少以下指标列: {missing}")


def sort_panel_data(df: pd.DataFrame) -> pd.DataFrame:
    """统一按年份、省份排序，确保时序分析和权重计算顺序稳定。"""
    return df.sort_values(["年份", "省份"]).reset_index(drop=True)


def attach_region(df: pd.DataFrame) -> pd.DataFrame:
    """为省份面板数据补充四大区域标签。"""
    result = df.copy()
    result["区域"] = result["省份"].map(REGION_MAP)
    if result["区域"].isna().any():
        missing = sorted(result.loc[result["区域"].isna(), "省份"].unique().tolist())
        raise ValueError(f"以下省份未配置区域映射: {missing}")
    return result


def merge_system_panels(low_df: pd.DataFrame, green_df: pd.DataFrame) -> pd.DataFrame:
    """合并双系统标准化数据，形成后续综合指数计算所需的联合面板。"""
    validate_indicator_columns(low_df, LOW_ALTITUDE_INDICATORS, "低空经济")
    validate_indicator_columns(green_df, GREEN_TRANSPORT_INDICATORS, "绿色交通")

    low_panel = sort_panel_data(low_df[["年份", "省份", *LOW_ALTITUDE_INDICATORS]].copy())
    green_panel = sort_panel_data(green_df[["年份", "省份", *GREEN_TRANSPORT_INDICATORS]].copy())

    merged = pd.merge(low_panel, green_panel, on=["年份", "省份"], how="inner", validate="one_to_one")
    merged = attach_region(merged)
    return merged


def get_indicator_groups() -> Dict[str, List[str]]:
    """返回双系统指标分组字典，便于模型层统一调用。"""
    return {
        "low": LOW_ALTITUDE_INDICATORS,
        "green": GREEN_TRANSPORT_INDICATORS,
        "all": LOW_ALTITUDE_INDICATORS + GREEN_TRANSPORT_INDICATORS,
    }


def save_dataframe(df: pd.DataFrame, file_name: str) -> Path:
    """将结果数据表保存到结果目录，并返回保存路径。"""
    ensure_output_dirs()
    output_path = RESULT_DIR / file_name
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def get_year_biennial_mapping() -> Dict[int, str]:
    """生成双年度分组标签，用于地图数据导出。"""
    return {
        2016: "2016-2017",
        2017: "2016-2017",
        2018: "2018-2019",
        2019: "2018-2019",
        2020: "2020-2021",
        2021: "2020-2021",
        2022: "2022-2023",
        2023: "2022-2023",
    }


def check_year_integrity(df: pd.DataFrame) -> None:
    """检查年份范围是否覆盖研究期，避免缺年导致评价结果失真。"""
    existing_years = sorted(df["年份"].unique().tolist())
    missing_years = [year for year in YEARS if year not in existing_years]
    if missing_years:
        raise ValueError(f"标准化数据缺少以下年份: {missing_years}")


def load_raw_data_if_exists() -> Dict[str, pd.DataFrame]:
    """尝试读取原始Excel数据，便于后续扩展核验；当前主流程不强依赖。"""
    raw_data: Dict[str, pd.DataFrame] = {}
    for file_name in ["低空经济指标数据.xlsx", "绿色交通指标数据.xlsx"]:
        file_path = RAW_DIR / file_name
        if file_path.exists():
            raw_data[file_name] = pd.read_excel(file_path)
    return raw_data
