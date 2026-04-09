"""项目配置文件：集中管理路径、变量和参数。"""

from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FIG_DIR = BASE_DIR / "output" / "figures"
RESULT_DIR = BASE_DIR / "output" / "results"

YEARS = list(range(2016, 2024))
RANDOM_SEED = 2026

# 控制变量：尽量选取绿色交通中的宏观背景变量，避免与被解释变量内涵过度重叠
CONTROL_VARS = [
    "green_城市化率",
    "green_人口密度",
    "green_人均GDP",
]

# 三条并行中介路径对应变量，统一使用本轮新数据字段
MEDIATORS = {
    "技术创新传导": "low_规模以上工业企业R&D经费",
    "产业结构升级": "low_低空经济产业规模",
    "交通结构优化": "green_新能源车辆占有率",
}

# 四大区域划分
REGION_MAP = {
    "北京": "东部",
    "天津": "东部",
    "河北": "东部",
    "上海": "东部",
    "江苏": "东部",
    "浙江": "东部",
    "福建": "东部",
    "山东": "东部",
    "广东": "东部",
    "海南": "东部",
    "辽宁": "东北",
    "吉林": "东北",
    "黑龙江": "东北",
    "山西": "中部",
    "安徽": "中部",
    "江西": "中部",
    "河南": "中部",
    "湖北": "中部",
    "湖南": "中部",
    "内蒙古": "西部",
    "广西": "西部",
    "重庆": "西部",
    "四川": "西部",
    "贵州": "西部",
    "云南": "西部",
    "陕西": "西部",
    "甘肃": "西部",
    "青海": "西部",
    "宁夏": "西部",
    "新疆": "西部",
}

# 兼容“根目录文件名不统一”的候选输入
ROOT_RAW_CANDIDATES = [
    "data.xlsx",
    "低空经济指标数据.xlsx",
    "绿色交通指标数据.xlsx",
]

ROOT_PROCESSED_CANDIDATES = [
    "低空经济_标准化结果.xlsx",
    "绿色交通_标准化结果.xlsx",
]
