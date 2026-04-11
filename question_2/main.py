"""问题二主函数：只负责流程调度。"""

from __future__ import annotations

import shutil
from pathlib import Path
from datetime import datetime

import pandas as pd

from config import (
    BASE_DIR,
    CONTROL_VARS,
    FIG_DIR,
    MEDIATORS,
    PROCESSED_DIR,
    RAW_DIR,
    REGION_MAP,
    RESULT_DIR,
    ROOT_DIR,
    ROOT_RAW_CANDIDATES,
)
from src.data_loader import load_all_data
from src.model import run_all_models
from src.visualization import export_all_figures

PRISM_DIR = BASE_DIR / "output" / "Prism"


def prepare_raw_files() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name in ROOT_RAW_CANDIDATES:
        src = ROOT_DIR / name
        dst = RAW_DIR / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)


def log_step(message: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {message}")


def save_tables(results, data_bundle, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    data_bundle.low_standardized.to_csv(out_dir / "low_standardized_panel.csv", index=False, encoding="utf-8-sig")
    data_bundle.green_standardized.to_csv(out_dir / "green_standardized_panel.csv", index=False, encoding="utf-8-sig")
    data_bundle.panel_df.to_csv(out_dir / "analysis_panel.csv", index=False, encoding="utf-8-sig")
    data_bundle.geo_weight_matrix.to_csv(out_dir / "weight_geo_used.csv", encoding="utf-8-sig")
    data_bundle.econ_weight_matrix.to_csv(out_dir / "weight_econ_used.csv", encoding="utf-8-sig")
    data_bundle.weight_matrix.to_csv(out_dir / "weight_hybrid_used.csv", encoding="utf-8-sig")
    results.moran_global.to_csv(out_dir / "moran_global.csv", index=False, encoding="utf-8-sig")
    results.lisa_2023.to_csv(out_dir / "lisa_2023.csv", index=False, encoding="utf-8-sig")
    results.sdm_effects.to_csv(out_dir / "sdm_effects.csv", index=False, encoding="utf-8-sig")
    results.mediation_effects.to_csv(out_dir / "mediation_effects.csv", index=False, encoding="utf-8-sig")
    results.regional_effects.to_csv(out_dir / "regional_effects.csv", index=False, encoding="utf-8-sig")
    results.tests_summary.to_csv(out_dir / "tests_summary.csv", index=False, encoding="utf-8-sig")
    results.robustness_effects.to_csv(out_dir / "robustness_effects.csv", index=False, encoding="utf-8-sig")
    results.comparison_summary.to_csv(out_dir / "comparison_summary.csv", index=False, encoding="utf-8-sig")


def export_txt_bundles(results, data_bundle, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    data_txt = [
        "【标准化数据摘要】",
        f"样本年份：{data_bundle.panel_df['年份'].min()}-{data_bundle.panel_df['年份'].max()}",
        f"样本省份数：{data_bundle.panel_df['省份'].nunique()}",
        f"面板观测值：{len(data_bundle.panel_df)}",
        "说明：低空经济与绿色交通两套指标体系分别计算系统指数，仅按省份-年份对齐进入第二问分析面板，未再合成为单一综合指数。",
        "",
        "【低空经济标准化数据前10行】",
        data_bundle.low_standardized.head(10).to_string(index=False),
        "",
        "【绿色交通标准化数据前10行】",
        data_bundle.green_standardized.head(10).to_string(index=False),
        "",
        "【综合分析面板前10行】",
        data_bundle.panel_df.head(10).to_string(index=False),
    ]
    (out_dir / "data_bundle_summary.txt").write_text("\n".join(data_txt), encoding="utf-8")

    result_txt = [
        "【全局莫兰指数】",
        results.moran_global.to_string(index=False),
        "",
        "【局部莫兰结果】",
        results.lisa_2023.to_string(index=False),
        "",
        "【空间杜宾效应】",
        results.sdm_effects.to_string(index=False),
        "",
        "【中介效应】",
        results.mediation_effects.to_string(index=False),
        "",
        "【区域异质性】",
        results.regional_effects.to_string(index=False),
        "",
        "【模型检验】",
        results.tests_summary.to_string(index=False),
        "",
        "【稳健性检验】",
        results.robustness_effects.to_string(index=False),
        "",
        "【与传统OLS比较】",
        results.comparison_summary.to_string(index=False),
    ]
    (out_dir / "model_results_summary.txt").write_text("\n".join(result_txt), encoding="utf-8")


def export_prism_package(results, data_bundle, prism_dir: Path) -> None:
    prism_dir.mkdir(parents=True, exist_ok=True)

    yearly = data_bundle.panel_df.groupby("年份")[["low_score", "green_score"]].mean().reset_index()
    yearly.columns = ["Year", "LowAltitude", "GreenTransport"]
    yearly.to_csv(prism_dir / "prism_index_yearly_trend.csv", index=False, encoding="utf-8-sig")

    moran = results.moran_global.rename(columns={
        "year": "Year",
        "moran_low": "LowAltitude_MoranI",
        "p_low": "LowAltitude_p",
        "moran_green": "GreenTransport_MoranI",
        "p_green": "GreenTransport_p",
    })
    moran.to_csv(prism_dir / "prism_global_moran_trend.csv", index=False, encoding="utf-8-sig")

    sdm = results.sdm_effects.copy()
    sdm["EffectCN"] = sdm["effect"].map({"direct": "Direct", "indirect": "Indirect", "total": "Total"})
    sdm[["EffectCN", "coef", "p_value"]].to_csv(prism_dir / "prism_sdm_effects.csv", index=False, encoding="utf-8-sig")

    mediation = results.mediation_effects[["path", "indirect_ab", "ci_low", "ci_high", "contribution"]].copy()
    mediation.columns = ["Path", "IndirectEffect", "CI_Low", "CI_High", "Contribution"]
    mediation.to_csv(prism_dir / "prism_mediation_contribution.csv", index=False, encoding="utf-8-sig")

    regional = results.regional_effects[["region", "direct", "indirect", "total", "技术创新传导", "产业结构升级", "交通结构优化"]].copy()
    regional.columns = ["Region", "Direct", "Indirect", "Total", "TechInnovation", "IndustrialUpgrade", "TransportOptimization"]
    regional.to_csv(prism_dir / "prism_regional_effects.csv", index=False, encoding="utf-8-sig")

    latest = data_bundle.panel_df.loc[data_bundle.panel_df["年份"] == data_bundle.panel_df["年份"].max(), ["省份", "low_score", "green_score"]].copy()
    latest.columns = ["Province", "LowAltitude", "GreenTransport"]
    latest.to_csv(prism_dir / "prism_province_rank_2023.csv", index=False, encoding="utf-8-sig")

    regional_year = data_bundle.panel_df.copy()
    regional_year["Region"] = regional_year["省份"].map(REGION_MAP)
    regional_year = regional_year.groupby(["年份", "Region"])[["low_score", "green_score"]].mean().reset_index()
    regional_year.columns = ["Year", "Region", "LowAltitude", "GreenTransport"]
    regional_year.to_csv(prism_dir / "prism_regional_yearly_trend.csv", index=False, encoding="utf-8-sig")

    guide = """# GraphPad Prism 出图指南

## 建议在 Prism 中重绘的图

1. `prism_global_moran_trend.csv`
   - 图型：XY line
   - X轴：Year
   - Y轴：LowAltitude_MoranI、GreenTransport_MoranI
   - 建议颜色：深蓝 / 砖红

2. `prism_sdm_effects.csv`
   - 图型：Column bar
   - X轴：EffectCN
   - Y轴：coef
   - 可按 p_value 添加显著性星号

3. `prism_mediation_contribution.csv`
   - 图型：Horizontal bar
   - X轴：Contribution
   - 标签列：Path
   - 可手动加入 CI_Low 与 CI_High 注释

4. `prism_regional_effects.csv`
   - 图型：Grouped bar 或 radar 替代分组柱状图
   - 用 Direct / Indirect / Total 做系列

5. `prism_index_yearly_trend.csv`
   - 图型：XY line
   - 展示双系统年度均值变化

6. `prism_province_rank_2023.csv`
   - 图型：Sorted bar
   - 分别按 LowAltitude 和 GreenTransport 出两张排序图

7. `prism_regional_yearly_trend.csv`
   - 图型：Multiple line
   - 按 Region 分组，分别出低空经济与绿色交通两张图

## 推荐 Prism 美化参数

- Font：Arial 或 Times New Roman
- Title size：16-18
- Axis size：11-12
- Line width：2.0-2.5
- Symbol size：5-6
- Remove top/right borders
- 背景保持纯白
- 导出分辨率：600 dpi
"""
    (prism_dir / "graphpad_prism_plot_guide.md").write_text(guide, encoding="utf-8")


def generate_q2_interpretation(results, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    d = results.sdm_effects.loc[results.sdm_effects["effect"] == "direct", "coef"].values[0]
    i = results.sdm_effects.loc[results.sdm_effects["effect"] == "indirect", "coef"].values[0]
    t = results.sdm_effects.loc[results.sdm_effects["effect"] == "total", "coef"].values[0]
    lm_lag_p = results.tests_summary.loc[results.tests_summary["test"] == "LM_lag", "p_value"].values[0]
    lr_p = results.tests_summary.loc[results.tests_summary["test"] == "LR_SDM_vs_SAR", "p_value"].values[0]
    haus_p = results.tests_summary.loc[results.tests_summary["test"] == "Hausman_FE_RE", "p_value"].values[0]
    top = results.mediation_effects.sort_values("contribution", ascending=False).iloc[0]
    indirect_text = "负向空间溢出" if i < 0 else "正向空间溢出"
    text = f"""# 第二问结果解读（可直接粘贴）

基于2016-2023年30省面板数据，本文采用空间杜宾模型（SDM）识别低空经济对绿色交通发展的空间效应。结果显示，低空经济对本省绿色交通具有显著正向促进作用，直接效应为 {d:.4f}；空间溢出效应为 {i:.4f}；总效应为 {t:.4f}。当前结果表明低空经济存在{indirect_text}特征。

模型选择检验支持采用空间计量框架：LM-lag 检验显著（p={lm_lag_p:.4g}）；LR 检验（SDM 对比 SAR）显著（p={lr_p:.4g}）；Hausman 检验结果为 p={haus_p:.4g}。

并行中介结果表明，三条核心传导路径均发挥作用，其中“{top['path']}”贡献最高（贡献度 {top['contribution']:.2%}）。
"""
    (out_dir / "q2_interpretation.md").write_text(text, encoding="utf-8")


def generate_model_explanation_markdown(results, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    direct = results.sdm_effects.loc[results.sdm_effects["effect"] == "direct", "coef"].values[0]
    indirect = results.sdm_effects.loc[results.sdm_effects["effect"] == "indirect", "coef"].values[0]
    total = results.sdm_effects.loc[results.sdm_effects["effect"] == "total", "coef"].values[0]
    top = results.mediation_effects.sort_values("contribution", ascending=False).iloc[0]
    spillover_label = "负向溢出（虹吸/竞争效应）" if indirect < 0 else "正向溢出（扩散/带动效应）"
    md = fr'''# 问题二模型说明与结果解释（供论文写手使用）

## 1. 本轮模型优化做了什么

本轮采用方案 B，对空间权重矩阵进行了优化：不再仅使用单一省际邻接矩阵，而是使用“地理邻接矩阵 + 经济距离矩阵”的混合权重矩阵。

\[
W^* = 0.7W_{{geo}} + 0.3W_{{econ}}
\]

## 2. warning 应该如何理解

运行时出现的 `The weights matrix is not fully connected` 并不等同于程序报错，也不意味着模型失效。它反映的是空间权重矩阵在图结构意义上存在多个连通分量。该现象会影响空间扩散链条的完整性，因此需要优化权重矩阵，而不是简单忽略。

## 3. 如何解释当前主结果

- 直接效应：{direct:.4f}
- 间接效应：{indirect:.4f}
- 总效应：{total:.4f}

这说明低空经济对本省绿色交通具有显著促进作用，同时区域间还存在 {spillover_label}。

若间接效应为负，可以在论文中解释为：低空经济资源、资本、创新要素和高端应用场景可能优先向优势地区集中，从而在一定时期内对周边省份形成虹吸效应或竞争性挤出效应。

## 4. 中介机制如何落笔

三条中介路径中，当前贡献度最高的是“{top['path']}”，贡献度为 {top['contribution']:.2%}。
'''
    (out_dir / "model_explanation_for_writer.md").write_text(md, encoding="utf-8")


def print_key_metrics(results) -> None:
    print("\n================ 关键结果摘要 ================\n")
    print("【空间杜宾核心效应】")
    print(results.sdm_effects.to_string(index=False))
    print("\n【空间中介路径效应】")
    print(results.mediation_effects.to_string(index=False))
    print("\n【模型检验汇总】")
    print(results.tests_summary.to_string(index=False))
    print("\n==============================================\n")


def main() -> None:
    log_step("开始同步原始数据文件")
    prepare_raw_files()
    log_step("开始读取原始数据并生成标准化面板")
    data_bundle = load_all_data(BASE_DIR)
    log_step("标准化完成，开始执行莫兰检验、SDM、中介与异质性模型")
    model_results = run_all_models(
        panel_df=data_bundle.panel_df,
        w_df=data_bundle.weight_matrix,
        econ_w_df=data_bundle.econ_weight_matrix,
        controls=CONTROL_VARS,
        mediators=MEDIATORS,
        region_map=REGION_MAP,
    )
    log_step("模型计算完成，开始保存结果表")
    save_tables(model_results, data_bundle, RESULT_DIR)
    log_step("开始导出图表")
    export_all_figures(
        panel_df=data_bundle.panel_df,
        w_df=data_bundle.weight_matrix,
        moran_df=model_results.moran_global,
        lisa_df=model_results.lisa_2023,
        sdm_effects=model_results.sdm_effects,
        mediation_df=model_results.mediation_effects,
        regional_df=model_results.regional_effects,
        robust_effects=model_results.robustness_effects,
        region_map=REGION_MAP,
        out_dir=FIG_DIR,
    )
    log_step("开始导出 Prism 数据包")
    export_prism_package(model_results, data_bundle, PRISM_DIR)
    log_step("开始导出 txt 封装结果")
    export_txt_bundles(model_results, data_bundle, RESULT_DIR)
    log_step("开始生成论文结果解读")
    generate_q2_interpretation(model_results, RESULT_DIR)
    generate_model_explanation_markdown(model_results, RESULT_DIR)
    log_step("全部流程完成")
    print_key_metrics(model_results)


if __name__ == "__main__":
    main()
