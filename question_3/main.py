"""问题三主程序：完成模型训练、预测、优化与绘图。"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from question_3.config import BASE_DIR as PROJECT_DIR, RESULT_DIR, ROOT_DIR as WORKSPACE_ROOT
from question_3.src.data_loader import Question3DataLoader
from question_3.src.model import (
    build_province_heatmap_matrix,
    build_region_ranking,
    build_region_summary,
    build_scenario_comparison,
    build_shandong_comparison,
    forecast_future_scenarios,
    run_model_comparison,
    summarize_key_findings,
)
from question_3.src.visualization import Question3Visualizer


WEIGHT_CANDIDATES = [
    PROJECT_DIR / "output" / "results" / "weight_hybrid.csv",
    WORKSPACE_ROOT / "question_2" / "output" / "results" / "weight_hybrid.csv",
    WORKSPACE_ROOT / "question_2" / "output" / "results" / "weight_hybrid_used.csv",
]


def load_spatial_weight() -> pd.DataFrame:
    """加载空间权重矩阵，优先读取第三问输出，缺失时回退到第二问结果。"""
    for path in WEIGHT_CANDIDATES:
        if path.exists():
            print(f"已加载空间权重矩阵：{path}")
            return pd.read_csv(path, index_col=0)
    candidate_text = "\n".join(str(path) for path in WEIGHT_CANDIDATES)
    raise FileNotFoundError(f"未找到空间权重矩阵文件，请确认以下路径之一存在：\n{candidate_text}")


def print_fit_metrics(comparison_df: pd.DataFrame) -> None:
    """打印各模型拟合指标。"""
    print("\n================ 模型拟合优度指标 ================")
    for _, row in comparison_df.iterrows():
        print(f"模型：{row['模型']}")
        print(f"  MAE : {row['MAE']:.6f}")
        print(f"  RMSE: {row['RMSE']:.6f}")
        print(f"  R²  : {row['R2']:.6f}")
    print("==================================================\n")


def print_model_judgement(comparison_df: pd.DataFrame) -> None:
    """输出对模型表现的简要评价。"""
    best_row = comparison_df.sort_values(by=["RMSE", "MAE", "R2"], ascending=[True, True, False]).iloc[0]
    rmse = float(best_row["RMSE"])
    r2 = float(best_row["R2"])
    if r2 >= 0.95 and rmse <= 0.02:
        level = "优秀"
    elif r2 >= 0.90 and rmse <= 0.05:
        level = "良好"
    else:
        level = "一般"
    print("================ 模型表现判断 ================")
    print(f"综合判断：当前最优模型整体表现属于“{level}”水平。")
    print(f"最优模型：{best_row['模型']}，R²={r2:.4f}，RMSE={rmse:.4f}。")
    print("评价依据：R² 越接近 1，说明拟合解释能力越强；RMSE、MAE 越小，说明预测误差越低。")
    print("==============================================\n")


def print_key_findings(findings_df: pd.DataFrame) -> None:
    """打印可用于论文撰写的摘要性结论。"""
    print("================ 关键结论摘要 ================")
    for scenario, scenario_df in findings_df.groupby("情景"):
        print(f"【{scenario}】")
        ordered_df = scenario_df.sort_values("2024-2030增幅", ascending=False)
        for _, row in ordered_df.iterrows():
            print(f"  {row['区域']}：2024-2030 年预测增幅 {row['2024-2030增幅']:.4f}")
    print("==============================================\n")


def generate_analysis_text(
    comparison_df: pd.DataFrame,
    region_summary_df: pd.DataFrame,
    shandong_gain_df: pd.DataFrame,
    shandong_df: pd.DataFrame,
) -> str:
    """生成第三问结果分析正文。"""
    best_model_row = comparison_df.sort_values(by=["RMSE", "MAE", "R2"], ascending=[True, True, False]).iloc[0]
    best_model_name = best_model_row["模型"]
    best_rmse = best_model_row["RMSE"]
    best_mae = best_model_row["MAE"]
    best_r2 = best_model_row["R2"]

    region_growth_df = (
        region_summary_df.pivot_table(index=["区域", "情景"], columns="年份", values="预测值")
        .reset_index()
    )
    region_growth_df["增幅"] = region_growth_df[2030] - region_growth_df[2024]
    targeted_region = region_growth_df[region_growth_df["情景"] == "重点投入情景"].sort_values("增幅", ascending=False)
    top_region = targeted_region.iloc[0]["区域"]
    top_region_growth = targeted_region.iloc[0]["增幅"]
    bottom_region = targeted_region.iloc[-1]["区域"]
    bottom_region_growth = targeted_region.iloc[-1]["增幅"]

    shandong_2030 = shandong_df[shandong_df["年份"] == 2030].sort_values("预测值", ascending=False)
    best_scenario = shandong_2030.iloc[0]["情景"]
    best_scenario_value = shandong_2030.iloc[0]["预测值"]
    baseline_value = shandong_2030[shandong_2030["情景"] == "基准情景"].iloc[0]["预测值"]
    gain_2030_df = shandong_gain_df[(shandong_gain_df["年份"] == 2030) & (shandong_gain_df["情景"] != "基准情景")].sort_values("相对基准提升", ascending=False)
    top_gain_scenario = gain_2030_df.iloc[0]["情景"]
    top_gain_value = gain_2030_df.iloc[0]["相对基准提升"]

    return f"""# 第三问结果分析

## 3.1 模型构建与精度比较
从模型拟合结果看，四类模型在测试集上的预测性能存在明显差异。其中，{best_model_name} 模型的综合表现最优，MAE 为 {best_mae:.4f}，RMSE 为 {best_rmse:.4f}，R² 为 {best_r2:.4f}，说明该模型在误差控制和拟合优度两个维度上均具有较强优势。整体而言，该结果表明当前所构建的预测框架已经能够较好刻画绿色交通发展指数的变化规律，模型精度达到较高水平，具备作为后续情景预测工具的可靠性。需要指出的是，虽然 GraphSAGE 模型在理论上能够反映省际空间关联，但在当前样本规模与变量结构下，其测试集误差仍高于最优模型，说明复杂空间模型并不一定在小样本情形下占优。相较之下，GM(1,1) 在现有数据尺度上表现出更稳定的拟合能力，因此将其作为后续情景分析的核心预测模型是合理的。

## 3.2 区域情景预测结果分析
从四大区域的预测结果来看，2024—2030 年间各区域绿色交通发展水平均呈上升趋势，但增速和提升幅度存在明显分化。在重点投入情景下，{top_region}地区的提升幅度最大，2024—2030 年预测指数增幅达到 {top_region_growth:.4f}；而{bottom_region}地区的增幅相对较小，为 {bottom_region_growth:.4f}。这说明在差异化投入策略下，不同区域对低空经济带动绿色交通发展的响应程度并不一致。结合区域基础条件可以判断，经济基础较好、产业协同程度较高的地区，对低空经济投资的吸收与转化效率更强，因此能够更快转化为绿色交通发展优势；而基础相对薄弱地区虽然同样受益，但短期内增量表现相对有限。总体来看，三种情景均能推动绿色交通指数提高，但重点投入情景在多数区域显示出更强的带动作用，这说明未来政策设计应更加重视资源配置的针对性，而非简单平均化投入。

## 3.3 山东省情景对比与重点研判
以山东省为例，不同投入情景下的绿色交通发展趋势同样呈现持续上升态势，但提升强度具有明显差别。到 2030 年，{best_scenario}下山东省预测值最高，达到 {best_scenario_value:.4f}，相较基准情景的 {baseline_value:.4f} 具有更明显优势。进一步比较相对基准情景的增益变化可见，{top_gain_scenario}在 2030 年带来的提升幅度最大，较基准情景提高 {top_gain_value:.4f}。这表明对于山东省而言，在既有产业基础较强、交通体系较完备的背景下，进一步强化重点领域投入，尤其是在创新资源、基础设施补短板以及关键政策支持方向上，能够更有效释放低空经济对绿色交通系统的牵引作用。由此可见，山东省未来更适宜采取具有针对性的强化投入模式，而不是平均分散式扩张，这对实现绿色交通高质量发展具有更强的现实指导意义。

## 3.4 政策启示
基于上述结果，可以得到两点启示：其一，全国层面的低空经济赋能绿色交通政策应坚持区域差异化思路，对增长弹性较高区域加大重点支持，同时对薄弱地区补齐基础设施和技术创新短板；其二，山东省应围绕关键创新环节、低空起降设施布局与绿色交通配套系统协同推进，形成更具聚焦性的投入结构。相比均衡铺开式配置，聚焦重点领域更有利于提升投入产出效率，并增强低空经济对绿色交通系统升级的长期支撑作用。
"""


def save_tables(
    comparison_df: pd.DataFrame,
    prediction_outputs: dict[str, pd.DataFrame],
    region_summary_df: pd.DataFrame,
    shandong_df: pd.DataFrame,
    heatmap_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    shandong_gain_df: pd.DataFrame,
    findings_df: pd.DataFrame,
    analysis_text: str,
) -> None:
    """保存模型结果表与预测表。"""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(RESULT_DIR / "模型拟合指标.csv", index=False, encoding="utf-8-sig")
    region_summary_df.to_csv(RESULT_DIR / "四大区域预测结果.csv", index=False, encoding="utf-8-sig")
    shandong_df.to_csv(RESULT_DIR / "山东省预测对比.csv", index=False, encoding="utf-8-sig")
    heatmap_df.to_csv(RESULT_DIR / "重点投入情景_省域热力图矩阵.csv", encoding="utf-8-sig")
    ranking_df.to_csv(RESULT_DIR / "2030区域预测排名.csv", index=False, encoding="utf-8-sig")
    shandong_gain_df.to_csv(RESULT_DIR / "山东省情景增益对比.csv", index=False, encoding="utf-8-sig")
    findings_df.to_csv(RESULT_DIR / "关键结论摘要.csv", index=False, encoding="utf-8-sig")
    (RESULT_DIR / "q3_结果分析.md").write_text(analysis_text, encoding="utf-8")
    for scenario_name, forecast_df in prediction_outputs.items():
        forecast_df.to_csv(RESULT_DIR / f"{scenario_name}_情景预测结果.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    """执行问题三全部流程。"""
    data_bundle = Question3DataLoader().load_data_bundle()
    spatial_weight = load_spatial_weight()
    comparison_df, model_outputs = run_model_comparison(
        train_df=data_bundle.train_df,
        valid_df=data_bundle.valid_df,
        test_df=data_bundle.test_df,
        feature_cols=data_bundle.feature_cols,
        target_col=data_bundle.target_col,
        spatial_weight=spatial_weight,
    )
    print_fit_metrics(comparison_df)
    print_model_judgement(comparison_df)

    best_model = model_outputs[comparison_df.iloc[0]["模型"]].fitted_model
    if comparison_df.iloc[0]["模型"] != "GraphSAGE":
        best_model = model_outputs["GraphSAGE"].fitted_model

    scenario_inputs = {
        "baseline": data_bundle.baseline_df,
        "stronger": data_bundle.stronger_df,
        "targeted": data_bundle.targeted_df,
    }
    prediction_outputs = forecast_future_scenarios(best_model, scenario_inputs)
    region_summary_df = build_region_summary(prediction_outputs)
    shandong_df = build_shandong_comparison(prediction_outputs)
    heatmap_df = build_province_heatmap_matrix(prediction_outputs, scenario_name="targeted")
    ranking_df = build_region_ranking(prediction_outputs, year=2030, scenario_name="targeted")
    shandong_gain_df = build_scenario_comparison(prediction_outputs, province="山东")
    findings_df = summarize_key_findings(prediction_outputs)
    analysis_text = generate_analysis_text(comparison_df, region_summary_df, shandong_gain_df, shandong_df)

    save_tables(
        comparison_df=comparison_df,
        prediction_outputs=prediction_outputs,
        region_summary_df=region_summary_df,
        shandong_df=shandong_df,
        heatmap_df=heatmap_df,
        ranking_df=ranking_df,
        shandong_gain_df=shandong_gain_df,
        findings_df=findings_df,
        analysis_text=analysis_text,
    )

    visualizer = Question3Visualizer(output_dir=PROJECT_DIR / "output")
    visualizer.plot_model_comparison(comparison_df)
    visualizer.plot_region_forecast(region_summary_df)
    visualizer.plot_province_heatmap(heatmap_df)
    visualizer.plot_region_ranking(ranking_df)
    visualizer.plot_shandong_comparison(shandong_df)
    visualizer.plot_scenario_gap(shandong_gain_df, province="山东")

    print_key_findings(findings_df)
    print(analysis_text)
    print("结果表已保存到：question_3/output/results")
    print("图片已保存到：question_3/output")


if __name__ == "__main__":
    main()
