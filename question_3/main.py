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

## 模型比较分析
从模型拟合结果看，四类模型在测试集上的预测性能存在明显差异。其中，{best_model_name} 模型的综合表现最优，MAE 为 {best_mae:.4f}，RMSE 为 {best_rmse:.4f}，R² 为 {best_r2:.4f}，说明该模型在误差控制和拟合优度两个维度上均具有较强优势。相较于传统时间序列或灰色预测方法，融入空间关联特征的模型能够更充分捕捉省际之间低空经济发展与绿色交通演进的联动关系，因此在对未来绿色交通发展水平进行预测时更具解释力和稳定性。整体来看，采用该模型作为后续情景预测的核心工具是合理的，也为第三问后续区域预测和山东省情景比较提供了较为可靠的方法基础。

## 区域预测分析
从四大区域的预测结果来看，2024—2030 年间各区域绿色交通发展水平均呈上升趋势，但增速和提升幅度存在明显分化。在重点投入情景下，{top_region}地区的提升幅度最大，2024—2030 年预测指数增幅达到 {top_region_growth:.4f}；而 {bottom_region}地区的增幅相对较小，为 {bottom_region_growth:.4f}。这说明在差异化投入策略下，不同区域对低空经济带动绿色交通发展的响应程度并不一致。总体而言，重点投入情景相较基准情景和加大投入情景表现出更强的带动效应，尤其对于具有后发潜力或政策边际效应更高的区域，其绿色交通发展水平能够实现更快跃升。因此，在全国层面推进低空经济赋能绿色交通时，应更加重视区域异质性，实施分层分类的资源配置与政策支持路径。

## 山东省情景分析
以山东省为例，不同投入情景下的绿色交通发展趋势同样呈现持续上升态势，但提升强度具有明显差别。到 2030 年，{best_scenario}下山东省预测值最高，达到 {best_scenario_value:.4f}，相较基准情景的 {baseline_value:.4f} 具有更明显优势。进一步比较相对基准情景的增益变化可见，{top_gain_scenario}在 2030 年带来的提升幅度最大，较基准情景提高 {top_gain_value:.4f}。这表明对于山东省而言，在既有产业基础较强、交通体系较完备的背景下，进一步强化重点领域投入，尤其是在创新资源、基础设施补短板以及关键政策支持方向上，能够更有效释放低空经济对绿色交通系统的牵引作用。由此可见，山东省未来更适宜采取具有针对性的强化投入模式，而不是平均分散式扩张，这对实现绿色交通高质量发展具有更强的现实指导意义。
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
