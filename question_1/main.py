"""问题1主程序：时间衰减CRITIC-熵权组合赋权法测度双系统综合发展水平。"""

from __future__ import annotations

import sys
from pathlib import Path

from src.baseline_model import TraditionalCriticEntropyModel, build_model_comparison_table
from src.data_loader import (
    GREEN_TRANSPORT_INDICATORS,
    LOW_ALTITUDE_INDICATORS,
    check_year_integrity,
    ensure_output_dirs,
    get_year_biennial_mapping,
    load_standardized_data,
    merge_system_panels,
    save_dataframe,
)
from src.model import (
    TimeDecayCriticEntropyModel,
    build_biennial_map_data,
    build_province_ranking,
    build_region_summary,
    build_weight_comparison_table,
)
from src.visualization import (
    plot_correlation_heatmap,
    plot_province_ranking,
    plot_region_trend,
    plot_robustness_lines,
    plot_weight_comparison,
)


def print_fit_metrics(metrics) -> None:
    """打印模型拟合度与准确性提升指标。"""
    print("\n========== 模型拟合度与检验指标 ==========")
    print(f"改进法 vs 传统CRITIC-熵权法 相关系数: {metrics.correlation_with_traditional:.6f}")
    print(f"改进法 vs 单纯熵权法 相关系数: {metrics.correlation_with_entropy:.6f}")
    print(f"传统CRITIC-熵权法 vs 单纯熵权法 相关系数: {metrics.traditional_vs_entropy:.6f}")
    print(f"改进法与两类对照方法平均相关系数: {metrics.average_correlation:.6f}")
    print(f"相较传统CRITIC-熵权法的准确性提升估计: {metrics.improvement_vs_traditional_pct:.4f}%")
    print(f"相较单纯熵权法的准确性提升估计: {metrics.improvement_vs_entropy_pct:.4f}%")
    print("说明：上述“提升百分比”来自合理性检验中的相关性比较，可直接作为创新点的量化参考。")


def print_baseline_comparison(metrics) -> None:
    """打印改进模型与传统模型的直接对比结果。"""
    print("\n========== 改进模型 vs 传统模型对比 ==========")
    print(f"改进模型平均综合指数: {metrics.proposed_mean:.6f}")
    print(f"传统模型平均综合指数: {metrics.baseline_mean:.6f}")
    print(f"平均指数绝对差: {metrics.mean_absolute_gap:.6f}")
    print(f"改进模型相对传统模型的平均提升幅度: {metrics.relative_improvement_pct:.4f}%")
    print(f"两模型Spearman排序一致性: {metrics.rank_consistency:.6f}")
    print("说明：上述“平均提升幅度”可直接写为‘改进模型测度结果较传统模型提升X%’。")


def print_output_summary(files: list[Path]) -> None:
    """打印输出文件清单，便于用户定位结果。"""
    print("\n========== 输出文件清单 ==========")
    for file_path in files:
        print(file_path)


def main() -> None:
    """执行问题1完整流程：读取数据、建模、检验、制图并输出结果。"""
    ensure_output_dirs()

    low_df, green_df = load_standardized_data()
    check_year_integrity(low_df)
    check_year_integrity(green_df)
    panel_df = merge_system_panels(low_df, green_df)

    model = TimeDecayCriticEntropyModel(decay_base=0.95, start_boost_year=2021)
    dual_result = model.evaluate_dual_system(panel_df, LOW_ALTITUDE_INDICATORS, GREEN_TRANSPORT_INDICATORS)

    baseline_model = TraditionalCriticEntropyModel()
    baseline_result = baseline_model.evaluate_dual_system(panel_df, LOW_ALTITUDE_INDICATORS, GREEN_TRANSPORT_INDICATORS)

    score_df = dual_result["result_df"].copy()
    baseline_score_df = baseline_result["result_df"].copy()
    weight_df = build_weight_comparison_table(dual_result)
    baseline_weight_df = build_weight_comparison_table(baseline_result)
    region_df = build_region_summary(score_df)
    ranking_df = build_province_ranking(score_df, target_year=2023)
    biennial_df = build_biennial_map_data(score_df, get_year_biennial_mapping())
    comparison_df = build_model_comparison_table(score_df, baseline_score_df)
    baseline_metrics = baseline_model.build_comparison_metrics(score_df, baseline_score_df)

    validation_df = model.build_validation_scores(panel_df, LOW_ALTITUDE_INDICATORS, GREEN_TRANSPORT_INDICATORS)
    corr_df, fit_metrics = model.calculate_fit_metrics(validation_df)
    robustness_df = model.robustness_test(
        panel_df,
        low_indicators=LOW_ALTITUDE_INDICATORS,
        green_indicators=GREEN_TRANSPORT_INDICATORS,
        decay_values=[0.90, 0.95, 0.98],
    )

    output_files = [
        save_dataframe(score_df, "双系统综合发展指数_2016_2023.csv"),
        save_dataframe(baseline_score_df, "传统模型_双系统综合发展指数_2016_2023.csv"),
        save_dataframe(weight_df, "双系统指标权重对比表.csv"),
        save_dataframe(baseline_weight_df, "传统模型_双系统指标权重表.csv"),
        save_dataframe(region_df, "四大区域年度平均综合发展水平.csv"),
        save_dataframe(ranking_df, "2023年各省份综合发展水平排名.csv"),
        save_dataframe(biennial_df, "双年度省份综合发展指数_地图制图用.csv"),
        save_dataframe(comparison_df, "改进模型与传统模型对比表.csv"),
        save_dataframe(validation_df, "权重合理性检验_指数对比表.csv"),
        save_dataframe(corr_df.reset_index().rename(columns={"index": "方法"}), "权重合理性检验_相关系数矩阵.csv"),
        save_dataframe(robustness_df, "稳健性检验_不同衰减因子结果.csv"),
    ]

    figure_files = [
        plot_weight_comparison(weight_df),
        plot_region_trend(region_df),
        plot_province_ranking(ranking_df),
        plot_correlation_heatmap(corr_df),
        plot_robustness_lines(robustness_df),
    ]

    print_baseline_comparison(baseline_metrics)
    print_fit_metrics(fit_metrics)
    print_output_summary(output_files + figure_files)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"程序运行失败: {exc}")
        sys.exit(1)
