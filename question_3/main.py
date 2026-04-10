"""问题三主程序：完成模型训练、预测、优化与绘图。"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from question_3.config import BASE_DIR as PROJECT_DIR, RESULT_DIR
from question_3.src.data_loader import Question3DataLoader
from question_3.src.model import (
    build_region_summary,
    build_shandong_comparison,
    forecast_future_scenarios,
    run_model_comparison,
)
from question_3.src.visualization import Question3Visualizer


def print_fit_metrics(comparison_df: pd.DataFrame) -> None:
    """打印各模型拟合指标。"""
    print("\n================ 模型拟合优度指标 ================")
    for _, row in comparison_df.iterrows():
        print(f"模型：{row['模型']}")
        print(f"  MAE : {row['MAE']:.6f}")
        print(f"  RMSE: {row['RMSE']:.6f}")
        print(f"  R²  : {row['R2']:.6f}")
    print("==================================================\n")


def save_tables(
    comparison_df: pd.DataFrame,
    prediction_outputs: dict[str, pd.DataFrame],
    region_summary_df: pd.DataFrame,
    shandong_df: pd.DataFrame,
) -> None:
    """保存模型结果表与预测表。"""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(RESULT_DIR / "模型拟合指标.csv", index=False, encoding="utf-8-sig")
    region_summary_df.to_csv(RESULT_DIR / "四大区域预测结果.csv", index=False, encoding="utf-8-sig")
    shandong_df.to_csv(RESULT_DIR / "山东省预测对比.csv", index=False, encoding="utf-8-sig")
    for scenario_name, forecast_df in prediction_outputs.items():
        forecast_df.to_csv(RESULT_DIR / f"{scenario_name}_情景预测结果.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    """执行问题三全部流程。"""
    data_bundle = Question3DataLoader().load_data_bundle()
    comparison_df, model_outputs = run_model_comparison(
        train_df=data_bundle.train_df,
        valid_df=data_bundle.valid_df,
        test_df=data_bundle.test_df,
        feature_cols=data_bundle.feature_cols,
        target_col=data_bundle.target_col,
        spatial_weight=pd.read_csv(PROJECT_DIR / "output" / "results" / "weight_hybrid.csv", index_col=0),
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
    save_tables(comparison_df, prediction_outputs, region_summary_df, shandong_df)

    visualizer = Question3Visualizer(output_dir=PROJECT_DIR / "output")
    visualizer.plot_model_comparison(comparison_df)
    visualizer.plot_region_forecast(region_summary_df)
    visualizer.plot_shandong_comparison(shandong_df)

    print("结果表已保存到：question_3/output/results")
    print("图片已保存到：question_3/output")


if __name__ == "__main__":
    main()
