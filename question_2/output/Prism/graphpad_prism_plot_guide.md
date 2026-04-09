# GraphPad Prism 出图指南

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
