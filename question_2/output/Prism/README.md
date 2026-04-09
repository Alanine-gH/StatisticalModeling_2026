# GraphPad Prism 出图数据包（Question 2）

本目录为 GraphPad Prism 重绘准备的数据文件，统一放在 `question_2/output/Prism`。

## 文件说明

### 1. `prism_index_yearly_trend.csv`
- 用途：双系统年度均值趋势图
- 建议图型：XY line
- 列说明：
  - `Year`
  - `LowAltitude`
  - `GreenTransport`

### 2. `prism_global_moran_trend.csv`
- 用途：全局莫兰指数趋势图
- 建议图型：XY line
- 列说明：
  - `Year`
  - `LowAltitude_MoranI`
  - `LowAltitude_p`
  - `GreenTransport_MoranI`
  - `GreenTransport_p`

### 3. `prism_sdm_effects.csv`
- 用途：空间杜宾模型直接/间接/总效应柱状图
- 建议图型：Column bar
- 列说明：
  - `Effect`
  - `Coef`
  - `PValue`

### 4. `prism_mediation_contribution.csv`
- 用途：并行中介路径贡献图
- 建议图型：Horizontal bar
- 列说明：
  - `Path`
  - `IndirectEffect`
  - `CI_Low`
  - `CI_High`
  - `Contribution`

### 5. `prism_regional_effects.csv`
- 用途：四大区域效应对比图
- 建议图型：Grouped bar
- 列说明：
  - `Region`
  - `Direct`
  - `Indirect`
  - `Total`
  - `TechInnovation`
  - `IndustrialUpgrade`
  - `TransportOptimization`

### 6. `prism_province_rank_2023.csv`
- 用途：2023年各省排序图
- 建议图型：Sorted bar
- 列说明：
  - `Province`
  - `LowAltitude`
  - `GreenTransport`

### 7. `prism_regional_yearly_trend.csv`
- 用途：区域年度趋势图
- 建议图型：Multiple line
- 列说明：
  - `Year`
  - `Region`
  - `LowAltitude`
  - `GreenTransport`

## Prism 作图建议

- 字体：`Times New Roman` 或 `Arial`
- 标题字号：16–18
- 坐标轴字号：11–12
- 线宽：2.0–2.5
- 点大小：5–6
- 画布背景：白色
- 导出分辨率：600 dpi
