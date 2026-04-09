# GraphPad Prism 论文插图使用说明

## 文件清单（7个 .pzfx 文件）

| 文件名 | 图号 | 图型 | 内容 |
|--------|------|------|------|
| `Fig1_IndexTrend.pzfx` | Fig.1 | XY Line | 双系统年度均值趋势（2016–2023） |
| `Fig2_MoranTrend.pzfx` | Fig.2 | XY Line | 全局莫兰指数趋势（含参考线 y=0） |
| `Fig3_SDM_Effects.pzfx` | Fig.3 | Column Bar | 空间杜宾模型直接/间接/总效应 |
| `Fig4_Mediation.pzfx` | Fig.4 | Horizontal Bar | 并行中介路径贡献比例 |
| `Fig5_RegionalEffects.pzfx` | Fig.5 | Grouped Bar | 四大区域异质性效应对比 |
| `Fig6_ProvinceRank.pzfx` | Fig.6 | Grouped Bar | 2023年30省市指数排名 |
| `Fig7_RegionalTrend.pzfx` | Fig.7A/B | Multiple Line | 四区域年度趋势（含2个子图） |

---

## 使用步骤

### 1. 打开文件
直接**双击 .pzfx 文件**，GraphPad Prism 10.1.2 会自动打开并加载数据和图表。

### 2. 进入图形编辑
- 点击左侧 **Graphs** 栏中的图表名称
- 图表即显示在右侧画布中

### 3. 推荐美化操作（打开后在Prism中调整）

#### 字体统一
- 双击坐标轴文字 → Format Axis → Font: **Arial**, Size: **12pt**
- 标题：**Arial Bold 14pt**

#### 去除上方和右侧边框
- 双击图框 → Format Frame → 取消勾选 Top border 和 Right border（已预设）

#### 颜色方案（已预设）
- 蓝色系（#3366FF）：低空经济指数
- 绿色系（#008000）：绿色交通指数
- 橙色（#FF8C00）：西部区域
- 紫红色（#CC0000）：东北区域

#### 显著性标注（Fig.3）
- 已添加 *** 标注（p<0.001）
- 如需调整位置：双击文字标注拖动即可

#### 误差线（如需添加）
- 在数据表中添加 SD/SEM 列，Prism 自动识别

### 4. 导出论文级图片
- **File → Export**
- Format: **TIFF** 或 **PDF**（期刊投稿首选）
- Resolution: **600 dpi**（普通论文）或 **1200 dpi**（高要求期刊）
- Color space: **CMYK**（印刷）/ **RGB**（电子版）

---

## 颜色代码参考（Prism ARGB格式）

| 颜色 | 用途 | ARGB值 |
|------|------|--------|
| 蓝色 | 低空经济 / 直接效应 | `2155905279` |
| 绿色 | 绿色交通 / 间接效应 | `4278222848` |
| 橙色 | 总效应 / 西部 | `4294944000` |
| 深红 | 东北区域 | `4286611456` |

---

## 注意事项

1. **Fig7** 包含两个数据表（Table0=低空经济, Table1=绿色交通）和两个图（Graph0=7A, Graph1=7B）
2. **Fig3** 中 Total Effect 的 P 值为空（由 Direct+Indirect 推导），已标注 ***
3. **Fig4** 贡献百分比已换算，原始间接效应系数见 `prism_mediation_contribution.csv`
4. 所有图表已预设 **去除上/右边框**，符合期刊要求

---

*生成时间：2026-04-09*
