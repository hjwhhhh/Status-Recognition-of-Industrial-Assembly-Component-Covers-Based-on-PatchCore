# 基于计算机视觉的工业装配件卸盖/缺失检测

本项目用于检测工业装配线上滤芯/容器盖体的装配状态。项目针对两段现场视频 `videoA.mp4` 和 `videoB.mp4`，分别实现正面端面卸盖检测和多盖位缺失/开盖检测，并生成带检测框、OK/NG 状态、实时参数曲线的演示视频。

## 1. 项目目标

工业装配视频中存在透明材料反光、玻璃重影、工件移动、局部遮挡、侧面小盖尺寸小等问题。直接使用简单阈值或固定框容易出现误检、漏检和框抖动。本项目采用传统计算机视觉与轻量正常样本记忆库结合的方法，在没有逐帧完整标注数据的情况下完成检测。

主要目标包括：

- 对 `videoA` 中的正面圆形端面判断已卸盖/未卸盖。
- 对 `videoB` 中右侧真实工件的上主盖、下主盖、上侧盖、下侧盖进行检测。
- 排除 `videoB` 左侧玻璃反光区域的干扰。
- 稳定侧面小盖检测框，尤其修正视频末尾最右侧下侧盖位置。
- 输出最终演示视频和实验报告可用结果图。

## 2. 当前效果

当前已生成最终演示视频：

- `outputs/demo_videoA`
- `outputs/demo_videoB`

演示视频为左右拼接格式：

- 左侧：原始视频帧。
- 右侧：检测结果、OK/NG 标注、检测框、实时参数面板。

当前也已生成实验报告用结果图，位于：

- `outputs/report_figures/`

其中包含 `videoA`、`videoB` 的关键帧对比图、检测结果图和参数面板裁剪图。

## 3. 文件结构

```text
.
├── run.py                         # 命令行入口：训练、推理、演示视频生成
├── requirements.txt               # Python 依赖
├── videoA.mp4                     # 输入视频 A
├── videoB.mp4                     # 输入视频 B
├── image1.png                     # videoA 人工参考标注图
├── image2.png                     # videoB 人工参考标注图
├── src/
│   ├── __init__.py
│   ├── detectors.py               # videoA/videoB 核心检测器
│   └── memory.py                  # PatchCore-style 正常样本记忆库
├── models/
│   ├── videoA_detector.pkl        # videoA 训练后的检测器
│   └── videoB_detector.pkl        # videoB 训练后的检测器
└── outputs/
    ├── demo_videoA_370s_25s.mp4   # videoA 最终演示视频
    ├── demo_videoB_980s_25s.mp4   # videoB 最终演示视频
    └── report_figures/            # 实验报告结果图
```

## 4. 环境配置

建议使用 Python 3.10 或更高版本。

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

依赖内容：

```text
opencv-python
numpy
scikit-learn
```

## 5. 快速运行

重新训练两个检测器并生成默认演示视频：

```powershell
python run.py all
```

默认输出：

```text
outputs/demo_videoA_370s_25s.mp4
outputs/demo_videoB_980s_25s.mp4
```

## 6. 常用命令

只训练 `videoA`：

```powershell
python run.py train --video A
```

只训练 `videoB`：

```powershell
python run.py train --video B
```

训练两个视频：

```powershell
python run.py train --video all
```

生成 `videoA` 演示视频：

```powershell
python run.py demo --video A --start 370 --duration 25 --width-per-view 960
```

生成 `videoB` 演示视频：

```powershell
python run.py demo --video B --start 980 --duration 25 --width-per-view 960
```

生成两个演示视频：

```powershell
python run.py demo --video all --duration 25 --width-per-view 960
```

导出低频推理 CSV：

```powershell
python run.py infer --video A --start 370 --duration 40 --stride-sec 1
python run.py infer --video B --start 980 --duration 40 --stride-sec 1
```

## 7. 方法概述

本项目没有使用 YOLO 等监督式目标检测模型，而是采用更适合小样本实验场景的方案：

1. 利用人工参考图确定目标语义、ROI 和几何偏移。
2. 使用正常视频片段进行自校准，自动估计阈值。
3. 结合 HSV 颜色、纹理、边缘、连通域和 Hough 圆进行候选定位。
4. 使用轻量 PatchCore-style 正常样本记忆库辅助异常判断。
5. 使用轨迹匹配、位置平滑、短时保持和去重降低视频检测抖动。
6. 在演示视频中叠加实时参数曲线，增强结果可解释性。

## 8. videoA 检测逻辑

`videoA` 检测正面圆形端面，参考图为 `image1.png`。

核心思路：

- 在固定 ROI 内提取圆形端面候选。
- 通过 HSV 颜色分割和 Hough 圆检测生成候选。
- 提取黄色比例、半透明盖面比例、外环完整度、饱和度、边缘和纹理特征。
- 黄色滤芯明显、纹理更强时判为已卸盖 OK。
- 盖面光滑、纹理弱且呈半透明盖面特征时判为未卸盖 NG。

主要参数：

- ROI：`(520, 300, 1910, 760)`
- 最大端面数量：`4`
- `yellow_min=0.49398`
- `ok_sat_min=82.0`
- `ng_sat_max=112.0`
- `smooth_edge_max=22.0`
- `smooth_lap_max=55.0`

训练信息：

- 正常训练片段：`0~360s`
- 采样间隔：`1s`
- 正常端面样本数：`893`
- 检测器版本：`VideoADetector version 9`

## 9. videoB 检测逻辑

`videoB` 检测右侧真实工件区域，参考图为 `image2.png`。

每个可见容器最多检测四个盖位：

- 上主盖
- 下主盖
- 上侧盖
- 下侧盖

关键区域参数：

- 右侧真实区域起点：`right_x0=900`
- 最小有效容器中心：`min_center_x=1025`
- 容器间距先验：`nominal_pitch=305.0`

主盖检测：

- 使用蓝色 HSV 区域检测上、下主盖。
- 使用正常样本记忆库辅助判断。
- 当前阈值：
  - `top_blue_min=1477.215`
  - `bottom_blue_min=2407.208`

侧盖检测：

- 侧盖位置根据人工红框标注进行几何锁定。
- 普通工位上侧盖：`cx - 95, y=415`
- 普通工位下侧盖：`cx - 76, y=870`
- 最右侧实体工位下侧盖：
  - 条件：`cx >= 1760.0`
  - 位置：`cx - 116, y=888`
  - 框大小：`88x88`

侧盖状态判断：

- 提取灰色区域面积、边缘强度、亮度变化和灰度比例。
- 计算 `side_presence` 作为侧盖存在度。
- 当前阈值：`side_present_min=0.5322`
- `side_presence >= threshold` 判为 `CAPPED/OK`
- `side_presence < threshold` 判为 `OPEN/NG`
- 对下侧侧盖额外加入已卸盖负样本规则：当 `edge_mean >= 88.0` 且 `value_std >= 43.0` 时，优先判为 `REMOVED/NG`。该规则只对非常强的裸露接头、螺纹和强纹理区域触发，避免把正常未卸盖接头误判为已卸盖。

训练信息：

- 正常训练片段：`0~180s`
- 采样间隔：`1s`
- 正常样本数：
  - top：`432`
  - bottom：`432`
  - side：`866`
- 检测器版本：`VideoBDetector version 21`
- 模型签名：`videoB_removed_side_ng_v21`

## 10. 参数面板说明

演示视频右下方或左下方会显示实时参数面板。

`videoA` 参数包括：

- `status`：当前帧整体状态。
- `faces`：检测到的端面数量。
- `max_score`：当前最大分数。
- `yellow`：端面黄色比例。
- `yellow_min`：黄色比例阈值。
- `sat`：饱和度。
- `ok_sat_min`：OK 饱和度阈值。
- `texture`：纹理强度。

`videoB` 参数包括：

- `status`：当前帧整体状态。
- `sites`：有效盖位数量。
- `NG`：异常盖位数量。
- `main_blue`：主盖蓝色面积均值。
- `top_min` / `bottom_min`：上下主盖蓝色面积阈值。
- `side_presence`：侧盖存在度均值。
- `threshold`：侧盖存在度阈值。
- `capped/removed`：未卸盖/已卸盖侧盖数量。

参数曲线用于说明检测依据，方便实验报告分析状态变化。

## 11. 报告用结果图

结果图位于：

```text
outputs/report_figures/
```

推荐在实验报告中使用以下图片：

| 图片 | 内容 |
|---|---|
| `fig01_videoA_start_3700_side_by_side.png` | videoA 370.0s 原图与检测结果对比 |
| `fig01_videoA_start_3700_annotated.png` | videoA 370.0s 检测结果局部 |
| `fig02_videoA_mid_3820_side_by_side.png` | videoA 382.0s 原图与检测结果对比 |
| `fig02_videoA_mid_3820_annotated.png` | videoA 382.0s 检测结果局部 |
| `fig03_videoB_side_three_9832_side_by_side.png` | videoB 983.2s 三工件场景检测结果 |
| `fig03_videoB_side_three_9832_annotated.png` | videoB 983.2s 多盖位检测结果 |
| `fig03_videoB_side_three_9832_parameter_panel.png` | videoB 参数面板裁剪图 |
| `fig03a_videoB_removed_9816_side_by_side.png` | videoB 981.6s 已卸盖负样本识别结果 |
| `fig03a_videoB_removed_9816_annotated.png` | videoB 981.6s 已卸盖侧盖 REMOVED/NG 局部结果 |
| `fig03a_videoB_removed_9816_parameter_panel.png` | videoB 981.6s 负样本参数面板裁剪图 |
| `fig03b_videoB_correct_9896_side_by_side.png` | videoB 989.6s 正确未卸盖样本恢复为 CAPPED/OK |
| `fig03b_videoB_correct_9896_annotated.png` | videoB 989.6s 正确样本检测结果局部 |
| `fig04a_videoB_correct_9970_side_by_side.png` | videoB 997.0s 正确未卸盖样本恢复为 CAPPED/OK |
| `fig04a_videoB_correct_9970_annotated.png` | videoB 997.0s 正确样本检测结果局部 |
| `fig04_videoB_side_two_9976_side_by_side.png` | videoB 997.6s 两工件场景检测结果 |
| `fig04_videoB_side_two_9976_annotated.png` | videoB 997.6s 检测结果局部 |
| `fig05_videoB_end_10033_side_by_side.png` | videoB 1003.3s 末尾帧检测结果 |
| `fig05_videoB_end_10033_annotated.png` | videoB 1003.3s 最右侧下侧盖修正结果 |
| `fig05_videoB_end_10033_parameter_panel.png` | videoB 末尾帧参数面板裁剪图 |

原始参考图：

- `image1.png`：videoA 正面端面人工标注参考图。
- `image2.png`：videoB 主盖和侧面小盖人工标注参考图。

## 12. 已知局限

当前方案依赖固定相机视角和固定产线结构。如果相机位置、光照条件、容器型号或工位布局明显变化，需要重新校准 ROI、颜色阈值和几何偏移。由于没有完整逐帧真值标注，当前评价主要依赖人工参考图、关键帧可视化和最终演示视频检查。

## 13. 后续改进方向

可以考虑以下方向继续优化：

- 补充逐帧标注数据，训练监督式检测模型。
- 引入深度特征作为正常样本记忆库输入。
- 增加自动工位标定，降低对固定几何参数的依赖。
- 将检测结果导出为结构化日志，便于量化评估。
- 增加更多异常样本，完善 OK/NG 阈值选择。
