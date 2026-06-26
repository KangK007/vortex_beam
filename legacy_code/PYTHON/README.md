# FVVB Python scripts

本文件夹用于保存服务于毕业设计论文的 Python 程序，核心围绕分数阶矢量涡旋光束（FVVB）、OAM/LG 模态分析、大气湍流传播与检测。部分脚本由既有 MATLAB 程序转换而来，但当前组织方式以论文主线和章节结构为准，而不是以原 MATLAB 文件夹名为准。

## 依赖

```bash
cd D:/Project/FVVB/PYTHON
python -m pip install -r requirements.txt
```

主要依赖：`numpy`、`scipy`、`matplotlib`。

## 物理约定

公共计算函数集中在 `fvvb_core.py`，默认单位为 SI。

- 波长：`632.8e-9 m`
- 束腰：`2.0e-3 m`
- 传播距离：`0.2 m`
- Berry 系数：`c_n = exp(i*pi*alpha) sin(pi*alpha) / [pi (alpha-n)]`
- 整数 `alpha` 使用极限处理：`c_n = delta(n, alpha)`，避免 `0/0`。
- FVVB 线偏振分量沿用 MATLAB 约定：
  - `Ex = sum_n c_n E'_n(r,z) sin(n phi)`
  - `Ey = i sum_n c_n E'_n(r,z) cos(n phi)`
- 圆偏振分量：
  - `E_sigma_plus  = (Ex + i Ey) / sqrt(2)`
  - `E_sigma_minus = (Ex - i Ey) / sqrt(2)`
- 叉形/螺旋相位板：`H(phi) = exp(i * fork_ell * phi)`。

## 图序映射

| 论文图 | Python 脚本 | MATLAB 来源 | 说明 |
|---|---|---|---|
| Fig. 1 | 无 | 无对应 `.m` | 光路示意图，非数值程序。 |
| Fig. 2 | `fig02_fvvb_intensity.py` | `MATLAB/图1/simulate_FVVB_intensity_polarization.m` 等 | FVVB 强度分布。 |
| Fig. 3 | `fig03_fvvb_oam_spectrum.py` | `MATLAB/图3图4图6_FVB_FVVB谱图/fvvb_fvb_oam_paper_ready.m` | FVVB 总 OAM 谱。 |
| Fig. 4 | `fig04_fvb_fvvb_oam_compare.py` | 同上 | FVB/FVVB OAM 谱比较。 |
| Fig. 5 | `fig05_oam_width_vs_alpha.py` | `MATLAB/图7_OAM谱-方差图/*.m` | FVB/FVVB OAM 谱宽 vs alpha。 |
| Fig. 6 | `fig06_oam_spp_varying.py` | `MATLAB/图9_OAM谱-叉形相位板/fvvb_oam_fork_total_only.m` | 固定入射 alpha，变 SPP 拓扑荷。 |
| Fig. 7 | `fig07_oam_incident_varying.py` | `MATLAB/图8_OAM谱-叉形相位板/fvvb_oam_fork_total_only2.m` | 固定 SPP，变入射 alpha。 |
| Fig. 8 | `fig08_modal_convergence.py` | 由现有 OAM/radial/sigma 脚本核心函数重建 | 模态分解收敛。 |
| Fig. 9 | `fig09_oam_width_alpha_spp.py` | `MATLAB/图10_FVVB-方差图/*.m` | OAM 谱宽 vs alpha / SPP。 |
| Fig. 10 | `fig10_radial_spectrum_diff.py` | `MATLAB/图11图12_FVVB-径向谱/fvvb_oam_fork_radial_spectrum.m` | 径向谱与差分。 |
| Fig. 11 | `fig11_radial_width_alpha_spp.py` | `MATLAB/图11图12_FVVB-径向谱/*.m` | 径向谱宽 vs alpha / SPP。 |
| Fig. S2 | `supp_fig_s2_lg_waist_sensitivity.py` | 未发现明确独立 `.m`，用公共核心函数重建 | 补充/附录的 LG 基腰敏感性验证图，保留。 |
| 毕设第 4 章 | `demo_turbulence_oam_evolution.py` | 新增 Python 趋势验证 | 单相位屏下的 FVVB 湍流 OAM 谱演化演示。 |
| 毕设主线 B | `run_turbulence_scan.py` | 新增 Python 批量扫描 | 湍流强度、传播距离、分数阶拓扑荷的 Monte Carlo OAM 谱统计。 |

## 运行方式

每个脚本可单独运行，例如：

```bash
python fig03_fvvb_oam_spectrum.py
```

输出默认保存到：

```text
PYTHON/outputs/<script_name>/
```

通常包括：

- `.png` 图像
- `.npz` 数值数组
- `_params.json` 参数记录

## 主线 B：大气湍流传播与 OAM 谱演化

主线 B 的入口脚本包括：

```bash
python demo_turbulence_oam_evolution.py
python run_turbulence_scan.py --quick
```

`demo_turbulence_oam_evolution.py` 用于单相位屏趋势演示；`run_turbulence_scan.py` 用于扫描湍流强度、传播距离和分数阶拓扑荷，并输出 Monte Carlo 均值与标准差。

当前湍流模型采用 modified von Kármán 共同标量相位屏，默认单位为 SI。相位屏同时作用于 `Ex/Ey` 或 `sigma+/sigma-` 两个分量，假设各向同性湍流且不包含偏振相关散射、双折射或真实退偏振。因此相关结果适合用于数值趋势分析；写入论文定量结论前，应结合最终实验传播距离、光束尺寸、SLM/相机采样和湍流加载方式再次校验。

`run_turbulence_scan.py` 的输出保存到：

```text
PYTHON/outputs/run_turbulence_scan/
```

主要包括 OAM 谱、谱宽、谱熵、与无湍流参考谱相似度、`alpha -> ell` 谱扩散矩阵、Stokes 摘要、相位屏统计验证图、弱/中/强湍流代表性强度图，以及对应的 `.npz` 和 `_params.json`。

## 重要说明

1. 本转换保留 MATLAB 的物理公式与归一化约定，不重新定义物理模型。
2. 扫描型脚本计算量较大；脚本顶部均保留可调参数。若要快速测试，可先减小 `grid_n`、`alpha_list`、`lmax` 或 `pmax`。
3. Fig. S2 的原始 MATLAB 源文件未在当前目录中明确找到，因此按补充材料文字和公共核心函数重建。
4. Python 图像样式与 MATLAB/Word 中嵌入图不保证像素级一致，但数值趋势和物理量定义应保持一致。
