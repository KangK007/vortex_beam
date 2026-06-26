# vortex_beam_thesis

## 项目简介 / Project Overview

本项目是面向毕业论文整理的分数阶矢量涡旋光束（Fractional Vector Vortex Beam, FVVB）仿真代码库，用于光束生成、OAM/LG 模态分析、自由空间传播以及大气湍流传播研究。

This is a thesis-oriented Python project for fractional vector vortex beam (FVVB) simulation, including beam generation, OAM/LG modal analysis, free-space propagation, and turbulence propagation studies.

本项目只重组代码结构和运行入口，不改变原有物理模型：

This repository reorganizes the code structure and execution workflow without changing the original physical model:

- 分数阶拓扑荷的 Berry 展开。 / Berry expansion for fractional topological charge.
- FVVB 的 Jones 场表达约定。 / FVVB Jones-field convention.
- 同一平面上的 SPP/叉形相位调制。 / Same-plane SPP/fork phase modulation.
- 角谱法传播。 / Angular-spectrum propagation.
- 湍流采用共同标量 modified von Karman 相位屏近似。 / Common scalar modified von Karman phase-screen approximation for turbulence.

## 项目结构 / Project Layout

```text
vortex_beam_thesis/
|-- configs/                         # YAML 参数文件 / YAML configuration files
|-- src/                             # 核心函数库 / reusable source modules
|-- scripts/
|   |-- field_generation/            # 单束光场生成与基础诊断 / single-beam simulation
|   |-- modal_analysis/              # OAM/LG 模态分析预留入口 / reserved modal-analysis workflows
|   |-- propagation_studies/         # 传播与湍流扫描 / propagation and turbulence scans
|   `-- reproduction/                # 复现实验结果 / reproduction workflows
|-- notebooks/                       # 结果检查 Notebook / lightweight inspection notebooks
|-- results/                         # 自动生成结果 / generated outputs
|-- thesis_figures/                  # 最终论文图 / manually selected thesis figures
|-- legacy_code/PYTHON/
|   |-- core/                        # 旧版核心模块 / legacy core modules
|   |-- chapter_03_free_space_and_spp/
|   `-- chapter_04_turbulence/
`-- tests/                           # 测试 / tests
```

## 安装依赖 / Installation

```bash
cd D:/Project/毕设/github_release/vortex_beam_thesis
python -m pip install -r requirements.txt
```

如果路径中的中文在终端中显示异常，可以先切换到仓库实际所在目录后再运行命令。

If Chinese characters in the path are not displayed correctly in your terminal, first change into the actual repository directory and then run the commands.

## 主要工作流 / Main Workflows

### 1. 单束 FVVB 仿真 / Single FVVB Beam Simulation

用于检查一组光束参数、SPP 参数、传播参数和湍流参数下的光场、强度、相位和 OAM 谱。

Use this workflow to inspect the field, intensity, phase, and OAM spectrum for one set of beam, SPP, propagation, and turbulence parameters.

```bash
python scripts/field_generation/simulate_single_fvvb_beam.py --config configs/default.yaml
```

运行后会保存：

Saved outputs:

- `results/logs/<run>/config_used.yaml`：本次运行使用的完整参数。 / full merged configuration.
- `results/logs/<run>/summary.json`：能量、OAM、偏振和诊断指标。 / energy, OAM, polarization, and diagnostic summaries.
- `results/data/<run>/single_simulation_raw.npz`：原始 numpy 数组。 / raw numpy arrays.
- `results/figures/<run>/intensity_phase.png`：强度和相位图。 / intensity and phase figure.
- `results/figures/<run>/oam_spectrum.png`：OAM 谱图。 / OAM spectrum figure.

### 2. 湍流响应扫描 / Turbulence Response Scan

用于快速检查不同 `alpha`、`Cn2` 和传播距离下的 OAM 谱展宽趋势。

Use this workflow for quick checks of OAM spectral broadening under different `alpha`, `Cn2`, and propagation distance settings.

```bash
python scripts/propagation_studies/scan_turbulence_response.py --config configs/propagation_scan.yaml
```

默认扫描参数较小，只用于快速验证代码流程。作为论文结果前，应提高 `grid_n`、`lmax`、`nr_fourier`、`nphi_fourier` 和 `n_mc`，并检查 `Pcap`、边缘能量比例、有限值诊断和数值收敛性。

The default scan is intentionally small and is only for workflow validation. Before using results as thesis-grade data, increase `grid_n`, `lmax`, `nr_fourier`, `nphi_fourier`, and `n_mc`, then check `Pcap`, edge energy, finite-value diagnostics, and numerical convergence.

### 3. 复现默认输出 / Reproduce Default Outputs

```bash
python scripts/reproduction/reproduce_default_outputs.py
```

该命令会重新运行默认单束仿真流程。最终论文图应基于人工确认过的 YAML 参数重新生成，并在检查后复制到 `thesis_figures/`。

This command reruns the default single-beam workflow. Final thesis figures should be generated from reviewed YAML configurations and copied into `thesis_figures/` only after manual inspection.

## 配置文件 / Configuration Files

- `configs/default.yaml`：快速单束光场 sanity check。 / fast single-beam sanity check.
- `configs/lg_beam.yaml`：较高分辨率的自由空间/SPP 模态分析配置。 / higher-resolution free-space/SPP modal-analysis setup.
- `configs/propagation_scan.yaml`：轻量级湍流扫描配置。 / lightweight turbulence scan setup.

建议所有物理参数和数值参数都在 YAML 文件中修改，不要直接写死在脚本里。

All physical and numerical parameters should be edited in YAML files rather than hard-coded in scripts.

## 重要物理假设 / Important Physical Assumptions

- 默认使用 SI 单位。 / SI units are used unless explicitly stated.
- FVVB 线偏振 Jones 分量沿用原模型约定： / FVVB uses the original Jones convention:
  - `Ex = sum_n c_n E'_n sin(n phi)`
  - `Ey = i sum_n c_n E'_n cos(n phi)`
- 圆偏振分量定义为： / Circular components are defined as:
  - `sigma+ = (Ex + i Ey) / sqrt(2)`
  - `sigma- = (Ex - i Ey) / sqrt(2)`
- SPP/叉形相位在同一平面上乘以 `exp(i * fork_ell * phi)`。 / The SPP/fork phase is applied as `exp(i * fork_ell * phi)` on the same plane.
- 当前湍流模型是共同标量相位屏近似，同时作用于两个偏振分量。 / The turbulence model uses a common scalar phase screen acting on both polarization components.
- 当前模型不包含偏振相关散射、双折射或完整退偏振。 / It does not model polarization-dependent scattering, birefringence, or full depolarization.

## 验证 / Verification

```bash
python -m unittest discover -s tests
python -m compileall -q src scripts tests
```

测试会检查核心数值约定和项目组织方式，避免后续调试时退回按图号命名的脚本结构。

The tests check numerical conventions and repository organization so later debugging does not drift back to figure-number-based script names.

## 旧代码归档 / Legacy Code

旧代码保留在 `legacy_code/PYTHON/` 中，用于追溯早期结果。文件已经按研究主题重命名，而不是按图号命名：

The legacy source code is kept under `legacy_code/PYTHON/` for traceability. Files have been renamed by research topic instead of figure number:

- `legacy_code/PYTHON/core/`
- `legacy_code/PYTHON/chapter_03_free_space_and_spp/`
- `legacy_code/PYTHON/chapter_04_turbulence/`

旧代码生成的 `.npz`、`.png` 和参数 JSON 默认不进入 Git。

Generated legacy `.npz`, `.png`, and parameter JSON files are intentionally excluded from Git.

