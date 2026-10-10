# Robust Steering

![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white) ![PyTorch 2.9.1](https://img.shields.io/badge/PyTorch-2.9.1-EE4C2C?logo=pytorch&logoColor=white) ![Version 0.1.0](https://img.shields.io/badge/version-0.1.0-007C7C) ![arXiv: XXXX](https://img.shields.io/badge/arXiv-XXXX-B31B1B?logo=arxiv&logoColor=white)

<p align="center">
  <img src="assets/steering-demo.gif" alt="Illustration of closed-loop activation steering under a language distribution shift." width="776">
</p>

Robust Steering is a Python package for steering language-model representations with robust control. It models representation dynamics across transformer layers and uses finite-horizon H∞ control to steer toward semantic targets under distribution shift.

The package brings together controller synthesis, activation interventions, and evaluation across truthfulness, safety, multilingual reasoning, and long-context tasks. It supports closed-loop steering without updating model weights, with LQR and activation-steering baselines for comparison.

[Installation](#installation) · [Quick start](#quick-start) · [Benchmarks](#benchmarks) · [CLI](#command-line-interface) · [Documentation](#documentation)

## Features

- **Robust controller synthesis.** Finite-horizon H∞ control with disturbance geometry estimated from representation-dynamics residuals.
- **Representation-level steering.** Semantic target calibration, reduced-state dynamics, and activation interventions during generation.
- **A shared controller interface.** H∞, LQR, PID, and activation-addition controllers, with model adapters separated from controller mathematics.
- **Benchmark integration.** Task-specific calibration and evaluation of steering effectiveness, transfer, and capability preservation.

## Installation

Requires Python 3.11 or newer.

```bash
git clone https://github.com/rgs2151/controller.git
cd controller
python -m pip install -e .
```

Import the package as `robust_steerability`. A Conda environment is also provided in [environment.yml](environment.yml):

```bash
conda env create -f environment.yml
conda activate robust-steerability
```

Language-model benchmarks require CUDA GPUs and access to the selected models. Benchmark-specific requirements are documented in the guides below.

## Quick start

Synthesize and apply an H∞ controller using the standalone control API. This example runs on CPU:

```python
import torch
from robust_steerability import FiniteHorizonControlProblem, HInfinityController

# Four-step dynamics with a two-dimensional state.
identity = torch.eye(2)
layers = identity.repeat(4, 1, 1)

problem = FiniteHorizonControlProblem(
    dynamics=0.9 * layers,
    control_channels=layers,
    disturbance_channels=0.1 * layers,
    state_costs=layers,
    control_costs=layers,
    terminal_cost=identity,
)
controller = HInfinityController.synthesize(problem, device="cpu")

state_deviation = torch.tensor([[1.0, -0.5]])
intervention = controller.intervention(0, state_deviation)
```

`FiniteHorizonControlProblem` specifies the dynamics, control and disturbance channels, and performance costs. The synthesized controller maps the current state deviation to an intervention. In language-model applications, runtime policies connect this interface to model representations and semantic targets.

See the [controller API](robust_steerability/control/README.md) and [package guide](robust_steerability/README.md) for integration details.

## Benchmarks

The repository includes benchmarks for both steering the intended behavior and preserving capabilities while steering a separate behavior.

| Benchmark | Steering target | Evaluation |
|---|---|---|
| [TruthfulQA](benchmarks/truthfulness/README.md) | Truthfulness | English and Spanish-input questions, with English output. |
| [HarmBench](benchmarks/harmful/README.md) | Safe refusal | Direct harmful requests and jailbreak variants. |
| [MGSM](benchmarks/mgsm/README.md) | Spanish-language output | Mathematical accuracy across input languages. |
| [L-CiteEval](benchmarks/lciteeval/README.md) | Positive sentiment | Answer and citation quality in long contexts. |
| [RealToxicityPrompts](benchmarks/toxicity/README.md) | Reduced toxicity | Toxicity and response quality. |

Each benchmark provides artifact preparation, calibration, evaluation, and scoring through `python -m robust_steerability.benchmarks.<benchmark>`. The linked guides contain supported models, commands, and evaluation protocols.

## Command-line interface

Run benchmark commands from the repository root. The following example evaluates Llama-3-8B on TruthfulQA, comparing the original model, A-LQR, and H∞ on English and Spanish-input questions:

```bash
# Prepare model and dataset artifacts.
python -m robust_steerability.benchmarks.truthfulness artifacts \
  --model llama8b --devices auto

# Calibrate the steering methods.
python -m robust_steerability.benchmarks.truthfulness calibrate \
  --model llama8b --methods alqr,h_infinity --devices auto

# Evaluate the selected methods and conditions.
python -m robust_steerability.benchmarks.truthfulness evaluate \
  --model llama8b --methods original,alqr,h_infinity \
  --datasets id,spanish --devices auto

# Score the generated responses.
python -m robust_steerability.benchmarks.truthfulness score \
  --model llama8b --methods original,alqr,h_infinity \
  --datasets id,spanish --scorers default --devices auto
```

`--devices auto` uses the available CUDA GPUs; use `--devices cuda:0` to select a single GPU. API-based judges require `OPENAI_API_KEY`. Model access and scorer requirements are described in each benchmark guide.

HarmBench, MGSM, L-CiteEval, and RealToxicityPrompts use the same stage interface through the `harmful`, `mgsm`, `lciteeval`, and `toxicity` modules, respectively. Consult each module's help for supported model keys and options:

```bash
python -m robust_steerability.benchmarks.truthfulness --help
python -m robust_steerability.benchmarks.harmful --help
```

## Documentation

- [Package guide](robust_steerability/README.md) — model integration, calibration, and runtime architecture.
- [Controller API](robust_steerability/control/README.md) — synthesis interfaces and mathematical conventions.
- [Benchmark guide](robust_steerability/benchmarks/README.md) — evaluation pipelines and benchmark configuration.
- [Reproducibility conventions](DECISIONS.md) — scientific settings and baseline choices.
- [Baseline implementations](robust_steerability/source_methods/README.md) — source methods and attribution.

## Repository structure

The implementation lives in `robust_steerability/`. Benchmark specifications and results are in `benchmarks/`; supporting figures and analyses are maintained separately in `figs/` and `parking/`. Reference material remains in `ref/`.

The Python package is independent of the supplementary analysis folders. Incorporated reference implementations retain their upstream attribution and applicable licenses.

## Citation

If you use Robust Steering in your research, please cite the accompanying manuscript:

```bibtex
@unpublished{fallah2026mindcontrol,
  title  = {Mind Control of {LLMs} via Closed-Loop Stimulation: Out-of-Distribution Robustness},
  author = {Fallah, Kasra and Singha, Rudramani and Ghanei, Hannah and Sajda, Paul and Anderson, James and Zabeh, Erfan},
  year   = {2026},
  note   = {Manuscript}
}
```
