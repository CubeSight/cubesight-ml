# CubeSight-ML

**CubeSight-ML** is a high-performance, stage-based sequential CFOP (Fridrich method) Rubik's Cube solver powered by deep reinforcement learning. It trains independent **ResMLP** cost-to-go value models using **DeepCubeA-compliant Deep Approximate Value Iteration (DAVI)** and solves cubes sequentially using **Beam Search** across four classical CFOP stages:

$$\text{Scramble} \xrightarrow[\text{Color-Neutral}]{\text{Cross}} \text{Cross on } f \xrightarrow{R_{f \to D}} \text{Cross on D} \xrightarrow{\text{F2L}} \text{F2L Solved} \xrightarrow{\text{OLL}} \text{OLL Solved} \xrightarrow{\text{PLL}} \text{Solved Cube}$$

---

## Key Features

1. **Slice-Turn Metric (STM) & 54-Move Action Space**:
   - Complete 54-move speedcubing action space enabled by default:
     - **18 Face turns**: $U, U2, U', D, D2, D', L, L2, L', R, R2, R', F, F2, F', B, B2, B'$
     - **9 Slice turns**: $M, M2, M', E, E2, E', S, S2, S'$
     - **18 Wide turns**: $u, u2, u', d, d2, d', l, l2, l', r, r2, r', f, f2, f', b, b2, b'$ (e.g. $r = R \cdot M'$)
     - **9 Whole-cube rotations**: $x, x2, x', y, y2, y', z, z2, z'$ (e.g. $x = r \cdot L'$)
   - **Double moves enabled by default** across all turn and rotation classes.
   - **STM Cost Model**: Face, slice, and wide turns cost 1 move; whole-cube rotations cost 0 moves in STM.

2. **Configurable Move Execution Penalties**:
   - Add ergonomics and execution difficulty penalties $\lambda_m \ge 0.0$ for each possible move during Beam Search.
   - Beam Search candidate ranking prioritizes candidates by path cost plus heuristic: $f(s) = g(s) + h(s)$, where $g(s) = \sum [ \text{base\_cost}(m) + \lambda_m ]$.
   - Includes speedcubing fingertrick preset in `presets/ergonomic_penalties.json` to penalize awkward regrips and moves like $B, D, S, E$ or excessive cube rotations.

3. **Strict ResMLP Architecture**:
   - Deep residual multilayer perceptron with batch normalization, ReLU activations, and residual skip connections.
   - Default per-sticker one-hot representation ($54 \times 6 = 324$ inputs) or per-piece categorical representation ($20 \times 2 + 12 \times 12 + 8 \times 8 = 256$ inputs).
   - Zero spatial inductive bias, no convolution, and no attention mechanisms.

4. **Rigorous DAVI Implementation (DeepCubeA Compliance)**:
   - **Bellman Target Masking**: Goal children receive target value $0.0$; goal parents receive target value $0.0$ (`clamp_min(0.0)`).
   - **Target Network Synchronization**: Two supported policies:
     - `deepcubea`: Sync active weights to target network every $C$ steps if interval training loss is below threshold $\epsilon$.
     - `plateau`: Sync when validation loss or solve rate plateaus.
   - **FLOP Accounting**: Integrated parameter and MACs profiling via `fvcore`.

5. **Approach 2: Native Multi-Goal Color-Neutral Cross**:
   - Cross model evaluates distances to any of the 6 possible cross faces (White, Yellow, Green, Blue, Red, Orange).
   - Inference automatically detects the solved cross face $f$ and applies a rigid whole-cube reorientation $R_{f \to D}$ before transitioning to F2L.
   - Bit-exact 3D spatial rotation mapping with full color and piece permutation parity.

6. **Curriculum & Precondition State Mixing**:
   - Infinite on-the-fly streaming dataloader mixing backward scrambles from goal states with legal CFOP precondition states:
     - Cross: 0% precondition (starts from any legal scrambled cube).
     - F2L: 35% precondition (cubes with D-cross solved).
     - OLL: 35% precondition (cubes with F2L solved).
     - PLL: 35% precondition (cubes with OLL solved).

7. **Beam Search Evaluation with Execution Penalties**:
   - Single-stage isolated evaluation (`scripts/eval_stage.py`).
   - End-to-end 4-stage pipeline solver (`scripts/solve_staged.py`).
   - Tracks both STM length and execution costs, reporting structured JSON and TensorBoard artifacts in a unified `runs/` layout.

---

## Directory Structure

```text
cubesight-ml/
├── src/cubesight_ml/
│   ├── cube/                  # Flat representation (40-d), 3D rotations, vectorized PyTorch ops
│   ├── datasets/              # Stage-aware DAVI streaming IterableDataset & DataLoaders
│   ├── envs/                  # Rubik's cube vectorized environment & step transitions
│   ├── models/                # ResMLP architecture & value wrapper
│   ├── search/                # Batched Beam Search & 4-stage StagedSolver
│   ├── stages/                # Stage definitions, goal checkers, and legal goal samplers
│   └── training/              # DAVI training loop, Bellman targets, target sync, checkpointing
├── scripts/
│   ├── train_stage_davi.py    # CLI to train a stage model with DAVI
│   ├── eval_stage.py          # CLI to evaluate a single stage model via Beam Search
│   └── solve_staged.py        # CLI to evaluate the full 4-stage sequential CFOP pipeline
├── tests/                     # Comprehensive unit & integration test suite (40+ tests)
└── runs/                      # Experiment logs, checkpoints, metrics, and profiles
```

---

## Installation & Setup

CubeSight-ML uses Python 3.12+ and `uv` (or `pip`).

```bash
# Clone the repository
git clone https://github.com/your-username/cubesight-ml.git
cd cubesight-ml

# Install dependencies with uv
uv sync

# Or with pip
pip install -e .
```

Verify the installation by running the test suite:

```bash
uv run pytest
```

---

## Workflow & CLI Usage

### 1. Training a Stage Model (`train_stage_davi.py`)

Train each of the four CFOP stages independently. Checkpoints, metrics, configs, and TensorBoard logs are written automatically to `runs/davi/<stage>/<timestamp>[_<exp_name>]`.

#### Train Cross (Color-Neutral, 0% precondition fraction)
```bash
uv run python scripts/train_stage_davi.py \
  --stage cross \
  --batch_size 10000 \
  --steps 1000000 \
  --scramble_depth 30 \
  --precondition_fraction 0.0 \
  --sync_policy deepcubea \
  --sync_interval 1000 \
  --sync_loss_threshold 0.05 \
  --exp_name baseline
```

#### Train F2L (D-Cross precondition fraction: 0.35)
```bash
uv run python scripts/train_stage_davi.py \
  --stage f2l \
  --batch_size 10000 \
  --steps 1000000 \
  --scramble_depth 30 \
  --precondition_fraction 0.35 \
  --sync_policy deepcubea \
  --exp_name baseline
```

#### Train OLL (F2L precondition fraction: 0.35)
```bash
uv run python scripts/train_stage_davi.py \
  --stage oll \
  --batch_size 10000 \
  --steps 500000 \
  --scramble_depth 20 \
  --precondition_fraction 0.35 \
  --sync_policy deepcubea \
  --exp_name baseline
```

#### Train PLL (OLL precondition fraction: 0.35)
```bash
uv run python scripts/train_stage_davi.py \
  --stage pll \
  --batch_size 10000 \
  --steps 500000 \
  --scramble_depth 20 \
  --precondition_fraction 0.35 \
  --sync_policy deepcubea \
  --exp_name baseline
```

Key CLI Flags for `train_stage_davi.py`:
- `--stage`: Stage to train (`cross`, `f2l`, `oll`, `pll`).
- `--state_representation`: `per_sticker` (default, 324-d) or `per_piece` (256-d).
- `--n1`, `--n2`, `--n_r`: Width of linear projection (default: 5000), width of residual blocks (default: 1000), and number of residual blocks (default: 4).
- `--sync_policy`: `deepcubea` (interval + threshold) or `plateau` (moving average threshold).
- `--precondition_fraction`: Proportion of training samples drawn from valid preceding stage states (defaults: 0.0 for cross, 0.35 for f2l/oll/pll).
- `--num_workers`: PyTorch DataLoader worker count.
- `--eval_interval`: Run inline validation Beam Search hook every $N$ steps.

---

### 2. Evaluating a Single Stage (`eval_stage.py`)

Test a single stage model in isolation on legal precondition states using STM and ergonomic move penalties:

```bash
uv run python scripts/eval_stage.py \
  --stage f2l \
  --ckpt runs/davi/f2l/20261024_baseline/best_model.pt \
  --beam_width 1000 \
  --max_steps 50 \
  --n_states 100 \
  --move_penalties presets/ergonomic_penalties.json
```

Outputs written to `runs/eval/stage/<stage>/<timestamp>/`:
- `metrics.json`: Solve rate, mean/median STM length, mean execution cost, mean nodes expanded.
- `lengths_distribution.json`: Frequency table of solution lengths.
- `solutions_sample.json`: First 10 solution action sequences.
- `config.json`: Evaluation parameters and environment configuration.

---

### 3. End-to-End 4-Stage CFOP Pipeline Solver (`solve_staged.py`)

Evaluate the full sequential pipeline on full random scrambles with STM and ergonomic execution penalties:

```bash
uv run python scripts/solve_staged.py \
  --cross_ckpt runs/davi/cross/best_model.pt \
  --f2l_ckpt runs/davi/f2l/best_model.pt \
  --oll_ckpt runs/davi/oll/best_model.pt \
  --pll_ckpt runs/davi/pll/best_model.pt \
  --beam_width 1000 \
  --max_steps 50 \
  --n_scrambles 100 \
  --scramble_depth 30 \
  --move_penalties presets/ergonomic_penalties.json
```

You can customize search parameters per stage using `--beam_widths` and `--max_steps_per_stage` (order: Cross, F2L, OLL, PLL):

```bash
uv run python scripts/solve_staged.py \
  --cross_ckpt runs/davi/cross/best_model.pt \
  --f2l_ckpt runs/davi/f2l/best_model.pt \
  --oll_ckpt runs/davi/oll/best_model.pt \
  --pll_ckpt runs/davi/pll/best_model.pt \
  --beam_widths 500 1000 500 500 \
  --max_steps_per_stage 30 60 30 30 \
  --n_scrambles 100
```

Outputs written to `runs/eval/pipeline/<timestamp>/`:
- `pipeline_results.json`: Overall solve rate, total move lengths, mean runtime per cube.
- `stage_breakdown.json`: Stage-by-stage solve rates, move lengths, and failure counts.
- `solutions.json`: Full solution move sequences and stage transition states.
- `config.json`: Pipeline configuration.

---

### 4. Interactive 3D Rubik's Cube Solver Studio (`visualize_solver.py`)

Launch the native desktop 3D Rubik's Cube window to run and inspect Beam Search solving visually:

```bash
# Launch native desktop 3D window (auto-discovers latest checkpoints in runs/)
uv run python scripts/visualize_solver.py

# Or launch with explicit checkpoints
uv run python scripts/visualize_solver.py \
  --cross_ckpt runs/davi/cross/best_model.pt \
  --f2l_ckpt runs/davi/f2l/best_model.pt \
  --oll_ckpt runs/davi/oll/best_model.pt \
  --pll_ckpt runs/davi/pll/best_model.pt \
  --move_penalties presets/ergonomic_penalties.json

# Or open in default web browser instead of desktop window
uv run python scripts/visualize_solver.py --browser
```

#### Studio Features:
- **Real-Time 3D WebGL Cube**: 360° mouse orbit controls, smooth zoom, realistic sticker materials, and dark-mode glassmorphic interface.
- **Stage & Pipeline Solving**: Solve individual stages (Cross, F2L, OLL, PLL) or the entire 4-stage CFOP pipeline sequentially.
- **Smooth 3D Layer Animations**: Continuous animated layer turns for all 54 STM moves (faces, slices $M, E, S$, wide turns $r, l, u, d, f, b$, and rotations $x, y, z$).
- **Timeline & Playback Controls**: Play, pause, step forward, step backward, jump to start/end, and adjustable move animation speed.
- **Interactive Move Chips**: Clickable solution timeline chips that jump directly to any step.
- **Scramblers & Preconditions**: 1-click random scrambles (configurable depth 1..100), custom move sequence input (e.g. `R U R' U'`), or legal stage precondition states.
- **Live Metrics Inspector**: Real-time display of STM length, total length, execution cost, search time, and stage breakdown.

---

## Artifacts & Runs Layout

All runs output to a clean, standardized hierarchy:

```text
runs/
├── davi/
│   └── <stage>/
│       └── <timestamp>_<exp_name>/
│           ├── config.json
│           ├── console.log
│           ├── model_profile.json
│           ├── seeds.json
│           ├── checkpoint_<step>.pt
│           ├── checkpoint_<step>_stats.json
│           ├── best_model.pt
│           ├── best_model_stats.json
│           ├── final_model.pt
│           ├── final_model_stats.json
│           └── tensorboard/
│               └── events.out.tfevents.*
└── eval/
    ├── stage/
    │   └── <stage>/
    │       └── <timestamp>/
    │           ├── config.json
    │           ├── metrics.json
    │           ├── lengths_distribution.json
    │           └── solutions_sample.json
    └── pipeline/
        └── <timestamp>/
            ├── config.json
            ├── pipeline_results.json
            ├── stage_breakdown.json
            └── solutions.json
```

---

## Code Quality & Standards

Code is formatted and linted according to PEP 8 standards via Ruff:

```bash
# Check linting
uv run ruff check .

# Check code formatting
uv run ruff format --check .

# Auto-format
uv run ruff format .
```
