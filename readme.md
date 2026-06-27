# IASCM: Input-Aware and Generalizable Error Resilience Prediction for Reliable GPU Computing

A graph-based deep learning framework for GPU soft error resilience assessment across diverse inputs. IASCM leverages multi-relation graph attention over instruction-level control-flow and data-flow graphs to predict soft error outcomes (Masked / SDC / DUE) under varying program inputs.

---

## Sera-DI Dataset

**Sera-DI** (Soft Error Resilience Assessment across Diverse Inputs) is a publicly available dataset containing GPU fault injection results across 12 benchmark programs with diverse input configurations.

### Download

Sera-DI is available for download at: [**Sera-DI on Google Drive**](https://drive.google.com/file/d/1ar3go8fFTCCM8Q9HqIi4fuheWef4yNcL/view?usp=drive_link)

After downloading, extract the archive and place the dataset directories under the `datasets/` folder, so that the project structure matches the layout described below.

### Dataset Structure

The dataset is organized by benchmark program. Each program has its own directory under `datasets/`:

```
datasets/
├── 2mm/
├── atax/
├── backprop/
├── conv2d/
├── gaussian/
├── gemm/
├── lavaMD/
├── lud/
├── mvt/
├── nn/
├── nw/
└── pathfinder/
```

Each program directory contains the following files:

| File                       | Description                                                               |
| ----------------------------| ---------------------------------------------------------------------------|
| `dataset_train.csv`        | Training set with fault injection records                                 |
| `dataset_val_cleaned.csv`  | Validation / Test-1 set (cleaned)                                         |
| `dataset_test_cleaned.csv` | Test-2 set (cleaned, unseen inputs)                                       |
| `encode_setting.json`      | Feature encoding configuration (max values for one-hot / binary encoding) |
| `node_mapping.json`        | Mapping from `(kernel_name, pcOffset)` pairs to graph node indices        |
| `inputs_di_count.csv`      | Per-input dynamic instruction execution counts for each node              |
| `instr_embeddings.npy`     | Pre-computed instruction node embeddings (NumPy array)                    |
| `{program}_lrm_clean.txt`  | Cleaned CUDA binary live register map (LRM) used for graph extraction     |
| `pc_extraction.txt`        | Extracted PC (program counter) offsets grouped by kernel                  |

### Data Fields in CSV Files

Each row in the training/test CSV files represents a single fault injection trial. Key columns include:

| Column          | Description                                             |
| -----------------| ---------------------------------------------------------|
| `kernel_name`   | Name of the CUDA kernel where the fault was injected    |
| `kernel_index`  | Index of the kernel invocation                          |
| `pcOffset`      | Program counter offset of the target instruction        |
| `instID`        | Instruction ID within the kernel                        |
| `regNo`         | Target register number                                  |
| `fip_pos`       | Bit position of the injected flip (0–31)                |
| `fip_tp`        | Flip type                                               |
| `blockID`       | CUDA thread block ID                                    |
| `blockTID`      | Thread ID within the block                              |
| `input`         | Input configuration identifier                          |
| `inject_result` | Fault outcome label: `0` = Masked, `1` = SDC, `2` = DUE |

---

## Requirements

### Hardware
- NVIDIA GPU with CUDA support

### Software

| Dependency | Version |
|------------|---------|
| CUDA | 12.1 |
| NVIDIA Driver | ≥ 537.70 |
| Python | 3.9.20 |
| pandas | 2.2.3 |
| numpy | 1.25.1 |
| scikit-learn | 1.6.1 |
| PyTorch | 2.1.0 (with CUDA 12.1) |
| tqdm | ≥ 4.65 |

### Installation

```bash
# Create a conda environment (recommended)
conda create -n iascm python=3.9.20
conda activate iascm

# Install PyTorch with CUDA 12.1
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install pandas==2.2.3 numpy==1.25.1 scikit-learn==1.6.1 tqdm
```

---

## Project Structure

```
code_script/
├── README.md                     # This file
├── IASCM.py                     # Main model: training, evaluation, and inference
├── create_dataset.py             # Data preprocessing and feature engineering
├── extract_ifg_from_lrm.py       # Instruction-level graph extraction (control-flow & data-flow)
└── extract_bbcfg_from_lrm.py     # Basic-block-level control flow graph extraction from LRM
```

### Code Description

| File | Description |
|------|-------------|
| `IASCM.py` | Entry point. Defines the **MultiRelationGraphTransformer** model, the **Trainer** class, data loading pipeline, and the main training loop. Iterates over all 12 benchmark programs. |
| `create_dataset.py` | Converts raw CSV records into model-ready features: encodes categorical fields (kernel index, instruction ID, register number, bit position, block ID, thread ID) via one-hot or binary encoding, and constructs per-input dynamic instruction count vectors. |
| `extract_ifg_from_lrm.py` | Parses the cleaned LRM file to extract **instruction-level control-flow graph (ICFG)** and **instruction-level data-flow graph (IDFG)** edges, then builds adjacency matrices for the two graph relations used by the model. |
| `extract_bbcfg_from_lrm.py` | Parses the cleaned LRM file to extract **basic-block-level control-flow graph (BBCFG)** structure, including basic block boundaries, sequential fall-through edges, and branch/jump edges. |

---

## Usage

### Step 1: Download and Prepare the Dataset

1. Download **Sera-DI** from [Google Drive](https://drive.google.com/file/d/1ar3go8fFTCCM8Q9HqIi4fuheWef4yNcL/view?usp=drive_link).
2. Extract the archive.
3. Place the program directories (e.g., `2mm/`, `atax/`, ...) under the `datasets/` folder at the project root, so that the relative path `../datasets/{program}/` is accessible from `code_script/`.

### Step 2: Run Training

```bash
cd code_script
python IASCM.py
```

This will sequentially train and evaluate IASCM on all 12 benchmark programs. For each program, the script:
1. Loads the pre-computed instruction embeddings and builds the control-flow and data-flow adjacency matrices from the LRM file.
2. Samples up to 15,000 training instances and 15,000 validation instances from `dataset_train.csv`.
3. Trains the **MultiRelationGraphTransformer** for 30 epochs with learning rate scheduling.
4. Saves the best model checkpoint (based on validation accuracy + macro-F1) to `code_script/checkpoints/{program}/`.
5. Appends evaluation results to `code_script/test_results.txt`.

### Step 3: View Results

After training completes, results are saved in:
- `code_script/test_results.txt` — per-program accuracy and macro-F1 scores.
- `code_script/checkpoints/{program}/best_model.pt` — saved model weights.

---

## Hyperparameters

The default hyperparameters are configured in `IASCM.py`:

| Parameter       | Default | Description                            |
| -----------------| ---------| ----------------------------------------|
| `BATCH_SIZE`    | 64      | Mini-batch size                        |
| `LEARNING_RATE` | 1e-3    | Initial learning rate                  |
| `EPOCHS`        | 30      | Number of training epochs              |
| `NUM_LAYERS`    | 1       | Number of graph attention layers       |
| `DROPOUT`       | 0.2     | Dropout rate                           |
| `TRAIN_SIZES`   | [15000] | Number of training samples per program |

---

## Model Architecture

IASCM uses a **Multi-Relation Graph Attention** mechanism that operates on two types of instruction-level graphs:

1. **Control-Flow Graph (CFG)**: Captures the sequential and branch execution order between instructions.
2. **Register Data-Flow Graph (RDG)**: Captures register-level data dependencies between instructions.

For each relation type, the model computes attention-weighted message passing over the graph structure, conditioned on both the static instruction embeddings and dynamic input features. The multi-relation outputs are fused and passed through a classifier to predict the fault outcome.
