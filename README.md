# TD-MoE (Tensor Decomposition for MoE Compression)

This is a lightweight implementation of the TD-MoE method described in the paper `TD-MoE: Tensor Decomposition for MoE Models` (ICLR 2026).

## What’s implemented
- Cross-expert tensorization: stack all experts’ weights in a layer into a 3D tensor `(K, d_out, d_in)`.
- Multilinear whitening (input, output, or both): compute covariance of activations/gradients and apply whitening before decomposition.
- Tucker decomposition (internal PyTorch implementation) and re-coloring of factors.
- 3D rank allocation to match a target compression ratio.

## Install

```bash
python -m pip install torch transformers
```

Optional for evaluation pipelines and WikiText-2 calibration:

```bash
python -m pip install datasets "lm_eval[hf]"
```

## Usage

Minimal compression run:

```bash
python -m tdmoe.compress \
  --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --target-reduction 0.2 \
  --whitening output \
  --calib-text "The quick brown fox jumps over the lazy dog." \
  --calib-text "Compression of MoE models can be done with tensor decomposition." \
  --max-batches 8
```

Use a calibration text file:

```bash
python -m tdmoe.compress \
  --model Qwen/Qwen2-57B-A14B \
  --target-reduction 0.4 \
  --whitening input \
  --calib-text-file /path/to/calib.txt \
  --max-batches 16
```

Use WikiText-2 for calibration:

```bash
python -m tdmoe.compress \
  --model Qwen/Qwen2-57B-A14B \
  --target-reduction 0.2 \
  --whitening output \
  --calib-dataset wikitext-2 \
  --calib-split train \
  --calib-max-samples 1024
```

Compress, save, and evaluate perplexity plus downstream tasks in one run:

```bash
python src/compress.py \
  --model Qwen/Qwen2-57B-A14B \
  --target-reduction 0.2 \
  --whitening output \
  --calib-dataset wikitext-2 \
  --calib-split train \
  --calib-max-samples 256 \
  --save-path outputs/qwen2_57b_a14b_tdmoe.pt \
  --save-pretrained-path outputs/qwen2_57b_a14b_tdmoe_hf \
  --eval-perplexity-datasets wiki,ptb,c4 \
  --lm-eval-tasks mmlu,arc_challenge \
  --eval-report-path outputs/qwen2_57b_a14b_eval.json
```

Evaluate a previously saved compressed checkpoint:

```bash
python src/evaluate.py \
  --model Qwen/Qwen2-57B-A14B \
  --checkpoint-path outputs/qwen2_57b_a14b_tdmoe.pt \
  --eval-perplexity-datasets wiki,ptb,c4 \
  --lm-eval-tasks mmlu,arc_challenge \
  --report-path outputs/qwen2_57b_a14b_eval.json
```

Notes:
- `--target-reduction 0.2` means 20% parameter reduction in the Tucker representation.
- `--whitening` can be `none`, `input`, `output`, or `both`.
- `--linear-names` controls which expert linear submodules are compressed (default: `w1,w2,w3,up_proj,down_proj,gate_proj`).
- `--save-pretrained-path` exports a Hugging Face checkpoint directory, which is the format used by `lm-evaluation-harness`.
- `--eval-perplexity-datasets` supports `wiki`, `ptb`, and `c4`. The default evaluation split is `test` for WikiText-2/PTB and `validation` for C4.
- `--lm-eval-tasks` passes the compressed model through `lm-evaluation-harness` using the Hugging Face backend. Task names like `mmlu` and `arc_challenge` follow the harness naming.

## Project layout

- `tdmoe/compress.py`: CLI entry point.
- `tdmoe/evaluate.py`: evaluate compressed checkpoints on perplexity datasets and `lm-evaluation-harness`.
- `tdmoe/rank_allocation.py`: rank search for target compression.
- `tdmoe/tucker.py`: Tucker decomposition helpers.
- `tdmoe/stats.py`: activation/gradient covariance collection.
- `tdmoe/whitening.py`: whitening utilities.
- `tdmoe/model_adapters.py`: expert discovery and weight handling.

## Caveats
- This implementation is designed to be readable and modifiable. It prioritizes correctness over speed.
- Large models (Mixtral/Qwen2-MoE) require significant GPU memory; consider smaller checkpoints for smoke tests.
