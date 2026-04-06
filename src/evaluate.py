from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _require_datasets():
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError("datasets package is required for perplexity evaluation") from exc
    return load_dataset


def parse_csv_arg(value: str | None) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _normalize_dtype(dtype: str | None):
    if not dtype:
        return None
    return getattr(torch, dtype)


def _extract_state_dict(payload: Any) -> Dict[str, torch.Tensor]:
    if isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict):
        return payload["model"]
    if isinstance(payload, dict):
        return payload
    raise ValueError("Unsupported checkpoint format; expected a state_dict or {'model': state_dict}")


def load_model_and_tokenizer(
    model_name_or_path: str,
    dtype: str = "float16",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    checkpoint_path: str | None = None,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=_normalize_dtype(dtype),
        device_map="auto" if device.startswith("cuda") or device == "auto" else None,
    )
    if checkpoint_path:
        payload = torch.load(checkpoint_path, map_location="cpu")
        state_dict = _extract_state_dict(payload)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Warning: missing keys when loading checkpoint: {len(missing)}", file=sys.stderr)
        if unexpected:
            print(f"Warning: unexpected keys when loading checkpoint: {len(unexpected)}", file=sys.stderr)
    if not (device.startswith("cuda") or device == "auto"):
        model.to(device)
    model.eval()
    return model, tokenizer


def save_hf_checkpoint(model, tokenizer, output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    return str(out)


def _dataset_spec(name: str, split: str, max_samples: int | None):
    normalized = name.strip().lower()
    split_expr = f"{split}[:{max_samples}]" if max_samples is not None else split
    if normalized in {"wiki", "wikitext", "wikitext2", "wikitext-2"}:
        return ("wikitext", "wikitext-2-raw-v1", split_expr, "text", "WikiText-2")
    if normalized in {"ptb", "penn_treebank", "penn-treebank"}:
        return ("ptb_text_only", "penn_treebank", split_expr, "sentence", "PTB")
    if normalized in {"c4"}:
        return ("allenai/c4", "en", split_expr, "text", "C4")
    raise ValueError(f"Unsupported perplexity dataset: {name}")


def load_eval_texts(dataset_name: str, split: str, max_samples: int | None) -> List[str]:
    load_dataset = _require_datasets()
    path, config, split_expr, field, _ = _dataset_spec(dataset_name, split, max_samples)
    try:
        ds = load_dataset(path, config, split=split_expr)
    except Exception:
        if dataset_name.strip().lower() == "c4":
            ds = load_dataset("c4", "en", split=split_expr)
        else:
            raise

    texts: List[str] = []
    for item in ds:
        text = item.get(field, "")
        if isinstance(text, str):
            text = text.strip()
        if text:
            texts.append(text)
    if not texts:
        raise RuntimeError(f"{dataset_name} yielded no usable text")
    return texts


def _infer_eval_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise RuntimeError("Model has no parameters") from exc


def _infer_max_length(model, tokenizer, seq_len: int | None) -> int:
    if seq_len is not None:
        return seq_len
    config_len = getattr(model.config, "max_position_embeddings", None)
    if isinstance(config_len, int) and config_len > 0:
        return min(config_len, 4096)
    tok_len = getattr(tokenizer, "model_max_length", None)
    if isinstance(tok_len, int) and 0 < tok_len < 1_000_000:
        return min(tok_len, 4096)
    return 2048


def evaluate_perplexity(
    model,
    tokenizer,
    dataset_name: str,
    split: str,
    max_samples: int | None = None,
    seq_len: int | None = None,
    stride: int | None = None,
):
    texts = load_eval_texts(dataset_name, split=split, max_samples=max_samples)
    text = "\n\n".join(texts)
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings["input_ids"][0]
    if input_ids.numel() < 2:
        raise RuntimeError(f"Not enough tokens for perplexity evaluation on {dataset_name}")

    max_length = _infer_max_length(model, tokenizer, seq_len)
    stride = stride or max_length
    device = _infer_eval_device(model)

    nll_sum = torch.tensor(0.0, device=device)
    total_tokens = 0
    prev_end = 0

    for begin in range(0, input_ids.size(0), stride):
        end = min(begin + max_length, input_ids.size(0))
        target_len = end - prev_end
        if target_len <= 0:
            continue

        window = input_ids[begin:end].unsqueeze(0).to(device)
        labels = window.clone()
        labels[:, :-target_len] = -100

        with torch.no_grad():
            outputs = model(input_ids=window, labels=labels)

        nll_sum += outputs.loss * target_len
        total_tokens += target_len
        prev_end = end
        if end >= input_ids.size(0):
            break

    perplexity = torch.exp(nll_sum / total_tokens).item()
    _, _, _, _, canonical_name = _dataset_spec(dataset_name, split, max_samples)
    return {
        "dataset": canonical_name,
        "split": split,
        "num_texts": len(texts),
        "num_tokens": int(total_tokens),
        "seq_len": int(max_length),
        "stride": int(stride),
        "perplexity": float(perplexity),
    }


def evaluate_perplexity_suite(
    model,
    tokenizer,
    dataset_names: Iterable[str],
    split_overrides: Dict[str, str] | None = None,
    max_samples: int | None = None,
    seq_len: int | None = None,
    stride: int | None = None,
):
    results = {}
    split_overrides = split_overrides or {}
    for dataset_name in dataset_names:
        split = split_overrides.get(dataset_name.strip().lower(), "test")
        results[dataset_name] = evaluate_perplexity(
            model=model,
            tokenizer=tokenizer,
            dataset_name=dataset_name,
            split=split,
            max_samples=max_samples,
            seq_len=seq_len,
            stride=stride,
        )
    return results


def run_lm_eval_harness(
    pretrained_path: str,
    tasks: Iterable[str],
    device: str,
    batch_size: str,
    num_fewshot: int = 0,
    limit: int | None = None,
    output_path: str | None = None,
):
    tasks = [task.strip() for task in tasks if task.strip()]
    if not tasks:
        return None

    lm_eval_device = "cuda:0" if device == "cuda" else device
    output_dir = output_path or tempfile.mkdtemp(prefix="tdmoe_lm_eval_")
    cmd = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={pretrained_path}",
        "--tasks",
        ",".join(tasks),
        "--device",
        lm_eval_device,
        "--batch_size",
        str(batch_size),
        "--num_fewshot",
        str(num_fewshot),
        "--output_path",
        output_dir,
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return {
        "tasks": tasks,
        "output_path": output_dir,
        "command": cmd,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _default_split_for(dataset_name: str) -> str:
    if dataset_name.strip().lower() == "c4":
        return "validation"
    return "test"


def _print_perplexity_results(results: Dict[str, Dict[str, Any]]):
    print("Perplexity results:")
    for key, result in results.items():
        print(
            f"- {result['dataset']} ({result['split']}): "
            f"ppl={result['perplexity']:.4f} tokens={result['num_tokens']} texts={result['num_texts']}"
        )


def _print_lm_eval_result(result: Dict[str, Any] | None):
    if not result:
        return
    print("lm-evaluation-harness:")
    print(f"- tasks: {', '.join(result['tasks'])}")
    print(f"- output_path: {result['output_path']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Base model name or a local HF checkpoint directory")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Compressed checkpoint produced by compress.py --save-path")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-perplexity-datasets", type=str, default="wiki,ptb,c4")
    parser.add_argument("--eval-max-samples", type=int, default=None)
    parser.add_argument("--eval-seq-len", type=int, default=None)
    parser.add_argument("--eval-stride", type=int, default=None)
    parser.add_argument("--lm-eval-tasks", type=str, default="mmlu,arc_challenge")
    parser.add_argument("--lm-eval-batch-size", type=str, default="auto")
    parser.add_argument("--lm-eval-num-fewshot", type=int, default=0)
    parser.add_argument("--lm-eval-limit", type=int, default=None)
    parser.add_argument("--lm-eval-output-path", type=str, default=None)
    parser.add_argument("--export-pretrained-path", type=str, default=None, help="Optional HF-format export dir used for lm-eval")
    parser.add_argument("--report-path", type=str, default=None)
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(
        model_name_or_path=args.model,
        dtype=args.dtype,
        device=args.device,
        checkpoint_path=args.checkpoint_path,
    )

    perplexity_datasets = parse_csv_arg(args.eval_perplexity_datasets)
    split_overrides = {name: _default_split_for(name) for name in perplexity_datasets}
    ppl_results = evaluate_perplexity_suite(
        model=model,
        tokenizer=tokenizer,
        dataset_names=perplexity_datasets,
        split_overrides=split_overrides,
        max_samples=args.eval_max_samples,
        seq_len=args.eval_seq_len,
        stride=args.eval_stride,
    ) if perplexity_datasets else {}
    _print_perplexity_results(ppl_results)

    lm_eval_tasks = parse_csv_arg(args.lm_eval_tasks)
    lm_eval_result = None
    if lm_eval_tasks:
        export_dir = args.export_pretrained_path
        if export_dir is None:
            tmpdir = tempfile.TemporaryDirectory(prefix="tdmoe_export_")
            export_dir = tmpdir.name
        save_hf_checkpoint(model, tokenizer, export_dir)
        lm_eval_result = run_lm_eval_harness(
            pretrained_path=export_dir,
            tasks=lm_eval_tasks,
            device=args.device,
            batch_size=args.lm_eval_batch_size,
            num_fewshot=args.lm_eval_num_fewshot,
            limit=args.lm_eval_limit,
            output_path=args.lm_eval_output_path,
        )
        _print_lm_eval_result(lm_eval_result)

    if args.report_path:
        report = {
            "model": args.model,
            "checkpoint_path": args.checkpoint_path,
            "perplexity": ppl_results,
            "lm_eval": lm_eval_result,
        }
        Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
