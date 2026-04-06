from __future__ import annotations

import argparse
import json
import os
from typing import List
import tempfile

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate import (
    evaluate_perplexity_suite,
    parse_csv_arg,
    run_lm_eval_harness,
    save_hf_checkpoint,
)
from model_adapters import find_expert_groups, stack_expert_weights, apply_expert_weights
from rank_allocation import RankSearchConfig, search_ranks
from tucker import whiten_tensor, recolor_factors, tucker_decompose, reconstruct
from stats import OnlineCovariance, whitening_from_cov


def load_calibration_texts(path: str | None, extra: List[str] | None) -> List[str]:
    texts = []
    if path:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
    if extra:
        texts.extend(extra)
    if not texts:
        raise ValueError("No calibration texts provided")
    return texts


def load_wikitext2(split: str, max_samples: int | None) -> List[str]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError("datasets package is required for WikiText-2") from exc

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    texts = []
    for item in ds:
        text = item.get("text", "").strip()
        if text:
            texts.append(text)
        if max_samples is not None and len(texts) >= max_samples:
            break
    if not texts:
        raise RuntimeError("WikiText-2 yielded no usable text samples")
    return texts


def make_dataloader(tokenizer, texts: List[str], batch_size: int, seq_len: int):
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=seq_len,
        return_tensors="pt",
    )
    dataset = torch.utils.data.TensorDataset(encodings["input_ids"], encodings["attention_mask"])

    def collate(batch):
        input_ids = torch.stack([b[0] for b in batch])
        attention_mask = torch.stack([b[1] for b in batch])
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids.clone()}

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)


def collect_group_covariances(
    model,
    modules_by_name: dict[str, List[torch.nn.Module]],
    dataloader,
    device,
    max_batches: int,
    collect_grad: bool,
    eps: float,
    cache_device: str | torch.device = "cpu",
):
    covariances = {}
    for linear_name, modules in modules_by_name.items():
        if not modules:
            continue
        sample = modules[0]
        d_out, d_in = sample.weight.shape
        stats_device = sample.weight.device
        covariances[linear_name] = {
            "cov_in": OnlineCovariance(d_in, device=stats_device, dtype=sample.weight.dtype, eps=eps),
            "cov_out": OnlineCovariance(d_out, device=stats_device, dtype=sample.weight.dtype, eps=eps) if collect_grad else None,
        }

    handles = []

    def make_fwd_hook(linear_name: str):
        def fwd_hook(_mod, inp, _out):
            x = inp[0].detach()
            if not torch.isfinite(x).all():
                print(f"Forward for {linear_name} is not all finite")
                x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
            covariances[linear_name]["cov_in"].update(x)

        return fwd_hook

    def make_bwd_hook(linear_name: str):
        def bwd_hook(_mod, _grad_inp, grad_out):
            cov_out = covariances[linear_name]["cov_out"]
            if cov_out is None:
                return
            if grad_out and grad_out[0] is not None:
                g = grad_out[0].detach()
                if not torch.isfinite(g).all():
                    print(f"Gradient for {linear_name} is not all finite")
                    g = torch.nan_to_num(g, nan=0.0, posinf=1e4, neginf=-1e4)
                cov_out.update(g)

        return bwd_hook

    for linear_name, modules in modules_by_name.items():
        fwd_hook = make_fwd_hook(linear_name)
        bwd_hook = make_bwd_hook(linear_name)
        for module in modules:
            handles.append(module.register_forward_hook(fwd_hook))
            if collect_grad:
                handles.append(module.register_full_backward_hook(bwd_hook))

    model.eval()
    for step, batch in tqdm(enumerate(dataloader), desc="Calibrating..."):
        if step >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        model.zero_grad(set_to_none=True)
        if collect_grad:
            out = model(**batch)
            out.loss.backward()
        else:
            with torch.no_grad():
                model(**batch)

    for h in handles:
        h.remove()

    cached = {}
    for linear_name, stats in covariances.items():
        cov_in_mat, _ = stats["cov_in"].finalize()
        cov_out_mat = None
        if stats["cov_out"] is not None:
            cov_out_mat, _ = stats["cov_out"].finalize()

        cached[linear_name] = {
            "cov_in": cov_in_mat.to(cache_device),
            "cov_out": cov_out_mat.to(cache_device) if cov_out_mat is not None else None,
        }

    return cached


def compress_group(model, group, linear_names: List[str], dataloader, device, max_batches: int, target_reduction: float, whitening: str, eps: float, tucker_device: str = "auto"):
    results = {}
    available_linear_names = [linear_name for linear_name in linear_names if hasattr(group.experts[0], linear_name)]
    covariance_cache = {}
    if whitening in {"input", "both", "output"} and available_linear_names:
        modules_by_name = {
            linear_name: [getattr(exp, linear_name) for exp in group.experts]
            for linear_name in available_linear_names
        }
        covariance_cache = collect_group_covariances(
            model=model,
            modules_by_name=modules_by_name,
            dataloader=dataloader,
            device=device,
            max_batches=max_batches,
            collect_grad=whitening in {"output", "both"},
            eps=eps,
            cache_device="cpu",
        )

    for linear_name in available_linear_names:
        weight_tensor = stack_expert_weights(group.experts, linear_name).to(device)
        k, d_out, d_in = weight_tensor.shape

        rank_cfg = RankSearchConfig(target_reduction=target_reduction)
        rank_res = search_ranks(k, d_out, d_in, rank_cfg)

        s_in = s_in_inv = None
        s_out = s_out_inv = None

        if whitening in {"input", "both", "output"}:
            cached_covariances = covariance_cache.get(linear_name)
            if cached_covariances is None:
                raise RuntimeError(f"Missing cached covariances for {group.name}.{linear_name}")

            cov_in = cached_covariances["cov_in"]
            cov_out = cached_covariances["cov_out"]
            if whitening in {"input", "both"}:
                s_in, s_in_inv = whitening_from_cov(cov_in)
                s_in = s_in.to(weight_tensor.device)
                s_in_inv = s_in_inv.to(weight_tensor.device)
            if whitening in {"output", "both"}:
                if cov_out is None:
                    raise RuntimeError("Output covariance requested but not collected")
                s_out, s_out_inv = whitening_from_cov(cov_out)
                s_out = s_out.to(weight_tensor.device)
                s_out_inv = s_out_inv.to(weight_tensor.device)

        t_whitened = whiten_tensor(weight_tensor, s_out, s_in)
        device_override = None
        if tucker_device in {"cpu", "cuda"}:
            device_override = tucker_device
        core, factors = tucker_decompose(
            t_whitened,
            ranks=(rank_res.r1, rank_res.r2, rank_res.r3),
            device_override=device_override,
        )
        factors = recolor_factors(factors, s_out_inv, s_in_inv)
        t_rec = reconstruct(core, factors)

        apply_expert_weights(group.experts, linear_name, t_rec)

        results[linear_name] = {
            "ranks": (rank_res.r1, rank_res.r2, rank_res.r3),
            "params": rank_res.params,
            "target_params": rank_res.target_params,
            "diff": rank_res.diff,
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--target-reduction", type=float, required=True, help="fraction of params to remove (e.g., 0.2)")
    parser.add_argument("--whitening", choices=["none", "input", "output", "both"], default="input")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--calib-text-file", type=str, default=None)
    parser.add_argument("--calib-text", type=str, action="append", default=None)
    parser.add_argument("--calib-dataset", type=str, default=None, help="e.g. wikitext-2")
    parser.add_argument("--calib-split", type=str, default="train")
    parser.add_argument("--calib-max-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--max-batches", type=int, default=16)
    parser.add_argument("--linear-names", type=str, default="w1,w2,w3,up_proj,down_proj,gate_proj")
    parser.add_argument("--tucker-device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--linalg-backend", type=str, choices=["default", "magma", "cusolver"], default="default")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--save-pretrained-path", type=str, default=None, help="Optional HF-format export path for the compressed model")
    parser.add_argument("--eval-perplexity-datasets", type=str, default=None, help="Comma-separated list, e.g. wiki,ptb,c4")
    parser.add_argument("--eval-max-samples", type=int, default=None)
    parser.add_argument("--eval-seq-len", type=int, default=None)
    parser.add_argument("--eval-stride", type=int, default=None)
    parser.add_argument("--lm-eval-tasks", type=str, default=None, help="Comma-separated lm-evaluation-harness tasks, e.g. mmlu,arc_challenge")
    parser.add_argument("--lm-eval-batch-size", type=str, default="auto")
    parser.add_argument("--lm-eval-num-fewshot", type=int, default=0)
    parser.add_argument("--lm-eval-limit", type=int, default=None)
    parser.add_argument("--lm-eval-output-path", type=str, default=None)
    parser.add_argument("--eval-report-path", type=str, default=None)

    args = parser.parse_args()

    if args.calib_dataset:
        if args.calib_dataset.lower() in {"wikitext-2", "wikitext2", "wikitext_2"}:
            texts = load_wikitext2(args.calib_split, args.calib_max_samples)
        else:
            raise ValueError(f"Unsupported calib-dataset: {args.calib_dataset}")
    else:
        texts = load_calibration_texts(args.calib_text_file, args.calib_text)

    if args.linalg_backend != "default":
        torch.backends.cuda.preferred_linalg_library(args.linalg_backend)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype), device_map="auto")
    # model.to(args.device)

    dataloader = make_dataloader(tokenizer, texts, args.batch_size, args.seq_len)

    expert_groups = find_expert_groups(model)
    if not expert_groups:
        raise RuntimeError("No expert groups found in model")

    linear_names = [n.strip() for n in args.linear_names.split(",") if n.strip()]

    all_results = {}
    for group in expert_groups:
        res = compress_group(
            model=model,
            group=group,
            linear_names=linear_names,
            dataloader=dataloader,
            device=args.device,
            max_batches=args.max_batches,
            target_reduction=args.target_reduction,
            whitening=args.whitening,
            eps=args.eps,
            tucker_device=args.tucker_device,
        )
        if res:
            all_results[group.name] = res

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
        torch.save({"model": model.state_dict(), "results": all_results}, args.save_path)

    perplexity_results = {}
    perplexity_datasets = parse_csv_arg(args.eval_perplexity_datasets)
    if perplexity_datasets:
        split_overrides = {
            dataset_name: ("validation" if dataset_name.strip().lower() == "c4" else "test")
            for dataset_name in perplexity_datasets
        }
        perplexity_results = evaluate_perplexity_suite(
            model=model,
            tokenizer=tokenizer,
            dataset_names=perplexity_datasets,
            split_overrides=split_overrides,
            max_samples=args.eval_max_samples,
            seq_len=args.eval_seq_len,
            stride=args.eval_stride,
        )

    lm_eval_results = None
    lm_eval_tasks = parse_csv_arg(args.lm_eval_tasks)
    if args.save_pretrained_path:
        save_hf_checkpoint(model, tokenizer, args.save_pretrained_path)
    if lm_eval_tasks:
        export_dir = args.save_pretrained_path
        if export_dir is None:
            tmpdir = tempfile.TemporaryDirectory(prefix="tdmoe_export_")
            export_dir = tmpdir.name
            save_hf_checkpoint(model, tokenizer, export_dir)
        lm_eval_results = run_lm_eval_harness(
            pretrained_path=export_dir,
            tasks=lm_eval_tasks,
            device=args.device,
            batch_size=args.lm_eval_batch_size,
            num_fewshot=args.lm_eval_num_fewshot,
            limit=args.lm_eval_limit,
            output_path=args.lm_eval_output_path,
        )

    print("Compression summary:")
    for gname, res in all_results.items():
        print(f"- {gname}")
        for lname, info in res.items():
            print(f"  {lname}: ranks={info['ranks']} params={info['params']} target={info['target_params']} diff={info['diff']}")

    if perplexity_results:
        print("Perplexity results:")
        for result in perplexity_results.values():
            print(f"- {result['dataset']} ({result['split']}): ppl={result['perplexity']:.4f}")

    if lm_eval_results:
        print("lm-evaluation-harness:")
        print(f"- tasks: {', '.join(lm_eval_results['tasks'])}")
        print(f"- output_path: {lm_eval_results['output_path']}")

    if args.eval_report_path:
        payload = {
            "compression": all_results,
            "perplexity": perplexity_results,
            "lm_eval": lm_eval_results,
        }
        os.makedirs(os.path.dirname(args.eval_report_path) or ".", exist_ok=True)
        with open(args.eval_report_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
