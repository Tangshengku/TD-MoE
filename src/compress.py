from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List
import tempfile

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate import (
    evaluate_perplexity_suite,
    log as eval_log,
    parse_csv_arg,
    run_lm_eval_harness,
    save_hf_checkpoint,
)
from model_adapters import ExpertGroup, find_expert_groups, stack_expert_weights, apply_expert_weights
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
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id for calibration packing")
    packed_token_ids = []
    for text in texts:
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not token_ids:
            continue
        packed_token_ids.extend(token_ids)
        packed_token_ids.append(eos_token_id)
    if not packed_token_ids:
        raise ValueError("Calibration corpus produced no tokens")
    input_ids = torch.tensor(packed_token_ids, dtype=torch.long)
    total_tokens = (input_ids.numel() // seq_len) * seq_len
    if total_tokens == 0:
        raise ValueError("Calibration corpus is too small for the configured seq_len")
    input_ids = input_ids[:total_tokens].view(-1, seq_len)
    attention_mask = torch.ones_like(input_ids)
    dataset = torch.utils.data.TensorDataset(input_ids, attention_mask)

    def collate(batch):
        input_ids = torch.stack([b[0] for b in batch])
        attention_mask = torch.stack([b[1] for b in batch])
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

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


@dataclass
class GroupAllocation:
    importance: float
    weight: float
    original_params: int
    target_params: int
    target_reduction: float


def _infer_router_top_k(model, group: ExpertGroup) -> int:
    for attr in ("num_experts_per_tok", "num_local_experts_per_tok"):
        value = getattr(getattr(model, "config", None), attr, None)
        if isinstance(value, int) and value > 0:
            return min(value, len(group.experts))
    for attr in ("top_k", "num_experts_per_tok"):
        value = getattr(group.module, attr, None)
        if isinstance(value, int) and value > 0:
            return min(value, len(group.experts))
    return min(2, len(group.experts))


def _extract_group_order(name: str) -> int | None:
    match = re.search(r"\.(\d+)\.", name)
    if match:
        return int(match.group(1))
    return None


def _smooth_scores(groups: List[ExpertGroup], scores: Dict[str, float], alpha: float) -> Dict[str, float]:
    if alpha <= 0:
        return scores
    ordered = sorted(
        ((group, _extract_group_order(group.name)) for group in groups),
        key=lambda item: (item[1] is None, item[1] if item[1] is not None else item[0].name),
    )
    result = dict(scores)
    for idx, (group, layer_idx) in enumerate(ordered):
        if layer_idx is None:
            continue
        neighbors = [scores[group.name]]
        if idx > 0 and ordered[idx - 1][1] is not None:
            neighbors.append(scores[ordered[idx - 1][0].name])
        if idx + 1 < len(ordered) and ordered[idx + 1][1] is not None:
            neighbors.append(scores[ordered[idx + 1][0].name])
        neighborhood_mean = sum(neighbors) / len(neighbors)
        result[group.name] = (1.0 - alpha) * scores[group.name] + alpha * neighborhood_mean
    return result


def collect_group_importance_scores(
    model,
    expert_groups: List[ExpertGroup],
    dataloader,
    device,
    max_batches: int,
    smoothing: float = 0.25,
):
    stats = {}
    handles = []

    for group in expert_groups:
        gate = group.gate
        if gate is None:
            continue
        stats[group.name] = {
            "counts": torch.zeros(len(group.experts), dtype=torch.float64),
            "selected_prob_sum": 0.0,
            "tokens": 0,
            "top_k": _infer_router_top_k(model, group),
        }

        def make_hook(group_name: str):
            def hook(_mod, inp, out):
                logits = out[0] if isinstance(out, tuple) else out
                if not isinstance(logits, torch.Tensor):
                    return
                if logits.shape[-1] != len(stats[group_name]["counts"]):
                    return
                probs = torch.softmax(logits.detach().float(), dim=-1)
                top_k = min(stats[group_name]["top_k"], probs.shape[-1])
                topk_probs, topk_idx = torch.topk(probs, k=top_k, dim=-1)
                flat_idx = topk_idx.reshape(-1).cpu()
                counts = torch.bincount(flat_idx, minlength=probs.shape[-1]).to(torch.float64)
                stats[group_name]["counts"] += counts
                stats[group_name]["selected_prob_sum"] += float(topk_probs.sum().item())
                stats[group_name]["tokens"] += int(topk_idx.numel() // top_k)

            return hook

        handles.append(gate.register_forward_hook(make_hook(group.name)))

    if not handles:
        return {group.name: 1.0 for group in expert_groups}

    model.eval()
    for step, batch in tqdm(enumerate(dataloader), desc="Collecting layer importance..."):
        if step >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

    for handle in handles:
        handle.remove()

    raw_scores = {}
    for group in expert_groups:
        stat = stats.get(group.name)
        if not stat or stat["tokens"] == 0 or stat["counts"].sum().item() == 0:
            raw_scores[group.name] = 1.0
            continue
        freq = stat["counts"] / stat["counts"].sum()
        concentration = float(torch.linalg.vector_norm(freq, ord=2).item() * (len(freq) ** 0.5))
        avg_selected_prob = stat["selected_prob_sum"] / stat["tokens"]
        raw_scores[group.name] = max(concentration * avg_selected_prob, 1e-8)

    return _smooth_scores(expert_groups, raw_scores, smoothing)


def _group_original_params(group: ExpertGroup, linear_names: List[str]) -> int:
    total = 0
    for linear_name in linear_names:
        if hasattr(group.experts[0], linear_name):
            weight = getattr(group.experts[0], linear_name).weight
            total += len(group.experts) * weight.shape[0] * weight.shape[1]
    return total


def allocate_group_reductions(
    expert_groups: List[ExpertGroup],
    linear_names: List[str],
    global_target_reduction: float,
    importance_scores: Dict[str, float],
):
    original_params = {group.name: _group_original_params(group, linear_names) for group in expert_groups}
    total_original = sum(original_params.values())
    if total_original <= 0:
        raise RuntimeError("No compressible expert parameters found for allocation")

    score_sum = sum(max(importance_scores.get(group.name, 1.0), 1e-8) for group in expert_groups)
    total_target = int(round((1.0 - global_target_reduction) * total_original))

    allocations: Dict[str, GroupAllocation] = {}
    running_target = 0
    for idx, group in enumerate(expert_groups):
        importance = max(importance_scores.get(group.name, 1.0), 1e-8)
        weight = importance / score_sum
        if idx == len(expert_groups) - 1:
            target_params = max(1, total_target - running_target)
        else:
            target_params = int(round(total_target * weight))
            running_target += target_params
        target_params = max(1, min(original_params[group.name], target_params))
        target_reduction = 1.0 - (target_params / original_params[group.name])
        target_reduction = min(max(target_reduction, 1e-6), 0.999999)
        allocations[group.name] = GroupAllocation(
            importance=importance,
            weight=weight,
            original_params=original_params[group.name],
            target_params=target_params,
            target_reduction=target_reduction,
        )
    return allocations


def compress_group(
    model,
    group,
    linear_names: List[str],
    dataloader,
    device,
    max_batches: int,
    target_reduction: float,
    whitening: str,
    eps: float,
    tucker_device: str = "auto",
    preserve_expert_dim: bool = False,
):
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
        print(f"Compressing {linear_name}")
        weight_tensor = stack_expert_weights(group.experts, linear_name).to(device)
        k, d_out, d_in = weight_tensor.shape

        rank_cfg = RankSearchConfig(
            target_reduction=target_reduction,
            fixed_r1=k if preserve_expert_dim else None,
        )
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
    parser.add_argument("--preserve-expert-dim", action="store_true", help="Keep the expert-mode Tucker rank fixed to the original number of experts")
    parser.add_argument("--auto-preserve-expert-dim", action="store_true", help="Automatically preserve expert dimension for small-expert MoE layers such as Mixtral")
    parser.add_argument("--layer-allocation", choices=["uniform", "router"], default="uniform", help="How to distribute the compression budget across MoE layers")
    parser.add_argument("--layer-allocation-smoothing", type=float, default=0.25, help="Neighbor smoothing strength for router-based layer allocation")
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
    preserve_expert_dim = args.preserve_expert_dim
    if args.auto_preserve_expert_dim and expert_groups:
        max_num_experts = max(len(group.experts) for group in expert_groups)
        if max_num_experts <= 8:
            preserve_expert_dim = True
            print("[tdmoe] Enabling expert-dimension preservation automatically for small-expert MoE layers")

    linear_names = [n.strip() for n in args.linear_names.split(",") if n.strip()]
    group_allocations = None
    if args.layer_allocation == "router":
        print("[tdmoe] Collecting router-based layer importance scores")
        importance_scores = collect_group_importance_scores(
            model=model,
            expert_groups=expert_groups,
            dataloader=dataloader,
            device=args.device,
            max_batches=args.max_batches,
            smoothing=args.layer_allocation_smoothing,
        )
        group_allocations = allocate_group_reductions(
            expert_groups=expert_groups,
            linear_names=linear_names,
            global_target_reduction=args.target_reduction,
            importance_scores=importance_scores,
        )
        print("[tdmoe] Layer-wise compression allocation:")
        for group in expert_groups:
            alloc = group_allocations[group.name]
            print(
                f"  {group.name}: importance={alloc.importance:.4f} weight={alloc.weight:.4f} "
                f"orig={alloc.original_params} target={alloc.target_params} "
                f"reduction={alloc.target_reduction:.4f}"
            )

    all_results = {}
    for i, group in enumerate(expert_groups):
        print(f"Compressing group {i+1} of {len(expert_groups)}")
        group_target_reduction = args.target_reduction
        if group_allocations is not None:
            group_target_reduction = group_allocations[group.name].target_reduction
        res = compress_group(
            model=model,
            group=group,
            linear_names=linear_names,
            dataloader=dataloader,
            device=args.device,
            max_batches=args.max_batches,
            target_reduction=group_target_reduction,
            whitening=args.whitening,
            eps=args.eps,
            tucker_device=args.tucker_device,
            preserve_expert_dim=preserve_expert_dim,
        )
        if res:
            all_results[group.name] = res

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
        torch.save({"model": model.state_dict(), "results": all_results}, args.save_path)

    perplexity_results = {}
    perplexity_datasets = parse_csv_arg(args.eval_perplexity_datasets)
    if perplexity_datasets:
        eval_log(f"Starting post-compression perplexity evaluation for datasets={','.join(perplexity_datasets)}")
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
        eval_log(f"Exporting compressed model to Hugging Face format at {args.save_pretrained_path}")
        save_hf_checkpoint(model, tokenizer, args.save_pretrained_path)
    if lm_eval_tasks:
        eval_log(f"Starting post-compression lm-eval tasks={','.join(lm_eval_tasks)}")
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
            "layer_allocations": {
                name: {
                    "importance": alloc.importance,
                    "weight": alloc.weight,
                    "original_params": alloc.original_params,
                    "target_params": alloc.target_params,
                    "target_reduction": alloc.target_reduction,
                }
                for name, alloc in (group_allocations or {}).items()
            },
            "perplexity": perplexity_results,
            "lm_eval": lm_eval_results,
        }
        os.makedirs(os.path.dirname(args.eval_report_path) or ".", exist_ok=True)
        with open(args.eval_report_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
