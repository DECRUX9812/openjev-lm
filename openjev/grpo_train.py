"""openjev/grpo_train.py -- DeepSeek-style GRPO post-training for the open-Jev LM arm (CPU).

This is the "deepseek harness": GRPO (Group Relative Policy Optimization), the algorithm
behind DeepSeek-R1/R1-Zero, adapted to typed decisions instead of math proofs. Same skeleton:
no critic, group-sampled rollouts, deterministic verifiable reward, group-normalised
advantages, KL penalty against a frozen reference policy. What changes vs R1 is only the
reward: ours is a per-field exact-match score against the Jev teacher answers already stored
in every corpus row (row["jev"]), so the whole loop stays $0 and offline.

Per step:
  1. draw `prompts` rows from the train file
  2. sample `k` completions per prompt from the current policy (temperature decoding)
  3. reward each completion against the stored Jev answer (see reward())
  4. advantage = (r - group_mean) / (group_std + eps)   -- the "group relative" in GRPO
  5. loss = -A * mean_t log pi_theta(y_t|x) + beta * KL(pi_theta || pi_ref)
     KL uses the k3 estimator: exp(logp_ref - logp) - (logp_ref - logp) - 1
  6. AdamW on LoRA params only

Degenerate groups (all rewards identical -> zero advantage -> no gradient) are skipped;
this is the standard dynamic-sampling fix and matters here because a warm SFT policy
makes most prompts all-correct/all-wrong under sampling.

Reference policy: the adapter state captured when GRPO starts (normally the SFT adapter).
It is kept as a small state dict (~2.2M params, ~9 MB) and swapped into the LoRA modules
for no-grad reference passes -- far cheaper than holding a second 0.5B model.

Checkpoints are written in the train_v2 format {"format": 2, "lora": ..., "opt": ...,
"step": ...} plus "ref_lora", so eval_openjev.py and eval_likelihood.py read them
unchanged and --resume works after a kill.

usage:
  python openjev/grpo_train.py --run-name rl1 --init runs/sft/adapter.pt --steps 100
  python openjev/grpo_train.py --run-name rl1 --resume            # continue same run
  python openjev/grpo_train.py --run-name rl1 --init runs/sft/adapter.pt --smoke
"""
from __future__ import annotations

import argparse, json, math, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_lora import apply_lora  # noqa: E402

ROOT = HERE.parent
DATA = ROOT / "data"
RUNS = HERE / "runs"          # same run-dir convention as train_v2.py
DEFAULT_MODEL = str(ROOT / "models" / "Qwen2.5-0.5B-Instruct")

BUCKETS = ("service_lead", "staff_role", "generic_job", "junk")
BOOLS = ("technical_need", "business_buyer", "small_firm_doable",
         "pay_stated", "evergreen_repost")
SCHEMA_KEYS = ("bucket",) + BOOLS + ("fit",)


def rss_gb() -> float:
    try:
        return int(Path("/proc/self/statm").read_text().split()[1]) * 4096 / 1e9
    except Exception:  # macOS etc.
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
        except Exception:
            return 0.0


# ------------------------------------------------------------------ reward
def parse_completion(text: str):
    """Pull the first JSON object out of a completion. Returns dict or None."""
    text = text.split("<|im_end|>")[0]
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    return None
    return None


def expected_from_row(row: dict) -> dict:
    """The teacher answer for a corpus row, normalised to python types."""
    jev = row.get("jev")
    if isinstance(jev, dict) and jev.get("bucket"):
        return {"bucket": jev["bucket"],
                **{b: bool(jev[b]) for b in BOOLS},
                "fit": int(round(float(jev["fit"])))}
    target = row["target"].split("<|im_end|>")[0]
    return json.loads(target)


def reward(comp: dict | None, exp: dict) -> tuple[float, dict]:
    """Verifiable reward: exact-match shaping on the typed answer.

    bucket exact-match            +1.00
    each of 5 booleans            +0.10
    fit closeness                 +0.25 * (1 - |pred - exp| / 4)
    parses + exact schema keys    +0.15
    unparseable                   -0.20 flat, no further credit
    """
    if comp is None:
        return -0.2, {"parsed": 0}
    detail = {"parsed": 1}
    r = 0.0
    if set(comp.keys()) == set(SCHEMA_KEYS):
        r += 0.15
        detail["schema"] = 1
    if comp.get("bucket") == exp["bucket"]:
        r += 1.0
        detail["bucket"] = 1
    bool_hits = sum(1 for b in BOOLS if comp.get(b) == exp[b])
    r += 0.10 * bool_hits
    detail["bools"] = bool_hits
    try:
        fit_err = abs(int(comp.get("fit")) - int(exp["fit"]))
        r += 0.25 * (1.0 - min(fit_err, 4) / 4)
        detail["fit_err"] = fit_err
    except Exception:
        detail["fit_err"] = None
    return r, detail


# ------------------------------------------------------------- model helpers
def lora_state(model) -> dict:
    return {k: v.detach().clone() for k, v in model.state_dict().items()
            if "A.weight" in k or "B.weight" in k}


def swap_lora(model, state: dict) -> dict:
    """Load `state` into the LoRA modules; returns the displaced state."""
    cur = lora_state(model)
    model.load_state_dict(state, strict=False)
    return cur


# -------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--run-name", default="rl1")
    ap.add_argument("--init", default=None, help="adapter to start from (SFT ckpt)")
    ap.add_argument("--steps", type=int, default=100, help="GRPO steps for THIS invocation")
    ap.add_argument("--prompts", type=int, default=2, help="prompts per group step")
    ap.add_argument("--k", type=int, default=8, help="samples per prompt")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.04, help="KL-to-ref coefficient")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new", type=int, default=80)
    ap.add_argument("--micro-batch", type=int, default=4,
                    help="sequences per forward/backward chunk (logits are big)")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--time-budget-min", type=float, default=0.0)
    ap.add_argument("--max-rss-gb", type=float, default=13.0)
    ap.add_argument("--train-file", default=str(DATA / "train_jev.jsonl"))
    ap.add_argument("--resume", action="store_true", help="resume this run's ckpt")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    run_dir = RUNS / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = run_dir / "adapter.pt"
    ledger = (run_dir / "grpo_log.jsonl").open("a")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    n_lora = apply_lora(model, args.rank, args.alpha, dropout=0.0)
    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    if n_train > 50e6:
        sys.exit("REFUSING TO TRAIN: LoRA freeze missing")
    print(f"[load] {time.time()-t0:.1f}s | LoRA layers={n_lora} | trainable={n_train/1e6:.2f}M "
          f"| rss={rss_gb():.2f}GB", flush=True)

    start_step, ref_sd = 0, None
    if args.resume and ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state["lora"], strict=False)
        start_step = int(state["step"])
        ref_sd = state.get("ref_lora")
        print(f"[resume] step {start_step}", flush=True)
    elif args.init:
        obj = torch.load(args.init, map_location="cpu", weights_only=False)
        sd = obj.get("lora") or obj.get("model") or obj
        model.load_state_dict(sd, strict=False)
        print(f"[init] {args.init}", flush=True)
    if ref_sd is None:
        ref_sd = lora_state(model)           # ref = the policy GRPO starts from

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    if args.resume and ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        if state.get("opt"):
            opt.load_state_dict(state["opt"])

    rows = [json.loads(l) for l in Path(args.train_file).read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("prompt") and (r.get("jev") or r.get("target"))]
    print(f"[data] {len(rows)} prompts | beta={args.beta} K={args.k} T={args.temperature}",
          flush=True)

    eos_ids = [tok.eos_token_id]
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int):
        eos_ids.append(im_end)
    rng = torch.Generator().manual_seed(args.seed + start_step)
    t_start = time.time()
    deadline = t_start + args.time_budget_min * 60 if args.time_budget_min else None
    step = start_step
    total = start_step + (2 if args.smoke else args.steps)

    while step < total:
        step += 1
        t_step = time.time()

        # ---------------- rollout: K samples per prompt, one generate call (left-padded)
        idx = torch.randperm(len(rows), generator=rng)[:args.prompts].tolist()
        batch_rows = [rows[i] for i in idx]
        exps = [expected_from_row(r) for r in batch_rows]
        prompt_ids = [tok(r["prompt"], add_special_tokens=False)["input_ids"]
                      for r in batch_rows]
        plen = max(len(p) for p in prompt_ids)
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        left = torch.full((len(batch_rows), plen), pad_id, dtype=torch.long)
        lattn = torch.zeros((len(batch_rows), plen), dtype=torch.long)
        for i, p in enumerate(prompt_ids):
            left[i, plen - len(p):] = torch.tensor(p)
            lattn[i, plen - len(p):] = 1

        model.eval()
        with torch.no_grad():
            gen = model.generate(
                input_ids=left, attention_mask=lattn, do_sample=True,
                temperature=args.temperature, top_p=args.top_p,
                num_return_sequences=args.k, max_new_tokens=args.max_new,
                eos_token_id=eos_ids, pad_token_id=pad_id)
        completions = gen[:, plen:].tolist()          # completion ids only (right-padded)

        groups, stats, n_seq = [], {"parsed": 0, "bucket": 0, "schema": 0}, 0
        for b in range(len(batch_rows)):
            group = []
            for k_i in range(args.k):
                ids = completions[b * args.k + k_i]
                ids = ids[:ids.index(eos_ids[-1]) + 1] if eos_ids[-1] in ids else ids
                comp = parse_completion(tok.decode(ids, skip_special_tokens=True))
                r, det = reward(comp, exps[b])
                stats["parsed"] += det["parsed"]
                stats["bucket"] += det.get("bucket", 0)
                stats["schema"] += det.get("schema", 0)
                group.append((prompt_ids[b] + ids, len(ids), r))   # (seq, comp_len, reward)
                n_seq += 1
            groups.append(group)

        # ---------------- group-relative advantages; skip degenerate groups
        flat = []
        skipped = 0
        for group in groups:
            rs = [r for _, _, r in group]
            mu = sum(rs) / len(rs)
            sd = math.sqrt(sum((r - mu) ** 2 for r in rs) / len(rs))
            if sd < 1e-4:                    # no intra-group signal -> no gradient anyway
                skipped += 1
                continue
            for seq, clen, r in group:
                flat.append((seq, clen, (r - mu) / (sd + 1e-4), r))
        mean_r = sum(r for g in groups for _, _, r in g) / max(n_seq, 1)
        if not flat:
            ledger.write(json.dumps({"step": step, "skipped_all": 1,
                                     "mean_reward": round(mean_r, 3)}) + "\n")
            ledger.flush()
            print(f"[step {step}] all groups degenerate (mean_r={mean_r:.3f}) -- skipped",
                  flush=True)
            continue

        # ---------------- right-aligned batch tensor: [pad | prompt | completion]
        # row i's content sits at [Lmax - seq_len, Lmax); its completion is the last
        # clen tokens -> batch positions [Lmax - clen, Lmax), predicted by logits
        # [Lmax - clen - 1, Lmax - 1).
        seq_lens = [len(seq) for seq, _, _, _ in flat]
        Lmax = max(seq_lens)
        full = torch.full((len(flat), Lmax), pad_id, dtype=torch.long)
        attn = torch.zeros((len(flat), Lmax), dtype=torch.long)
        for i, (seq, _, _, _) in enumerate(flat):
            full[i, Lmax - len(seq):] = torch.tensor(seq)
            attn[i, Lmax - len(seq):] = 1
        pos_ids = (attn.cumsum(dim=-1) - 1).clamp(min=0)   # pads all sit at position 0

        def row_logps(logits, j, gi):
            """Per-token logp of row j's completion in flat[gi] (right-aligned layout)."""
            clen = flat[gi][1]
            lo, hi = Lmax - clen - 1, Lmax - 1
            lp = F.log_softmax(logits[j, lo:hi].float(), dim=-1)
            tgt = full[gi, Lmax - clen:Lmax]
            return lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)

        # ---------------- reference logps (ref adapter swapped in, no grad)
        displaced = swap_lora(model, ref_sd)
        ref_logps = []
        with torch.no_grad():
            for m0 in range(0, len(flat), args.micro_batch):
                mb = full[m0:m0 + args.micro_batch]
                out = model(input_ids=mb, attention_mask=attn[m0:m0 + args.micro_batch],
                            position_ids=pos_ids[m0:m0 + args.micro_batch], use_cache=False)
                for j in range(mb.shape[0]):
                    ref_logps.append(row_logps(out.logits, j, m0 + j))
                del out
        swap_lora(model, displaced)

        # ---------------- policy update: advantage-weighted logp + k3 KL to ref
        model.train()
        opt.zero_grad(set_to_none=True)
        n_used = len(flat)
        agg_kl, agg_loss = 0.0, 0.0
        for m0 in range(0, len(flat), args.micro_batch):
            mb = full[m0:m0 + args.micro_batch]
            out = model(input_ids=mb, attention_mask=attn[m0:m0 + args.micro_batch],
                        position_ids=pos_ids[m0:m0 + args.micro_batch], use_cache=False)
            chunk_loss = None
            for j in range(mb.shape[0]):
                gi = m0 + j
                adv = flat[gi][2]
                plp = row_logps(out.logits, j, gi)
                rlp = ref_logps[gi]
                klen = min(len(plp), len(rlp))
                kl_t = torch.exp(rlp[:klen] - plp[:klen]) - (rlp[:klen] - plp[:klen]) - 1.0
                seq_loss = -(adv * plp[:klen].mean()) + args.beta * kl_t.mean()
                chunk_loss = seq_loss / n_used if chunk_loss is None else chunk_loss + seq_loss / n_used
                agg_kl += float(kl_t.detach().mean()) / n_used
            chunk_loss.backward()
            agg_loss += float(chunk_loss.detach())
            del out
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        dt = time.time() - t_step
        rec = {"step": step,
               "mean_reward": round(mean_r, 4),
               "parse_rate": round(stats["parsed"] / n_seq, 3),
               "bucket_rate": round(stats["bucket"] / n_seq, 3),
               "schema_rate": round(stats["schema"] / n_seq, 3),
               "groups_used": len(groups) - skipped, "groups_skipped": skipped,
               "kl": round(agg_kl, 4), "loss": round(agg_loss, 4),
               "step_s": round(dt, 1), "rss_gb": round(rss_gb(), 2),
               "elapsed_min": round((time.time() - t_start) / 60, 1)}
        ledger.write(json.dumps(rec) + "\n")
        ledger.flush()
        print(f"[step {step}/{total}] r={mean_r:.3f} bucket={rec['bucket_rate']:.2f} "
              f"kl={agg_kl:.4f} {dt:.0f}s rss={rss_gb():.2f}GB", flush=True)

        if step % args.save_every == 0 or step == total:
            torch.save({"format": 2, "lora": lora_state(model), "opt": opt.state_dict(),
                        "step": step, "ref_lora": ref_sd}, ckpt)
            (run_dir / "progress.json").write_text(json.dumps(
                {"step": step, "mean_reward": mean_r}, indent=1))
            print(f"  [save] step {step}", flush=True)
        if rss_gb() > args.max_rss_gb:
            torch.save({"format": 2, "lora": lora_state(model), "opt": opt.state_dict(),
                        "step": step, "ref_lora": ref_sd}, ckpt)
            print(f"[guard] rss cap hit -- checkpointed at step {step}", flush=True)
            break
        if deadline and time.time() > deadline and step < total:
            print(f"[budget] hit {args.time_budget_min}min at step {step}/{total}", flush=True)
            break
        if args.smoke:
            break

    cfg = {"model": args.model, "run_name": args.run_name, "algo": "grpo",
           "steps_done": step, "beta": args.beta, "k": args.k,
           "temperature": args.temperature, "lr": args.lr,
           "init": args.init, "wall_min": round((time.time() - t_start) / 60, 1)}
    (run_dir / "grpo_config.json").write_text(json.dumps(cfg, indent=1))
    print(f"[done] {json.dumps(cfg)}", flush=True)


if __name__ == "__main__":
    main()
