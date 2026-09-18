# Weights

The trained adapter lives with the run that produced it:

- `runs/run-final/adapter.pt` — the shipped open-Jev LM adapter (LoRA r=16 on
  `Qwen/Qwen2.5-0.5B-Instruct`; 2,162,688 trainable parameters; trained 400 steps in 89 minutes
  on six vCPUs, no GPU). sha256 `e49b717438fa54ea` (first 16 hex).

Hugging Face mirrors (created by `hf/upload_model.py`, run with your own token):

- model: `https://huggingface.co/{HF_USER}/openjev-0.5b`
- dataset: `https://huggingface.co/datasets/{HF_USER}/openjev-jev-labelled`

The adapter is a `torch.save`-style state dict of the LoRA parameters only — it is small (~25 MB)
because nothing else was trained. To use it you need the base model from Qwen (Apache-2.0) and
the loader in `openjev/` — see `hf/QUICKSTART.md`.

Earlier snapshots (`runs/snap_0433/`, `runs/snap_0405/`) are kept for the audit trail but are not
the shipped model.
