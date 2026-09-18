"""openjev-lm router demo — the model picks WHICH ML primitive to run.

JevML (github.com/gamesonrblx/JevML) ships 4 primitives — PCA, MCMC,
text-diffusion, NCA — with a registry designed for a decision model to
read and route between. That's what this does: a natural-language task
comes in, the 0.5B adapter scores each primitive by staged likelihood,
then Node runs the chosen primitive for real. No cloud, no LLM API.
"""
import subprocess, sys, time

sys.path.insert(0, '/home/ubuntu/repos/openjev-lm/openjev')
import eval_likelihood as E

NODE = '/home/ubuntu/node-v22.20.0-linux-x64/bin/node'
HARNESS = '/tmp/JevML/run_primitive.ts'
PRIMS = ["pca", "mcmc", "text-diffusion", "nca"]
# fuller option text gives the likelihood comparison more tokens to work with
OPT = {
    "pca": "pca — find axes of variation, compress a table",
    "mcmc": "mcmc — sample from a density you can only score",
    "text-diffusion": "text-diffusion — restore or refine corrupted text",
    "nca": "nca — simulate local rules on a grid",
}

TASKS = [
    "this sensor table has 8 correlated channels — find the axes that actually matter",
    "I can score candidates but can't sample — draw from a two-peaked posterior",
    "this log line got corrupted in transit — restore it",
    "simulate a field of cells living and dying by neighbor rules",
    "before I cluster this gene table, squeeze it down to the axes that carry variance",
]

def route_prompt(task):
    return ("<|im_start|>system\nDecide.<|im_end|>\n"
            "<|im_start|>user\n"
            f"task: {task}\n"
            "primitive(pca|mcmc|text-diffusion|nca)<|im_end|>\n"
            "<|im_start|>assistant\n")

NULL_TASK = "a task"

def decide(model, tok, prefix, stats, task):
    """PMI-style calibration: score each option on the task AND on a null
    context; the winner is the biggest delta — the option's own prior cancels."""
    v0 = len('{"primitive": "')
    cands = [(p, '{"primitive": "' + OPT[p] + '"}') for p in PRIMS]
    spans = [(v0, v0 + len(OPT[p])) for p in PRIMS]
    on_task, _ = E.score_stage(model, tok, route_prompt(task), cands, spans, prefix, stats)
    on_null, _ = E.score_stage(model, tok, route_prompt(NULL_TASK), cands, spans, prefix, stats)
    merged = []
    for t, n in zip(on_task, on_null):
        merged.append({"label": t["label"],
                       "mean_logp": t["mean_logp"] - n["mean_logp"],
                       "sum_logp": t["sum_logp"],
                       "raw": t["mean_logp"]})
    return E.rank(merged)

def main():
    print('openjev-lm × JevML — the decision model routes to the right ML primitive\n')
    model, tok, _ = E.load_model('/home/ubuntu/repos/openjev-lm/models/Qwen2.5-0.5B-Instruct', 8, want_lora=True)
    state, step = E.load_lora_state('/home/ubuntu/repos/openjev-lm/openjev/runs/retrain-check/adapter.pt')
    print(E.apply_adapter(model, state), f'(adapter step {step})\n')
    prefix = E.Prefix(model)
    stats = {"boundary_fallbacks": 0}
    time.sleep(2)

    for i, task in enumerate(TASKS):
        print(f'━━ TASK {i+1}: {task}')
        t = time.time()
        ranked = decide(model, tok, prefix, stats, task)
        sec = time.time() - t
        win, runner = ranked[0], ranked[1]
        margin = win["mean_logp"] - runner["mean_logp"]
        print(f'   ▶ routes to {win["label"]:<15} (task-vs-null delta {win["mean_logp"]:+.2f}, '
              f'margin {margin:4.2f} over {runner["label"]}, {sec:.1f}s)\n')
        time.sleep(1.2)
        out = subprocess.run(
            [NODE, '--experimental-strip-types', HARNESS, win["label"]],
            capture_output=True, text=True, cwd='/tmp/JevML')
        print(out.stdout.rstrip())
        print()
        time.sleep(1.6)

    print('5 tasks, 4 primitives, zero LLM calls — the adapter is the router.')
    print('openjev-lm (the brain): github.com/DECRUX9812/openjev-lm')
    print('JevML (the toolbox):  github.com/gamesonrblx/JevML')

if __name__ == '__main__':
    main()
