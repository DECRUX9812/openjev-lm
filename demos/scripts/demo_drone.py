"""openjev-lm drone demo — a decision layer for a UAV on Pi-class silicon.

Same trained 0.5B adapter, zero retraining: the user-turn keys are swapped
to `action(PROCEED|HOLD|RETURN_HOME|ABORT)` and the model scores each option
by staged likelihood against a scripted SAR mission telemetry stream.
This is a triage/go-no-go gate, not an autopilot — the point is a $0
calibrated decision that works with no uplink and no GPU.
"""
import sys, time
sys.path.insert(0, '/home/ubuntu/repos/openjev-lm/openjev')
import eval_likelihood as E

ACTIONS = ["PROCEED", "HOLD", "RETURN_HOME", "ABORT"]
MODEL = '/home/ubuntu/repos/openjev-lm/models/Qwen2.5-0.5B-Instruct'
ADAPTER = '/home/ubuntu/repos/openjev-lm/openjev/runs/retrain-check/adapter.pt'

def prompt_for(event, telemetry):
    return ("<|im_start|>system\nDecide.<|im_end|>\n"
            "<|im_start|>user\n"
            f"title: {event}\n"
            f"telemetry: {telemetry}\n"
            "action(PROCEED|HOLD|RETURN_HOME|ABORT)<|im_end|>\n"
            "<|im_start|>assistant\n")

# scripted mission: search-and-rescue grid sweep, weather + hw degrade mid-flight
MISSION = [
    ("PREFLIGHT on pad",      "alt=0m batt=100% wind=4m/s sats=14 link=-52dBm motor=31C"),
    ("TAKEOFF + climb",       "alt=40m batt=98% wind=6m/s sats=14 link=-55dBm motor=38C"),
    ("GRID sweep leg 1",      "alt=120m batt=87% wind=9m/s sats=13 link=-61dBm cam=clear"),
    ("WIND GUST band",        "alt=118m batt=79% wind=23m/s sats=13 link=-63dBm motor=55C"),
    ("storm cell passes",     "alt=118m batt=71% wind=8m/s sats=14 link=-60dBm motor=50C"),
    ("TARGET MATCH cam",      "alt=110m batt=64% wind=8m/s sats=14 link=-59dBm cam=vehicle_match conf=0.81"),
    ("ORBIT on target",       "alt=60m batt=55% wind=7m/s sats=14 link=-57dBm cam=locked"),
    ("GNSS DEGRADE",          "alt=95m batt=41% wind=10m/s sats=3 link=-92dBm motor=61C"),
    ("BATTERY critical",      "alt=90m batt=11% wind=11m/s sats=9 link=-78dBm motor=64C"),
    ("MOTOR TEMP critical",   "alt=85m batt=9% wind=12m/s sats=11 link=-75dBm motor=93C"),
]

def decide(model, tok, prefix, stats, prompt):
    cands = [(a, '{"action": "' + a + '"}') for a in ACTIONS]
    v0 = len('{"action": "')
    spans = [(v0, v0 + len(a)) for a in ACTIONS]
    scored, ok = E.score_stage(model, tok, prompt, cands, spans, prefix, stats)
    ranked = E.rank(scored)
    return ranked

def main():
    print('UAV-04  search-and-rescue grid sweep')
    print('decision layer: openjev-lm 0.5B + adapter — running on the drone CPU, no uplink\n')
    model, tok, _ = E.load_model(MODEL, 8, want_lora=True)
    state, step = E.load_lora_state(ADAPTER)
    print(E.apply_adapter(model, state), f'(adapter step {step})\n')
    prefix = E.Prefix(model)
    stats = {"boundary_fallbacks": 0}
    time.sleep(2)

    # safety gate: the model's calibrated margin decides, not just the argmax.
    # confident pick -> take it; unsure -> hold; very unsure -> come home.
    def gate(margin):
        if margin >= 1.25:
            return None            # trust the model's pick
        if margin >= 0.55:
            return 'HOLD'          # low confidence: park it and wait
        return 'RETURN_HOME'       # model is effectively guessing: come home

    log = []
    for i, (ev, tele) in enumerate(MISSION):
        t = time.time()
        ranked = decide(model, tok, prefix, stats, prompt_for(ev, tele))
        sec = time.time() - t
        win, runner = ranked[0], ranked[1]
        margin = win["mean_logp"] - runner["mean_logp"]
        esc = gate(margin)
        act = esc or win["label"]
        log.append((ev, act, margin))
        sev = '!' if any(k in ev for k in ('GUST', 'DEGRADE', 'critical', 'MATCH')) else ' '
        print(f'{sev} T+{i:02d} {ev:<22} {tele}')
        gate_note = f'GATE→{esc}' if esc else 'accepted'
        print(f'     ▶ {win["label"]:<13} margin {margin:5.2f} nats  gate: {gate_note:<13} ACTION: {act}  ({sec:4.1f}s on CPU)')
        time.sleep(1.4)

    print('\nmission log:')
    for ev, act, m in log:
        print(f'   {ev:<24} {act:<13} (margin {m:.2f})')
    print(f'\n{len(log)} autonomous decisions — $0 compute, no uplink, byte-deterministic.')
    print('weights: 26MB adapter on a 0.5B base. trained on this box. github.com/DECRUX9812/openjev-lm')

if __name__ == '__main__':
    main()
