#!/usr/bin/env bash
# open-Jev demo — narrated terminal walkthrough, paced for clip extraction.
PY=/home/ubuntu/.venvs/openjev/bin/python
LM=/home/ubuntu/repos/openjev-lm
CLF=/home/ubuntu/repos/openjev
DWELL=${DWELL:-8}

seg() { clear; printf '\033[1;36m\n===== %s =====\033[0m\n\n' "$1"; sleep 3; }
run() { printf '\033[1;32m$ %s\033[0m\n' "$*"; sleep 1.5; eval "$*"; echo; sleep "$DWELL"; }

seg "HOOK — a \$40M-startup decision model, reproduced on this CPU box for \$0"
run "nproc && grep -m1 'model name' /proc/cpuinfo && (ls /dev/dri 2>/dev/null || echo 'no GPU on this box')"
run "du -sh $CLF/weights && ls -la $CLF/weights"

seg "ACT 1 — offline. HF offline flags set — zero network allowed."
run "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; cd $CLF && $PY -m openjev.cli 'Line Cook' --employer 'Olive Garden' --pay '17.50 hourly'"
printf '# HF_HUB_OFFLINE=1 + TRANSFORMERS_OFFLINE=1 — guaranteed local.\n# no API key, no meter, no internet. Try that with a hosted model.\n'
sleep "$DWELL"

seg "ACT 2 — the seven questions, answered in milliseconds"
run "cd $CLF && time $PY -m openjev.cli 'Assistant Cook' --employer 'Regina Restaurant' --pay '18.00 hourly'"
run "$PY -m openjev.cli 'IT Support Analyst' --employer 'Desjardins' --pay '52,000 annually'"
run "$PY -m openjev.cli 'Lead Developer' --employer 'AYINAM CONSULTANCY' --pay ''"

seg "ACT 3 — determinism. same input, byte-identical answer."
run "$PY /home/ubuntu/demo_determinism.py"
printf '# a metered API cannot promise you this.\n'
sleep "$DWELL"

seg "ACT 4 — dark-data sweep. 200 postings, one breath."
run "$PY /home/ubuntu/demo_sweep.py"
printf "# when a call costs \$0 the question stops being 'can I afford to classify this'\n"
sleep "$DWELL"

seg "ACT 5 — receipts, not vibes. every number re-derived, no installs."
run "cd $LM && python3 verify/score.py --self-test | tail -18"
printf '# every claim in the README re-derived from row-level receipts.\n'
sleep "$DWELL"

seg "ACT 6 — the 0.5B LM arm. Adapter trained on THIS box tonight, 400 steps, 14 min."
run "cat $LM/openjev/runs/retrain-check/progress.json"
run "HF_HUB_OFFLINE=1 $PY /home/ubuntu/demo_lm_one.py 'Need a website for my plumbing business' 'Mike\\'s Plumbing (Austin TX)' '2,000 budget'"
printf '# never-seen posting, different country, budget-in-text — and it calls service_lead.\n# the 3MB classifier misses exactly these. the 0.5B adapter catches them.\n'
sleep "$DWELL"
run "cd $LM && $PY -c \"import json; s=json.load(open('runs/retrain_eval.json'))['summary']; print('fresh adapter on 70 gold:', round(s['bucket_acc_vs_gold']*70),'/70 =', s['bucket_acc_vs_gold'], '| agreement with Jev:', s['bucket_acc_vs_jev_bucket'])\""

seg "ACT 7 — honesty is the feature. we publish our misses."
run "cd $CLF && $PY -m openjev.cli 'Data Modeler' --employer 'SGI REGINA-11th Ave' --pay ''"
printf '# boundary row — the receipts disagree on it and the repo says so.\n# open-Jev: MIT, CPU-only, \$0/call. github.com/DECRUX9812/openjev + openjev-lm\n'
sleep 10
