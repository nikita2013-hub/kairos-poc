#!/usr/bin/env bash
# Full proof-of-concept pipeline. Pass the ASSISTments CSV as $1 to use real data.
set -euo pipefail
DATA=${1:-}
ARGS=(); [ -n "$DATA" ] && ARGS=(--assistments "$DATA" --n-skills 100)
python -m pytest -q tests
python experiments/run_learner_model.py "${ARGS[@]}" --epochs 20
python experiments/run_policy_sim.py "${ARGS[@]}" --n-learners 400 --steps 60
python experiments/run_policy_sim.py "${ARGS[@]}" --belief grkt --grkt-ckpt results/grkt.pt \
       --n-learners 100 --steps 40 --conditions K,K-FAIR --swap-learners 30
python experiments/run_verifier_eval.py generate --n 300 --caa anthropic
echo ">>> Now have two annotators fill label_human in results/verifier_pairs.csv, then run:"
echo "    python experiments/run_verifier_eval.py evaluate --nli hf:microsoft/deberta-large-mnli --extractor anthropic"
python experiments/run_latency.py --turns 200 --caa anthropic --nli hf:microsoft/deberta-large-mnli \
       --extractor anthropic --realise anthropic --belief grkt --grkt-ckpt results/grkt.pt
