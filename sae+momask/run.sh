#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
# Fragmentation control; no forced CUDA memory reservation.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PYTHON="${PYTHON:-python}"
MODE="${1:-all}"
SEED="${SEED:-3407}"
LSA_WEIGHT="${LSA_WEIGHT:-0.001}"
RUN_DIR="${RUN_DIR:-$HERE/runs/lsa${LSA_WEIGHT}_momask272_s${SEED}}"
mkdir -p "$RUN_DIR"
COMMON=(--data-root "${DATA_ROOT:-/mnt/user/lh3/MotionStreamer-main/humanml3d_272}"
        --evaluator "${EVALUATOR:-/mnt/user/lh3/MotionStreamer-main/Evaluator_272/epoch=99.ckpt}"
        --eval-text-model "${EVAL_TEXT_MODEL:-distilbert-base-uncased}"
        --run-dir "$RUN_DIR" --lsa-weight "$LSA_WEIGHT" --seed "$SEED"
        --workers "${WORKERS:-8}" --device "${DEVICE:-cuda:0}"
        --precision "${PRECISION:-auto}"
        --sample-steps "${SAMPLE_STEPS:-18}" --cfg "${CFG:-4}" --res-cfg "${RES_CFG:-5}")
if [[ -n "${BABEL_DIR:-}" ]]; then COMMON+=(--babel-dir "$BABEL_DIR"); fi
if [[ -n "${VQ_CHECKPOINT:-}" ]]; then COMMON+=(--vq-checkpoint "$VQ_CHECKPOINT"); fi
if [[ -n "${MASK_CHECKPOINT:-}" ]]; then COMMON+=(--mask-checkpoint "$MASK_CHECKPOINT"); fi
if [[ -n "${RES_CHECKPOINT:-}" ]]; then COMMON+=(--res-checkpoint "$RES_CHECKPOINT"); fi
run_stage() {
    local stage="$1" steps="$2" batch="$3"
    local extras=()
    if [[ -f "$RUN_DIR/$stage/latest.pt" ]]; then extras+=(--resume); fi
    if [[ "$stage" == vq && -n "${INIT_VQ:-}" ]]; then extras+=(--init-vq "$INIT_VQ"); fi
    "$PYTHON" -u train.py --stage "$stage" "${COMMON[@]}" --steps "$steps" --batch-size "$batch" \
        --eval-every "${EVAL_EVERY:-10000}" --save-every "${SAVE_EVERY:-2000}" \
        --memory-fraction "${MEMORY_FRACTION:-0.88}" --max-auto-batch "${MAX_AUTO_BATCH:-1024}" \
        "${extras[@]}" 2>&1 | tee -a "$RUN_DIR/${stage}.log"
}
evaluate() {
    "$PYTHON" -u train.py --stage eval "${COMMON[@]}" --repeat "${REPEAT:-20}" \
        --split "${SPLIT:-test}" --num-res-layers "${NUM_RES_LAYERS:--1}" \
        2>&1 | tee -a "$RUN_DIR/eval.log"
}
case "$MODE" in
    check) "$PYTHON" -u train.py --stage check "${COMMON[@]}" ;;
    vq) run_stage vq "${VQ_STEPS:-50000}" "${VQ_BATCH:-auto}" ;;
    mask) run_stage mask "${MASK_STEPS:-100000}" "${MASK_BATCH:-auto}" ;;
    res) run_stage res "${RES_STEPS:-100000}" "${RES_BATCH:-auto}" ;;
    eval) evaluate ;;
    all)
        "$PYTHON" -u train.py --stage check "${COMMON[@]}"
        run_stage vq "${VQ_STEPS:-50000}" "${VQ_BATCH:-auto}"
        run_stage mask "${MASK_STEPS:-100000}" "${MASK_BATCH:-auto}"
        run_stage res "${RES_STEPS:-100000}" "${RES_BATCH:-auto}"
        evaluate ;;
    *) echo 'Usage: bash run.sh {check|vq|mask|res|eval|all}' >&2; exit 2 ;;
esac
