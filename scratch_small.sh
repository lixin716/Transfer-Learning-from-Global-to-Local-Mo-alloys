#!/usr/bin/env bash
set -euo pipefail

# ==========================================================
# MACE scratch training (NO foundation / NO fine-tuning)
# Model size: medium (num_channels=128, max_L=0)
# Now: loop over fold_00 ... fold_09
# ==========================================================

NAME="${NAME:-mace_scratch_small}"

# 10 folds root
DATA_ROOT="${DATA_ROOT:-./data/processed_data}"

# Per-fold filenames (override if needed)
TRAIN_BASENAME="${TRAIN_BASENAME:-train_ref.xyz}"
VALID_BASENAME="${VALID_BASENAME:-valid_ref.xyz}"
TEST_BASENAME="${TEST_BASENAME:-test_ref.xyz}"

DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-8}"
VALID_BATCH_SIZE="${VALID_BATCH_SIZE:-8}"
MAX_EPOCHS="${MAX_EPOCHS:-60}"

ENERGY_WEIGHT="${ENERGY_WEIGHT:-30.0}"
FORCES_WEIGHT="${FORCES_WEIGHT:-30.0}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-6}"

NUM_CHANNELS="${NUM_CHANNELS:-128}"
MAX_L="${MAX_L:-0}"

# 用 REF_* key
ENERGY_KEY="${ENERGY_KEY:-REF_energy}"
FORCES_KEY="${FORCES_KEY:-REF_forces}"

E0S="${E0S:-average}"
RUN_DIR_BASE="${RUN_DIR_BASE:-./runs/${NAME}}"

# choose CLI entrypoint
if command -v mace_run_train >/dev/null 2>&1; then
  RUN_CMD=(mace_run_train)
else
  RUN_CMD=(python -m mace.cli.run_train)
fi

mkdir -p "$RUN_DIR_BASE"
echo "Base run dir: $RUN_DIR_BASE"
echo "Data root: $DATA_ROOT"

# -------------------------
# Loop folds: fold_00 .. fold_09
# -------------------------
for i in $(seq -w 0 9); do
  FOLD="fold_${i}"

  TRAIN_FILE="${DATA_ROOT}/${FOLD}/${TRAIN_BASENAME}"
  VALID_FILE="${DATA_ROOT}/${FOLD}/${VALID_BASENAME}"
  TEST_FILE="${DATA_ROOT}/${FOLD}/${TEST_BASENAME}"

  # 每折单独的 name / log_dir（避免覆盖）
  FOLD_NAME="${NAME}_${FOLD}"
  RUN_DIR="${RUN_DIR_BASE}/${FOLD}"

  echo "=============================="
  echo ">>> Scratch training ${FOLD}"
  echo "Train: $TRAIN_FILE"
  echo "Valid: $VALID_FILE"
  echo "Test : $TEST_FILE"
  echo "Run dir: $RUN_DIR"
  echo "=============================="

  # --------- basic checks ----------
  for f in "$TRAIN_FILE" "$VALID_FILE" "$TEST_FILE"; do
    [[ -f "$f" ]] || { echo "ERROR: file not found: $f" >&2; exit 1; }
  done

  mkdir -p "$RUN_DIR"

  # 训练前硬校验
  python - "$TRAIN_FILE" "$VALID_FILE" "$TEST_FILE" "$ENERGY_KEY" "$FORCES_KEY" <<'PY'
import sys
from ase.io import read

train_fn, valid_fn, test_fn, ekey, fkey = sys.argv[1:6]

def check(fn):
    at = read(fn, index=0, format="extxyz")
    assert ekey in at.info, f"Missing {ekey} in atoms.info (first frame) of {fn}"
    assert fkey in at.arrays, f"Missing {fkey} in atoms.arrays (first frame) of {fn}"
    f = at.arrays[fkey]
    assert f.shape == (len(at), 3), f"Bad {fkey} shape in {fn}: {f.shape}, expected {(len(at),3)}"
    print(f"OK: {fn} has {ekey} and {fkey}; forces shape={f.shape}")

check(train_fn)
check(valid_fn)
check(test_fn)
PY

  # 自动检测 atomic_numbers
  ATOMIC_NUMBERS="$(python - "$TRAIN_FILE" <<'PY'
import sys
from ase.io import iread
nums=set()
for at in iread(sys.argv[1], index=":", format="extxyz"):
    nums.update(map(int, at.numbers))
nums=sorted(nums)
print("[" + ",".join(map(str, nums)) + "]")
PY
)"
  echo "Detected atomic_numbers (${FOLD}): ${ATOMIC_NUMBERS}"

  # ----------------------------------------------------------
  # ✅ Scratch training (NO --foundation_model)
  # ----------------------------------------------------------
  set -x
  "${RUN_CMD[@]}" \
    --name "${FOLD_NAME}" \
    --train_file "${TRAIN_FILE}" \
    --valid_file "${VALID_FILE}" \
    --test_file "${TEST_FILE}" \
    --atomic_numbers "${ATOMIC_NUMBERS}" \
    --num_channels "${NUM_CHANNELS}" \
    --max_L "${MAX_L}" \
    --batch_size "${BATCH_SIZE}" \
    --valid_batch_size "${VALID_BATCH_SIZE}" \
    --max_num_epochs "${MAX_EPOCHS}" \
    --energy_weight "${ENERGY_WEIGHT}" \
    --forces_weight "${FORCES_WEIGHT}" \
    --lr "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --E0s "${E0S}" \
    --energy_key "${ENERGY_KEY}" \
    --forces_key "${FORCES_KEY}" \
    --ema \
    --ema_decay 0.99 \
    --amsgrad \
    --default_dtype float32 \
    --device "${DEVICE}" \
    --seed 42 \
    --log_dir "${RUN_DIR}"
  set +x

  echo ">>> Done ${FOLD}. Logs/models under: ${RUN_DIR}"
done

echo "All folds done. Base logs/models under: ${RUN_DIR_BASE}"
