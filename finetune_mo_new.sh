#!/usr/bin/env bash
set -euo pipefail

# -------------------------
# Base config (shared)
# -------------------------
NAME="${NAME:-mace_ft_mo_new_medium}"

#  10 折目录根路径
DATA_ROOT="${DATA_ROOT:-./data/processed_new_ft}"

# 每个 fold 下默认文件名（如有不同可通过环境变量覆盖）
TRAIN_BASENAME="${TRAIN_BASENAME:-train_ref.xyz}"
VALID_BASENAME="${VALID_BASENAME:-valid_ref.xyz}"
TEST_BASENAME="${TEST_BASENAME:-test_ref.xyz}"

FOUNDATION_MODEL="${FOUNDATION_MODEL:-medium}"

DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-8}"
VALID_BATCH_SIZE="${VALID_BATCH_SIZE:-8}"
MAX_EPOCHS="${MAX_EPOCHS:-200}"

ENERGY_WEIGHT="${ENERGY_WEIGHT:-30.0}"
FORCES_WEIGHT="${FORCES_WEIGHT:-30.0}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-6}"

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
for i in $(seq -w 0 2); do
  FOLD="fold_${i}"

  TRAIN_FILE="${DATA_ROOT}/${FOLD}/${TRAIN_BASENAME}"
  VALID_FILE="${DATA_ROOT}/${FOLD}/${VALID_BASENAME}"
  TEST_FILE="${DATA_ROOT}/${FOLD}/${TEST_BASENAME}"

  # 每折单独的 name / log_dir（避免覆盖）
  FOLD_NAME="${NAME}_${FOLD}"
  RUN_DIR="${RUN_DIR_BASE}/${FOLD}"

  echo "=============================="
  echo ">>> Training ${FOLD}"
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

  # ✅ 训练前硬校验：确保 key 真在 info/arrays 里（train/valid/test 都检查）
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

# ✅ 自动检测所有原子序数
ATOMIC_NUMBERS="$(python - "$TRAIN_FILE" "$VALID_FILE" "$TEST_FILE" <<'PY'
import sys
from ase.io import iread
nums=set()
for fn in sys.argv[1:]:
    for at in iread(fn, index=":", format="extxyz"):
        nums.update(map(int, at.numbers))
nums=sorted(nums)
print("[" + ",".join(map(str, nums)) + "]")
PY
)"
  echo "Detected atomic_numbers (${FOLD}): ${ATOMIC_NUMBERS}"

  set -x
  "${RUN_CMD[@]}" \
    --name "${FOLD_NAME}" \
    --train_file "${TRAIN_FILE}" \
    --valid_file "${VALID_FILE}" \
    --test_file "${TEST_FILE}" \
    --foundation_model "${FOUNDATION_MODEL}" \
    --multiheads_finetuning False \
    --atomic_numbers "${ATOMIC_NUMBERS}" \
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
