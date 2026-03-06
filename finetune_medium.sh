#!/usr/bin/env bash
set -euo pipefail

NAME="${NAME:-mace_ft_medium_fold9}"

# ✅ 8:1:1 三分割数据
TRAIN_FILE="${TRAIN_FILE:-./data/processed_data/fold_09/train_ref.xyz}"
VALID_FILE="${VALID_FILE:-./data/processed_data/fold_09/valid_ref.xyz}"
TEST_FILE="${TEST_FILE:-./data/processed_data/fold_09/test_ref.xyz}"

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
RUN_DIR="${RUN_DIR:-./runs/${NAME}}"

# --------- basic checks ----------
for f in "$TRAIN_FILE" "$VALID_FILE" "$TEST_FILE"; do
  [[ -f "$f" ]] || { echo "ERROR: file not found: $f" >&2; exit 1; }
done

mkdir -p "$RUN_DIR"
echo "Run dir: $RUN_DIR"

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
echo "Detected atomic_numbers: ${ATOMIC_NUMBERS}"

# choose CLI entrypoint
if command -v mace_run_train >/dev/null 2>&1; then
  RUN_CMD=(mace_run_train)
else
  RUN_CMD=(python -m mace.cli.run_train)
fi

set -x
"${RUN_CMD[@]}" \
  --name "${NAME}" \
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

echo "Done. Logs/models under: ${RUN_DIR}"
