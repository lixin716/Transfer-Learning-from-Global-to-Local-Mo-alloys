#!/usr/bin/env bash
set -euo pipefail


FOLDS_ROOT="${FOLDS_ROOT:-./data/processed_lodo/folds}"
ELEMENTS="${ELEMENTS:-}"   # 可选：只训练指定元素（空=全部）。例：ELEMENTS="Zr Ti V"

FOUNDATION_MODEL="${FOUNDATION_MODEL:-medium}"

DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-8}"
VALID_BATCH_SIZE="${VALID_BATCH_SIZE:-8}"
MAX_EPOCHS="${MAX_EPOCHS:-200}"

ENERGY_WEIGHT="${ENERGY_WEIGHT:-30.0}"
FORCES_WEIGHT="${FORCES_WEIGHT:-30.0}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-6}"

ENERGY_KEY="${ENERGY_KEY:-REF_energy}"
FORCES_KEY="${FORCES_KEY:-REF_forces}"

E0S="${E0S:-average}"

RUNS_ROOT="${RUNS_ROOT:-./runs}"

if command -v mace_run_train >/dev/null 2>&1; then
  RUN_CMD=(mace_run_train)
else
  RUN_CMD=(python -m mace.cli.run_train)
fi

want_elem () {
  local e="$1"
  if [[ -z "${ELEMENTS}" ]]; then
    return 0
  fi
  for x in ${ELEMENTS}; do
    [[ "${x}" == "${e}" ]] && return 0
  done
  return 1
}

banner () {
  local msg="$1"
  echo
  echo "===================================================================="
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${msg}"
  echo "===================================================================="
}

[[ -d "${FOLDS_ROOT}" ]] || { echo "ERROR: folds root not found: ${FOLDS_ROOT}" >&2; exit 1; }

shopt -s nullglob
elem_dirs=( "${FOLDS_ROOT}"/* )
shopt -u nullglob

if [[ ${#elem_dirs[@]} -eq 0 ]]; then
  echo "ERROR: no element dirs under: ${FOLDS_ROOT}" >&2
  exit 1
fi

# 统计最终会训练哪些元素（用于 i/total 打印）
selected=()
for d in "${elem_dirs[@]}"; do
  [[ -d "$d" ]] || continue
  e="$(basename "$d")"
  want_elem "$e" || continue
  selected+=( "$e" )
done

if [[ ${#selected[@]} -eq 0 ]]; then
  echo "ERROR: no elements selected under: ${FOLDS_ROOT} (ELEMENTS='${ELEMENTS}')" >&2
  exit 1
fi

TOTAL="${#selected[@]}"
banner "Training will run on ${TOTAL} element(s): ${selected[*]}"

i=0
for d in "${elem_dirs[@]}"; do
  [[ -d "$d" ]] || continue
  elem="$(basename "$d")"
  want_elem "${elem}" || continue

  i=$((i+1))
  banner "START TRAINING (${i}/${TOTAL})  >>>  Element = ${elem}"

  TRAIN_FILE="${d}/train_ref.xyz"
  VALID_FILE="${d}/valid_ref.xyz"
  TEST_FILE="${d}/test_ref.xyz"

  for f in "$TRAIN_FILE" "$VALID_FILE" "$TEST_FILE"; do
    [[ -f "$f" ]] || { echo "SKIP ${elem}: file not found: $f" >&2; continue 2; }
  done

  NAME="${NAME_PREFIX:-mace_ft_lodo}_${elem}"
  RUN_DIR="${RUN_DIR_PREFIX:-${RUNS_ROOT}/${NAME}}"
  mkdir -p "$RUN_DIR"

  echo "[INFO] Element      : ${elem}"
  echo "[INFO] Run dir      : ${RUN_DIR}"
  echo "[INFO] Train/Valid/Test:"
  echo "       - ${TRAIN_FILE}"
  echo "       - ${VALID_FILE}"
  echo "       - ${TEST_FILE}"

  # 训练前硬校验：确保 key 在 info/arrays 里（train/valid/test 都检查）
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

check(train_fn); check(valid_fn); check(test_fn)
PY

 
  ATOMIC_NUMBERS="$(python - "$TRAIN_FILE" "$VALID_FILE" "$TEST_FILE" <<'PY'
import sys
from ase.io import iread

def scan(fn):
    zs=set()
    for at in iread(fn, index=":", format="extxyz"):
        zs.update(map(int, at.numbers))
    return zs

train_fn, valid_fn, test_fn = sys.argv[1:4]
z_tr = scan(train_fn)
z_va = scan(valid_fn)
z_te = scan(test_fn)
z_all = sorted(z_tr | z_va | z_te)

def fmt(zs):
    return "[" + ",".join(map(str, sorted(zs))) + "]"

# 这一行输出给 bash 变量用
print(fmt(z_all))

# 额外日志：写 stderr（不会污染 bash 变量）
print(f"[INFO] Z in train: {fmt(z_tr)}", file=sys.stderr)
print(f"[INFO] Z in valid: {fmt(z_va)}", file=sys.stderr)
print(f"[INFO] Z in test : {fmt(z_te)}", file=sys.stderr)

missing = (z_va | z_te) - z_tr
if missing:
    print(f"[WARN] Z present in valid/test but not in train: {fmt(missing)}", file=sys.stderr)
PY
)"
  echo "[INFO] Using atomic_numbers (train∪valid∪test): ${ATOMIC_NUMBERS}"

  echo "[INFO] Launching training command..."
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

  banner "FINISH TRAINING (${i}/${TOTAL})  <<<  Element = ${elem}"
  echo "[INFO] Logs/models under: ${RUN_DIR}"
done

banner "ALL DONE."
