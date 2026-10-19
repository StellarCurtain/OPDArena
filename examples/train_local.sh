#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root; override these relative paths as needed.
student_model="${STUDENT_MODEL:-./models/student-base}"
teacher_model="${TEACHER_MODEL:-./models/teacher-base}"
training_data="${TRAINING_DATA:-./data/openr1/train.parquet}"
output_directory="${OUTPUT_DIRECTORY:-./outputs/opd}"
gpu_count="${GPU_COUNT:-1}"
python -m opdarena.train --method opd --model "$student_model" \
  --teacher "$teacher_model" --data "$training_data" --output "$output_directory" \
  --gpus "$gpu_count" "$@"
