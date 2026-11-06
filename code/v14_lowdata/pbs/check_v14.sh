#!/bin/bash
# INTEGRITY GATE -- run BEFORE reading any accuracy number.
#   ./check_v14.sh smoke        ./check_v14.sh grid
cd "$(dirname "$0")"
MODE="${1:?usage: ./check_v14.sh smoke|grid}"
python3 ../scripts/gate_v14.py "$MODE"
