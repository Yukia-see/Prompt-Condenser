#!/bin/sh
# Rebuild after changing the Python reference (keywords, prompts, instructions):
#   1. regenerates the data tables inside src/prompt_condenser.rs from reference/prompt_condenser_dual_pass.py
#   2. regenerates tests/parity.json from the same Python module
# then run: cargo test --offline
set -e
cd "$(dirname "$0")/.."
python3 tools/gen_data.py > tools/data.rs.txt
cat tools/part_head.rs tools/data.rs.txt tools/part_logic.rs tools/part_tests.rs > src/prompt_condenser.rs
python3 tools/gen_parity.py > tests/parity.json 2>/dev/null
