#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../venv/bin/activate"

RECIPE_DIR="${SCRIPT_DIR}/recipes/vllm/minimax-m2.5/find-decode"

for recipe in tp2 tp4 tp8 tep2 tep4 tep8 dp2 dp4 dp8 dep2 dep4 dep8 dep16; do
    echo "Submitting ${recipe}..."
    srtctl apply -f "${RECIPE_DIR}/${recipe}.yaml" || echo "FAILED: ${recipe}"
done
