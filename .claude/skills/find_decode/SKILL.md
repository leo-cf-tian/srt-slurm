---
name: find-decode
description: Generate a set of disaggregated vLLM decode experiment recipes to sweep different decode parallelism configs (TP, EP, DP combinations) and find the best one. User describes which configs to generate, the base model/prefill setup, and output directory.
argument-hint: <description of decode configs to generate>
allowed-tools: Bash Read Write Glob Grep Edit
---

Generate a batch of disaggregated vLLM decode experiment YAML recipes and a `submit_all.sh` script based on the user's description.

## How to interpret the user's request

The user will describe:
- **Which decode configs to sweep** using shorthand like TP2, TEP4, DP2, DEP4, etc.
  - **T** = tensor parallel (`tensor-parallel-size`)
  - **E** = expert parallel (`enable-expert-parallel: true`)
  - **D** = data parallel (`data-parallel-size`)
  - The number is the parallelism size
  - Examples: TP2 = tensor-parallel-size 2; TEP4 = tensor-parallel-size 4 + expert parallel; DP2 = data-parallel-size 2; DEP4 = data-parallel-size 4 + expert parallel
- **Base recipe** to derive from (an existing YAML file path), OR model/prefill details
- **Output directory** for the generated recipes (or default to a sensible path)
- Any **overrides** to prefill config, decode config, benchmark settings, etc.

If the user doesn't specify a base recipe, ask which existing recipe to use as a template, or ask for model/prefill details.

## GPU calculation rules

- GPUs per decode worker = product of all parallelism dimensions for that worker
  - TP-only: GPUs = tensor-parallel-size
  - DP-only: GPUs = data-parallel-size
  - TP+DP combos: GPUs = tensor-parallel-size * data-parallel-size
- Expert parallel does NOT add GPUs — it's a flag that distributes MoE experts across existing parallel ranks
- `decode_nodes` = ceil(gpus_per_decode / gpus_per_node)
- All configs use `decode_workers: 1` unless the user specifies otherwise

## Recipe generation rules

For each decode config, generate a YAML file named after the config (lowercase), e.g., `tp2.yaml`, `tep4.yaml`, `dp2.yaml`, `dep4.yaml`.

### Decode vllm_config per config type

**TP configs** (e.g., TP2, TP4):
```yaml
tensor-parallel-size: <N>
pipeline-parallel-size: 1
enable-expert-parallel: false
```

**TEP configs** (e.g., TEP2, TEP4):
```yaml
tensor-parallel-size: <N>
pipeline-parallel-size: 1
enable-expert-parallel: true
```

**DP configs** (e.g., DP2, DP4):
```yaml
tensor-parallel-size: 1
pipeline-parallel-size: 1
data-parallel-size: <N>
data-parallel-rpc-port: 13345
enable-expert-parallel: false
```

**DEP configs** (e.g., DEP2, DEP4):
```yaml
tensor-parallel-size: 1
pipeline-parallel-size: 1
data-parallel-size: <N>
data-parallel-rpc-port: 13345
enable-expert-parallel: true
```

**Combined TP+DP** (e.g., TP2DP2): set both tensor-parallel-size and data-parallel-size accordingly.

### Benchmark concurrencies

Use a wide sweep by default: `"2x4x8x16x32x64x128x256x512x1024x1536x2048x3072x4096x6144x8192"`

**Cap concurrencies** at `gpus_per_decode * max_num_seqs` (default `max_num_seqs` = 1024 unless overridden in the decode vllm_config). Remove any concurrency values above this cap.

### Decode kv-transfer-config

For decode-only benchmarking (finding best decode config in isolation), use:
```yaml
kv-transfer-config: '{"kv_connector": "DecodeBenchConnector", "kv_role": "kv_both"}'
```

For end-to-end disaggregated serving with real prefill+decode, use:
```yaml
kv-transfer-config: '{"kv_connector": "NixlConnector", "kv_role": "kv_both"}'
```

Default to `DecodeBenchConnector` unless the user says otherwise.

### Name field

Set the `name` field to: `<model>-vllm-disagg-<gpu_type>-decode-<config>`, e.g., `minimax-m2.5-vllm-disagg-gb200-decode-tp2`.

## Submit script

Generate a `submit_all.sh` at the **project root** that runs `srtctl apply -f <path>` for each recipe:

```bash
#!/bin/bash
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
RECIPE_DIR="${DIR}/<relative/path/to/recipes>"

for recipe in <list of config names>; do
    echo "Submitting ${recipe}..."
    srtctl apply -f "${RECIPE_DIR}/${recipe}.yaml"
done
```

Make the script executable.

## Reference: complete recipe template

Below is a complete example recipe for reference. Adapt the model, resources, environments, and vllm_config sections as needed based on the user's base recipe or description.

```yaml
name: "minimax-m2.5-vllm-disagg-gb200-decode-tp2"

model:
  path: "minimax-m2.5-fp8"
  container: "v0.19.0"
  precision: "fp8"

dynamo:
  top_of_tree: true
  install: true

setup_script: install-deps.sh

resources:
  gpu_type: "gb200"
  gpus_per_node: 4
  prefill_nodes: 1
  decode_nodes: 1
  prefill_workers: 2
  decode_workers: 1
  gpus_per_prefill: 2
  gpus_per_decode: 2

frontend:
  type: dynamo
  enable_multiple_frontends: false

backend:
  type: vllm
  connector: null

  prefill_environment:
    VLLM_ENGINE_READY_TIMEOUT_S: "3600"
    VLLM_FLASHINFER_ALLREDUCE_BACKEND: "mnnvl"

  decode_environment:
    VLLM_ENGINE_READY_TIMEOUT_S: "3600"
    VLLM_FLASHINFER_ALLREDUCE_BACKEND: "mnnvl"

  vllm_config:
    prefill:
      kv-transfer-config: '{"kv_connector": "NixlConnector", "kv_role": "kv_both"}'
      kv-cache-dtype: "fp8"
      tensor-parallel-size: 1
      pipeline-parallel-size: 1
      data-parallel-size: 2
      data-parallel-rpc-port: 13345
      enable-expert-parallel: true
      safetensors-load-strategy: "prefetch"
      trust-remote-code: true
      no-enable-prefix-caching: true
      max-num-batched-tokens: 16384
      stream-interval: 20

    decode:
      kv-transfer-config: '{"kv_connector": "DecodeBenchConnector", "kv_role": "kv_both"}'
      kv-cache-dtype: "fp8"
      tensor-parallel-size: 2
      pipeline-parallel-size: 1
      enable-expert-parallel: false
      safetensors-load-strategy: "prefetch"
      trust-remote-code: true
      no-enable-prefix-caching: true
      stream-interval: 20

benchmark:
  type: "sa-bench"
  isl: 1024
  osl: 1024
  concurrencies: "2x4x8x16x32x64x128x256x512x1024x1536x2048"
  warmup_prompts: 1
```

## Workflow

1. Read the base recipe (if provided) to understand the full config structure
2. Determine the list of decode configs to generate from the user's description
3. For each config, compute gpus_per_decode, decode_nodes, and concurrency cap
4. Generate all recipe YAML files in the output directory
5. Generate `submit_all.sh` at the project root
6. Summarize what was generated in a table (config name, GPUs, max concurrency)