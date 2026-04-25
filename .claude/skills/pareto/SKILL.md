---
name: pareto
description: Generate a Pareto chart from benchmark-rollup.json results, plotting per-GPU throughput vs per-user decode speed across concurrency levels.
argument-hint: <path-or-dir-or-description-of-dirs> [--gpus N]
allowed-tools: Bash Read Glob Write
---

Generate a Pareto frontier chart from benchmark results.

## Input

The user provides either:
- A path to a `benchmark-rollup.json` file, OR
- A path to a job output directory (e.g., `outputs/1467165/` or `outputs/1467165/logs/`) OR
- A description of multiple job output directories located under `outputs` — may include natural-language filters like "all XpYd directories except cutlassmoe"

If `--gpus N` is provided, use that as the total GPU count. Otherwise, determine GPU count automatically from the `config.yaml` in the job output directory:
- Total GPUs = `(gpus_per_prefill * prefill_workers) + (gpus_per_decode * decode_workers)`
- The `config.yaml` is at `outputs/<job_id>/config.yaml` (one level above `logs/`)
- If `gpus_per_prefill` / `gpus_per_decode` are not set, fall back to: `(gpus_per_node / prefill_workers_per_node)` logic, or ask the user

If GPU count cannot be determined, ask the user.

## Chart specification

Generate a Python script using matplotlib and run it to produce a PNG. Install python dependencies as needed.

**Axes:**
- **X-axis**: "Tokens/s per User" = `1000 / tpot_mean_ms`
- **Y-axis**: "Output Tokens/s per GPU" = `throughput_toks / NUM_GPUS`

**Chart elements:**
- Scatter plot with each concurrency level as a labeled point (`c128`, `c256`, etc.)
- Pareto frontier: connect non-dominated points with a solid colored line (different colors for different configs). Also use lightly faded line for dominated points.
- A point is Pareto-optimal if no other point is better on BOTH axes simultaneously
- Annotate each point with its concurrency value
- Title should include: model name (from `config.model`), topology info if available from config, and ISL/OSL if available. For example: "minimax-m2.5-fp8 GB200 1p1d dep2-dep4 1k/1k" for 1 prefill worker (DEP2=data parallel size 2 with expert parallel enabled), 1 decode worker (DEP4).
- Grid enabled, clear axis labels with units and formula

**Output:**
- Save PNG next to the rollup JSON file (same directory), named `pareto_chart.png` if working on single output. For multiple outputs, save the image named `aggregate-pareto-N.png` and put it under `outputs/aggregate` alongside a textfile `aggregate-pareto-N.txt` with the name of all the output runs as well as a table summary of their datapoints.
- For aggregate charts, auto-increment N by checking existing `aggregate-pareto-*.png` files in `outputs/aggregate/`.

**Summary text (`aggregate-pareto-N.txt`) should include:**
- Header with model, GPU type, and ISL/OSL
- Per-config table with columns: Concurrency, Tok/s/User, Tok/s/GPU, Throughput, TPOT(ms), Pareto flag
- Sorted by concurrency within each config

**Chart styling notes:**
- Use distinct colors AND distinct markers per config (circle, square, triangle, diamond, etc.) for readability
- Bold annotations for Pareto-optimal points, normal weight for dominated points
- Dashed faded line connecting all points in order, solid line for Pareto frontier only
- White edge on scatter markers for visibility against lines
