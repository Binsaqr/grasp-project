# Benchmark User Guide

Commands and runtime paths use `app/` as the working directory unless stated otherwise.

Benchmark Lab compares sensor and feature combinations for research reporting.
It does not change production grasp decisions.

## Start Benchmark Lab

```bash
cd <checkout>/app
./scripts/check_demo_system.sh
./run_final_grasp_demo.sh --with-davis
```

Open:

```text
http://127.0.0.1:8080
```

Then open the Benchmark Lab tab.

## Bottle-Only Workflow

1. Place bottle in front of camera.
2. Open Benchmark Lab.
3. Select Bottle-Only mode.
4. Choose features or presets.
5. Run proof benchmark.
6. Run full benchmark.
7. Export report.

Use Bottle-Only mode when comparing sensor evidence for a visible bottle
without a real gripper/contact trial.

## Feature Mask Proof

Run proof benchmark before trusting a comparison table.

Expected proof behavior:

- Disabled features are inaccessible.
- Missing dependencies are reported.
- Not-applicable features are reported.
- `used_features` contains only features enabled for that configuration.
- `blocked_features`, `missing_dependencies`, `not_applicable_features`,
  `unavailable_features`, and `stale_features` explain why data was not used.
- `feature_leakage_detected` is `false`.
- Production pipeline fields are isolated from benchmark results.

If many configurations look too similar:

1. Run proof benchmark again.
2. Open the result JSON/table.
3. Check `used_features`.
4. Check `feature_leakage_detected`.
5. Check whether the scenario is bottle-only, where contact/epsilon metrics may
   be not applicable.

## Full Benchmark

Run full benchmark after proof benchmark passes.

Recommended steps:

1. Keep the bottle visible and stable.
2. Confirm RealSense status is fresh.
3. Confirm DAVIS is active or clearly unavailable.
4. Confirm point cloud is current if point-cloud features are selected.
5. Confirm the Sync card says `software_timestamp` and that selected
   modalities are current enough for benchmarking.
6. Select the comparison presets/features.
7. Start the full run.
8. Wait for completion.
9. Export HTML/CSV/tables/charts.

## Reading Results

- Bottle-only results are Estimated Performance, not Ground-Truth Accuracy.
- DAVIS events measure motion; DAVIS Events Only cannot claim stationary bottle
  identity by itself.
- Point Cloud Only depends on depth/intrinsics/ROI context.
- Point cloud is asynchronous and may update slower than RealSense. If
  `pointcloud_freshness` is `STALE`, the configuration should be marked stale or
  valid with limitations.
- Sync fields to check in exported JSON: `sync_mode`, `sync_quality`,
  `sync_offset_ms`, `stale_modalities`, `bundle_valid_for_benchmark`, and
  `used_features`.
- Epsilon and force-closure require valid contact geometry.
- Two-contact epsilon is not applicable to vacuum, soft, or three-finger
  embodiments unless a separate validated metric enables it.
- Parallel jaw is the production reference; other embodiments are display or
  benchmark-only unless their release state says otherwise.

## Useful Outputs

Benchmark outputs are written under:

```text
output/benchmarks/
output/final_benchmark/
```

Report-ready exports usually include:

- HTML report.
- CSV results.
- Comparison tables.
- Charts.
- Proof summary JSON.

Use `docs/experiment_results_index.md` when deciding which exported table to
use in a research report.

## Troubleshooting

- Benchmark Lab does not start: confirm the main dashboard is open at
  `http://127.0.0.1:8080`.
- Results look identical: inspect `used_features`, run Feature Mask Proof, and
  check `feature_leakage_detected`.
- Feature toggles not changing results: disabled features should appear under
  blocked/disabled fields and must not be listed in `used_features`.
- Stale data wins ranking: check `result_validity`, `stale_data_count`, and
  `winner_eligible`. Stale configurations should not be treated as clean
  winners.
- Point cloud metrics missing: make sure point cloud is current and depth is
  active.
- DAVIS metrics missing: check the DAVIS Motion tab for event stream status.
- Epsilon missing: verify contact geometry exists; bottle-only mode can mark
  epsilon as not applicable.
- Export missing: wait for the run to complete before exporting.
