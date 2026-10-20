# Offline clock correspondence diagnostics

Commands and runtime paths use `app/` as the working directory unless stated otherwise.

`scripts/estimate_clock_mapping.py` fits an affine relationship between two
explicitly declared clocks using supplied paired correspondences. It reports
offset, dimensionless rate ratio, drift in parts per million, residuals and
conditional regression uncertainty for each explicit clock epoch.

The existing `sync/sensor_sync.py` bundles latest software status timestamps and
diagnoses skew. It does not fit native clocks. This offline tool does not change
that runtime, live decisions or recorded timestamps, find event correspondences,
repair resets, calibrate sensor clocks, or establish physical latency.

## Run

```bash
python scripts/estimate_clock_mapping.py correspondences.json --output output/clock-diagnostics.json
python -m pytest tests/test_clock_mapping.py -o addopts='' -q
```

To extract **host receipt associations only** from a verified raw archive:

```bash
python scripts/estimate_clock_mapping.py --recording output/raw-trial/dashboard --stream davis_poll --output output/clock-diagnostics.json
```

`--recording` and a correspondence file are mutually exclusive; archive mode
requires `--stream davis_poll`, `rgb`, `depth` or `camera_info`. The archive reader
verifies the complete manifest, configuration/revision snapshots and every raw
array before extraction. Reports retain the manifest SHA-256, selected stream,
all used pairs, recorded native clock metadata, unavailable observation reasons,
drop records and source-sequence gaps. Output cannot be placed inside the input
archive or overwrite one of its files through an alias.

For DAVIS, each nonempty integer event packet contributes its **last native event
timestamp** associated with the recorded host receipt nanoseconds. Its recorded
normalization epoch and optional native epoch ID remain explicit. A normalization
epoch describes the recorded host mapping segment; it does not establish a
physical device boot. Empty polls contribute no timestamp or pair and remain
counted as unavailable. Floating native event arrays are unavailable even when
their displayed values appear integral, because original integer precision
cannot be established by rounding them. Decreasing counters or invalid/missing
clock metadata block the affected segment until a new recorded epoch is
available; the extractor does not bridge unknown/reset segments. Duplicate
timestamps are disclosed as unavailable distinct correspondences.

For ROS streams, extraction requires an explicitly documented native clock
domain and `native_clock.epoch_id`, plus an exact matching integer
`header_stamp_sec * 1000000000 + header_stamp_nanosec`. The original header parts
are retained. Current captures declare `ros_header_unspecified` without a native
epoch; archive mode returns an explicit unavailable result for them and never
supplies a clock identity or epoch. A researcher can provide a separate
correspondence file when an independently documented clock/epoch is available.

The target epoch in archive mode is scoped only to verified observed monotonic
continuity within that archive. It cannot join host boots or archives. Missing
receipt endpoint metadata remains explicitly unrecorded. Archive mode always
emits `host_receipt_association`, never `simultaneous_event`, and retains the
acquisition/transport/queue-delay limitation. An unavailable result exits 0 with
its reasons; corruption or invalid CLI input exits 2. No archive bytes or live
clock policy are changed.

Exit 0 means the input contract was processed, including epochs with too few
pairs or queries outside the permitted range. Inspect every epoch/query status.
Exit 2 means invalid JSON or a contract error. The CLI reuses the existing strict
evaluation JSON reader: duplicate keys, nonfinite values, malformed JSON and
inputs larger than 64 MiB are rejected. The output must not overwrite the input.

## Correspondence contract

The input is a JSON object with this shape. The values below are synthetic
contract examples, not measurements:

```json
{
  "schema_version": "grasp_clock_correspondences_v1",
  "source_clock": {"domain": "device-A", "unit": "us"},
  "target_clock": {"domain": "host-monotonic", "unit": "ns"},
  "residual_alert_ns": 1000,
  "epochs": [{
    "epoch_id": "mapping-epoch-0",
    "source_epoch_id": "device-A-boot-0",
    "target_epoch_id": "host-boot-0",
    "correspondence_kind": "simultaneous_event",
    "provenance": {
      "method": "synthetic_fixture",
      "reference": "test-only://paired-events"
    },
    "pairs": [
      {"pair_id": "p0", "source_timestamp": 0, "target_timestamp": 1700000000000000017},
      {"pair_id": "p1", "source_timestamp": 1000000, "target_timestamp": 1700000001000050017},
      {"pair_id": "p2", "source_timestamp": 2000000, "target_timestamp": 1700000002000100017}
    ]
  }],
  "queries": []
}
```

Clock domains must identify the actual source clock. Supported units are `ns`,
`us`, `ms` and `s`. Native timestamp fields must be exact signed 128-bit JSON
integers; booleans and floating-point timestamps are rejected. Choose an exact
integer tick unit that preserves the source representation. Do not round a
recorded floating-point timestamp or relabel a clock merely to satisfy this
contract. For ROS seconds/nanoseconds, retain the original fields and construct
an exact nanosecond integer using integer arithmetic.

All timestamps and origins remain integers in input/report JSON. Consumers need
an integer-safe JSON parser; ordinary JavaScript `Number` cannot preserve every
integer above `2**53`. Offset, rate and query estimates also have exact
numerator/denominator representations. Rounded native query estimates are JSON
integers and disclose their rounding error in nanoseconds.

Every epoch requires its own identifier, source and target epoch identifiers,
correspondence kind, provenance method/reference and ordered pairs with globally
unique pair IDs. A device restart, native counter rollover or target clock reset
requires a separate epoch. Both native clocks must increase strictly within an
epoch. The tool rejects decreasing or repeated values; it never sorts pairs,
unwraps counters or joins clock epochs automatically. Declared epoch boundaries
cannot prove that a hidden reset did not occur.

Use the appropriate correspondence kind:

- `simultaneous_event` declares that both timestamps correspond to the same
  physical event through an independent protocol. The tool retains that
  declaration and its evidence reference; it cannot authenticate simultaneity.
- `host_receipt_association` associates a device timestamp with application
  receipt, such as `device.get_event()` return or a ROS callback entry. This
  includes acquisition, transport and queue delay. Its fitted offset/drift is
  an association diagnostic, not a mapping of simultaneous sensor events.

Reports always distinguish `DECLARED_SIMULTANEOUS_EVENTS_UNVALIDATED` from
`HOST_RECEIPT_ASSOCIATION_ONLY`. Both retain
`physical_alignment_established: false` and `applied_to_live_policy: false`.
Do not promote a receipt association into an onset or lead-time measurement.

## Fit and uncertainty

At least three pairs in an epoch are required. Fewer pairs produce
`UNAVAILABLE_INSUFFICIENT_PAIRS` with no model. This is a software fitting
minimum, not a claim of sample adequacy or physical validation.

The estimator subtracts the first native origins and scales the resulting
integer differences to nanoseconds. It computes ordinary least-squares sums,
offset and rate as exact integer/rational expressions. Only human-readable
summaries, residual statistics and standard errors are converted to floating
point. The model explicitly records origins, unit factors and its reconstruction
formula; it avoids fitting a large floating-point epoch intercept.

All declared pairs remain in the fit. An outlier can bias offset and drift;
residual alerts do not remove it or refit thresholds. The optional nonnegative
integer `residual_alert_ns` is an explicit diagnostic threshold, independent of
live policy. Every pair retains its residual and alert result. Without a supplied
threshold, alert status remains null. The report also provides residual RMSE,
maximum absolute residual and residual standard error with `n - 2` degrees of
freedom.

A pair may optionally declare nonnegative integer `source_uncertainty_ns` and
`target_uncertainty_ns`. When both are provided, the report compares its residual
with `target_uncertainty_ns + abs(rate_ratio) * source_uncertainty_ns`. These are
supplied uncertainty declarations, not fitted weights or authenticated bounds.
They do not enter the OLS parameter or query uncertainty calculations.

Reported parameter and query standard errors are conditional on independent,
homoscedastic target errors and exact source timestamps under an affine model.
They do not cover wrong pairing, correlated delay, source uncertainty, model
drift, hidden resets or systematic acquisition/transport delay. A perfect fit
can report zero statistical residual/error while physical uncertainty remains
unknown. These diagnostics are not a physical clock error bound or a hardware
synchronization acceptance test.

## Queries and extrapolation

An optional query must identify the exact source domain, unit, epoch and native
timestamp, for example:

```json
{
  "query_id": "q0",
  "epoch_id": "mapping-epoch-0",
  "source_epoch_id": "device-A-boot-0",
  "source_clock_domain": "device-A",
  "source_unit": "us",
  "source_timestamp": 1000000,
  "max_extrapolation_ns": 0
}
```

Unknown or mismatched clock/epoch declarations fail the contract. Interpolation
includes the observed endpoints and reports `INTERPOLATED_UNVALIDATED`. Outside
the observed source range, the default allowance of zero produces
`UNAVAILABLE_EXTRAPOLATION_LIMIT` with no numeric target estimate. A positive
`max_extrapolation_ns` explicitly permits only that distance beyond the nearest
observed source endpoint and reports `EXTRAPOLATED_UNVALIDATED`. A query never
selects an adjacent epoch. Conditional mean-fit and new-pair prediction standard
errors are shown separately and generally grow away from the sample center;
neither is an extrapolation guarantee.

Keep the correspondence file, actual evidence, pairing protocol, source clock
metadata and revision alongside the report. The report includes a canonical
input SHA-256 for provenance. It cannot establish that correspondences are
correct, recordings are adequate, clocks are physically synchronized, or a slip
warning preceded an independently measured onset.
