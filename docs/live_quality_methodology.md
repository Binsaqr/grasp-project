# Live Operational Grasp Quality

Commands and runtime paths use `app/` as the working directory unless stated otherwise.

This document defines the final live operational grasp-quality monitor.

The monitor reports two distinct values:

- **Analytic Quality**: immutable quality from the locked RealSense contact geometry and epsilon/wrench-space analysis.
- **Live Quality**: current operational quality after live RGB-D, point-cloud, DAVIS, and synchronization evidence applies penalties or hard gates.

Live Quality does **not** recompute epsilon and does **not** modify the baseline quality matrix.

## Formula

The baseline is:

```text
analytic_score = normalized locked epsilon/wrench quality
```

The live score is:

```text
raw_live_score = analytic_score - sum(weighted_current_risk_penalties)
live_score = temporal_smooth(raw_live_score)
```

Hard gates bypass smoothing and cap the score immediately.

## Penalty Weights

Configured in `config/live_quality.yaml`:

- alignment: current center/contact alignment deviation
- angle: current grasp-box angle deviation
- depth: current depth deviation from lock
- grip_width: current grip width deviation from lock
- contact: required contact loss or missing bilateral contact
- pointcloud: fresh 3D contact inconsistency
- incipient_slip: DAVIS early slip warning
- gross_slip: DAVIS gross slip risk
- rotation: DAVIS rotation/spin risk
- vibration: DAVIS vibration risk
- target_loss: target lost or invalid tracking state
- uncertainty: stale frames, bad sync, or missing optional sensors

Missing optional sensors reduce confidence through uncertainty. They do not invent a slip/contact risk.

## Hard Gates

The monitor can immediately block the live quality result for:

- target lost
- gross slip
- missing required contacts
- severe rotation
- invalid analytic baseline
- clearly off-surface 3D contacts when the point cloud is fresh
- unacceptable synchronization

Target loss and gross slip bypass smoothing so the dashboard does not coast through a severe risk.

## Smoothing And Recovery

The monitor keeps a small state per writer/session/lock/quality compute identity.

- First sample: no smoothing.
- Normal updates: exponential smoothing.
- Fast score drops: high-alpha smoothing so risk appears quickly.
- Hard gates: no smoothing.
- Recovery after severe risk: slower alpha for several frames.

The analytic baseline remains fixed through all smoothing.

## Sensor Contributions

The dashboard `Why This Score?` cards show:

- RGB: target alignment, angle, and contact evidence
- depth: depth shift and grip-width consistency
- point cloud: fresh 3D contact validity
- DAVIS APS: contextual availability and confidence
- DAVIS events: incipient slip, gross slip, rotation, and vibration
- synchronization: sensor skew, bundle age, and optional-source uncertainty

Each contribution exposes status, confidence, penalty, and a short explanation.

## Outputs

The dashboard publishes:

- `output/live_quality/latest_status.json`
- `output/live_quality/timeline.jsonl`

The status payload includes:

- analytic quality score and label
- live quality score and label
- delta from analytic
- trend
- primary risk
- decision
- next action
- hard gates
- sensor contributions
- recent timeline markers

## Operator Meaning

Analytic Quality answers:

```text
Was the locked grasp geometrically good when computed?
```

Live Quality answers:

```text
Is that locked grasp still operationally safe and stable right now?
```

The two numbers are intentionally different when the bottle moves, contacts are lost, DAVIS detects slip, the point cloud disagrees, or sensors become stale.
