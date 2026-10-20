# Recording a runtime trial

Commands and runtime paths use `app/` as the working directory unless stated otherwise.

Use this procedure for a supervised, hand-operated trial. A high score, completed
archive or passing smoke test does not establish physical grasp validity.

- Retain the source revision, image identity and configuration.
- Freeze the protocol and grouped split plan before inspecting held-out results.
- Use fresh recording-enabled processes and unique, honest trial identities.
- Stop gracefully and retain both archives, logs, failures and unavailable evidence.
- Evaluate only against independent labels and the frozen study.

## 1. Prepare the acquisition environment

Use the Linux computer connected to the D405 and DAVIS346, the Ubuntu 22.04 /
ROS 2 Humble container, USB permissions and Docker Compose v2. Read the
[container prerequisites](container.md). A pip-only environment does not provide
ROS or native sensor drivers. Platform procedures do not establish hardware
commissioning or capture reliability.

Commands below run in Bash from `app/` on the acquisition host. Post-capture
Python commands require the supported project environment. Preserve existing
configuration and data; use a clean source tree and retain its exact revision.

## 2. Freeze the trial plan before held-out data

Choose and record the task/vocabulary, physical outcome and slip-onset definitions,
label protocol, required evidence, grouping axes, configurations/ablations,
durations, repetitions, stopping/exclusion rules and scientific acceptance
thresholds. Fit on train only; select on train/validation; freeze before looking
at test results. No success-rate, residual, lead-time, sample-size or accuracy
target is supplied by this software.

Keep actual identities: `session_id` identifies a real capture session and
`object_id` the physical object. Do not rename the same session/object to evade
a holdout. Each trial gets a new ID/root. All three evaluation splits must be
nonempty, with each claimed grouping axis disjoint; a one-session/one-object
pilot cannot support a session-and-object-held-out claim.

Start with a nonfragile object, secured mounts/cables and a contained surface.
Keep the default loopback dashboard binding. A responsible operator handles the
object manually; do not connect or command a robot/gripper for this checklist.
Stop existing demo processes through their own launch terminals before starting.
There must be one DAVIS owner and one intended RGB-D producer.

## 3. Start one fresh, recording-enabled trial

Replace these example IDs with the actual planned session/object/trial. Use a
new metadata directory and raw root; stop if any command fails:

```bash
set -euo pipefail
trial_id=s01-o01-t01
session_id=s01
object_id=o01
trial_root="output/raw/$trial_id"
result_dir="output/research-runs/$trial_id"
test ! -e "$trial_root"
mkdir -p output/research-runs
mkdir "$result_dir"
git rev-parse HEAD > "$result_dir/source-sha.txt"
git status --porcelain > "$result_dir/source-status.txt"
test ! -s "$result_dir/source-status.txt"
docker compose build grasp-dashboard 2>&1 | tee "$result_dir/build.log"
docker image inspect localhost/grasp-project:release --format '{{.Id}}' > "$result_dir/image-id.txt"
docker compose run --rm --no-deps --name "grasp-recording-$trial_id" \
  -e GRASP_RECORDING_ROOT="/app/output/raw/$trial_id" \
  -e GRASP_RECORDING_TRIAL_ID="$trial_id" \
  -e GRASP_RECORDING_SESSION_ID="$session_id" \
  -e GRASP_RECORDING_OBJECT_ID="$object_id" \
  -e HOST=127.0.0.1 -e PORT=8080 -e ROS_DOMAIN_ID=67 \
  -e USE_YOLO_BOTTLE_TRIGGER=0 \
  grasp-dashboard
```

This foreground command uses the existing Compose USB/network/volume setup and
launcher. It does not replace a running container. Host `output/` retains
`raw/TRIAL/dashboard` and `raw/TRIAL/rgbd`. Container Git metadata may be
unavailable because `.git` is excluded: retain the host SHA/status, exact image ID,
build output and configuration, rather than inventing clean provenance.

**Do not substitute `./run_docker_demo.sh` after exporting recording variables:**
that wrapper does not forward them. Do not attach an unrecorded running stack.
Do not run `record_native_trial.py` alongside this DAVIS owner; native-only
capture does not retain the actual worker/quality journal.

In another host terminal, substitute the same trial ID and retain diagnostics:

```bash
docker exec grasp-recording-s01-o01-t01 ./scripts/check_demo_system.sh \
  > output/research-runs/s01-o01-t01/system-check.txt 2>&1
```

At `http://127.0.0.1:8080`, check advancing RGB/depth/intrinsics, current event
evidence and the locked analytic baseline. Missing or excluded events remain
unavailable. Follow the [manual operator sequence](operator-guide.md#documented-hand-operated-sequence):
steady view/lock → manual approach → operator post-grasp declaration → settle/hold
→ manual open/end. Record phase declarations separately; clicks do not measure
contact, physical closure or slip onset.

Use separate trial IDs for quiet-object, lighting-change and safe camera/global-
motion controls. Keep object-local motion distinct from these confounds. Do not
provoke an unsafe drop. Missing/stale/excluded evidence and corruption are
already covered by software fixtures; offline ablations supply event-disabled
and diagnostic-removal controls without unplugging a live device.

## 4. Stop, retain and verify both archives

Press Ctrl+C in the **foreground acquisition terminal** and wait for shutdown.
Ctrl+C on a separate log viewer does not stop capture. Save all `output/logs/`
files into this trial's result directory before another launch rotates them.
Retain failures and interruptions too.

Both role manifests must say `complete` and pass the reader's integrity checks.
A shutdown message alone is insufficient. Forced termination or failed producer
teardown leaves an incomplete archive: retain it for diagnosis, do not edit its
status, and use a new root for any retake. Completion does not prove loss-free
sensor capture.

On a machine with the supported ROS/Qt runtime and existing project Python
environment, use new report paths outside the archives:

If ROS/Qt is available only in the retained image, first open this offline shell
from the host application directory (`app/`), then run the Python commands below inside it:

```bash
recorded_image_id="$(cat output/research-runs/s01-o01-t01/image-id.txt)"
docker run --rm -it --network none --cap-drop ALL \
  --security-opt no-new-privileges --entrypoint bash \
  -v "$PWD/output:/app/output" "$recorded_image_id" \
  -lc 'source /opt/ros/humble/setup.bash && exec bash'
```

This shell mounts the retained output, without camera devices or the hardware
launcher. Put study JSON/media under that mounted output too.

```bash
python3 scripts/replay_recording.py output/raw/s01-o01-t01/dashboard \
  --output output/research-runs/s01-o01-t01/dashboard-replay.json
python3 scripts/replay_recording.py output/raw/s01-o01-t01/rgbd \
  --rgbd-evaluation --output output/research-runs/s01-o01-t01/rgbd-replay.json
python3 scripts/estimate_clock_mapping.py --recording output/raw/s01-o01-t01/dashboard \
  --stream davis_poll --output output/research-runs/s01-o01-t01/clock-diagnostics.json
```

Synthetic fixtures do not establish real-archive parity or capture throughput.
`verify_grasp_trial.py` verifies bounded snapshots, not these
raw archives. See the [raw contract](raw_recording_contract.md) and
[RGB-D journal](rgbd_trace.md) for failure details.

| Software correctness acceptance | What it does not establish |
|---|---|
| Each replay exits 0 after checking the complete manifest, config/revision hashes and arrays. Inspect counts, unsupported observations, unknown losses, gaps, score, decision, validity and reasons together. | Measured stability, complete sensor coverage or correct physical predictions. Unknown loss accounting is not zero loss. |
| Supported RGB-D replay has `rgbd_evaluation.verified: true` and no mismatches. Preserve any mismatch for investigation. | Full asynchronous-producer recomputation: DAVIS/point-cloud/analytic-completion replies are recorded external inputs. |
| Clock tool processes the contract and discloses usable/unavailable pairs, epochs, fit status and uncertainty. Exit 0 can mean unavailable data. | Physical alignment. Current ROS headers are `ros_header_unspecified`; receipt fits include queue/transport delay. |
| Evaluation validates the supplied study and identical selected observation support across arms. Inspect labelled/paired counts, coverage, abstentions, confusion matrices and reasons. Exit 0 can include zero valid pairs. | A scientific performance threshold, label independence, test blinding or sample adequacy. Required missing evidence must abstain despite a high score. |

## 5. Collect independent evidence, then evaluate

Keep independent raw media or observation records **without prediction
overlays**, actual annotator identity, media hashes, clock domain/units and
uncertainty. Join labels to the recorded `trial_id/observation_id`; retain
unknown/unlabelled cases and disagreements. Declare independence/blinding only
when true. Retain sensor serials, actual intrinsics/depth scale, RGB/depth alignment,
camera/ROI mapping, original calibration images/reference measurements and their
uncertainty. Check geometry/depth against known references under the declared
calibration protocol; a saved configuration is not a validated calibration.
For timing claims, separately observe physical onset and visible alert
availability, and retain independently paired simultaneous-event correspondences
with reset epochs and mapping uncertainty. In-memory report readiness and the
legacy startup-reference latency are not physical warning latency or lead time.

Prepare the existing [annotation/split contracts](offline_evaluation.md) and
[arm declaration](recording_experiments.md#explicit-study-declarations); do not copy
synthetic smoke labels or infer ground truth from predictions. Motion
(`davis_poll`) and quality-decision (`live_quality_inputs`) tasks have different
vocabularies. Supply every actual archive part for every train/validation/test
trial, repeating `--recording`:

```bash
python3 scripts/evaluate_recordings.py \
  --recording output/raw/s01-o01-t01/dashboard --recording output/raw/s01-o01-t01/rgbd \
  --recording output/raw/s02-o02-t01/dashboard --recording output/raw/s02-o02-t01/rgbd \
  --recording output/raw/s03-o03-t01/dashboard --recording output/raw/s03-o03-t01/rgbd \
  --annotations output/research-runs/annotations.json \
  --splits output/research-runs/splits.json \
  --declarations output/research-runs/experiment.json \
  --all-ablations --output output/research-runs/study-report-new.json
```

Those IDs illustrate three genuinely distinct grouped trials, not a prescribed
sample size. Disabled events may deliberately yield zero valid pairs when events
remain required. Sync-diagnostic removal does not remove a calibrated clock
alignment system. Do not tune thresholds from held-out results.

For retained contact geometry, use the existing
[epsilon manifest and controls](epsilon_sensitivity.md), with a fixed declared
constant baseline, constructed invariances and recorded distributions separated:

```bash
python3 scripts/epsilon_sensitivity.py output/research-runs/contacts.json --output output/research-runs/epsilon-new.json
```

Exit 1 retains unavailable/degenerate/failed results (or no samples); exit 2 is an
input/file error. Neither constant agreement nor constructed invariance establishes
physical grasp ranking. Do not manufacture recorded contacts to fill this input.

## 6. Retain a reviewable result package

Retain both unmodified raw archives
(`manifest.json` plus every `.npy`), source/status/image/build/environment/config
provenance, trial plan and phase notes, diagnostics and all logs, independent raw
media/hashes and labels, splits/declarations, clock correspondences, and every
replay/clock/evaluation/epsilon report with command, exit code and stderr.
List missing streams, unknown loss counts, mismatches, exclusions and failed
retakes; do not discard them. Keep raw recordings and unreviewed research results outside the source repository.

Optional YOLO-enabled worker replay is refused without archived model/backend support;
long-session memory/throughput, real-source parity, calibration, physical accuracy,
slip lead time, displayed-alert latency, grasp success and sample adequacy remain
unvalidated. The [synthetic smoke guide](offline_smoke.md) is the reproducible
software exercise; it does not supply physical evidence.

Process ownership remains unchanged: each process owns a separate new role
archive; sinks attach before its first inference; a role finalizes only after
its producers stop. Advanced callers may inject `recording_sink` into the backend;
an explicitly supplied writer remains caller-owned. Recording is off unless all
four recording variables are supplied. No score threshold or actuation behavior
is changed by recording.
