# Troubleshooting

Commands and runtime paths use `app/` as the working directory unless stated otherwise.

For the complete startup flow, see the [operator guide](operator-guide.md) and
[camera setup guide](camera_setup_guide.md).

## RealSense

- Camera not detected: check USB connection, then run `./scripts/check_demo_system.sh`.
- ROS domain mismatch: use `ROS_DOMAIN_ID=67` for ROS2 commands and demo launches.
- Depth missing: verify `/camera/camera/aligned_depth_to_color/image_raw` is publishing.
- Low FPS: confirm the active RealSense profile and CPU thread limits before changing tracking code.
- Stale quality matrix: first hold one feasible bottle steady and allow the
  automatic lock/baseline retry to complete. Use Fresh Lock + Quality only for
  an explicit new baseline after changing targets or when the UI requests it.
- Target lost: keep the bottle within the RealSense view and avoid using contact/depth ROI as full bottle extent.
- Pre-grasp distance: the current hand-mounted defaults are a 160–200 mm ready
  range with a 180 mm nominal camera depth (`MIN_CLOSE_DISTANCE_MM`,
  `MAX_CLOSE_DISTANCE_MM`, and `DESIRED_CLOSE_DISTANCE_MM` can override them).
  This is derived from the approximate 170 mm camera-to-hand Z offset plus a
  10 mm desired gap. If the display still uses 80–100 mm or the legacy
  150–260 mm / 200 mm range, it is showing an old build/profile rather than
  the current hand-mounted readiness fields.
  Close-range tracking and guidance are synthetic/software-tested only and are
  `NOT_PHYSICALLY_VALIDATED`; `READY` is not measured collision clearance or
  proof that a physical grasp is safe.
- Close-range lock procedure: first show the complete bottle farther from the
  camera and wait for `TARGET_LOCKED`/`TRACKING`. Then approach slowly while
  keeping the remembered magenta grasp band in frame. The same lock is carried
  toward 160–200 mm; do not try to acquire a new lock from only a cap, label,
  or shoulder fragment at close range.
- If `identity_tracking_ok=true` and `contact_patch_tracking_ok=false`, the
  same bottle is still locked but closing is intentionally blocked. Hold
  steady; if contact depth does not recover, move back slightly, re-center the
  band, and approach more slowly. A held last-good overlay is an estimate, not
  a current measurement.
- If `identity_tracking_ok=false` or `roi_candidate_reject_reason` is nonempty,
  move back until the full bottle is visible and reacquire it. The existing
  identity-loss counter remains active and is not bypassed.

## DAVIS

- USB permission error `errno=13`: run `bash scripts/check_davis_permissions.sh` to identify the current node and access state.
- Permanent udev fix: run `bash scripts/setup_davis_udev.sh`, then unplug/replug the DAVIS. The rule follows VID/PID `152a:841a` and is not tied to a USB port.
- Temporary chmod is diagnostic only. It applies to the current `/dev/bus/usb/<BUS>/<DEVICE>` node and will be lost after reconnect.
- `pyaer` OK but no events: inspect the DAVIS Motion cards separately:
  `Raw Poll Events` proves the backend received packets, `Global / Object
  Window` shows how many current events exist globally versus inside the
  mapped bottle ROI, and `Event Timestamp` must be `CURRENT`. Global events
  with zero object-window events indicate an ROI/alignment issue, not a dead
  camera. Raw packets with zero global-window events indicate stale timing.
- `dv` or `dv_processing` missing: not fatal if `pyaer` works for this demo.
- GUI not launching: make sure the Qt platform plugin path points to `/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms`.
- RealSense target disappears from DAVIS after grasp: use the main
  `I Have Grasped` control only after the RealSense target is freshly locked.
  The dashboard freezes that last valid DAVIS ROI for local post-grasp motion;
  `Release / End Motion Test` clears it and returns to pre-close monitoring.
- Camera alignment: the default RealSense-to-DAVIS scale/offset is approximate,
  not camera stitching. With both cameras rigidly mounted, record at least four
  matching physical image points in DAVIS APS and RealSense RGB at the intended
  task distance (currently the estimated 160–200 mm range, nominal 180 mm),
  then run
  `scripts/calibrate_davis_realsense_2d.sh correspondences.json`. Do not create
  or copy a homography without measured correspondences. The live mapper rejects
  files with a missing or greater-than-8-pixel reprojection error. The input
  JSON must also record pixel resolutions as
  `davis_frame_size: [346, 260]` and the active
  `realsense_frame_size: [width, height]`; a profile mismatch is rejected.

## Dashboard

- Port already in use: set `PORT=8081` or stop the old dashboard process.
- Old backend holding writer lock: stop duplicate `real_time_live.py --backend-only` processes.
- Stale images: check file age cards in Diagnostics and recompute quality for the current lock.
- DAVIS-adjusted preview unavailable: add `config/davis_to_realsense_homography.json` only after real calibration.

## Sensor Synchronization

- Current sync is software-only. No microcontroller, trigger wire, external
  pulse, or hardware timestamp link is required for normal operation.
- `display_latest` means the dashboard is showing latest sensor outputs side by
  side. It does not mean the values are strictly synchronized.
- If the Sync card shows stale DAVIS, stale point cloud, missing timestamp, or
  offset too large, live quality and Benchmark Lab should treat the affected
  modality as limited or unavailable.
- Point cloud is asynchronous and slower than the RealSense overlay. A stale
  point cloud should be refreshed or excluded from current-quality claims.
- See `docs/sensor_sync.md` for thresholds and DAVIS event windows.

## Benchmark Lab

- Results all look the same: check Feature Mask Proof, inspect `used_features`,
  and run leakage tests from the Benchmark Lab controls if available.
- Bottle-only benchmark: report values as Estimated Performance, not
  ground-truth accuracy.
- Contact/epsilon missing: expected when no gripper/contact trial is running.
- Stale modality: inspect `sync_mode`, `sync_quality`, `stale_modalities`, and
  `bundle_valid_for_benchmark` in the exported results. Stale data should not be
  treated as a clean winner.

## Point Cloud

- Point cloud stale: open the Point Cloud tab, click Refresh Point Cloud, and
  confirm RealSense depth is active.
- Bottle cloud unavailable: check that the bottle is visible and the depth
  stream is healthy.

## Release Candidate Gates

- Gate files live in `output/release/`.
- `READY_WITH_LIMITATIONS` is acceptable for demos when limitations are
  documented and the affected subsystem is not controlling production actions.
- A stale or missing gate should not be treated as a pass. Re-run
  `./scripts/check_demo_system.sh` and refresh the final validation report.
- Unit tests prove code paths and import safety; they do not prove live bottle,
  finger, DAVIS motion, or multi-view reconstruction behavior.

## Dashboard Looks Like Old Debug UI

- Open `http://127.0.0.1:8080`.
- Keep the dashboard in Presentation mode for the final demo.
- Use the top release tabs: Operator, DAVIS Motion, Grasp Quality, Point Cloud,
  Multi-View Scan, Benchmark Lab, Embodiment, Diagnostics.
- Raw RGB/depth and long debug views are intentionally under Diagnostics or
  Engineering mode.
