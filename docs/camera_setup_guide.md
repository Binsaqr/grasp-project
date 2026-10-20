# Camera Setup Guide

Commands and runtime paths use `app/` as the working directory unless stated otherwise.

Use this before running the final dashboard.

Final launch command after the camera checks pass:

```bash
cd <checkout>/app
./run_final_grasp_demo.sh --with-davis
```

## RealSense

1. Plug the Intel RealSense camera into a USB 3 port.
2. Check USB detection:

```bash
lsusb | grep -i -E "realsense|intel|8086"
```

3. Run the project system check:

```bash
cd <checkout>/app
./scripts/check_demo_system.sh
```

Expected:

- RealSense is detected dynamically.
- RGB stream is active when the demo is running.
- Depth stream is active when the demo is running.
- Camera info/intrinsics are available or a clear warning is shown.

If RealSense is not detected:

- Unplug/replug the camera.
- Try another USB 3 port.
- Avoid hubs while debugging.
- Rerun `./scripts/check_demo_system.sh`.

## DAVIS

1. Plug the DAVIS camera into a USB 3 port.
2. Check USB detection:

```bash
lsusb | grep -i -E "davis|ini|inivation|thesycon|152a"
```

3. Run:

```bash
cd <checkout>/app
./scripts/check_demo_system.sh
```

Expected:

- DAVIS is detected dynamically.
- Permissions are OK.
- The dashboard shows APS/events or a clear unavailable reason.
- If no motion is present, the event state may be active with no events.

## DAVIS Permissions

If the check reports permission denied, install the udev rule:

```bash
cd <checkout>/app
./scripts/setup_davis_udev.sh
```

Then:

1. Unplug the DAVIS.
2. Plug it back in.
3. Rerun:

```bash
./scripts/check_demo_system.sh
```

If permissions still do not apply, log out and back in or reboot, then run the
check again.

## Do Not Hardcode USB Paths

Do not hardcode paths such as:

```text
/dev/bus/usb/002/008
```

Those paths change when a camera is unplugged, replugged, or moved to another
USB port. Use dynamic discovery through the project scripts. If multiple
matching cameras are connected, configure the intended device by serial in
`config/hardware.yaml` or the documented environment variables.

## Common Camera Problems

- RealSense not detected: use USB 3, unplug/replug, rerun the system check.
- RealSense topics unavailable: make sure the RealSense launch/backend is
  running through the existing demo script.
- DAVIS permission denied: run `./scripts/setup_davis_udev.sh`, unplug/replug,
  and check again.
- DAVIS no events: move a hand/object in front of the DAVIS; stationary scenes
  can show zero events.
- Camera moved to another USB port: rerun the system check. Do not reuse an old
  `/dev/bus/usb/...` path.
- Stale backend/writer lock: stop the app with `Ctrl+C`, rerun the system
  check, then restart the demo.

## Quick Camera Checklist

- [ ] RealSense detected by `lsusb`.
- [ ] DAVIS detected by `lsusb` if using event-camera motion.
- [ ] DAVIS permissions installed if needed.
- [ ] `./scripts/check_demo_system.sh` passes or reports only understood
      hardware warnings.
- [ ] Final demo launches with `./run_final_grasp_demo.sh --with-davis`.
- [ ] Dashboard opens at `http://127.0.0.1:8080`.
