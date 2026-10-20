#!/usr/bin/env bash
set -euo pipefail

if ! command -v lsusb >/dev/null 2>&1; then
    echo "FAIL: lsusb is not installed."
    exit 1
fi

mapfile -t DEVICES < <(lsusb | grep -Ei 'DAViS|DAVIS|iniVation|INI DAV|Thesycon|libcaer' || true)

if [ "${#DEVICES[@]}" -eq 0 ]; then
    echo "FAIL: No DAVIS/iniVation USB device found."
    exit 1
fi

echo "This is a temporary permission fix. It will be lost after unplug/replug."
echo "Permanent fix: bash scripts/setup_davis_udev.sh"
echo

for line in "${DEVICES[@]}"; do
    bus="$(echo "$line" | awk '{print $2}')"
    dev="$(echo "$line" | awk '{print $4}' | tr -d ':')"
    path="/dev/bus/usb/$bus/$dev"
    echo "Applying temporary chmod to $path"
    sudo chmod a+rw "$path"
    ls -l "$path"
done

echo
echo "Now test:"
echo "  python3 event_postgrasp_monitor.py --enable --duration 3.0 --live-print"
