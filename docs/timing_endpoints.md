# Software timing endpoints

Commands and runtime paths use `app/` as the working directory unless stated otherwise.

`DavisMotionAnalyzer.update()` records `software_timing` in the motion report
and its compatibility evaluation output. These are software processing
intervals, not measured physical slip delays.

All interval objects identify their start/end endpoints, clock domain, endpoint
timestamps in seconds, duration in milliseconds, status, and unavailable/invalid
reason. Only finite, ordered endpoints in the same explicit clock domain can
produce `MEASURED` intervals. Missing timestamps remain `UNAVAILABLE`; clock
resets, device-counter rollover, nonfinite values and cross-clock comparisons
produce `INVALID` intervals with no numeric duration. A recorded zero duration
requires two observed, equal timestamps.

The host monotonic endpoints are:

| Endpoint | Meaning |
| --- | --- |
| `analysis_epoch_start` | Analyzer start or its latest baseline reset. Stage-only evidence resets retain this origin, matching the existing analyzer reference. |
| `update_entry` | Entry to the current update, before polling. |
| `last_nonempty_backend_poll_return` | Host stamp immediately after `device.get_event()` returns the last nonempty event packet consumed in this update. This is application receipt, not device, bus, kernel or hardware arrival. |
| `inference_complete` | Classification, boundary evidence fusion and validity gates have completed. |
| `first_accepted_slip_label` | First completed inference accepted by the existing usable-for-grasp gate as incipient or gross slip since the last motion-evidence reset. |
| `report_payload_ready` | Decision payload assembled in Python memory, before file serialization or transport. |

The last-packet intervals concern one current packet endpoint, not all events in
the rolling analysis window. An update using buffered evidence can have a
measured processing interval and unavailable packet-receipt intervals. Missing
receipt metadata is not replaced with a later analyzer timestamp, device event
time or an estimated wall-clock timestamp. `application_receipt_endpoint_basis`
explicitly identifies `device_get_event_return`; hardware arrival is unmeasured.
`analysis_epoch_index` increments when motion evidence is
reset; first-label and first-report endpoints are then cleared.

`slip_detection_latency_ms` retains its previous numeric behavior. Its new basis
field explicitly identifies **analysis-epoch start to the first accepted slip's
pre-inference window reference**. That legacy reference is sampled before
classification, so it is distinct from the new actual inference-completion
endpoint. Neither value measures physical slip onset. Scoring, decision rules
and thresholds are unchanged.

`physical_onset_to_alert_ms` remains unavailable until independent onset labels
and a validated mapping to the alert clock exist. `browser_display_latency_ms`
remains unavailable because browser receipt and render endpoints are not
instrumented. A report-ready endpoint does not measure a displayed warning,
hardware acquisition latency, physical onset delay or slip lead time.

Camera-free regression command:

```bash
python -m pytest tests/test_latency_endpoints.py tests/test_davis_motion_estimation.py tests/test_davis_closeup_monitoring.py tests/test_davis_fixed_center_roi.py -o addopts='' -q
```

The tests inject deterministic clocks and synthetic packets. They validate the
software contract and existing inference integration, not sensor performance.
