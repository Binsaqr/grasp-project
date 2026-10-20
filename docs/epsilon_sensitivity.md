# Offline epsilon sensitivity and constant controls

Commands and runtime paths use `app/` as the working directory unless stated otherwise.

`scripts/epsilon_sensitivity.py` calls the canonical
`grasp_project.realsense.epsilon_wrench.compute_epsilon_quality` implementation.
It never opens a sensor, changes a live score or threshold, or evaluates physical
success. NumPy, SciPy and Matplotlib are existing metric dependencies; there are
no new dependencies. This tool fills the gap between the existing mathematical
consistency report and a reproducible analysis of explicitly supplied contacts.

Run from the application directory (`app/`):

```powershell
python scripts/epsilon_sensitivity.py samples.json --output output/epsilon_sensitivity.json
```

Exit code 0 means all reference rows and expected invariances computed and passed;
1 means the report contains invalid, degenerate or unavailable reference rows,
a failed/unavailable expected invariance, or no supplied samples; 2 means a
malformed manifest/configuration or a file error. A failed expected invariance is retained
as `invariance_passed: false`; inspect controls as well as the reference status.
Reports are deterministic for the same manifest and installed numerical stack.
The manifest path and SHA-256 are included; dependency-version pinning and
artifact verification belong to the recording/provenance workflow.
Duplicate JSON keys, nonfinite JSON numbers and unknown configuration fields are
rejected so that the manifest has one unambiguous interpretation.

The manifest separates two collections. This minimal example is **constructed
geometry only**:

```json
{
  "schema": "grasp_epsilon_samples/v1",
  "parameters": {
    "reference_mu": 0.5,
    "mu_values": [0.0, 0.1, 0.5, 1.0],
    "samples_per_contact": 8,
    "constant_baseline": 0.05,
    "tolerance": 1e-9
  },
  "constructed_samples": [
    {
      "sample_id": "constructed-antipodal",
      "source": {"kind": "constructed", "description": "analytic antipodal control"},
      "contacts": [
        {"point": [-1, 0], "inward_normal": [1, 0]},
        {"point": [1, 0], "inward_normal": [-1, 0]}
      ],
      "object_centroid": [0, 0],
      "object_radius": 1
    }
  ],
  "recorded_samples": []
}
```

Constructed samples get deterministic translation (`[17,-11]` times the source
object radius), uniform scale
(`3x`, including radius), and joint planar rotation (`0.7` radians) controls.
These should preserve epsilon under the normalized planar wrench model. Radius
alone (`2x`) and the first normal (`0.25` radian tilt) are sensitivity probes,
with **no expected invariance**. Friction sweeps are separate from the reference
distribution. Zero-friction or other lower-dimensional hulls are reported as
degenerate, with null epsilon and force closure, rather than valid zero quality.
An origin outside an otherwise valid hull is a valid epsilon of zero and false
force closure. Invalid radius, malformed contacts and zero normals are rejected
before the metric's live fallback behavior can obscure the input problem.
Floating-point precision still limits extreme coordinates or scales; failed or
unavailable invariances remain visible rather than being forced to pass.

For an actual recorded contact sample, keep the same geometry fields, put it
only in `recorded_samples`, and declare its source:

```json
{
  "kind": "recorded",
  "recording_id": "YOUR_RETAINED_RECORDING_ID",
  "artifact": "relative/path/to/epsilon_result.json",
  "artifact_sha256": "SHA256_OF_THE_RETAINED_ARTIFACT_AS_64_HEX_CHARACTERS",
  "contact_source": "EXACT_RECORDED_CONTACT_PROVENANCE"
}
```

Copy contacts, centroid and radius from the retained artifact, not from a
prediction-dependent label or a newly constructed surrogate. The source fields
are required declarations and are **not verified against the artifact by this
tool**. Preserve authoritative planner/locked contacts versus diagnostic visual
contacts in `contact_source`; do not pool them silently. Recorded samples are
flagged when contact sources differ and include distributions per contact source.
They are never expanded with constructed transforms or pooled into the constructed
distribution. Their friction sweeps are recomputations of the contact model,
not new recordings. Duplicate IDs within a collection are invalid.

Each collection reports reference epsilon range, mean, median and population
standard deviation, force-closure counts, and the number of omitted invalid
rows. Near-constant means range is within the declared absolute tolerance; it
is unknown for fewer than two valid rows. A fixed constant baseline (default
`0.05`, explicitly configurable and not fitted) has zero spread and reports
absolute/RMS differences **from computed epsilon**, never prediction accuracy.
The default is an offline comparison parameter and changes no live threshold.

Similar epsilon values under constructed transforms demonstrate the geometry
normalization only. A recorded distribution may also be near constant if contact
construction or object-radius normalization imposes similar geometry. Neither
observation establishes discrimination between real grasp outcomes. Independent
physical labels, diverse retained recordings, held-out grouping, and analysis
of the true acquisition procedure are still required. This report makes no
claim about sample adequacy, hardware latency or grasp success.
