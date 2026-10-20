# Source attribution and third-party components

## Project source

The application in `app/` is sourced from
[Binsaqr/GRASP_PROJECT](https://github.com/Binsaqr/GRASP_PROJECT), revision
[`af65e105fcb18e0cbf3a1c509d5ac5da13cbca8f`](https://github.com/Binsaqr/GRASP_PROJECT/tree/af65e105fcb18e0cbf3a1c509d5ac5da13cbca8f).
Its application-relative layout, existing launchers, optional workflows and
regression tests are retained. The method diagrams and technical guides derive
from that revision; public-facing prose and links are curated for this layout.
The root launcher delegates to the existing application launcher.

No project reuse license is specified in the source. Do not describe the project
as MIT-licensed or infer a general reuse grant from public availability. GitHub's
[repository licensing guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository)
explains this distinction. This document does not establish the authorship or
permission history of every original file.

## Runtime and development dependencies

`app/requirements.txt` and `app/Dockerfile` declare the dependencies. Versions and
the ROS base tag are not fully pinned. Exact resolved versions, transitive
components and binary notices therefore depend on the build; this table is a
source inventory, not a complete software bill of materials.

| Component | Upstream information | Distribution considerations |
|---|---|---|
| NumPy · SciPy | [NumPy license](https://github.com/numpy/numpy/blob/main/LICENSE.txt) · [SciPy license](https://github.com/scipy/scipy/blob/main/LICENSE.txt) | Retain notices for the installed distributions and their included native libraries. |
| OpenCV / opencv-python | [Python packaging source and license](https://github.com/opencv/opencv-python) | Packaging, OpenCV and bundled native components have separate notices; inspect the resolved wheel or apt packages. |
| Matplotlib · Pillow | [Matplotlib license](https://matplotlib.org/stable/project/license.html) · [Pillow source](https://github.com/python-pillow/Pillow) | Preserve the notices supplied with the resolved distributions. |
| Flask · PyYAML · pytest | [Flask license](https://github.com/pallets/flask/blob/main/LICENSE.txt) · [PyYAML source](https://github.com/yaml/pyyaml) · [pytest license](https://github.com/pytest-dev/pytest/blob/main/LICENSE) | Retain applicable license and copyright notices. |
| PyQt5 / Qt | [Riverbank licensing](https://www.riverbankcomputing.com/software/pyqt/) · [license FAQ](https://www.riverbankcomputing.com/commercial/license-faq) | PyQt is offered under GPL v3 or commercial terms, separately from Qt. Confirm the applicable terms and source obligations before distributing a packaged application. No commercial entitlement is recorded here. |
| pyrealsense2 / librealsense / realsense2_camera | [RealSense SDK](https://github.com/IntelRealSense/librealsense) · [ROS driver](https://github.com/IntelRealSense/realsense-ros) | Check the resolved Python, SDK, ROS and native-library versions and notices separately. |
| pyaer / libcaer | [pyaer distribution](https://pypi.org/project/pyaer/) · [ROS libcaer vendor package](https://index.ros.org/p/libcaer_vendor/) | The Python wrapper and native driver are separate components; retain their respective licenses and applicable source records. |
| ROS Humble / Ubuntu and system packages | [ROS distribution](https://docs.ros.org/en/humble/) · `app/Dockerfile` | A container contains independently licensed packages. Retain an image identity, package inventory and matching notices/source records when distributing an image. |
| three.js | [r149 license](https://github.com/mrdoob/three.js/blob/r149/LICENSE) · [r128 license](https://github.com/mrdoob/three.js/blob/r128/LICENSE) | The dashboard serves the system `libjs-three` asset and has CDN fallbacks. These upstream releases use MIT terms requiring preservation of notices. Check the actual apt version; fallback requests contact external CDNs. |

Dependencies are installed as separate distributions; their terms do not license
this project's source. Distribution of a built image or application requires
review of the actual included versions, licenses, notices and any corresponding
source obligations.

## CI dependencies

The original CI recipe retained at `app/.github/workflows/camera-free.yml` uses
`actions/checkout` v7.0.1 at
`3d3c42e5aac5ba805825da76410c181273ba90b1`:
[upstream source](https://github.com/actions/checkout/tree/3d3c42e5aac5ba805825da76410c181273ba90b1)
and [MIT license](https://github.com/actions/checkout/blob/3d3c42e5aac5ba805825da76410c181273ba90b1/LICENSE).
The action is intended for GitHub Actions; its source is not bundled with the
application. The retained recipe is a reference, not active automation in this
repository layout.

The CI test image obtains Node from `node:22-bookworm-slim`, copying the bundled
license to `/usr/local/share/doc/node/LICENSE`. See the
[official image recipe](https://github.com/nodejs/docker-node/blob/main/22/bookworm-slim/Dockerfile)
and [Node license](https://github.com/nodejs/node/blob/v22.x/LICENSE). This is a
test-only dependency. Retain the resolved image identity and complete notices if
redistributing that test image.

## Optional model workflows

Existing optional YOLO, Ultralytics and SAM2 entrypoints require separately
provided software or model assets. Their presence does not supply a license for
weights, datasets or checkpoints. No model weights or raw capture dataset are
included in this source package. Before using or redistributing these assets,
record their exact identity, origin, terms and required attribution; do not
apply one implementation's license to another implementation or its weights.
