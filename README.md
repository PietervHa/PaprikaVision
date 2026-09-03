# PaprikaVisionMDE

Vision system for orientation-driven paprika placement. One question per fruit:
**which way does the stem point, and can that answer be trusted enough to act
on?**

Fresh project, but not from scratch — the camera handling, logging, config
loading, per-user maintenance login, database/JSONL result writing and the PLC
trigger server are carried over from VisionSoftwareMDE, because they were the
parts that had already been debugged against real hardware. What was left
behind: OCR (Paddle/Tesseract/text locator), the classifier and template
backends, and the Roboflow workflow path — none of it applies here, and
carrying it would have meant maintaining dead branches.

---

## The idea

Finding the stem is the obvious way to know which end is up, and it breaks on
the fruit you care about — stems snap off in handling. So orientation comes
from **two independent estimators**, and they cross-check each other:

**1. Keypoints.** A YOLO pose model predicts two landmarks per fruit,
`stem_end` (the calyx) and `blossom_end`. The vector between them is the
orientation — full 360°, no ambiguity. Primary estimator.

**2. Shape.** A paprika isn't symmetric along its long axis: the stem end
carries the shoulder and is measurably wider, the blossom end tapers to the
lobed tip. PCA on the fruit silhouette gives the axis; the width profile along
it says which end is the shoulder. **This never looks at the stem**, so it
still works on a fruit that has none.

`fuse()` combines them. Keypoints win the axis; shape can override the *flip*
when the model is unsure which end is the stem but the silhouette clearly
isn't. Disagreements are recorded, never averaged — averaging two directions
170° apart gives you a number confidently perpendicular to both, which is the
worst possible outcome for a placement machine. A flagged disagreement can be
rejected; a plausible wrong angle cannot.

**Standing fruit** is treated as a first-class state, not an error. When the
two landmarks project onto nearly the same point, the long axis is pointing at
the camera, so there is no in-plane rotation to report — and the *visibility*
of `stem_end` distinguishes `standing_stem_up` from `standing_stem_down`.
Returning a plausible-looking angle here would be worse than returning nothing.

**Colour is not a class.** You need colour *robustness*, not colour
*selection*, so there's one class (`paprika`) and the shape estimator
thresholds on saturation — red, yellow, orange and green all cross the same
line while the belt doesn't. See ANNOTATION_SPEC §4 for why splitting by
colour would actively hurt.

---

## Three backends

Set `paprika.backend` in `config/default.yaml`:

| Backend | Needs a model? | Use |
|---|---|---|
| `shape` | No | **Start here.** Segments fruit against the belt, orients from silhouette. Runs today, before a single image is annotated. |
| `pose` | Yes | Production. Handles touching fruit and stemless fruit properly. |
| `simulator` | No | Synthetic fruit rotating at a known 30°/s, camera ignored. For proving the PLC angle convention on a bench. |

All three return an identical contract, so switching is a config edit and
nothing downstream can tell the difference.

`shape` existing is the answer to "build the backend and frontend while waiting
for the dataset": the HMI, the arrow, the angle convention and the PLC
handshake can all be commissioned and signed off now, and swapping in the
trained model later changes one config line.

---

## Getting started

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python -m tools.manage_users add <username>    # maintenance login
python -m backend.main
```

HMI at `http://localhost:5000`. Press `Q` in the terminal to fire a cycle
manually.

`ultralytics`/`torch` are only needed for `backend: "pose"`.

---

## Commissioning order

**1. Lock the camera — before collecting any data.** Set
`camera.manual_settings.enabled: true` and pin exposure, gain and white
balance. Auto white balance wandering during a collection session poisons the
shape estimator (it thresholds on absolute saturation) *and* the training set,
by teaching the model that a paprika's appearance is inherently unstable.
Re-collecting 2000 images costs far more than doing this now.

**2. Tune the belt threshold.**
```powershell
python -m tools.tune_shape --live --frames 12
```
Sweeps `saturation_floor` and writes `fruit | mask | overlay` montages to
`data/debug/tune_shape/`. Pick the value that maximises the stem-end signal,
not the detection count — finding a blob is easy.

**3. Fix the angle convention.** Set `backend: "simulator"`, watch the PLC
against the HMI. Because the true angle is known exactly, any mismatch is
unambiguously a convention error. Then set `paprika.frame.angle_offset_deg` and
`angle_invert`. These are commissioning values, set once per machine — never
patch the convention in code, the next machine will need different numbers.

**4. Collect and annotate.** Read `docs/ANNOTATION_SPEC.md` **before**
labelling. The schema decisions there are the expensive ones: a wrong threshold
costs an afternoon, a wrong keypoint convention costs re-annotating 2000
images.

**5. Validate the export.**
```powershell
python -m tools.dataset_check data/datasets/paprika
```
Catches swapped keypoints, missing visibility flags, angle-coverage holes and
class-balance misses. The swap check matters most: swapped landmarks leave a
dataset that is perfectly self-consistent and trains to a healthy loss, then
produces a model that points every arrow backwards. Nothing in a training
metric will tell you — only a geometric check against the fruit's own shape
will.

**6. Train, then switch backend.**
```powershell
yolo pose train model=yolo11n-pose.pt data=paprika.yaml imgsz=640 epochs=150
```
Point `paprika.pose.model_path` at `best.pt`, set `backend: "pose"`.

---

## PLC interface

The PLC needs an angle, not a verdict. On a placeable fruit:

```
OK;<angle>;<x>;<y>;<pose>;<confidence>\n
OK;137.4;612;388;lying;0.91
```

`<angle>` is already mapped through `paprika.frame`, so the PLC uses it
directly with no conversion.

Anything not placeable — no fruit, standing fruit, uncertain stem end — returns
the plain `NOK` string, **never a number**. That asymmetry is deliberate: if an
angle came back with a confidence attached and the PLC decided whether to trust
it, the trust threshold would end up living in ladder logic, far from the
confidence values that justify it and from anyone who could retune it.
Deciding here keeps the PLC rule trivial — a number arrived, so place;
otherwise recirculate.

Set `trigger.response_format: "okonly"` for bare OK/NOK while first bringing up
a PLC that doesn't parse fields yet.

The placement policy is conservative on purpose. A wrong angle places a fruit
backwards and the error leaves the cell; an honest "don't know" just sends it
round again. The costs aren't symmetric, so the thresholds in
`paprika.policy` aren't either. Start strict, relax with evidence from real
production logs.

---

## Layout

```
backend/
  core/
    camera.py            carried over - threaded capture, reconnect, manual exposure
    paprika_engine.py    detect -> orient -> placement decision -> result contract
    overlay_worker.py    background scan thread feeding the live HMI overlay
    vision.py            cycle entry point
    state.py             thread-safe shared state
    tcp_trigger_server.py  carried over, structured angle response added
    auth.py / db.py      carried over
  detection/paprika/
    orientation.py       THE CORE - keypoints, shape analysis, fusion, pose classes
    pose_detector.py     the three backends
  utils/
    annotate.py          overlay: box, landmarks, orientation arrow
frontend/                FastAPI + HMI (angle dial, confidence meters)
tools/
  dataset_check.py       validate an export before training
  tune_shape.py          calibrate saturation_floor against the real belt
  manage_users.py        carried over
docs/ANNOTATION_SPEC.md  read this before labelling
```

`orientation.py` is where the actual thinking is, and it has no dependency on
any model — which is what let it be verified against synthetic fruit at every
angle, in four colours, including the stemless case where the keypoints guess
the wrong end and the silhouette overrules them.

## Known limits

- The `shape` backend cannot separate two touching paprikas — they segment as
  one blob and the axis lands between them. `pose` is what fixes that.
- `shape` needs a belt markedly less saturated than the fruit. Verify with
  `tune_shape` before relying on it.
- Standing fruit gives no angle by design. If the infeed produces a lot of
  them, that's a mechanical problem to solve upstream, not a vision one.
- Angles are measured in the frame as displayed. If you change
  `camera.flip` or the HMI rotation, re-check step 3.
