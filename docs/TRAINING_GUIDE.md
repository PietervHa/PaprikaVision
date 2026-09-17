# Training the pose model — start to finish

`ANNOTATION_SPEC.md` decides *what* a label means. This decides *how* to
produce them, turn them into a model, and find out whether the model is any
better than what you already have.

Read the spec first. Everything here assumes it.

---

## 0. Where this sits

The `shape` backend is the commissioning path. It finds the stem by colour,
which works well on red, orange and yellow fruit and cannot work on green,
where stem and flesh share a hue. Everything downstream — the self-check, the
silhouette estimator, the groove axis — exists to paper over that, and on green
it runs out.

The `pose` backend replaces the stem search with a model that has seen
examples. `config/default.yaml` already points at `models/paprika_pose.pt` and
`pose_detector.py` already loads it. The file has simply never existed. This
guide produces it.

You can stop at any stage. `backend: shape` keeps running throughout, and
switching back is one line.

**Benchmark to beat.** Measured on 470 hand-labelled fruit against the `shape`
backend:

| | |
|---|---|
| placed fruit, median error | **2.4°** |
| placed fruit, p90 | **10.2°** |
| within 10° | **90%** |
| placements worse than 45° | **8** (2.3%) |

The first number is the one to beat. The last number is the one that matters —
those are fruit that reached the actuator badly wrong with nothing flagged.

---

## 1. What you are clicking

Two landmarks per fruit, in this order. The order is fixed by the spec and
baked into the model, the exporter and the validator.

| # | name | what to click |
|---|---|---|
| 0 | `stem_end` | the **calyx** — where the stem meets the shoulder. **Not the stem tip.** |
| 1 | `blossom_end` | centre of the blossom scar at the opposite end, in the middle of the lobes |

**The calyx rule is the expensive one to get wrong.** Stems vary in length and
snap off in handling. Annotate the tip and the same fruit produces a different
landmark depending on how roughly it was picked, and the model learns that the
target moves for reasons that have nothing to do with orientation. The calyx is
on every fruit, always in the same anatomical place, including one whose stem
is gone — there is still a scar and a shoulder.

### Visibility — the part that makes stemless fruit work

Which mouse button you use sets the flag:

| button | flag | meaning |
|---|---|---|
| **left** | 2 | I can see this landmark |
| **right** | 1 | it is hidden, but I know where it is — click the position anyway |
| `n` key | 0 | genuinely absent: outside the frame, or I cannot judge at all |

Right-click is not a fallback for "I'm not sure". It is for landmarks that are
definitely there and definitely hidden — the underside of a standing fruit, a
calyx facing away. Skipping those instead of positioning them teaches the model
that every fruit shows both ends, and standing fruit can then never be learned.
`dataset_check` rejects a dataset with no occluded flags at all for this reason.

### The cases

**Lying on its side** — both visible. Calyx at one end, blossom at the other,
roughly along the long axis. Two left clicks.

**Standing, calyx facing you** — you can see the calyx star in the middle.
Calyx: left click at the centre. Blossom: right click at the *same* centre,
because it is directly underneath.

**Standing, lobes facing you** — no calyx visible, lobes converging in the
middle. Blossom: left click at the centre. Calyx: right click at the same
centre.

Both landmarks landing on top of each other is correct here, not a mistake.
It is the signal the engine reads: two predicted landmarks close together means
`standing`, and the *visibility* of `stem_end` decides stem-up from stem-down.

**How to tell the two standing cases apart:** look at what is in the middle. A
green star-shaped calyx, or lobes converging with nothing in the centre.

---

## 2. Calyx pass

```
python -m tools.label_stems data/raw
python -m tools.label_stems data/raw --colour green --limit 200
```

One click per fruit. `s` skips, `u` undoes the last fruit, `q` saves and quits.

Resumable — frames already carrying labels are skipped, so this can be done in
sittings. Labels are written after every frame, via a temporary file, so an
interrupted session cannot destroy the ones already taken.

Labels land in `data/debug/labels/stem_labels.json`, in **full-frame pixel
coordinates**. That is deliberate: the whole point is to compare one version of
the detector against another, and a label tied to a detection index would
silently re-point at a different fruit the first time anything was tuned.

> Needs desktop OpenCV. `opencv-python-headless` has no GUI; the tool says so
> rather than failing with a stack trace.

---

## 3. Blossom pass

```
python -m tools.label_stems data/raw --blossom
```

Walks the fruit that already have a calyx and asks only for the second
landmark. The calyx you marked is drawn as a green ring while you click, so a
blossom placed on the same side as the calyx is obvious immediately.

Made a mess of it? Start that pass over without losing the calyx work:

```
python -m tools.label_stems --reset-blossom
```

Clears every blossom field, keeps all calyx labels, and copies the file to
`stem_labels.backup-<timestamp>.json` first. This is the only command here that
destroys labelling, and it never destroys the only copy.

---

## 4. Verify before spending a training run

```
python -m tools.label_stems --verify
```

Instant, and it catches the mistakes that a training curve would not:

- **Both landmarks on the same side of the centre** — the blossom clicked next
  to the calyx rather than across from it. The most likely systematic error.
- **Landmarks nearly on top of each other** — correct end-on, wrong lying down.
- **A landmark outside its own fruit's box** — a stray click.
- **No occluded flags anywhere** — hidden landmarks were skipped, not
  positioned. Fix this before exporting.

None are automatically wrong. A paprika really can be almost round. They are
prompts to open a handful and look.

---

## 5. Export

```
python -m tools.export_dataset data/raw
python -m tools.export_dataset data/raw --out data/datasets/paprika --val 0.2
```

Merges your clicks over `pre_annotate.py`'s automatic guesses. **A hand label
always wins** — the pre-annotator finds stems by colour, so it is strong
exactly where you are weak and absent exactly where you are strong.

Fruit with no landmark from either source are written with visibility 0 rather
than dropped. A fruit whose calyx cannot be seen is a real case the model has
to handle; deleting those examples would teach it that every fruit has one.

**The split is by frame, never by fruit.** Two fruit in one frame share
lighting, belt, focus and are often the same paprika a moment apart. Splitting
by fruit puts near-duplicates on both sides of the line and gives a validation
score that flatters the model. Frames are assigned by a hash of the filename,
so adding labels later never moves anyone across.

Output:

```
data/datasets/paprika/train/images, train/labels
                      val/images,   val/labels
                      data.yaml
```

---

## 6. Check the dataset

```
python -m tools.dataset_check data/datasets/paprika
```

Errors block training. Warnings do not, but read them — they describe what the
model will be blind to.

Expect these on a first pass, and understand what they mean:

| warning | what it actually means |
|---|---|
| under 15% standing | a normal belt run gives almost none. Put some through deliberately and label them. |
| under 20% stemless | this is the case the model exists for. |
| empty angle buckets | the model will be blind at those angles, and a val split drawn from the same distribution will not reveal it. Hand-rotate fruit to fill them. |

That last one is worth taking seriously. A validation score can look excellent
while the model has never seen a fruit pointing at 200°.

---

## 7. Train

`ultralytics` and `torch` are already in `requirements.txt`.

```
yolo pose train data=data/datasets/paprika/data.yaml model=yolo11n-pose.pt epochs=100 imgsz=640
```

`imgsz=640` matches `paprika.pose.imgsz` in the config. Change one, change both.

---

## 8. Install and switch

```
copy runs\pose\train\weights\best.pt models\paprika_pose.pt
```

Then in `config/default.yaml`:

```yaml
paprika:
  backend: "pose"
```

Weights are gitignored on purpose. **Keep each `best.pt` together with the
dataset version it came from.** A model whose training data you cannot identify
is one you cannot debug, and you will want to go back.

The model loads lazily and logs clearly if the path is wrong or the file will
not open, so a mistake here fails loudly rather than quietly falling back.

---

## 9. Score it — the step that decides everything

```
python -m tools.label_stems data/raw --score --worst 15
```

Same labels, same command, now measuring the pose backend. Compare against the
`shape` numbers in section 0.

The output gives error by verdict and by estimator, then the number that
matters on its own: how far out the fruit the machine actually **placed** were.
`--worst` names the worst offenders by filename so they can be opened and
looked at, which is how every fix in this project that stuck was found.

**What to expect, honestly.** The first model will probably lose to `shape` on
red fruit, where colour-based stem finding is already excellent. Score green
alone — that is where `shape` has no good route and where the model should win.
If it does, the answer may be to run pose on green and shape on red rather than
to pick one.

If it loses everywhere, nothing is lost: `backend: shape` is one line back, and
you now have a labelled dataset that makes the next attempt cheap.

---

## Recovery

| situation | command |
|---|---|
| labels do not match at scoring time | `python -m tools.label_stems data/raw --repair` |
| blossom pass done wrong | `python -m tools.label_stems --reset-blossom` |
| labels look wrong after a reset | rename `stem_labels.backup-*.json` over `stem_labels.json` |
| model misbehaving | set `backend: "shape"` in the config |

`--repair` recomputes `centroid_xy` and `true_angle_deg` from the frames. It
exists because an early version of the labeller stored the centroid with the
bbox offset applied twice, which put it outside its own fruit and made every
label unmatchable. Clicks were never affected. It is idempotent.

---

## Things that are true and easy to forget

**Stability is not accuracy.** Most numbers this project produced before the
labels existed measured how much an estimate *moved*, not how *right* it was.
Those can move in opposite directions — a measure that improved stability by
picking a consistently wrong point looked like progress for a full round.

**Check the position before the metric.** Three candidate fixes to the calyx
estimator improved the aggregate and were each putting the point in a worse
place. The aggregate was measured first every time.

**Small samples lie.** Five separate measures separated a hand-picked set of
eight to twelve fruit perfectly and then failed on the full set — one of them
scored 61% of stemless fruit as having a stem. Nothing is believed here until
it has been run over every frame available.

**Source frames are never modified.** Every tool reads them and writes
elsewhere, and refuses to run if the output would land inside the source
folder.
