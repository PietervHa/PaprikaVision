# The pose model, start to finish

From no model at all, through the first trained one, to growing it with fruit
it got wrong. `ANNOTATION_SPEC.md` decides what a label *means*; read it first.
This decides how to get from there to a model on the line and keep it improving.

---

## Contents

1. [What you are building](#1-what-you-are-building)
2. [Labelling](#2-labelling)
3. [The first model](#3-the-first-model)
4. [Putting it on the line](#4-putting-it-on-the-line)
5. [Growing it with failures](#5-growing-it-with-failures)
6. [What to keep and what to delete](#6-what-to-keep-and-what-to-delete)
7. [Recovery](#7-recovery)
8. [Things that cost us weeks](#8-things-that-cost-us-weeks)

---

## 1. What you are building

The `shape` backend finds the stem by colour. That works on red, orange and
yellow and cannot work on green, where stem and flesh share a hue. Every
workaround downstream exists because of that, and on green they run out.

The `pose` backend replaces the stem search with a model that has seen
examples. `config/default.yaml` already points at `models/paprika_pose.pt`; you
are producing that file.

**The benchmark to beat.** Measured on 470 hand-labelled fruit with `shape`:

| | |
|---|---|
| placed fruit, median error | **2.4°** |
| within 10° | **90%** |
| placements worse than 45° | **8** (2.3%) |

The last row is the one that matters: fruit that reached the actuator badly
wrong with nothing flagged.

**Three artefacts, and what each is for.**

| | what it is | needed for |
|---|---|---|
| `data/raw`, `data/captures` | source frames | labelling |
| `data/debug/labels/stem_labels.json` | your clicks | labelling, re-labelling |
| `data/datasets/paprika_vN/` | **the dataset** — images + labels together | training |

Only the third is needed to train. That is what lets the other two go.

---

## 2. Labelling

Two landmarks per fruit, in this order. The order is fixed by the spec and
baked into the model, exporter and validator.

| # | name | what to click |
|---|---|---|
| 0 | `stem_end` | the **calyx** — where the stem meets the shoulder. **Not the stem tip.** |
| 1 | `blossom_end` | centre of the blossom scar, in the middle of the lobes |

Stems vary in length and snap off in handling. Annotate the tip and the same
fruit gives a different landmark depending on how roughly it was picked. The
calyx is on every fruit, always in the same anatomical place.

### Which mouse button

| button | flag | meaning |
|---|---|---|
| **left** | 2 | I can see this landmark |
| **right** | 1 | it is hidden, but I know where it is — click the position anyway |
| `n` key | 0 | genuinely absent: outside the frame, or I cannot judge |

Right-click is not "I'm unsure". It is for landmarks that are definitely there
and definitely hidden — the underside of a standing fruit, a calyx facing away.
Skip those instead of positioning them and the model learns that every fruit
shows both ends, and standing fruit can never work. `dataset_check` rejects a
dataset with no occluded flags at all for exactly this reason.

### The three cases

**Lying on its side** — both visible, two left clicks, one at each end.

**Standing, calyx facing you** — you can see the calyx star in the middle.
Calyx: left click at the centre. Blossom: right click at the *same* centre,
because it is directly underneath.

**Standing, lobes facing you** — no calyx, lobes converging. Blossom: left
click at the centre. Calyx: right click at the same centre.

Both landmarks on top of each other is correct here. It is the signal the
engine reads: two landmarks close together means `standing`, and the visibility
of `stem_end` decides stem-up from stem-down.

*How to tell the standing cases apart:* look at what is in the middle. A green
star-shaped calyx, or lobes converging with nothing in the centre.

### Running it

```
python -m tools.label_stems data/raw
python -m tools.label_stems data/raw --colour green --limit 200
python -m tools.label_stems data/raw --blossom
```

Calyx pass, then blossom pass. Both resumable — frames already done are
skipped, so this can be spread over several sittings. Labels are written after
every frame via a temporary file, so an interrupted session cannot lose the
ones already taken.

`s` skips, `u` undoes the last fruit, `q` saves and quits.

> Needs desktop OpenCV. `opencv-python-headless` has no GUI; the tool says so
> rather than failing with a stack trace.

### Check before spending a training run

```
python -m tools.label_stems --verify
```

Instant, and it catches what a training curve never would:

- both landmarks on the same side of the centre — the blossom clicked *next to*
  the calyx rather than *across from* it
- landmarks nearly on top of each other — correct end-on, wrong lying down
- a landmark outside its own fruit's box
- **no occluded flags anywhere** — fix this before exporting

---

## 3. The first model

One command does export, dataset check, training and scoring:

```
python -m tools.retrain --name v1 --dry-run
python -m tools.retrain --name v1
```

`--dry-run` prints the plan without doing anything. Worth running first.

**What it does for you.** Source folders are derived from the labels file, so
you cannot accidentally export only the captures. A fresh `--name` is required,
because the exporter copies in without clearing and reusing a folder mixes two
datasets. It identifies the run directory *it* created, not whichever is newest
— `train10` sorts before `train2`. It backs up the live weights before
replacing them. It scores on the held-out split, not on frames the model
trained on. And it writes a manifest recording what produced what.

**What it will not do.** Read `dataset_check`'s warnings for you. Standing at
7% and occluded at 1% will train perfectly well and produce a model that cannot
do standing fruit. No script can tell that from a good run.

Useful options:

```
--epochs 100 --patience 10        stop when validation stops improving
--batch 16                        drop to 8 if VRAM complains
--imgsz 640                       must match paprika.pose.imgsz
--base yolo11n-pose.pt            train fresh (default)
--base models/paprika_pose.pt     fine-tune from existing weights
```

### Reading `dataset_check`

Errors block. Warnings do not, but they describe what the model will be blind
to:

| warning | what it means |
|---|---|
| under 15% standing | a normal belt run gives almost none. Stand some up by hand and run them past. |
| under 20% stemless/occluded | the case the model exists for. |
| empty angle buckets | the model will be blind at those angles, and a val split from the same distribution will not reveal it. |

If you accept a limitation deliberately:

```
python -m tools.dataset_check "data/datasets/paprika_v1" --i-know-better
```

It prints every error as usual and exits 0 instead of 1. **Write down which
ones you accepted, next to the weights.** A model with a known blind spot is
fine; one with a forgotten blind spot is how a standing fruit gets placed
confidently at some arbitrary angle six months later.

---

## 4. Putting it on the line

`retrain` installs the new weights for scoring and prints the numbers. Read
them against the model you had, and against `shape`'s 2.4° median.

In `config/default.yaml`:

```yaml
paprika:
  backend: "pose"
```

Restart the app — config is read at startup. The HMI's Detector field should
read `pose`, and `data/logs/vision.log` should show:

```
Paprika pose model loaded: models/paprika_pose.pt
Paprika pose model warmed up in NNN ms
```

The warm-up runs one throwaway inference at load. A cold CUDA model compiles
kernels on its first call — seconds, not milliseconds — and without this that
lands on whichever fruit arrives first after a restart.

### Scoring honestly

```
python -m tools.label_stems data/datasets/paprika_v1/val/images --score --worst 10
```

Val frames only. The model trained on the rest, so scoring on `data/raw`
flatters it. `--worst` names the worst placed fruit by filename, which is how
every fix in this project that stuck was found.

Expect the model to lose to `shape` on red, where colour-based stem finding is
already excellent. **Score green separately** — that is where `shape` has no
good route and where the model should win. The answer may be pose on green and
shape on red rather than picking one.

### Speed

```
python -m tools.bench_pose --device cuda:0
```

On a Quadro M1200 the forward pass is ~88 ms and framework overhead is 0.1 ms —
measured. ONNX and TensorRT have nothing to remove, `half=True` does nothing
below compute 7.0, and no architecture in the family is faster. 40 ms is not
reachable on that card. Re-run this if you change GPU, model or image size.

---

## 5. Growing it with failures

### Capture what it gets wrong

Press **`c`** while the app runs. It saves the raw frame to `data/captures`
with the verdict in the filename and a JSON sidecar holding what the machine
decided.

Raw, not overlaid: an overlaid frame is a picture of an answer and cannot be
re-run, re-labelled or trained on. PNG, not JPEG — measured, JPEG at quality 95
shifted the reported stem angle by up to 4.1° and the placement gate is 6°.

> If the source is a folder of files rather than a live camera, the capture is
> a *copy* of a frame already in `data/raw`. The exporter deduplicates on image
> content, so this costs nothing — but it means you are not collecting anything
> new, only marking which frames were wrong.

### Label, then grow the dataset

```
python -m tools.label_stems data/captures
python -m tools.label_stems data/captures --blossom
python -m tools.label_stems --verify

python -m tools.retrain --name v2 --base-dataset data/datasets/paprika_v1
```

`--base-dataset` brings v1's images across **with their original train/val
side**, then adds whatever has been labelled since. That split matters: a frame
crossing from train to val between versions would look like an improvement and
is not one.

Repeatable — pass several to combine versions.

### Fine-tune or start fresh?

Default is fresh from `yolo11n-pose.pt`. The old data is still there via
`--base-dataset`, so nothing is lost.

`--base models/paprika_pose.pt` continues from the existing weights. On a
dataset this size that tends to overfit the additions. It is offered rather
than recommended, and the scoring step will say whether it worked.

**There is no way to update a network with only new data and keep the old.**
The weights have no memory of data they can no longer see, and a few hundred
captured failures would teach it the hard cases while losing the easy ones.
Keeping the exported dataset is the cheap version of remembering.

---

## 6. What to keep and what to delete

After a successful export, the dataset folder is self-contained.

```
data/datasets/paprika_vN/     KEEP - the artefact
data/raw/                     can go
data/captures/                can go
data/debug/labels/            can go (but see below)
```

Check first:

```
python -m tools.datasets --verify
```

`all datasets intact` means the copies are good. A mismatch means sort that out
before throwing away the only other copy.

### What the datasets know about themselves

```
python -m tools.datasets
python -m tools.datasets --show v2
python -m tools.datasets --trace models/paprika_pose.pt
```

```
  ok paprika_v1    620 frames   802 fruit   id ee1096ee70df   from: -
  ok paprika_v2    715 frames   984 fruit   id e84e48a042d5   from: paprika_v1
```

Each carries a `DATASET.json`: counts, landmark visibility, what it was built
on, and a **content id** hashed over every label file. A dataset folder is
ordinary files — someone drops images in, a sync half-copies it, an editor
rewrites a label — and none of that announces itself. `--verify` catches it and
exits non-zero.

Parents are recorded by content id as well as path, because folders get moved
and what was actually trained on does not change when they do.

`--trace` follows a model back through its datasets *and verifies each link*,
so a model trained on a dataset that has since been altered says so rather than
looking fine.

### The one reason to keep the raw frames

Re-labelling. If you ever want to fix a bad click or change a convention, you
need the original frames and `stem_labels.json` — the exported dataset holds
YOLO-format labels only, and there is no path back to the clicking tool.

If disk is cheap, zip `data/raw` and `stem_labels.json` once and put them
aside. If not, you are fine: the exported labels are what trains the model.

---

## 7. Recovery

| situation | command |
|---|---|
| model misbehaving | set `backend: "shape"` in the config |
| new model worse than the old | `copy models\paprika_pose_before_*.pt models\paprika_pose.pt` |
| blossom pass done wrong | `python -m tools.label_stems --reset-blossom` |
| labels look wrong after a reset | rename `stem_labels.backup-*.json` over `stem_labels.json` |
| labels do not match at scoring time | `python -m tools.label_stems data/raw --repair` |
| dataset fails verification | rebuild it from its parent with `retrain --base-dataset` |

`retrain` backs up the live weights before replacing them and prints the exact
command to go back.

---

## 8. Things that cost us weeks

**Stability is not accuracy.** Most numbers produced before the hand labels
existed measured how much an estimate *moved*, not how *right* it was. Those
can move in opposite directions: one candidate fix improved stability by
choosing a consistently wrong point, and looked like progress for a full round.

**Check the position before the metric.** Three separate fixes to the calyx
estimator improved the aggregate while putting the landmark in a worse place.
The aggregate was measured first every time.

**Small samples lie.** Five different measures separated a hand-picked set of
eight to twelve fruit perfectly and then failed on the full set. One scored 61%
of stemless fruit as having a stem. Nothing is believed here until it has been
run over every frame available.

**Two estimators agreeing proves nothing.** They agree when both are right and
when both are wrong the same way. A silhouette estimator was condemned as
worse-than-random on that basis when the real fault was a mirrored coordinate
conversion in the test.

**A missing field is not a zero.** The pose backend emits no `stem_area_ratio`,
and a gate reading the default of `0.0` ran an expensive classifier on every
fruit and discarded good landmarks. The same shape twice: a field set but never
read, so half-visible fruit were placed with confident angles.

**Source frames are never modified.** Every tool reads them and writes
elsewhere, and refuses to run if the output would land inside the source
folder.
