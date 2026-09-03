# Annotation spec — read before labelling anything

Your machine is collecting right now, and you have ~2000 images to label. The
schema decisions below are the expensive ones: getting a threshold wrong costs
an afternoon of retuning, getting the keypoint convention wrong costs
re-annotating 2000 images. Settle these first.

---

## 1. Why keypoints, and not a "stem" class

The obvious approach is two detection classes, `paprika` and `stem`, then pair
each stem with the nearest paprika. Don't. It fails in three ways that
keypoints simply don't have:

- **Association.** Two paprikas touching on the belt give you two boxes and
  two stems, and no reliable way to say which stem belongs to which fruit. On
  a belt, touching fruit is normal, not an edge case.
- **Stemless fruit.** A broken-off stem means no `stem` box, which means no
  orientation at all. That's the fruit you specifically said you need to
  handle.
- **Precision.** A box around a stem gives you a blob centre. The angle you
  need is the line through the fruit, and a stem box only tells you roughly
  where one end of it is.

A pose model attaches the landmarks *to the fruit instance*, so association is
free, and it predicts a landmark position even when the landmark is hidden —
which is exactly what a stemless or stem-down fruit needs.

## 2. The two landmarks

Fixed order. This order is baked into `orientation.KEYPOINT_NAMES`, the
dataset validator and the trained model. Changing it means re-exporting.

| # | Name | Where exactly |
|---|---|---|
| 0 | `stem_end` | Centre of the **calyx** — where the stem meets the fruit shoulder. **Not** the tip of the stem. |
| 1 | `blossom_end` | Centre of the blossom scar at the opposite end, in the middle of the lobes. |

**`stem_end` is the calyx, not the stem tip.** This is the single most
important rule here. Stem length varies enormously and stems snap off in
handling — if you annotate the tip, then a fruit with a 40 mm stem and the same
fruit with a 5 mm stem produce landmarks in different places, and the model
learns that the orientation target moves depending on how roughly the fruit was
picked. The calyx is a fixed anatomical point that exists on every fruit,
including one whose stem is completely gone: there is still a scar and a
shoulder where the stem *was*.

## 3. Visibility flags — the rule that makes stemless fruit work

COCO convention, and both Roboflow and Ultralytics use it:

| Flag | Meaning | When |
|---|---|---|
| `2` | visible | You can see the feature. |
| `1` | labelled but occluded | You know where it is, you can't see it. |
| `0` | not labelled | Genuinely unknowable. |

**Annotate both landmarks on every fruit, using flag 1 when occluded.** Do not
skip a landmark just because you can't see it.

This is what separates a working system from one that fails on the fruit you
care about:

- Fruit standing on its **blossom end**, stem pointing at the camera → `stem_end`
  is flag 2 (you see it, centred in the fruit), `blossom_end` is flag 1
  (underneath, hidden, but you know it's directly below the centre). Put it
  roughly at the fruit centre.
- Fruit standing on its **stem end**, stem hidden underneath → `stem_end` flag 1
  at the centre, `blossom_end` flag 2.
- **Stemless fruit lying down** → `stem_end` flag 2 at the calyx scar, which is
  still visible. Flag 1 only if the scar itself is facing away.

If you only label what you can see, the model can never learn a standing fruit,
because it will never have seen a training example where the two landmarks
overlap. The engine reads exactly this signal: when the two predicted landmarks
land close together, it reports `standing`, and the *visibility* of `stem_end`
decides `standing_stem_up` versus `standing_stem_down`.

## 4. One class, not one per colour

Class list is exactly:

```
paprika
```

You said colour isn't a selection criterion. Don't annotate it. Splitting into
`red` / `yellow` / `green` / `orange` would:

- cut your effective data per class from 2000 to ~500,
- teach the model that colour is a meaningful feature when you want it to key
  on shape, so a colour it saw rarely detects worse,
- produce a class-confusion metric you'd have to explain to someone, that has
  no effect on what the machine does.

You need colour *robustness*, which comes from covering every colour in the
data. That's a sampling requirement, not an annotation one.

## 5. Bounding box

Tight around the fruit body, **including the stem** if it's attached. The box
matters less than the landmarks here, but it feeds two things: the crop the
shape cross-check segments, and the `min_span_ratio` standing test, which
compares landmark separation to the box diagonal. Inconsistent box padding
makes that ratio noisy, so be consistent — tight, every time.

---

## 6. What to collect — act on this now, while the machine is running

Natural belt output will be badly unbalanced for your purposes. The model
learns what it sees, so deliberately feed it the hard cases. Targets as a
share of the 2000:

| Case | Target | Why |
|---|---|---|
| Stemless / broken stem | **≥ 20%** | Natural rate is maybe 5%. This is your stated requirement — at 5% the model will treat it as noise. |
| Standing (either end up) | **≥ 15%** | The pose the belt creates and the one with no valid angle. Needs enough examples to be recognised, not guessed. |
| Touching / overlapping fruit | **≥ 15%** | Where a naive stem-box approach breaks. Prove the pose model handles it. |
| Each colour | **roughly even** | This is where colour robustness comes from. |
| Angle coverage | every 30° bucket | See below. |

**Angle uniformity is the sneaky one.** If your infeed tends to drop fruit the
same way, most of your 2000 images will cluster around a couple of angles, and
the model will be quietly excellent at those and poor everywhere else — and
your validation set, drawn from the same distribution, won't reveal it. Rotate
fruit by hand during collection to fill the gaps. `tools/dataset_check.py`
plots the angle histogram so you can see the holes.

**Lock the camera before you collect.** Set `camera.manual_settings.enabled:
true` and pin exposure, gain and white balance. Auto white balance wandering
across a collection session teaches the model that a paprika's appearance is
inherently unstable, and it poisons the shape estimator, which thresholds on
absolute saturation. Re-collecting is far more expensive than doing this now.

## 7. Labelling workflow

Don't label 2000 images by hand.

1. Label **150–200** by hand, covering the hard cases above.
2. Train a quick YOLO11n-pose run on those.
3. Use it for model-assisted labelling (Roboflow's Label Assist, or predict
   into your annotation tool) on the rest — you correct rather than place.
4. Re-train on the full set.

Step 3 is roughly a 3–5× speedup, and the corrections you make are concentrated
exactly where the model is weak, which is where your attention is worth most.

## 8. Augmentation

- **Rotation: full ±180°, yes.** Orientation is the target, so you want every
  angle represented. Roboflow and Ultralytics rotate keypoints with the image
  correctly.
- **Flips: safe here, but know why.** Horizontal/vertical flip is the classic
  pose-dataset bug — with left/right paired landmarks (like wrists) a flip
  swaps their identities and you must swap the labels too. Your two landmarks
  aren't a mirror pair, so there's nothing to swap. Flip freely.
- **Hue/saturation jitter: moderate, and this is where colour robustness comes
  from.** Keep it modest, though: push hue too far and a green fruit with a
  green stem becomes ambiguous.
- **Blur/noise: light.** Matches real motion on a belt.
- **Avoid heavy crop augmentation** — it pushes landmarks outside the image and
  the visibility semantics get muddled.

## 9. Training

```bash
yolo pose train model=yolo11n-pose.pt data=paprika.yaml \
     imgsz=640 epochs=150 batch=16
```

`paprika.yaml`:

```yaml
path: data/datasets/paprika
train: images/train
val: images/val

kpt_shape: [2, 3]        # 2 landmarks, (x, y, visibility)
flip_idx: [0, 1]         # no mirror pair to swap - see section 8
names:
  0: paprika
```

Then point `paprika.pose.model_path` at `runs/pose/train/weights/best.pt` and
flip `paprika.backend` to `"pose"`.

## 10. What to check before you trust it

Run `tools/dataset_check.py` on the export. It catches the errors that are
invisible until they've cost you a training run:

- landmark count ≠ 2, or the wrong `kpt_shape`
- **flipped keypoint order** — detected by measuring whether `stem_end` sits at
  the wider end of the fruit across the set. If most of your data says
  otherwise, two landmarks got swapped somewhere.
- visibility flags never being 1, meaning somebody skipped occluded landmarks
  and standing fruit will not work
- angle histogram gaps
- stemless / standing proportions against the targets above

---

## 11. Gemeten op jouw eigen beelden (160 stuks, juli 2026)

Twee dingen die het plan veranderen, gemeten in plaats van aangenomen.

**De blauwe band is goud waard.** Fruit en band zijn puur op hue te scheiden:
de band ligt strak op hue 100-125, al het fruit valt daarbuiten, en de bleke
randstroken vallen af op saturatie. Segmentatie lukte op 160 van de 160
beelden, 243 vruchten. Daarom zijn **alle bounding boxes gratis** - laat
`tools/pre_annotate.py` ze genereren en teken ze niet met de hand.

**De vormschatter werkt niet op blokpaprika, en dat verandert wat het model
moet leren.** Het idee was dat de steelkant aan de bredere schouder zit, zodat
het silhouet kon uitwijzen welk uiteinde welk is bij een vrucht zonder steel.
Op 87 vruchten met zichtbare steel klopt dat in 66% van de gevallen, mediane
signaalsterkte 0,074, mediane langwerpigheid 1,33. Te rond en te symmetrisch.
Dus staat `shape_crosscheck` uit, en wordt er in de pre-annotatie geen keypoint
geraden op basis van vorm.

Wat dat betekent voor jou: het model moet het onderscheid tussen steelkant en
bloemkant leren van **fijne kenmerken** - het calyx-litteken, het lobbenpatroon
aan de onderkant - en niet van de grove omtrek. Dat kan een CNN prima, en jij
kunt het met het blote oog ook, maar het stelt wel een eis aan de data:

> Zorg dat een steelloze vrucht altijd zó gefotografeerd is dat het
> calyx-litteken zichtbaar is. Is dat niet zo, label het uiteinde dan met
> vlag 1 op de plek waar je het vermoedt, en niet met vlag 2 alsof je het ziet.
> Anders leert het model dat een raadsel een zeker antwoord heeft.

**Verdeling van de steekproef**, als richtlijn voor wat er nog bij moet:

| | aantal | aandeel |
|---|---|---|
| rood | 89 | 37% |
| groen | 95 | 39% |
| oranje | 43 | 18% |
| geel | 16 | 7% |
| steel gevonden via kleur | 129 | 53% |
| rechtopstaand | 7 | 3% |
| geen steel zichtbaar | 19 | 8% |

Geel is ondervertegenwoordigd en rechtopstaande vruchten zitten ver onder de
15% uit sectie 6. Beide zijn nu nog bij te sturen, want de machine draait.
