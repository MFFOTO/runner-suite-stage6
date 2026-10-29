# -*- coding: utf-8 -*-
"""performance.fast_path must change how long a photo takes, never what is saved.

Runs the engine's real per-photo loop on synthetic photos with a fake detector, with
the fast path off and on, across randomised runners and many settings, and requires
the same crops to come out. Also covers quality_scoring.premium_only.

    python test_fast_path.py
"""

import itertools
import random

from test_support import chk, finish, make_engine, fresh_root, crisp_image, keypoints, run_photo, cv2, np, core

W, H = 2400, 1600
rng0 = np.random.default_rng(11)
IMAGES = []
for sigma in (0, 2.5, 7):                       # smooth photos: soft at the scale the subject measure looks at
    base = (rng0.random((H // 8, W // 8, 3)) * 255).astype(np.uint8)
    im = cv2.resize(base, (W, H), interpolation=cv2.INTER_CUBIC)
    IMAGES.append(cv2.GaussianBlur(im, (0, 0), sigma) if sigma else im)
IMAGES.append(crisp_image(W, H))                # a crisp one: small runners can pass the sharpness lines


def scenario(rnd):
    """Runners of mixed kinds: whole and well placed, small and unsure (low score ceiling),
    partly out of view, and touching the frame edge."""
    boxes, confs, kps = [], [], []
    for _ in range(rnd.randint(1, 4)):
        kind = rnd.choices(("healthy", "marginal", "partial", "edge"), (0.40, 0.30, 0.12, 0.18))[0]
        h = (rnd.uniform(0.16, 0.28) if kind == "marginal" else rnd.uniform(0.35, 0.9)) * H
        w = h * rnd.uniform(0.28, 0.45)
        cx = rnd.uniform(0.2 * W, 0.8 * W)
        x1 = min(max(8.0, cx - w / 2), W - w - 8.0)
        y1 = rnd.uniform(10.0, max(11.0, H - h - 10.0))
        if kind == "edge":
            side = rnd.choice(("left", "right", "top"))
            x1 = 0.0 if side == "left" else (W - w if side == "right" else x1)
            y1 = 0.0 if side == "top" else y1
        b = (x1, y1, min(W, x1 + w), min(H, y1 + h))
        visible_p = 0.30 if kind == "partial" else 0.97
        k = [[rnd.uniform(b[0], b[2]), rnd.uniform(b[1], b[3]), 0.95 if rnd.random() < visible_p else 0.05] for _ in range(17)]
        boxes.append(b); kps.append(k)
        confs.append(rnd.uniform(0.34, 0.42) if kind == "marginal" else rnd.uniform(0.5, 0.97))
    return boxes, confs, kps


def engine(fast, variant):
    return make_engine(fresh_root(), [("performance", "fast_path", fast), ("image_quality", "target_height", 500)] + list(variant))


# --- premium_only itself --------------------------------------------------------------
img = IMAGES[3]; box = (1000.0, 300.0, 1700.0, 1500.0)
import re
probe = run_photo(engine(False, [("quality_scoring", "classes", {"premium": 1, "good": 0, "review": 0}), ("selection_filters", "premium_min_sharpness", 0)]),
                  img, [box], [0.9], [keypoints(box)])
score = float(re.search(r"_q(\d+)_", probe[0]).group(1))          # what this runner actually scores
line = [("quality_scoring", "classes", {"premium": score + 3, "good": score - 10, "review": 0}), ("selection_filters", "premium_min_sharpness", 0)]
default_saved = run_photo(engine(False, line), img, [box], [0.9], [keypoints(box)])
only_saved = run_photo(engine(False, line + [("quality_scoring", "premium_only", True)]), img, [box], [0.9], [keypoints(box)])
chk(f"default: a runner scoring {score:.0f} with the Premium line at {score + 3:.0f} is Good and saved to B_Good, as before",
    len(default_saved) == 1 and "B_Good" in default_saved[0], default_saved)
chk("premium_only: the same runner is not written at all", only_saved == [], only_saved)
low = [("quality_scoring", "classes", {"premium": score - 5, "good": score - 15, "review": 0}), ("quality_scoring", "premium_only", True)]
chk("premium_only: a Premium runner is still saved", len(run_photo(engine(False, low), img, [box], [0.9], [keypoints(box)])) == 1)
e = engine(False, [("quality_scoring", "premium_only", True), ("quality_scoring", "save_review", True)])
chk("premium_only wins over save_review", e._should_save_class("premium", 90) and not e._should_save_class("good", 80)
    and not e._should_save_class("review", 50) and not e._should_save_class("reject", 70))
chk("default _should_save_class is unchanged", engine(False, [])._should_save_class("good", 65)
    and not engine(False, [])._should_save_class("review", 50))
f = engine(True, [])
chk("score floor: Good line by default, Premium line when premium_only, none when Good+Review are both saved",
    f._saveable_floor() == 60.0
    and engine(True, [("quality_scoring", "premium_only", True)])._saveable_floor() == 73.0
    and engine(True, [("quality_scoring", "save_review", True)])._saveable_floor() is None)

# --- same crops with the fast path off and on -----------------------------------------
VARIANTS = []
for premium_min, premium_only, save_review, edge, completeness, metric, p in itertools.product(
        (0, 180, 600), (False, True), (False, True), (True, False), (True, False), ("subject", "box"), (73, 35)):
    VARIANTS.append([("selection_filters", "premium_min_sharpness", premium_min), ("quality_scoring", "premium_only", premium_only),
                     ("quality_scoring", "save_review", save_review), ("completeness_guard", "edge_truncation_guard", edge),
                     ("completeness_guard", "enabled", completeness), ("selection_filters", "sharpness_metric", metric),
                     ("quality_scoring", "classes", {"premium": p, "good": max(0, p - 13), "review": 0})])

rnd = random.Random(2026)
mismatches = photos = total_saved = 0
first_bad = None
skips = {}
for variant in rnd.sample(VARIANTS, 80):
    for _ in range(4):
        im = rnd.choice(IMAGES); boxes, confs, kps = scenario(rnd)
        outs = {}
        for fast in (False, True):
            e = engine(fast, variant)
            outs[fast] = run_photo(e, im, boxes, confs, kps)
            if fast:
                total_saved += len(outs[fast])
                for k, v in e.stats.items():
                    if k.startswith("skipped_"):
                        skips[k] = skips.get(k, 0) + v
        photos += 1
        if outs[False] != outs[True]:
            mismatches += 1; first_bad = first_bad or (variant, outs)
chk(f"the same crops are saved with the fast path on and off ({photos} randomised photos, 80 setting combinations)", mismatches == 0, first_bad)
for rule in ("skipped_early_partial", "skipped_fence_soft", "skipped_fence_score"):
    chk(f"the scenarios exercised the rule {rule}", skips.get(rule, 0) > 0, skips)
chk("and crops were genuinely saved, so identical output is not just identical emptiness", total_saved >= 50, total_saved)

# the score-ceiling rule on its own: marginal runners only, several score lines
rnd2 = random.Random(77)
mism2 = n2 = ceiling_skips = saved2 = 0
for (p, g), premium_only in itertools.product(((92, 88), (84, 80), (78, 74), (40, 35)), (False, True)):
    variant = [("selection_filters", "premium_min_sharpness", 0), ("quality_scoring", "premium_only", premium_only),
               ("quality_scoring", "classes", {"premium": p, "good": g, "review": 0})]
    for _ in range(14):
        boxes, confs, kps = [], [], []
        for _ in range(rnd2.randint(1, 3)):
            h = rnd2.uniform(0.16, 0.34) * H; w = h * rnd2.uniform(0.28, 0.45)
            x1 = rnd2.uniform(0.2 * W, 0.7 * W); y1 = rnd2.uniform(20.0, H - h - 20.0)
            boxes.append((x1, y1, x1 + w, y1 + h)); confs.append(rnd2.uniform(0.34, 0.6))
            kps.append([[rnd2.uniform(x1, x1 + w), rnd2.uniform(y1, y1 + h), 0.95] for _ in range(17)])
        outs = {}
        for fast in (False, True):
            e = engine(fast, variant)
            outs[fast] = run_photo(e, IMAGES[3], boxes, confs, kps)
            if fast:
                ceiling_skips += e.stats.get("skipped_fence_score", 0); saved2 += len(outs[fast])
        n2 += 1; mism2 += outs[False] != outs[True]
chk(f"score-ceiling rule: same crops with the fast path on and off ({n2} marginal-runner photos, 4 score lines)", mism2 == 0)
chk("...the rule fired often and some marginal runners were still saved", ceiling_skips >= 10 and saved2 >= 5, (ceiling_skips, saved2))

# report: a skipped candidate says why, and the unmeasured column is blank
e = engine(True, [("quality_scoring", "premium_only", True), ("selection_filters", "premium_min_sharpness", 5000)])
e.cfg["debug"]["write_csv"] = True
run_photo(e, IMAGES[3], [box], [0.9], [keypoints(box)])
chk("a skipped candidate is reported as skipped_<reason>", any(str(r.get("reason", "")).startswith("skipped_") for r in e.report_rows), e.report_rows)
e = engine(True, low)
e.cfg["debug"]["write_csv"] = True
run_photo(e, IMAGES[3], [box], [0.9], [keypoints(box)])
rows = [r for r in e.report_rows if r.get("decision") == "accept"]
chk("an accepted crop's sharp_box is blank in fast mode, measured otherwise",
    rows and rows[0]["sharp_box"] is None and engine(False, low) is not None)
e = engine(False, low); e.cfg["debug"]["write_csv"] = True
run_photo(e, IMAGES[3], [box], [0.9], [keypoints(box)])
chk("...and measured with the fast path off", [r for r in e.report_rows if r.get("decision") == "accept"][0]["sharp_box"] > 0)

# rejects saved to disk for inspection must still be written: the fast path stands down
e = engine(True, [("quality_scoring", "premium_only", True), ("selection_filters", "premium_min_sharpness", 5000), ("debug", "save_rejects", True)])
run_photo(e, IMAGES[3], [box], [0.9], [keypoints(box)])
chk("with debug.save_rejects on, nothing is skipped (the rejects must still be saved)",
    not any(k.startswith("skipped_") for k in e.stats) and any((e.rejects_folder).glob("*.jpg")), e.stats)
finish()
