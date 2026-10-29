# -*- coding: utf-8 -*-
"""The whole-image fallback must respect quality_scoring.premium_only, and nothing else about it changes.

    python test_fallback.py
"""

from test_support import chk, finish, make_engine, fresh_root, crisp_image, cv2

img = crisp_image(2400, 1600)


def run(overrides):
    e = make_engine(fresh_root(), [("whole_image_fallback", "enabled", True), ("whole_image_fallback", "min_original_sharpness", 0),
                                   ("whole_image_fallback", "reject_if_fence", False), ("image_quality", "target_height", 500)] + overrides)
    e.cfg["debug"]["write_csv"] = True
    src = e.output_folder / "photo.jpg"; cv2.imwrite(str(src), img)
    e._process_detection_results(img, src, [])                   # no person found: the fallback runs
    written = sorted(str(p.relative_to(e.output_folder)) for p in e.output_folder.rglob("*.jpg") if p.name != "photo.jpg")
    return e, written


e, w = run([])
chk("default: the fallback saves its crop to C_Review, exactly as before", len(w) == 1 and "C_Review" in w[0], w)
e, w = run([("quality_scoring", "save_review", False)])
chk("...even though save_review is off (enabling the fallback is an explicit request)", len(w) == 1, w)
e, w = run([("quality_scoring", "premium_only", True)])
chk("premium_only: a fallback that would save a Review crop writes nothing", w == [], w)
chk("...it is counted and reported with the reason, and no work was done for it",
    e.stats.get("fallback_not_saved") == 1 and any("premium_only" in str(r.get("reason")) for r in e.report_rows)
    and "fallback_saved" not in e.stats, (e.stats, e.report_rows))
e, w = run([("quality_scoring", "premium_only", True), ("whole_image_fallback", "save_class", "good")])
chk("premium_only: save_class good is not saved either", w == [], w)
e, w = run([("quality_scoring", "premium_only", True), ("whole_image_fallback", "save_class", "premium")])
chk("premium_only with save_class premium: still saved, to A_Premium", len(w) == 1 and "A_Premium" in w[0], w)
e, w = run([("whole_image_fallback", "enabled", False)])
chk("fallback disabled: nothing is written", w == [], w)
finish()
