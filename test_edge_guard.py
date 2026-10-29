# -*- coding: utf-8 -*-
"""A runner cut off by the photo's edge must not be graded Premium/Good.

    python test_edge_guard.py
"""

from test_support import *      # noqa: F401,F403
from test_support import (chk, finish, make_engine, fresh_root, crisp_image, keypoints, run_photo, core, np)

H, W = 4128, 6192
KP = np.array([[100, 100, 0.9]] * 17, dtype=float)       # every keypoint confident: a complete athlete
on, off = make_engine(fresh_root()), make_engine(fresh_root(), [("completeness_guard", "edge_truncation_guard", False)])

chk("guard is on by default", core.DEFAULT_CONFIG["completeness_guard"]["edge_truncation_guard"] is True)

cut_close_up = (4900.0, 0.0, 6192.0, 4128.0)             # big close-up cut by the right and top edge
chk("without the guard such a close-up takes the full-frame path (the gap)", off._is_full_frame_runner(cut_close_up, (H, W)) is True)
chk("with it, the close-up is judged like any other runner", on._is_full_frame_runner(cut_close_up, (H, W)) is False)
chk("...so the completeness guard demotes it", on._assess_completeness(KP, cut_close_up, (H, W))["ok_for_premium"] is False)

right_only = (5100.0, 900.0, 6192.0, 3900.0)
chk("without the guard a runner cut by the right edge is fine for Premium (the gap)",
    off._assess_completeness(KP, right_only, (H, W))["ok_for_premium"] is True)
c = on._assess_completeness(KP, right_only, (H, W))
chk("with it: demoted, reason truncated_side", c["ok_for_premium"] is False and c["reason"] == "truncated_side" and c["truncated_side"])
c = on._assess_completeness(KP, (0.0, 900.0, 900.0, 3900.0), (H, W))
chk("...and so is one cut by the left edge", c["ok_for_premium"] is False and c["reason"] == "truncated_side")

whole = (1000.0, 200.0, 3000.0, 4000.0)
chk("a whole close-up runner still takes the full-frame path", on._is_full_frame_runner(whole, (H, W)) is True)
chk("a complete runner inside the frame is still Premium-eligible",
    on._assess_completeness(KP, (2000.0, 500.0, 2800.0, 3000.0), (H, W))["ok_for_premium"] is True)
feet = (2000.0, 500.0, 2800.0, float(H))
chk("feet cut by the bottom edge are still fine", on._assess_completeness(KP, feet, (H, W))["ok_for_premium"] is True
    and not on._cut_by_frame(feet, (H, W), top=True))
chk("a box 7 px from the edge is not cut off, 6 px is", not on._cut_by_frame((5000.0, 900.0, W - 7.0, 3900.0), (H, W))
    and on._cut_by_frame((5000.0, 900.0, W - 6.0, 3900.0), (H, W)))
chk("the engine's own top-edge rule is untouched",
    on._assess_completeness(KP, (2000.0, 0.0, 2800.0, 3000.0), (H, W))["reason"] == "truncated_head")
c = off._assess_completeness(KP, right_only, (H, W))
chk("guard off: behaves exactly like before (flag False, still eligible)", c["truncated_side"] is False and c["ok_for_premium"] is True)

# the engine's real per-photo loop: what ends up in A_Premium
img = crisp_image(3000, 2000)
for label, box, expect_on in [("a whole runner inside the frame", (1100.0, 300.0, 1900.0, 1900.0), 1),
                              ("a runner cut by the right edge", (1400.0, 500.0, 3000.0, 1700.0), 0),
                              ("a big close-up cut by the right and top edge", (1500.0, 0.0, 3000.0, 2000.0), 0)]:
    def premium(guard):
        e = make_engine(fresh_root(), [("completeness_guard", "edge_truncation_guard", guard),
                                       ("image_quality", "target_height", 800)])
        return len([n for n in run_photo(e, img, [box], [0.9], [keypoints(box)]) if "A_Premium" in n])
    chk(f"loop, guard off: {label} -> Premium crop", premium(False) == 1)
    chk(f"loop, guard on: {label} -> {'Premium crop' if expect_on else 'no Premium crop'}", premium(True) == expect_on)
finish()
