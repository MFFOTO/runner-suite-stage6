# -*- coding: utf-8 -*-
"""The engine must carry the original photo's EXIF onto every crop.

    python test_metadata.py            (needs piexif and Pillow, which the engine's setup installs)
"""

import contextlib
import io
from unittest import mock

from test_support import chk, finish, make_engine, fresh_root, crisp_image, cv2, np, core
import piexif
from PIL import Image

root = fresh_root()
e = make_engine(root)

# a camera-style original: capture date, make, orientation 6, a thumbnail, true size 6192x4128, a maker note
thumb = io.BytesIO(); Image.new("RGB", (160, 120), (200, 30, 30)).save(thumb, "JPEG")
src = root / "orig.jpg"
Image.fromarray(crisp_image(400, 300)).save(src, "JPEG", quality=90)
piexif.insert(piexif.dump({
    "0th": {piexif.ImageIFD.Make: b"NIKON CORPORATION", piexif.ImageIFD.Model: b"NIKON Z 8", piexif.ImageIFD.Orientation: 6,
            piexif.ImageIFD.Software: b"Ver.03.10"},
    "Exif": {piexif.ExifIFD.DateTimeOriginal: b"2026:10:25 11:34:40", piexif.ExifIFD.SubSecTimeOriginal: b"58",
             piexif.ExifIFD.PixelXDimension: 6192, piexif.ExifIFD.PixelYDimension: 4128, piexif.ExifIFD.MakerNote: b"\x01" * 20000},
    "GPS": {}, "Interop": {}, "1st": {piexif.ImageIFD.Compression: 6}, "thumbnail": thumb.getvalue()}), str(src))
chk("the test original really has a thumbnail and orientation 6",
    piexif.load(str(src))["thumbnail"] and piexif.load(str(src))["0th"][piexif.ImageIFD.Orientation] == 6)

crop = root / "crop.jpg"; cv2.imwrite(str(crop), crisp_image(200, 300))
ok = e._copy_exif_without_orientation(src, crop)
got = piexif.load(str(crop))
chk("returns True when it wrote the EXIF", ok is True)
chk("capture date and sub-seconds are carried over unchanged",
    got["Exif"][piexif.ExifIFD.DateTimeOriginal] == b"2026:10:25 11:34:40" and got["Exif"][piexif.ExifIFD.SubSecTimeOriginal] == b"58")
chk("camera make, model and software are kept", got["0th"][piexif.ImageIFD.Make] == b"NIKON CORPORATION"
    and got["0th"][piexif.ImageIFD.Model] == b"NIKON Z 8" and got["0th"][piexif.ImageIFD.Software] == b"Ver.03.10")
chk("orientation is dropped (the pixels are already rotated)", piexif.ImageIFD.Orientation not in got["0th"])
chk("the thumbnail of the uncropped original is dropped", not got["thumbnail"])
chk("the pixel dimensions are the crop's, not the original's",
    (got["Exif"][piexif.ExifIFD.PixelXDimension], got["Exif"][piexif.ExifIFD.PixelYDimension]) == (200, 300))
chk("the crop still decodes", cv2.imread(str(crop)) is not None)

crop2 = root / "crop2.jpg"; cv2.imwrite(str(crop2), crisp_image(200, 300))
e._copy_exif_without_orientation(src, crop2, size=(4000, 6000))
g2 = piexif.load(str(crop2))["Exif"]
chk("an explicit size (the real output size) is used for the dimensions",
    (g2[piexif.ExifIFD.PixelXDimension], g2[piexif.ExifIFD.PixelYDimension]) == (4000, 6000))

# a maker note too big for one segment: dropped, everything else kept, no crash
big = piexif.load(str(src)); big["Exif"][piexif.ExifIFD.MakerNote] = b"\x02" * 70000
crop3 = root / "crop3.jpg"; cv2.imwrite(str(crop3), crisp_image(200, 300))
with mock.patch.object(core.piexif, "load", return_value=big):
    ok3 = e._copy_exif_without_orientation(src, crop3)
g3 = piexif.load(str(crop3))
chk("an oversize maker note is dropped, the capture date survives, no crash",
    ok3 and g3["Exif"][piexif.ExifIFD.DateTimeOriginal] == b"2026:10:25 11:34:40" and piexif.ExifIFD.MakerNote not in g3["Exif"])

# failures are reported, not swallowed
bad = root / "notajpeg.jpg"; bad.write_bytes(b"not a jpeg")
crop4 = root / "crop4.jpg"; cv2.imwrite(str(crop4), crisp_image(200, 300))
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    r1 = e._copy_exif_without_orientation(bad, crop4); r2 = e._copy_exif_without_orientation(bad, crop4)
chk("an unreadable source returns False and says so once, not silently and not every time",
    r1 is False and r2 is False and buf.getvalue().count("[WARN]") == 1, buf.getvalue())
chk("...and leaves the crop untouched and valid", cv2.imread(str(crop4)) is not None and not piexif.load(str(crop4))["Exif"])

e2 = make_engine(fresh_root(), [("image_quality", "preserve_exif", False)])
crop5 = root / "crop5.jpg"; cv2.imwrite(str(crop5), crisp_image(200, 300))
chk("preserve_exif off: nothing is written", e2._copy_exif_without_orientation(src, crop5) is False and not piexif.load(str(crop5))["Exif"])

# the engine's own write path (write_output) uses it with the real output size
final = crisp_image(300, 450)
out = e.write_output(src, 0, final, 500.0, 0.9, 80.0, "premium")
g = piexif.load(str(out))
chk("write_output: the saved crop has the original capture date and its own size",
    g["Exif"][piexif.ExifIFD.DateTimeOriginal] == b"2026:10:25 11:34:40"
    and (g["Exif"][piexif.ExifIFD.PixelXDimension], g["Exif"][piexif.ExifIFD.PixelYDimension]) == (300, 450)
    and piexif.ImageIFD.Orientation not in g["0th"] and not g["thumbnail"])

wh = core.HighResRunnerSuite._jpeg_size
prog = root / "progressive.jpg"; Image.fromarray(crisp_image(123, 77)).save(prog, "JPEG", progressive=True)
chk("_jpeg_size reads a baseline and a progressive JPEG", wh(crop) == (200, 300) and wh(prog) == (123, 77), (wh(crop), wh(prog)))
chk("_jpeg_size gives None for junk and for a missing file", wh(bad) is None and wh(root / "missing.jpg") is None)
# a big EXIF segment in front of the frame header must not confuse it
fat = root / "fat.jpg"; Image.fromarray(crisp_image(90, 60)).save(fat, "JPEG")
piexif.insert(piexif.dump({"0th": {}, "Exif": {piexif.ExifIFD.MakerNote: b"\x03" * 40000}, "GPS": {}, "Interop": {}, "1st": {}, "thumbnail": None}), str(fat))
chk("_jpeg_size skips a 40 KB EXIF segment", wh(fat) == (90, 60), wh(fat))
finish()
