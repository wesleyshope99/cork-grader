"""Generate synthetic "cork disk photo" test fixtures so the app's crop/train/
grade pipeline can be exercised end-to-end without real cork photos. Not part
of the shipped app -- delete test-fixtures/ before deploying.

Each image simulates a disk of varying porosity (grade proxy: more/larger
dark pore speckles = worse grade) placed at a random offset/size within a
larger background frame, so the crop UI's pan/zoom actually has something to
do. Pure stdlib PNG writer, no Pillow needed.
"""
import struct
import zlib
import os
import random

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "test-fixtures")

GRADE_PORE_DENSITY = {
    "Flor": 0.01,
    "Extra": 0.03,
    "AAA": 0.06,
    "A": 0.10,
    "B": 0.16,
    "C": 0.24,
}

FRAME = 500
BG = (60, 58, 55)


def write_png(path, size, pixel_fn):
    rows = []
    for y in range(size):
        row = bytearray([0])
        for x in range(size):
            r, g, b = pixel_fn(x, y)
            row += bytes((r, g, b, 255))
        rows.append(bytes(row))
    raw = b"".join(rows)

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)
    idat = zlib.compress(raw, 6)
    with open(path, "wb") as f:
        f.write(sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b""))


def make_disk_photo(path, grade, seed):
    rng = random.Random(seed)
    disk_r = rng.uniform(FRAME * 0.28, FRAME * 0.42)
    cx = rng.uniform(disk_r + 10, FRAME - disk_r - 10)
    cy = rng.uniform(disk_r + 10, FRAME - disk_r - 10)
    base = (int(200 + rng.uniform(-15, 15)), int(165 + rng.uniform(-15, 15)), int(110 + rng.uniform(-15, 15)))
    density = GRADE_PORE_DENSITY[grade]

    pores = []
    n_pores = int(density * 900)
    for _ in range(n_pores):
        ang = rng.uniform(0, 6.283)
        rad = rng.uniform(0, disk_r * 0.92)
        px = cx + rad * (rng.random() ** 0.5) * (1 if True else 1)
        py = cy + rad * (rng.random() ** 0.5)
        # recompute properly with angle
        px = cx + (rad) * 0.0
        pores.append((cx + rad_cos(ang, rad), cy + rad_sin(ang, rad), rng.uniform(1.5, 5.5)))

    def pixel_fn(x, y):
        dx, dy = x - cx, y - cy
        if dx * dx + dy * dy > disk_r * disk_r:
            return BG
        r, g, b = base
        for (px, py, pr) in pores:
            ddx, ddy = x - px, y - py
            if ddx * ddx + ddy * ddy <= pr * pr:
                shade = 0.35
                return (int(r * shade), int(g * shade), int(b * shade))
        return (r, g, b)

    write_png(path, FRAME, pixel_fn)


def rad_cos(ang, rad):
    import math
    return rad * math.cos(ang)


def rad_sin(ang, rad):
    import math
    return rad * math.sin(ang)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = []
    for grade in GRADE_PORE_DENSITY:
        for i in range(6):
            fname = f"{grade}_{i}.png"
            make_disk_photo(os.path.join(OUT_DIR, fname), grade, seed=hash((grade, i)))
            manifest.append({"file": fname, "grade": grade})
            print("wrote", fname)
    import json
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
