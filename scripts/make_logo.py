#!/usr/bin/env python3
"""Rasterize the teeter logo to logo.png (and a raw RGBA buffer) using only
Python's stdlib (traced shapes + glow), so we don't add image-crate deps to
the project. Matches assets/logo.svg."""
import math, struct, zlib, os

SIZE = 512  # logo.png dims

# palette
BG      = (5, 9, 12)
GLOW    = (24, 242, 226)
PANEL_B = (76, 255, 241)
PANEL_MID=(21, 201, 186)
PANEL_D = (6, 125, 116)
def lerp(a, b, t):
    return a + (b - a) * t

def mix(c1, c2, t):
    return tuple(int(round(lerp(c1[i], c2[i], t))) for i in range(3))

# rounded-rect signed distance (in local coords, half-extent h)
def sd_rrect(px, py, h, r):
    qx = abs(px) - (h - r)
    qy = abs(py) - (h - r)
    ox = max(qx, 0.0)
    oy = max(qy, 0.0)
    outside = math.hypot(ox, oy)
    inside = min(max(qx, qy), 0.0)
    return outside + inside - r

def rot(x, y, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return (c * x - s * y, s * x + c * y)

def panel_color():
    return PANEL_B, PANEL_MID, PANEL_D

def build(size, out_rgba):
    cx = cy = size / 2
    scale = size / 512.0
    center_y = (size - 60) / 2 # place pivot a touch below center in raster
    center_y = size * 0.526

    rows = []
    for y in range(size):
        row = []
        for x in range(size):
            # transform into logo space (0..512), centered at (256,268)
            px = (x - cx)
            py = (y - center_y)
            # to logo-space scale
            X = px / scale
            Y = py / scale

            # start from bg
            col = BG
            alpha = 255

            # background gradient glow (radial)
            gdist = math.hypot(X - 0, Y - (268 - 268) ) # glow centered at logo center in logo space = (256,268) -> local 0,0? recompute below in screen
            # simpler: use local space; glow center at (0,10) in logo-relative (panel area)
            glow_c = math.hypot(X, Y - 12)
            t = max(0.0, 1.0 - glow_c / 216.0)
            if t > 0.0:
                glow_t = t * t * 0.55
                gcol = mix(BG, GLOW, glow_t)
                col = gcol

            # echoes (top-most = furthest/back), drawn in logo coords around (0,0)
            # we evaluate the rounded rect shadow copies
            draw_panel = None
            # iterate echoes back-to-front
            echoes = [(60, 10, 24, 0.16), (36, 6, 16, 0.30), (16, 2, 8, 0.55)]
            for (ex, ey, eang, eop) in echoes:
                # undo echo translate+rotate to place the rect in local coords
                tx, ty = rot(X - ex, Y - ey, -eang)
                d = sd_rrect(tx, ty, 88, 36)
                sdf = 9.0  # stroke width of the outline
                cov = max(0.0, min(1.0, 1.0 - (abs(d) - sdf / 2)))  # ring band
                if cov > 0:
                    ncol = mix(BG, GLOW, 0.9 + 0.1 * (eop))
                    # thin crisp outline color
                    ring = mix(BG, (34, 233, 216), 1.0)
                    # combine: blend ring color by coverage*opacity
                    f = cov * eop
                    col = mix(col, ring, f)

            # main tilted panel
            tx, ty = rot(X, Y, 16.0)  # undo tilt (-16)
            d = sd_rrect(tx, ty, 88, 36)

            # glitch notch is ignored in simple raster (kept in svg only)
            outline = 9.0
            fill_cov = max(0.0, 1.0 - d)  # sdf<0 inside
            if fill_cov > 0 and d < -1:
                # interior: dark void teal-black (matches svg #0a2a2b), not solid
                pcol = (10, 42, 43)
                col = mix(col, pcol, min(fill_cov, 1.0))
            band_cov = max(0.0, min(1.0, 1.0 - (abs(d) - outline / 2)))
            if band_cov > 0:
                ring = mix(BG, (76, 255, 241), 1.0)
                col = mix(col, ring, band_cov)

            # glow rim around the panel
            if d > -4 and d < 24:
                gt = 1.0 - (d - (-4)) / 28.0
                gt = max(0.0, gt) * 0.35
                col = mix(col, GLOW, gt)

            # pivot point (below panel, at logo coords (0,104))
            pdis = math.hypot(X - 0, Y - 104)
            if pdis < 40:
                # halo
                hcol = mix(BG, GLOW, 0.45)
                col = mix(col, hcol, max(0.0, 1.0 - pdis / 40.0))
            if pdis < 16:
                col = mix(col, (5, 9, 12), 0.95)
            if pdis < 16:
                ring = mix(BG, (90, 255, 238), 1.0)
                col = mix(col, ring, max(0.0, min(1.0, 1.0 - pdis / 16.0)))
            if pdis < 5:
                col = mix(col, (183, 255, 248), 1.0)

            row.append((col[0], col[1], col[2], alpha))
        rows.append(row)

    # write raw RGBA or PNG
    if out_rgba:
        raw = bytearray()
        for y in range(size):
            for x in range(size):
                r, g, b, a = rows[y][x]
                raw += bytes((r, g, b, a))
        return bytes(raw)
    else:
        # build PNG
        rawidat = bytearray()
        for y in range(size):
            rawidat.append(0)
            for x in range(size):
                # convert linear-ish: keep as-is
                r, g, b, a = rows[y][x]
                rawidat += bytes((r, g, b, a))
        return _png(size, size, bytes(rawidat))

def _png(w, h, raw):
    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        c += struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff)
        return c
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)  # 8-bit RGBA
    idat = zlib.compress(raw, 9)
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")

if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "/mnt/c/Users/mrpisterino/dev/teeter/assets/logo.png"
    if out.endswith(".rgba"):
        with open(out, "wb") as f:
            f.write(build(SIZE, True))
        # also emit a size header for the exe
        print("saved rgba", out, "bytes:", SIZE * SIZE * 4)
    else:
        with open(out, "wb") as f:
            f.write(build(SIZE, False))
        print("saved png", out)
