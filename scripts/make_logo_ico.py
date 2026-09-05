#!/usr/bin/env python3
"""Pack multi-size PNG-encoded ICO entries (16,32,48,64,128,256) into logo.ico.
Windows reads PNG-compressed icon entries for sizes >= 256 and is tolerant of
them for smaller sizes on modern builds. Reuses make_logo.build() for pixels."""
import io, struct, sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_logo

SIZES = [16, 32, 48, 64, 128, 256]


def main(out):
    pngs = {s: make_logo.build(s, False) for s in SIZES}
    # keep a copy of raw 256 RGBA for the compiler resource generator later
    rgba256 = make_logo.build(256, True)

    headers = b""
    blobs = b""
    offset = 6 + 16 * len(SIZES)
    for s in SIZES:
        png = pngs[s]
        headers += struct.pack("<BBBBHHII", s % 256, s % 256, 0, 0, 1, 32, len(png), offset)
        blobs += png
        offset += len(png)
    with open(out, "wb") as f:
        f.write(struct.pack("<HHH", 0, 1, len(SIZES)))
        f.write(headers)
        f.write(blobs)
    print("saved ico", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/mnt/c/Users/mrpisterino/dev/teeter/assets/logo.ico")
