"""Changeling Tonsor: anyone's hairstyle on you (HairFastGAN).

    python main.py                                   the window
    python main.py you.jpg hairstyle.jpg -o out.png  on the command line (--colour another.jpg for the colour)
    python main.py --download-models                 fetch what's missing and stop
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def read_rgb(path):
    import cv2
    import numpy as np
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def write_rgb(path, image):
    import cv2
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise SystemExit(f"Could not write image: {path}")
    buf.tofile(path)


def main():
    ap = argparse.ArgumentParser(description="Changeling Tonsor: anyone's hairstyle on you (HairFastGAN).")
    ap.add_argument("images", nargs="*", help="you.jpg hairstyle.jpg (omit to open the window)")
    ap.add_argument("-o", "--output", help="where to save (default: <you>_with_<hairstyle>.png)")
    ap.add_argument("--colour", "--color", help="take the hair colour from this photo instead of the hairstyle's")
    ap.add_argument("--device", default="cuda", help="cuda (HairFastGAN needs an NVIDIA GPU)")
    ap.add_argument("--download-models", action="store_true", help="fetch any missing models and stop")
    args = ap.parse_args()

    from tonsor import models, patch_hairfast
    if models.missing():
        models.fetch()
    patch_hairfast.patch()
    if args.download_models:
        return

    from tonsor.pipeline import Tonsor
    if not args.images:
        from window import Window
        Window(Tonsor, args.device).run()
        return
    if len(args.images) != 2:
        raise SystemExit("Give two photos: you, then the hairstyle.")
    you, style = args.images
    out = args.output or os.path.join(os.path.dirname(you) or ".", "{}_with_{}.png".format(
        os.path.splitext(os.path.basename(you))[0], os.path.splitext(os.path.basename(style))[0]))
    t0 = time.time()
    tonsor = Tonsor(args.device, progress=lambda text: print(text, flush=True))
    print(f"  loaded in {time.time() - t0:.0f} s", flush=True)
    t0 = time.time()
    result, _ = tonsor.transfer(read_rgb(you), read_rgb(style), read_rgb(args.colour) if args.colour else None)
    write_rgb(out, result)
    print(f"  saved {out} ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
