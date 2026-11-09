"""models.py - HairFastGAN's model files: only what hairstyle transfer needs (about 5 GB, plus e4e).

They live in F:\\AIModels\\HairFastGAN\\pretrained_models when that drive is there (TONSOR_MODELS overrides), so
every app on this PC shares one copy; else in this repo's HairFastGAN/pretrained_models. e4e (1.1 GB) is the same
file Changeling Tiresias, Chronos and Natio use: it's copied from them instead of downloaded.

    python -m tonsor.models      fetch whatever is missing
"""
import http.client
import os
import shutil
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "https://huggingface.co/AIRI-Institute/HairFastGAN/resolve/main/pretrained_models"
SHARED = r"F:\AIModels\HairFastGAN\pretrained_models"
FILES = [
    "StyleGAN/ffhq.pt",
    "FeatureStyleEncoder/psp_ffhq_encode.pt",
    "FeatureStyleEncoder/143_enc.pth",
    "FeatureStyleEncoder/backbone.pth",
    "FeatureStyleEncoder/79999_iter.pth",
    "Rotate/rotate_best.pth",
    "Blending/checkpoint.pth",
    "PostProcess/pp_model.pth",
    "PostProcess/latent_avg.pt",
    "ShapeAdaptor/mask_generator.pth",
    "ShapeAdaptor/shape_predictor_68_face_landmarks.dat",
    "BiSeNet/face_parsing_79999_iter.pth",
    "BiSeNet/seg.pth",
    "ArcFace/backbone_ir50.pth",
    "STAR/WFLW_STARLoss_NME_4_02_FR_2_32_AUC_0_605.pkl",
    "encoder4editing/e4e_ffhq_encode.pt",
    "sean_checkpoints/CelebA-HQ_pretrained/latest_net_G.pth",
]
# where e4e already is (Tiresias, Chronos, Natio)
E4E = [os.path.join(os.path.dirname(HERE), name, "pretrained", "e4e_ffhq_encode.pt")
       for name in ("Changeling-Tiresias", "Changeling-Chronos", "Changeling-Natio")]


def folder():
    override = os.environ.get("TONSOR_MODELS")
    if override:
        return override
    if os.path.isdir(os.path.dirname(os.path.dirname(SHARED))):
        return SHARED
    return os.path.join(HERE, "HairFastGAN", "pretrained_models")


def missing():
    return [f for f in FILES if not os.path.exists(os.path.join(folder(), f))]


def _download(url, path, tries=30):
    part = path + ".part"
    for attempt in range(tries):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
            with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab" if have else "wb") as out:
                if have and r.status != 206:
                    out.truncate(0)
                    have = 0
                total = have + int(r.headers.get("Content-Length", 0))
                last = 0.0
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    have += len(chunk)
                    if time.monotonic() - last > 15:
                        last = time.monotonic()
                        print(f"  {os.path.basename(path)}: {have / 2**20:.0f} / {total / 2**20:.0f} MB", flush=True)
            if total and have < total:
                raise OSError(f"stopped at {have} of {total} bytes")
            os.replace(part, path)
            return
        except (OSError, urllib.error.URLError, http.client.HTTPException) as e:
            if attempt == tries - 1:
                raise
            print(f"  {os.path.basename(path)}: {e}; trying again ({attempt + 2}/{tries})", flush=True)
            time.sleep(min(60, 3 * (attempt + 1)))


def fetch():
    root = folder()
    for f in missing():
        path = os.path.join(root, f)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        local = next((p for p in E4E if f.endswith("e4e_ffhq_encode.pt") and os.path.exists(p)), None)
        if local:
            print(f"{f}: copying from {os.path.dirname(local)}", flush=True)
            shutil.copy(local, path)
        else:
            print(f"{f}: downloading", flush=True)
            _download(f"{BASE}/{f}", path)
    print("all there:", root, flush=True)


if __name__ == "__main__":
    fetch()
    sys.exit(0)
