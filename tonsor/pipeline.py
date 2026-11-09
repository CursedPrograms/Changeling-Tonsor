"""pipeline.py - Changeling Tonsor's hairstyle transfer: HairFastGAN (AIRI Institute, MIT), wrapped.

    from tonsor.pipeline import Tonsor
    t = Tonsor()
    result = t.transfer(you, hairstyle, colour=None)     # RGB numpy arrays in, a 1024 x 1024 RGB face frame out

`hairstyle` gives the hair's shape (the cut, the length, the parting); `colour` gives its colour, and is the same
picture when None. Each photo is found, aligned and cropped to the face (as FFHQ is), so the result is that face
frame, as Changeling Tiresias's and Chronos's are. HairFastGAN works in StyleGAN2's FS space: it encodes the three
faces, moves the hair's shape from one and its colour from the other onto yours, and blends. A few seconds a face
on a GPU, nothing learned per photo.
"""
import os
import sys
import time
from argparse import Namespace

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAIRFAST = os.path.join(HERE, "HairFastGAN")


def _args(device):
    return Namespace(save_all_dir="output", size=1024, ckpt="pretrained_models/StyleGAN/ffhq.pt", channel_multiplier=2,
                     latent=512, n_mlp=8, device=device, batch_size=3, save_all=False, mixing=0.95, smooth=5,
                     rotate_checkpoint="pretrained_models/Rotate/rotate_best.pth",
                     blending_checkpoint="pretrained_models/Blending/checkpoint.pth",
                     pp_checkpoint="pretrained_models/PostProcess/pp_model.pth")


class Tonsor:
    def __init__(self, device="cuda", progress=None):
        say = progress or (lambda text: None)
        say("Loading HairFastGAN")
        if HAIRFAST not in sys.path:
            sys.path.insert(0, HAIRFAST)
        self._cwd = os.getcwd()
        os.chdir(HAIRFAST)                          # it reads pretrained_models/ from its own folder
        try:
            from hair_swap import HairFast
            self.model = HairFast(_args(device))
        finally:
            os.chdir(self._cwd)
        self.device = device

    def transfer(self, you, hairstyle, colour=None, seed=3407):
        """(result, your aligned face) as 1024 x 1024 RGB uint8 arrays."""
        import torch
        import torchvision.transforms.functional as F
        from PIL import Image

        tensors = [F.to_tensor(Image.fromarray(np.ascontiguousarray(img))) for img in
                   (you, hairstyle, hairstyle if colour is None else colour)]
        os.chdir(HAIRFAST)
        try:
            with torch.inference_mode():
                out = self.model.swap(*tensors, align=True, seed=seed)
        finally:
            os.chdir(self._cwd)
        final, aligned_you = out[0], out[1]

        def to_rgb(t):
            t = t.detach().float().cpu()
            if t.ndim == 4:
                t = t[0]
            if t.min() < 0:                         # -1 .. 1
                t = (t + 1) / 2
            elif t.max() > 1.5:                     # 0 .. 255
                t = t / 255
            return (t.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        return to_rgb(final), to_rgb(aligned_you)


if __name__ == "__main__":       # python -m tonsor.pipeline you.jpg hairstyle.jpg [colour.jpg] out.jpg
    import cv2

    def read(p):
        return cv2.cvtColor(cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    paths = sys.argv[1:]
    t0 = time.time()
    t = Tonsor(progress=print)
    print(f"loaded in {time.time() - t0:.0f} s")
    t0 = time.time()
    colour = read(paths[2]) if len(paths) == 4 else None
    result, _ = t.transfer(read(paths[0]), read(paths[1]), colour)
    cv2.imwrite(paths[-1], cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    print(f"done in {time.time() - t0:.1f} s: {paths[-1]}")
