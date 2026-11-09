"""patch_hairfast.py - make the HairFastGAN clone run on any PC: no compiler, models from where they are.

    python -m tonsor.patch_hairfast        (run.bat does it after cloning; doing it twice is harmless)

1. Its StyleGAN2 ops (fused_act, upfirdn2d, in three copies) are CUDA extensions compiled when first imported:
   that needs Visual Studio's C++ compiler and the CUDA toolkit, and fails on most PCs. These are the same ops in
   plain PyTorch (rosinality's own fallbacks): a little slower, the same results.
2. CLIP ViT-B/32 (two of its encoders use it) is kept in F:\\AIModels\\clip (TONSOR_CLIP overrides), so it's
   downloaded once for every app on the PC, not into each user's cache.
3. It reads its model files from pretrained_models/ next to its code: that's F:\\AIModels\\HairFastGAN when the
   models are there (see tonsor/models.py), so every path is turned into one under the folder models.py chose.
"""
import os
import re

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAIRFAST = os.path.join(HERE, "HairFastGAN")
OP_DIRS = ["models/stylegan2/op", "models/encoder4editing/models/stylegan2/op",
           "models/FeatureStyleEncoder/pixel2style2pixel/models/stylegan2/op"]
MARK = "# Changeling Tonsor: plain PyTorch"

FUSED_ACT = MARK + ''' (no compiled CUDA extension).
import torch
from torch import nn
from torch.nn import functional as F


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    rest_dim = [1] * (input.ndim - bias.ndim - 1)
    return F.leaky_relu(input + bias.view(1, bias.shape[0], *rest_dim), negative_slope=negative_slope) * scale
'''

UPFIRDN2D = MARK + ''' (no compiled CUDA extension).
import torch
from torch.nn import functional as F


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    return upfirdn2d_native(input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1])


def upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1):
    _, channel, in_h, in_w = input.shape
    input = input.reshape(-1, in_h, in_w, 1)
    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape

    out = input.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)
    out = F.pad(out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])
    out = out[:, max(-pad_y0, 0): out.shape[1] - max(-pad_y1, 0), max(-pad_x0, 0): out.shape[2] - max(-pad_x1, 0), :]

    out = out.permute(0, 3, 1, 2)
    out = out.reshape([-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w).to(out.dtype)
    out = F.conv2d(out, w)
    out = out.reshape(-1, minor, in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
                      in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1)
    out = out.permute(0, 2, 3, 1)
    out = out[:, ::down_y, ::down_x, :]
    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1
    return out.view(-1, channel, out_h, out_w)
'''

CLIP_LOAD = 'clip.load({model}, device="cuda", download_root=__import__("os").environ.get("TONSOR_CLIP", r"F:\\AIModels\\clip"))'


def patch():
    changed = 0
    for d in OP_DIRS:
        for name, text in (("fused_act.py", FUSED_ACT), ("upfirdn2d.py", UPFIRDN2D)):
            path = os.path.join(HAIRFAST, d, name)
            if not open(path, encoding="utf-8").read().startswith(MARK):
                open(path, "w", encoding="utf-8").write(text)
                changed += 1
    enc = os.path.join(HAIRFAST, "models", "Encoders.py")
    s = open(enc, encoding="utf-8").read()
    s2 = re.sub(r'clip\.load\((clip_model|"ViT-B/32"), device="cuda"\)', lambda m: CLIP_LOAD.format(model=m.group(1)), s)
    if s2 != s:
        open(enc, "w", encoding="utf-8").write(s2)
        changed += 1
    link_models()
    print(f"HairFastGAN patched ({changed} files changed)")


def link_models():
    import subprocess
    import sys
    sys.path.insert(0, HERE)
    from tonsor.models import folder
    target = os.path.abspath(folder())
    here = os.path.join(HAIRFAST, "pretrained_models")
    if os.path.normcase(target) == os.path.normcase(os.path.abspath(here)) or os.path.exists(here):
        return
    if os.name == "nt":
        subprocess.run(["cmd", "/c", "mklink", "/J", here, target], check=True, stdout=subprocess.DEVNULL)
    else:
        os.symlink(target, here, target_is_directory=True)
    print("pretrained_models ->", target)


if __name__ == "__main__":
    patch()
