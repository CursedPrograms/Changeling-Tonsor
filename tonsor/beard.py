"""beard.py - the barber's other half: colour a beard, paint stubble in, shave it off (moved here from Changeling
Cosmeta, so everything hair is in Tonsor).

    barber = Barber()
    face = barber.read(rgb)                 # dlib's 68 face points and BiSeNet's skin and lips
    painter = BeardPainter(face)
    out = painter.apply(Beard(colour=(True, "#8a8a88", 0.8)))

The beard area is the jaw from cheek to cheek under the chin, up to the nostrils, without the lips (dlib's 68
points). Inside it, "beard" is whatever is clearly darker than the face's own skin, so it works on any skin tone
and any beard. Both models are already here: dlib's shape predictor and BiSeNet come with HairFastGAN's models.

    colour   tint the beard and moustache (grey to black, a dyed red, bleached)
    stubble  an even shade with a fine grain over the area: a shadow of a beard
    shave    the face's own skin back over the beard (stubble and short beards; a full beard needs a
             generative edit: the AI beards, through AetherCanvas AI)
"""
import os
from dataclasses import dataclass

import cv2
import numpy as np

BEARD_COLOURS = {"Black": "#151210", "Dark brown": "#3a2618", "Brown": "#5e4028", "Ginger": "#a0502a",
                 "Blonde": "#b89060", "Grey": "#8a8a88", "White": "#d8d6d0"}
STUBBLE_COLOURS = {"Dark": "#1a1410", "Brown": "#3a2618", "Grey": "#6a6a68"}
BISENET_LABELS = ["background", "skin", "l_brow", "r_brow", "l_eye", "r_eye", "eye_g", "l_ear", "r_ear", "ear_r",
                  "nose", "mouth", "u_lip", "l_lip", "neck", "neck_l", "cloth", "hair", "hat"]
ID = {n: i for i, n in enumerate(BISENET_LABELS)}
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


@dataclass
class Beard:
    """What to do: each is (on, colour, strength); shave has no colour."""
    colour: tuple = (False, BEARD_COLOURS["Dark brown"], 0.7)
    stubble: tuple = (False, STUBBLE_COLOURS["Dark"], 0.5)
    shave: tuple = (False, None, 0.8)

    def anything(self):
        return self.colour[0] or self.stubble[0] or self.shave[0]


class BeardFace:
    def __init__(self, image, labels, points):
        self.image, self.labels, self.points = image, labels, points

    def mask(self, *names, feather=0):
        m = np.isin(self.labels, [ID[n] for n in names]).astype(np.float32)
        return cv2.GaussianBlur(m, (0, 0), feather) if feather else m

    @property
    def size(self):
        return float(np.ptp(self.points[:, 0]))


class Barber:
    """Reads a face for the beard tools: dlib's face finder and 68 points, BiSeNet's labels on the graphics card."""

    def __init__(self):
        import dlib
        import torch
        from .bisenet import BiSeNet
        from .models import folder
        root = folder()
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(os.path.join(root, "ShapeAdaptor", "shape_predictor_68_face_landmarks.dat"))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        net = BiSeNet(n_classes=19)
        net.load_state_dict(torch.load(os.path.join(root, "BiSeNet", "face_parsing_79999_iter.pth"), map_location="cpu",
                                       weights_only=True))
        self.net = net.to(self.device).eval()

    def read(self, rgb):
        """A BeardFace, or ValueError when there's no face."""
        h, w = rgb.shape[:2]
        k = min(1.0, 800 / max(h, w))                       # dlib finds faces quicker on a smaller copy
        small = cv2.resize(rgb, (int(w * k), int(h * k))) if k < 1 else rgb
        found = self.detector(small, 1)
        if not found:
            raise ValueError("No face found in the photo.")
        box = max(found, key=lambda r: r.width() * r.height())
        import dlib
        box = dlib.rectangle(int(box.left() / k), int(box.top() / k), int(box.right() / k), int(box.bottom() / k))
        shape = self.predictor(rgb, box)
        pts = np.array([(p.x, p.y) for p in shape.parts()], np.float32)
        return BeardFace(rgb, self._labels(rgb, pts), pts)

    def _labels(self, rgb, pts):
        """BiSeNet's labels in a close square around the face (as CelebA-HQ's faces are cropped), background elsewhere."""
        import torch
        h, w = rgb.shape[:2]
        x0, y0 = pts.min(0)
        x1, y1 = pts.max(0)
        side = int(max(x1 - x0, y1 - y0) * 1.9)
        cx, cy = int((x0 + x1) / 2), int((y0 + y1) / 2 - side * 0.08)
        left, top = cx - side // 2, cy - side // 2
        pad = max(0, -left, -top, left + side - w, top + side - h)
        padded = cv2.copyMakeBorder(rgb, pad, pad, pad, pad, cv2.BORDER_REPLICATE) if pad else rgb
        crop = padded[top + pad:top + pad + side, left + pad:left + pad + side]
        x = (cv2.resize(crop, (512, 512), interpolation=cv2.INTER_AREA).astype(np.float32) / 255 - MEAN) / STD
        with torch.inference_mode():
            t = torch.from_numpy(x.transpose(2, 0, 1)[None]).to(self.device)
            out = self.net(t)[0][0].argmax(0).byte().cpu().numpy()
        labels = np.zeros(padded.shape[:2], np.uint8)
        labels[top + pad:top + pad + side, left + pad:left + pad + side] = cv2.resize(out, (side, side),
                                                                                       interpolation=cv2.INTER_NEAREST)
        return labels[pad:pad + h, pad:pad + w] if pad else labels


# ---- the masks and the painting (the same maths as Changeling Cosmeta' had)
def rgb_of(hex_colour):
    h = hex_colour.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _lab(img):
    return cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)


def area_mask(face):
    """The whole beard area (0..1): dlib's jaw from the lower cheeks round the chin, up past the nostrils."""
    p = face.points
    cheek_r = p[3] + (p[31] - p[3]) * 0.35            # (dlib's 0..16 runs from the image's left side round the chin)
    cheek_l = p[13] + (p[35] - p[13]) * 0.35
    jaw = p[33] + (p[3:14] - p[33]) * 1.15          # a beard grows past the jaw line (and dlib's sits high on beards)
    poly = np.concatenate([jaw, [cheek_l, p[35], p[33], p[31], cheek_r]])
    m = np.zeros(face.image.shape[:2], np.float32)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1.0, cv2.LINE_AA)
    lips = np.zeros_like(m)
    cv2.fillPoly(lips, [np.round(p[48:60]).astype(np.int32)], 1.0, cv2.LINE_AA)
    lips = np.maximum(lips, face.mask("u_lip", "l_lip", "mouth"))
    m *= 1 - cv2.GaussianBlur(lips, (0, 0), 2)
    m *= 1 - face.mask("background", "cloth", feather=2)     # never the wall behind or the collar
    return cv2.GaussianBlur(m, (0, 0), max(1.0, face.size * 0.02))


def beard_mask(face, lab, area):
    """The beard itself, inside the area: darker than the face's skin (a dark beard on any skin), or lighter and
    greyer than it (a grey or white beard on darker skin)."""
    skin = face.mask("skin", "nose") > 0.5
    cheeks = skin & (area < 0.2)
    pick = cheeks if cheeks.any() else skin
    if not pick.any():
        pick = np.ones(area.shape, bool)
    L = lab[..., 0]
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    ref, ref_chroma = np.median(L[pick]), np.median(chroma[pick])
    dark = np.clip((ref - L - 6) / 18, 0, 1)
    grey = np.clip((L - ref - 4) / 16, 0, 1) * np.clip((ref_chroma * 0.75 - chroma) / (ref_chroma * 0.35 + 1), 0, 1)
    m = cv2.GaussianBlur(np.maximum(dark, grey) * area, (0, 0), max(1.0, face.size * 0.006))
    return np.clip(m * 1.5, 0, 1)


def stubble_texture(shape, size, seed=7):
    rng = np.random.default_rng(seed)
    grain = cv2.GaussianBlur(rng.random(shape[:2]).astype(np.float32), (0, 0), max(0.6, size * 0.0022))
    grain = (grain - grain.min()) / max(np.ptp(grain), 1e-6)
    return 0.65 + 0.35 * grain


def shave(img_rgb, mask, strength, skin):
    """The face's own skin over the beard: a normalised blur of the skin that isn't beard, with a little of the
    photo's own light and shade kept so it isn't flat."""
    src = img_rgb.astype(np.float32)
    clean = (skin * (1 - np.clip(mask * 3, 0, 1))).astype(np.float32)
    sigma = max(4.0, min(img_rgb.shape[:2]) * 0.02)
    num = cv2.GaussianBlur(src * clean[..., None], (0, 0), sigma)
    den = cv2.GaussianBlur(clean, (0, 0), sigma)[..., None]
    tone = num / np.maximum(den, 1e-4)
    light = cv2.GaussianBlur(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), sigma * 2)
    tone *= (0.85 + 0.15 * light / max(light.mean(), 1))[..., None]
    a = np.clip(mask * strength * 1.4, 0, 1)[..., None]
    return np.clip(src * (1 - a) + tone * a, 0, 255).astype(np.uint8)


def _tint(lab, mask, colour, strength, light=0.6, keep=0.35):
    if strength <= 0:
        return lab
    ys, xs = np.nonzero(mask > 0.002)
    if not len(ys):
        return lab
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    region, m = lab[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    sel = m > 0.05
    if not sel.any():
        return lab
    t = _lab(np.uint8([[rgb_of(colour)]]))[0, 0]
    wsum = m.sum()
    mean = (region * m[..., None]).sum((0, 1)) / wsum     # weighted by the mask: the beard's own colour, not the skin
    out = np.empty_like(region)
    out[..., 0] = region[..., 0] + (t[0] - mean[0]) * light
    out[..., 1] = t[1] + (region[..., 1] - mean[1]) * keep
    out[..., 2] = t[2] + (region[..., 2] - mean[2]) * keep
    a = np.clip(m * strength, 0, 1)[..., None]
    lab = lab.copy()
    lab[y0:y1, x0:x1] = region * (1 - a) + out * a
    return lab


class BeardPainter:
    """The masks once per picture; apply() is then quick enough for every slider move."""

    def __init__(self, face):
        self.face = face
        self.lab = _lab(face.image)
        self.area = area_mask(face)
        self.beard = beard_mask(face, self.lab, self.area)
        self.skin = face.mask("skin", "nose")
        self._shaved = {}
        self._stubble = None

    def apply(self, beard):
        if not beard.anything():
            return self.face.image
        lab = self.lab
        on, _, strength = beard.shave
        if on and strength > 0:
            key = round(strength, 2)
            if key not in self._shaved:
                self._shaved = {key: _lab(shave(self.face.image, self.beard, key, self.skin))}
            lab = self._shaved[key]
        on, colour, strength = beard.colour
        if on:
            lab = _tint(lab, self.beard, colour, strength, light=0.95, keep=0.5)
        on, colour, strength = beard.stubble
        if on and strength > 0:
            if self._stubble is None:
                self._stubble = stubble_texture(self.face.image.shape, self.face.size)
            lab = _tint(lab, self.area * self._stubble, colour, strength * 0.85, light=0.7, keep=0.4)
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return np.clip(out * 255, 0, 255).astype(np.uint8)
