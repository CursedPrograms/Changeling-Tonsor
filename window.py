"""window.py - Changeling Tonsor's window, in Changeling's look (customtkinter, the Cursed pink).

Open a photo of you, then a hairstyle: any photo of someone whose hair you want (or one from hairstyles/). Its
colour comes with it, or from a third photo. Cut (Enter) runs on its own thread with a completion bar; the result is
the face frame, as the other Changelings' are. Save writes it next to your photo, or wherever you choose.
"""
import os
import threading
import time
import tkinter as tk
import webbrowser
from tkinter import filedialog

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = "Changeling Tonsor"
TAGLINE = "the barber: anyone's hairstyle, on you"
GITHUB = "https://github.com/CursedPrograms/Changeling-Tonsor"
KOFI = "https://ko-fi.com/cursedentertainment"
PINK = "#e0408a"
HAIRSTYLES = os.path.join(HERE, "hairstyles")
IMAGE_TYPES = [("Pictures", "*.png *.jpg *.jpeg *.webp *.bmp *.jfif"), ("All files", "*.*")]
PICTURES = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".jfif")


def read_rgb(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return None if image is None else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def write_rgb(path, image):
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if ok:
        buf.tofile(path)
    return ok


def set_logo(root, here):
    """The Changeling logo as the window's icon. customtkinter puts its own in at about 200 ms, so this is later."""
    ico, png = os.path.join(here, "logo.ico"), os.path.join(here, "logo.png")

    def apply():
        try:
            if os.name == "nt" and os.path.exists(ico):
                root.iconbitmap(ico)
            elif os.path.exists(png):
                root._logo_icon = tk.PhotoImage(file=png)
                root.iconphoto(True, root._logo_icon)
        except tk.TclError:
            pass
    root.after(300, apply)


def hairstyle_files():
    if not os.path.isdir(HAIRSTYLES):
        return []
    return sorted(f for f in os.listdir(HAIRSTYLES) if f.lower().endswith(PICTURES))


class Window:
    def __init__(self, make_tonsor, device=None):
        self.make_tonsor = make_tonsor
        self.device = device
        self.tonsor = None
        self.photos = {"you": None, "style": None, "colour": None}     # RGB
        self.paths = {"you": None, "style": None, "colour": None}
        self.result = None
        self.job = None
        self.state = {"busy": False, "stage": "", "t0": 0.0, "message": ""}
        self.lock = threading.Lock()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme(os.path.join(HERE, "theme.json"))
        self.root = ctk.CTk()
        set_logo(self.root, HERE)
        self.root.title(NAME)
        self.root.geometry("1280x800")
        self.root.minsize(1000, 640)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        self._build()
        self.root.bind("<Control-o>", lambda e: self.open("you"))
        self.root.bind("<Control-s>", lambda e: self.save())
        self.root.bind("<Return>", lambda e: self.cut())
        self.root.bind("<Escape>", lambda e: self.root.destroy())
        self.root.after(100, self.watch)
        self._run(self._load_models, "Loading HairFastGAN (once, about half a minute)")

    # ---- the window
    def _build(self):
        stage = ctk.CTkFrame(self.root, corner_radius=12)
        stage.grid(row=0, column=0, sticky="nsew", padx=(14, 7), pady=14)
        stage.grid_columnconfigure((0, 1), weight=1, uniform="half")
        stage.grid_rowconfigure(1, weight=1)
        for col, text in enumerate(("You", "After")):
            ctk.CTkLabel(stage, text=text.upper(), text_color="gray50",
                         font=ctk.CTkFont(size=11, weight="bold")).grid(row=0, column=col, pady=(8, 0))
        self.before = tk.Label(stage, bg="#16161a", bd=0, text="Open a photo of you  (Ctrl+O)", fg="gray60",
                               font=("Segoe UI", 12), cursor="hand2")
        self.before.grid(row=1, column=0, sticky="nsew", padx=(8, 4), pady=8)
        self.before.bind("<Button-1>", lambda e: self.open("you"))
        self.after_view = tk.Label(stage, bg="#16161a", bd=0, fg="gray60", font=("Segoe UI", 12))
        self.after_view.grid(row=1, column=1, sticky="nsew", padx=(4, 8), pady=8)
        self.before.bind("<Configure>", lambda e: self.root.after_idle(self._refresh))

        side = ctk.CTkScrollableFrame(self.root, width=300, corner_radius=12)
        side.grid(row=0, column=1, sticky="ns", padx=(7, 14), pady=14)
        logo = os.path.join(HERE, "logo.png")
        if os.path.exists(logo):
            ctk.CTkLabel(side, image=ctk.CTkImage(Image.open(logo), size=(56, 56)), text="").pack(anchor="w", padx=12, pady=(10, 0))
        ctk.CTkLabel(side, text="CHANGELING", text_color="gray55",
                     font=ctk.CTkFont(size=11, weight="bold")).pack(anchor="w", padx=12, pady=(12, 0))
        ctk.CTkLabel(side, text="Tonsor", font=ctk.CTkFont(size=24, weight="bold"), text_color=PINK).pack(anchor="w", padx=12)
        ctk.CTkLabel(side, text=TAGLINE, text_color="gray60", wraplength=270, justify="left").pack(anchor="w", padx=12)

        ctk.CTkButton(side, text="Open photo of you  (Ctrl+O)", command=lambda: self.open("you")).pack(fill="x", padx=12, pady=(16, 4))

        self._section(side, "The hairstyle")
        self.thumbs = {}
        self.thumbs["style"] = ctk.CTkLabel(side, text="no hairstyle yet", width=140, height=140, fg_color="#16161a",
                                            corner_radius=8, cursor="hand2")
        self.thumbs["style"].pack(anchor="w", padx=12, pady=4)
        self.thumbs["style"].bind("<Button-1>", lambda e: self.open("style"))
        row = ctk.CTkFrame(side, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=2)
        ctk.CTkButton(row, text="Pick a photo ...", command=lambda: self.open("style")).pack(side="left", expand=True, fill="x")
        styles = hairstyle_files()
        self.style_pick = ctk.CTkOptionMenu(side, values=styles or ["(hairstyles/ is empty)"], command=self.pick_style)
        self.style_pick.set("From hairstyles/ ..." if styles else "(hairstyles/ is empty)")
        self.style_pick.pack(fill="x", padx=12, pady=4)
        ctk.CTkLabel(side, text="Any photo of someone whose hair you want, face to the camera. Put favourites in "
                     "hairstyles/.", text_color="gray55", wraplength=270, justify="left").pack(anchor="w", padx=12)

        self._section(side, "The colour")
        self.colour_mode = ctk.CTkSegmentedButton(side, values=["The hairstyle's", "Another photo's"],
                                                  command=self._colour_mode)
        self.colour_mode.set("The hairstyle's")
        self.colour_mode.pack(fill="x", padx=12, pady=4)
        self.thumbs["colour"] = ctk.CTkLabel(side, text="", width=90, height=90, fg_color="#16161a", corner_radius=8,
                                             cursor="hand2")
        self.thumbs["colour"].bind("<Button-1>", lambda e: self.open("colour"))

        self.cut_btn = ctk.CTkButton(side, text="Cut  (Enter)", height=42, font=ctk.CTkFont(size=15, weight="bold"),
                                     command=self.cut)
        self.cut_btn.pack(fill="x", padx=12, pady=(18, 6))
        self.bar = ctk.CTkProgressBar(side, mode="determinate")
        self.bar.set(0)
        self.bar.pack(fill="x", padx=12, pady=(4, 2))
        self.status = ctk.CTkLabel(side, text="", text_color="gray65", wraplength=270, justify="left")
        self.status.pack(anchor="w", padx=12)
        self.save_btn = ctk.CTkButton(side, text="Save  (Ctrl+S)", command=self.save, state="disabled")
        self.save_btn.pack(fill="x", padx=12, pady=(10, 0))

        foot = ctk.CTkFrame(side, fg_color="transparent")
        foot.pack(fill="x", padx=12, pady=(18, 6))
        links = ctk.CTkFrame(foot, fg_color="transparent")
        links.pack(anchor="w")
        for text, url in (("GitHub", GITHUB), ("Ko-fi", KOFI)):
            link = ctk.CTkLabel(links, text=text, text_color=("#c2186b", "#f29cc0"), cursor="hand2")
            link.pack(side="left", padx=(0, 14))
            link.bind("<Button-1>", lambda e, u=url: webbrowser.open(u))
        ctk.CTkLabel(foot, text="© Cursed Entertainment 2026", text_color="gray45").pack(anchor="w")

    def _section(self, parent, text):
        ctk.CTkLabel(parent, text=text.upper(), text_color="gray50",
                     font=ctk.CTkFont(size=11, weight="bold")).pack(anchor="w", padx=12, pady=(16, 1))

    def _colour_mode(self, mode):
        if mode == "Another photo's":
            self.thumbs["colour"].pack(anchor="w", padx=12, pady=4, after=self.colour_mode)
            if self.photos["colour"] is None:
                self.thumbs["colour"].configure(text="click: a photo\nfor the colour")
        else:
            self.thumbs["colour"].pack_forget()

    def _show(self, label, image):
        if image is None:
            label.configure(image="")
            label.image = None
            return
        w, h = max(label.winfo_width(), 64), max(label.winfo_height(), 64)
        ih, iw = image.shape[:2]
        k = min(w / iw, h / ih)
        small = cv2.resize(image, (max(int(iw * k), 1), max(int(ih * k), 1)), interpolation=cv2.INTER_AREA)
        photo = ImageTk.PhotoImage(Image.fromarray(small))
        label.configure(image=photo, text="")
        label.image = photo

    def _thumb(self, key, image, size):
        h, w = image.shape[:2]
        s = min(h, w)
        top = max(0, int((h - s) * 0.25))
        crop = image[top:top + s, (w - s) // 2:(w - s) // 2 + s]
        self.thumbs[key].configure(image=ctk.CTkImage(Image.fromarray(crop), size=(size, size)), text="")

    def _refresh(self):
        self._show(self.before, self.photos["you"])
        self._show(self.after_view, self.result)

    # ---- the job, on its own thread; the window follows self.state
    def _run(self, work, stage):
        with self.lock:
            self.state.update(busy=True, stage=stage, t0=time.monotonic(), message="")
        self.cut_btn.configure(state="disabled")
        self.job = threading.Thread(target=self._guard, args=(work,), daemon=True)
        self.job.start()

    def _guard(self, work):
        try:
            message = work()
        except Exception as e:  # noqa: BLE001 - shown in the window
            message = f"Couldn't: {e}"
        with self.lock:
            self.state.update(busy=False, message=message or "")

    def _stage(self, text):
        with self.lock:
            self.state["stage"] = text

    def watch(self):
        with self.lock:
            s = dict(self.state)
        if s["busy"]:
            if self.bar.cget("mode") != "indeterminate":
                self.bar.configure(mode="indeterminate")
                self.bar.start()
            self.status.configure(text=f"{s['stage']} ... {int(time.monotonic() - s['t0'])} s")
        elif self.job is not None and not self.job.is_alive():
            self.job = None
            self.bar.stop()
            self.bar.configure(mode="determinate")
            self.bar.set(1 if self.result is not None else 0)
            self.status.configure(text=s["message"])
            self.cut_btn.configure(state="normal" if self.tonsor else "disabled")
            self.save_btn.configure(state="normal" if self.result is not None else "disabled")
            self._refresh()
        self.root.after(100, self.watch)

    def _load_models(self):
        self.tonsor = self.make_tonsor(self.device or "cuda", progress=self._stage)
        return "Ready: a photo of you, then a hairstyle."

    # ---- what the buttons do
    def open(self, key, path=None):
        titles = {"you": "A photo of you", "style": "The hairstyle: a photo of someone with it",
                  "colour": "The hair colour: a photo of someone with it"}
        path = path or filedialog.askopenfilename(title=titles[key], filetypes=IMAGE_TYPES,
                                                  initialdir=HAIRSTYLES if key != "you" and os.path.isdir(HAIRSTYLES) else None)
        if not path:
            return
        image = read_rgb(path)
        if image is None:
            self.status.configure(text=f"Can't read {os.path.basename(path)}")
            return
        self.photos[key], self.paths[key] = image, path
        if key == "you":
            self.result = None
            self.save_btn.configure(state="disabled")
            self._refresh()
        else:
            self._thumb(key, image, 140 if key == "style" else 90)
        if self.job is None:
            self.status.configure(text="Now: Cut (Enter)." if self.photos["you"] is not None and
                                  self.photos["style"] is not None else "")

    def pick_style(self, name):
        path = os.path.join(HAIRSTYLES, name)
        if os.path.exists(path):
            self.open("style", path)

    def cut(self):
        if self.tonsor is None or self.job is not None:
            return
        if self.photos["you"] is None:
            self.open("you")
            return
        if self.photos["style"] is None:
            self.open("style")
            return
        colour = self.photos["colour"] if self.colour_mode.get() == "Another photo's" else None
        you, style = self.photos["you"], self.photos["style"]

        def work():
            t0 = time.monotonic()
            self.result, _ = self.tonsor.transfer(you, style, colour)
            return f"Done in {time.monotonic() - t0:.0f} s."
        self.result = None
        self._run(work, "Finding the faces, moving the hair, blending")

    def save(self):
        if self.result is None:
            return
        stem = os.path.splitext(os.path.basename(self.paths["you"] or "photo.png"))[0]
        style = os.path.splitext(os.path.basename(self.paths["style"] or "hair"))[0][:40]
        path = filedialog.asksaveasfilename(title="Save the result", defaultextension=".png",
                                            initialdir=os.path.dirname(self.paths["you"] or HERE),
                                            initialfile=f"{stem}_with_{style}.png",
                                            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if path and write_rgb(path, self.result):
            self.status.configure(text=f"Saved {os.path.basename(path)}")

    def run(self):
        self.root.mainloop()
