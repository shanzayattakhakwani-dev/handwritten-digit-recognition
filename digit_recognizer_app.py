import tkinter as tk
from tkinter import font as tkfont
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import os

# ── Try loading the model ──────────────────────────────────────────────────────
try:
    from tensorflow.keras.models import load_model
    MODEL_PATH = "digit_model.keras"
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        MODEL_LOADED = True
    else:
        model = None
        MODEL_LOADED = False
except Exception as e:
    model = None
    MODEL_LOADED = False

# ── Colour palette ─────────────────────────────────────────────────────────────
BG          = "#0D0D0D"
CANVAS_BG   = "#1A1A2E"
ACCENT      = "#00F5D4"
ACCENT2     = "#F72585"
PANEL_BG    = "#111122"
TEXT_LIGHT  = "#E0E0FF"
TEXT_DIM    = "#555577"
BAR_BASE    = "#1E1E3A"

# ── App ────────────────────────────────────────────────────────────────────────
class DigitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognizer")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)

        self.canvas_size = 280       # display canvas (px)
        self.mnist_size  = 28        # model expects 28×28

        self.pil_image = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        self.pil_draw  = ImageDraw.Draw(self.pil_image)

        self._build_ui()
        self._reset_bars()

        if not MODEL_LOADED:
            self._set_status(
                "⚠  digit_model.keras not found — place it in the same folder and restart.",
                color=ACCENT2
            )

    # ── UI construction ────────────────────────────────────────────────────────
    def _build_ui(self):
        # Title bar
        title_frame = tk.Frame(self.root, bg=BG, pady=14)
        title_frame.pack(fill="x")

        tk.Label(
            title_frame,
            text="DIGIT  RECOGNIZER",
            font=("Courier", 18, "bold"),
            fg=ACCENT, bg=BG
        ).pack()
        tk.Label(
            title_frame,
            text="Draw a digit (0–9) and press  P R E D I C T",
            font=("Courier", 9),
            fg=TEXT_DIM, bg=BG
        ).pack()

        # ── Main row ──
        main = tk.Frame(self.root, bg=BG, padx=20, pady=4)
        main.pack()

        # Drawing canvas
        canvas_frame = tk.Frame(main, bg=ACCENT, padx=2, pady=2)
        canvas_frame.grid(row=0, column=0, padx=(0, 20))

        self.canvas = tk.Canvas(
            canvas_frame,
            width=self.canvas_size,
            height=self.canvas_size,
            bg=CANVAS_BG,
            cursor="crosshair",
            highlightthickness=0
        )
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>",       self._on_draw)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        # Right panel
        right = tk.Frame(main, bg=BG)
        right.grid(row=0, column=1, sticky="ns")

        # Prediction box
        pred_frame = tk.Frame(right, bg=PANEL_BG, padx=20, pady=16,
                              highlightbackground=ACCENT, highlightthickness=1)
        pred_frame.pack(fill="x", pady=(0, 12))

        tk.Label(pred_frame, text="PREDICTION", font=("Courier", 9, "bold"),
                 fg=TEXT_DIM, bg=PANEL_BG).pack()

        self.pred_label = tk.Label(
            pred_frame, text="?",
            font=("Courier", 72, "bold"),
            fg=ACCENT, bg=PANEL_BG, width=3
        )
        self.pred_label.pack()

        self.conf_label = tk.Label(
            pred_frame, text="confidence: —",
            font=("Courier", 10),
            fg=TEXT_DIM, bg=PANEL_BG
        )
        self.conf_label.pack()

        # Probability bars
        bars_frame = tk.Frame(right, bg=PANEL_BG, padx=14, pady=12,
                              highlightbackground=TEXT_DIM, highlightthickness=1)
        bars_frame.pack(fill="x", pady=(0, 12))

        tk.Label(bars_frame, text="CLASS PROBABILITIES", font=("Courier", 8, "bold"),
                 fg=TEXT_DIM, bg=PANEL_BG).pack(anchor="w", pady=(0, 6))

        self.bar_frames  = []
        self.bar_fills   = []
        self.bar_pct_lbs = []

        BAR_W = 160
        for i in range(10):
            row = tk.Frame(bars_frame, bg=PANEL_BG)
            row.pack(fill="x", pady=1)

            tk.Label(row, text=str(i), font=("Courier", 9, "bold"),
                     fg=TEXT_LIGHT, bg=PANEL_BG, width=2).pack(side="left")

            bg_bar = tk.Frame(row, bg=BAR_BASE, width=BAR_W, height=12)
            bg_bar.pack(side="left", padx=4)
            bg_bar.pack_propagate(False)

            fill = tk.Frame(bg_bar, bg=ACCENT, width=0, height=12)
            fill.place(x=0, y=0, height=12)

            pct_lb = tk.Label(row, text="0%", font=("Courier", 8),
                              fg=TEXT_DIM, bg=PANEL_BG, width=5)
            pct_lb.pack(side="left")

            self.bar_frames.append((bg_bar, BAR_W))
            self.bar_fills.append(fill)
            self.bar_pct_lbs.append(pct_lb)

        # Buttons
        btn_frame = tk.Frame(right, bg=BG)
        btn_frame.pack(fill="x")

        self._make_btn(btn_frame, "PREDICT", self._predict, ACCENT,   "#0D0D0D").pack(fill="x", pady=(0, 6))
        self._make_btn(btn_frame, "CLEAR",   self._clear,   ACCENT2,  "#0D0D0D").pack(fill="x")

        # Status bar
        self.status_var = tk.StringVar(value="Ready — draw something!")
        self.status_lbl = tk.Label(
            self.root,
            textvariable=self.status_var,
            font=("Courier", 9),
            fg=TEXT_DIM, bg=BG, pady=10
        )
        self.status_lbl.pack()

    def _make_btn(self, parent, text, cmd, bg, fg):
        return tk.Button(
            parent, text=text, command=cmd,
            font=("Courier", 11, "bold"),
            bg=bg, fg=fg,
            activebackground=fg, activeforeground=bg,
            relief="flat", padx=16, pady=8,
            cursor="hand2", bd=0
        )

    # ── Drawing logic ──────────────────────────────────────────────────────────
    def _on_draw(self, event):
        r = 10
        x, y = event.x, event.y
        self.canvas.create_oval(x-r, y-r, x+r, y+r,
                                fill="white", outline="white")
        self.pil_draw.ellipse([x-r, y-r, x+r, y+r], fill=255)

    def _on_release(self, event):
        self._set_status("Digit drawn — press PREDICT when ready.")

    # ── Prediction ─────────────────────────────────────────────────────────────
    def _predict(self):
        if not MODEL_LOADED or model is None:
            self._set_status("⚠  No model loaded. Put digit_model.keras here and restart.", ACCENT2)
            return

        # Preprocess: blur slightly, resize to 28×28, normalise
        img = self.pil_image.filter(ImageFilter.GaussianBlur(radius=1))
        img = img.resize((self.mnist_size, self.mnist_size), Image.LANCZOS)
        arr = np.array(img) / 255.0
        arr = arr.reshape(1, self.mnist_size, self.mnist_size, 1)

        probs  = model.predict(arr, verbose=0)[0]   # shape (10,)
        digit  = int(np.argmax(probs))
        conf   = float(probs[digit]) * 100

        self.pred_label.config(text=str(digit))
        self.conf_label.config(text=f"confidence: {conf:.1f}%",
                               fg=ACCENT if conf >= 70 else ACCENT2)

        self._update_bars(probs, digit)
        self._set_status(f"Predicted  {digit}  with {conf:.1f}% confidence.", ACCENT)

    def _update_bars(self, probs, winner):
        for i, prob in enumerate(probs):
            bg_bar, max_w = self.bar_frames[i]
            w = int(prob * max_w)
            color = ACCENT if i == winner else "#334466"
            self.bar_fills[i].place(x=0, y=0, width=w, height=12)
            self.bar_fills[i].config(bg=color)
            self.bar_pct_lbs[i].config(
                text=f"{prob*100:.1f}%",
                fg=ACCENT if i == winner else TEXT_DIM
            )

    # ── Clear ──────────────────────────────────────────────────────────────────
    def _clear(self):
        self.canvas.delete("all")
        self.pil_image = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        self.pil_draw  = ImageDraw.Draw(self.pil_image)
        self.pred_label.config(text="?", fg=ACCENT)
        self.conf_label.config(text="confidence: —", fg=TEXT_DIM)
        self._reset_bars()
        self._set_status("Canvas cleared — draw a digit!")

    def _reset_bars(self):
        for i in range(10):
            self.bar_fills[i].place(x=0, y=0, width=0, height=12)
            self.bar_pct_lbs[i].config(text="0%", fg=TEXT_DIM)

    def _set_status(self, msg, color=None):
        self.status_var.set(msg)
        self.status_lbl.config(fg=color or TEXT_DIM)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = DigitApp(root)
    root.mainloop()
