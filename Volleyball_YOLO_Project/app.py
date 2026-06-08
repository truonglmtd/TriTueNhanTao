"""
Volleyball YOLO Detection - Modern Dashboard UI (CustomTkinter)

Features:
- Modern light theme dashboard using CustomTkinter
- Left sidebar (upload, params, filters)
- Right content with tabs: Bounding Box, Raw Labels, Save
- Stats cards and detection table
- YOLOv8 integration (ultralytics.YOLO)
- Runs detection in a background thread
- Light/Dark mode toggle

Note: Drag & drop is represented by the upload area; native OS drag-drop may need extra packages.

"""

import threading
import io
import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import tkinter.font as tkfont
import customtkinter as ctk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np
import cv2
import json

# Try to import YOLO model
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# ----------------- UI Constants -----------------
PRIMARY = "#4F7CFF"      # Blue
ACCENT = "#8B5CF6"       # Light purple
BG_LIGHT = "#FFFFFF"
BG_PANEL = "#F5F7FA"
CARD_BG = "#FFFFFF"
TEXT = "#0f172a"
RADIUS = 14
FONT_FAMILY = "Segoe UI"
# Shared font variables (will be initialized per-application)
DEFAULT_FONT = None
TITLE_FONT = None
LOGO_FONT = None
BUTTON_FONT = None
STAT_FONT = None
TABLE_FONT = None
STATUS_FONT = None

# ----------------- Helpers copied from existing app -----------------

def box_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter_area == 0.0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter_area / (area_a + area_b - inter_area)


def class_nms(filtered_boxes, iou_thresh=0.45):
    keep = []
    boxes = sorted(filtered_boxes, key=lambda x: x[4], reverse=True)
    while boxes:
        current = boxes.pop(0)
        keep.append(current)
        boxes = [b for b in boxes if box_iou(current, b) < iou_thresh]
    return keep


# Default thresholds for label filtering
CLASS_THRESHOLDS = {"Player": 0.18, "Referee": 0.20, "Ball": 0.05}
CLASS_MIN_AREA = {"Player": 3800, "Referee": 3000, "Ball": 150}


def filter_detections(boxes, names, default_conf, image_height=None, image_width=None):
    filtered = []
    for box in boxes:
        cls_id = int(box.cls[0])
        label = names[cls_id]
        score = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        area = width * height
        min_conf = CLASS_THRESHOLDS.get(label, default_conf)
        min_area = CLASS_MIN_AREA.get(label, 0)

        if score < min_conf:
            continue
        if width < 8 or height < 8:
            continue
        if area < min_area:
            continue

        ratio = width / max(1.0, height)
        if label == "Ball":
            if area > 22000:
                continue
            if ratio > 3.0 or ratio < 0.28:
                continue
        elif label == "Referee":
            if ratio > 1.8 or ratio < 0.35:
                continue
            if image_height and y2 < image_height * 0.25:
                continue
        else:  # Player
            if ratio > 1.9 or ratio < 0.35:
                continue
            if image_height and y2 < image_height * 0.30:
                continue

        if image_height and label in {"Player", "Referee"} and y2 < image_height * 0.20:
            continue

        filtered.append([x1, y1, x2, y2, score, cls_id])
    return class_nms(filtered, iou_thresh=0.30)


def draw_boxes_on_image_pil(image, filtered_boxes, names):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("Segoe UI.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    CLASS_COLORS = {"Player": (79, 124, 255), "Referee": (34, 197, 94), "Ball": (249, 115, 22)}
    for x1, y1, x2, y2, score, cls_id in filtered_boxes:
        label = names[cls_id]
        color = CLASS_COLORS.get(label, (255, 255, 255))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text = f"{label} {score:.2f}"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_bg = [x1, y1 - text_height - 6, x1 + text_width + 8, y1]
        draw.rectangle(text_bg, fill=color)
        draw.text((x1 + 4, y1 - text_height - 4), text, fill=(255, 255, 255), font=font)
    return image


# ----------------- Main App -----------------
class VolleyballDashboardApp:
    def __init__(self, root, model_path=None):
        # Set customtkinter appearance
        ctk.set_appearance_mode("light")  # support dark/light
        ctk.set_default_color_theme(self._create_ctk_theme())

        self.root = root
        # initialize fonts now that root exists
        self._setup_fonts()
        self.root.title("Volleyball YOLO Detection - Modern Dashboard")
        self.root.geometry("1200x800")

        # Model
        self.model_path = model_path or self._find_default_model()
        self.model = None
        if YOLO and self.model_path:
            try:
                self.model = YOLO(self.model_path)
            except Exception:
                self.model = None

        # State
        self.image_path = None
        self.original_pil = None
        self.result_pil = None
        self.last_detections = []

        # Build UI
        self._build_ui()

    def _find_default_model(self):
        candidates = [
            "runs/detect/runs/volleyball_yolov8-5/weights/best.pt",
            "runs/detect/runs/volleyball_yolov8/weights/best.pt",
            "yolov8s.pt",
            "yolo26n.pt",
        ]
        for c in candidates:
            if Path(c).exists():
                return str(Path(c))
        return None

    def _create_ctk_theme(self):
        # Use default theme but we can customize if needed
        return "blue"

    def _setup_fonts(self):
        """Choose available font family (Segoe UI -> Inter -> Helvetica) and
        create shared tkfont.Font objects used across the UI.
        """
        global DEFAULT_FONT, TITLE_FONT, LOGO_FONT, BUTTON_FONT, STAT_FONT, TABLE_FONT
        # Query available families
        families = list(tkfont.families())
        preferred = None
        for name in ("Segoe UI", "Inter", "Helvetica"):
            if name in families:
                preferred = name
                break
        if preferred is None:
            preferred = families[0] if families else "TkDefaultFont"

        # Create fonts per requirements
        # For CustomTkinter widgets use ctk.CTkFont (or tuple); for tkinter Text use tkfont.Font
        LOGO_FONT = ctk.CTkFont(family=preferred, size=24, weight="bold")
        TITLE_FONT = ctk.CTkFont(family=preferred, size=16, weight="bold")
        DEFAULT_FONT = ctk.CTkFont(family=preferred, size=12)
        BUTTON_FONT = ctk.CTkFont(family=preferred, size=13, weight="bold")
        # STAT_FONT will be used for numbers (bold large)
        STAT_FONT = ctk.CTkFont(family=preferred, size=28, weight="bold")
        STATUS_FONT = ctk.CTkFont(family=preferred, size=10)

        # Table: prefer Consolas monospace for tkinter.Text
        table_family = "Consolas" if "Consolas" in families else ("Courier New" if "Courier New" in families else preferred)
        TABLE_FONT = tkfont.Font(root=self.root, family=table_family, size=11)

        # assign to module globals (CTk fonts for CTk widgets, tk font for table)
        globals()["DEFAULT_FONT"] = DEFAULT_FONT
        globals()["TITLE_FONT"] = TITLE_FONT
        globals()["LOGO_FONT"] = LOGO_FONT
        globals()["BUTTON_FONT"] = BUTTON_FONT
        globals()["STAT_FONT"] = STAT_FONT
        globals()["TABLE_FONT"] = TABLE_FONT
        globals()["STATUS_FONT"] = STATUS_FONT

    def _build_ui(self):
        # Root grid config
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        # LEFT SIDEBAR (20%)
        sidebar = ctk.CTkFrame(self.root, width=300, corner_radius=RADIUS, fg_color=BG_PANEL)
        sidebar.grid(row=0, column=0, sticky="nsew", padx=16, pady=16)
        sidebar.grid_propagate(False)

        # Logo and Title
        logo_frame = ctk.CTkFrame(sidebar, height=100, fg_color=BG_PANEL, corner_radius=RADIUS)
        logo_frame.pack(pady=(12, 8), padx=12, fill="x")
        # simple circle volleyball icon
        self._create_logo(logo_frame)
        ctk.CTkLabel(logo_frame, text="VOLLEYBALL\nYOLO DETECTION", font=LOGO_FONT, text_color=TEXT).pack(side="right", padx=8)

        # Upload area
        upload_card = ctk.CTkFrame(sidebar, corner_radius=RADIUS, fg_color=CARD_BG, height=180)
        upload_card.pack(padx=12, pady=(8, 12), fill="x")
        upload_card.pack_propagate(False)

        self.upload_preview = ctk.CTkLabel(upload_card, text="Click to upload or drag & drop", width=200, height=120, fg_color=BG_PANEL, text_color=TEXT, font=DEFAULT_FONT)
        self.upload_preview.pack(padx=12, pady=12, fill="both", expand=True)
        self.upload_preview.bind("<Button-1>", lambda e: self.load_image())

        # Upload button large
        upload_btn = ctk.CTkButton(sidebar, text="Chọn ảnh", command=self.load_image, corner_radius=12, font=BUTTON_FONT)
        upload_btn.pack(padx=12, pady=(6, 8), fill="x")

        # Parameters card
        params_card = ctk.CTkFrame(sidebar, corner_radius=RADIUS, fg_color=CARD_BG)
        params_card.pack(padx=12, pady=(6, 6), fill="x")

        ctk.CTkLabel(params_card, text="Confidence", anchor="w", text_color=TEXT, font=DEFAULT_FONT).pack(fill="x", padx=10, pady=(8, 0))
        self.conf_slider = ctk.CTkSlider(params_card, from_=0.01, to=1.0, progress_color=PRIMARY)
        self.conf_slider.set(0.12)
        self.conf_slider.pack(fill="x", padx=10, pady=(2, 6))

        ctk.CTkLabel(params_card, text="IoU", anchor="w", text_color=TEXT, font=DEFAULT_FONT).pack(fill="x", padx=10)
        self.iou_slider = ctk.CTkSlider(params_card, from_=0.01, to=1.0, progress_color=PRIMARY)
        self.iou_slider.set(0.25)
        self.iou_slider.pack(fill="x", padx=10, pady=(2, 6))

        ctk.CTkLabel(params_card, text="Image Size", anchor="w", text_color=TEXT, font=DEFAULT_FONT).pack(fill="x", padx=10)
        self.imgsz_option = ctk.CTkOptionMenu(params_card, values=["640", "960", "1280"], command=None, font=DEFAULT_FONT)
        self.imgsz_option.set("960")
        self.imgsz_option.pack(fill="x", padx=10, pady=(6, 12))

        # Filters
        filter_card = ctk.CTkFrame(sidebar, corner_radius=RADIUS, fg_color=CARD_BG)
        filter_card.pack(padx=12, pady=(6, 6), fill="x")
        ctk.CTkLabel(filter_card, text="Filter Classes", anchor="w", text_color=TEXT, font=DEFAULT_FONT).pack(fill="x", padx=10, pady=(8, 0))
        self.var_player = tk.BooleanVar(value=True)
        self.var_referee = tk.BooleanVar(value=True)
        self.var_ball = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(filter_card, text="Player", variable=self.var_player, text_color=TEXT, font=DEFAULT_FONT, command=self._on_filter_changed).pack(anchor="w", padx=10, pady=6)
        ctk.CTkCheckBox(filter_card, text="Referee", variable=self.var_referee, text_color=TEXT, font=DEFAULT_FONT, command=self._on_filter_changed).pack(anchor="w", padx=10, pady=6)
        ctk.CTkCheckBox(filter_card, text="Ball", variable=self.var_ball, text_color=TEXT, font=DEFAULT_FONT, command=self._on_filter_changed).pack(anchor="w", padx=10, pady=6)

        # Detect button with gradient
        detect_img = self._create_gradient_button_image(320, 48, PRIMARY, ACCENT)
        self.detect_btn = ctk.CTkButton(sidebar, text="▶  CHẠY DETECT", command=self._on_detect_clicked, fg_color=(PRIMARY, ACCENT), text_color=BG_LIGHT, corner_radius=14, font=BUTTON_FONT)
        self.detect_btn.pack(padx=12, pady=12, fill="x")

        # Theme toggle
        theme_frame = ctk.CTkFrame(sidebar, fg_color=BG_PANEL, corner_radius=10)
        theme_frame.pack(padx=12, pady=(6, 12), fill="x")
        ctk.CTkLabel(theme_frame, text="Theme", text_color=TEXT, font=DEFAULT_FONT).pack(side="left", padx=(10, 8))
        self.theme_option = ctk.CTkSegmentedButton(theme_frame, values=["Light", "Dark"], command=self._on_theme_change, font=BUTTON_FONT)
        self.theme_option.set("Light")
        self.theme_option.pack(side="right", padx=10, pady=8)

        # RIGHT CONTENT (80%)
        content = ctk.CTkFrame(self.root, fg_color=BG_LIGHT)
        content.grid(row=0, column=1, sticky="nsew", padx=(0,16), pady=16)
        content.grid_rowconfigure(0, weight=1)
        content.grid_columnconfigure(0, weight=1)
        # Top content area: two image cards side by side
        top_content = ctk.CTkFrame(content, fg_color=BG_LIGHT)
        top_content.grid(row=0, column=0, sticky="nsew", padx=16, pady=12)
        top_content.grid_columnconfigure((0,1), weight=1)
        top_content.grid_rowconfigure(0, weight=1)

        # Original Image Card
        orig_card = ctk.CTkFrame(top_content, corner_radius=RADIUS, fg_color=CARD_BG)
        orig_card.grid(row=0, column=0, sticky="nsew", padx=(0,8), pady=4)
        ctk.CTkLabel(orig_card, text="Ảnh gốc", font=TITLE_FONT, text_color="#1E293B").pack(anchor="nw", padx=12, pady=(12,0))
        self.orig_canvas = tk.Canvas(orig_card, bg=BG_PANEL, highlightthickness=0)
        self.orig_canvas.pack(fill="both", expand=True, padx=12, pady=12)

        # Result Image Card
        res_card = ctk.CTkFrame(top_content, corner_radius=RADIUS, fg_color=CARD_BG)
        res_card.grid(row=0, column=1, sticky="nsew", padx=(8,0), pady=4)
        ctk.CTkLabel(res_card, text="Kết quả Detection", font=TITLE_FONT, text_color="#1E293B").pack(anchor="nw", padx=12, pady=(12,0))
        self.res_canvas = tk.Canvas(res_card, bg=BG_PANEL, highlightthickness=0)
        self.res_canvas.pack(fill="both", expand=True, padx=12, pady=12)
        # Save result button (disabled until a result exists)
        self.save_btn = ctk.CTkButton(res_card, text="Tải ảnh kết quả", command=self._save_result, corner_radius=12, font=BUTTON_FONT, state="disabled")
        self.save_btn.pack(padx=12, pady=(0,12), anchor="se")

        # Bottom area: stats and table
        bottom_area = ctk.CTkFrame(content, fg_color=BG_LIGHT)
        bottom_area.grid(row=1, column=0, sticky="nsew", padx=16, pady=(6,12))
        bottom_area.grid_columnconfigure((0,1), weight=1)

        # Stats cards
        stats_frame = ctk.CTkFrame(bottom_area, fg_color=BG_LIGHT)
        stats_frame.grid(row=0, column=0, sticky="nsew", padx=(0,8))
        stats_frame.pack_propagate(False)
        self._create_stats_cards(stats_frame)

        # Table
        table_frame = ctk.CTkFrame(bottom_area, fg_color=CARD_BG, corner_radius=RADIUS)
        table_frame.grid(row=0, column=1, sticky="nsew", padx=(8,0))
        ctk.CTkLabel(table_frame, text="Detections", font=TITLE_FONT, text_color=TEXT).pack(anchor="nw", padx=12, pady=(12,0))
        # Use a tkinter Text with scrollbar for table simplicity
        self.table_text = tk.Text(table_frame, height=8, bg=BG_PANEL, fg=TEXT, font=TABLE_FONT)
        self.table_text.pack(fill="both", expand=True, padx=12, pady=12)

        # Status bar
        self.status_label = ctk.CTkLabel(content, text="Chưa chọn ảnh.", anchor="w", text_color=TEXT, font=STATUS_FONT)
        self.status_label.grid(row=2, column=0, sticky="ew", padx=16, pady=(0,12))

    # ----------------- UI helpers -----------------
    def _create_logo(self, parent):
        # draw a simple volleyball-like circle
        canvas = tk.Canvas(parent, width=64, height=64, highlightthickness=0, bg=BG_PANEL)
        canvas.pack(side="left", padx=8, pady=8)
        canvas.create_oval(6,6,58,58, fill=PRIMARY, outline=ACCENT, width=3)
        # simple lines to simulate ball
        canvas.create_arc(10,20,54,60, start=200, extent=60, style="arc", width=3)

    def _create_gradient_button_image(self, w, h, c1, c2):
        # returns a PhotoImage but we use CTkButton fg_color tuple instead; keep for extension
        img = Image.new("RGB", (w, h), color=c1)
        draw = ImageDraw.Draw(img)
        for i in range(w):
            r = int(int(c1[1:3],16)*(1 - i/w) + int(c2[1:3],16)*(i/w))
        return None

    def _create_stats_cards(self, parent):
        # create 4 small stat frames
        card_frame = ctk.CTkFrame(parent, fg_color=BG_LIGHT)
        card_frame.pack(fill="both", expand=True, padx=4, pady=4)
        # Use labels for each stat
        self.stat_player = ctk.CTkFrame(card_frame, fg_color=CARD_BG, corner_radius=12)
        self.stat_player.pack(side="left", expand=True, fill="both", padx=6, pady=6)
        ctk.CTkLabel(self.stat_player, text="Player", text_color=TEXT, font=DEFAULT_FONT).pack(anchor="nw", padx=8, pady=(8,0))
        self.player_count = ctk.CTkLabel(self.stat_player, text="0", font=STAT_FONT, text_color=PRIMARY)
        self.player_count.pack(anchor="center", pady=12)

        self.stat_referee = ctk.CTkFrame(card_frame, fg_color=CARD_BG, corner_radius=12)
        self.stat_referee.pack(side="left", expand=True, fill="both", padx=6, pady=6)
        ctk.CTkLabel(self.stat_referee, text="Referee", text_color=TEXT, font=DEFAULT_FONT).pack(anchor="nw", padx=8, pady=(8,0))
        self.referee_count = ctk.CTkLabel(self.stat_referee, text="0", font=STAT_FONT, text_color="#10B981")
        self.referee_count.pack(anchor="center", pady=12)

        self.stat_ball = ctk.CTkFrame(card_frame, fg_color=CARD_BG, corner_radius=12)
        self.stat_ball.pack(side="left", expand=True, fill="both", padx=6, pady=6)
        ctk.CTkLabel(self.stat_ball, text="Ball", text_color=TEXT, font=DEFAULT_FONT).pack(anchor="nw", padx=8, pady=(8,0))
        self.ball_count = ctk.CTkLabel(self.stat_ball, text="0", font=STAT_FONT, text_color="#F59E0B")
        self.ball_count.pack(anchor="center", pady=12)

        self.stat_total = ctk.CTkFrame(card_frame, fg_color=CARD_BG, corner_radius=12)
        self.stat_total.pack(side="left", expand=True, fill="both", padx=6, pady=6)
        ctk.CTkLabel(self.stat_total, text="Total", text_color=TEXT, font=DEFAULT_FONT).pack(anchor="nw", padx=8, pady=(8,0))
        self.total_count = ctk.CTkLabel(self.stat_total, text="0", font=STAT_FONT, text_color=ACCENT)
        self.total_count.pack(anchor="center", pady=12)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*")])
        if not path:
            return
        self.image_path = path
        self.original_pil = Image.open(path).convert("RGB")
        self._display_image_on_canvas(self.original_pil, self.orig_canvas)
        self.status_label.configure(text=f"Ảnh đã chọn: {Path(path).name}")
        self.table_text.delete("1.0", "end")
        # no result yet
        if hasattr(self, 'save_btn'):
            self.save_btn.configure(state="disabled")

    def _display_image_on_canvas(self, pil_img, canvas):
        # Resize image to fit canvas keeping ratio
        canvas.update_idletasks()
        w = canvas.winfo_width() or 400
        h = canvas.winfo_height() or 300
        img = pil_img.copy()
        img.thumbnail((w - 10, h - 10), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(img)
        # Keep reference per-canvas to avoid garbage collection and avoid overwriting
        if canvas == getattr(self, 'orig_canvas', None):
            self.orig_tk_img = tk_img
        elif canvas == getattr(self, 'res_canvas', None):
            self.res_tk_img = tk_img
        else:
            # fallback
            self._tk_img = tk_img
        canvas.delete("all")
        canvas.create_image(w//2, h//2, image=tk_img)

    def _on_theme_change(self, value):
        if value == "Dark":
            ctk.set_appearance_mode("dark")
        else:
            ctk.set_appearance_mode("light")
    

    def _on_filter_changed(self):
        """Called when any filter checkbox toggles. Recompute counts and refresh UI."""
        # If no detections yet, nothing to do
        if not hasattr(self, 'last_detections') or not getattr(self, 'last_detections'):
            return
        names = getattr(self, 'last_names', None)
        if names is None:
            return
        # Build counts dict from raw detections (unfiltered)
        counts = {}
        for _, _, _, _, _, cls_id in self.last_detections:
            label = names[int(cls_id)]
            counts[label] = counts.get(label, 0) + 1
        # Reapply UI update using existing method
        self._on_detection_done(counts, names)

    def _on_detect_clicked(self):
        # Start detection in background thread
        if not self.image_path:
            self.status_label.configure(text="Vui lòng chọn ảnh trước khi chạy detect.")
            return
        if not self.model:
            self.status_label.configure(text="Model chưa được load hoặc ultralytics không khả dụng.")
            return
        # disable button and show loading
        self.detect_btn.configure(state="disabled")
        self.status_label.configure(text="Đang chạy detect...")
        thread = threading.Thread(target=self._run_detection_thread, daemon=True)
        thread.start()

    def _run_detection_thread(self):
        try:
            imgsz = int(self.imgsz_option.get())
            conf = float(self.conf_slider.get())
            iou = float(self.iou_slider.get())
        except Exception:
            imgsz = 960; conf = 0.12; iou = 0.25

        try:
            results = self.model.predict(source=self.image_path, imgsz=imgsz, conf=min(conf,0.12), iou=iou, max_det=200, augment=False, save=False)
            boxes = results[0].boxes
            names = results[0].names
            if boxes is not None and len(boxes) > 0:
                filtered = filter_detections(boxes, names, default_conf=conf, image_height=results[0].orig_shape[0], image_width=results[0].orig_shape[1])
            else:
                filtered = []

            self.last_detections = filtered
            # store names for later UI filtering
            self.last_names = names
            # draw result image
            pil = self.original_pil.copy()
            pil = draw_boxes_on_image_pil(pil, filtered, names)
            self.result_pil = pil

            # prepare stats
            counts = {}
            for _, _, _, _, _, cls_id in filtered:
                label = names[int(cls_id)]
                counts[label] = counts.get(label, 0) + 1

            # update UI in main thread
            self.root.after(0, lambda: self._on_detection_done(counts, names))
        except Exception as e:
            self.root.after(0, lambda: self.status_label.configure(text=f"Lỗi khi detect: {str(e)[:80]}"))
            self.root.after(0, lambda: self.detect_btn.configure(state="normal"))

    def _on_detection_done(self, counts, names):
        # Re-apply UI filters to the stored detections
        names = getattr(self, 'last_names', names)
        allowed = set()
        if self.var_player.get():
            allowed.add("Player")
        if self.var_referee.get():
            allowed.add("Referee")
        if self.var_ball.get():
            allowed.add("Ball")

        if allowed:
            filtered_ui = [d for d in self.last_detections if names[int(d[5])] in allowed]
        else:
            filtered_ui = []

        # draw filtered result image
        if self.original_pil:
            pil = self.original_pil.copy()
            pil = draw_boxes_on_image_pil(pil, filtered_ui, names)
            self.result_pil = pil
            self._display_image_on_canvas(self.result_pil, self.res_canvas)
            # enable save button when a result image exists
            if hasattr(self, 'save_btn'):
                self.save_btn.configure(state="normal")

        # update stats based on filtered_ui
        counts_ui = {}
        for _, _, _, _, _, cls_id in filtered_ui:
            label = names[int(cls_id)]
            counts_ui[label] = counts_ui.get(label, 0) + 1

        self.player_count.configure(text=str(counts_ui.get("Player", 0)))
        self.referee_count.configure(text=str(counts_ui.get("Referee", 0)))
        self.ball_count.configure(text=str(counts_ui.get("Ball", 0)))
        self.total_count.configure(text=str(sum(counts_ui.values())))

        # update table showing only filtered_ui
        self.table_text.delete("1.0", "end")
        if filtered_ui:
            for idx, (x1, y1, x2, y2, score, cls_id) in enumerate(filtered_ui, 1):
                label = names[int(cls_id)]
                area = int((x2 - x1) * (y2 - y1))
                self.table_text.insert("end", f"{idx}\t{label}\t{score:.3f}\t({int(x1)},{int(y1)},{int(x2)},{int(y2)})\t{area}\n")
        else:
            self.table_text.insert("end", "No detections\n")

        self.status_label.configure(text="Hoàn tất.")
        self.detect_btn.configure(state="normal")

    def _save_result(self):
        """Save the currently displayed result image (respecting selected classes)."""
        if not hasattr(self, 'result_pil') or self.result_pil is None:
            self.status_label.configure(text="Không có ảnh kết quả để lưu.")
            return
        # Suggest a default filename based on source
        default_name = Path(self.image_path).stem + "_result.jpg" if self.image_path else "result.jpg"
        file_path = filedialog.asksaveasfilename(defaultextension=".jpg", initialfile=default_name, filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("All", "*")])
        if not file_path:
            return
        try:
            # Ensure we don't overwrite existing files: add (1), (2), ... if needed
            dest = Path(file_path)
            parent = dest.parent
            stem = dest.stem
            suffix = dest.suffix
            candidate = dest
            i = 1
            while candidate.exists():
                candidate = parent / f"{stem} ({i}){suffix}"
                i += 1

            # Save a copy to avoid modifying in-memory image
            save_img = self.result_pil.copy()
            if candidate.suffix.lower() in (".jpg", ".jpeg"):
                save_img = save_img.convert("RGB")
            save_img.save(candidate)
            self.status_label.configure(text=f"Đã lưu: {candidate.name}")
        except Exception as e:
            self.status_label.configure(text=f"Lỗi khi lưu: {str(e)[:80]}")


def main():
    root = ctk.CTk()
    app = VolleyballDashboardApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
