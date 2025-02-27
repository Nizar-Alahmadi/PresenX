#!/usr/bin/env python

# ====================================================
# Imports
# ====================================================
import os
import sys
import time
import pickle
import sqlite3
import json
import signal
import threading
import csv
import webbrowser
import cv2
import numpy as np
import face_recognition
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from datetime import datetime
from PIL import Image, ImageTk, Image as PilImage
import matplotlib.pyplot as plt
import pandas as pd
import pystray
from openvino.runtime import Core

if sys.platform.startswith('win'):
    import ctypes

# ====================================================
# Configuration Section
# ====================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SCRIPT_DIR, "config.json")
DEFAULT_CONFIG = {
    "detection_confidence": 0.1,
    "inactive_threshold": 0.0,
    "cosine_similarity_cutoff": 0.9,
    "camera_index": 0
}

def load_config():
    """Load configuration from config.json or create with defaults if not present."""
    if not os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
        return DEFAULT_CONFIG
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)

def save_config(config):
    """Save configuration to config.json."""
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)

config = load_config()
INACTIVE_THRESHOLD = config.get("inactive_threshold")

if sys.platform.startswith('win'):
    whnd = ctypes.windll.kernel32.GetConsoleWindow()
    if whnd != 0:
        ctypes.windll.user32.ShowWindow(whnd, 0)
        ctypes.windll.kernel32.FreeConsole()

# ====================================================
# Database Section
# ====================================================
DB_FILE = os.path.join(SCRIPT_DIR, "Log.db")

def init_db():
    """Initialize SQLite database with logs, attendance, and users tables."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            date_str TEXT NOT NULL,
            status TEXT NOT NULL,
            start_time_str TEXT NOT NULL,
            end_time_str TEXT NOT NULL,
            duration_str TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            datetime_str TEXT NOT NULL,
            status TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            embedding BLOB NOT NULL
        );
    """)
    conn.commit()
    conn.close()

def append_db_row(user_id, date_str, status, start_time_str, end_time_str, duration_str):
    """Append a row to the logs table."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO logs (user_id, date_str, status, start_time_str, end_time_str, duration_str)
        VALUES (?, ?, ?, ?, ?, ?);
    """, (user_id, date_str, status, start_time_str, end_time_str, duration_str))
    conn.commit()
    conn.close()

def append_attendance_row(user_id, datetime_str, status):
    """Append a row to the Attendance table."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO Attendance (user_id, datetime_str, status)
        VALUES (?, ?, ?);
    """, (user_id, datetime_str, status))
    conn.commit()
    conn.close()

def format_time_diff(start, end):
    """Format time difference in HH:MM:SS."""
    diff_sec = int(end - start)
    hours = diff_sec // 3600
    minutes = (diff_sec % 3600) // 60
    seconds = diff_sec % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

# ====================================================
# Face Processing Section
# ====================================================
def get_face_embedding(face_bgr, fr_compiled, fr_input, fr_output, fr_h, fr_w):
    """Compute normalized face embedding using OpenVINO model."""
    face_resized = cv2.resize(face_bgr, (fr_w, fr_h)).transpose(2, 0, 1)
    face_resized = np.expand_dims(face_resized, axis=0)
    emb = fr_compiled.infer_new_request({fr_input.any_name: face_resized})[fr_output]
    emb = np.squeeze(emb)
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 0 else emb

def cosine_distance(e1, e2):
    """Calculate cosine distance between two embeddings."""
    if e1 is None or e2 is None:
        return 9999
    return 1.0 - np.dot(e1, e2)

def detect_cameras(max_test=10):
    """Detect available camera indices."""
    available_cameras = []
    for i in range(max_test):
        cap_test = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap_test.isOpened():
            available_cameras.append(i)
        cap_test.release()
    return available_cameras

# ====================================================
# Model Loading Section
# ====================================================
def load_models():
    """Load OpenVINO face detection and re-identification models."""
    ie = Core()
    fd_model_xml = os.path.join(SCRIPT_DIR, "models", "face-detection-adas-0001.xml")
    fd_model_bin = os.path.join(SCRIPT_DIR, "models", "face-detection-adas-0001.bin")
    fd_model = ie.read_model(model=fd_model_xml, weights=fd_model_bin)
    fd_compiled = ie.compile_model(fd_model, device_name="GPU")
    fd_input = fd_compiled.input(0)
    fd_output = fd_compiled.output(0)
    fd_h, fd_w = fd_input.shape[2], fd_input.shape[3]

    fr_model_xml = os.path.join(SCRIPT_DIR, "models", "face-reidentification-retail-0095.xml")
    fr_model_bin = os.path.join(SCRIPT_DIR, "models", "face-reidentification-retail-0095.bin")
    fr_model = ie.read_model(fr_model_xml, weights=fr_model_bin)
    fr_compiled = ie.compile_model(fr_model, device_name="GPU")
    fr_input = fr_compiled.input(0)
    fr_output = fr_compiled.output(0)
    fr_h, fr_w = fr_input.shape[2], fr_input.shape[3]

    return {
        "fd_compiled": fd_compiled, "fd_input": fd_input, "fd_output": fd_output, "fd_h": fd_h, "fd_w": fd_w,
        "fr_compiled": fr_compiled, "fr_input": fr_input, "fr_output": fr_output, "fr_h": fr_h, "fr_w": fr_w
    }

models = load_models()
fd_compiled = models["fd_compiled"]
fd_input = models["fd_input"]
fd_output = models["fd_output"]
fd_h, fd_w = models["fd_h"], models["fd_w"]
fr_compiled = models["fr_compiled"]
fr_input = models["fr_input"]
fr_output = models["fr_output"]
fr_h, fr_w = models["fr_h"], models["fr_w"]

# ====================================================
# Main Application Class
# ====================================================
class PresenX:
    def __init__(self, master):
        """Initialize the PresenX application."""
        self.master = master
        self.master.title("PresenX")
        self.master.geometry("1000x800")
        self.master.resizable(True, True)
        self.init_database()
        self.handle_user_login()
        self.setup_gui()
        self.initialize_variables()

    def init_database(self):
        """Set up the SQLite database."""
        init_db()

    def handle_user_login(self):
        """Prompt for username and handle login or signup."""
        username = simpledialog.askstring("Login", "Please enter your username:")
        if not username:
            messagebox.showerror("Error", "No username provided. Exiting.")
            sys.exit(0)
        user_data = self.find_user_by_name(username)
        if user_data is None:
            if messagebox.askyesno("Sign Up", f"User '{username}' not found. Sign up?"):
                new_id = self.create_new_user(username)
                self.current_user_id = new_id
                self.current_username = username
            else:
                messagebox.showinfo("Goodbye", "Exiting application.")
                sys.exit(0)
        else:
            self.current_user_id = user_data["id"]
            self.current_username = user_data["username"]
        self.user_embedding = self.load_user_embedding_from_db(self.current_user_id)

    def setup_gui(self):
        """Configure and set up GUI components."""
        self.style = ttk.Style(theme='lumen')
        self.style.configure("TLabel", font=("Segoe UI", 10))
        self.style.configure("Header.TLabel", font=("Segoe UI", 20, "bold"), foreground="#2b2b2b")
        self.style.configure("Status.TLabel", font=("Segoe UI", 12), foreground="#0066cc")
        self.style.configure("TButton", font=("Segoe UI", 10), padding=6)
        self.style.configure("TCombobox", font=("Segoe UI", 10))

        # --------------------------------------------------------------------
        # ADDITIONAL STYLING FOR THE LEFT PANEL (HIGHLIGHTED PART)
        # --------------------------------------------------------------------
        # Frame background
        self.style.configure("Controls.TFrame", background="#F7F7F7")

        # Label style (for “Select Camera:” etc.)
        self.style.configure("ControlsLabel.TLabel", font=("Segoe UI", 11, "bold"),
                             foreground="#333333", background="#F7F7F7")

        # Combobox style
        self.style.configure("ControlsCombobox.TCombobox", font=("Segoe UI", 10),
                             padding=5)

        # Button style (left panel)
        # Note: We keep bootstyle (SUCCESS, DANGER, etc.) for color but set fonts here.
        self.style.configure("Controls.TButton", font=("Segoe UI", 10, "bold"), padding=8)
        # --------------------------------------------------------------------

        # Header Frame
        header_frame = ttk.Frame(self.master, padding=(20, 10))
        header_frame.pack(side=TOP, fill=X)
        logo_path = os.path.join(SCRIPT_DIR, "icons", "PresenX_Logo.png")
        try:
            header_logo = ImageTk.PhotoImage(PilImage.open(logo_path).resize((200, 182)))
            logo_label = ttk.Label(header_frame, image=header_logo)
            logo_label.image = header_logo
            logo_label.pack(side=LEFT, padx=10)
        except FileNotFoundError as e:
            print(f"Error loading header logo: {e}")

        self.status_label = ttk.Label(header_frame, text="Status: Ready", style="Status.TLabel")
        self.status_label.pack(side=TOP)
        self.datetime_label = ttk.Label(header_frame, style="TLabel")
        self.datetime_label.pack(side=TOP, pady=(5, 0))
        self.update_datetime()
        self.attendance_status_label = ttk.Label(header_frame, text="Last Check-in: N/A", style="TLabel")
        self.attendance_status_label.pack(side=TOP, pady=(5, 0))

        # Main Frame
        main_frame = ttk.Frame(self.master, padding=20)
        main_frame.pack(fill=BOTH, expand=TRUE)
        # Apply custom style to controls_frame
        controls_frame = ttk.Frame(main_frame, style="Controls.TFrame", padding=10, relief="groove")
        controls_frame.grid(row=0, column=0, sticky=N+S+W, padx=(0, 20))
        video_frame = ttk.Frame(main_frame)
        video_frame.grid(row=0, column=1, sticky=N+S+E, padx=(0, 20))
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # Camera Selection
        ttk.Label(controls_frame, text="Select Camera:", style="ControlsLabel.TLabel").grid(
            row=0, column=0, sticky=W, pady=5
        )
        self.camera_list = detect_cameras()
        if not self.camera_list:
            messagebox.showerror("Error", "No cameras found.", parent=self.master)
            self.master.destroy()
            return
        default_camera = self.camera_list[0]
        try:
            from screeninfo import get_monitors
            if len(get_monitors()) == 1 and len(self.camera_list) >= 2:
                default_camera = self.camera_list[1]
        except ImportError:
            pass
        self.current_camera_index = default_camera
        self.camera_var = tk.StringVar()
        camera_names = [f"Camera {i}" for i in self.camera_list]
        self.camera_combobox = ttk.Combobox(
            controls_frame, textvariable=self.camera_var,
            values=camera_names, state="readonly",
            style="ControlsCombobox.TCombobox"
        )
        self.camera_combobox.grid(row=1, column=0, sticky=W, pady=5)
        self.camera_combobox.current(self.camera_list.index(default_camera))
        self.camera_combobox.bind("<<ComboboxSelected>>", self.on_camera_change)

        # Control Buttons
        self.control_button = ttk.Button(
            controls_frame, text="Start", bootstyle=SUCCESS,
            command=self.control_button_pressed, style="Controls.TButton"
        )
        self.control_button.grid(row=2, column=0, sticky=EW, pady=5)

        self.stop_button = ttk.Button(
            controls_frame, text="Stop", bootstyle=DANGER,
            command=self.stop_recognition, style="Controls.TButton"
        )
        self.stop_button.grid(row=3, column=0, sticky=EW, pady=5)
        self.stop_button.grid_remove()

        self.record_button = ttk.Button(
            controls_frame, text="Record Embedding", bootstyle=INFO,
            command=self.record_embedding, style="Controls.TButton"
        )
        self.record_button.grid(row=4, column=0, sticky=EW, pady=5)

        self.view_button = ttk.Button(
            controls_frame, text="View Records", bootstyle=SECONDARY,
            command=self.view_records, style="Controls.TButton"
        )
        self.view_button.grid(row=5, column=0, sticky=EW, pady=5)

        self.pause_button = ttk.Button(
            controls_frame, text="Pause", bootstyle=WARNING,
            command=self.on_pause, style="Controls.TButton"
        )
        self.pause_button.grid(row=6, column=0, sticky=EW, pady=5)
        self.pause_button.grid_remove()

        self.pause_reason_label = ttk.Label(
            controls_frame, text="Select pause reason:", style="ControlsLabel.TLabel"
        )
        self.pause_reason_label.grid(row=7, column=0, sticky=EW, pady=5)
        self.pause_reason_label.grid_remove()

        self.pause_reason_var = tk.StringVar()
        self.pause_reason_combobox = ttk.Combobox(
            controls_frame, textvariable=self.pause_reason_var,
            values=["break", "meeting", "phone call", "appointment",
                    "break fast", "lunch", "praying", "training",
                    "emergency", "bathroom", "other"],
            state="readonly", style="ControlsCombobox.TCombobox"
        )
        self.pause_reason_combobox.grid(row=8, column=0, sticky=EW, pady=5)
        self.pause_reason_combobox.grid_remove()

        self.resume_button = ttk.Button(
            controls_frame, text="Resume", bootstyle=SUCCESS,
            command=self.on_resume, style="Controls.TButton"
        )
        self.resume_button.grid(row=9, column=0, sticky=EW, pady=5)
        self.resume_button.grid_remove()

        self.integrated_video_label = ttk.Label(video_frame)
        self.integrated_video_label.pack(fill=BOTH, expand=TRUE)
        self.integrated_video_label.pack_forget()

        # Status Bar
        status_bar_frame = ttk.Frame(self.master, padding=(10, 5), relief=FLAT)
        status_bar_frame.pack(side=BOTTOM, fill=X)
        self.status_bar_label = ttk.Label(status_bar_frame, text="Camera: Disconnected | User: Unknown")
        self.status_bar_label.pack(side=LEFT)

        self.cap = cv2.VideoCapture(self.current_camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Cannot access camera {self.current_camera_index}.", parent=self.master)
            self.master.destroy()
            return
        self.update_status_bar()
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def initialize_variables(self):
        """Initialize application state variables."""
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        append_attendance_row(self.current_user_id, current_datetime, "Check-in")
        self.last_check_in = current_datetime
        self.last_check_out = None
        self.update_attendance_status()
        self.previous_status = None
        self.status_start_time = None
        self.last_seen_time = None
        self.started = False
        self.session_active = False

    # Database Helpers
    def find_user_by_name(self, username):
        """Find user by username in the database."""
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, embedding FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        conn.close()
        return {"id": row[0], "username": row[1], "embedding": row[2]} if row else None

    def create_new_user(self, username):
        """Create a new user and capture their face embedding."""
        user_emb = self.capture_embedding_interactively(username)
        emb_blob = pickle.dumps(user_emb)
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, embedding) VALUES (?, ?)", (username, emb_blob))
        new_id = cursor.lastrowid
        conn.commit()
        conn.close()
        messagebox.showinfo("Success", f"User '{username}' has been created.")
        return new_id

    def capture_embedding_interactively(self, username):
        """Capture face embedding interactively for a new user."""
        messagebox.showinfo("Sign Up", f"Hello {username}, recording a 1-minute video.\nPress 'q' to stop early.")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            messagebox.showerror("Error", "Cannot access camera for sign-up.")
            sys.exit(0)
        start_time = time.time()
        embeddings = []
        max_duration = 60
        cv2.namedWindow("Sign Up Embedding - Press 'q' to stop", cv2.WINDOW_NORMAL)
        while time.time() - start_time < max_duration:
            ret, frame = cap.read()
            if not ret:
                break
            det_frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (fd_w, fd_h)).transpose(2, 0, 1)
            det_frame = np.expand_dims(det_frame, axis=0)
            results = fd_compiled.infer_new_request({fd_input.any_name: det_frame})[fd_output]
            height, width = frame.shape[:2]
            for detection in results[0][0]:
                if float(detection[2]) > load_config().get("detection_confidence"):
                    xmin = int(detection[3] * width)
                    ymin = int(detection[4] * height)
                    xmax = int(detection[5] * width)
                    ymax = int(detection[6] * height)
                    face_bgr = frame[ymin:ymax, xmin:xmax]
                    if face_bgr.size > 0:
                        emb = get_face_embedding(face_bgr, fr_compiled, fr_input, fr_output, fr_h, fr_w)
                        embeddings.append(emb)
                        break
            elapsed = int(time.time() - start_time)
            cv2.putText(frame, f"Recording... {elapsed} sec", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Move slowly in a 360° circle", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow("Sign Up Embedding - Press 'q' to stop", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        if not embeddings:
            messagebox.showerror("Error", "No face detected in sign-up video.")
            sys.exit(0)
        user_emb = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(user_emb)
        return user_emb / norm if norm > 0 else user_emb

    def load_user_embedding_from_db(self, user_id):
        """Load user embedding from the database."""
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT embedding FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        conn.close()
        return pickle.loads(row[0]) if row and row[0] else None

    # GUI Update Methods
    def update_status_bar(self):
        """Update the status bar with camera and user information."""
        cam_status = "Connected" if self.cap.isOpened() else "Disconnected"
        user_name = self.current_username or "UnknownUser"
        emb_status = "(Embedding Loaded)" if self.user_embedding is not None else "(No Embedding)"
        self.status_bar_label.config(text=f"Camera: {cam_status} | User: {user_name} {emb_status}")

    def update_datetime(self):
        """Update the datetime label every second."""
        self.datetime_label.config(text=datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
        self.master.after(1000, self.update_datetime)

    def update_attendance_status(self):
        """Update the attendance status label."""
        text = f"Last Check-in: {self.last_check_in or 'N/A'}"
        self.attendance_status_label.config(text=text)

    def on_camera_change(self, event):
        """Handle camera selection change."""
        try:
            new_index = int(self.camera_var.get().split()[1])
            if new_index != self.current_camera_index:
                if self.cap.isOpened():
                    self.cap.release()
                self.cap = cv2.VideoCapture(new_index, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    messagebox.showerror("Error", f"Cannot access camera {new_index}.", parent=self.master)
                    return
                self.current_camera_index = new_index
                self.update_status_bar()
        except (ValueError, IndexError) as e:
            print(f"Error changing camera: {e}")

    # Control Methods
    def is_face_detected(self, frame):
        """Check if a face is detected in the frame."""
        det_frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (fd_w, fd_h)).transpose(2, 0, 1)
        det_frame = np.expand_dims(det_frame, axis=0)
        results = fd_compiled.infer_new_request({fd_input.any_name: det_frame})[fd_output]
        return any(float(detection[2]) > load_config().get("detection_confidence") for detection in results[0][0])

    def control_button_pressed(self):
        """Start PresenX and update GUI."""
        if not self.started:
            self.started = self.session_active = True
            ret, frame = self.cap.read()
            found_face = self.is_face_detected(frame) if ret else False
            self.control_button.grid_remove()
            self.pause_button.grid()
            self.stop_button.config(state="normal")
            self.stop_button.grid()
            self.record_button.grid_remove()
            self.status_start_time = self.last_seen_time = time.time()
            self.status_label.config(text=f"Status: {'active' if found_face else 'waiting...'}",
                                     foreground="green" if found_face else "black")
            self.integrated_video_label.pack(fill=BOTH, expand=TRUE)
            self.update_video()

    def on_pause(self):
        """Show pause reason selection."""
        self.pause_button.grid_remove()
        self.pause_reason_label.grid()
        self.pause_reason_combobox.grid()

    def on_reason_selected(self, event):
        """Handle pause reason selection."""
        selected_reason = self.pause_reason_var.get()
        if selected_reason:
            current_time = time.time()
            if self.previous_status:
                self.log_status_change(self.previous_status, self.status_start_time, current_time, "")
            self.previous_status = "paused"
            self.status_start_time = current_time
            self.pause_reason_label.grid_remove()
            self.pause_reason_combobox.grid_remove()
            self.resume_button.grid()
            self.status_label.config(text=f"Status: {selected_reason}", foreground="orange")

    def on_resume(self):
        """Resume recognition after pause."""
        self.resume_button.grid_remove()
        self.pause_button.grid()
        current_time = time.time()
        self.log_status_change("paused", self.status_start_time, current_time, self.pause_reason_var.get())
        self.previous_status = "active"
        self.status_start_time = current_time
        self.status_label.config(text="Status: active", foreground="green")

    def stop_recognition(self):
        """Stop PresenX and reset GUI."""
        current_time = time.time()
        if self.session_active and self.previous_status:
            if self.previous_status == "paused":
                self.log_status_change("paused", self.status_start_time, current_time, self.pause_reason_var.get())
            else:
                self.log_status_change(self.previous_status, self.status_start_time, current_time, "")
        self.started = self.session_active = False
        self.control_button.config(text="Start")
        self.control_button.grid()
        self.stop_button.config(state="disabled")
        self.stop_button.grid_remove()
        self.record_button.grid()
        self.previous_status = self.status_start_time = self.last_seen_time = None
        self.status_label.config(text="Status: Ready", foreground="black")
        self.integrated_video_label.pack_forget()
        self.pause_button.grid_remove()
        self.pause_reason_label.grid_remove()
        self.pause_reason_combobox.grid_remove()
        self.resume_button.grid_remove()

    def update_video(self):
        """Update video feed and recognition status."""
        if not self.started:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.status_label.config(text="Status: Camera disconnected", foreground="red")
            self.started = False
            return
        height, width = frame.shape[:2]
        det_frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (fd_w, fd_h)).transpose(2, 0, 1)
        det_frame = np.expand_dims(det_frame, axis=0)
        results = fd_compiled.infer_new_request({fd_input.any_name: det_frame})[fd_output]
        user_detected = False
        for detection in results[0][0]:
            conf = float(detection[2])
            if conf > load_config().get("detection_confidence"):
                xmin = int(detection[3] * width)
                ymin = int(detection[4] * height)
                xmax = int(detection[5] * width)
                ymax = int(detection[6] * height)
                face_bgr = frame[ymin:ymax, xmin:xmax]
                if face_bgr.size == 0:
                    continue
                try:
                    emb = get_face_embedding(face_bgr, fr_compiled, fr_input, fr_output, fr_h, fr_w)
                    if cosine_distance(emb, self.user_embedding) < load_config().get("cosine_similarity_cutoff"):
                        color = (0, 255, 0)
                        label = self.current_username
                        user_detected = True
                    else:
                        color = (0, 0, 255)
                        label = "Unknown"
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                    cv2.putText(frame, label, (xmin, max(ymin-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                except cv2.error as e:
                    print(f"Error processing face: {e}")
        try:
            im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=im)
            self.integrated_video_label.imgtk = imgtk
            self.integrated_video_label.configure(image=imgtk)
        except (AttributeError, ValueError) as e:
            print(f"Error updating video feed: {e}")
        self.update_status(user_detected)
        if self.started:
            self.master.after(30, self.update_video)

    def update_status(self, user_detected):
        """Update application status based on face detection."""
        if self.user_embedding is None:
            self.status_label.config(text="Status: No embedding found.", foreground="red")
            return
        current_time = time.time()
        if self.previous_status != "paused":
            if user_detected:
                self.last_seen_time = current_time
                if self.previous_status != "active":
                    self.log_status_change(self.previous_status, self.status_start_time, current_time, "")
                    self.previous_status = "active"
                    self.status_start_time = current_time
                    self.status_label.config(text="Status: active", foreground="green")
            elif self.last_seen_time and (current_time - self.last_seen_time) > INACTIVE_THRESHOLD:
                if self.previous_status != "inactive":
                    self.log_status_change(self.previous_status, self.status_start_time, current_time, "")
                    self.previous_status = "inactive"
                    self.status_start_time = current_time
                    self.status_label.config(text="Status: inactive", foreground="red")
        else:
            self.status_label.config(text=f"Status: {self.pause_reason_var.get()}", foreground="orange")

    def log_status_change(self, old_status, start_t, end_t, pause_reason):
        """Log status change to the database."""
        if not old_status:
            return
        date_str = datetime.now().strftime("%m/%d/%Y")
        final_status = pause_reason if old_status == "paused" and pause_reason else old_status if old_status != "paused" else "inactive"
        start_time_str = time.strftime('%H:%M:%S', time.localtime(start_t)) if start_t else "00:00:00"
        end_time_str = time.strftime('%H:%M:%S', time.localtime(end_t))
        duration_str = format_time_diff(start_t or end_t, end_t)
        append_db_row(self.current_user_id, date_str, final_status, start_time_str, end_time_str, duration_str)

    def record_embedding(self):
        """Record a new face embedding for the current user."""
        messagebox.showinfo("Record Embedding", "Move your face in a 360° circle.\nPress 'q' to stop early.")
        cap = cv2.VideoCapture(self.current_camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            messagebox.showerror("Error", f"Cannot access camera {self.current_camera_index}.", parent=self.master)
            return
        start_time = time.time()
        embeddings = []
        max_duration = 60
        cv2.namedWindow("Recording Embedding - Press 'q' to stop", cv2.WINDOW_NORMAL)
        while time.time() - start_time < max_duration:
            ret, frame = cap.read()
            if not ret:
                break
            det_frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (fd_w, fd_h)).transpose(2, 0, 1)
            det_frame = np.expand_dims(det_frame, axis=0)
            results = fd_compiled.infer_new_request({fd_input.any_name: det_frame})[fd_output]
            height, width = frame.shape[:2]
            for detection in results[0][0]:
                if float(detection[2]) > load_config().get("detection_confidence"):
                    xmin = int(detection[3] * width)
                    ymin = int(detection[4] * height)
                    xmax = int(detection[5] * width)
                    ymax = int(detection[6] * height)
                    face_bgr = frame[ymin:ymax, xmin:xmax]
                    if face_bgr.size > 0:
                        emb = get_face_embedding(face_bgr, fr_compiled, fr_input, fr_output, fr_h, fr_w)
                        embeddings.append(emb)
                        break
            elapsed = int(time.time() - start_time)
            cv2.putText(frame, f"Recording... {elapsed} sec", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Move slowly in a 360° circle", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow("Recording Embedding - Press 'q' to stop", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        if not embeddings:
            messagebox.showerror("Error", "No face detected in video.", parent=self.master)
            return
        user_emb = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(user_emb)
        user_emb = user_emb / norm if norm > 0 else user_emb
        emb_blob = pickle.dumps(user_emb)
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET embedding = ? WHERE id = ?", (emb_blob, self.current_user_id))
        conn.commit()
        conn.close()
        self.user_embedding = user_emb
        messagebox.showinfo("Success", "Embedding updated!", parent=self.master)
        self.update_status_bar()
        restart_app()

    def view_records(self):
        """Display user records in a new window."""
        extended_view_records(self)

    def on_closing(self):
        """Handle application shutdown."""
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        append_attendance_row(self.current_user_id, current_datetime, "Check-out")
        self.last_check_out = current_datetime
        self.update_attendance_status()
        if self.session_active and self.previous_status:
            self.log_status_change(self.previous_status, self.status_start_time, time.time(), "")
        if self.cap.isOpened():
            self.cap.release()
        self.master.destroy()

class PresenXIntegrated(PresenX):
    """Integrated subclass for additional functionality if needed."""
    pass

# ====================================================
# Extra Features Section
# ====================================================
def extended_view_records(app_instance):
    """Display logs and attendance records for the current user."""
    view_win = tk.Toplevel(app_instance.master)
    try:
        view_win.iconbitmap(os.path.join(SCRIPT_DIR, "icons", "table_icon.ico"))
    except Exception:
        pass
    view_win.title("Records")
    view_win.geometry("900x600")
    view_win.resizable(True, True)
    notebook = ttk.Notebook(view_win)
    notebook.pack(fill=BOTH, expand=True, padx=5, pady=5)

    # Logs Tab
    logs_frame = ttk.Frame(notebook)
    notebook.add(logs_frame, text="Logs")
    logs_current_sort = [None, False]
    logs_columns = ("ID", "Date", "Status", "Start", "End", "Duration")
    logs_filter_frame = ttk.Frame(logs_frame, padding=10)
    logs_filter_frame.pack(fill=X)
    ttk.Label(logs_filter_frame, text="Status:", font="-size 10 -weight bold").grid(row=0, column=0, padx=5, pady=5, sticky="w")
    logs_text_filter_var = tk.StringVar()
    ttk.Entry(logs_filter_frame, textvariable=logs_text_filter_var).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
    ttk.Label(logs_filter_frame, text="From Date:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
    logs_from_date_var = tk.StringVar()
    ttk.Entry(logs_filter_frame, textvariable=logs_from_date_var).grid(row=1, column=1, padx=5, pady=5, sticky="ew")
    ttk.Label(logs_filter_frame, text="To Date:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
    logs_to_date_var = tk.StringVar()
    ttk.Entry(logs_filter_frame, textvariable=logs_to_date_var).grid(row=2, column=1, padx=5, pady=5, sticky="ew")
    ttk.Button(logs_filter_frame, text="Apply Filter", bootstyle=INFO, command=lambda: logs_load_data()).grid(row=3, column=1, padx=5, pady=5, sticky="e")
    logs_filter_frame.columnconfigure(1, weight=1)

    logs_tree = ttk.Treeview(logs_frame, columns=logs_columns, show="headings")
    logs_vsb = ttk.Scrollbar(logs_frame, orient="vertical", command=logs_tree.yview)
    logs_tree.configure(yscrollcommand=logs_vsb.set)
    logs_vsb.pack(side=RIGHT, fill=Y)
    logs_tree.pack(fill=BOTH, expand=True)

    def logs_sort_column(tv, col, reverse):
        l = [(tv.set(k, col), k) for k in tv.get_children('')]
        try:
            l.sort(key=lambda t: int(t[0]), reverse=reverse)
        except ValueError:
            l.sort(reverse=reverse)
        for index, (val, k) in enumerate(l):
            tv.move(k, '', index)
        for c in logs_columns:
            tv.heading(c, text=c, command=lambda _col=c: logs_sort_column(tv, _col, False))
        tv.heading(col, text=col + (" ▲" if not reverse else " ▼"), command=lambda: logs_sort_column(tv, col, not reverse))
        logs_current_sort[0] = col
        logs_current_sort[1] = reverse

    for col in logs_columns:
        logs_tree.heading(col, text=col, command=lambda _col=col: logs_sort_column(logs_tree, _col, False))
        logs_tree.column(col, anchor="center", width=80 if col != "Status" else 100)

    def logs_load_data():
        text_filter = logs_text_filter_var.get().lower().strip()
        from_date = datetime.strptime(logs_from_date_var.get().strip(), "%m/%d/%Y") if logs_from_date_var.get().strip() else None
        to_date = datetime.strptime(logs_to_date_var.get().strip(), "%m/%d/%Y") if logs_to_date_var.get().strip() else None
        for i in logs_tree.get_children():
            logs_tree.delete(i)
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT id, date_str, status, start_time_str, end_time_str, duration_str FROM logs WHERE user_id = ?",
                       (app_instance.current_user_id,))
        rows = cursor.fetchall()
        conn.close()
        for row in rows:
            row_date = datetime.strptime(row[1], "%m/%d/%Y") if row[1] else None
            if (from_date and row_date and row_date < from_date) or (to_date and row_date and row_date > to_date):
                continue
            if text_filter and text_filter not in " ".join(str(item).lower() for item in row):
                continue
            logs_tree.insert("", tk.END, values=row)
        if logs_current_sort[0]:
            logs_sort_column(logs_tree, logs_current_sort[0], logs_current_sort[1])

    logs_load_data()

    # Attendance Tab
    attendance_frame = ttk.Frame(notebook)
    notebook.add(attendance_frame, text="Attendance")
    attendance_current_sort = [None, False]
    attendance_columns = ("ID", "Datetime", "Status")
    att_filter_frame = ttk.Frame(attendance_frame, padding=10)
    att_filter_frame.pack(fill=X)
    ttk.Label(att_filter_frame, text="Status:", font="-size 10 -weight bold").grid(row=0, column=0, padx=5, pady=5, sticky="w")
    attendance_text_filter_var = tk.StringVar()
    ttk.Entry(att_filter_frame, textvariable=attendance_text_filter_var).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
    ttk.Label(att_filter_frame, text="From Date:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
    attendance_from_date_var = tk.StringVar()
    ttk.Entry(att_filter_frame, textvariable=attendance_from_date_var).grid(row=1, column=1, padx=5, pady=5, sticky="ew")
    ttk.Label(att_filter_frame, text="To Date:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
    attendance_to_date_var = tk.StringVar()
    ttk.Entry(att_filter_frame, textvariable=attendance_to_date_var).grid(row=2, column=1, padx=5, pady=5, sticky="ew")
    ttk.Button(att_filter_frame, text="Apply Filter", bootstyle=INFO, command=lambda: attendance_load_data()).grid(row=3, column=1, padx=5, pady=5, sticky="e")
    att_filter_frame.columnconfigure(1, weight=1)

    attendance_tree = ttk.Treeview(attendance_frame, columns=attendance_columns, show="headings")
    attendance_vsb = ttk.Scrollbar(attendance_frame, orient="vertical", command=attendance_tree.yview)
    attendance_tree.configure(yscrollcommand=attendance_vsb.set)
    attendance_vsb.pack(side=RIGHT, fill=Y)
    attendance_tree.pack(fill=BOTH, expand=True)

    def attendance_sort_column(tv, col, reverse):
        l = [(tv.set(k, col), k) for k in tv.get_children('')]
        try:
            l.sort(key=lambda t: int(t[0]), reverse=reverse)
        except ValueError:
            l.sort(reverse=reverse)
        for index, (val, k) in enumerate(l):
            tv.move(k, '', index)
        for c in attendance_columns:
            tv.heading(c, text=c, command=lambda _col=c: attendance_sort_column(tv, _col, False))
        tv.heading(col, text=col + (" ▲" if not reverse else " ▼"), command=lambda: attendance_sort_column(tv, col, not reverse))
        attendance_current_sort[0] = col
        attendance_current_sort[1] = reverse

    for col in attendance_columns:
        attendance_tree.heading(col, text=col, command=lambda _col=col: attendance_sort_column(attendance_tree, _col, False))
        attendance_tree.column(col, anchor="center", width=200 if col == "Datetime" else 100 if col == "Status" else 30)

    def attendance_load_data():
        text_filter = attendance_text_filter_var.get().lower().strip()
        from_date_str = attendance_from_date_var.get().strip()
        to_date_str = attendance_to_date_var.get().strip()
        dt_format = "%Y-%m-%d %H:%M:%S"
        from_date = datetime.strptime(from_date_str + " 00:00:00", dt_format) if from_date_str else None
        to_date = datetime.strptime(to_date_str + " 23:59:59", dt_format) if to_date_str else None
        for i in attendance_tree.get_children():
            attendance_tree.delete(i)
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT id, datetime_str, status FROM Attendance WHERE user_id = ?", (app_instance.current_user_id,))
        rows = cursor.fetchall()
        conn.close()
        for row in rows:
            row_dt = datetime.strptime(row[1], dt_format) if row[1] else None
            if (from_date and row_dt and row_dt < from_date) or (to_date and row_dt and row_dt > to_date):
                continue
            if text_filter and text_filter not in " ".join(str(item).lower() for item in row):
                continue
            attendance_tree.insert("", tk.END, values=row)
        if attendance_current_sort[0]:
            attendance_sort_column(attendance_tree, attendance_current_sort[0], attendance_current_sort[1])

    attendance_load_data()

    def refresh():
        logs_load_data()
        attendance_load_data()
        view_win.after(1000, refresh)

    refresh()

def export_logs_to_csv():
    """Export logs to a CSV file."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, user_id, date_str, status, start_time_str, end_time_str, duration_str FROM logs")
    rows = cursor.fetchall()
    conn.close()
    filename = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
    if filename:
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["ID", "UserID", "Date", "Status", "Start Time", "End Time", "Duration"])
            writer.writerows(rows)
        messagebox.showinfo("Export", "Logs exported to CSV successfully.")

def export_logs_to_excel():
    """Export logs to an Excel file."""
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT id, user_id, date_str, status, start_time_str, end_time_str, duration_str FROM logs", conn)
    conn.close()
    filename = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel Files", "*.xlsx")])
    if filename:
        df.to_excel(filename, index=False)
        messagebox.showinfo("Export", "Logs exported to Excel successfully.")

def export_attendance_to_csv():
    """Export attendance records to a CSV file."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, user_id, datetime_str, status FROM Attendance")
    rows = cursor.fetchall()
    conn.close()
    filename = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
    if filename:
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["ID", "UserID", "Datetime", "Status"])
            writer.writerows(rows)
        messagebox.showinfo("Export", "Attendance exported to CSV successfully.")

def export_attendance_to_excel():
    """Export attendance records to an Excel file."""
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT id, user_id, datetime_str, status FROM Attendance", conn)
    conn.close()
    filename = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel Files", "*.xlsx")])
    if filename:
        df.to_excel(filename, index=False)
        messagebox.showinfo("Export", "Attendance exported to Excel successfully.")

def show_configuration_panel(app_instance, config, save_config_func):
    """Display a configuration panel for adjusting settings."""
    config_win = tk.Toplevel(app_instance.master)
    config_win.title("Configuration Panel")
    config_win.geometry("400x300")
    entries = {}
    for row, (label, key) in enumerate([("Detection Confidence:", "detection_confidence"),
                                        ("Inactive Threshold (sec):", "inactive_threshold"),
                                        ("Cosine Similarity Cutoff:", "cosine_similarity_cutoff"),
                                        ("Default Camera Index:", "camera_index")]):
        ttk.Label(config_win, text=label).grid(row=row, column=0, padx=5, pady=5, sticky="w")
        var = tk.StringVar(value=str(config.get(key)))
        ttk.Entry(config_win, textvariable=var).grid(row=row, column=1, padx=5, pady=5, sticky="ew")
        entries[key] = var
    config_win.columnconfigure(1, weight=1)

    def save_settings():
        try:
            config.update({
                "detection_confidence": float(entries["detection_confidence"].get()),
                "inactive_threshold": float(entries["inactive_threshold"].get()),
                "cosine_similarity_cutoff": float(entries["cosine_similarity_cutoff"].get()),
                "camera_index": int(entries["camera_index"].get())
            })
            save_config_func(config)
            messagebox.showinfo("Configuration", "Settings saved. Restart to apply.")
            config_win.destroy()
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input: {e}")

    ttk.Button(config_win, text="Save", command=save_settings).grid(row=4, column=0, columnspan=2, pady=10)

def create_system_tray(app_instance, script_dir):
    """Create a system tray icon for the application."""
    def on_restore(icon, item):
        app_instance.master.deiconify()
    def on_exit(icon, item):
        icon.stop()
        app_instance.master.destroy()
    icon_path = os.path.join(script_dir, "icons", "PresenX_Logo.ico")
    icon_image = PilImage.open(icon_path)
    tray_menu = (pystray.MenuItem("Restore", on_restore), pystray.MenuItem("Exit", on_exit))
    tray_icon = pystray.Icon("PresenX", icon_image, "PresenX", menu=pystray.Menu(*tray_menu))
    tray_icon.run()

def start_tray_icon(app_instance, script_dir):
    """Start the system tray icon in a separate thread."""
    threading.Thread(target=lambda: create_system_tray(app_instance, script_dir), daemon=True).start()

def restart_app():
    """Restart the application."""
    os.execl(sys.executable, sys.executable, *sys.argv)

# ====================================================
# Main Section
# ====================================================
if __name__ == "__main__":
    root = ttk.Window(themename="lumen")
    try:
        app_icon = tk.PhotoImage(file=os.path.join(SCRIPT_DIR, "icons", "PresenX_Logo.png"))
        root.iconphoto(False, app_icon)
    except Exception as e:
        print(f"Error loading app icon: {e}")
    try:
        root.iconbitmap(os.path.join(SCRIPT_DIR, "icons", "PresenX_Logo.ico"))
    except Exception:
        pass
    app = PresenXIntegrated(root)

    def handle_sigterm(signum, frame):
        print("Received SIGTERM. Closing gracefully.")
        app.on_closing()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)

    menubar = tk.Menu(root)
    file_menu = tk.Menu(menubar, tearoff=0)
    file_menu.add_command(label="Export Logs to CSV", command=export_logs_to_csv)
    file_menu.add_command(label="Export Logs to Excel", command=export_logs_to_excel)
    file_menu.add_command(label="Export Attendance to CSV", command=export_attendance_to_csv)
    file_menu.add_command(label="Export Attendance to Excel", command=export_attendance_to_excel)
    file_menu.add_separator()
    file_menu.add_command(label="Configuration Panel", command=lambda: show_configuration_panel(app, config, save_config))
    file_menu.add_command(label="Restart App", command=restart_app)
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=root.quit)
    menubar.add_cascade(label="File", menu=file_menu)
    root.config(menu=menubar)

    start_tray_icon(app, SCRIPT_DIR)
    app.pause_reason_combobox.bind("<<ComboboxSelected>>", app.on_reason_selected)
    root.mainloop()
