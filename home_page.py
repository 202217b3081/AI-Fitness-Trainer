import streamlit as st
import mediapipe as mp
import numpy as np
import time
import math
import cv2
import pandas as pd
import os
import re
import io
from queue import Queue, Empty, Full
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
from EmailingSystem import email_user
from graph import single_plot,double_plot
from calc import cals

import sqlite3
import secrets
import hashlib
from datetime import datetime

# Authentication database and secure password/OTP helpers
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUTH_DB = os.path.join(BASE_DIR, "fitness_users.db")

def hash_password(password):
    salt = secrets.token_hex(16)
    derived = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 200000
    ).hex()
    return f"{salt}${derived}"

def verify_password(password, stored_hash):
    try:
        salt, expected = stored_hash.split("$", 1)
        actual = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt.encode("utf-8"), 200000
        ).hex()
        return secrets.compare_digest(actual, expected)
    except (ValueError, AttributeError):
        return False

def init_auth_db():
    with sqlite3.connect(AUTH_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                email TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

def user_exists(email_address):
    with sqlite3.connect(AUTH_DB) as conn:
        row = conn.execute(
            "SELECT email FROM users WHERE email = ?", (email_address.lower().strip(),)
        ).fetchone()
    return row is not None

def create_user(name, email_address, password):
    with sqlite3.connect(AUTH_DB) as conn:
        conn.execute(
            "INSERT INTO users (email, name, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (email_address.lower().strip(), name.strip(), hash_password(password),
             datetime.now().isoformat(timespec="seconds"))
        )

def check_login(email_address, password):
    with sqlite3.connect(AUTH_DB) as conn:
        row = conn.execute(
            "SELECT name, password_hash FROM users WHERE email = ?",
            (email_address.lower().strip(),)
        ).fetchone()
    return row[0] if row and verify_password(password, row[1]) else None

init_auth_db()


def init_feature_db():
    """Store per-user goals without changing the existing workout CSV."""
    with sqlite3.connect(AUTH_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_goals (
                email TEXT PRIMARY KEY,
                weekly_workouts INTEGER NOT NULL DEFAULT 3,
                daily_calories REAL NOT NULL DEFAULT 150,
                updated_at TEXT NOT NULL
            )
        """)

def get_user_goals(email_address):
    with sqlite3.connect(AUTH_DB) as conn:
        row = conn.execute(
            "SELECT weekly_workouts, daily_calories FROM user_goals WHERE email = ?",
            (email_address.lower().strip(),)
        ).fetchone()
    return row if row else (3, 150.0)

def save_user_goals(email_address, weekly_workouts, daily_calories):
    with sqlite3.connect(AUTH_DB) as conn:
        conn.execute("""
            INSERT INTO user_goals (email, weekly_workouts, daily_calories, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(email) DO UPDATE SET
                weekly_workouts=excluded.weekly_workouts,
                daily_calories=excluded.daily_calories,
                updated_at=excluded.updated_at
        """, (email_address.lower().strip(), int(weekly_workouts),
              float(daily_calories), datetime.now().isoformat(timespec="seconds")))

def load_user_history(email_address):
    history_file = os.path.join(BASE_DIR, "workout_history.csv")
    if not os.path.exists(history_file):
        return pd.DataFrame(columns=["Date", "Name", "Email", "Exercise", "Summary"])
    try:
        data = pd.read_csv(history_file)
        if "Email" in data.columns:
            data = data[data["Email"].fillna("").astype(str).str.lower()
                        == email_address.lower()]
        return data.copy()
    except Exception:
        return pd.DataFrame(columns=["Date", "Name", "Email", "Exercise", "Summary"])

def history_totals(data):
    reps = 0
    calories = 0.0
    if not data.empty and "Summary" in data.columns:
        for summary in data["Summary"].fillna("").astype(str):
            reps_match = re.search(r"for\s+(\d+)\s+times", summary, re.IGNORECASE)
            calories_match = re.search(
                r"burnt\s+([0-9]+(?:\.[0-9]+)?)\s+calories", summary, re.IGNORECASE
            )
            if reps_match:
                reps += int(reps_match.group(1))
            if calories_match:
                calories += float(calories_match.group(1))
    return len(data), reps, calories

init_feature_db()


class BrowserCameraProcessor(VideoProcessorBase):
    """Receives camera frames from the visitor's browser (not the cloud server)."""
    def __init__(self):
        self.frame_queue = Queue(maxsize=1)

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        # Keep only the newest frame so a slow 4 GB laptop does not build a backlog.
        try:
            self.frame_queue.put_nowait(image)
        except Full:
            try:
                self.frame_queue.get_nowait()
            except Empty:
                pass
            try:
                self.frame_queue.put_nowait(image)
            except Full:
                pass
        return frame


class BrowserCameraCapture:
    """Small OpenCV-like adapter around frames received through WebRTC."""
    def __init__(self, frame_queue, context):
        self.frame_queue = frame_queue
        self.context = context

    def isOpened(self):
        return bool(self.context.state.playing)

    def read(self):
        try:
            return True, self.frame_queue.get(timeout=1.0)
        except Empty:
            return False, None

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return 640
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return 480
        if prop == cv2.CAP_PROP_FPS:
            return 10
        return 0

    def release(self):
        # The browser owns the camera; the visitor stops it with WebRTC's STOP button.
        return None


flag=1
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# user position
stage = None
message=" "
# count of correct movement
counter = 0
l_angles=[]
r_angles=[]
frames=[]
frame_count=0



@st.cache_resource()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)

    return resized


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def get_pos(img, results):
    landmarks = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        landmarks.append([id, cx, cy])
    return landmarks

def drawOn(img, p1, p2, p3, angle,lmList):
    x1, y1 = lmList[p1][1:]
    x2, y2 = lmList[p2][1:]
    x3, y3 = lmList[p3][1:]

    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv2.line(img, (x3, y3), (x2, y2), (0, 255, 0), 1)
    cv2.circle(img, (x1, y1), 5, (255, 0, 0), cv2.FILLED)
    cv2.circle(img, (x2, y2), 5, (255, 0, 0), cv2.FILLED)
    cv2.circle(img, (x3, y3), 5, (255, 0, 0), cv2.FILLED)
    cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)





st.markdown(
    """
    <style>
    :root {
        --bg: #fff7fb;
        --panel: #ffffff;
        --panel-soft: #fff0f6;
        --green: #7c5ce7;
        --green-light: #6847d8;
        --muted: #6f6680;
    }

    .stApp {
        background:
            radial-gradient(circle at 8% 5%, rgba(255, 190, 214, .42), transparent 27%),
            radial-gradient(circle at 92% 12%, rgba(183, 218, 255, .42), transparent 29%),
            linear-gradient(135deg, #fff8fc 0%, #f5f3ff 52%, #eff8ff 100%);
        color: #342d45;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8eaff 0%, #eaf4ff 100%);
        border-right: 1px solid #eadcf7;
    }
    [data-testid="stSidebar"] * { color: #403451; }

    h1, h2, h3 {
        color: #6847d8 !important;
        font-weight: 800 !important;
        letter-spacing: -.4px;
    }
    h1 { font-size: clamp(1.65rem, 3vw, 2.25rem) !important; }

    p, label, [data-testid="stMarkdownContainer"] { color: #514860; }

    .stTextInput input, .stSelectbox div[data-baseweb="select"],
    .stNumberInput input {
        background: #ffffff;
        border: 1px solid #e4d7f4;
        border-radius: 16px;
        color: #342d45;
        min-height: 44px;
    }

    /* Keep text readable in Streamlit's disabled, pre-filled profile fields. */
    .stTextInput input:disabled,
    .stTextInput input[disabled],
    [data-testid="stTextInput"] input:disabled {
        background: #ffffff !important;
        color: #342d45 !important;
        -webkit-text-fill-color: #342d45 !important;
        opacity: 1 !important;
        border: 1.5px solid #d9c6ed !important;
        box-shadow: none !important;
        cursor: default !important;
    }
    .stTextInput input::placeholder,
    [data-testid="stTextInput"] input::placeholder {
        color: #756b83 !important;
        -webkit-text-fill-color: #756b83 !important;
        opacity: 1 !important;
    }

    /* Keep Streamlit's selected dropdown and its opened menu in the same soft theme. */
    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-baseweb="select"] div[role="combobox"],
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {
        background: linear-gradient(110deg, #fff0f7 0%, #f1eaff 58%, #eaf6ff 100%) !important;
        border: 1px solid #d9bce9 !important;
        border-radius: 14px !important;
        color: #493653 !important;
        box-shadow: 0 3px 10px rgba(145, 105, 170, .10) !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] *,
    [data-testid="stSidebar"] [data-baseweb="select"] svg {
        color: #493653 !important;
        fill: #8c68ad !important;
    }
    [data-baseweb="popover"] [role="listbox"],
    [data-baseweb="popover"] [data-baseweb="menu"],
    [data-baseweb="popover"] ul {
        background: #fff7fb !important;
        border: 1px solid #e7d3f1 !important;
        border-radius: 14px !important;
        box-shadow: 0 10px 24px rgba(80, 55, 100, .16) !important;
        overflow: hidden !important;
    }
    [data-baseweb="popover"] [role="option"],
    [data-baseweb="popover"] li {
        background: #fff7fb !important;
        color: #493653 !important;
    }
    [data-baseweb="popover"] [role="option"]:hover,
    [data-baseweb="popover"] li:hover {
        background: #f5e6fb !important;
        color: #493653 !important;
    }
    [data-baseweb="popover"] [aria-selected="true"] {
        background: #eadcf8 !important;
        color: #493653 !important;
        font-weight: 750 !important;
    }

    .login-wave-overlay {
        position: fixed;
        inset: 0;
        z-index: 999999;
        display: flex;
        align-items: center;
        justify-content: center;
        pointer-events: none;
        background: rgba(255, 247, 251, .20);
        animation: wave-overlay-out 1.8s ease forwards;
    }
    .login-wave-hand {
        font-size: clamp(100px, 24vw, 210px);
        line-height: 1;
        filter: drop-shadow(0 12px 18px rgba(116, 78, 145, .20));
        transform-origin: 70% 80%;
        animation: hand-wave-zoom 1.8s cubic-bezier(.2,.75,.25,1) forwards;
    }
    @keyframes hand-wave-zoom {
        0% { opacity: 0; transform: scale(.18) rotate(-18deg); }
        18% { opacity: 1; transform: scale(.75) rotate(10deg); }
        38% { opacity: 1; transform: scale(1.05) rotate(-13deg); }
        54% { opacity: 1; transform: scale(1.28) rotate(12deg); }
        72% { opacity: .9; transform: scale(1.65) rotate(-8deg); }
        100% { opacity: 0; transform: scale(2.4) rotate(5deg); }
    }
    @keyframes wave-overlay-out {
        0%, 72% { opacity: 1; }
        100% { opacity: 0; visibility: hidden; }
    }

    [data-testid="stWidgetLabel"] p {
        color: #514860 !important;
        font-weight: 700;
    }

    .stButton button, .stDownloadButton button, [data-testid="stFormSubmitButton"] button {
        background: linear-gradient(100deg, #a78bfa, #f4a6c8 55%, #8ecdf5);
        color: #302342;
        border: 0;
        border-radius: 999px;
        font-weight: 800;
        min-height: 44px;
        padding: .55rem 1.35rem;
        box-shadow: 0 5px 14px rgba(137, 103, 190, .15);
        transition: transform .18s ease, box-shadow .18s ease;
    }
    .stButton button:hover, .stDownloadButton button:hover,
    [data-testid="stFormSubmitButton"] button:hover {
        color: #302342;
        border: 0;
        transform: translateY(-2px);
        box-shadow: 0 8px 18px rgba(137, 103, 190, .22);
    }

    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #ffffff, #f8efff);
        padding: 18px;
        border-radius: 22px;
        border: 1px solid #eadcf7;
        box-shadow: 0 5px 18px rgba(91, 68, 130, .07);
    }
    [data-testid="stMetricValue"] { color: #6847d8 !important; }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255,255,255,.78);
        border: 1px solid #eadcf7;
        border-radius: 20px;
        box-shadow: 0 4px 16px rgba(91, 68, 130, .05);
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #eadcf7;
        border-radius: 16px;
        overflow: hidden;
    }

    [data-testid="stVideo"] {
        background: #f5efff;
        border: 1px solid #e5d8f5;
        border-radius: 18px;
        padding: 8px;
        overflow: hidden;
        margin: 8px 0 28px 0;
    }
    [data-testid="stImage"] img {
        border: 1px solid #e5d8f5;
        border-radius: 18px;
    }
    .tutorial-link {
        display: inline-block;
        background: linear-gradient(100deg,#8b70e8,#e88db9);
        color: #ffffff !important;
        padding: 9px 16px;
        border-radius: 999px;
        text-decoration: none !important;
        font-weight: 800;
        margin: 0 0 8px 0;
    }
    .page-hero {
        background: linear-gradient(115deg,#f4e8ff 0%,#ffeaf2 55%,#e8f5ff 100%);
        border: 1px solid #e8d9f5;
        border-left: 6px solid #a78bfa;
        border-radius: 24px;
        padding: 24px 26px;
        margin: 4px 0 22px 0;
        box-shadow: 0 7px 22px rgba(120, 91, 160, .08);
    }
    .page-hero p { color: #514860 !important; }
    .info-card {
        background: linear-gradient(145deg,rgba(255,255,255,.98),rgba(250,242,255,.98));
        border: 1px solid #eadcf7;
        border-radius: 20px;
        padding: 17px 19px;
        min-height: 90px;
        margin-bottom: 12px;
        box-shadow: 0 4px 15px rgba(91, 68, 130, .05);
    }
    .step-number {
        display:inline-flex;
        align-items:center;
        justify-content:center;
        width:30px;
        height:30px;
        border-radius:50%;
        background: linear-gradient(135deg,#a78bfa,#f4a6c8);
        color:#ffffff;
        font-weight:800;
        margin-right:10px;
    }
    hr { border-color: #e9dff3; }

    [data-testid="stCheckbox"] label {
        background: #f5edff;
        border: 1px solid #dfcff4;
        border-radius: 16px;
        padding: 10px 14px;
        font-weight: 750;
    }

    a { color: #7656d8 !important; }

    @media (max-width: 768px) {
        .block-container { padding-left: 1rem; padding-right: 1rem; }
    }

    /* Pastel dropdown fix: style both the closed select and the opened option menu. */
    [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        background: linear-gradient(110deg, #fff0f7 0%, #f1eaff 58%, #eaf6ff 100%) !important;
        color: #493653 !important;
        border: 1px solid #d9bce9 !important;
        border-radius: 14px !important;
    }
    [data-baseweb="popover"],
    [data-baseweb="popover"] > div,
    [data-baseweb="popover"] [data-baseweb="menu"],
    [data-baseweb="popover"] [role="listbox"],
    [data-baseweb="popover"] ul,
    ul[role="listbox"] {
        background: #fff7fb !important;
        color: #493653 !important;
        border: 1px solid #e7d3f1 !important;
        border-radius: 14px !important;
    }
    [data-baseweb="popover"] *,
    [data-baseweb="popover"] [role="option"],
    [data-baseweb="popover"] li,
    li[role="option"] {
        background-color: #fff7fb !important;
        color: #493653 !important;
    }
    [data-baseweb="popover"] [role="option"]:hover,
    [data-baseweb="popover"] li:hover,
    li[role="option"]:hover {
        background-color: #f5e6fb !important;
        color: #493653 !important;
    }
    [data-baseweb="popover"] [aria-selected="true"],
    [role="option"][aria-selected="true"] {
        background-color: #eadcf8 !important;
        color: #493653 !important;
        font-weight: 750 !important;
    }
    /* Professional finishing layer: restrained colours, clear hierarchy, consistent controls. */
    .stApp { color: #393448; }
    h1, h2, h3 { letter-spacing: -0.025em; }
    h2 { font-size: clamp(1.25rem, 2.2vw, 1.65rem) !important; }
    h3 { font-size: 1.12rem !important; }
    [data-testid="stSidebar"] { box-shadow: 2px 0 14px rgba(80, 65, 110, .04); }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #5A5369; }
    .page-hero { border-left-width: 4px; box-shadow: 0 5px 16px rgba(87, 72, 120, .06); }
    .info-card { box-shadow: 0 3px 12px rgba(87, 72, 120, .045); }
    .stButton button, .stDownloadButton button,
    [data-testid="stFormSubmitButton"] button {
        background: #E9E1F6 !important;
        color: #493D63 !important;
        border: 1px solid #D8CBEA !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        box-shadow: 0 2px 7px rgba(87, 72, 120, .07) !important;
    }
    .stButton button:hover, .stDownloadButton button:hover,
    [data-testid="stFormSubmitButton"] button:hover {
        background: #DFD3F0 !important;
        border-color: #CDBBE5 !important;
        transform: translateY(-1px);
    }
    .tutorial-link {
        background: #E8E1F4 !important;
        color: #514466 !important;
        border: 1px solid #D8CBEA;
        border-radius: 10px;
        padding: 8px 13px;
        font-weight: 700;
        box-shadow: none;
    }
    .tutorial-link:hover { background: #DED3EF !important; }
    [data-testid="stMetric"] { border-radius: 16px; }
    [data-testid="stDataFrame"] { background: #FFFFFF; }
    /* Replace any remaining dark native select surfaces with a soft pastel surface. */
    select, option {
        background-color: #fff7fb !important;
        color: #493653 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Sign in / Sign up gate: the workout application is shown only after login.
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "pending_signup" not in st.session_state:
    st.session_state.pending_signup = False

if not st.session_state.logged_in:
    st.markdown("""
    <div class="page-hero">
      <h1 style="margin:0;color:#8B6BCB !important;">AI Fitness Trainer</h1>
      <p style="margin:8px 0 0 0;color:#514860;">Sign in to continue your personal fitness journey.</p>
    </div>
    """, unsafe_allow_html=True)

    auth_tab, signup_tab = st.tabs(["Sign In", "Create Account"])

    with auth_tab:
        with st.form("signin_form"):
            login_email = st.text_input("Email Address", placeholder="Enter your registered email")
            login_password = st.text_input("Password", type="password", placeholder="Enter your password")
            login_submit = st.form_submit_button("Sign In", use_container_width=True)
        if login_submit:
            if not login_email.strip() or not login_password:
                st.error("Please enter both email and password.")
            else:
                account_name = check_login(login_email, login_password)
                if account_name:
                    st.session_state.logged_in = True
                    st.session_state.show_login_wave = True
                    st.session_state.user_name = account_name
                    st.session_state.user_email = login_email.strip().lower()
                    st.rerun()
                else:
                    st.error("Incorrect email or password. Please try again.")

    with signup_tab:
        with st.form("signup_form"):
            signup_name = st.text_input("Full Name", placeholder="Enter your name")
            signup_email = st.text_input("Email Address", placeholder="Enter your email")
            signup_password = st.text_input("Create Password", type="password")
            signup_confirm = st.text_input("Confirm Password", type="password")
            signup_submit = st.form_submit_button("Create Account", use_container_width=True)

        if signup_submit:
            clean_email = signup_email.strip().lower()
            if not signup_name.strip() or not clean_email or not signup_password:
                st.error("Please fill in all fields.")
            elif not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", clean_email):
                st.error("Please enter a valid email address.")
            elif len(signup_password) < 8:
                st.error("Password must contain at least 8 characters.")
            elif signup_password != signup_confirm:
                st.error("Passwords do not match.")
            elif user_exists(clean_email):
                st.error("An account with this email already exists. Please sign in.")
            else:
                try:
                    create_user(signup_name, clean_email, signup_password)
                    st.session_state.logged_in = True
                    st.session_state.show_login_wave = True
                    st.session_state.user_name = signup_name.strip()
                    st.session_state.user_email = clean_email
                    st.success("Account created successfully!")
                    st.rerun()
                except sqlite3.IntegrityError:
                    st.error("An account with this email already exists. Please sign in.")
                except Exception as exc:
                    st.error(f"Could not create account: {exc}")
    st.stop()

# One-time full-screen waving-hand greeting immediately after successful login/signup.
if st.session_state.pop("show_login_wave", False):
    st.markdown(
        '<div class="login-wave-overlay"><div class="login-wave-hand">👋</div></div>',
        unsafe_allow_html=True,
    )

st.sidebar.markdown("""
<div style="padding:12px 2px 18px 2px;">
  <div style="font-size:25px;font-weight:800;color:#8B6BCB;line-height:1.2;">
    💜 AI Fitness Trainer ✨
  </div>
  <div style="font-size:13px;color:#766B86;margin-top:8px;">
    👋 Powered by MediaPipe &amp; OpenCV 💪
  </div>
</div>
""", unsafe_allow_html=True)
app_mode = st.sidebar.selectbox('Menu', ['Training', 'Progress Dashboard', 'Goals & Achievements', 'Rest Timer', 'About App', 'Demos and Tutorials', 'History'])

st.sidebar.markdown("---")
st.sidebar.caption("Signed in as")
st.sidebar.markdown(f"**{st.session_state.user_name}**")
st.sidebar.caption(st.session_state.user_email)
if st.sidebar.button("Logout", use_container_width=True):
    for key in ["logged_in", "user_name", "user_email"]:
        st.session_state.pop(key, None)
    st.rerun()


if app_mode == 'Training':
    st.markdown("""
    <div style="
        background: linear-gradient(120deg, #F2E8FA, #EAF3FF);
        padding: 30px;
        border-radius: 18px;
        border-left: 6px solid #C7A7E8;
        margin-bottom: 25px;">
        <h1 style="color:#8B6BCB; margin:0;">
            Welcome to AI Fitness Trainer
        </h1>
        <p style="color:#514860; font-size:18px; font-weight:650;">
            Train Smart. Move Better. Stay Fit.
        </p>
        <p style="color:#6D607D;">
            Your AI-powered workout companion using MediaPipe and OpenCV.
        </p>
    </div>
    """, unsafe_allow_html=True)


if app_mode == 'Progress Dashboard':
    st.markdown("""
    <div class="page-hero">
      <h1 style="margin:0;color:#8B6BCB !important;">📈 Weekly Progress Dashboard</h1>
      <p style="margin:8px 0 0 0;color:#514860;">A simple view of your saved workout activity.</p>
    </div>
    """, unsafe_allow_html=True)
    dashboard_data = load_user_history(st.session_state.user_email)
    total_workouts, total_reps, total_calories = history_totals(dashboard_data)
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Workouts", total_workouts)
    m2.metric("Total Repetitions", total_reps)
    m3.metric("Estimated Calories", f"{total_calories:.1f} kcal")
    if not dashboard_data.empty and "Date" in dashboard_data.columns:
        dashboard_data["Date"] = pd.to_datetime(dashboard_data["Date"], errors="coerce")
        dashboard_data = dashboard_data.dropna(subset=["Date"])
        if not dashboard_data.empty:
            dashboard_data["Week"] = dashboard_data["Date"].dt.to_period("W").astype(str)
            weekly = dashboard_data.groupby("Week").size().tail(8).rename("Workouts")
            st.markdown("### Workouts per week")
            st.bar_chart(weekly)
            recent = dashboard_data.sort_values("Date").tail(10).set_index("Date")
            st.markdown("### Recent workout activity")
            st.dataframe(recent[["Exercise"]], use_container_width=True)
        else:
            st.info("Workout dates could not be read yet. Complete another workout to build your chart.")
    else:
        st.info("Complete a workout to start building your progress dashboard.")

elif app_mode == 'Goals & Achievements':
    st.markdown("""
    <div class="page-hero">
      <h1 style="margin:0;color:#8B6BCB !important;">🎯 Goals & Achievements</h1>
      <p style="margin:8px 0 0 0;color:#514860;">Set personal targets and earn badges from saved workouts.</p>
    </div>
    """, unsafe_allow_html=True)
    current_week_goal, current_calorie_goal = get_user_goals(st.session_state.user_email)
    with st.form("fitness_goals_form"):
        weekly_goal = st.number_input("Weekly workout goal", min_value=1, max_value=14,
                                      value=int(current_week_goal), step=1)
        calorie_goal = st.number_input("Daily estimated calorie goal (kcal)", min_value=10.0,
                                       max_value=5000.0, value=float(current_calorie_goal), step=10.0)
        save_goals = st.form_submit_button("Save My Goals", use_container_width=True)
    if save_goals:
        save_user_goals(st.session_state.user_email, weekly_goal, calorie_goal)
        st.success("Your goals have been saved.")
        st.rerun()

    goal_data = load_user_history(st.session_state.user_email)
    total_w, total_r, total_c = history_totals(goal_data)
    if not goal_data.empty and "Date" in goal_data.columns:
        dates = pd.to_datetime(goal_data["Date"], errors="coerce")
        today = pd.Timestamp.today().normalize()
        this_week_count = int((dates >= (today - pd.Timedelta(days=today.weekday()))).sum())
        today_count = int((dates.dt.date == today.date()).sum())
    else:
        this_week_count, today_count = 0, 0

    st.markdown("### Your goal progress")
    st.progress(min(this_week_count / max(int(weekly_goal), 1), 1.0))
    st.write(f"Weekly workouts: **{this_week_count} / {int(weekly_goal)}**")
    today_calories = 0.0
    if not goal_data.empty and "Date" in goal_data.columns and "Summary" in goal_data.columns:
        goal_dates = pd.to_datetime(goal_data["Date"], errors="coerce")
        today_mask = goal_dates.dt.date == pd.Timestamp.today().date()
        for summary in goal_data.loc[today_mask, "Summary"].fillna("").astype(str):
            match = re.search(r"burnt\\s+([0-9]+(?:\\.[0-9]+)?)\\s+calories", summary, re.IGNORECASE)
            if match:
                today_calories += float(match.group(1))
    st.progress(min(today_calories / max(float(calorie_goal), 1.0), 1.0))
    st.write(f"Today's estimated calories: **{today_calories:.1f} / {float(calorie_goal):.0f} kcal**")
    st.caption("Calories are estimates from saved workouts, not a medical or nutrition measurement.")

    st.markdown("### 🏅 Achievement badges")
    badges = [
        ("First Workout", total_w >= 1, "Complete your first workout."),
        ("50 Repetitions", total_r >= 50, "Reach 50 saved repetitions."),
        ("100 Repetitions", total_r >= 100, "Reach 100 saved repetitions."),
        ("5 Workouts", total_w >= 5, "Save five workout sessions."),
        ("Weekly Goal", this_week_count >= int(weekly_goal), "Meet your weekly workout target.")
    ]
    badge_cols = st.columns(2)
    for idx, (badge_name, unlocked, description) in enumerate(badges):
        with badge_cols[idx % 2]:
            if unlocked:
                st.success(f"🏆 {badge_name}\n\n{description}")
            else:
                st.info(f"🔒 {badge_name}\n\n{description}")

elif app_mode == 'Rest Timer':
    st.markdown("""
    <div class="page-hero">
      <h1 style="margin:0;color:#8B6BCB !important;">⏱️ Workout Rest Timer</h1>
      <p style="margin:8px 0 0 0;color:#514860;">Take a short break between exercise sets.</p>
    </div>
    """, unsafe_allow_html=True)
    rest_seconds = st.select_slider("Choose rest duration (seconds)",
                                    options=[15, 30, 45, 60, 90, 120], value=30)
    if st.button("Start Rest Timer", use_container_width=True):
        timer_display = st.empty()
        progress_display = st.progress(0)
        for remaining in range(int(rest_seconds), 0, -1):
            timer_display.markdown(
                f"<h1 style='text-align:center;color:#8B6BCB;font-size:56px'>{remaining}s</h1>",
                unsafe_allow_html=True
            )
            progress_display.progress((int(rest_seconds) - remaining + 1) / int(rest_seconds))
            time.sleep(1)
        timer_display.success("Rest completed! Ready for your next set?")
        progress_display.progress(1.0)
    st.caption("The timer runs in the app session. On a 4 GB RAM laptop it uses very little extra memory.")

if app_mode == 'About App':
    st.markdown("""
    <div class="page-hero">
      <h1 style="margin:0;color:#8B6BCB !important;">About AI Fitness Trainer</h1>
      <p style="margin:8px 0 0 0;color:#514860;">An AI-powered exercise guidance application built with MediaPipe, OpenCV and Streamlit.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
      <h3 style="margin:0 0 8px 0;">🤖 About the Application</h3>
      <p style="margin:0;color:#514860;">MediaPipe detects body pose landmarks, OpenCV processes frames received securely from the browser camera through WebRTC, and Streamlit provides the web-based graphical user interface.</p>
    </div>
    <h2 style="margin-top:22px;">📋 How to Use</h2>
    """, unsafe_allow_html=True)

    steps = [
        "Choose Training from the menu to start your fitness session.",
        "If you are unsure how to perform an exercise, visit Demos and Tutorials.",
        "Enter your name and email address on the Training page.",
        "Select an exercise from the dropdown list.",
        "Choose how many repetitions you want to perform.",
        "Tick Start Workout to activate the trainer.",
        "Allow the application to access and capture your pose through the camera.",
        "Start performing the selected exercise in front of the camera.",
        "Your repetition count and guidance suggestions will appear on the screen.",
        "After completing your target, check your email inbox for the performance report."
    ]
    for idx, step in enumerate(steps, start=1):
        st.markdown('<div class="info-card" style="display:flex;align-items:center;padding:13px 16px;min-height:unset;"><span class="step-number">' + str(idx) + '</span><span style="color:#514860;">' + step + '</span></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card" style="border-left:4px solid #E6B84A;margin-top:18px;">
      <h3 style="margin:0 0 10px 0;">⚠️ Before You Start</h3>
      <ul style="margin:0;color:#514860;padding-left:22px;">
        <li>Make sure the room has proper lighting.</li>
        <li>Ensure your camera is working properly.</li>
        <li>Keep your full body visible within the camera frame.</li>
        <li>Perform exercises in a safe space with enough room to move.</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)
elif app_mode == 'Demos and Tutorials':
    st.markdown("""
    <div class="page-hero">
      <h1 style="margin:0;color:#8B6BCB !important;">🎬 Exercise Demos &amp; Tutorials</h1>
      <p style="margin:8px 0 0 0;color:#514860;">Choose an exercise, watch its demonstration and follow the linked tutorial before starting your workout.</p>
    </div>
    """, unsafe_allow_html=True)
    #HIGH KNESS
    st.title("HIGH KNEES")
    link_html = '<a href="https://youtu.be/oDdkytliOqE?si=Mferh8hYmi14Rh7m" class="tutorial-link">Video tutorial &rarr;</a>'
    st.markdown(link_html, unsafe_allow_html=True)
    hk_video_url = "https://www.shutterstock.com/shutterstock/videos/1060403879/preview/stock-footage-athletic-woman-doing-high-knee-exercise-at-home-home-training-workout-home-fitness-there-is-some.webm"
    st.video(hk_video_url)
    
    #SQUATS
    st.title("SQUATS")
    link_html = '<a href="https://youtu.be/4KmY44Xsg2w?si=eh8-DmSgjZptopSs" class="tutorial-link">Video tutorial &rarr;</a>'
    st.markdown(link_html, unsafe_allow_html=True)
    s_video_url = "https://www.shutterstock.com/shutterstock/videos/1102020407/preview/stock-footage-female-butt-workout-squats-at-home-athletic-asian-woman-squats-workout-in-living-room-female.webm"
    st.video(s_video_url)
    
    #SHOULDER PRESS
    st.title("SHOULDER PRESS")
    link_html = '<a href="https://youtu.be/xe19t2_6yis?si=SIpCkA04UFBoJfs_" class="tutorial-link">Video tutorial &rarr;</a>'
    st.markdown(link_html, unsafe_allow_html=True)
    shoulder_video_url = "https://ak.picdn.net/shutterstock/videos/1104019445/preview/stock-footage-medium-shot-of-young-sportsman-performing-barbell-shoulder-press-in-gym-man-holds-barbell-on-the.mp4"
    st.video(shoulder_video_url)
    
    #LATERAL CURLS
    st.title("LATERAL CURLS")
    link_html = '<a href="https://youtu.be/PzsMitRdI_8?si=Xt_bteImzFf_vP7m" class="tutorial-link">Video tutorial &rarr;</a>'
    st.markdown(link_html, unsafe_allow_html=True)
    lcurls_video_url = "https://www.shutterstock.com/shutterstock/videos/1012307186/preview/stock-footage-bodybuilder-performs-seated-side-lateral-raise-reflection-of-exercise-in-mirror.webm"
    st.video(lcurls_video_url)
    #CURLS
    st.title("CURLS")
    link_html = '<a href="https://youtu.be/sYV-ki-1blM?si=Oi9cMYUAKZnPggP_" class="tutorial-link">Video tutorial &rarr;</a>'
    st.markdown(link_html, unsafe_allow_html=True)
    curls_video_url = "https://www.shutterstock.com/shutterstock/videos/1097465161/preview/stock-footage-athlete-exercising-with-dumbbells-in-indoor-gym-fit-man-doing-biceps-triceps-curls-workout-in-gym.webm"
    st.video(curls_video_url)
    
    
    
    
    #LEFT CURLS
    st.title("Left curls")
    link_html = '<a href="https://www.youtube.com/shorts/cHxRJdSVIkA?feature=share" class="tutorial-link">Video tutorial &rarr;</a>'
    st.markdown(link_html, unsafe_allow_html=True)
    lecurls_video_url ="https://www.shutterstock.com/shutterstock/videos/1030360235/preview/stock-footage-fitness-sport-weightlifting-and-bodybuilding-concept-man-exercising-with-dumbbells-at-home.webm"
    st.video(lecurls_video_url)
    
    
    #PUSHUPS
    st.title("PUSH UPS")
    link_html = '<a href="https://youtu.be/_UBOxUl0Sl4" class="tutorial-link">Video tutorial &rarr;</a>'
    st.markdown(link_html, unsafe_allow_html=True)
    pushups_video_url = "https://www.shutterstock.com/shutterstock/videos/1100041747/preview/stock-footage-calisthenics-outdoor-bodyweight-workout-doing-push-ups-in-sunny-day.webm"
    st.video(pushups_video_url)
    
    #PLANK
    st.title("PLANK")
    link_html = '<a href="https://youtu.be/pvIjsG5Svck?si=ry3kol3TxHF7hu3g" class="tutorial-link">Video tutorial &rarr;</a>'
    st.markdown(link_html, unsafe_allow_html=True)
    p_video_url = "https://www.shutterstock.com/shutterstock/videos/1063989619/preview/stock-footage-fitness-indian-woman-doing-plank-exercise-workout-in-gym-indoors.webm"
    st.video(p_video_url)
    
    #SIT UPS
    st.title("SIT UPS")
    link_html = '<a href="https://youtu.be/1fbU_MkV7NE?si=bcHWl9xa7PkFSsAu" class="tutorial-link">Video tutorial &rarr;</a>'
    st.markdown(link_html, unsafe_allow_html=True)
    situps_video_url = "https://cdn.coverr.co/videos/coverr-ab-workout-in-the-park-7546/1080p.mp4"
    st.video(situps_video_url)
    
    
    #KNEE BEND
    st.title("KNEE BEND")
    link_html = '<a href="https://youtu.be/W4LuS9rK0gU?si=A5dTYz_OHyQmvHHd" class="tutorial-link">Video tutorial &rarr;</a>'
    st.markdown(link_html, unsafe_allow_html=True)
    kb_video_url ="https://www.shutterstock.com/shutterstock/videos/1061478490/preview/stock-footage-caucasian-woman-spending-time-at-home-in-living-room-exercising-with-dumbbells-doing-squats-in.webm"
    st.video(kb_video_url)
    
    #CRUNCHES
    st.title("CRUNCHES")
    link_html = '<a href="https://youtu.be/Xyd_fa5zoEU?si=uK846BgYpam0H4X3" class="tutorial-link">Video tutorial &rarr;</a>'
    st.markdown(link_html, unsafe_allow_html=True)
    cru_video_url = "https://www.shutterstock.com/shutterstock/videos/1041475927/preview/stock-footage-long-haired-girl-doing-bicycle-crunches-exercise-working-out-at-empty-studio-in-front-the-windows.webm"
    st.video(cru_video_url)
    
    #HALF PLOUGH POSE
    st.title("HALF PLOUGH POSE")
    link_html = '<a href="https://youtu.be/uqKbLaXSfWE" class="tutorial-link">Video tutorial &rarr;</a>'
    st.markdown(link_html, unsafe_allow_html=True)
    hp_video_url = "https://www.shutterstock.com/shutterstock/videos/1105807867/preview/stock-footage-video-of-woman-performing-alternate-leg-ardha-halasana-this-exercise-strengthen-the-muscles-of.webm"
    st.video(hp_video_url)
    
    
    
elif app_mode == 'Training':
    st.markdown("""
    <div class="section-heading">
        <h2 style="margin:0 0 6px 0;">🏋️ Start Your Workout</h2>
        <p style="color:#766B86; font-size:16px; margin:0;">
            Set up your session and let your AI trainer guide your movement.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="
        background:linear-gradient(145deg, rgba(248, 239, 255, .98), rgba(255, 240, 247, .98));
        border:1px solid rgba(199, 167, 232, .45);
        border-radius:18px;
        padding:22px 24px 10px 24px;
        margin:16px 0 22px 0;">
        <div style="color:#8B6BCB;font-size:12px;font-weight:800;letter-spacing:1.5px;">
            PERSONAL WORKOUT SETUP
        </div>
        <div style="color:#493653;font-size:14px;margin-top:5px;">
            Add your details, choose an exercise and select your repetition target.
        </div>
    </div>
    """, unsafe_allow_html=True)

    suggestion = " "
    details_col1, details_col2 = st.columns(2, gap="large")
    with details_col1:
        name = st.text_input("Your Name", value=st.session_state.user_name, disabled=True)
    with details_col2:
        email = st.text_input("Email Address", value=st.session_state.user_email, disabled=True)

    choice_col, target_col = st.columns([1.35, 1], gap="large")
    with choice_col:
        option = st.selectbox(
            'Choose Exercise',
            ("Select One", "High Knees","Squats","Shoulder Press","Lateral Curls", "Curls", "Left Curl", "Push Ups", "Plank","Sit Up",
             "Knee Bend", "Crunches","Half Plough Pose"))
    with target_col:
        press_time = st.select_slider(
            'Target Repetitions',
            options=['0', '2', '10', '15', '20', '25', '30'])

    form_tips = {
        "High Knees": "Keep your chest upright, lift knees comfortably, and land softly.",
        "Squats": "Keep heels grounded, knees aligned with toes, and chest lifted.",
        "Shoulder Press": "Brace your core, avoid arching your lower back, and press smoothly.",
        "Lateral Curls": "Raise arms with a slight elbow bend; avoid swinging your body.",
        "Curls": "Keep elbows close to your sides and avoid swinging the weights.",
        "Left Curl": "Keep your upper arm steady and move through a controlled range.",
        "Push Ups": "Keep your body in a straight line and lower with control.",
        "Plank": "Keep shoulders, hips and ankles aligned; breathe steadily.",
        "Sit Up": "Move smoothly and avoid pulling your neck.",
        "Knee Bend": "Keep knees tracking in line with toes and move within a comfortable range.",
        "Crunches": "Lift shoulders gently; avoid pulling your head forward.",
        "Half Plough Pose": "Move slowly and stop if you feel pain or strain."
    }
    if option in form_tips:
        st.info("Form guidance: " + form_tips[option])

    cal_value=cals(option,int(press_time))
    message = "You have done " + option + " for " + press_time + " times and burnt "+str(cal_value)+" calories"

    st.markdown("""
    <div style="
        background:rgba(199, 167, 232, .18);
        border-left:4px solid #C7A7E8;
        padding:12px 16px;
        border-radius:10px;
        color:#514860;
        font-size:14px;
        margin:14px 0 18px 0;">
        💡 Tip: Keep your full body visible in the camera and exercise in a well-lit space.
    </div>
    """, unsafe_allow_html=True)

    run = st.checkbox("Start Workout")

    if option == "Left Curl":
        count = 0
        direction = 0
        prev_per = 0
        e = 0

    if option == "Push Ups":
        dir = 0

    if option == "Knee Bend":
        relax_counter = 0
        bent_counter = 0
        bent_time = 0
        relax_time = 0


    if option == "Crunches":
        f=0

    while run:
        stframe = st.empty()
        st.markdown("### Browser camera")
        st.caption("Allow camera access in your browser. For a 4 GB RAM laptop, the app processes a small 320 × 240 frame at a low frame rate.")
        camera_ctx = webrtc_streamer(
            key="ai-fitness-trainer-camera",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=BrowserCameraProcessor,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640},
                    "height": {"ideal": 480},
                    "frameRate": {"ideal": 10, "max": 10},
                },
                "audio": False,
            },
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            async_processing=True,
        )
        if not camera_ctx.state.playing or camera_ctx.video_processor is None:
            st.info("Click START in the camera panel above and allow camera permission. The workout will begin when the browser camera is connected.")
            st.stop()

        cap = BrowserCameraCapture(camera_ctx.video_processor.frame_queue, camera_ctx)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_input = int(cap.get(cv2.CAP_PROP_FPS))
        st.caption(f"Low-resource mode: camera {width} × {height} | AI processing: 320 × 240 | target 10 FPS")
        fps = 0
        i = 0

        st.markdown("""
        <div class="page-hero" style="padding:16px 20px;margin:10px 0 16px 0;">
          <h2 style="margin:0;color:#8B6BCB !important;">📹 Live Workout Session</h2>
          <p style="margin:5px 0 0 0;color:#514860;">Follow the on-screen guidance and keep your posture visible to the camera.</p>
        </div>
        """, unsafe_allow_html=True)
        ltitle, ctitle, rtitle = st.columns(3, gap="medium")

        with ltitle:
            with st.container(border=True):
                st.markdown("**FRAME RATE**")
                kpi1_text = st.empty()
                kpi1_text.markdown("<h2 style='text-align:center;color:#8B6BCB;'>0</h2>", unsafe_allow_html=True)

        with ctitle:
            with st.container(border=True):
                st.markdown("**REPETITIONS**")
                kpi2_text = st.empty()
                kpi2_text.markdown("<h2 style='text-align:center;color:#8B6BCB;'>0</h2>", unsafe_allow_html=True)

        with rtitle:
            with st.container(border=True):
                st.markdown("**TARGET**")
                kpi3_text = st.empty()
                kpi3_text.markdown("<h2 style='text-align:center;color:#8B6BCB;'>0</h2>", unsafe_allow_html=True)

        st.markdown("**💡 Trainer Suggestions**")
        sugg = st.empty()
        sugg.markdown("---")

        st.markdown("<hr/>", unsafe_allow_html=True)

        with mp_pose.Pose(static_image_mode=False,
                          model_complexity=1,
                          smooth_landmarks=False,
                          enable_segmentation=False,
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as pose:
            prevTime = 0

            failed_reads = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    failed_reads += 1
                    # Avoid a tight infinite loop if the webcam disconnects/freezes.
                    if failed_reads >= 10:
                        st.error("Browser camera stopped sending frames. Check camera permission or press STOP and START again.")
                        break
                    time.sleep(0.05)
                    continue

                failed_reads = 0
                frame_count += 1

                # Mirror the webcam view, then downscale BEFORE MediaPipe.
                frame = cv2.flip(frame, 1)
                frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                result = pose.process(frame)
                try:
                    landmarks = get_pos(frame, result)

                    rshoulder = landmarks[12][1:]
                    relbow = landmarks[14][1:]
                    rwrist = landmarks[16][1:]

                    lshoulder = landmarks[11][1:]
                    lelbow = landmarks[13][1:]
                    lwrist = landmarks[15][1:]

                    rhip = landmarks[24][1:]
                    rknee = landmarks[26][1:]
                    rankle = landmarks[28][1:]

                    lhip = landmarks[23][1:]
                    lknee = landmarks[25][1:]
                    lankle = landmarks[27][1:]

                    # angle_right_elbow = round(calculate_angle(rshoulder, relbow, rwrist))
                    # angle_left_elbow = round(calculate_angle(lshoulder, lelbow, lwrist))
                    #
                    # angle_right_knee = round(calculate_angle(rhip, rknee, rankle))
                    # angle_left_knee = round(calculate_angle(lhip, lknee, lankle))
                    #
                    # angle_right_shoulder = round(calculate_angle(rwrist, rshoulder, rhip))
                    # angle_left_shoulder = round(calculate_angle(lwrist, lshoulder, lhip))

                    # cv2.putText(frame, str(angle_right_elbow), (relbow[0], relbow[1] - 10),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    # cv2.putText(frame, str(angle_left_elbow), (lelbow[0], lelbow[1] - 10),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    # cv2.putText(frame, str(angle_right_knee), (rknee[0], rknee[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    #             0.5, (255, 255, 255), 1)
                    # cv2.putText(frame, str(angle_left_knee), (lknee[0], lknee[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    #             (255, 255, 255), 1)
                    # cv2.putText(frame, str(angle_right_shoulder), (rshoulder[0] + 10, rshoulder[1] + 10),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    # cv2.putText(frame, str(angle_left_shoulder), (lshoulder[0] - 50, lshoulder[1] + 10),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    if option == "Left Curl":
                        angle_left_elbow = round(calculate_angle(lshoulder, lelbow, lwrist))
                        l_angles.append(angle_left_elbow)
                        frames.append(frame_count)
                        per = np.interp(angle_left_elbow, (45, 155), (100, 0))
                        bar = np.interp(angle_left_elbow, (45, 155), (60, 420))
                        drawOn(frame, 11, 13, 15, angle_left_elbow, landmarks)

                        color = (255, 0, 255)
                        if per == 100:
                            color = (0, 255, 0)
                            if direction == 0:
                                direction = 1
                                counter += 1
                            else:
                                suggestion="Lift Arm Down"

                        if per == 0:
                            if direction == 1:
                                direction = 0
                            else:
                                suggestion="Lift Arm Up"


                        cv2.putText(frame, f"stage: {str(suggestion)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                                    cv2.LINE_AA)


                        cv2.rectangle(frame, (520, 60), (590, 420), color, 2)
                        cv2.rectangle(frame, (520, int(bar)), (590, 420), color, cv2.FILLED)
                        cv2.putText(frame, f'{int(per)} %', (520, 50), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)



                    if option == "Curls":
                        angle_right_elbow = round(calculate_angle(rshoulder, relbow, rwrist))
                        angle_left_elbow = round(calculate_angle(lshoulder, lelbow, lwrist))
                        drawOn(frame, 11, 13, 15, angle_left_elbow, landmarks)
                        l_angles.append(angle_left_elbow)
                        r_angles.append(angle_right_elbow)
                        frames.append(frame_count)
                        drawOn(frame, 12, 14, 16, angle_right_elbow, landmarks)
                        color = (255, 0, 255)
                        if angle_left_elbow >= 165 and angle_right_elbow >= 165:
                            stage = "open"
                            suggestion = "Keep elbows close."
                        elif angle_left_elbow <= 34 and angle_right_elbow <= 34 and stage == "open":
                            stage = "close"
                            counter += 1
                            suggestion = "Make sure you are lifting the weights fully"

                        # visualizing angle_left_elbow
                        per1 = np.interp(angle_left_elbow, (34, 165), (100, 0))
                        bar1 = np.interp(angle_left_elbow, (34, 165), (60, 420))
                        per2 = np.interp(angle_right_elbow, (34, 165), (100, 0))
                        bar2 = np.interp(angle_right_elbow, (34, 165), (60, 420))

                        if per1 == 100 and per2==100:
                            color=(0,255,0)

                        cv2.rectangle(frame, (520, 60), (590, 420), color, 2)
                        cv2.rectangle(frame, (520, int(bar1)), (590, 420), color, cv2.FILLED)
                        cv2.putText(frame, f'{int(per1)} %', (520, 50), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
                        cv2.rectangle(frame, (20, 60), (90, 420), color, 2)
                        cv2.rectangle(frame, (20, int(bar2)), (90, 420), color, cv2.FILLED)
                        cv2.putText(frame, f'{int(per2)} %', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)


                        cv2.putText(frame, f"stage: {str(suggestion)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2, cv2.LINE_AA)

                    if option == "Lateral Curls":
                        angle_right_shoulder = round(calculate_angle(rwrist, rshoulder, rhip))
                        angle_left_shoulder = round(calculate_angle(lwrist, lshoulder, lhip))
                        l_angles.append(angle_left_shoulder)
                        r_angles.append(angle_right_shoulder)
                        frames.append(frame_count)
                        drawOn(frame, 15, 11, 23, angle_left_shoulder, landmarks)
                        drawOn(frame, 16, 12, 24, angle_right_shoulder, landmarks)
                        color = (255, 0, 255)
                        if angle_right_shoulder >= 100 and angle_left_shoulder >= 100 and stage =="down":
                            stage = "up"
                            counter += 1
                            suggestion = "Down"
                        elif 30 >= angle_right_shoulder and 30 >= angle_left_shoulder:
                            stage = "down"
                            suggestion = "Up"



                        # visualizing angle_left_shoulder
                        per1 = np.interp(angle_left_shoulder, (30, 100), (0, 100))
                        bar1 = np.interp(angle_left_shoulder, (30, 100), (420, 60))
                        per2 = np.interp(angle_right_shoulder, (30, 100), (0, 100))
                        bar2 = np.interp(angle_right_shoulder, (30, 100), (420, 60))

                        if per1 == 100 and per2==100:
                            color=(0,255,0)

                        cv2.rectangle(frame, (520, 60), (590, 420), color, 2)
                        cv2.rectangle(frame, (520, int(bar1)), (590, 420), color, cv2.FILLED)
                        cv2.putText(frame, f'{int(per1)} %', (520, 50), cv2.FONT_HERSHEY_PLAIN, 2, color,2)
                        cv2.rectangle(frame, (20, 60), (90, 420), color, 2)
                        cv2.rectangle(frame, (20, int(bar2)), (90, 420), color, cv2.FILLED)
                        cv2.putText(frame, f'{int(per2)} %', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)



                        cv2.putText(frame, f"stage: {str(suggestion)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2, cv2.LINE_AA)



                    if option == "Squats":
                        angle_right_knee = round(calculate_angle(rhip, rknee, rankle))
                        angle_left_knee = round(calculate_angle(lhip, lknee, lankle))
                        l_angles.append(angle_left_knee)
                        r_angles.append(angle_right_knee)
                        frames.append(frame_count)
                        drawOn(frame, 24, 26, 28, angle_right_knee, landmarks)
                        drawOn(frame, 23, 25, 27, angle_left_knee, landmarks)
                        color = (255, 0, 255)
                        if angle_right_knee >= 173 and angle_left_knee >= 173:
                            stage = "down"
                            suggestion = "Go Down"
                        elif angle_right_knee <= 90 and angle_left_knee <= 90 and stage == 'down':
                            stage = "up"
                            counter += 1
                            suggestion = "Go Up"
                        cv2.putText(frame, f"stage: {str(suggestion)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2, cv2.LINE_AA)

                        # visualizing angle_left_knee
                        per1 = np.interp(angle_left_knee, (90, 173), (100, 0))
                        bar1 = np.interp(angle_left_knee, (90, 173), (60, 420))


                        # visualizing angle_right_knee
                        per2 = np.interp(angle_right_knee, (90, 173), (100, 0))
                        bar2 = np.interp(angle_right_knee, (90, 173), (60, 420))

                        if per1 == 100 and per2==100:
                            color=(0,255,0)

                        cv2.rectangle(frame, (520, 60), (590, 420), color, 2)
                        cv2.rectangle(frame, (520, int(bar1)), (590, 420), color, cv2.FILLED)
                        cv2.putText(frame, f'{int(per1)} %', (520, 50), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
                        cv2.rectangle(frame, (20, 60), (90, 420), color, 2)
                        cv2.rectangle(frame, (20, int(bar2)), (90, 420), color, cv2.FILLED)
                        cv2.putText(frame, f'{int(per2)} %', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

                    if option == "Push Ups":
                        angle_left_elbow = round(calculate_angle(lshoulder, lelbow, lwrist))
                        drawOn(frame, 11, 13, 15, angle_left_elbow, landmarks)
                        color = (255, 0, 255)
                        l_angles.append(angle_left_elbow)
                        frames.append(frame_count)
                        per = np.interp(angle_left_elbow, (70, 140), (100, 0))
                        bar = np.interp(angle_left_elbow, (70, 140), (60, 420))
                        if per == 100:
                            color = (0, 255, 0)
                            if dir == 0:
                                counter += 1
                                dir = 1
                            suggestion="Up"
                        if per == 0:
                            if dir == 1:
                                dir = 0
                            suggestion="Down"
                        cv2.putText(frame, f"stage: {str(suggestion)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.rectangle(frame, (520, 60), (590, 420), color, 2)
                        cv2.rectangle(frame, (520, int(bar)), (590, 420), color, cv2.FILLED)
                        cv2.putText(frame, f'{int(per)} %', (520, 50), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

                    if option == "Shoulder Press":
                        angle_right_elbow = round(calculate_angle(rshoulder, relbow, rwrist))
                        angle_left_elbow = round(calculate_angle(lshoulder, lelbow, lwrist))
                        l_angles.append(angle_left_elbow)
                        r_angles.append(angle_right_elbow)
                        frames.append(frame_count)
                        drawOn(frame, 11, 13, 15, angle_left_elbow, landmarks)
                        drawOn(frame, 12, 14, 16, angle_right_elbow, landmarks)
                        color=(255,0,255)
                        if angle_left_elbow > 170 and angle_right_elbow > 170:
                            stage = "Up"
                        if angle_left_elbow < 90 and angle_right_elbow < 90 and stage == 'Up':
                            stage = "Down"
                            counter += 1

                        cv2.putText(frame, f"stage: {str(suggestion)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2, cv2.LINE_AA)
                        # visualizing angle_left_elbow
                        per1 = np.interp(angle_left_elbow, (90, 170), (0, 100))
                        bar1 = np.interp(angle_left_elbow, (90, 170), (420, 60))
                        per2 = np.interp(angle_right_elbow, (90, 170), (0, 100))
                        bar2 = np.interp(angle_right_elbow, (90, 170), (420, 60))
                        if per1==100 and per2==100:
                            color=(0,255,0)
                        cv2.rectangle(frame, (520, 60), (590, 420), color, 2)
                        cv2.rectangle(frame, (520, int(bar1)), (590, 420), color, cv2.FILLED)
                        cv2.putText(frame, f'{int(per1)} %', (520, 50), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

                        # visua;izing angle_right_elbow

                        cv2.rectangle(frame, (20, 60), (90, 420), color, 2)
                        cv2.rectangle(frame, (20, int(bar2)), (90, 420), color, cv2.FILLED)
                        cv2.putText(frame, f'{int(per2)} %', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

                    if option == "High Knees":
                        angle_left_hip = round(calculate_angle(lshoulder, lhip, lknee))
                        drawOn(frame, 11, 23, 25, angle_left_hip, landmarks)
                        if angle_left_hip > 160:
                            stage = "Down"
                        if angle_left_hip < 80 and stage == 'Down':
                            stage = "Up"
                            counter += 1
                        cv2.putText(frame, f"stage: {str(suggestion)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2, cv2.LINE_AA)
                        per = np.interp(angle_left_hip, (80, 160), (0, 100))
                        bar = np.interp(angle_left_hip, (80, 160), (420, 60))
                        cv2.rectangle(frame, (520, 60), (590, 420), color, 2)
                        cv2.rectangle(frame, (520, int(bar)), (590, 420), color, cv2.FILLED)
                        cv2.putText(frame, f'{int(per)} %', (520, 50), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

                    if option == "Knee Bend":
                        angle_left_knee = round(calculate_angle(lhip, lknee, lankle))
                        drawOn(frame, 23, 25, 27, angle_left_knee, landmarks)
                        if angle_left_knee > 140:
                            relax_counter += 1
                            bent_counter = 0
                            stage = "Relaxed"
                            suggestion = ""

                        if angle_left_knee < 140:
                            relax_counter = 0
                            bent_counter += 1
                            stage = "Bent"
                            suggestion = ""

                            # rep
                        if bent_counter == 8:
                            counter += 1
                            suggestion = 'Rep completed. Relax knee'

                        elif bent_counter < 8 and stage == 'Bent':
                            suggestion = 'Keep Your Knee Bent'

                        else:
                            suggestion = " "

                        cv2.putText(frame, f"stage: {str(suggestion)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2, cv2.LINE_AA)

                    if option == "Sit Up":
                        shoulder_avg = [(rshoulder[0] + lshoulder[0]) / 2, (rshoulder[1] + lshoulder[1]) / 2]
                        hip_avg = [(rhip[0] + lhip[0]) / 2, (rhip[1] + lhip[1]) / 2]
                        knee_avg = [(rknee[0] + lknee[0]) / 2, (rknee[1] + lknee[1]) / 2]
                        angle = calculate_angle(shoulder_avg, hip_avg, knee_avg)
                        drawOn(frame, 11, 23, 25, angle_left_knee, landmarks)

                        if angle <= 84 and status=="Up":
                            counter += 1
                            status = "Down"
                            suggestion = "Down"

                        if angle >= 90:
                            status = "Up"
                            suggestion = "Up"

                        cv2.putText(frame, f"stage: {str(suggestion)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2, cv2.LINE_AA)
                        per = np.interp(angle, (84, 90), (0, 100))
                        bar = np.interp(angle, (84, 90), (420, 60))
                        cv2.rectangle(frame, (520, 60), (590, 420), color, 2)
                        cv2.rectangle(frame, (520, int(bar)), (590, 420), color, cv2.FILLED)
                        cv2.putText(frame, f'{int(per)} %', (520, 50), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

                    if option == "Crunches":
                        x1 = landmarks[0][1]
                        x2 = landmarks[12][1]

                        length = x1 - x2
                        print(length)
                        if length >= 0 and f == 0:
                            f = 1
                            stage = "Bend Forward"
                            suggestion=stage
                        elif length < 0 and f == 1:
                            f = 0
                            stage = "Relax"
                            suggestion=stage
                            counter += 1
                        drawOn(frame, 0, 12, 24, length, landmarks)

                        cv2.putText(frame, f"stage: {str(stage)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 2, cv2.LINE_AA)


                    if option == "Plank":
                        angle1 = round(calculate_angle(lshoulder, lelbow, lwrist))
                        angle2 = round(calculate_angle(lshoulder, lhip, lknee))
                        angle3 = angle_left_knee = round(calculate_angle(lhip, lknee, lankle))
                        drawOn(frame, 11, 13, 15, angle1, landmarks)
                        drawOn(frame, 11, 23, 25, angle2, landmarks)
                        drawOn(frame, 23, 25, 27, angle3, landmarks)
                        if not 75 <= angle1 <= 105:
                            suggestion1='Bring your shoulder vertically above your elbow'
                            cv2.putText(frame, f"{str(suggestion1)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 0, 255), 2, cv2.LINE_AA)
                        elif angle2 < 140:
                            suggestion2='Make your back straight. Bring your buttocks DOWN'
                            cv2.putText(frame, f"{str(suggestion2)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 0, 255), 2, cv2.LINE_AA)
                        elif angle2 > 170:
                            suggestion3='Make your back straight. Bring your buttocks UP'
                            cv2.putText(frame, f"{str(suggestion3)}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 0, 255), 2, cv2.LINE_AA)
                        elif angle3 <= 160:
                            suggestion4='Do not bend your knee. Stretch your legs'
                            cv2.putText(frame, f"{str(suggestion4)}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 0, 255), 2, cv2.LINE_AA)
                        else:
                            counter+=1
                            time.sleep(1)

                    if option == "Half Plough Pose":
                        # Calculate angles
                        angle_left_hip = round(calculate_angle(lshoulder, lhip, lknee))
                        drawOn(frame, 11, 23, 25, angle_left_hip, landmarks)

                        print(lhip[1], lknee[1])
                        # Determine if Half Plough Pose is being performed
                        if angle_left_hip<83:
                            suggestion1="Half Plough Pose detected!"
                            cv2.putText(frame, f"{str(suggestion1)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 0, 255), 2, cv2.LINE_AA)
                            if lhip[1] > lknee[1]:
                                suggestion2="Your form looks good."
                                cv2.putText(frame, f"{str(suggestion2)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 0, 255), 2, cv2.LINE_AA)
                                counter += 1
                                time.sleep(1)
                            else:
                                suggestion3="Your hips should be below your knees for proper form."
                                cv2.putText(frame, f"{str(suggestion3)}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 0, 255), 2, cv2.LINE_AA)
                        else:
                            suggestion4="Not Half Plough Pose."
                            cv2.putText(frame, f"{str(suggestion4)}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 0, 255), 2, cv2.LINE_AA)


                    if counter == int(press_time):
                        completed_exercise = option
                        completed_message = message

                        # Save the completed workout to the CSV history file
                        history_file = os.path.join(
                            os.path.dirname(os.path.abspath(__file__)),
                            "workout_history.csv"
                        )
                        history_entry = pd.DataFrame([{
                            "Date": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "Name": st.session_state.user_name,
                            "Email": st.session_state.user_email,
                            "Exercise": completed_exercise,
                            "Summary": completed_message
                        }])

                        try:
                            if os.path.exists(history_file):
                                old_history = pd.read_csv(history_file)
                                old_history = pd.concat(
                                    [old_history, history_entry],
                                    ignore_index=True
                                )
                            else:
                                old_history = history_entry

                            old_history.to_csv(history_file, index=False)
                            st.success("Workout saved to history successfully!")
                        except Exception as e:
                            st.error(f"Could not save workout history: {e}")

                        suggestion = "Completed" + completed_message

                        if completed_exercise == "Left Curl":
                            single_plot(frames, l_angles, "Left Elbow", 45, 155)
                        if completed_exercise == "Curls":
                            double_plot(frames, l_angles, r_angles, "Left Elbow", "Right Elbow", 34, 165)
                        if completed_exercise == "Lateral Curls":
                            double_plot(frames, l_angles, r_angles, "Left Shoulder", "Right Shoulder", 30, 85)
                        if completed_exercise == "Squats":
                            double_plot(frames, l_angles, r_angles, "Left Knee", "Right Knee", 90, 173)
                        if completed_exercise == "Shoulder Press":
                            double_plot(frames, l_angles, r_angles, "Left Knee", "Right Knee", 90, 170)
                        if completed_exercise == "Push Ups":
                            single_plot(frames, l_angles, "Left Elbow", 70, 140)

                        if flag:
                            try:
                                email_user(email, name, completed_message)
                                flag = 0
                            except Exception as e:
                                st.warning(f"Workout saved, but email could not be sent: {e}")

                        counter = 0
                        press_time = 0
                        option = False
                        run = False
                        frame_count = 0
                        frames = []
                        l_angles = []
                        r_angles = []
                        cap.release()
                        break

                except:
                    pass

                mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                currTime = time.time()
                fps = 1 / max(currTime - prevTime, 1e-6)
                prevTime = currTime

                kpi1_text.markdown(f"<h2 style='text-align:center;color:#8B6BCB;'>{int(fps)}</h2>", unsafe_allow_html=True)
                kpi2_text.markdown(f"<h2 style='text-align:center;color:#8B6BCB;'>{counter}</h2>", unsafe_allow_html=True)
                kpi3_text.markdown(f"<h2 style='text-align:center;color:#8B6BCB;'>{press_time}</h2>", unsafe_allow_html=True)
                sugg.markdown(f"<div class='info-card' style='min-height:unset;color:#8B6BCB;font-size:18px;font-weight:700;'>{suggestion}</div>", unsafe_allow_html=True)
                # Keep RGB channel order consistent for correct preview colours.
                # Display at native processing size; do not enlarge and waste memory.
                stframe.image(frame, channels='RGB', use_container_width=False, width=320)

        # Always release the webcam after the processing loop exits.
        cap.release()
        run = False


elif app_mode == 'History':
    st.markdown("""
    <div class="page-hero">
      <h1 style="margin:0;color:#8B6BCB !important;">📊 Workout History</h1>
      <p style="margin:8px 0 0 0;color:#514860;">Review your completed sessions and track your fitness activity.</p>
    </div>
    """, unsafe_allow_html=True)
    history_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workout_history.csv")

    if os.path.exists(history_file):
        history_data = pd.read_csv(history_file)
        if "Email" in history_data.columns:
            history_data = history_data[
                history_data["Email"].fillna("").astype(str).str.lower()
                == st.session_state.user_email.lower()
            ]

        if not history_data.empty:
            total_workouts = len(history_data)
            total_reps = 0
            total_calories = 0.0
            if "Summary" in history_data.columns:
                for summary in history_data["Summary"].fillna("").astype(str):
                    reps_match = re.search(r"for\s+(\d+)\s+times", summary, re.IGNORECASE)
                    calories_match = re.search(r"burnt\s+([0-9]+(?:\.[0-9]+)?)\s+calories", summary, re.IGNORECASE)
                    if reps_match:
                        total_reps += int(reps_match.group(1))
                    if calories_match:
                        total_calories += float(calories_match.group(1))

            metric1, metric2, metric3 = st.columns(3, gap="medium")
            metric1.metric("Total Workouts", total_workouts)
            metric2.metric("Total Repetitions", total_reps)
            metric3.metric("Estimated Calories", f"{total_calories:.1f} kcal")

            st.markdown("### 🗂️ Your Workout Records")
            st.dataframe(history_data.reset_index(drop=True), use_container_width=True, hide_index=True)
            st.caption("Showing workout records for your signed-in account.")

            # Generate a compact PDF only when the user requests it.
            try:
                from reportlab.lib import colors
                from reportlab.lib.pagesizes import A4
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
                from reportlab.lib.styles import getSampleStyleSheet
                pdf_buffer = io.BytesIO()
                pdf_doc = SimpleDocTemplate(pdf_buffer, pagesize=A4)
                styles = getSampleStyleSheet()
                story = [
                    Paragraph("AI Fitness Trainer - Workout Report", styles["Title"]),
                    Spacer(1, 12),
                    Paragraph("User: " + str(st.session_state.user_name), styles["Normal"]),
                    Paragraph("Email: " + str(st.session_state.user_email), styles["Normal"]),
                    Spacer(1, 12)
                ]
                pdf_rows = [["Date", "Exercise", "Summary"]]
                for _, record in history_data.tail(40).iterrows():
                    pdf_rows.append([
                        str(record.get("Date", ""))[:19],
                        str(record.get("Exercise", ""))[:24],
                        str(record.get("Summary", ""))[:70]
                    ])
                pdf_table = Table(pdf_rows, repeatRows=1, colWidths=[80, 95, 260])
                pdf_table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F2E8FA")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#514860")),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 7),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4EFFA")])
                ]))
                story.append(pdf_table)
                pdf_doc.build(story)
                st.download_button(
                    "📄 Download Workout Report (PDF)",
                    data=pdf_buffer.getvalue(),
                    file_name="AI_Fitness_Trainer_Report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except ImportError:
                st.warning("PDF library is not installed yet. Run: python -m pip install reportlab")
        else:
            st.info("No workout records available yet. Complete a workout to see your progress here.")
    else:
        st.info("No workout history found. Complete a workout first to create your history.")
