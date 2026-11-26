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
        --bg: #0b1512;
        --panel: #14251f;
        --panel-soft: #1a3028;
        --green: #39D98A;
        --green-light: #8BFFB0;
        --muted: #B8D8C8;
    }

    .stApp {
        background:
            radial-gradient(circle at 85% 8%, rgba(57,217,138,.10), transparent 28%),
            linear-gradient(135deg, #0b1512 0%, #14251f 55%, #101b17 100%);
        color: #F4FFF8;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #14251f 0%, #0d1714 100%);
        border-right: 1px solid rgba(139,255,176,.12);
    }
    [data-testid="stSidebar"] * { color: #F4FFF8; }

    h1, h2, h3 {
        color: var(--green-light) !important;
        font-weight: 750 !important;
        letter-spacing: -.4px;
    }
    h1 { font-size: clamp(1.65rem, 3vw, 2.25rem) !important; }

    p, label, [data-testid="stMarkdownContainer"] { color: #E5F4EB; }

    .stTextInput input, .stSelectbox div[data-baseweb="select"],
    .stNumberInput input {
        background: #202b29;
        border: 1px solid #345347;
        border-radius: 11px;
        color: #FFFFFF;
        min-height: 44px;
    }

    [data-testid="stWidgetLabel"] p {
        color: #D7F2E2 !important;
        font-weight: 650;
    }

    .stButton button {
        background: linear-gradient(90deg, #39D98A, #78F0A5);
        color: #092016;
        border: 0;
        border-radius: 12px;
        font-weight: 750;
        min-height: 44px;
        padding: .55rem 1.3rem;
        transition: all .2s ease;
    }
    .stButton button:hover {
        background: #8BFFB0;
        color: #092016;
        border: 0;
        transform: translateY(-1px);
    }

    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #1b382d, #142820);
        padding: 18px;
        border-radius: 16px;
        border: 1px solid rgba(57,217,138,.35);
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(20,37,31,.62);
        border: 1px solid rgba(139,255,176,.16);
        border-radius: 16px;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid rgba(139,255,176,.2);
        border-radius: 12px;
        overflow: hidden;
    }

    /* Tutorial and live-workout presentation */
    [data-testid="stVideo"] {
        background: #10221b;
        border: 1px solid rgba(139,255,176,.22);
        border-radius: 16px;
        padding: 8px;
        overflow: hidden;
        margin: 8px 0 28px 0;
    }
    [data-testid="stImage"] img {
        border: 1px solid rgba(139,255,176,.28);
        border-radius: 16px;
    }
    .tutorial-link {
        display: inline-block;
        background: linear-gradient(90deg,#16794d,#239b65);
        color: #ffffff !important;
        padding: 9px 15px;
        border-radius: 9px;
        text-decoration: none !important;
        font-weight: 700;
        margin: 0 0 8px 0;
    }
    .page-hero {
        background: linear-gradient(120deg,#174d38,#102820);
        border: 1px solid rgba(139,255,176,.2);
        border-left: 5px solid #39D98A;
        border-radius: 17px;
        padding: 22px 25px;
        margin: 4px 0 22px 0;
    }
    .info-card {
        background: linear-gradient(145deg,rgba(26,48,40,.96),rgba(17,35,28,.96));
        border: 1px solid rgba(139,255,176,.2);
        border-radius: 14px;
        padding: 16px 18px;
        min-height: 90px;
        margin-bottom: 12px;
    }
    .step-number {
        display:inline-flex;
        align-items:center;
        justify-content:center;
        width:30px;
        height:30px;
        border-radius:50%;
        background:#39D98A;
        color:#092016;
        font-weight:800;
        margin-right:10px;
    }
    hr { border-color: rgba(139,255,176,.18); }

    [data-testid="stCheckbox"] label {
        background: rgba(57,217,138,.10);
        border: 1px solid rgba(57,217,138,.35);
        border-radius: 12px;
        padding: 10px 14px;
        font-weight: 700;
    }

    a { color: #8BFFB0 !important; }

    @media (max-width: 768px) {
        .block-container { padding-left: 1rem; padding-right: 1rem; }
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
      <h1 style="margin:0;color:#8BFFB0 !important;">AI Fitness Trainer</h1>
      <p style="margin:8px 0 0 0;color:#E5F4EB;">Sign in to continue your personal fitness journey.</p>
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
                    st.session_state.user_name = signup_name.strip()
                    st.session_state.user_email = clean_email
                    st.success("Account created successfully!")
                    st.rerun()
                except sqlite3.IntegrityError:
                    st.error("An account with this email already exists. Please sign in.")
                except Exception as exc:
                    st.error(f"Could not create account: {exc}")
    st.stop()

st.sidebar.markdown("""
<div style="padding:12px 2px 18px 2px;">
  <div style="font-size:25px;font-weight:800;color:#8BFFB0;line-height:1.2;">
    AI Fitness Trainer
  </div>
  <div style="font-size:13px;color:#B8D8C8;margin-top:8px;">
    Powered by MediaPipe &amp; OpenCV
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
        background: linear-gradient(120deg, #174D38, #102820);
        padding: 30px;
        border-radius: 18px;
        border-left: 6px solid #39D98A;
        margin-bottom: 25px;">
        <h1 style="color:#7CFC9A; margin:0;">
            Welcome to AI Fitness Trainer
        </h1>
        <p style="color:white; font-size:18px;">
            Train Smart. Move Better. Stay Fit.
        </p>
        <p style="color:#C5E8D5;">
            Your AI-powered workout companion using MediaPipe and OpenCV.
        </p>
    </div>
    """, unsafe_allow_html=True)


if app_mode == 'Progress Dashboard':
    st.markdown("""
    <div class="page-hero">
      <h1 style="margin:0;color:#8BFFB0 !important;">📈 Weekly Progress Dashboard</h1>
      <p style="margin:8px 0 0 0;color:#E5F4EB;">A simple view of your saved workout activity.</p>
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
      <h1 style="margin:0;color:#8BFFB0 !important;">🎯 Goals & Achievements</h1>
      <p style="margin:8px 0 0 0;color:#E5F4EB;">Set personal targets and earn badges from saved workouts.</p>
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
      <h1 style="margin:0;color:#8BFFB0 !important;">⏱️ Workout Rest Timer</h1>
      <p style="margin:8px 0 0 0;color:#E5F4EB;">Take a short break between exercise sets.</p>
    </div>
    """, unsafe_allow_html=True)
    rest_seconds = st.select_slider("Choose rest duration (seconds)",
                                    options=[15, 30, 45, 60, 90, 120], value=30)
    if st.button("Start Rest Timer", use_container_width=True):
        timer_display = st.empty()
        progress_display = st.progress(0)
        for remaining in range(int(rest_seconds), 0, -1):
            timer_display.markdown(
                f"<h1 style='text-align:center;color:#8BFFB0;font-size:56px'>{remaining}s</h1>",
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
      <h1 style="margin:0;color:#8BFFB0 !important;">About AI Fitness Trainer</h1>
      <p style="margin:8px 0 0 0;color:#E5F4EB;">An AI-powered exercise guidance application built with MediaPipe, OpenCV and Streamlit.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
      <h3 style="margin:0 0 8px 0;">🤖 About the Application</h3>
      <p style="margin:0;color:#D7EDE1;">MediaPipe is used to detect body pose landmarks, OpenCV reads and processes webcam frames, and Streamlit provides the web-based graphical user interface.</p>
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
        st.markdown('<div class="info-card" style="display:flex;align-items:center;padding:13px 16px;min-height:unset;"><span class="step-number">' + str(idx) + '</span><span style="color:#E5F4EB;">' + step + '</span></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card" style="border-left:4px solid #E6B84A;margin-top:18px;">
      <h3 style="margin:0 0 10px 0;">⚠️ Before You Start</h3>
      <ul style="margin:0;color:#E5F4EB;padding-left:22px;">
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
      <h1 style="margin:0;color:#8BFFB0 !important;">🎬 Exercise Demos &amp; Tutorials</h1>
      <p style="margin:8px 0 0 0;color:#E5F4EB;">Choose an exercise, watch its demonstration and follow the linked tutorial before starting your workout.</p>
    </div>
    """, unsafe_allow_html=True)
    #HIGH KNESS
    st.title("HIGH KNEES")
    link_html = '<a href="https://youtu.be/oDdkytliOqE?si=Mferh8hYmi14Rh7m" style="background-color: rgb(0, 128, 131); color: #fff; text-decoration:none; padding: 5px;">Video tutorial &rarr;</a>'
    st.markdown(link_html, unsafe_allow_html=True)
    hk_video_url = "https://www.shutterstock.com/shutterstock/videos/1060403879/preview/stock-footage-athletic-woman-doing-high-knee-exercise-at-home-home-training-workout-home-fitness-there-is-some.webm"
    st.video(hk_video_url)
    
    #SQUATS
    st.title("SQUATS")
    link_html = '<a href="https://youtu.be/4KmY44Xsg2w?si=eh8-DmSgjZptopSs" style="background-color: rgb(0, 128, 131); color: #fff; text-decoration:none; padding: 5px;">Video tutorial &rarr;</a>'
    st.markdown(link_html, unsafe_allow_html=True)
    s_video_url = "https://www.shutterstock.com/shutterstock/videos/1102020407/preview/stock-footage-female-butt-workout-squats-at-home-athletic-asian-woman-squats-workout-in-living-room-female.webm"
    st.video(s_video_url)
    
    #SHOULDER PRESS
    st.title("SHOULDER PRESS")
    link_html = '<a href="https://youtu.be/xe19t2_6yis?si=SIpCkA04UFBoJfs_" style="background-color: rgb(0, 128, 131); color: #fff; text-decoration:none; padding: 5px;">Video tutorial &rarr;</a>'
    st.markdown(link_html, unsafe_allow_html=True)
    shoulder_video_url = "https://ak.picdn.net/shutterstock/videos/1104019445/preview/stock-footage-medium-shot-of-young-sportsman-performing-barbell-shoulder-press-in-gym-man-holds-barbell-on-the.mp4"
    st.video(shoulder_video_url)
    
    #LATERAL CURLS
    st.title("LATERAL CURLS")
    link_html = '<a href="https://youtu.be/PzsMitRdI_8?si=Xt_bteImzFf_vP7m" style="background-color: rgb(0, 128, 131); color: #fff; text-decoration:none; padding: 5px;">Video tutorial &rarr;</a>'
    st.markdown(link_html, unsafe_allow_html=True)
    lcurls_video_url = "https://www.shutterstock.com/shutterstock/videos/1012307186/preview/stock-footage-bodybuilder-performs-seated-side-lateral-raise-reflection-of-exercise-in-mirror.webm"
    st.video(lcurls_video_url)
    #CURLS
    st.title("CURLS")
    link_html = '<a href="https://youtu.be/sYV-ki-1blM?si=Oi9cMYUAKZnPggP_" style="background-color: rgb(0, 128, 131); color: #fff; text-decoration:none; padding: 5px;">Video tutorial &rarr;</a>'
    st.markdown(link_html, unsafe_allow_html=True)
    curls_video_url = "https://www.shutterstock.com/shutterstock/videos/1097465161/preview/stock-footage-athlete-exercising-with-dumbbells-in-indoor-gym-fit-man-doing-biceps-triceps-curls-workout-in-gym.webm"
    st.video(curls_video_url)
    
    
    
    
    #LEFT CURLS
    st.title("Left curls")
    link_html = '<a href="https://www.youtube.com/shorts/cHxRJdSVIkA?feature=share" style="background-color: rgb(0, 128, 131); color: #fff; text-decoration:none; padding: 5px;">Video tutorial &rarr;</a>'
    st.markdown(link_html, unsafe_allow_html=True)
    lecurls_video_url ="https://www.shutterstock.com/shutterstock/videos/1030360235/preview/stock-footage-fitness-sport-weightlifting-and-bodybuilding-concept-man-exercising-with-dumbbells-at-home.webm"
    st.video(lecurls_video_url)
    
    
    #PUSHUPS
    st.title("PUSH UPS")
    link_html = '<a href="https://youtu.be/_UBOxUl0Sl4" style="background-color: rgb(0, 128, 131); color: #fff; text-decoration:none; padding: 5px;">Video tutorial &rarr;</a>'
    st.markdown(link_html, unsafe_allow_html=True)
    pushups_video_url = "https://www.shutterstock.com/shutterstock/videos/1100041747/preview/stock-footage-calisthenics-outdoor-bodyweight-workout-doing-push-ups-in-sunny-day.webm"
    st.video(pushups_video_url)
    
    #PLANK
    st.title("PLANK")
    link_html = '<a href="https://youtu.be/pvIjsG5Svck?si=ry3kol3TxHF7hu3g" style="background-color: rgb(0, 128, 131); color: #fff; text-decoration:none; padding: 5px;">Video tutorial &rarr;</a>'
    st.markdown(link_html, unsafe_allow_html=True)
    p_video_url = "https://www.shutterstock.com/shutterstock/videos/1063989619/preview/stock-footage-fitness-indian-woman-doing-plank-exercise-workout-in-gym-indoors.webm"
    st.video(p_video_url)
    
    #SIT UPS
    st.title("SIT UPS")
    link_html = '<a href="https://youtu.be/1fbU_MkV7NE?si=bcHWl9xa7PkFSsAu" style="background-color: rgb(0, 128, 131); color: #fff; text-decoration:none; padding: 5px;">Video tutorial &rarr;</a>'
    st.markdown(link_html, unsafe_allow_html=True)
    situps_video_url = "https://cdn.coverr.co/videos/coverr-ab-workout-in-the-park-7546/1080p.mp4"
    st.video(situps_video_url)
    
    
    #KNEE BEND
    st.title("KNEE BEND")
    link_html = '<a href="https://youtu.be/W4LuS9rK0gU?si=A5dTYz_OHyQmvHHd" style="background-color: rgb(0, 128, 131); color: #fff; text-decoration:none; padding: 5px;">Video tutorial &rarr;</a>'
    st.markdown(link_html, unsafe_allow_html=True)
    kb_video_url ="https://www.shutterstock.com/shutterstock/videos/1061478490/preview/stock-footage-caucasian-woman-spending-time-at-home-in-living-room-exercising-with-dumbbells-doing-squats-in.webm"
    st.video(kb_video_url)
    
    #CRUNCHES
    st.title("CRUNCHES")
    link_html = '<a href="https://youtu.be/Xyd_fa5zoEU?si=uK846BgYpam0H4X3" style="background-color: rgb(0, 128, 131); color: #fff; text-decoration:none; padding: 5px;">Video tutorial &rarr;</a>'
    st.markdown(link_html, unsafe_allow_html=True)
    cru_video_url = "https://www.shutterstock.com/shutterstock/videos/1041475927/preview/stock-footage-long-haired-girl-doing-bicycle-crunches-exercise-working-out-at-empty-studio-in-front-the-windows.webm"
    st.video(cru_video_url)
    
    #HALF PLOUGH POSE
    st.title("HALF PLOUGH POSE")
    link_html = '<a href="https://youtu.be/uqKbLaXSfWE" style="background-color: rgb(0, 128, 131); color: #fff; text-decoration:none; padding: 5px;">Video tutorial &rarr;</a>'
    st.markdown(link_html, unsafe_allow_html=True)
    hp_video_url = "https://www.shutterstock.com/shutterstock/videos/1105807867/preview/stock-footage-video-of-woman-performing-alternate-leg-ardha-halasana-this-exercise-strengthen-the-muscles-of.webm"
    st.video(hp_video_url)
    
    
    
elif app_mode == 'Training':
    st.markdown("""
    <div class="section-heading">
        <h2 style="margin:0 0 6px 0;">🏋️ Start Your Workout</h2>
        <p style="color:#B8D8C8; font-size:16px; margin:0;">
            Set up your session and let your AI trainer guide your movement.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="
        background:linear-gradient(145deg, rgba(26,48,40,.95), rgba(17,35,28,.95));
        border:1px solid rgba(139,255,176,.18);
        border-radius:18px;
        padding:22px 24px 10px 24px;
        margin:16px 0 22px 0;">
        <div style="color:#8BFFB0;font-size:12px;font-weight:800;letter-spacing:1.5px;">
            PERSONAL WORKOUT SETUP
        </div>
        <div style="color:#F4FFF8;font-size:14px;margin-top:5px;">
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
        background:rgba(57,217,138,.07);
        border-left:4px solid #39D98A;
        padding:12px 16px;
        border-radius:10px;
        color:#CDEBDD;
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

    path=0
    # path="test/left_curl_demo.mp4"
    # path = "test/letral.mp4"
    # path = "test/curls_demo.mp4"
    # path = "test/squats1.mp4"
    # path = "test/shoulderpress_demo.mp4"
    # path = "test/highkness.mp4"
    # path = "test/pushup2_demo.mp4"
    # path = "test/KneeBendVideo.mp4"
    # path = "test/situps.mp4"
    # path = "test/crunches 1.mp4"
    # path = "test/plough.mp4"
    # time.sleep(2)
    while run:
        stframe = st.empty()
        # Request a clearer webcam stream. Actual quality depends on the camera hardware.
        cap = cv2.VideoCapture(path, cv2.CAP_DSHOW) if isinstance(path, int) else cv2.VideoCapture(path)
        if isinstance(path, int):
            # LOW-RESOURCE CAMERA SETTINGS for laptops with 4 GB RAM.
            # Do not request 720p: it can overload older CPUs/webcams.
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 10)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            st.error("Camera could not be opened. Check camera permission and close other apps using the camera.")
            st.stop()

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_input = int(cap.get(cv2.CAP_PROP_FPS))
        st.caption(f"Low-resource mode: camera {width} × {height} | AI processing: 320 × 240 | target 10 FPS")
        fps = 0
        i = 0

        st.markdown("""
        <div class="page-hero" style="padding:16px 20px;margin:10px 0 16px 0;">
          <h2 style="margin:0;color:#8BFFB0 !important;">📹 Live Workout Session</h2>
          <p style="margin:5px 0 0 0;color:#D7F2E2;">Follow the on-screen guidance and keep your posture visible to the camera.</p>
        </div>
        """, unsafe_allow_html=True)
        ltitle, ctitle, rtitle = st.columns(3, gap="medium")

        with ltitle:
            with st.container(border=True):
                st.markdown("**FRAME RATE**")
                kpi1_text = st.empty()
                kpi1_text.markdown("<h2 style='text-align:center;color:#8BFFB0;'>0</h2>", unsafe_allow_html=True)

        with ctitle:
            with st.container(border=True):
                st.markdown("**REPETITIONS**")
                kpi2_text = st.empty()
                kpi2_text.markdown("<h2 style='text-align:center;color:#8BFFB0;'>0</h2>", unsafe_allow_html=True)

        with rtitle:
            with st.container(border=True):
                st.markdown("**TARGET**")
                kpi3_text = st.empty()
                kpi3_text.markdown("<h2 style='text-align:center;color:#8BFFB0;'>0</h2>", unsafe_allow_html=True)

        st.markdown("**💡 Trainer Suggestions**")
        sugg = st.empty()
        sugg.markdown("---")

        st.markdown("<hr/>", unsafe_allow_html=True)

        with mp_pose.Pose(static_image_mode=False,
                          model_complexity=0,
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
                        st.error("Camera stopped responding. Close the workout and check the webcam.")
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
                fps = 1 / (currTime - prevTime)
                prevTime = currTime

                kpi1_text.markdown(f"<h2 style='text-align:center;color:#8BFFB0;'>{int(fps)}</h2>", unsafe_allow_html=True)
                kpi2_text.markdown(f"<h2 style='text-align:center;color:#8BFFB0;'>{counter}</h2>", unsafe_allow_html=True)
                kpi3_text.markdown(f"<h2 style='text-align:center;color:#8BFFB0;'>{press_time}</h2>", unsafe_allow_html=True)
                sugg.markdown(f"<div class='info-card' style='min-height:unset;color:#8BFFB0;font-size:18px;font-weight:700;'>{suggestion}</div>", unsafe_allow_html=True)
                # Keep RGB channel order consistent for correct preview colours.
                # Display at native processing size; do not enlarge and waste memory.
                stframe.image(frame, channels='RGB', use_container_width=False, width=320)

        # Always release the webcam after the processing loop exits.
        cap.release()
        run = False


elif app_mode == 'History':
    st.markdown("""
    <div class="page-hero">
      <h1 style="margin:0;color:#8BFFB0 !important;">📊 Workout History</h1>
      <p style="margin:8px 0 0 0;color:#E5F4EB;">Review your completed sessions and track your fitness activity.</p>
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
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#174D38")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 7),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EAF5EE")])
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

