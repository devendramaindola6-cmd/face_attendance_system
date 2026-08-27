from __future__ import annotations

import base64
import csv
import hmac
import hashlib
import json
import os
import shutil
import time
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from attendance_system import (
    ATTENDANCE_PATH,
    DEFAULT_CONFIDENCE_LIMIT,
    FACES_DIR,
    LABELS_PATH,
    crop_face,
    current_time_ist,
    detect_largest_face,
    draw_box,
    ensure_attendance_file,
    ensure_directories,
    load_face_detector,
    load_model,
    mark_attendance,
    mark_leaving,
    MODEL_PATH,
    next_person_id,
    normalize_name,
    train,
)

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
HERO_IMAGE_PATH = ASSETS_DIR / "aisha-scanner-facerec.png"
ATTENDANCE_COLUMNS = ["date", "time", "person_id", "name", "status", "duration"]
ADMIN_USERNAME_KEY = "ADMIN_USERNAME"
ADMIN_PASSWORD_KEY = "ADMIN_PASSWORD"
RESET_SECONDS = 30


st.set_page_config(
    page_title="Employee Attendance System",
    page_icon=":camera:",
    layout="wide",
)


def apply_theme() -> None:
    st.markdown(
        """
        <style>
            [data-testid="stAppViewContainer"] {
                background:
                    radial-gradient(circle at top left, rgba(18, 184, 134, 0.14), transparent 32rem),
                    radial-gradient(circle at top right, rgba(245, 158, 11, 0.14), transparent 30rem),
                    linear-gradient(135deg, #f8fbfb 0%, #eef8f4 46%, #fff8ed 100%);
                color: #102027;
            }

            [data-testid="stHeader"] {
                background: transparent;
            }

            .block-container {
                max-width: 1180px;
                padding-top: 2rem;
                padding-bottom: 3rem;
            }

            .hero-eyebrow {
                color: #0f766e;
                font-size: 0.78rem;
                font-weight: 800;
                letter-spacing: 0;
                margin-bottom: 0.45rem;
                text-transform: uppercase;
            }

            .hero-title {
                color: #102027;
                font-size: 3rem;
                font-weight: 900;
                line-height: 1.02;
                margin: 0;
            }

            .hero-copy {
                color: #52646c;
                font-size: 1.08rem;
                line-height: 1.65;
                margin-top: 1rem;
                max-width: 38rem;
            }

            .hero-pill-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.65rem;
                margin-top: 1.3rem;
            }

            .hero-pill {
                background: rgba(255, 255, 255, 0.78);
                border: 1px solid rgba(15, 118, 110, 0.18);
                border-radius: 999px;
                color: #17464a;
                font-size: 0.86rem;
                font-weight: 700;
                padding: 0.5rem 0.8rem;
            }

            .scanner-frame {
                border: 1px solid rgba(15, 118, 110, 0.2);
                border-radius: 8px;
                box-shadow: 0 24px 54px rgba(16, 32, 39, 0.16);
                overflow: hidden;
                position: relative;
            }

            .scanner-frame img {
                border-radius: 0;
                display: block;
                width: 100%;
            }

            .scanner-line {
                animation: scanLine 3.2s ease-in-out infinite;
                background: linear-gradient(
                    90deg,
                    transparent,
                    rgba(132, 255, 214, 0.28),
                    rgba(132, 255, 214, 0.96),
                    rgba(132, 255, 214, 0.28),
                    transparent
                );
                box-shadow: 0 0 22px rgba(45, 212, 191, 0.85);
                height: 3px;
                left: 10%;
                position: absolute;
                right: 21%;
                top: 13%;
                z-index: 2;
            }

            @keyframes scanLine {
                0%, 100% {
                    top: 13%;
                    opacity: 0.65;
                }
                50% {
                    top: 77%;
                    opacity: 1;
                }
            }

            [data-testid="stMetric"] {
                background: rgba(255, 255, 255, 0.82);
                border: 1px solid rgba(16, 32, 39, 0.08);
                border-radius: 8px;
                box-shadow: 0 14px 35px rgba(16, 32, 39, 0.08);
                padding: 1rem;
            }

            [data-testid="stMetricLabel"] {
                color: #52646c;
                font-weight: 700;
            }

            [data-testid="stMetricValue"] {
                color: #102027;
                font-weight: 900;
            }

            .stTabs [data-baseweb="tab-list"] {
                background: rgba(255, 255, 255, 0.74);
                border: 1px solid rgba(16, 32, 39, 0.08);
                border-radius: 8px;
                gap: 0.25rem;
                padding: 0.35rem;
            }

            .stTabs [data-baseweb="tab"] {
                border-radius: 7px;
                color: #52646c;
                font-weight: 800;
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .stTabs [aria-selected="true"] {
                background: #0f766e;
                color: #ffffff;
            }

            .stButton > button {
                border-radius: 8px;
                border: 1px solid rgba(15, 118, 110, 0.28);
                font-weight: 800;
                min-height: 2.8rem;
            }

            .stButton > button[kind="primary"] {
                background: linear-gradient(135deg, #0f766e, #16a34a);
                border: 0;
                color: #ffffff;
                box-shadow: 0 12px 26px rgba(15, 118, 110, 0.24);
            }

            [data-testid="stSidebar"] {
                background:
                    linear-gradient(180deg, #ffffff 0%, #f1fbf7 52%, #fff8ed 100%);
                border-right: 1px solid rgba(16, 32, 39, 0.08);
            }

            .admin-panel {
                background: linear-gradient(135deg, #0f766e, #102027);
                border-radius: 8px;
                box-shadow: 0 16px 34px rgba(16, 32, 39, 0.18);
                color: #ffffff;
                margin-top: 0.8rem;
                padding: 1rem;
            }

            .admin-panel-title {
                font-size: 1rem;
                font-weight: 900;
                margin-bottom: 0.35rem;
            }

            .admin-panel-copy {
                color: rgba(255, 255, 255, 0.78);
                font-size: 0.82rem;
                line-height: 1.45;
                margin: 0;
            }

            .admin-status {
                background: rgba(255, 255, 255, 0.76);
                border: 1px solid rgba(16, 32, 39, 0.08);
                border-radius: 8px;
                color: #52646c;
                font-size: 0.84rem;
                font-weight: 800;
                margin: 0.75rem 0;
                padding: 0.75rem;
            }

            .admin-status strong {
                color: #102027;
                display: block;
                font-size: 0.92rem;
                margin-bottom: 0.1rem;
            }

            .admin-status.signed-in {
                background: rgba(22, 163, 74, 0.12);
                border-color: rgba(22, 163, 74, 0.28);
            }

            .admin-status.warning {
                background: rgba(245, 158, 11, 0.14);
                border-color: rgba(245, 158, 11, 0.28);
            }

            h2, h3 {
                color: #102027;
                font-weight: 900;
            }

            div[data-testid="stDataFrame"] {
                border: 1px solid rgba(16, 32, 39, 0.08);
                border-radius: 8px;
                overflow: hidden;
            }

            img {
                border-radius: 8px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def read_attendance() -> pd.DataFrame:
    ensure_attendance_file()
    return pd.read_csv(ATTENDANCE_PATH)


def write_attendance(attendance: pd.DataFrame) -> None:
    attendance = attendance.reindex(columns=ATTENDANCE_COLUMNS)
    attendance.to_csv(ATTENDANCE_PATH, index=False)


def clear_attendance() -> None:
    write_attendance(pd.DataFrame(columns=ATTENDANCE_COLUMNS))


def schedule_reset(*reset_names: str) -> None:
    deadline = time.time() + RESET_SECONDS
    for reset_name in reset_names:
        st.session_state[f"{reset_name}_reset_at"] = deadline


def apply_pending_resets() -> bool:
    now = time.time()
    did_reset = False

    if st.session_state.get("enrollment_reset_at", float("inf")) <= now:
        st.session_state.enrollment_name_key_version += 1
        st.session_state.enrollment_camera_key_version += 1
        st.session_state.enrollment_upload_key_version += 1
        st.session_state.pop("enrollment_name", None)
        st.session_state.pop("enrollment_dir", None)
        st.session_state.pop("saved_sample_hashes", None)
        st.session_state.pop("enrollment_reset_at", None)
        did_reset = True

    if st.session_state.get("attendance_reset_at", float("inf")) <= now:
        st.session_state.show_attendance_camera = False
        st.session_state.attendance_camera_key_version += 1
        st.session_state.pop("attendance_reset_at", None)
        did_reset = True

    if st.session_state.get("leaving_reset_at", float("inf")) <= now:
        st.session_state.show_leaving_camera = False
        st.session_state.leaving_camera_key_version += 1
        st.session_state.pop("leaving_reset_at", None)
        did_reset = True

    return did_reset


@st.fragment(run_every="1s")
def render_pending_reset_refresh() -> None:
    now = time.time()
    has_pending_reset = any(
        key.endswith("_reset_at") and isinstance(deadline, float) and deadline > now
        for key, deadline in st.session_state.items()
    )
    if not has_pending_reset:
        return

    if apply_pending_resets():
        st.rerun()


def speak_message(message: str, button_label: str) -> None:
    safe_button_id = "speak-" + hashlib.sha256(message.encode("utf-8")).hexdigest()[:12]

    components.html(
        f"""
        <div>
            <button id="{safe_button_id}">
                {button_label}
            </button>
        </div>
        <script>
            const message = {json.dumps(message)};
            const speak = () => {{
                if (!("speechSynthesis" in window)) {{
                    return;
                }}

                window.speechSynthesis.cancel();
                const utterance = new SpeechSynthesisUtterance(message);
                utterance.rate = 0.95;
                utterance.pitch = 1;
                utterance.volume = 1;
                window.speechSynthesis.speak(utterance);
            }};

            const button = document.getElementById("{safe_button_id}");
            button.addEventListener("click", speak);
            setTimeout(speak, 250);
        </script>
        <style>
            body {{
                margin: 0;
                font-family: "Source Sans Pro", sans-serif;
            }}

            #{safe_button_id} {{
                background: linear-gradient(135deg, #0f766e, #16a34a);
                border: 0;
                border-radius: 8px;
                box-shadow: 0 10px 22px rgba(15, 118, 110, 0.24);
                color: #ffffff;
                cursor: pointer;
                font-size: 0.95rem;
                font-weight: 800;
                min-height: 2.6rem;
                padding: 0 1rem;
                width: 100%;
            }}
        </style>
        """,
        height=48,
    )


def speak_attendance(names: list[str]) -> None:
    if not names:
        return

    unique_names = sorted(set(names))
    if len(unique_names) == 1:
        message = f"{unique_names[0]} is present"
    else:
        message = f"{', '.join(unique_names[:-1])}, and {unique_names[-1]} are present"

    speak_message(message, "Play Voice Announcement")


def speak_leaving(names: list[str]) -> None:
    if not names:
        return

    unique_names = sorted(set(names))
    if len(unique_names) == 1:
        message = f"{unique_names[0]} checked out"
    else:
        message = f"{', '.join(unique_names[:-1])}, and {unique_names[-1]} checked out"

    speak_message(message, "Play Checkout Announcement")


def secret_value(key: str) -> str | None:
    try:
        value = st.secrets.get(key)
    except Exception:
        value = None

    if value is None:
        value = os.getenv(key)

    return str(value) if value else None


def admin_credentials() -> tuple[str | None, str | None]:
    return secret_value(ADMIN_USERNAME_KEY), secret_value(ADMIN_PASSWORD_KEY)


def check_admin_credentials(username: str, password: str) -> bool:
    admin_username, admin_password = admin_credentials()
    if not admin_username or not admin_password:
        return False

    return hmac.compare_digest(username, admin_username) and hmac.compare_digest(
        password,
        admin_password,
    )


def sync_confidence_from_train() -> None:
    st.session_state.recognition_confidence = st.session_state.train_recognition_confidence


def show_admin_login() -> bool:
    st.sidebar.divider()
    st.sidebar.markdown(
        """
        <div class="admin-panel">
            <div class="admin-panel-title">Admin Console</div>
            <p class="admin-panel-copy">
                Sign in to manage training, records, exports, and cleanup tools.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.get("is_admin", False):
        st.sidebar.markdown(
            """
            <div class="admin-status signed-in">
                <strong>Access granted</strong>
                Train and Records are unlocked.
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.sidebar.button("Log Out"):
            st.session_state.is_admin = False
            st.rerun()
        return True

    admin_username, admin_password = admin_credentials()
    if not admin_username or not admin_password:
        st.sidebar.markdown(
            """
            <div class="admin-status warning">
                <strong>Login not configured</strong>
                Add admin credentials in Streamlit secrets.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return False

    with st.sidebar.form("admin_login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Admin Login")

    if submitted:
        if check_admin_credentials(username, password):
            st.session_state.is_admin = True
            st.sidebar.success("Admin signed in.")
            st.rerun()
        else:
            st.sidebar.error("Invalid admin credentials.")

    return False


def show_attendance_table() -> None:
    attendance = read_attendance()
    st.subheader("Attendance")
    if st.button("Delete CSV Data", type="secondary"):
        clear_attendance()
        st.success("Attendance CSV data deleted.")
        attendance = read_attendance()

    if attendance.empty:
        st.info("No attendance marked yet.")
        return

    editable_attendance = attendance.reset_index(names="row_id")
    editable_attendance.insert(0, "delete", False)

    edited_attendance = st.data_editor(
        editable_attendance,
        column_config={
            "delete": st.column_config.CheckboxColumn("Delete"),
            "row_id": st.column_config.NumberColumn("Row"),
        },
        disabled=ATTENDANCE_COLUMNS + ["row_id"],
        hide_index=True,
        use_container_width=True,
        key="attendance_row_editor",
    )

    selected_rows = edited_attendance.loc[edited_attendance["delete"], "row_id"].tolist()
    if st.button("Delete Selected Rows", type="secondary"):
        if not selected_rows:
            st.error("Select one or more rows to delete.")
        else:
            updated_attendance = attendance.drop(index=selected_rows).reset_index(drop=True)
            write_attendance(updated_attendance)
            st.success(f"Deleted {len(selected_rows)} selected row(s).")
            st.rerun()

    st.download_button(
        "Download CSV",
        data=ATTENDANCE_PATH.read_bytes(),
        file_name="attendance.csv",
        mime="text/csv",
    )


def enrolled_people() -> pd.DataFrame:
    ensure_directories()
    rows = []
    for person_dir in sorted(FACES_DIR.iterdir()):
        if not person_dir.is_dir():
            continue

        prefix, _, raw_name = person_dir.name.partition("_")
        if not prefix.isdigit() or not raw_name:
            continue

        sample_count = len(list(person_dir.glob("*.png")))
        if sample_count == 0:
            continue

        rows.append(
            {
                "person_id": int(prefix),
                "name": raw_name.replace("_", " "),
                "samples": sample_count,
                "status": "Ready for training",
            }
        )

    return pd.DataFrame(rows)


def read_attendance_rows() -> list[dict[str, str]]:
    ensure_attendance_file()
    with ATTENDANCE_PATH.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def attendance_dates(rows: list[dict[str, str]]) -> list[date]:
    dates: set[date] = set()
    for row in rows:
        try:
            dates.add(date.fromisoformat(row.get("date", "")))
        except ValueError:
            continue

    return sorted(dates)


def add_absent_candidates() -> tuple[int, list[str]]:
    people = enrolled_people()
    if people.empty:
        return 0, []

    attendance_rows = read_attendance_rows()
    marked_by_date: dict[str, set[int]] = {}
    for row in attendance_rows:
        person_id = row.get("person_id", "")
        if person_id.isdigit():
            marked_by_date.setdefault(row.get("date", ""), set()).add(int(person_id))

    pending_dates = attendance_dates(attendance_rows)
    now = current_time_ist()
    absent_dates: set[str] = set()
    absent_count = 0

    with ATTENDANCE_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        for pending_date in pending_dates:
            date_text = pending_date.isoformat()
            marked_ids = marked_by_date.setdefault(date_text, set())

            for row in people.to_dict("records"):
                person_id = int(row["person_id"])
                if person_id in marked_ids:
                    continue

                writer.writerow(
                    [
                        date_text,
                        now.strftime("%H:%M:%S"),
                        person_id,
                        str(row["name"]),
                        "Absent",
                        "",
                    ]
                )
                absent_count += 1
                absent_dates.add(date_text)
                marked_ids.add(person_id)

    return absent_count, sorted(absent_dates)


def delete_trained_model() -> None:
    for path in (MODEL_PATH, LABELS_PATH):
        if path.exists():
            path.unlink()


def delete_training_samples() -> None:
    if FACES_DIR.exists():
        shutil.rmtree(FACES_DIR)
    FACES_DIR.mkdir(parents=True, exist_ok=True)
    st.session_state.pop("enrollment_name", None)
    st.session_state.pop("enrollment_dir", None)
    st.session_state.pop("saved_sample_hashes", None)


def delete_training_people(person_ids: list[int]) -> int:
    deleted_count = 0
    selected_ids = set(person_ids)

    for person_dir in FACES_DIR.iterdir():
        if not person_dir.is_dir():
            continue

        prefix = person_dir.name.split("_", 1)[0]
        if prefix.isdigit() and int(prefix) in selected_ids:
            shutil.rmtree(person_dir)
            deleted_count += 1

    if deleted_count:
        delete_trained_model()
        st.session_state.pop("enrollment_name", None)
        st.session_state.pop("enrollment_dir", None)
        st.session_state.pop("saved_sample_hashes", None)

    return deleted_count


def model_status() -> str:
    return "Ready" if MODEL_PATH.exists() and LABELS_PATH.exists() else "Not trained"


def image_as_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def show_hero(is_admin: bool) -> None:
    people = enrolled_people()
    attendance = read_attendance()
    sample_total = int(people["samples"].sum()) if not people.empty else 0

    left, right = st.columns([1.04, 0.96], vertical_alignment="center")
    with left:
        st.markdown(
            """
            <div class="hero-eyebrow">Workplace Attendance</div>
            <h1 class="hero-title">Employee Attendance System</h1>
            <p class="hero-copy">
                Enroll employees, train the recognizer, and mark attendance from secure
                browser camera snapshots. Built for Streamlit Cloud with IST timestamps.
            </p>
            <div class="hero-pill-row">
                <span class="hero-pill">Cloud ready</span>
                <span class="hero-pill">Browser camera</span>
                <span class="hero-pill">IST records</span>
                <span class="hero-pill">Admin controls</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        if HERO_IMAGE_PATH.exists():
            image_data = image_as_base64(HERO_IMAGE_PATH)
            st.markdown(
                f"""
                <div class="scanner-frame">
                    <img
                        src="data:image/png;base64,{image_data}"
                        alt="Aisha shown inside a face attendance scanner interface"
                    />
                    <div class="scanner-line"></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if is_admin:
        metric_1, metric_2, metric_3, metric_4 = st.columns(4)
        metric_1.metric("Enrolled Employees", len(people))
        metric_2.metric("Face Samples", sample_total)
        metric_3.metric("Attendance Rows", len(attendance))
        metric_4.metric("Model", model_status())


def decode_image(image_bytes: bytes) -> np.ndarray:
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not read that image. Try a JPG or PNG file.")
    return frame


def get_enrollment_dir(name: str, create: bool = True) -> Path | None:
    ensure_directories()
    safe_name = normalize_name(name)

    if (
        st.session_state.get("enrollment_name") != safe_name
        or "enrollment_dir" not in st.session_state
    ):
        if not create:
            return None

        person_id = next_person_id()
        person_dir = FACES_DIR / f"{person_id}_{safe_name}"
        person_dir.mkdir(parents=True, exist_ok=False)
        st.session_state.enrollment_name = safe_name
        st.session_state.enrollment_dir = str(person_dir)
        st.session_state.saved_sample_hashes = set()

    return Path(st.session_state.enrollment_dir)


def save_face_sample(name: str, image_bytes: bytes) -> tuple[bool, str]:
    detector = load_face_detector()
    person_dir = get_enrollment_dir(name)
    if person_dir is None:
        raise RuntimeError("Could not create the enrollment folder.")

    image_hash = hashlib.sha256(image_bytes).hexdigest()
    saved_hashes: set[str] = st.session_state.setdefault("saved_sample_hashes", set())

    if image_hash in saved_hashes:
        return False, "This image was already saved."

    frame = decode_image(image_bytes)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = detect_largest_face(detector, gray)
    if face is None:
        return False, "No face detected in this image."

    next_sample_number = len(list(person_dir.glob("*.png"))) + 1
    sample_path = person_dir / f"{next_sample_number:03d}.png"
    face_image = crop_face(gray, face)
    cv2.imwrite(str(sample_path), face_image)
    saved_hashes.add(image_hash)
    return True, f"Saved sample {next_sample_number}."


def recognize_attendance_image(
    image_bytes: bytes,
    confidence_limit: float,
    action: str,
) -> tuple[np.ndarray, list[str], list[str], int, list[str]]:
    detector = load_face_detector()
    recognizer, labels = load_model()
    frame = decode_image(image_bytes)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    marked_names: list[str] = []
    recognized_names: list[str] = []
    messages: list[str] = []

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80, 80),
    )

    for face in faces:
        face_tuple = tuple(int(value) for value in face)
        face_image = crop_face(gray, face_tuple)
        person_id, confidence = recognizer.predict(face_image)

        if confidence <= confidence_limit and person_id in labels:
            name = labels[person_id]
            recognized_names.append(name)

            if action == "leaving":
                was_marked, detail = mark_leaving(person_id, name)
                status = f"Checked out - {detail}" if was_marked else detail
            else:
                was_marked = mark_attendance(person_id, name)
                status = "Marked" if was_marked else "Present today"

            if was_marked:
                marked_names.append(name)
            messages.append(f"{name}: {status}")

            draw_box(frame, face_tuple, f"{name} - {status}", (0, 180, 0))
        else:
            draw_box(frame, face_tuple, "Unknown", (0, 0, 255))

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return rgb_frame, marked_names, recognized_names, len(faces), messages


def main() -> None:
    ensure_directories()
    ensure_attendance_file()
    apply_theme()
    st.session_state.setdefault("recognition_confidence", int(DEFAULT_CONFIDENCE_LIMIT))
    st.session_state.setdefault(
        "train_recognition_confidence",
        st.session_state.recognition_confidence,
    )
    st.session_state.setdefault("enrollment_name_key_version", 0)
    st.session_state.setdefault("enrollment_camera_key_version", 0)
    st.session_state.setdefault("enrollment_upload_key_version", 0)
    st.session_state.setdefault("attendance_camera_key_version", 0)
    st.session_state.setdefault("leaving_camera_key_version", 0)
    apply_pending_resets()

    with st.sidebar:
        is_admin = show_admin_login()

    show_hero(is_admin)

    tab_names = ["Enroll", "Take Attendance", "Leaving"]
    if is_admin:
        tab_names.extend(["Train", "Records"])

    tabs = st.tabs(tab_names)
    enroll_tab = tabs[0]
    attendance_tab = tabs[1]
    leaving_tab = tabs[2]
    if is_admin:
        train_tab = tabs[3]
        records_tab = tabs[4]

    with enroll_tab:
        st.subheader("Enroll Employee")
        name = st.text_input(
            "Name",
            key=f"enrollment_name_input_{st.session_state.enrollment_name_key_version}",
        )
        if name.strip():
            try:
                person_dir = get_enrollment_dir(name, create=False)
                saved_count = len(list(person_dir.glob("*.png"))) if person_dir else 0
                st.caption(f"{saved_count} sample(s) saved.")
            except Exception as error:
                st.error(str(error))

        if st.button("Open Enrollment Camera"):
            st.session_state.show_enrollment_camera = True

        camera_image = None
        if st.session_state.get("show_enrollment_camera", False):
            camera_image = st.camera_input(
                "Capture a face sample",
                key=f"enrollment_camera_input_{st.session_state.enrollment_camera_key_version}",
            )

        uploaded_samples = st.file_uploader(
            "Or upload face sample images",
            type=("jpg", "jpeg", "png"),
            accept_multiple_files=True,
            key=f"enrollment_file_uploader_{st.session_state.enrollment_upload_key_version}",
        )

        if st.button("Save Samples", type="primary"):
            if not name.strip():
                st.error("Enter a name first.")
            else:
                inputs = []
                if camera_image is not None:
                    inputs.append(camera_image)
                inputs.extend(uploaded_samples or [])

                if not inputs:
                    st.error("Capture or upload at least one image.")
                else:
                    saved = 0
                    messages = []
                    for image_file in inputs:
                        try:
                            did_save, message = save_face_sample(name, image_file.getvalue())
                            saved += int(did_save)
                            messages.append(message)
                        except Exception as error:
                            messages.append(str(error))

                    if saved:
                        st.success(f"Saved {saved} new sample(s).")
                        st.info("This enrollment is now available in the Train tab.")
                        st.caption("Enrollment fields will reset in 30 seconds.")
                        schedule_reset("enrollment")
                    else:
                        st.warning("No new samples were saved.")
                    for message in messages:
                        st.caption(message)

    with attendance_tab:
        st.subheader("Mark Attendance")
        if st.button("Open Attendance Camera"):
            st.session_state.show_attendance_camera = True

        attendance_image = None
        if st.session_state.get("show_attendance_camera", False):
            attendance_image = st.camera_input(
                "Capture attendance image",
                key=f"attendance_camera_input_{st.session_state.attendance_camera_key_version}",
            )

        if st.button("Mark Attendance", type="primary"):
            if attendance_image is None:
                st.error("Open the attendance camera and capture an image first.")
                return

            try:
                (
                    annotated_frame,
                    marked_names,
                    recognized_names,
                    face_count,
                    recognition_messages,
                ) = recognize_attendance_image(
                    attendance_image.getvalue(),
                    float(st.session_state.recognition_confidence),
                    "attendance",
                )
                st.image(annotated_frame, channels="RGB", use_container_width=True)
                if marked_names:
                    st.success("Marked: " + ", ".join(sorted(set(marked_names))))
                    speak_attendance(marked_names)
                elif recognized_names:
                    st.info("; ".join(recognition_messages))
                elif face_count:
                    st.warning("Face detected, but it did not match a trained employee.")
                else:
                    st.warning("No face detected. Try better lighting and face the camera.")
                st.caption(f"Processed at {current_time_ist().strftime('%H:%M:%S')} IST.")
                st.caption("Attendance image will reset in 30 seconds.")
                schedule_reset("attendance")
            except Exception as error:
                st.error(str(error))

    with leaving_tab:
        st.subheader("Mark Leaving")
        if st.button("Open Leaving Camera"):
            st.session_state.show_leaving_camera = True

        leaving_image = None
        if st.session_state.get("show_leaving_camera", False):
            leaving_image = st.camera_input(
                "Capture leaving image",
                key=f"leaving_camera_input_{st.session_state.leaving_camera_key_version}",
            )

        if st.button("Mark Leaving", type="primary"):
            if leaving_image is None:
                st.error("Open the leaving camera and capture an image first.")
                return

            try:
                (
                    annotated_frame,
                    marked_names,
                    recognized_names,
                    face_count,
                    recognition_messages,
                ) = recognize_attendance_image(
                    leaving_image.getvalue(),
                    float(st.session_state.recognition_confidence),
                    "leaving",
                )
                st.image(annotated_frame, channels="RGB", use_container_width=True)
                if marked_names:
                    st.success("Leaving marked: " + ", ".join(sorted(set(marked_names))))
                    speak_leaving(marked_names)
                elif recognized_names:
                    st.info("; ".join(recognition_messages))
                elif face_count:
                    st.warning("Face detected, but it did not match a trained employee.")
                else:
                    st.warning("No face detected. Try better lighting and face the camera.")
                st.caption(f"Processed at {current_time_ist().strftime('%H:%M:%S')} IST.")
                st.caption("Leaving image will reset in 30 seconds.")
                schedule_reset("leaving")
            except Exception as error:
                st.error(str(error))

    if is_admin:
        with train_tab:
            st.subheader("Train Model")
            st.slider(
                "Recognition confidence",
                min_value=40,
                max_value=100,
                help="Lower values are stricter. This same value is used while taking attendance.",
                key="train_recognition_confidence",
                on_change=sync_confidence_from_train,
            )
            training_people = enrolled_people()
            if training_people.empty:
                st.info("No enrolled employees are available for training yet.")
            else:
                training_editor = training_people.copy()
                training_editor.insert(0, "delete", False)
                edited_training = st.data_editor(
                    training_editor,
                    column_config={
                        "delete": st.column_config.CheckboxColumn("Delete"),
                        "person_id": st.column_config.NumberColumn("ID"),
                    },
                    disabled=["person_id", "name", "samples", "status"],
                    hide_index=True,
                    use_container_width=True,
                    key="training_row_editor",
                )

                selected_people = edited_training.loc[
                    edited_training["delete"], "person_id"
                ].tolist()
                if st.button("Delete Selected Training Data", type="secondary"):
                    if not selected_people:
                        st.error("Select one or more enrolled employees to delete.")
                    else:
                        deleted_count = delete_training_people(
                            [int(person_id) for person_id in selected_people]
                        )
                        st.success(
                            f"Deleted training data for {deleted_count} enrolled employees. "
                            "Train the model again before taking attendance."
                        )
                        st.rerun()

            if st.button("Train Face Model", type="primary"):
                try:
                    train()
                    st.success("Model trained successfully.")
                except Exception as error:
                    st.error(str(error))

            st.divider()
            st.subheader("Delete Training Data")

            delete_model_confirmed = st.checkbox(
                "Confirm delete trained model",
                key="delete_model_confirmed",
            )
            if st.button("Delete Trained Model", type="secondary"):
                if not delete_model_confirmed:
                    st.error("Confirm before deleting the trained model.")
                else:
                    delete_trained_model()
                    st.success("Trained model deleted.")

            delete_samples_confirmed = st.checkbox(
                "Confirm delete enrolled face samples",
                key="delete_samples_confirmed",
            )
            if st.button("Delete Enrolled Face Samples", type="secondary"):
                if not delete_samples_confirmed:
                    st.error("Confirm before deleting enrolled face samples.")
                else:
                    delete_training_samples()
                    delete_trained_model()
                    st.success("Enrolled face samples and trained model deleted.")

        with records_tab:
            show_attendance_table()
            st.divider()
            if st.button("Add Absent For Recorded Dates", type="secondary"):
                try:
                    absent_count, absent_dates = add_absent_candidates()
                    if absent_count:
                        st.success(
                            f"Added {absent_count} absent record(s) for "
                            + ", ".join(absent_dates)
                            + "."
                        )
                    else:
                        st.info("No absent candidates to add for recorded attendance dates.")
                except Exception as error:
                    st.error(str(error))

    render_pending_reset_refresh()


if __name__ == "__main__":
    main()
