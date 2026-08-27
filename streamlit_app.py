from __future__ import annotations

import base64
import csv
import hashlib
import shutil
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

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
    MODEL_PATH,
    next_person_id,
    normalize_name,
    train,
)

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
HERO_IMAGE_PATH = ASSETS_DIR / "aisha-scanner-facerec.png"


st.set_page_config(
    page_title="Face Attendance",
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
                background: #ffffff;
                border-right: 1px solid rgba(16, 32, 39, 0.08);
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


def show_attendance_table() -> None:
    attendance = read_attendance()
    st.subheader("Attendance")
    if st.button("Delete CSV Data", type="secondary"):
        ensure_attendance_file()
        ATTENDANCE_PATH.write_text("date,time,person_id,name,status\n", encoding="utf-8")
        st.success("Attendance CSV data deleted.")
        attendance = read_attendance()

    if attendance.empty:
        st.info("No attendance marked yet.")
        return

    st.dataframe(attendance.tail(50), use_container_width=True, hide_index=True)
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


def model_status() -> str:
    return "Ready" if MODEL_PATH.exists() and LABELS_PATH.exists() else "Not trained"


def image_as_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def show_hero() -> None:
    people = enrolled_people()
    attendance = read_attendance()
    sample_total = int(people["samples"].sum()) if not people.empty else 0

    left, right = st.columns([1.04, 0.96], vertical_alignment="center")
    with left:
        st.markdown(
            """
            <div class="hero-eyebrow">Smart Classroom Attendance</div>
            <h1 class="hero-title">Face Attendance System</h1>
            <p class="hero-copy">
                Enroll students, train the recognizer, and mark attendance from secure
                browser camera snapshots. Built for Streamlit Cloud with IST timestamps.
            </p>
            <div class="hero-pill-row">
                <span class="hero-pill">Cloud ready</span>
                <span class="hero-pill">Browser camera</span>
                <span class="hero-pill">IST records</span>
                <span class="hero-pill">CSV export</span>
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

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Enrolled People", len(people))
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
) -> tuple[np.ndarray, list[str]]:
    detector = load_face_detector()
    recognizer, labels = load_model()
    frame = decode_image(image_bytes)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    marked_names: list[str] = []

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
            was_marked = mark_attendance(person_id, name)
            if was_marked:
                marked_names.append(name)

            status = "Marked" if was_marked else "Present today"
            draw_box(frame, face_tuple, f"{name} - {status}", (0, 180, 0))
        else:
            draw_box(frame, face_tuple, "Unknown", (0, 0, 255))

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return rgb_frame, marked_names


def main() -> None:
    ensure_directories()
    ensure_attendance_file()
    apply_theme()

    show_hero()

    with st.sidebar:
        st.header("Recognition")
        confidence_limit = st.slider(
            "Recognition confidence",
            min_value=40,
            max_value=100,
            value=int(DEFAULT_CONFIDENCE_LIMIT),
            help="Lower values are stricter.",
        )

    enroll_tab, train_tab, attendance_tab, records_tab = st.tabs(
        ["Enroll", "Train", "Take Attendance", "Records"]
    )

    with enroll_tab:
        st.subheader("Enroll Person")
        name = st.text_input("Name")
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
                key="enrollment_camera_input",
            )

        uploaded_samples = st.file_uploader(
            "Or upload face sample images",
            type=("jpg", "jpeg", "png"),
            accept_multiple_files=True,
            key="enrollment_file_uploader",
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
                    else:
                        st.warning("No new samples were saved.")
                    for message in messages:
                        st.caption(message)

    with train_tab:
        st.subheader("Train Model")
        training_people = enrolled_people()
        if training_people.empty:
            st.info("No enrolled people are available for training yet.")
        else:
            st.dataframe(training_people, use_container_width=True, hide_index=True)

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

    with attendance_tab:
        st.subheader("Mark Attendance")
        if st.button("Open Attendance Camera"):
            st.session_state.show_attendance_camera = True

        attendance_image = None
        if st.session_state.get("show_attendance_camera", False):
            attendance_image = st.camera_input(
                "Capture attendance image",
                key="attendance_camera_input",
            )

        if st.button("Mark Attendance", type="primary"):
            if attendance_image is None:
                st.error("Open the attendance camera and capture an image first.")
                return

            try:
                annotated_frame, marked_names = recognize_attendance_image(
                    attendance_image.getvalue(),
                    float(confidence_limit),
                )
                st.image(annotated_frame, channels="RGB", use_container_width=True)
                if marked_names:
                    st.success("Marked: " + ", ".join(sorted(set(marked_names))))
                else:
                    st.info(
                        "No new attendance was marked. Faces may be unknown or already present today."
                    )
                st.caption(f"Processed at {current_time_ist().strftime('%H:%M:%S')} IST.")
            except Exception as error:
                st.error(str(error))

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

    with records_tab:
        show_attendance_table()


if __name__ == "__main__":
    main()
