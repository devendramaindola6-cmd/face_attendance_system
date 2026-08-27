from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from attendance_system import (
    ATTENDANCE_PATH,
    DEFAULT_CONFIDENCE_LIMIT,
    DEFAULT_SAMPLE_COUNT,
    FACES_DIR,
    crop_face,
    detect_largest_face,
    draw_box,
    ensure_attendance_file,
    ensure_directories,
    load_face_detector,
    load_model,
    mark_attendance,
    next_person_id,
    normalize_name,
    train,
)


st.set_page_config(
    page_title="Face Attendance",
    page_icon=":camera:",
    layout="wide",
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

    st.title("Face Detection Attendance")

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
        target_sample_count = st.slider(
            "Target samples",
            min_value=5,
            max_value=80,
            value=DEFAULT_SAMPLE_COUNT,
            step=5,
        )
        if name.strip():
            try:
                person_dir = get_enrollment_dir(name, create=False)
                saved_count = len(list(person_dir.glob("*.png"))) if person_dir else 0
                st.progress(min(saved_count / target_sample_count, 1.0))
                st.caption(f"{saved_count} of {target_sample_count} target samples saved.")
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

        uploaded_attendance_image = st.file_uploader(
            "Or upload an attendance image",
            type=("jpg", "jpeg", "png"),
            key="attendance_file_uploader",
        )

        if st.button("Mark From Image", type="primary"):
            image_file = attendance_image or uploaded_attendance_image
            if image_file is None:
                st.error("Capture or upload an image first.")
                return

            try:
                annotated_frame, marked_names = recognize_attendance_image(
                    image_file.getvalue(),
                    float(confidence_limit),
                )
                st.image(annotated_frame, channels="RGB", use_container_width=True)
                if marked_names:
                    st.success("Marked: " + ", ".join(sorted(set(marked_names))))
                else:
                    st.info(
                        "No new attendance was marked. Faces may be unknown or already present today."
                    )
                st.caption(f"Processed at {datetime.now().strftime('%H:%M:%S')}.")
            except Exception as error:
                st.error(str(error))

    with records_tab:
        show_attendance_table()


if __name__ == "__main__":
    main()
