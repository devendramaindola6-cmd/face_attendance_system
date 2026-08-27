from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FACES_DIR = DATA_DIR / "faces"
MODEL_DIR = DATA_DIR / "model"
ATTENDANCE_DIR = BASE_DIR / "attendance"
MODEL_PATH = MODEL_DIR / "lbph_face_model.yml"
LABELS_PATH = MODEL_DIR / "labels.json"
ATTENDANCE_PATH = ATTENDANCE_DIR / "attendance.csv"
ATTENDANCE_COLUMNS = ["date", "time", "person_id", "name", "status", "duration"]

FACE_SIZE = (200, 200)
DEFAULT_CAMERA_INDEX = 0
DEFAULT_SAMPLE_COUNT = 40
DEFAULT_CONFIDENCE_LIMIT = 70.0
DEFAULT_MARK_COOLDOWN_SECONDS = 60
IST = timezone(timedelta(hours=5, minutes=30), name="IST")


@dataclass(frozen=True)
class Person:
    person_id: int
    name: str


def current_time_ist() -> datetime:
    return datetime.now(IST)


def ensure_directories() -> None:
    for path in (FACES_DIR, MODEL_DIR, ATTENDANCE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_face_detector() -> cv2.CascadeClassifier:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"Could not load face detector from {cascade_path}")
    return detector


def open_camera(camera_index: int) -> cv2.VideoCapture:
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise RuntimeError(
            f"Could not open camera index {camera_index}. "
            "Check webcam permission or try --camera 1."
        )
    return camera


def normalize_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in (" ", "-", "_") else "_" for ch in name)
    cleaned = "_".join(cleaned.strip().split())
    if not cleaned:
        raise ValueError("Name must contain at least one letter or number.")
    return cleaned


def next_person_id() -> int:
    ensure_directories()
    ids: list[int] = []
    for path in FACES_DIR.iterdir():
        if path.is_dir():
            prefix = path.name.split("_", 1)[0]
            if prefix.isdigit():
                ids.append(int(prefix))
    return (max(ids) + 1) if ids else 1


def detect_largest_face(
    detector: cv2.CascadeClassifier, gray_frame: np.ndarray
) -> tuple[int, int, int, int] | None:
    detected_faces = detect_faces(detector, gray_frame)
    if not detected_faces:
        return None

    return max(detected_faces, key=lambda face: face[2] * face[3])


def overlap_ratio(
    first_face: tuple[int, int, int, int],
    second_face: tuple[int, int, int, int],
) -> float:
    first_x, first_y, first_width, first_height = first_face
    second_x, second_y, second_width, second_height = second_face

    first_right = first_x + first_width
    first_bottom = first_y + first_height
    second_right = second_x + second_width
    second_bottom = second_y + second_height

    overlap_left = max(first_x, second_x)
    overlap_top = max(first_y, second_y)
    overlap_right = min(first_right, second_right)
    overlap_bottom = min(first_bottom, second_bottom)

    overlap_width = max(0, overlap_right - overlap_left)
    overlap_height = max(0, overlap_bottom - overlap_top)
    overlap_area = overlap_width * overlap_height

    first_area = first_width * first_height
    second_area = second_width * second_height
    smaller_area = min(first_area, second_area)
    if smaller_area == 0:
        return 0

    return overlap_area / smaller_area


def merge_overlapping_faces(
    faces: list[tuple[int, int, int, int]],
    overlap_limit: float = 0.45,
) -> list[tuple[int, int, int, int]]:
    merged_faces: list[tuple[int, int, int, int]] = []

    for face in sorted(faces, key=lambda item: item[2] * item[3], reverse=True):
        if any(overlap_ratio(face, kept_face) >= overlap_limit for kept_face in merged_faces):
            continue
        merged_faces.append(face)

    return merged_faces


def detect_faces(
    detector: cv2.CascadeClassifier, gray_frame: np.ndarray
) -> list[tuple[int, int, int, int]]:
    prepared_frames = [
        gray_frame,
        cv2.equalizeHist(gray_frame),
    ]
    detection_settings = [
        {"scaleFactor": 1.1, "minNeighbors": 5, "minSize": (60, 60)},
        {"scaleFactor": 1.08, "minNeighbors": 4, "minSize": (45, 45)},
        {"scaleFactor": 1.05, "minNeighbors": 3, "minSize": (35, 35)},
    ]

    detected_faces = []
    for prepared_frame in prepared_frames:
        for settings in detection_settings:
            faces = detector.detectMultiScale(prepared_frame, **settings)
            if len(faces):
                detected_faces.extend(
                    tuple(int(value) for value in face) for face in faces
                )

    return merge_overlapping_faces(detected_faces)


def crop_face(gray_frame: np.ndarray, face: tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = face
    cropped = gray_frame[y : y + height, x : x + width]
    resized = cv2.resize(cropped, FACE_SIZE)
    return cv2.equalizeHist(resized)


def draw_box(
    frame: np.ndarray,
    face: tuple[int, int, int, int],
    label: str,
    color: tuple[int, int, int],
) -> None:
    x, y, width, height = face
    cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
    cv2.putText(
        frame,
        label,
        (x, max(30, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        color,
        2,
        cv2.LINE_AA,
    )


def create_recognizer() -> cv2.face.LBPHFaceRecognizer:
    if not hasattr(cv2, "face"):
        raise RuntimeError(
            "OpenCV face recognizer is unavailable. Install opencv-contrib-python."
        )
    return cv2.face.LBPHFaceRecognizer_create(
        radius=2,
        neighbors=8,
        grid_x=8,
        grid_y=8,
    )


def enroll(name: str, camera_index: int, sample_count: int) -> Path:
    ensure_directories()
    person = Person(next_person_id(), normalize_name(name))
    person_dir = FACES_DIR / f"{person.person_id}_{person.name}"
    person_dir.mkdir(parents=True, exist_ok=False)

    detector = load_face_detector()
    camera = open_camera(camera_index)
    saved_count = 0

    print("Enrollment started. Look at the camera and slowly turn your head.")
    print("Press q to stop early.")

    try:
        while saved_count < sample_count:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Could not read from the camera.")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face = detect_largest_face(detector, gray)

            if face is not None:
                face_image = crop_face(gray, face)
                sample_path = person_dir / f"{saved_count + 1:03d}.png"
                cv2.imwrite(str(sample_path), face_image)
                saved_count += 1
                draw_box(
                    frame,
                    face,
                    f"Saved {saved_count}/{sample_count}",
                    (0, 180, 0),
                )
            else:
                cv2.putText(
                    frame,
                    "No face detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Enroll Face - press q to quit", frame)
            if cv2.waitKey(120) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    if saved_count == 0:
        person_dir.rmdir()
        raise RuntimeError("No face samples were saved.")

    print(f"Saved {saved_count} samples for {person.name} in {person_dir}")
    return person_dir


def iter_training_images() -> Iterable[tuple[np.ndarray, int, str]]:
    ensure_directories()
    for person_dir in sorted(FACES_DIR.iterdir()):
        if not person_dir.is_dir():
            continue
        prefix, _, name = person_dir.name.partition("_")
        if not prefix.isdigit() or not name:
            continue
        person_id = int(prefix)
        for image_path in sorted(person_dir.glob("*.png")):
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                yield cv2.resize(image, FACE_SIZE), person_id, name


def train() -> None:
    ensure_directories()
    images: list[np.ndarray] = []
    labels: list[int] = []
    label_names: dict[int, str] = {}

    for image, person_id, name in iter_training_images():
        images.append(image)
        labels.append(person_id)
        label_names[person_id] = name.replace("_", " ")

    if not images:
        raise RuntimeError("No enrollment images found. Run enroll first.")

    recognizer = create_recognizer()
    recognizer.train(images, np.array(labels, dtype=np.int32))
    recognizer.write(str(MODEL_PATH))
    LABELS_PATH.write_text(
        json.dumps({str(key): value for key, value in sorted(label_names.items())}, indent=2),
        encoding="utf-8",
    )
    print(f"Trained model with {len(images)} images for {len(label_names)} people.")
    print(f"Model: {MODEL_PATH}")


def labels_from_face_dirs() -> dict[int, str]:
    ensure_directories()
    labels: dict[int, str] = {}

    for person_dir in sorted(FACES_DIR.iterdir()):
        if not person_dir.is_dir():
            continue

        prefix, _, name = person_dir.name.partition("_")
        if prefix.isdigit() and name:
            labels[int(prefix)] = name.replace("_", " ")

    return labels


def load_model() -> tuple[cv2.face.LBPHFaceRecognizer, dict[int, str]]:
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        raise RuntimeError("Model not found. Run train after enrolling people.")

    recognizer = create_recognizer()
    recognizer.read(str(MODEL_PATH))
    raw_labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    labels = {int(key): value for key, value in raw_labels.items()}
    current_labels = labels_from_face_dirs()
    if current_labels:
        labels = current_labels

    return recognizer, labels


def ensure_attendance_file() -> None:
    ensure_directories()
    if not ATTENDANCE_PATH.exists():
        with ATTENDANCE_PATH.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(ATTENDANCE_COLUMNS)
        return

    with ATTENDANCE_PATH.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if fieldnames != ATTENDANCE_COLUMNS:
        with ATTENDANCE_PATH.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=ATTENDANCE_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row.get(column, "") for column in ATTENDANCE_COLUMNS})


def attendance_status_marked_today(person_id: int, status: str) -> bool:
    ensure_attendance_file()
    today = current_time_ist().date().isoformat()
    with ATTENDANCE_PATH.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return any(
            row["date"] == today
            and row["person_id"] == str(person_id)
            and row["status"] == status
            for row in reader
        )


def attendance_already_marked_today(person_id: int) -> bool:
    return attendance_status_marked_today(person_id, "Present")


def mark_attendance(person_id: int, name: str) -> bool:
    if attendance_already_marked_today(person_id):
        return False

    now = current_time_ist()
    with ATTENDANCE_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                now.date().isoformat(),
                now.strftime("%H:%M:%S"),
                person_id,
                name,
                "Present",
                "",
            ]
        )
    return True


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def comparable_name(name: str) -> str:
    return " ".join(name.strip().casefold().replace("_", " ").split())


def present_time_for_today(person_id: int, name: str) -> datetime | None:
    ensure_attendance_file()
    today = current_time_ist().date().isoformat()
    expected_name = comparable_name(name)

    with ATTENDANCE_PATH.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            row_name = comparable_name(row.get("name", ""))
            if (
                row.get("date") == today
                and row.get("status") == "Present"
                and (
                    row.get("person_id") == str(person_id)
                    or row_name == expected_name
                )
            ):
                try:
                    return datetime.strptime(
                        f"{row['date']} {row['time']}",
                        "%Y-%m-%d %H:%M:%S",
                    ).replace(tzinfo=IST)
                except ValueError:
                    return None

    return None


def mark_leaving(person_id: int, name: str) -> tuple[bool, str]:
    if attendance_status_marked_today(person_id, "Checked Out"):
        return False, "Already checked out today"

    present_time = present_time_for_today(person_id, name)
    if present_time is None:
        today = current_time_ist().date().isoformat()
        return False, f"No present record found for {today}"

    now = current_time_ist()
    duration = format_duration((now - present_time).total_seconds())
    with ATTENDANCE_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                now.date().isoformat(),
                now.strftime("%H:%M:%S"),
                person_id,
                name,
                "Checked Out",
                duration,
            ]
        )

    return True, duration


def recognize(
    camera_index: int,
    confidence_limit: float,
    mark_cooldown_seconds: int,
) -> None:
    detector = load_face_detector()
    recognizer, labels = load_model()
    camera = open_camera(camera_index)
    recent_marks: dict[int, datetime] = {}

    print("Recognition started. Attendance is marked once per person per day.")
    print("Press q to quit.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Could not read from the camera.")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(80, 80),
            )

            for face in faces:
                face_image = crop_face(gray, tuple(face))
                person_id, confidence = recognizer.predict(face_image)
                now = current_time_ist()

                if confidence <= confidence_limit and person_id in labels:
                    name = labels[person_id]
                    last_mark = recent_marks.get(person_id)
                    can_try_mark = (
                        last_mark is None
                        or (now - last_mark).total_seconds() >= mark_cooldown_seconds
                    )
                    if can_try_mark:
                        was_marked = mark_attendance(person_id, name)
                        recent_marks[person_id] = now
                    else:
                        was_marked = False

                    status = "Marked" if was_marked else "Present today"
                    label = f"{name} - {status}"
                    draw_box(frame, tuple(face), label, (0, 180, 0))
                else:
                    draw_box(frame, tuple(face), "Unknown", (0, 0, 255))

            cv2.imshow("Attendance Recognition - press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


def print_report() -> None:
    ensure_attendance_file()
    with ATTENDANCE_PATH.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        print("No attendance records yet.")
        return

    print(f"Attendance report: {ATTENDANCE_PATH}")
    for row in rows[-30:]:
        print(
            f"{row['date']} {row['time']} | "
            f"{row['person_id']} | {row['name']} | {row['status']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automatic attendance system using webcam face recognition."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    enroll_parser = subparsers.add_parser("enroll", help="Capture face samples.")
    enroll_parser.add_argument("--name", required=True, help="Person name.")
    enroll_parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX)
    enroll_parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLE_COUNT)

    subparsers.add_parser("train", help="Train the recognizer from saved samples.")

    recognize_parser = subparsers.add_parser(
        "recognize",
        help="Recognize faces and automatically mark attendance.",
    )
    recognize_parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX)
    recognize_parser.add_argument(
        "--confidence",
        type=float,
        default=DEFAULT_CONFIDENCE_LIMIT,
        help="Lower is stricter. Try 55-80 depending on lighting.",
    )
    recognize_parser.add_argument(
        "--cooldown",
        type=int,
        default=DEFAULT_MARK_COOLDOWN_SECONDS,
        help="Seconds before checking the same person again.",
    )

    subparsers.add_parser("report", help="Print recent attendance records.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "enroll":
        enroll(args.name, args.camera, args.samples)
    elif args.command == "train":
        train()
    elif args.command == "recognize":
        recognize(args.camera, args.confidence, args.cooldown)
    elif args.command == "report":
        print_report()


if __name__ == "__main__":
    main()
