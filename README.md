# Employee Attendance System

This project enrolls employees, trains a face recognizer, marks attendance from webcam snapshots or uploaded images, and records employee checkout duration. Attendance dates and times are saved in Indian Standard Time (IST).

## Setup

Use the existing virtual environment:

```powershell
.\face_detection_venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run With Streamlit

Start the local web app:

```powershell
streamlit run streamlit_app.py
```

If `streamlit` is not recognized, run it through Python:

```powershell
python -m streamlit run streamlit_app.py
```

The app opens in your browser at `http://localhost:8501`.

Use the tabs in order:

1. `Enroll` - enter a name and save webcam snapshots or uploaded face images.
2. `Take Attendance` - capture an attendance image and mark recognized employees present.
3. `Leaving` - capture a leaving image and mark recognized employees checked out.
4. `Train` - admin-only tab to train the face model.
5. `Records` - admin-only tab to view or download the attendance CSV.

## Deploy On Streamlit Community Cloud

1. Push this project to GitHub.
2. In Streamlit Community Cloud, create a new app from the repository.
3. Set the main file path to:

```text
streamlit_app.py
```

4. Deploy.

The web app uses `st.camera_input`, so the user's browser captures snapshots and sends them to the cloud app. A cloud server cannot access your local webcam through `cv2.VideoCapture`.

Add these secrets in Streamlit Community Cloud before using admin features:

```toml
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "change-this-password"
```

Without these secrets, public users can still use `Enroll`, `Take Attendance`, and `Leaving`, but `Train` and `Records` stay hidden.

`data/model/` is allowed by `.gitignore`, so you can deploy a pre-trained model if needed. `data/faces/` and `attendance/` are ignored because they can contain private or runtime-generated data. Files created on Streamlit Community Cloud are temporary and can disappear after the app restarts.

## 1. Enroll an employee

Run this once for each employee:

```powershell
python attendance_system.py enroll --name "Alice"
```

Look at the camera and slowly turn your head. The script saves face samples into `data/faces/`.

## 2. Train the model

After enrolling all employees:

```powershell
python attendance_system.py train
```

The trained model is saved under `data/model/`.

## 3. Start automatic attendance

```powershell
python attendance_system.py recognize
```

When a known face is detected, attendance is written automatically to:

```text
attendance/attendance.csv
```

Each employee is marked only once per day.

When `Mark Leaving` is used in the Streamlit app, the CSV gets a `Checked Out` row with the duration since that employee's `Present` record for the same IST date.

Attendance timestamps use Indian Standard Time (IST), even when the app runs on Streamlit Community Cloud.

## 4. View recent attendance

```powershell
python attendance_system.py report
```

## Useful options

If your webcam is not camera `0`, try:

```powershell
python attendance_system.py recognize --camera 1
```

If recognition is too strict or too loose, adjust confidence:

```powershell
python attendance_system.py recognize --confidence 60
```

Lower confidence values are stricter. Typical useful values are `55` to `80`.
