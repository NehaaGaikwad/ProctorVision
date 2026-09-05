# ProctorVision AI

### AI-Based Online Exam Proctoring and Violation Monitoring System

ProctorVision AI is an AI-based online examination proctoring system designed to monitor a candidate during an exam and automatically identify suspicious activities using computer vision, audio intelligence, and system-level monitoring.

The system continuously analyzes the candidate's camera feed, microphone input, objects visible in the examination environment, and active application/window state. When suspicious activity is detected, the system records the violation, assigns a severity level, captures supporting evidence, stores the event in a SQLite database, and generates a session-wise JSON report.

The project also includes a Streamlit dashboard for analyzing exam sessions, violation counts, severity distribution, timelines, and captured evidence.

---

## 1. Project Objective

The main objective of ProctorVision AI is to reduce the need for continuous manual supervision during online examinations.

Instead of relying on a human proctor to observe every candidate continuously, the system uses multiple AI-based monitoring modules to detect potentially suspicious behavior.

The system focuses on:

* Face presence detection
* Multiple-person detection
* Gaze direction monitoring
* Voice activity detection
* Mobile phone detection
* Book detection
* Window/application switching detection
* Evidence capture
* Violation severity classification
* Session-wise violation storage
* Automated exam reports
* Dashboard-based analysis

The system is designed as a modular pipeline so that each monitoring component can operate independently while all violations are ultimately managed through a centralized violation management system.

---

## 2. Main Features

### 2.1 Face Detection

The system uses MediaPipe Face Landmarker to detect faces from the webcam feed.

It determines:

* Whether a face is visible
* The number of faces present
* Facial landmark positions

The system supports detection of multiple faces in the examination frame.

---

### 2.2 Face Missing Detection

If the candidate's face disappears from the camera for a continuous number of frames, the system generates a face-not-detected warning.

Current threshold:

```text
15 frames
```

This helps identify situations where the candidate moves away from the camera or intentionally avoids camera monitoring.

---

### 2.3 Multiple Face Detection

The system monitors the number of detected faces.

If more than one face remains visible for the configured threshold, a multiple-face violation is generated.

Current threshold:

```text
10 frames
```

This can help identify situations where another person enters the camera frame.

---

### 2.4 Gaze Monitoring

Facial landmarks are used to estimate the candidate's gaze direction.

The system currently identifies:

* CENTER
* LOOKING LEFT
* LOOKING RIGHT
* LOOKING UP
* LOOKING DOWN

The gaze calculation uses relative positions of the eyes, nose, and chin rather than requiring a separate gaze-tracking model.

A sustained non-center gaze can trigger a gaze violation.

Current threshold:

```text
90 frames
```

The purpose is to avoid treating every small head or eye movement as a violation.

---

### 2.5 Voice Detection

The system continuously monitors microphone input using Silero VAD.

The voice monitoring pipeline:

```text
Microphone
     ↓
Audio Stream
     ↓
Silero VAD
     ↓
Speech Probability
     ↓
Speech Duration Tracking
     ↓
Suspicious Voice Detection
     ↓
Evidence Audio
     ↓
Database
```

Speech probability is continuously evaluated while the exam is active.

Current configuration treats speech lasting longer than:

```text
3 seconds
```

as suspicious activity.

The system does not continuously save microphone recordings.

Instead, audio is temporarily maintained in memory using buffers, and only suspicious voice activity is saved as evidence.

This reduces unnecessary storage of normal audio.

---

### 2.6 Mobile Phone Detection

The project uses YOLO11 for object detection.

The object detector currently focuses on:

```text
cell phone
book
```

Phone detection uses a confirmation period rather than immediately registering every single detection.

Current phone confirmation threshold:

```text
5 frames
```

Once a phone is confirmed:

1. The best-confidence phone detection is selected.
2. The detected object is marked in the evidence frame.
3. The evidence image is saved.
4. A violation is inserted into the database.
5. A severity level is assigned.

Phone severity is currently based on confidence.

---

### 2.7 Book Detection

Books are also detected using YOLO11.

The system monitors the detected book and generates a warning when the object is detected.

Book evidence is captured and stored when a book violation is registered.

Book severity is based on detection confidence.

---

### 2.8 Window/Application Switching Detection

The system also monitors the active Windows application.

The examination window is identified using the configured application/window name:

```text
ProctorVision AI
```

If another application becomes the active window during the exam, the system identifies it as a possible window-switch violation.

The system:

1. Detects the active window.
2. Checks whether it is the examination window.
3. Records the application/window title.
4. Captures a screenshot of the screen.
5. Tracks how long the candidate remains outside the exam window.
6. Assigns severity based on duration.
7. Stores the violation in the database.

Current window-switch severity rules:

```text
< 5 seconds   → LOW
5–9 seconds   → MEDIUM
10+ seconds   → HIGH
```

A cooldown mechanism is also used to reduce repeated immediate detections.

---

## 3. Overall System Architecture

The overall system follows a modular architecture:

```text
                         PROCTORVISION AI
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
          Camera Input      Microphone       System State
              │                 │                 │
              ▼                 ▼                 ▼
        Face Tracker       Voice Detector    Window Monitor
              │                 │                 │
              │                 │                 │
              ├────────────┬────┴─────────────┬───┤
                           │
                           ▼
                    Violation Manager
                           │
                    ┌──────┴──────┐
                    │             │
                    ▼             ▼
                 SQLite       Evidence
                 Database      Storage
                    │
                    ▼
              Report Generator
                    │
                    ▼
                JSON Report
                    │
                    ▼
               Streamlit
                Dashboard
```

Object detection runs alongside the face, voice, and window monitoring modules.

```text
Camera Frame
     │
     ├── FaceTracker
     │      ├── Face Detection
     │      ├── Multiple Faces
     │      └── Gaze Direction
     │
     └── ObjectDetector
            ├── Cell Phone
            └── Book
```

---

## 4. Technology Stack

### Programming Language

* Python

### Computer Vision

* OpenCV
* MediaPipe Face Landmarker

### Object Detection

* Ultralytics YOLO11

### Audio Processing

* Silero VAD
* SoundDevice
* Librosa
* SoundFile
* NumPy

### Data Management

* SQLite
* Pandas

### Dashboard

* Streamlit
* Plotly

### Reporting

* JSON

### Supporting Libraries

* Pillow
* Torch
* TorchVision

---

## 5. Project Structure

```text
ProctorVision/
│
├── database/
│   └── proctorvision.db
│
├── evidence/
│   ├── object_detection/
│   ├── window_switch/
│   ├── voice/
│   └── face_detection/
│
├── reports/
│   └── EXAM_YYYYMMDD_HHMMSS.json
│
├── src/
│   ├── audio_preprocessing.py
│   ├── camera.py
│   ├── dashboard.py
│   ├── face_landmarker.task
│   ├── face_tracker.py
│   ├── main.py
│   ├── monitor.py
│   ├── object_detector.py
│   ├── report_generator.py
│   ├── violation_manager.py
│   └── voice_detector.py
│
├── .gitignore
├── README.md
├── requirements.txt
├── yolo11n.pt
└── yoloe-11s-seg.pt
```

---

## 6. Important Modules

### `main.py`

This is the main execution file of the project.

It initializes and coordinates:

* Camera
* FaceTracker
* VoiceDetector
* WindowMonitor
* ObjectDetector
* ViolationManager
* ReportGenerator

It also controls the examination lifecycle.

The current exam duration is:

```text
120 seconds
```

The main loop continuously processes the camera frame and combines the outputs from all monitoring modules.

---

### `camera.py`

Responsible for:

* Camera initialization
* Reading frames
* FPS calculation
* Camera release

It provides the frames used by the face and object detection systems.

---

### `face_tracker.py`

Responsible for:

* MediaPipe Face Landmarker initialization
* Face landmark detection
* Face counting
* Face status
* Gaze estimation

The MediaPipe model file is:

```text
src/face_landmarker.task
```

The tracker supports up to five faces.

---

### `voice_detector.py`

Responsible for:

* Microphone input
* Audio streaming
* Silero VAD inference
* Speech probability
* Speech duration
* Suspicious voice detection
* Temporary audio buffering
* Suspicious audio evidence generation
* Voice violation database reporting

Voice monitoring runs in a background thread so that microphone processing does not block the main camera-processing loop.

---

### `object_detector.py`

Responsible for YOLO11-based object detection.

Currently monitored classes:

```text
cell phone
book
```

The model used is:

```text
yolo11n.pt
```

Only the required target classes are passed forward to the main monitoring pipeline.

---

### `monitor.py`

Responsible for system/window monitoring.

It:

* Gets the active Windows application
* Checks whether the examination window is active
* Detects application/window switching
* Captures screen evidence
* Measures switch duration
* Assigns severity

This module uses Windows-specific APIs and therefore the current implementation is intended for Windows.

---

### `violation_manager.py`

This is the central data-management component.

It creates and manages the SQLite database.

The database is:

```text
database/proctorvision.db
```

It manages:

* Exam sessions
* Violation records
* Session IDs
* Timestamps
* Duration
* Confidence
* Severity
* Evidence paths

Every violation is associated with the current examination session.

---

### `report_generator.py`

Responsible for generating a JSON report after an examination ends.

The report contains:

* Session ID
* Start time
* End time
* Exam duration
* Total violations
* Violation type summary
* Severity summary
* Final result
* Detailed violation records
* Evidence paths

Reports are stored inside:

```text
reports/
```

---

### `dashboard.py`

The dashboard provides a visual interface for analyzing recorded examination sessions.

It uses Streamlit and Plotly.

The dashboard provides:

* Total exam sessions
* Total violations
* Severity overview
* Violation-type charts
* Session selection
* Session-specific statistics
* Violation timeline
* Detailed violation information
* Evidence preview
* Voice evidence playback
* JSON report viewing
* JSON report download
* All session records

The dashboard supports evidence from:

```text
PHONE
BOOK
WINDOW_SWITCH
NO_FACE
MULTIPLE_FACES
GAZE_VIOLATION
VOICE
```

---

## 7. Examination Workflow

The complete workflow is:

```text
Start Application
      │
      ▼
Camera + Microphone Initialized
      │
      ▼
Press S
      │
      ▼
Exam Session Created
      │
      ▼
Continuous Monitoring
      │
      ├── Face Detection
      ├── Gaze Detection
      ├── Multiple Face Detection
      ├── Voice Detection
      ├── Phone Detection
      ├── Book Detection
      └── Window Monitoring
      │
      ▼
Suspicious Activity Detected
      │
      ▼
Evidence Captured
      │
      ▼
Violation Recorded
      │
      ▼
SQLite Database
      │
      ▼
Exam Ends
      │
      ▼
Session Closed
      │
      ▼
JSON Report Generated
      │
      ▼
Dashboard Analysis
```

---

## 8. Exam Controls

Before starting the exam:

```text
S → Start Exam
Q → Quit
```

During the exam:

```text
E → End Exam
Q → End/Exit
```

When the exam reaches the configured duration automatically, monitoring is stopped and the report is generated.

---

## 9. Violation Management

All violations are stored in a centralized database.

Each violation contains:

```text
ID
Type
Timestamp
Duration
Confidence
Severity
Evidence Path
Session ID
```

Example violation types:

```text
VOICE
PHONE
BOOK
WINDOW_SWITCH
NO_FACE
MULTIPLE_FACES
GAZE_VIOLATION
```

This centralized structure makes it possible to analyze all violations together while still keeping each exam session separate.

---

## 10. Severity System

The project uses three severity levels:

```text
LOW
MEDIUM
HIGH
```

Severity depends on the type of violation.

For example, window switching is based on duration, while phone and book violations use detection confidence.

Voice violations use a combination of speech duration and confidence.

The report generator uses the highest detected severity to determine the final session result.

```text
No violations
      ↓
NO VIOLATIONS

Any LOW violations
      ↓
LOW RISK

Any MEDIUM violations
      ↓
MEDIUM RISK

Any HIGH violations
      ↓
HIGH RISK
```

---

## 11. Evidence Management

ProctorVision AI does not only store a violation label.

It also attempts to preserve evidence supporting the detected event.

Examples include:

### Image Evidence

Used for:

* Phone detection
* Book detection
* Face-related violations
* Gaze violations
* Multiple-face violations
* Window switching

### Audio Evidence

Used for:

* Suspicious voice activity

### Screen Evidence

Used for:

* Window/application switching

Evidence is stored separately from the database.

The database stores the path to the evidence file rather than storing the complete binary evidence inside SQLite.

This keeps the database lightweight and makes evidence files independently accessible.

---

## 12. Database Design

The system uses SQLite for persistent storage.

### Sessions Table

The session information contains:

```text
id
session_id
start_time
end_time
status
```

A session ID follows the format:

```text
EXAM_YYYYMMDD_HHMMSS
```

Example:

```text
EXAM_20260905_210826
```

### Violations Table

The violation information contains:

```text
id
type
timestamp
duration
confidence
severity
evidence_path
session_id
```

The `session_id` connects each violation to the examination in which it occurred.

---

## 13. Report Generation

At the end of an examination, the system generates a JSON report.

Example structure:

```json
{
    "session": {
        "session_id": "EXAM_20260905_210826",
        "start_time": "2026-09-05 21:08:26",
        "end_time": "2026-09-05 21:10:26",
        "status": "COMPLETED",
        "total_duration_seconds": 120.0
    },
    "summary": {
        "total_violations": 17,
        "violation_types": {},
        "severity": {},
        "final_result": "HIGH RISK"
    },
    "violations": []
}
```

The actual report contains the complete violation information for that session.

---

## 14. Dashboard

The Streamlit dashboard is designed to provide a post-examination analysis interface.

### Overall Analytics

The dashboard displays:

* Total sessions
* Total violations
* High-severity violations
* Medium-severity violations
* Violation type distribution
* Severity distribution

### Session Analytics

After selecting a particular session, the dashboard displays:

* Session status
* Violation count
* Risk score
* Final result
* Exam duration
* Violation-type distribution
* Severity distribution
* Violation timeline
* Detailed violations
* Evidence
* JSON report

The charts allow hovering over individual columns to view their count while disabling zooming and unnecessary chart controls.

---

## 15. Installation

### Step 1 — Clone the Repository

```bash
git clone https://github.com/NehaaGaikwad/ProctorVision.git
cd ProctorVision
```

### Step 2 — Create a Virtual Environment

```bash
python -m venv env
```

Activate it on Windows:

```powershell
.\env\Scripts\Activate.ps1
```

If PowerShell activation is restricted, the environment can also be activated using:

```powershell
env\Scripts\activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 16. Running the Proctoring System

From the project root:

```bash
python src/main.py
```

The system opens the ProctorVision AI examination interface.

Press:

```text
S
```

to start the examination.

During the examination, all monitoring modules operate continuously.

---

## 17. Running the Dashboard

After an examination has been completed, run:

```bash
streamlit run src/dashboard.py
```

The dashboard reads the SQLite database and generated reports and displays the recorded examination information.

---

## 18. Required Model Files

The project requires the following model files:

### MediaPipe

```text
src/face_landmarker.task
```

### YOLO

```text
yolo11n.pt
```

These files are required by the corresponding detection modules.

---

## 19. Audio Pipeline

The voice monitoring system operates at:

```text
Sample Rate: 16000 Hz
```

The microphone stream is processed in small audio chunks.

The audio pipeline is:

```text
Microphone
     ↓
Audio Chunk
     ↓
Silero VAD
     ↓
Speech Probability
     ↓
Speech State
     ↓
Duration Tracking
     ↓
Violation Decision
```

A rolling buffer is maintained so that when suspicious speech is detected, the saved evidence can include audio surrounding the event rather than only the exact instant at which the threshold was crossed.

Normal speech is not permanently stored.

Only suspicious voice activity is saved as evidence.

---

## 20. Face and Gaze Pipeline

The face-monitoring pipeline uses MediaPipe Face Landmarker.

```text
Camera Frame
     ↓
BGR → RGB
     ↓
MediaPipe Face Landmarker
     ↓
Face Landmarks
     ↓
Face Count
     ↓
Gaze Calculation
     ↓
Monitoring Decision
```

Selected facial landmarks are used to calculate relative horizontal and vertical offsets.

These offsets are mapped to the gaze states:

```text
CENTER
LOOKING LEFT
LOOKING RIGHT
LOOKING UP
LOOKING DOWN
```

The system uses frame-based thresholds to reduce false positives caused by short movements.

---

## 21. Object Detection Pipeline

YOLO11 is used for object detection.

```text
Camera Frame
     ↓
YOLO11
     ↓
Detected Objects
     ↓
Filter Target Classes
     ↓
Phone / Book
     ↓
Confidence Check
     ↓
Violation Confirmation
```

The current target classes are:

```text
cell phone
book
```

The detector records:

* Object class
* Confidence
* Bounding box

The highest-confidence phone detection is used when generating phone evidence.

---

## 22. Window Monitoring Pipeline

The system checks the active Windows application during the examination.

```text
Active Window
     ↓
Compare With Exam Window
     ↓
Exam Window?
   /       \
 YES       NO
  │         │
Clean    Switch Detected
            │
            ▼
       Screenshot
            │
            ▼
       Duration Tracking
            │
            ▼
         Severity
            │
            ▼
         Database
```

The current implementation uses Windows APIs to identify the foreground application and Pillow's screen capture functionality to create evidence screenshots.

---

## 23. Data Flow

The complete data flow of the project is:

```text
Camera ───────────────┐
                      │
Microphone ───────────┤
                      │
Windows State ────────┤
                      ▼
              Detection Modules
                      │
                      ▼
              Violation Decision
                      │
             ┌────────┴────────┐
             │                 │
             ▼                 ▼
         Evidence          Violation
          Storage          Manager
             │                 │
             └────────┬────────┘
                      ▼
                   SQLite
                      │
                      ▼
               Report Generator
                      │
                      ▼
                 JSON Report
                      │
                      ▼
                  Dashboard
```

---

## 24. Design Approach

The project was developed using a modular architecture instead of placing every detection operation inside one large script.

Each major responsibility is separated into its own module:

```text
Camera              → camera.py
Face/Gaze            → face_tracker.py
Voice                → voice_detector.py
Objects              → object_detector.py
Window Monitoring    → monitor.py
Database             → violation_manager.py
Reports              → report_generator.py
Dashboard            → dashboard.py
System Controller    → main.py
```

This makes the system easier to:

* Debug
* Test
* Modify
* Extend
* Maintain

For example, a new object class can be added to the object detector without rewriting the database or report-generation system.

---

## 25. Current Detection Thresholds

The current project configuration includes:

| Detection          |   Threshold |
| ------------------ | ----------: |
| Face missing       |   15 frames |
| Multiple faces     |   10 frames |
| Gaze violation     |   90 frames |
| Phone confirmation |    5 frames |
| Book detection     |     1 frame |
| Book missed reset  |   20 frames |
| Phone missed reset |   20 frames |
| Suspicious voice   |   3 seconds |
| Window cooldown    |   2 seconds |
| Exam duration      | 120 seconds |

These values can be tuned according to camera quality, environment, hardware, and examination requirements.

---

## 26. Current Limitations

The current version is a working prototype and still has limitations.

### Camera Dependency

Face and object detection depend on camera quality, lighting, camera angle, and visibility.

### Gaze Estimation

Gaze direction is estimated using facial landmark geometry and is not a dedicated eye-tracking model.

### Voice Detection

Voice detection identifies speech activity and suspicious duration, but it does not identify the speaker or understand the spoken content.

### Window Monitoring

Window monitoring is currently implemented using Windows-specific APIs.

### Object Detection

Phone and book detection depend on YOLO detection confidence and the visual quality of the object.

### False Positives

Normal candidate movements may sometimes be classified as suspicious. Threshold tuning is therefore important.

---

## 27. Future Improvements

The current architecture provides a foundation for future improvements.

Possible extensions include:

* Better gaze estimation
* Head-pose estimation
* Face recognition for candidate identity verification
* Improved phone/book detection
* More object classes
* Audio classification
* Speaker identification
* Whisper-based speech transcription
* Suspicious keyword detection
* Better anti-spoofing
* Candidate re-identification
* More advanced risk scoring
* Real-time browser-based examination interface
* Multi-camera support
* Cloud-based monitoring
* Authentication
* Admin and examiner accounts
* Advanced analytics
* PDF report generation
* Real-time dashboard monitoring
* Email/report notification system

---

## 28. Security and Privacy Considerations

The system is designed to collect examination-related evidence only when suspicious activity is detected.

For voice monitoring, normal speech is temporarily processed in memory and suspicious voice activity is saved as evidence.

Evidence files should be treated as sensitive examination data and should not be publicly shared.

The project should be deployed with appropriate:

* Data retention policies
* Access controls
* Candidate consent
* Secure storage
* Evidence protection
* Institutional privacy policies

---

## 29. Testing

The project can be tested by intentionally creating different examination scenarios.

### Face Tests

* Move away from the camera
* Bring another person into the frame
* Look in different directions

### Voice Tests

* Remain silent
* Speak for less than the violation duration
* Speak continuously for more than the violation duration

### Object Tests

* Show a mobile phone
* Show a book
* Remove the object

### Window Tests

* Keep the exam window active
* Switch to another application
* Return to the exam window

After the exam, the database, evidence folders, JSON report, and Streamlit dashboard can be checked to verify the recorded events.

---

## 30. Example Output

A completed examination produces:

```text
database/
└── proctorvision.db

evidence/
├── object_detection/
├── window_switch/
├── voice/
└── face_detection/

reports/
└── EXAM_YYYYMMDD_HHMMSS.json
```

The JSON report summarizes the complete session and the dashboard provides a visual representation of the same recorded information.

---

## 31. Project Status

### Current Working Components

* [x] Webcam monitoring
* [x] Face detection
* [x] Multiple-face detection
* [x] Face-missing detection
* [x] Gaze monitoring
* [x] Voice activity detection
* [x] Suspicious voice detection
* [x] Voice evidence capture
* [x] Phone detection
* [x] Book detection
* [x] Window/application switch detection
* [x] Screenshot evidence
* [x] SQLite database
* [x] Session management
* [x] Violation management
* [x] JSON report generation
* [x] Streamlit dashboard
* [x] Violation analytics
* [x] Severity analytics
* [x] Evidence visualization
* [x] Session-wise analysis

---

## 32. Conclusion

ProctorVision AI combines computer vision, audio processing, object detection, system monitoring, database management, and visualization into a single online examination proctoring system.

The project is designed around the idea that suspicious examination behavior should not only be detected but also **recorded, supported with evidence, stored against a specific examination session, and made available for later analysis**.

The modular architecture allows additional monitoring techniques to be integrated without redesigning the entire system.

The current implementation provides the foundation for a more advanced AI-powered examination monitoring platform.

---

## Author

**Neha Gaikwad**

Computer Engineering

**Project:** ProctorVision AI

**Repository:**
https://github.com/NehaaGaikwad/ProctorVision
