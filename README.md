# Intelligent Vision-Based Examination Monitoring System (IVB-EMS)

## Overview

Intelligent Vision-Based Examination Monitoring System (IVB-EMS) is an AI-powered online proctoring solution designed to monitor candidates during online examinations and identify suspicious activities in real time.

The system combines computer vision, object detection, facial landmark tracking, and automated evidence collection to support examination integrity.

---

## Features

### Candidate Verification
- Validates student information before starting the examination.
- Verifies roll code format through a secure registration interface.

### Object Detection
- Uses YOLOv8 to detect prohibited objects such as:
  - Mobile Phones
  - Books
  - Laptops
- Detects multiple persons in the examination frame.

### Head Movement & Attention Monitoring
- Uses MediaPipe Face Mesh for facial landmark detection.
- Tracks head position and screen attention.
- Identifies:
  - Looking away from the screen
  - Head-down posture

### Real-Time Alerts
- Displays warning messages when suspicious activity is detected.
- Generates audible alerts using system notifications.

### Automated Evidence Collection
- Captures screenshots of detected violations.
- Saves time-stamped evidence for review and audit purposes.

### User Interface
- Built with Tkinter for examination registration and session management.

---

## Tech Stack

### Programming Language
- Python

### Libraries
- OpenCV
- YOLOv8 (Ultralytics)
- MediaPipe
- Tkinter

### Functional Modules
- Computer Vision
- Object Detection
- Face Landmark Tracking
- GUI Development

---

## Workflow

1. Student enters name and roll code.
2. System validates examination credentials.
3. Webcam monitoring starts.
4. YOLOv8 detects prohibited objects and additional persons.
5. MediaPipe tracks facial landmarks and head movement.
6. Suspicious activities trigger warnings.
7. Evidence is automatically captured and stored.
8. Examination session continues under monitoring.

---

## Future Improvements

- Database integration for candidate records
- Cloud-based evidence storage
- Admin dashboard for review
- Advanced gaze tracking
- Examination analytics reports

---

## Author

Nashrah Fatma
