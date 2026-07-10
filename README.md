# Intelligent Vision-Based Examination Monitoring System (IVB-EMS)

## Project Overview

Intelligent Vision-Based Examination Monitoring System (IVB-EMS) is an AI-powered online proctoring solution designed to monitor candidates during online examinations and identify suspicious activities in real time. The system uses computer vision techniques to detect prohibited objects, monitor candidate attention, generate alerts, and capture evidence of violations.

---

## Features

### Object Detection
- Uses YOLOv8 to detect prohibited objects such as:
  - Mobile Phones
  - Books
  - Laptops
- Detects multiple-person presence during examinations.

### Behavioral Monitoring
- Uses MediaPipe Face Mesh for facial landmark detection.
- Monitors head movement and screen attention.
- Identifies suspicious behaviors such as:
  - Looking away from the screen
  - Head-down posture

### Real-Time Alerts
- Generates visual warning messages for detected violations.
- Triggers audible alerts to notify candidates.

### Automated Evidence Collection
- Captures time-stamped screenshots of suspicious activities.
- Stores evidence for administrative review and auditing.

### Examination Registration Interface
- Provides a Tkinter-based GUI for candidate verification and exam session initiation.

---

## Tech Stack

### Programming Language
- Python

### Libraries & Frameworks
- OpenCV
- YOLOv8 (Ultralytics)
- MediaPipe
- Tkinter

### Concepts Used
- Computer Vision
- Object Detection
- Facial Landmark Tracking
- Real-Time Monitoring
- GUI Development

---

## Workflow

1. Candidate enters registration details.
2. System validates examination credentials.
3. Webcam monitoring begins.
4. YOLOv8 detects prohibited objects and additional persons.
5. MediaPipe tracks facial landmarks and head movement.
6. Suspicious activities trigger alerts.
7. Evidence is automatically captured and stored.
8. Examination continues under monitoring.

---

## Future Enhancements

- Cloud-based evidence storage
- Admin dashboard for monitoring
- Advanced gaze tracking
- Examination analytics and reporting

---

## Author

Nashrah Fatma
