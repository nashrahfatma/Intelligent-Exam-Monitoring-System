from __future__ import annotations
import cv2
import mediapipe as mp
from ultralytics import YOLO
import time
import os
import winsound
import tkinter as tk
from tkinter import messagebox

# --- 1. INITIALIZATION ---
if not os.path.exists('proctoring_evidence'):
    os.makedirs('proctoring_evidence')

try:
    model = YOLO('yolov8n.pt')
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
except Exception as e:
    print(f"Error loading models: {e}")


def start_proctoring():
    name = name_entry.get().strip()
    roll = roll_entry.get().strip()

    if not name or not roll:
        messagebox.showerror("Error", "All fields are required!")
        return
    if not roll.startswith("0834"):
        messagebox.showerror("Invalid Format", "Roll Code must start with '0834'")
        return

    root.withdraw()
    run_proctor_engine(name, roll)
    root.deiconify()


def run_proctor_engine(name, roll):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    look_away_frames = 0
    head_down_frames = 0
    last_sound_time = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1)  # Mirror view
        evidence_frame = frame.copy()
        current_time = time.time()
        h, w, _ = frame.shape

        alert_triggered = False
        warning_text = ""

        # --- 2. YOLO DETECTION (Objects & People) ---
        results = model(frame, verbose=False)
        person_count = 0
        for r in results:
            for box in r.boxes:
                label = model.names[int(box.cls[0])]
                if label == 'person': person_count += 1
                if label in ['cell phone', 'book', 'laptop']:
                    alert_triggered = True
                    warning_text = f"WARNING: {label.upper()} DETECTED"
                    b = box.xyxy[0].cpu().numpy()
                    cv2.rectangle(evidence_frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 3)

        if person_count > 1:
            alert_triggered = True
            warning_text = "WARNING: MULTIPLE PEOPLE"

        # --- 3. FACE MESH (Gaze & Head Movement) ---
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(rgb_frame)

        if face_results and face_results.multi_face_landmarks:
            landmarks = face_results.multi_face_landmarks[0].landmark
            nose = landmarks[1]

            # SIDEWAYS GAZE
            if nose.x < 0.44 or nose.x > 0.56:
                look_away_frames += 1
                if look_away_frames > 10:
                    alert_triggered = True
                    warning_text = "WARNING: LOOK AT THE SCREEN!"
            else:
                look_away_frames = 0

                # HEAD DOWN
            if nose.y > 0.60:
                head_down_frames += 1
                if head_down_frames > 10:
                    alert_triggered = True
                    warning_text = "WARNING: SIT STRAIGHT!"
            else:
                head_down_frames = 0

        # --- 4. THE POP-UP WARNING EFFECT ---
        if alert_triggered:
            # Draw a big RED Semi-Transparent Box in the middle
            overlay = frame.copy()
            cv2.rectangle(overlay, (w // 4, h // 3), (3 * w // 4, 2 * h // 3), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            # Warning Text in the box
            cv2.putText(frame, "CAUTION!", (w // 2 - 100, h // 2 - 40),
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 3)
            cv2.putText(frame, warning_text, (w // 2 - 250, h // 2 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # Beep and Save Evidence
            if current_time - last_sound_time > 2.5:
                winsound.Beep(1000, 300)
                # Save Evidence Header
                cv2.rectangle(evidence_frame, (0, 0), (w, 80), (0, 0, 255), -1)
                cv2.putText(evidence_frame, f"ROLL: {roll} | {warning_text}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imwrite(f"proctoring_evidence/{roll}_{int(current_time)}.jpg", evidence_frame)
                last_sound_time = current_time

        # --- 5. TOP STATUS BAR ---
        cv2.putText(frame, f"Exam Mode: {name} ({roll})", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('Secure Exam Portal - AI Monitoring', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


# --- 6. GUI ---
root = tk.Tk()
root.title("College Exam Portal")
root.geometry("450x400")
root.configure(bg="#f4f7f6")

tk.Label(root, text="EXAM REGISTRATION", font=("Arial", 16, "bold"), bg="#2c3e50", fg="white", pady=15).pack(fill="x")
tk.Label(root, text="Student Name:", bg="#f4f7f6").pack(pady=(30, 0))
name_entry = tk.Entry(root, font=("Arial", 12), width=30)
name_entry.pack(pady=5)
tk.Label(root, text="Roll Code (starts with 0834):", bg="#f4f7f6").pack(pady=(15, 0))
roll_entry = tk.Entry(root, font=("Arial", 12), width=30)
roll_entry.pack(pady=5)

tk.Button(root, text="VERIFY & START EXAM", command=start_proctoring, bg="#27ae60", fg="white",
          font=("Arial", 12, "bold"), pady=12, width=20).pack(pady=40)
root.mainloop()