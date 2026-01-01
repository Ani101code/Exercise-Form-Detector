import cv2
import numpy as np
import pyttsx3
import threading
import time
import os
from collections import deque
import queue

# =========================================================
# WINDOWS FIXES
# =========================================================
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_DSHOW'] = '1'

# =========================================================
# MEDIAPIPE INITIALIZATION
# =========================================================
print("=" * 60)
print("MULTI-EXERCISE FORM TRAINER")
print("=" * 60)

try:
    import mediapipe as mp
    print(f"✓ MediaPipe version: {mp.__version__}")
    
    # Use solutions API if available
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    # Create pose detector
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    )
    
    print("✓ Pose detection initialized")
    
except Exception as e:
    print(f"✗ MediaPipe error: {e}")
    exit()

# =========================================================
# AUDIO SYSTEM WITH MUTE/UNMUTE
# =========================================================
class ExerciseAudioFeedback:
    def __init__(self):
        self.enabled = True
        self.muted = False
        self.last_feedback_time = 0
        self.feedback_cooldown = 2
        self.last_form_praise_time = 0
        self.form_praise_cooldown = 5
        self.speech_queue = queue.Queue()
        self.is_speaking = False
        
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 160)
            self.engine.setProperty('volume', 1.0)
            
            threading.Thread(target=self._speech_worker, daemon=True).start()
            print("✓ Audio feedback ready")
        except Exception as e:
            print(f"⚠ Audio disabled: {e}")
            self.enabled = False
            self.muted = True
    
    def _speech_worker(self):
        """Background thread to handle speech queue"""
        while True:
            try:
                text = self.speech_queue.get(timeout=1)
                if text and not self.muted and self.enabled:
                    self.is_speaking = True
                    try:
                        self.engine.say(text)
                        self.engine.runAndWait()
                    except:
                        pass
                    self.is_speaking = False
                self.speech_queue.task_done()
            except queue.Empty:
                continue
            except Exception:
                pass
    
    def toggle_mute(self):
        """Toggle mute on/off"""
        if not self.enabled:
            return False
        
        self.muted = not self.muted
        
        if self.muted:
            while not self.speech_queue.empty():
                try:
                    self.speech_queue.get_nowait()
                    self.speech_queue.task_done()
                except queue.Empty:
                    break
            self.is_speaking = False
        
        status = "MUTED" if self.muted else "UNMUTED"
        
        if not self.muted:
            self.speak(f"Audio {status.lower()}", priority="startup", force=True)
        
        print(f"Audio {status}")
        return self.muted
    
    def speak(self, text, priority="normal", force=False):
        """Speak text with cooldown and mute check"""
        if not self.enabled or self.muted or not text:
            return
        
        if priority == "startup" or force:
            self.speech_queue.put(text)
            return
        
        if priority == "critical":
            self.speech_queue.put(text)
            return
        
        current_time = time.time()
        if current_time - self.last_feedback_time < self.feedback_cooldown:
            return
        
        self.speech_queue.put(text)
        self.last_feedback_time = current_time

# =========================================================
# FORM ANALYSIS ENGINE - SIMPLIFIED FOR RELIABLE COUNTING
# =========================================================
class ExerciseFormAnalyzer:
    def __init__(self):
        # Angle buffers
        self.elbow_buffer = deque(maxlen=5)
        self.shoulder_buffer = deque(maxlen=5)
        self.hip_buffer = deque(maxlen=5)
        
        # Exercise tracking
        self.exercises = ["PUSH-UP", "PULL-UP", "DIP"]
        self.current_exercise = 0
        self.stage = "UP"
        self.reps = 0
        self.form_score = 100
        
        # Session state
        self.session_active = True  # Start counting immediately
        
        # SESSION TRACKER - stores reps for all exercises
        self.session_reps = {
            "PUSH-UP": 0,
            "PULL-UP": 0,
            "DIP": 0
        }
        
        # Initialize form errors dictionary
        self.form_errors = {
            "hips_too_low": False,
            "hips_too_high": False,
            "elbows_flaring": False,
            "incomplete_extension": False,
            "insufficient_depth": False,
            "legs_moving": False,
            "torso_leaning": False
        }
        
        # Simple anti-false positive system
        self.last_rep_time = 0
        self.rep_cooldown = 1.0
        
        # Exercise requirements data
        self.exercise_requirements = {
            "PUSH-UP": {
                "title": "PUSH-UP REQUIREMENTS",
                "requirements": [
                    "1. Start with arms fully extended",
                    "2. Lower until elbows < 95°",
                    "3. Keep body straight",
                    "4. Elbows at 45° angle",
                    "5. Push back up fully",
                    "6. No sagging or raising hips"
                ],
                "form_tips": [
                    "✓ Keep core tight",
                    "✓ Look slightly ahead",
                    "✓ Breathe in on way down",
                    "✓ Breathe out on way up"
                ]
            },
            "PULL-UP": {
                "title": "PULL-UP REQUIREMENTS",
                "requirements": [
                    "1. Start with arms fully extended",
                    "2. Pull until chin over bar",
                    "3. No kipping or leg swinging",
                    "4. Shoulders engaged, not shrugged",
                    "5. Control both up and down",
                    "6. Full range of motion"
                ],
                "form_tips": [
                    "✓ Grip slightly wider than shoulders",
                    "✓ Pull with back, not just arms",
                    "✓ Keep chest up",
                    "✓ Lower with control"
                ]
            },
            "DIP": {
                "title": "DIP REQUIREMENTS",
                "requirements": [
                    "1. Start with arms locked out",
                    "2. Lower until elbows < 80°",
                    "3. Keep torso upright",
                    "4. Don't let shoulders rise to ears",
                    "5. Control descent and ascent",
                    "6. Full range of motion"
                ],
                "form_tips": [
                    "✓ Keep elbows slightly tucked",
                    "✓ Don't go too deep if painful",
                    "✓ Focus on triceps contraction",
                    "✓ Maintain control throughout"
                ]
            }
        }
        
        # SIMPLIFIED thresholds - just count based on elbow angle
        self.exercise_thresholds = {
            "PUSH-UP": {
                "elbow_down": 95,
                "elbow_up": 155,
                "shoulder_depth": 120,
                "hip_min": 140,
                "hip_max": 195,
                "stage_names": ["UP", "DOWN"],
            },
            "PULL-UP": {
                "elbow_bottom": 170,
                "elbow_top": 60,
                "shoulder_engaged": 40,
                "hip_min": 170,
                "stage_names": ["BOTTOM", "TOP"],
            },
            "DIP": {
                "elbow_top": 170,
                "elbow_bottom": 80,  # Slightly higher threshold for easier detection
                "shoulder_depth": 30,
                "torso_upright": 75,
                "stage_names": ["TOP", "BOTTOM"],
            }
        }
        
        # Debug info
        self.last_debug_time = 0
        self.debug_interval = 1.0
    
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        try:
            a = np.array([a.x, a.y]) if hasattr(a, 'x') else np.array(a)
            b = np.array([b.x, b.y]) if hasattr(b, 'x') else np.array(b)
            c = np.array([c.x, c.y]) if hasattr(c, 'x') else np.array(c)
            
            ba = a - b
            bc = c - b
            
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
            
            return np.degrees(np.arccos(cosine_angle))
        except:
            return 0
    
    def analyze_pushup(self, landmarks, thresholds):
        """SIMPLE push-up analysis - just count based on elbow angle"""
        # Get landmarks
        left_shoulder = landmarks[11]
        left_elbow = landmarks[13]
        left_wrist = landmarks[15]
        left_hip = landmarks[23]
        left_knee = landmarks[25]
        
        # Calculate angles
        elbow_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
        shoulder_angle = self.calculate_angle(left_hip, left_shoulder, left_elbow)
        hip_angle = self.calculate_angle(left_shoulder, left_hip, left_knee)
        
        # Smooth angles
        self.elbow_buffer.append(elbow_angle)
        self.shoulder_buffer.append(shoulder_angle)
        self.hip_buffer.append(hip_angle)
        
        smooth_elbow = np.median(list(self.elbow_buffer))
        smooth_shoulder = np.median(list(self.shoulder_buffer))
        smooth_hip = np.median(list(self.hip_buffer))
        
        current_time = time.time()
        
        # Debug info
        if current_time - self.last_debug_time > self.debug_interval:
            print(f"[PUSHUP] Elbow: {int(smooth_elbow)}°, Stage: {self.stage}, Reps: {self.reps}")
            self.last_debug_time = current_time
        
        # Reset form errors
        for key in self.form_errors:
            self.form_errors[key] = False
        
        feedback = ""
        form_deductions = 0
        
        # SIMPLE FORM CHECKS (optional)
        if smooth_hip < thresholds["hip_min"]:
            self.form_errors["hips_too_low"] = True
            feedback = "HIPS TOO LOW - Keep body straight!"
            form_deductions += 15
        elif smooth_hip > thresholds["hip_max"]:
            self.form_errors["hips_too_high"] = True
            feedback = "HIPS TOO HIGH - Lower to plank!"
            form_deductions += 10
        
        # SIMPLE REP COUNTING - JUST USE ELBOW ANGLE
        # Check for DOWN position (elbow bent)
        if smooth_elbow < thresholds["elbow_down"]:
            if self.stage == "UP":
                self.stage = "DOWN"
                if not feedback:
                    feedback = "At bottom"
        
        # Check for UP position (elbow extended)
        elif smooth_elbow > thresholds["elbow_up"]:
            if self.stage == "DOWN":
                if current_time - self.last_rep_time > self.rep_cooldown:
                    self.stage = "UP"
                    self.reps += 1
                    self.session_reps["PUSH-UP"] += 1
                    self.last_rep_time = current_time
                    
                    self.form_score = max(50, 100 - form_deductions)
                    
                    if self.form_score >= 70:
                        feedback = f"✓ PUSH-UP {self.reps}! Form: {self.form_score}%"
                        print(f"✓ Push-up #{self.reps} counted!")
                    else:
                        feedback = f"✗ PUSH-UP {self.reps} (Form: {self.form_score}%)"
                else:
                    feedback = "Wait before next rep"
        
        # Update form score gradually
        target_score = max(50, 100 - form_deductions)
        if target_score > self.form_score:
            self.form_score = min(100, self.form_score + 2)
        elif target_score < self.form_score:
            self.form_score = max(50, self.form_score - 2)
        
        # Default feedback
        if not feedback:
            if self.stage == "UP":
                feedback = "READY - Go down for push-up"
            else:
                feedback = "PUSH - Go up to complete"
        
        return {
            "angles": [smooth_elbow, smooth_shoulder, smooth_hip],
            "feedback": feedback,
            "form_score": int(self.form_score),
            "form_errors": self.form_errors.copy(),
        }
    
    def analyze_pullup(self, landmarks, thresholds):
        """SIMPLE pull-up analysis - just count based on elbow angle"""
        left_shoulder = landmarks[11]
        left_elbow = landmarks[13]
        left_wrist = landmarks[15]
        left_hip = landmarks[23]
        left_knee = landmarks[25]
        
        elbow_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
        shoulder_angle = self.calculate_angle(left_hip, left_shoulder, left_elbow)
        hip_angle = self.calculate_angle(left_shoulder, left_hip, left_knee)
        
        self.elbow_buffer.append(elbow_angle)
        self.shoulder_buffer.append(shoulder_angle)
        self.hip_buffer.append(hip_angle)
        
        smooth_elbow = np.median(list(self.elbow_buffer))
        smooth_shoulder = np.median(list(self.shoulder_buffer))
        smooth_hip = np.median(list(self.hip_buffer))
        
        current_time = time.time()
        
        # Debug info
        if current_time - self.last_debug_time > self.debug_interval:
            print(f"[PULLUP] Elbow: {int(smooth_elbow)}°, Stage: {self.stage}, Reps: {self.reps}")
            self.last_debug_time = current_time
        
        self.form_errors = {key: False for key in self.form_errors}
        
        feedback = ""
        
        # SIMPLE REP COUNTING FOR PULL-UPS
        # Check for TOP position (chin over bar - elbow bent)
        if smooth_elbow < thresholds["elbow_top"]:
            if self.stage == "BOTTOM":
                self.stage = "TOP"
                if not feedback:
                    feedback = "At top"
        
        # Check for BOTTOM position (arms extended)
        elif smooth_elbow > thresholds["elbow_bottom"]:
            if self.stage == "TOP":
                if current_time - self.last_rep_time > self.rep_cooldown:
                    self.stage = "BOTTOM"
                    self.reps += 1
                    self.session_reps["PULL-UP"] += 1
                    self.last_rep_time = current_time
                    feedback = f"✓ PULL-UP {self.reps}!"
                    print(f"✓ Pull-up #{self.reps} counted!")
                else:
                    feedback = "Wait before next rep"
        
        if not feedback:
            if self.stage == "BOTTOM":
                feedback = "PULL - Go up for pull-up"
            else:
                feedback = "LOWER - Go down to complete"
        
        return {
            "angles": [smooth_elbow, smooth_shoulder, smooth_hip],
            "feedback": feedback,
            "form_score": self.form_score,
            "form_errors": self.form_errors.copy(),
        }
    
    def analyze_dip(self, landmarks, thresholds):
        """SIMPLE dip analysis - just count based on elbow angle"""
        left_shoulder = landmarks[11]
        left_elbow = landmarks[13]
        left_wrist = landmarks[15]
        left_hip = landmarks[23]
        left_knee = landmarks[25]
        
        elbow_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
        shoulder_angle = self.calculate_angle(left_hip, left_shoulder, left_elbow)
        hip_angle = self.calculate_angle(left_shoulder, left_hip, left_knee)
        
        self.elbow_buffer.append(elbow_angle)
        self.shoulder_buffer.append(shoulder_angle)
        self.hip_buffer.append(hip_angle)
        
        smooth_elbow = np.median(list(self.elbow_buffer))
        smooth_shoulder = np.median(list(self.shoulder_buffer))
        smooth_hip = np.median(list(self.hip_buffer))
        
        current_time = time.time()
        
        # Debug info
        if current_time - self.last_debug_time > self.debug_interval:
            print(f"[DIP] Elbow: {int(smooth_elbow)}°, Stage: {self.stage}, Reps: {self.reps}")
            self.last_debug_time = current_time
        
        self.form_errors = {key: False for key in self.form_errors}
        
        feedback = ""
        
        # SIMPLE REP COUNTING FOR DIPS
        # Check for TOP position (arms extended)
        if smooth_elbow > thresholds["elbow_top"]:
            if self.stage == "BOTTOM":
                self.stage = "TOP"
                if not feedback:
                    feedback = "At top"
        
        # Check for BOTTOM position (elbows bent)
        elif smooth_elbow < thresholds["elbow_bottom"]:
            if self.stage == "TOP":
                if current_time - self.last_rep_time > self.rep_cooldown:
                    self.stage = "BOTTOM"
                    self.reps += 1
                    self.session_reps["DIP"] += 1
                    self.last_rep_time = current_time
                    feedback = f"✓ DIP {self.reps}!"
                    print(f"✓ Dip #{self.reps} counted!")
                else:
                    feedback = "Wait before next rep"
        
        if not feedback:
            if self.stage == "TOP":
                feedback = "LOWER - Go down for dip"
            else:
                feedback = "PUSH - Go up to complete"
        
        return {
            "angles": [smooth_elbow, smooth_shoulder, smooth_hip],
            "feedback": feedback,
            "form_score": self.form_score,
            "form_errors": self.form_errors.copy(),
        }
    
    def analyze_form(self, landmarks, exercise_name):
        """Analyze form for current exercise"""
        thresholds = self.exercise_thresholds[exercise_name]
        
        if not self.session_active:
            return {
                "angles": [0, 0, 0],
                "feedback": "Press 'S' to start session",
                "form_score": 100,
                "form_errors": self.form_errors.copy(),
            }
        
        if exercise_name == "PUSH-UP":
            return self.analyze_pushup(landmarks, thresholds)
        elif exercise_name == "PULL-UP":
            return self.analyze_pullup(landmarks, thresholds)
        elif exercise_name == "DIP":
            return self.analyze_dip(landmarks, thresholds)
        
        return {
            "angles": [0, 0, 0],
            "feedback": "",
            "form_score": 100,
            "form_errors": {},
        }
    
    def switch_exercise(self, new_index=None):
        """Switch to a different exercise"""
        if new_index is None:
            self.current_exercise = (self.current_exercise + 1) % len(self.exercises)
        else:
            self.current_exercise = new_index % len(self.exercises)
        
        # Reset current exercise reps but keep session total
        self.reps = 0
        self.form_score = 100
        self.stage = "TOP" if self.exercises[self.current_exercise] in ["PULL-UP", "DIP"] else "UP"
        
        self.elbow_buffer.clear()
        self.shoulder_buffer.clear()
        self.hip_buffer.clear()
        
        return self.exercises[self.current_exercise]
    
    def start_session(self):
        """Start a new session"""
        self.session_active = True
        print("✓ Session started! Counting now active.")
        return True
    
    def end_session(self):
        """End current session"""
        self.session_active = False
        print("✓ Session ended.")
        return False
    
    def manual_count(self):
        """Manually add a rep"""
        if not self.session_active:
            return False
        
        current_time = time.time()
        if current_time - self.last_rep_time > 0.5:
            self.reps += 1
            current_exercise = self.exercises[self.current_exercise]
            self.session_reps[current_exercise] += 1
            self.last_rep_time = current_time
            return True
        return False
    
    def reset_current_exercise(self):
        """Reset only current exercise counter"""
        self.reps = 0
        self.form_score = 100
        print(f"Current {self.exercises[self.current_exercise]} counter reset")
    
    def reset_session(self):
        """Reset entire session (all exercises)"""
        self.reps = 0
        self.form_score = 100
        for exercise in self.session_reps:
            self.session_reps[exercise] = 0
        print("Session reset - all counters cleared")
    
    def get_current_requirements(self):
        """Get requirements for current exercise"""
        current_exercise = self.exercises[self.current_exercise]
        return self.exercise_requirements.get(current_exercise, {})
    
    def get_session_summary(self):
        """Get formatted session summary"""
        summary = []
        for exercise, reps in self.session_reps.items():
            if reps > 0:
                summary.append(f"{exercise}: {reps} reps")
        return summary

# =========================================================
# MAIN APPLICATION
# =========================================================
def main():
    print("\n" + "=" * 60)
    print("MULTI-EXERCISE FORM TRAINER")
    print("=" * 60)
    print("SIMPLIFIED COUNTING - Should work reliably!")
    print("\n" + "=" * 60)
    
    audio = ExerciseAudioFeedback()
    form_analyzer = ExerciseFormAnalyzer()
    
    # Initialize camera
    print("\nInitializing camera...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("✗ Cannot open camera!")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("✓ Camera ready")
    
    window_name = "Multi-Exercise Form Trainer - Press ESC to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    print("✓ Window created")
    
    # Instructions
    print("\n" + "=" * 60)
    print("SIMPLE COUNTING RULES:")
    print("  PUSH-UP: Down when elbow < 95°, Up when elbow > 155°")
    print("  PULL-UP: Top when elbow < 60°, Bottom when elbow > 170°")
    print("  DIP: Bottom when elbow < 80°, Top when elbow > 170°")
    print("  1 second cooldown between reps")
    print("\nTIP: Position yourself SIDEWAYS to camera")
    print("     Watch console for angle debug info")
    print("\nCONTROLS:")
    print("  S = Start/Stop session")
    print("  1, 2, 3 = Select exercise")
    print("  SPACE = Next exercise")
    print("  R = Reset counter (current exercise)")
    print("  M = Mute/Unmute audio")
    print("  + = Manually add rep")
    print("  - = Manually subtract rep")
    print("  ESC or Q = Quit")
    print("=" * 60)
    
    if audio.enabled and not audio.muted:
        current_exercise = form_analyzer.exercises[form_analyzer.current_exercise]
        audio.speak(f"{current_exercise.replace('-', ' ')} trainer ready. Session started!", priority="startup")
    
    print("\nSession ACTIVE - start exercising!")
    print("Check console for angle readings every second")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        frame_count += 1
        
        if not ret:
            print("⚠ Frame read failed")
            break
        
        current_time = time.time()
        fps = frame_count / (current_time - start_time) if current_time > start_time > 0 else 0
        
        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        
        form_data = None
        feedback = ""
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            
            current_exercise_name = form_analyzer.exercises[form_analyzer.current_exercise]
            form_data = form_analyzer.analyze_form(results.pose_landmarks.landmark, current_exercise_name)
            feedback = form_data["feedback"]
            
            # SPEAK AUDIO FEEDBACK
            if feedback and audio.enabled and not audio.muted:
                # Speak form corrections
                if "HIPS" in feedback or "ELBOWS" in feedback or "Keep" in feedback:
                    audio.speak(feedback, priority="critical")
                # Speak good reps
                elif "✓" in feedback:
                    audio.speak(feedback.replace("✓", ""))
        
        # =========================================================
        # LEFT PANEL - EXERCISE INFO
        # =========================================================
        left_panel = frame.copy()
        cv2.rectangle(left_panel, (20, 20), (450, 350), (20, 20, 40), -1)
        cv2.rectangle(left_panel, (20, 20), (450, 350), (0, 200, 255), 2)
        frame = cv2.addWeighted(left_panel, 0.7, frame, 0.3, 0)
        
        # Session status
        session_status = "ACTIVE" if form_analyzer.session_active else "INACTIVE"
        session_color = (0, 255, 0) if form_analyzer.session_active else (255, 100, 100)
        cv2.putText(frame, f"SESSION: {session_status}", (40, 60),
                   cv2.FONT_HERSHEY_DUPLEX, 1.0, session_color, 2)
        
        # Title
        current_exercise = form_analyzer.exercises[form_analyzer.current_exercise]
        cv2.putText(frame, f"EXERCISE: {current_exercise}", (40, 100),
                   cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 200, 255), 2)
        
        # Rep count (current exercise)
        cv2.putText(frame, f"CURRENT REPS: {form_analyzer.reps}", (40, 150),
                   cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 0), 3)
        
        # Stage
        stage = form_analyzer.stage
        stage_color = (0, 255, 0) if stage == "UP" or stage == "TOP" else (255, 150, 0)
        cv2.putText(frame, f"STAGE: {stage}", (40, 190),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, stage_color, 2)
        
        # Form score
        form_score = form_data['form_score'] if form_data else 100
        score_color = (0, 255, 0) if form_score > 85 else (100, 200, 255) if form_score > 70 else (100, 100, 255)
        cv2.putText(frame, f"FORM: {int(form_score)}%", (40, 230),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, score_color, 2)
        
        # Audio status
        audio_status = "MUTED" if audio.muted else "ON"
        audio_color = (100, 100, 100) if audio.muted else (0, 255, 0)
        cv2.putText(frame, f"AUDIO: {audio_status}", (40, 270),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, audio_color, 2)
        
        # Angles display
        if form_data and form_data['angles']:
            y_pos = 310
            angle_names = ["Elbow", "Shoulder", "Hip"]
            for i, angle in enumerate(form_data['angles'][:3]):
                # Color code elbow angle
                if angle_names[i] == "Elbow":
                    if current_exercise == "PUSH-UP":
                        if angle < 95:
                            color = (255, 150, 0)  # Orange for DOWN
                        elif angle > 155:
                            color = (0, 255, 0)    # Green for UP
                        else:
                            color = (200, 200, 0)  # Yellow for in-between
                    elif current_exercise == "PULL-UP":
                        if angle < 60:
                            color = (0, 255, 0)    # Green for TOP
                        elif angle > 170:
                            color = (255, 150, 0)  # Orange for BOTTOM
                        else:
                            color = (200, 200, 0)
                    else:  # DIP
                        if angle < 80:
                            color = (255, 150, 0)  # Orange for BOTTOM
                        elif angle > 170:
                            color = (0, 255, 0)    # Green for TOP
                        else:
                            color = (200, 200, 0)
                else:
                    color = (200, 255, 200)
                
                cv2.putText(frame, f"{angle_names[i]}: {int(angle)}°", (40, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                y_pos += 30
        
        # =========================================================
        # MIDDLE PANEL - SESSION TRACKER
        # =========================================================
        middle_panel_x = 480
        middle_panel = frame.copy()
        cv2.rectangle(middle_panel, (middle_panel_x, 20), (middle_panel_x + 300, 350), (40, 20, 40), -1)
        cv2.rectangle(middle_panel, (middle_panel_x, 20), (middle_panel_x + 300, 350), (200, 100, 255), 2)
        frame = cv2.addWeighted(middle_panel, 0.7, frame, 0.3, 0)
        
        # Session tracker title
        cv2.putText(frame, "SESSION TRACKER", (middle_panel_x + 30, 60),
                   cv2.FONT_HERSHEY_DUPLEX, 1.0, (200, 100, 255), 2)
        
        # Show reps for each exercise
        y_pos = 100
        for exercise, reps in form_analyzer.session_reps.items():
            # Highlight current exercise
            if exercise == current_exercise:
                color = (255, 255, 0)  # Yellow for current exercise
                prefix = "> "
            else:
                color = (255, 200, 200)  # Pink for other exercises
                prefix = "  "
            
            cv2.putText(frame, f"{prefix}{exercise}: {reps} reps", 
                       (middle_panel_x + 40, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            y_pos += 40
        
        # Controls
        cv2.putText(frame, "Press 'S' to start/stop", (middle_panel_x + 40, 250),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 100), 1)
        cv2.putText(frame, "Press '+' to manually add", (middle_panel_x + 40, 280),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 100), 1)
        cv2.putText(frame, "Press '-' to subtract", (middle_panel_x + 40, 310),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 100), 1)
        
        # =========================================================
        # RIGHT PANEL - EXERCISE REQUIREMENTS
        # =========================================================
        right_panel_x = width - 450
        right_panel = frame.copy()
        cv2.rectangle(right_panel, (right_panel_x, 20), (width - 20, height - 100), (20, 40, 20), -1)
        cv2.rectangle(right_panel, (right_panel_x, 20), (width - 20, height - 100), (0, 200, 100), 2)
        frame = cv2.addWeighted(right_panel, 0.7, frame, 0.3, 0)
        
        # Get current exercise requirements
        requirements = form_analyzer.get_current_requirements()
        
        if requirements:
            # Title
            cv2.putText(frame, requirements["title"], (right_panel_x + 20, 60),
                       cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 150), 2)
            
            # Requirements list
            y_pos = 100
            for req in requirements["requirements"]:
                cv2.putText(frame, req, (right_panel_x + 30, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1)
                y_pos += 30
            
            # Form tips
            y_pos += 20
            cv2.putText(frame, "FORM TIPS:", (right_panel_x + 20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
            y_pos += 30
            
            for tip in requirements["form_tips"]:
                cv2.putText(frame, tip, (right_panel_x + 30, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 255, 150), 1)
                y_pos += 25
        
        # =========================================================
        # BOTTOM FEEDBACK BAR
        # =========================================================
        if feedback:
            fb_color = (0, 255, 0) if "✓" in feedback else (0, 0, 255) if "✗" in feedback else (200, 200, 0)
            
            text_size = cv2.getTextSize(feedback, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            fb_bg_x = width // 2 - text_size[0] // 2 - 20
            fb_bg_y = height - 80
            
            cv2.rectangle(frame, (fb_bg_x, fb_bg_y), 
                         (fb_bg_x + text_size[0] + 40, fb_bg_y + text_size[1] + 20),
                         (20, 20, 20), -1)
            cv2.rectangle(frame, (fb_bg_x, fb_bg_y), 
                         (fb_bg_x + text_size[0] + 40, fb_bg_y + text_size[1] + 20),
                         fb_color, 2)
            
            cv2.putText(frame, feedback, (width // 2 - text_size[0] // 2, height - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, fb_color, 2)
        
        # FPS
        cv2.putText(frame, f"FPS: {int(fps)}", (width - 120, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 255), 1)
        
        # Controls
        controls = "S:StartStop 1:Pushup 2:Pullup 3:Dip Space:Next M:Mute R:Reset Q:Quit"
        cv2.putText(frame, controls, (width // 2 - 250, height - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Show frame
        cv2.imshow(window_name, frame)
        
        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27 or key == ord('q'):
            break
        elif key == ord('s'):  # Start/Stop session
            if form_analyzer.session_active:
                form_analyzer.end_session()
                if audio.enabled and not audio.muted:
                    audio.speak("Session stopped", priority="startup")
            else:
                form_analyzer.start_session()
                if audio.enabled and not audio.muted:
                    audio.speak("Session started", priority="startup")
        elif key == ord('1'):
            new_exercise = form_analyzer.switch_exercise(0)
            print(f"Switched to {new_exercise}")
            if audio.enabled and not audio.muted:
                audio.speak(f"Push up mode", priority="startup")
        elif key == ord('2'):
            new_exercise = form_analyzer.switch_exercise(1)
            print(f"Switched to {new_exercise}")
            if audio.enabled and not audio.muted:
                audio.speak(f"Pull up mode", priority="startup")
        elif key == ord('3'):
            new_exercise = form_analyzer.switch_exercise(2)
            print(f"Switched to {new_exercise}")
            if audio.enabled and not audio.muted:
                audio.speak(f"Dip mode", priority="startup")
        elif key == ord(' '):
            new_exercise = form_analyzer.switch_exercise()
            print(f"Switched to {new_exercise}")
            if audio.enabled and not audio.muted:
                audio.speak(f"Switched to {new_exercise.replace('-', ' ').lower()}", priority="startup")
        elif key == ord('r'):
            form_analyzer.reset_current_exercise()
            if audio.enabled and not audio.muted:
                audio.speak("Current exercise reset")
        elif key == ord('m'):
            audio.toggle_mute()
        elif key == ord('+'):
            if form_analyzer.manual_count():
                if audio.enabled and not audio.muted:
                    audio.speak(f"Manual count {form_analyzer.reps}")
        elif key == ord('-'):
            if form_analyzer.session_active and form_analyzer.reps > 0:
                form_analyzer.reps -= 1
                current_exercise = form_analyzer.exercises[form_analyzer.current_exercise]
                if form_analyzer.session_reps[current_exercise] > 0:
                    form_analyzer.session_reps[current_exercise] -= 1
                print(f"Manual subtract: Rep {form_analyzer.reps}")
                if audio.enabled and not audio.muted:
                    audio.speak(f"Manual subtract {form_analyzer.reps + 1}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Final session summary
    print(f"\n" + "=" * 60)
    print("FINAL SESSION SUMMARY:")
    print("-" * 60)
    total_reps = 0
    for exercise, reps in form_analyzer.session_reps.items():
        if reps > 0:
            print(f"  {exercise}: {reps} reps")
            total_reps += reps
    print("-" * 60)
    print(f"  TOTAL: {total_reps} reps")
    print("=" * 60)
    
    if audio.enabled and not audio.muted:
        exercise_list = []
        for exercise, reps in form_analyzer.session_reps.items():
            if reps > 0:
                exercise_list.append(f"{reps} {exercise.replace('-', ' ').lower()}s")
        
        if exercise_list:
            summary_text = "Session complete. You did " + ", ".join(exercise_list)
            audio.speak(summary_text, priority="startup")

# =========================================================
# RUN PROGRAM
# =========================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SIMPLIFIED COUNTING - JUST ELBOW ANGLE:")
    print("  PUSH-UP: Down <95°, Up >155°")
    print("  PULL-UP: Top <60°, Bottom >170°")
    print("  DIP: Bottom <80°, Top >170°")
    print("\nTROUBLESHOOTING:")
    print("  1. Check console for elbow angle readings")
    print("  2. Make sure you're SIDEWAYS to camera")
    print("  3. Use '+' key to manually add reps if needed")
    print("=" * 60)
    
    input("\nPress Enter to start camera...")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nGreat workout! Check your session summary above.")