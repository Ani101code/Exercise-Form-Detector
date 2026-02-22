import cv2
import numpy as np
import threading
import time
import os
import queue
from collections import deque
import sys

# =========================================================
# WINDOWS FIXES
# =========================================================
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_DSHOW'] = '1'

# =========================================================
# MEDIAPIPE INITIALIZATION
# =========================================================
print("=" * 60)
print("MULTI-EXERCISE FORM TRAINER - GOOD FORM ONLY")
print("=" * 60)

try:
    import mediapipe as mp
    print(f"✓ MediaPipe version: {mp.__version__}")
    
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
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
# AUDIO SYSTEM
# =========================================================
class ExerciseAudioFeedback:
    def __init__(self):
        self.enabled = True
        self.muted = True
        self.last_feedback_time = 0
        self.feedback_cooldown = 4.0
        self.last_spoken = ""
        self.rep_cooldown = 2.0
        self.form_warning_cooldown = 6.0
        
        self.audio_queue = queue.Queue()
        self.audio_worker_running = True
        self.audio_worker_thread = threading.Thread(target=self._audio_worker, daemon=True)
        
        try:
            import win32com.client
            self.speaker = win32com.client.Dispatch("SAPI.SpVoice")
            self.voice = "Windows"
            print("✓ Using Windows TTS (win32com)")
            self.audio_worker_thread.start()
            print("✓ Audio worker thread started")
            
        except ImportError:
            try:
                import pyttsx3
                self.speaker = pyttsx3.init()
                self.speaker.setProperty('rate', 160)
                self.speaker.setProperty('volume', 1.0)
                self.voice = "pyttsx3"
                print("✓ Using pyttsx3")
                self.audio_worker_thread.start()
                print("✓ Audio worker thread started")
                
            except Exception as e:
                print(f"✗ No TTS available: {e}")
                self.enabled = False
                self.muted = True
                self.audio_worker_running = False
                return
        
        print("Audio system ready (initially MUTED - press 'M' to unmute)")
    
    def _audio_worker(self):
        while self.audio_worker_running:
            try:
                try:
                    text = self.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                if text is None:
                    break
                
                if self.voice == "Windows":
                    try:
                        self.speaker.Speak(text)
                    except Exception as e:
                        print(f"Windows TTS error: {e}")
                else:
                    try:
                        self.speaker.say(text)
                        self.speaker.runAndWait()
                    except Exception as e:
                        print(f"pyttsx3 error: {e}")
                
                self.audio_queue.task_done()
                
            except Exception as e:
                print(f"Audio worker error: {e}")
                time.sleep(0.1)
    
    def _queue_speech(self, text):
        if not self.enabled or self.muted or not text:
            return
        
        try:
            self.audio_queue.put(text, block=False)
        except queue.Full:
            print("Audio queue full, skipping message")
        except Exception as e:
            print(f"Queue error: {e}")
    
    def toggle_mute(self):
        if not self.enabled:
            return False
        
        self.muted = not self.muted
        status = "MUTED" if self.muted else "UNMUTED"
        
        self._clear_queue()
        
        feedback_text = f"Audio {status.lower()}"
        self._queue_speech(feedback_text)
        
        print(f"Audio {status}")
        return self.muted
    
    def _clear_queue(self):
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except queue.Empty:
                break
    
    def speak(self, text, priority="normal", force=False):
        if not self.enabled or self.muted or not text:
            return
        
        current_time = time.time()
        
        if priority == "startup":
            self._queue_speech(text)
            return
        
        if force:
            self._queue_speech(text)
            self.last_feedback_time = current_time
            self.last_spoken = text
            return
        
        cooldown_needed = self.feedback_cooldown
        
        if "Good" in text or "Excellent" in text:
            cooldown_needed = self.rep_cooldown
        elif any(word in text.lower() for word in ["too", "flaring", "kipping", "straight", "raise", "lower", "knees", "wide"]):
            cooldown_needed = self.form_warning_cooldown
        
        if current_time - self.last_feedback_time > cooldown_needed:
            if text != self.last_spoken or current_time - self.last_feedback_time > cooldown_needed * 2:
                self._queue_speech(text)
                self.last_feedback_time = current_time
                self.last_spoken = text
    
    def cleanup(self):
        self.audio_worker_running = False
        try:
            self.audio_queue.put(None)
        except:
            pass
        
        if self.audio_worker_thread.is_alive():
            self.audio_worker_thread.join(timeout=1.0)

# =========================================================
# FORM ANALYZER - GOOD FORM ONLY
# =========================================================
class ExerciseFormAnalyzer:
    def __init__(self):
        self.elbow_buffer = deque(maxlen=10)
        self.shoulder_buffer = deque(maxlen=10)
        self.hip_buffer = deque(maxlen=10)
        self.knee_width_buffer = deque(maxlen=10)  # For squat knee width
        
        # Exercises list
        self.exercises = ["PUSH-UP", "PULL-UP", "SQUAT"]
        self.current_exercise = 0
        self.stage = "UP"
        self.reps = 0
        self.form_score = 100
        
        self.session_active = False
        
        self.session_reps = {
            "PUSH-UP": 0,
            "PULL-UP": 0,
            "SQUAT": 0
        }
        
        self.last_rep_time = 0
        self.rep_cooldown = 2.0
        
        # GOOD FORM REQUIREMENTS
        self.exercise_thresholds = {
            "PUSH-UP": {
                "elbow_down": 90,
                "elbow_up": 160,
                "hip_min": 160,
                "hip_max": 180,
                "hip_ideal": 170,
                "hip_tolerance": 15,
                "stage_names": ["UP", "DOWN"],
                "hysteresis": 5,
                "min_down_time": 0.2,
                "min_up_time": 0.2,
                "debug": True,
            },
            "PULL-UP": {
                "elbow_bottom": 150,
                "elbow_top": 80,
                "hip_min": 160,
                "shoulder_range": (40, 100),
                "stage_names": ["BOTTOM", "TOP"],
                "hysteresis": 10,
                "min_top_time": 0.3,
                "min_bottom_time": 0.3,
            },
            "SQUAT": {
                "knee_up": 155,  # Must stand up to 155 deg to complete rep
                "knee_down": 125,  # Must squat to 125 deg - achievable parallel depth
                "hip_up": 150,
                "hip_down": 80,
                "back_min": 0,    # DISABLED - back angle unreliable from front-facing camera
                "back_max": 180,  # DISABLED - was causing silent form violations
                # FIX: Raised from 1.2 to 1.6 — 1.2 was far too strict.
                # Most people naturally squat with knees slightly wider than hips.
                # 1.6 allows knees up to 60% wider than hip width before flagging.
                "max_knee_width_ratio": 1.6,
                "stage_names": ["UP", "DOWN"],
                "hysteresis": 10,
                "min_down_time": 0.2,
                "min_up_time": 0.2,
            }
        }
        
        self.last_form_warning = None
        self.last_form_warning_time = 0
        self.warning_cooldown = 8.0
        self.stage_timer = 0
        self.current_stage_duration = 0
        
        self.exercise_switch_time = 0
        self.switch_cooldown = 1.0
        
        self.last_angle_display = 0
        self.debug_interval = 0.5
        self.last_angles = [0, 0, 0]
        
        # Track form quality for current rep
        self.current_rep_has_good_form = True
        self.form_violations_in_rep = []
        
        # Position validation
        self.in_exercise_position = False
        self.position_check_frames = 0
        self.frames_needed_for_position = 10

    def calculate_angle(self, a, b, c):
        try:
            a = np.array([a.x, a.y]) if hasattr(a, 'x') else np.array(a)
            b = np.array([b.x, b.y]) if hasattr(b, 'x') else np.array(b)
            c = np.array([c.x, c.y]) if hasattr(c, 'x') else np.array(c)
            
            if np.linalg.norm(a - b) < 0.001 or np.linalg.norm(c - b) < 0.001:
                return 180
            
            ba = a - b
            bc = c - b
            
            dot_product = np.dot(ba, bc)
            norm_ba = np.linalg.norm(ba)
            norm_bc = np.linalg.norm(bc)
            
            if norm_ba == 0 or norm_bc == 0:
                return 180
            
            cosine_angle = dot_product / (norm_ba * norm_bc)
            cosine_angle = max(-1.0, min(1.0, cosine_angle))
            angle = np.degrees(np.arccos(cosine_angle))
            
            return angle if not np.isnan(angle) else 180
            
        except Exception as e:
            return 180
    
    def calculate_knee_width_ratio(self, left_hip, right_hip, left_knee, right_knee):
        """
        Calculate knee width relative to hip width.
        
        FIX NOTE: MediaPipe x-coordinates are normalized 0-1 across the frame width.
        When squatting facing the camera, the knee and hip x-coords are used to
        measure lateral spread. This is only meaningful when facing the camera directly.
        
        The ratio compares knee spread to hip spread. A ratio of 1.0 means knees
        are exactly as wide as hips. Ratios above the threshold flag as "too wide".
        """
        try:
            left_hip_x = left_hip.x
            right_hip_x = right_hip.x
            left_knee_x = left_knee.x
            right_knee_x = right_knee.x
            
            hip_width = abs(left_hip_x - right_hip_x)
            knee_width = abs(left_knee_x - right_knee_x)
            
            if hip_width < 0.01:
                return 1.0
            
            ratio = knee_width / hip_width
            return ratio
            
        except Exception as e:
            return 1.0
    
    def analyze_pushup(self, landmarks, thresholds):
        """Pushup analysis - ONLY COUNT GOOD FORM"""
        try:
            left_shoulder = landmarks[11]
            left_elbow = landmarks[13]
            left_wrist = landmarks[15]
            left_hip = landmarks[23]
            left_knee = landmarks[25]
            left_ankle = landmarks[27]
            
            right_shoulder = landmarks[12]
            right_elbow = landmarks[14]
            right_wrist = landmarks[16]
            right_hip = landmarks[24]
            right_knee = landmarks[26]
            right_ankle = landmarks[28]
            
        except:
            return self._get_default_analysis()
        
        left_elbow_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
        left_hip_angle = self.calculate_angle(left_shoulder, left_hip, left_knee)
        left_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
        
        right_elbow_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
        right_hip_angle = self.calculate_angle(right_shoulder, right_hip, right_knee)
        right_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
        
        if abs(left_elbow_angle - 90) < abs(right_elbow_angle - 90):
            elbow_angle = left_elbow_angle
            hip_angle = left_hip_angle
            knee_angle = left_knee_angle
        else:
            elbow_angle = right_elbow_angle
            hip_angle = right_hip_angle
            knee_angle = right_knee_angle
        
        self.elbow_buffer.append(elbow_angle)
        self.hip_buffer.append(hip_angle)
        
        smooth_elbow = np.median(list(self.elbow_buffer))
        smooth_hip = np.median(list(self.hip_buffer))
        smooth_knee = knee_angle
        
        self.last_angles = [smooth_elbow, 0, smooth_hip]
        
        is_in_position = (
            smooth_elbow > 120 and
            smooth_hip > 140 and
            smooth_knee > 140 and
            abs(left_shoulder.y - right_shoulder.y) < 0.2
        )
        
        if is_in_position:
            self.position_check_frames += 1
        else:
            self.position_check_frames = 0
        
        if self.position_check_frames >= self.frames_needed_for_position:
            self.in_exercise_position = True
        
        current_time = time.time()
        if current_time - self.last_angle_display > self.debug_interval:
            print(f"[PUSHUP] Elbow: {int(smooth_elbow)}° | Hip: {int(smooth_hip)}° | Knee: {int(smooth_knee)}° | Position: {self.in_exercise_position}")
            if self.form_violations_in_rep:
                print(f"         Form issues: {', '.join(self.form_violations_in_rep)}")
            self.last_angle_display = current_time
        
        feedback = ""
        audio_feedback = ""
        form_issue = False
        
        if self.stage_timer == 0:
            self.stage_timer = current_time
        self.current_stage_duration = current_time - self.stage_timer
        
        time_since_switch = current_time - self.exercise_switch_time
        if time_since_switch < self.switch_cooldown:
            feedback = f"Get ready... ({int(self.switch_cooldown - time_since_switch)}s)"
            return {
                "angles": [smooth_elbow, 0, smooth_hip],
                "feedback": feedback,
                "audio_feedback": "",
                "rep_counted": False,
                "form_issue": False
            }
        
        hip_deviation = abs(smooth_hip - thresholds["hip_ideal"])
        
        if hip_deviation > thresholds["hip_tolerance"]:
            form_issue = True
            if smooth_hip < thresholds["hip_min"]:
                violation = "hips_low"
                if violation not in self.form_violations_in_rep:
                    self.form_violations_in_rep.append(violation)
                
                if current_time - self.last_form_warning_time > self.warning_cooldown:
                    feedback = "RAISE HIPS - Keep body straight"
                    audio_feedback = "Raise your hips"
                    self.last_form_warning = violation
                    self.last_form_warning_time = current_time
            elif smooth_hip > thresholds["hip_max"]:
                violation = "hips_high"
                if violation not in self.form_violations_in_rep:
                    self.form_violations_in_rep.append(violation)
                
                if current_time - self.last_form_warning_time > self.warning_cooldown:
                    feedback = "LOWER HIPS - Don't pike up"
                    audio_feedback = "Lower your hips"
                    self.last_form_warning = violation
                    self.last_form_warning_time = current_time
        
        if smooth_knee < 150:
            form_issue = True
            violation = "on_knees"
            if violation not in self.form_violations_in_rep:
                self.form_violations_in_rep.append(violation)
            if not feedback and current_time - self.last_form_warning_time > self.warning_cooldown:
                feedback = "GET ON YOUR TOES - No knee pushups"
                audio_feedback = "Get on your toes"
                self.last_form_warning = violation
                self.last_form_warning_time = current_time
        
        if smooth_elbow < thresholds["elbow_down"]:
            if self.stage == "UP":
                self.stage = "DOWN"
                self.stage_timer = current_time
                self.current_rep_has_good_form = True
                self.form_violations_in_rep = []
                if not form_issue:
                    feedback = "Good depth - push up!"
                print(f"✓ PUSHUP DOWN - Elbow: {int(smooth_elbow)}°")
            else:
                if form_issue:
                    self.current_rep_has_good_form = False
        
        elif smooth_elbow > thresholds["elbow_up"]:
            if self.stage == "DOWN":
                if self.current_stage_duration > thresholds["min_down_time"]:
                    if current_time - self.last_rep_time > self.rep_cooldown:
                        self.stage = "UP"
                        self.stage_timer = current_time
                        
                        if self.current_rep_has_good_form and not self.form_violations_in_rep:
                            self.reps += 1
                            self.session_reps["PUSH-UP"] += 1
                            self.last_rep_time = current_time
                            
                            feedback = f"PUSH-UP {self.reps} - EXCELLENT!"
                            audio_feedback = f"Good push up {self.reps}"
                            print(f"✓ Push-up #{self.reps} COUNTED - GOOD FORM!")
                            
                            self.last_form_warning = None
                        else:
                            violations_text = ""
                            if "hips_low" in self.form_violations_in_rep:
                                violations_text = "hips too low"
                            elif "hips_high" in self.form_violations_in_rep:
                                violations_text = "hips too high"
                            elif "on_knees" in self.form_violations_in_rep:
                                violations_text = "on knees"
                            else:
                                violations_text = ', '.join(self.form_violations_in_rep)
                            
                            feedback = f"Rep NOT COUNTED - {violations_text}"
                            audio_feedback = f"Rep not counted. {violations_text}. Keep body straight"
                            print(f"✗ Rep NOT counted - Form violations: {violations_text}")
                        
                        self.current_rep_has_good_form = True
                        self.form_violations_in_rep = []
                    else:
                        if not feedback:
                            feedback = "Good tempo"
            else:
                if form_issue:
                    self.current_rep_has_good_form = False
        
        if not feedback:
            if self.stage == "UP":
                feedback = "Lower down for push-up"
            else:
                feedback = "Push up to complete"
        
        return {
            "angles": [smooth_elbow, 0, smooth_hip],
            "feedback": feedback,
            "audio_feedback": audio_feedback,
            "rep_counted": "EXCELLENT" in feedback and self.reps > 0,
            "form_issue": form_issue
        }
    
    def analyze_pullup(self, landmarks, thresholds):
        """
        Pull-up counter.

        State machine:
          HANG -> arms extended (elbow > 150), form issue cleared while here
          UP   -> chin over bar (elbow < 90)
          HANG -> back to hang = rep complete

        Form checks (latch during rep, clear when back to full hang):
          - Kipping: hip angle < 150 (body swinging)
          - Asymmetry: left/right elbow differ by > 25 deg

        Must reach TOP_ANGLE to count - partial reps don't count.
        """
        try:
            left_shoulder  = landmarks[11]
            left_elbow     = landmarks[13]
            left_wrist     = landmarks[15]
            left_hip       = landmarks[23]
            left_knee      = landmarks[25]
            right_shoulder = landmarks[12]
            right_elbow    = landmarks[14]
            right_wrist    = landmarks[16]
            right_hip      = landmarks[24]
            right_knee     = landmarks[26]
        except:
            return self._get_default_analysis()

        # Angles
        l_elbow = self.calculate_angle(left_shoulder,  left_elbow,  left_wrist)
        r_elbow = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
        l_hip   = self.calculate_angle(left_shoulder,  left_hip,    left_knee)
        r_hip   = self.calculate_angle(right_shoulder, right_hip,   right_knee)

        elbow_angle = (l_elbow + r_elbow) / 2.0
        hip_angle   = (l_hip   + r_hip)   / 2.0
        asym        = abs(l_elbow - r_elbow)

        self.elbow_buffer.append(elbow_angle)
        self.hip_buffer.append(hip_angle)
        smooth_elbow = np.median(list(self.elbow_buffer))
        smooth_hip   = np.median(list(self.hip_buffer))

        current_time = time.time()
        if self.stage_timer == 0:
            self.stage_timer = current_time

        HANG_ANGLE = 150   # arms fully extended
        TOP_ANGLE  = 90    # chin over bar
        HIP_MIN    = 150   # below this = kipping
        ASYM_MAX   = 25    # elbow difference = uneven pull

        if not hasattr(self, '_form_issue'):
            self._form_issue = ""
        if not hasattr(self, '_reached_top'):
            self._reached_top = False

        in_rep = (self.stage == "UP")

        # Check chin over bar: nose y must be above (less than) wrist y
        try:
            nose = landmarks[0]
            avg_wrist_y = (left_wrist.y + right_wrist.y) / 2.0
            chin_over_bar = nose.y < avg_wrist_y
        except:
            chin_over_bar = True  # can't check, don't penalise

        # Track if chin cleared the bar at any point during this pull
        if in_rep and chin_over_bar:
            self._reached_top = True

        # Latch form issues during the pull
        if in_rep:
            if smooth_hip < HIP_MIN and not self._form_issue:
                self._form_issue = "kipping"
            elif asym > ASYM_MAX and not self._form_issue:
                self._form_issue = "uneven pull"

        # Debug
        if current_time - self.last_angle_display > self.debug_interval:
            print(f"[PULLUP] Elbow:{smooth_elbow:.1f} Hip:{smooth_hip:.1f} Asym:{asym:.1f} Stage:{self.stage} Reps:{self.reps}")
            print(f"         Issue:'{self._form_issue}'")
            self.last_angle_display = current_time

        # Switch cooldown
        if current_time - self.exercise_switch_time < self.switch_cooldown:
            self.stage = "HANG"
            self._form_issue = ""
            self._reached_top = False
            rem = self.switch_cooldown - (current_time - self.exercise_switch_time)
            return {"angles": [smooth_elbow, 0, smooth_hip], "feedback": f"Get ready... ({rem:.0f}s)",
                    "audio_feedback": "", "rep_counted": False, "form_issue": False}

        feedback = ""
        audio_feedback = ""
        rep_counted = False
        form_issue = False

        # ── STATE MACHINE ──────────────────────────────────────────

        # ── HANGING VALIDATION ────────────────────────────────────
        # Wrists must be ABOVE shoulders (lower y = higher in frame).
        # If not, person is not on a bar - ignore all rep counting.
        avg_wrist_y    = (left_wrist.y  + right_wrist.y)  / 2.0
        avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2.0
        is_hanging = avg_wrist_y < avg_shoulder_y  # wrists higher than shoulders

        if not is_hanging:
            return {"angles": [smooth_elbow, 0, smooth_hip],
                    "feedback": "Get on the bar - wrists must be above shoulders",
                    "audio_feedback": "", "rep_counted": False, "form_issue": False}

        if self.stage == "HANG":
            # While hanging with arms extended, clear the form flag
            if smooth_elbow >= HANG_ANGLE:
                self._form_issue = ""

            if smooth_elbow <= TOP_ANGLE:
                # Reached the top - chin over bar
                self.stage = "UP"
                self.stage_timer = current_time
                feedback = "Good! Lower with control."
                print(f"  -> TOP at {smooth_elbow:.1f}")
            else:
                # Live form warnings while on the way up
                if smooth_hip < HIP_MIN and current_time - self.last_form_warning_time > 3.0:
                    feedback = "NO KIPPING - keep body still!"
                    self.last_form_warning_time = current_time
                    form_issue = True
                elif asym > ASYM_MAX and current_time - self.last_form_warning_time > 3.0:
                    feedback = "PULL EVENLY - both arms!"
                    self.last_form_warning_time = current_time
                    form_issue = True
                if not feedback:
                    if smooth_elbow >= HANG_ANGLE:
                        feedback = "Pull up!"
                    else:
                        feedback = f"Keep pulling! ({smooth_elbow:.0f} deg)"

        elif self.stage == "UP":
            if smooth_elbow >= HANG_ANGLE:
                # Back to full hang - evaluate the rep
                issue = self._form_issue
                self._form_issue = ""
                self.stage = "HANG"
                self.stage_timer = current_time

                # Also check chin-over-bar was reached during this rep
                if not self._reached_top and not issue:
                    issue = "chin did not clear bar"
                reached = self._reached_top
                self._reached_top = False  # reset for next rep

                if current_time - self.last_rep_time > 1.0:
                    self.last_rep_time = current_time
                    if issue:
                        form_issue = True
                        feedback = f"NOT COUNTED - {issue}!"
                        audio_feedback = f"Rep not counted. {issue}."
                        print(f"  -> NOT COUNTED: {issue}")
                    else:
                        self.reps += 1
                        self.session_reps["PULL-UP"] += 1
                        rep_counted = True
                        feedback = f"PULL-UP {self.reps} - EXCELLENT!"
                        audio_feedback = f"Good pull up {self.reps}"
                        print(f"  -> REP #{self.reps} COUNTED!")
            else:
                # Still at top or lowering
                if smooth_elbow <= TOP_ANGLE:
                    feedback = "Lower with control."
                else:
                    feedback = f"All the way down! ({smooth_elbow:.0f} deg)"

        return {"angles": [smooth_elbow, 0, smooth_hip], "feedback": feedback,
                "audio_feedback": audio_feedback, "rep_counted": rep_counted, "form_issue": form_issue}

    def analyze_squat(self, landmarks, thresholds):
        try:
            left_hip    = landmarks[23]
            left_knee   = landmarks[25]
            left_ankle  = landmarks[27]
            right_hip   = landmarks[24]
            right_knee  = landmarks[26]
            right_ankle = landmarks[28]
        except:
            return self._get_default_analysis()

        lk = self.calculate_angle(left_hip,  left_knee,  left_ankle)
        rk = self.calculate_angle(right_hip, right_knee, right_ankle)
        self.elbow_buffer.append((lk + rk) / 2.0)
        smooth_knee = np.median(list(self.elbow_buffer))

        current_time = time.time()
        if self.stage_timer == 0:
            self.stage_timer = current_time

        KNEE_STANDING = 145
        KNEE_DOWN = thresholds["knee_down"]  # 125
        KNEE_UP   = thresholds["knee_up"]    # 155
        CAVE_MARGIN = 0.10
        FLARE_RATIO = 2.2

        if not hasattr(self, '_form_issue'):
            self._form_issue = ""

        # Cave / flare — only latched while in BOTTOM stage
        in_squat   = (self.stage == "BOTTOM")
        hip_width  = abs(left_hip.x  - right_hip.x)
        knee_width = abs(left_knee.x - right_knee.x)
        knees_caving  = in_squat and (
            left_knee.x  < (left_hip.x  - CAVE_MARGIN) or
            right_knee.x > (right_hip.x + CAVE_MARGIN)
        )
        knees_flaring = in_squat and (hip_width > 0.01) and (knee_width > hip_width * FLARE_RATIO)

        if in_squat and knees_caving  and not self._form_issue:
            self._form_issue = "knees caving in"
        if in_squat and knees_flaring and not self._form_issue:
            self._form_issue = "knees flaring out"

        # Debug
        if current_time - self.last_angle_display > self.debug_interval:
            print(f"[SQUAT] Knee:{smooth_knee:.1f} Stage:{self.stage} Reps:{self.reps} Issue:'{self._form_issue}'")
            print(f"        HipW:{hip_width:.3f} KneeW:{knee_width:.3f} Cave:{knees_caving} Flare:{knees_flaring}")
            self.last_angle_display = current_time

        # Switch cooldown
        if current_time - self.exercise_switch_time < self.switch_cooldown:
            self.stage = "WAITING"
            self._form_issue = ""
            rem = self.switch_cooldown - (current_time - self.exercise_switch_time)
            return {"angles": [smooth_knee, 0, 0], "feedback": f"Get ready... ({rem:.0f}s)",
                    "audio_feedback": "", "rep_counted": False, "form_issue": False}

        feedback = ""
        audio_feedback = ""
        rep_counted = False
        form_issue = False

        # ── STATE MACHINE ─────────────────────────────────────────

        if self.stage == "WAITING":
            self._form_issue = ""
            if smooth_knee >= KNEE_STANDING:
                self.stage = "ARMED"
                self.stage_timer = current_time
                feedback = "Ready! Squat down!"
                print(f"  -> ARMED at {smooth_knee:.1f}")
            else:
                feedback = f"Stand up straight! ({smooth_knee:.0f} need {KNEE_STANDING})"

        elif self.stage == "ARMED":
            # Clear form issue only while fully upright — guarantees clean slate
            if smooth_knee >= KNEE_UP:
                self._form_issue = ""
            if smooth_knee <= KNEE_DOWN:
                self.stage = "BOTTOM"
                self.stage_timer = current_time
                feedback = "Good depth! Stand up!"
                print(f"  -> BOTTOM at {smooth_knee:.1f}")
            else:
                feedback = "Squat down!" if smooth_knee >= KNEE_UP else f"Deeper! ({smooth_knee:.0f} need <{KNEE_DOWN})"

        elif self.stage == "BOTTOM":
            if smooth_knee >= KNEE_UP:
                issue = self._form_issue   # snapshot before clearing
                self._form_issue = ""      # clear immediately
                self.stage = "ARMED"
                self.stage_timer = current_time

                if current_time - self.last_rep_time > 1.0:
                    self.last_rep_time = current_time
                    if issue:
                        form_issue = True
                        feedback = f"NOT COUNTED - {issue}!"
                        audio_feedback = f"Rep not counted. {issue}."
                        print(f"  -> NOT COUNTED: {issue}")
                    else:
                        self.reps += 1
                        self.session_reps["SQUAT"] += 1
                        rep_counted = True
                        feedback = f"SQUAT {self.reps} - EXCELLENT!"
                        audio_feedback = f"Good squat {self.reps}"
                        print(f"  -> REP #{self.reps} COUNTED!")
            else:
                # Live warning while squatting
                if knees_caving and current_time - self.last_form_warning_time > 3.0:
                    feedback = "KNEES CAVING IN - push out!"
                    self.last_form_warning_time = current_time
                    form_issue = True
                elif knees_flaring and current_time - self.last_form_warning_time > 3.0:
                    feedback = "KNEES FLARING OUT - bring in!"
                    self.last_form_warning_time = current_time
                    form_issue = True
                if not feedback:
                    feedback = f"Stand up! ({smooth_knee:.0f})"

        return {"angles": [smooth_knee, 0, 0], "feedback": feedback,
                "audio_feedback": audio_feedback, "rep_counted": rep_counted, "form_issue": form_issue}

    def _get_default_analysis(self):
        if not self.session_active:
            return {
                "angles": [180, 180, 180],
                "feedback": "Press 'S' to START session",
                "audio_feedback": "",
                "rep_counted": False,
                "form_issue": False
            }
        
        return {
            "angles": [180, 180, 180],
            "feedback": "Adjust position - ensure full body is visible",
            "audio_feedback": "",
            "rep_counted": False,
            "form_issue": False
        }
    
    def analyze_form(self, landmarks, exercise_name):
        if not landmarks:
            return self._get_default_analysis()
        
        thresholds = self.exercise_thresholds[exercise_name]
        
        if not self.session_active:
            return self._get_default_analysis()
        
        if exercise_name == "PUSH-UP":
            return self.analyze_pushup(landmarks, thresholds)
        elif exercise_name == "PULL-UP":
            return self.analyze_pullup(landmarks, thresholds)
        elif exercise_name == "SQUAT":
            return self.analyze_squat(landmarks, thresholds)
        
        return self._get_default_analysis()
    
    def switch_exercise(self, new_index=None):
        if new_index is None:
            self.current_exercise = (self.current_exercise + 1) % len(self.exercises)
        else:
            self.current_exercise = new_index % len(self.exercises)
        
        self.reps = 0
        self.stage = "TOP" if self.exercises[self.current_exercise] == "PULL-UP" else "UP"
        self.stage_timer = 0
        self.exercise_switch_time = time.time()
        
        self.elbow_buffer.clear()
        self.shoulder_buffer.clear()
        self.hip_buffer.clear()
        self.knee_width_buffer.clear()
        self.last_form_warning = None
        self.last_form_warning_time = 0
        self.current_rep_has_good_form = True
        self.form_violations_in_rep = []
        
        self.in_exercise_position = False
        self.position_check_frames = 0
        
        print(f"Switched to {self.exercises[self.current_exercise]}")
        return self.exercises[self.current_exercise]
    
    def reset_current_exercise(self):
        self.reps = 0
        print(f"Current {self.exercises[self.current_exercise]} counter reset to 0")
        return True
    
    def reset_session(self):
        for exercise in self.session_reps:
            self.session_reps[exercise] = 0
        self.reps = 0
        print("ALL session counters reset to 0")
        return True

# =========================================================
# UI DRAWING
# =========================================================
def draw_ui(frame, form_analyzer, audio, width, height):
    cv2.rectangle(frame, (20, 20), (400, 300), (25, 25, 40), -1)
    cv2.rectangle(frame, (20, 20), (400, 300), (0, 200, 255), 2)
    
    status_text = "ACTIVE" if form_analyzer.session_active else "INACTIVE"
    status_color = (0, 255, 0) if form_analyzer.session_active else (0, 100, 255)
    cv2.putText(frame, f"Status: {status_text}", 
               (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    
    current_exercise = form_analyzer.exercises[form_analyzer.current_exercise]
    cv2.putText(frame, f"Exercise: {current_exercise}", 
               (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)
    
    cv2.putText(frame, f"REPS: {form_analyzer.reps}",
               (40, 160), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 255, 0), 3)
    
    current_session_reps = form_analyzer.session_reps[current_exercise]
    cv2.putText(frame, f"Session: {current_session_reps}",
               (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 255, 150), 2)
    
    audio_status = "MUTED" if audio.muted else "UNMUTED"
    audio_color = (100, 100, 100) if audio.muted else (0, 255, 0)
    cv2.putText(frame, f"Audio: {audio_status}",
               (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, audio_color, 1)
    
    cv2.putText(frame, "GOOD FORM ONLY!",
               (40, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    panel_width = 350
    panel_height = 290
    panel_x = width - panel_width - 20
    panel_y = 20
    
    cv2.rectangle(frame, (panel_x, panel_y), (width - 20, panel_y + panel_height), (40, 25, 25), -1)
    cv2.rectangle(frame, (panel_x, panel_y), (width - 20, panel_y + panel_height), (0, 200, 255), 2)
    
    cv2.putText(frame, "SESSION TOTALS",
               (panel_x + 20, panel_y + 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)
    
    y_offset = panel_y + 90
    for i, exercise in enumerate(form_analyzer.exercises):
        count = form_analyzer.session_reps[exercise]
        
        if i == form_analyzer.current_exercise:
            color = (0, 255, 0)
            cv2.putText(frame, ">", (panel_x + 20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        else:
            color = (200, 200, 200)
        
        cv2.putText(frame, f"{exercise}:", 
                   (panel_x + 50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"{count}", 
                   (panel_x + 280, y_offset), cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2)
        
        y_offset += 50
    
    total_reps = sum(form_analyzer.session_reps.values())
    cv2.line(frame, (panel_x + 20, y_offset), (width - 40, y_offset), (0, 200, 255), 2)
    y_offset += 40
    cv2.putText(frame, "TOTAL:", 
               (panel_x + 50, y_offset), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, f"{total_reps}", 
               (panel_x + 250, y_offset), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 0), 3)
    
    req_panel_width = 400
    req_panel_height = 230
    req_panel_x = width - req_panel_width - 20
    req_panel_y = height - req_panel_height - 20
    
    cv2.rectangle(frame, (req_panel_x, req_panel_y), (width - 20, height - 20), (25, 40, 25), -1)
    cv2.rectangle(frame, (req_panel_x, req_panel_y), (width - 20, height - 20), (0, 255, 100), 2)
    
    current_exercise = form_analyzer.exercises[form_analyzer.current_exercise]
    
    cv2.putText(frame, f"{current_exercise} REQUIREMENTS",
               (req_panel_x + 20, req_panel_y + 35), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 100), 2)
    
    req_y = req_panel_y + 70
    line_height = 30
    
    if current_exercise == "PUSH-UP":
        requirements = [
            "1) Down: Elbow < 90 deg",
            "2) Up: Elbow > 160 deg",
            "3) Body: 160-180 deg (straight)",
            "4) No sagging or piking",
            "5) Stand SIDEWAYS to camera"
        ]
    elif current_exercise == "PULL-UP":
        requirements = [
            "1) Top: Elbow < 80 deg",
            "2) Bottom: Elbow > 150 deg",
            "3) Body: > 160 deg (no kipping)",
            "4) Keep legs still",
            "5) Chin over bar at top",
            "6) Pull evenly with both arms"
        ]
    else:  # SQUAT
        requirements = [
            "1) Up: Knee > 150 deg (standing)",
            "2) Down: Knee < 110 deg (parallel)",
            "3) Chest up (back 60-120 deg)",
            "4) Knees: hip-width OK (ratio < 1.6)",
            "5) Face camera directly"
        ]
    
    for req in requirements:
        cv2.putText(frame, req,
                   (req_panel_x + 20, req_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1)
        req_y += line_height
    
    return frame

# =========================================================
# MAIN APPLICATION
# =========================================================
def main():
    print("\n" + "=" * 60)
    print("GOOD FORM ONLY - REP COUNTER")
    print("=" * 60)
    
    audio = ExerciseAudioFeedback()
    form_analyzer = ExerciseFormAnalyzer()
    
    print("\nInitializing camera...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("✗ Cannot open camera!")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✓ Camera ready: {width}x{height}")
    
    window_name = "Exercise Form Trainer - Good Form Only"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)
    
    print("\n" + "=" * 60)
    print("SQUAT FIX APPLIED:")
    print("  • Knee width threshold raised: 1.2 → 1.6")
    print("  • Live knee width ratio shown on screen + terminal")
    print("  • Shows ratio/threshold so you can tune further")
    print("\nCONTROLS:")
    print("  S = Start/Stop session")
    print("  M = Mute/Unmute audio")
    print("  1/2/3 = Switch exercise")
    print("  SPACE = Next exercise")
    print("  R = Reset current counter")
    print("  T = Reset ALL session counters")
    print("  ESC or Q = Quit")
    print("=" * 60)
    
    frame_count = 0
    start_time = time.time()
    
    try:
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
            audio_feedback = ""
            
            if results.pose_landmarks:
                # Person is visible - reset the lost-frame counter
                form_analyzer.lost_frames = 0

                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
                
                current_exercise_name = form_analyzer.exercises[form_analyzer.current_exercise]
                form_data = form_analyzer.analyze_form(results.pose_landmarks.landmark, current_exercise_name)
            else:
                # Person not detected - count lost frames
                form_analyzer.lost_frames = getattr(form_analyzer, 'lost_frames', 0) + 1
                # After 10 lost frames (~0.3s) reset squat state so it restarts cleanly
                if form_analyzer.lost_frames == 10:
                    form_analyzer.stage = "WAITING"
                    form_analyzer.elbow_buffer.clear()
                    form_analyzer.hip_buffer.clear()
                    form_analyzer.shoulder_buffer.clear()
                    form_analyzer.stage_timer = 0
                    form_analyzer._rep_form_issue = ""
                    print("  [RESET] Lost tracking - stage reset to WAITING")
                feedback = "Step back into frame..."
            
            # Safely extract feedback from form_data if available
            if form_data:
                feedback = form_data["feedback"]
                audio_feedback = form_data.get("audio_feedback", "")

            if audio_feedback and form_analyzer.session_active and audio.enabled and not audio.muted:
                    if "not counted" in audio_feedback.lower():
                        audio.speak(audio_feedback, force=True)
                    elif "Good" in audio_feedback or "Excellent" in audio_feedback:
                        audio.speak(audio_feedback, force=True)
                    elif form_data.get("form_issue", False):
                        audio.speak(audio_feedback)
            
            frame = draw_ui(frame, form_analyzer, audio, width, height)
            
            if feedback:
                fb_color = (0, 255, 0) if "EXCELLENT" in feedback else (0, 100, 255) if "NOT COUNTED" in feedback else (255, 255, 0)
                
                text_size = cv2.getTextSize(feedback, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                fb_bg_x = max(20, width // 2 - text_size[0] // 2 - 20)
                fb_bg_y = height - 80
                
                cv2.rectangle(frame,
                             (fb_bg_x, fb_bg_y),
                             (fb_bg_x + text_size[0] + 40, fb_bg_y + text_size[1] + 20),
                             (20, 20, 20), -1)
                cv2.rectangle(frame,
                             (fb_bg_x, fb_bg_y),
                             (fb_bg_x + text_size[0] + 40, fb_bg_y + text_size[1] + 20),
                             fb_color, 2)
                
                cv2.putText(frame, feedback,
                           (fb_bg_x + 20, fb_bg_y + text_size[1] + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, fb_color, 2)
            
            cv2.putText(frame, f"FPS: {int(fps)}",
                       (width - 100, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 200, 255), 1)
            
            cv2.imshow(window_name, frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27 or key == ord('q'):
                break
            elif key == ord('s'):
                form_analyzer.session_active = not form_analyzer.session_active
                status = "STARTED" if form_analyzer.session_active else "PAUSED"
                print(f"\nSession {status}")
                if audio.enabled and not audio.muted:
                    audio.speak(f"Session {status.lower()}", priority="startup")
            elif key == ord('m'):
                audio.toggle_mute()
            elif key == ord('1'):
                new_exercise = form_analyzer.switch_exercise(0)
                print(f"\nSwitched to {new_exercise}")
                if audio.enabled and not audio.muted:
                    audio.speak(f"Push up mode", priority="startup")
            elif key == ord('2'):
                new_exercise = form_analyzer.switch_exercise(1)
                print(f"\nSwitched to {new_exercise}")
                if audio.enabled and not audio.muted:
                    audio.speak(f"Pull up mode", priority="startup")
            elif key == ord('3'):
                new_exercise = form_analyzer.switch_exercise(2)
                print(f"\nSwitched to {new_exercise}")
                if audio.enabled and not audio.muted:
                    audio.speak(f"Squat mode", priority="startup")
            elif key == ord(' '):
                new_exercise = form_analyzer.switch_exercise()
                print(f"\nSwitched to {new_exercise}")
            elif key == ord('r'):
                if form_analyzer.reset_current_exercise():
                    if audio.enabled and not audio.muted:
                        audio.speak("Counter reset", priority="startup")
            elif key == ord('t'):
                if form_analyzer.reset_session():
                    if audio.enabled and not audio.muted:
                        audio.speak("All counters reset", priority="startup")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        audio.cleanup()
    
    print(f"\n" + "=" * 60)
    print("SESSION COMPLETE - FINAL RESULTS:")
    print("-" * 60)
    total_reps = 0
    for exercise, reps in form_analyzer.session_reps.items():
        if reps > 0:
            print(f"  {exercise}: {reps} reps")
            total_reps += reps
    print("-" * 60)
    print(f"  TOTAL REPS: {total_reps}")
    print("=" * 60)

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("EXERCISE FORM TRAINER - GOOD FORM ONLY")
    print("Reps only count with proper form!")
    print("=" * 60)
    
    input("\nPress Enter to start...")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram stopped by user")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
