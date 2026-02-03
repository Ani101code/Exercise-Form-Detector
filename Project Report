1. Introduction
Incorrect exercise technique is a major contributor to training-related injuries and inefficient movement patterns. Beginners in particular often struggle to maintain correct form without continuous supervision. Traditional fitness feedback methods rely heavily on mirrors or visual displays, which require constant visual attention and may disrupt movement execution.
This project investigates the use of real-time computer vision and pose estimation to automatically analyse exercise form and provide instant multimodal feedback, combining on-screen visual cues with audio feedback. The system focuses on push-ups, pull-ups, and squats and only counts repetitions performed with acceptable biomechanical form, encouraging quality over quantity.
 
2. Research Question
To what extent can a computer vision–based system using pose estimation and multimodal (visual and audio) feedback accurately detect incorrect exercise form, improve movement quality in real time and reduce the risk of exercise-related injury?
 
3. System Overview
The system is implemented in Python using:
• OpenCV for video capture and real-time rendering.
• MediaPipe Pose for 2D human pose estimation.
• Text-to-Speech (TTS) for real-time audio feedback.
• Multithreading to ensure non-blocking audio output.
A webcam captures live video, from which MediaPipe extracts body landmarks. Joint angles and body alignment metrics are calculated and evaluated against predefined biomechanical thresholds. Based on this analysis, the system provides immediate feedback and additionally counts repetitions.
 
4. Pose Estimation and Angle Calculation
MediaPipe Pose provides normalized landmark coordinates for major joints. Joint angles are calculated using vector geometry:
• Elbow angle (shoulder–elbow–wrist).
• Hip angle (shoulder–hip–knee).
• Knee angle (hip–knee–ankle).
• Torso angle relative to vertical (for squats)..
To reduce noise and false positives, temporal smoothing is applied using a median filter over multiple frames. This ensures stable angle values even with minor landmark jitter.
 
5. Exercise-Specific Form Analysis
5.1 Push-Up Form Analysis
For push-ups, the system analyzes:
• Elbow flexion angle to detect sufficient depth and lockout.
• Hip angle to assess body alignment (plank position).
• Knee angle to ensure the user is on their toes.
Detected form errors include:
• Sagging hips (lumbar extension).
• Excessive hip elevation (piking).
• Performing push-ups on the knees.
• Insufficient elbow flexion depth.
A repetition is only counted if:
• Elbow angle passes both the down and up thresholds.
• Body alignment remains within acceptable limits.
• No form violations persist during the repetition.
If incorrect form persists across multiple frames, an audio warning is triggered. Repetitions with poor form are explicitly not counted, reinforcing correct technique.
 
5.2 Pull-Up Form Analysis
For pull-ups, the system evaluates:
• Elbow angle to detect top and bottom positions.
• Hip angle to detect excessive swinging (kipping).
• Shoulder–elbow relationship to ensure controlled movement.
Detected errors include:
• Kipping (hip swing).
• Incomplete range of motion.
• Uncontrolled descent.
Only repetitions performed with controlled motion and minimal hip movement are counted. Audio feedback is provided to discourage momentum-based repetitions.
 
5.3 Squat Form Analysis
For squats, the system analyzes:
• Knee flexion angle to determine squat depth.
• Hip angle for movement consistency.
• Torso angle relative to vertical to assess spinal posture.
• Knee alignment relative to hip width to detect valgus collapse.
Detected errors include:
• Insufficient squat depth.
• Excessive forward or upright torso lean.
• Knee valgus (knees collapsing inward).
The system allows natural variation but rejects repetitions with biomechanically risky patterns. Audio cues are issued when errors persist, especially at the bottom of the squat where injury risk is highest.
 
6. Multimodal Feedback System
6.1 Visual Feedback
Visual feedback includes:
• Real-time skeleton overlay
• On-screen textual cues
• Live repetition counters
• Exercise-specific form requirements
• Session totals and status indicators
This allows users to understand form requirements before and during exercise.
 
6.2 Audio Feedback
To reduce reliance on constant screen monitoring, the system integrates real-time audio feedback using text-to-speech synthesis.
Key features include:
• Non-blocking audio processing via a dedicated thread.
• Feedback prioritization (errors > successful reps).
• Cooldown timers to avoid repetitive or distracting cues.
• Immediate warnings for uncounted repetitions.
Audio feedback enables users to maintain focus on movement execution rather than visual checking, particularly during physically demanding sets.
 
7. Evaluation and Metrics
The system is evaluated using both quantitative and qualitative criteria:
• Pose detection stability: consistency of landmark tracking across frames.
• Form classification accuracy: correct identification of valid vs invalid repetitions.
• Latency: time between error detection and feedback delivery.
• User response: observable improvement in form after feedback.
• Usability: reduced need for visual attention during exercise.
Comparisons are made between visual-only feedback and combined visual–audio feedback to assess effectiveness.
 
8. Injury Risk Reduction
While direct injury prevention cannot be conclusively measured without long-term clinical studies, the system targets known biomechanical risk factors, including:
• Poor joint alignment
• Excessive spinal deviation
• Momentum-based movement patterns
By rejecting repetitions with unsafe form and providing immediate corrective feedback, the system likely contributes to a reduction in injury risk, particularly for beginners and unsupervised training environments.
 
9. Limitations
• The system relies on 2D pose estimation, which may be less accurate for depth-related movements.
• Lighting and camera angle significantly affect landmark accuracy.
• Individual anatomical differences are not fully personalized.
 
10. Conclusion
This project demonstrates that a computer vision–based system using pose estimation and multimodal feedback can effectively detect incorrect exercise form and improve movement quality in real time. The integration of audio feedback reduces reliance on visual attention and enhances usability during exercise.
Although direct injury prevention cannot be conclusively proven, the system successfully minimizes biomechanical risk factors, suggesting a meaningful contribution to safer and more effective training.
