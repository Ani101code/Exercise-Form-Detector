# References and Related Work

This list captures publications, book chapters, and authoritative websites that are relevant to the current implementation (OpenCV + MediaPipe Pose + real-time form feedback for push-up/pull-up/squat analysis).

## 1) Core pose-estimation and computer-vision publications

1. Lugaresi, C., Tang, J., Nash, H., et al. (2019). **MediaPipe: A Framework for Building Perception Pipelines**. arXiv:1906.08172.  
   Relevance: foundational framework behind the real-time perception pipeline used here.

2. Bazarevsky, V., Grishchenko, I., Raveendran, K., et al. (2020). **BlazePose: On-device Real-time Body Pose tracking** (Google AI Blog / technical release notes and model card ecosystem).  
   Relevance: practical architecture family underlying MediaPipe Pose real-time body landmarking.

3. Cao, Z., Hidalgo, G., Simon, T., Wei, S.-E., & Sheikh, Y. (2021). **OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields**. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 43(1), 172–186. https://doi.org/10.1109/TPAMI.2019.2929257  
   Relevance: landmark baseline in real-time 2D pose estimation literature.

4. Toshev, A., & Szegedy, C. (2014). **DeepPose: Human Pose Estimation via Deep Neural Networks**. *CVPR 2014*. https://doi.org/10.1109/CVPR.2014.214  
   Relevance: early deep-learning approach for articulated pose estimation.

5. Newell, A., Yang, K., & Deng, J. (2016). **Stacked Hourglass Networks for Human Pose Estimation**. *ECCV 2016*. https://doi.org/10.1007/978-3-319-46484-8_29  
   Relevance: influential architecture for keypoint localization.

## 2) Biomechanics and training references (for threshold rationale)

6. American College of Sports Medicine (ACSM). **ACSM's Guidelines for Exercise Testing and Prescription** (latest edition).  
   Relevance: practical standards for exercise technique and safety cues.

7. NSCA (National Strength and Conditioning Association). **Essentials of Strength Training and Conditioning** (latest edition).  
   Relevance: movement-quality and coaching principles for squat, push-up, and pull-up form.

## 3) Documentation / websites used to guide implementation

8. OpenCV Documentation: https://docs.opencv.org/  
    Relevance: video capture, frame processing, and on-screen rendering.

9. MediaPipe Pose Documentation: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker  
    Relevance: pose landmarks, confidence settings, and runtime behavior.

10. NumPy Documentation: https://numpy.org/doc/  
    Relevance: vectorized angle computations.

11. Python `threading` and `queue` documentation:  
    https://docs.python.org/3/library/threading.html  
    https://docs.python.org/3/library/queue.html  
    Relevance: non-blocking TTS worker pattern.

12. pyttsx3 Documentation: https://pyttsx3.readthedocs.io/  
    Relevance: cross-platform text-to-speech fallback.

13. Microsoft SAPI / `win32com` references:  
    https://learn.microsoft.com/ (Speech API and COM automation docs)  
    Relevance: Windows-native TTS path used by the project.
