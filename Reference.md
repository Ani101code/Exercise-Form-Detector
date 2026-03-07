# References and Related Work

This bibliography is split into two groups:
- 1) Technical references (pose estimation, CV pipelines, implementation stack), and  
- 2) Anatomical / biomechanics references (exercise technique, joint angles, muscle activation, EMG) for push-ups, pull-ups/chin-ups, and squats.

---

# A) Technical References (Computer Vision / Pose Estimation / Implementation)

1. Lugaresi, C., Tang, J., Nash, H., et al. (2019). *MediaPipe: A Framework for Building Perception Pipelines.* arXiv:1906.08172.  
   https://arxiv.org/abs/1906.08172

2. Bazarevsky, V., Grishchenko, I., Raveendran, K., et al. (2020). *BlazePose: On-device Real-time Body Pose Tracking* (Google AI technical release ecosystem).  
   https://ai.googleblog.com/2020/08/on-device-real-time-body-pose-tracking.html

3. Cao, Z., Hidalgo, G., Simon, T., Wei, S.-E., & Sheikh, Y. (2021). *OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields.* IEEE TPAMI, 43(1), 172–186.  
   https://doi.org/10.1109/TPAMI.2019.2929257

4. Toshev, A., & Szegedy, C. (2014). *DeepPose: Human Pose Estimation via Deep Neural Networks.* CVPR.  
   https://doi.org/10.1109/CVPR.2014.214

5. Newell, A., Yang, K., & Deng, J. (2016). *Stacked Hourglass Networks for Human Pose Estimation.* ECCV.  
   https://doi.org/10.1007/978-3-319-46484-8_29

6. OpenCV documentation.  
   https://docs.opencv.org/

7. MediaPipe Pose / Pose Landmarker documentation.  
   https://developers.google.com/mediapipe/solutions/vision/pose_landmarker

8. NumPy documentation.  
   https://numpy.org/doc/

9. Python threading documentation.  
   https://docs.python.org/3/library/threading.html

10. Python queue documentation.  
    https://docs.python.org/3/library/queue.html

11. pyttsx3 documentation.  
    https://pyttsx3.readthedocs.io/

12. Microsoft SAPI/COM documentation (win32com context).  
    https://learn.microsoft.com/en-us/previous-versions/windows/desktop/ms723602(v=vs.85)

---

# B) Anatomical, Biomechanics, and EMG References (Exercise Form Logic)

## B1) Foundational Biomechanics / Kinesiology

13. American College of Sports Medicine. (2021). *ACSM's Guidelines for Exercise Testing and Prescription* (12th ed.). Wolters Kluwer.  
    https://shop.lww.com/ACSM-s-Guidelines-for-Exercise-Testing-and-Prescription/p/9781975150181

---

## B2) Push-Up

14. Ay, A. et al. (2015). *The effects of exercise type and elbow angle on vertical ground reaction force and muscle activity during a push-up plus exercise.* PMC / PLOS ONE.  
    **[PU-1 — 90° elbow depth criterion]**  
    https://pmc.ncbi.nlm.nih.gov/articles/PMC4327800/

15. Donatto, F. et al. *Biomechanical Analysis of the Elbow Joint Loading During Push-Up.* ResearchGate.  
    **[PU-2 — elbow extension at top, ~160–180°]**  
    https://www.researchgate.net/publication/247915095_BIOMECHANICAL_ANALYSIS_OF_THE_ELBOW_JOINT_LOADING_DURING_PUSH-UP

16. Winter, D.A. (1993). *Hand position affects elbow joint load during push-up exercise.* Journal of Orthopaedic & Sports Physical Therapy. PubMed.  
    **[PU-2 — elbow extension at top]**  
    https://pubmed.ncbi.nlm.nih.gov/8514808/

17. Seedman, J. (2015). *Tip: The Worst Way to Do Push-Ups.* Advanced Human Performance.  
    **[PU-3 — body alignment / hip angle, plank position 160–180°]**  
    https://www.advancedhumanperformance.com/tip-the-worst-way-to-do-pushups

18. Calatayud, J. et al. (2012). *The Biomechanics of the Push-up.* ResearchGate.  
    **[PU-3 — body alignment / hip angle]**  
    https://www.researchgate.net/publication/271794661_The_Biomechanics_of_the_Push-up

---

## B3) Pull-Up / Chin-Up

19. Seedman, J. (2020). *Pullups & Pulldowns: The Right & Wrong Way.* Advanced Human Performance.  
    **[PL-1 — ~90° elbow at top position]**  
    https://www.advancedhumanperformance.com/blog/pullups-best-technique

20. Youdas, J.W. et al. (2010). *Surface electromyographic activation patterns and elbow joint motion during a pull-up, chin-up, or Perfect-Pullup rotational exercise.* Journal of Strength and Conditioning Research. PubMed.  
    **[PL-2 — full-extension dead hang >150°]**  
    https://pubmed.ncbi.nlm.nih.gov/21068680/

21. ExRx.net. *Muscular Analysis of Pull-ups and Chin-ups.*  
    **[PL-2 — full-extension dead hang >150°]**  
    https://exrx.net/Questions/PullupAnalysis

22. Gymshark (2026). *Pull-Ups: How to Do Them and Get Your First Rep.*  
    **[PL-3 — chin over bar criterion]**  
    https://www.gymshark.com/blog/article/how-to-do-pull-ups

23. Elio's Health (2023). *Biomechanics of the Pull-Up.*  
    **[PL-3 — chin over bar criterion]**  
    https://www.elioshealth.com/blog/pull-up

24. Urbanczyk, C.A. et al. (2020). *Avoiding high-risk rotator cuff loading: Muscle force during three pull-up techniques.* Scandinavian Journal of Medicine & Science in Sports. Via Hooper's Beta.  
    **[PL-4 — kipping & asymmetry as form violations]**  
    https://www.hoopersbeta.com/library/how-pullups-work-wide-grip-standard-chin-up-biomechanics

25. Dickie, J.A. et al. (2017). *Electromyographic analysis of muscle activation during pull-up variations.* Journal of Electromyography and Kinesiology. ScienceDirect.  
    **[PL-4 — kipping & asymmetry as form violations]**  
    https://www.sciencedirect.com/science/article/am/pii/S0765159722001320

---

## B4) Squat

26. Muscle Evo (2023). *Parallel Squats vs Deep vs 90 Degrees: How Low to Go?* (citing Escamilla RF et al. review of 70 squat studies).  
    **[SQ-1 — parallel depth = ~125° of knee flexion]**  
    https://muscleevo.net/how-deep-should-you-squat/

27. Escamilla, R.F. (2001). *Knee biomechanics of the dynamic squat exercise.* Medicine & Science in Sports & Exercise. Via IJSPT review.  
    **[SQ-1 — parallel depth = ~125° of knee flexion]**  
    https://ijspt.scholasticahq.com/article/94600-a-biomechanical-review-of-the-squat-exercise-implications-for-clinical-practice

28. Straub, R.K. & Powers, C.M. (2024). *A Biomechanical Review of the Squat Exercise: Implications for Clinical Practice.* International Journal of Sports Physical Therapy. PMC.  
    **[SQ-2 — squat depth definitions; SQ-4 — knee valgus]**  
    https://pmc.ncbi.nlm.nih.gov/articles/PMC10987311/

29. Bloomquist, K. et al. (2016). *Muscle Activation Differs between Three Different Knee Joint-Angle Positions during a Maximal Isometric Back Squat.* PMC.  
    **[SQ-3 — muscle activation at 90° vs standing 140°+]**  
    https://pmc.ncbi.nlm.nih.gov/articles/PMC4967668/

30. McClure, C. (2020). *The Effects of Squatting Mechanics on the Soft Tissues of the Knee Joint.* UNC DPT Portfolios.  
    **[SQ-4 — knee valgus as form violation]**  
    https://dptportfolios.web.unc.edu/wp-content/uploads/sites/2565/2020/05/McClureC_AdvOrtho.pdf

31. Myer, G.D. et al. (2008). *The effects of plyometric vs. dynamic stabilization and balance training on power, balance, and landing force in female athletes.* Journal of Strength and Conditioning Research. PubMed.  
    **[SQ-5 — knee flare / width-to-hip ratio]**  
    https://pubmed.ncbi.nlm.nih.gov/19130646/

32. Swinton, P.A. et al. (2012). *A biomechanical analysis of straight and hexagonal barbell deadlifts using submaximal loads.* Journal of Strength and Conditioning Research. Via squat knee-width discussion.  
    **[SQ-5 — knee flare / width-to-hip ratio]**  
    https://www.acefitness.org/resources/everyone/blog/5428/what-happens-when-your-knees-cave-in-during-a-squat/

---

# Notes for Use in This Project

- Technical thresholds (e.g., elbow/knee/hip angle cutoffs) should be treated as heuristic ranges informed by literature and coaching guidelines, then calibrated empirically for camera setup and user population.

- EMG findings are exercise-variation dependent (grip width, cadence, tempo, load, depth, fatigue), so references are best used to justify rule directionality (what tends to increase/decrease activation), not as absolute universal cutoffs.
