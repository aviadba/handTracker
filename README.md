# handTracker
hand tracker - gui hand tracker

run:

  import handTracker_gui
  handTracker_gui.handTracker_gui()

description
GUI after tracking initialization. Trigger area framed in green
![screenshot of handTracker before hand detection]
(https://github.com/aviadba/handTracker/blob/master/screenshots/init_tracking.png)

Hand tracking (blue) frame
![screenshot of handTracker tracking]
(https://github.com/aviadba/handTracker/blob/master/screenshots/tracking.png)

Shows hand outline, center of moments and lines to tips of fingers
![screenshot of handTracker finger detection]
(https://github.com/aviadba/handTracker/blob/master/screenshots/finger_detection.png)

Usage:
  
  Video
    Camera {ip, nativ} - video input channel. ip for wifi camera (like phone)
      native for capturedevice(0) (laptop webcam)
    ip -  if Camera=ip, set this to address
    Start/Stop to start streaming (Notes: tested only with ip cam)
  
  Background:
    BG method: {avgbbox, histbp}
    BG TH {binary, otsu} - Background threshold. 
      binary val??
      otsu - Otsu method for threshold (2-mean clustering)
    
Tracking method: {csrt, mosse, kcf, static, simple}
  all tracking modes start with identification
  csrt,  mosse and kcf: openCV implemetation. bbox (bounding box) updated dynamically
 
IMPORTANT! 
  Start 'Video' BEFORE 'Tracking'. 
  For 'Tracking' all settings must be valid (e.g 'Background')
