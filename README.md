# handTracker
hand tracker - gui hand tracker
## Getting Started
run:
```
import handTracker_gui
handTracker_gui.handTracker_gui()
```
### Screenshots
GUI after tracking initialization. Trigger area framed in green
![screenshot of handTracker before hand detection](https://github.com/aviadba/handTracker/blob/master/screenshots/init_tracking.png)

Hand tracking (blue) frame
![screenshot of handTracker tracking](https://github.com/aviadba/handTracker/blob/master/screenshots/tracking.png)

Hand outline, center of moments and vectors to tips of fingers
![screenshot of handTracker finger detection](https://github.com/aviadba/handTracker/blob/master/screenshots/finger_detection.png)

### Manual
#### Video
* Camera {ip, nativ}
video input channel. ip for wifi camera (like phone) native for capturedevice(0) (laptop webcam)
* ip

if Camera=ip, set this to address
* Start/Stop

to start streaming (Notes: tested only with ip cam)
#### Background
* BG method: {avgbbox, histbp}

background removal method
  - avgbbox
  
  calculate initial background, update global background with dynamic averaging outside bbox
  - histbp
  
  histogram backpropagation. After hand initial detection, create base histogram. Threshold the color probability map of bbox given base histogram
* BG TH {binary, otsu}

Background threshold. 
  - binary val??
  - otsu
  
  Otsu method for threshold (2-mean clustering)
#### Tracking    
* Tracking method: {csrt, mosse, kcf, static, simple}

Tracking initializes after hand is detected

  - csrt,  mosse and kcf
  
openCV implemetation. bbox (bounding box) updated dynamically

  - static
  
bbox is limited to to initial area but the limits of the hand are dynamically updated

  - simple
  
bbox is updated with current hand position. Implemetation of simple tracker
 
IMPORTANT! 
  Start 'Video' BEFORE 'Tracking'. 
  For 'Tracking' all settings must be valid (e.g 'Background')
