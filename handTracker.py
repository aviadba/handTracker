"""Hand detection and basic finger counting.
Designed for image camera streamed from phone using the webip camera app"""


import cv2
import numpy as np


class Handtracker:
    def __init__(self, capture=None):
        self.cap = capture
        self.vid_delay = 15  # may not be required
        self.debug = True
        # define bounding box (x,y,w,h) 0.2 image size at the top right quadrant
        bboxh = int(self.cap.y * 0.45)
        bboxw = int(self.cap.x * 0.45)
        top = int(self.cap.y * 0.025)
        left = int(self.cap.x * 0.525)
        self.bbox = (left, top, bboxw, bboxh)
        self.bg = {'avg': [0, 0]}

    def drawbbox(self, img):
        # draw bbox on img
        (x,y,w,h) = self.bbox
        cv2.rectangle(img, pt1=(x,y), pt2=(x+w,y+h), color=(255,0,0), thickness=3)

    def process_frame(self, img, bbox='static', background='avgbbox'):
        """wrapper for all HandTracker outputs
                Parameters
                    bbox {'static'} bounding box mode
                        static-as defined by defaults
                    background {'avgbbox'} - background removal mode:
                        avgbbox-average of bounding box
                """
        if bbox == 'static':
            (x, y, w, h) = self.bbox
        roi = img[y:y + h, x:x + w].copy()
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # remove background and threshold
        if background == 'avgbbox':
            roi = self.remove_background_avgbbox(roi)
        elif background == 'histbp':
            pass
        if np.any(roi):
            roi = self.detect_hand(roi)
        else:
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        # draw lines from center of mass to fingertips
        if w/h > self.cap.x/self.cap.y:
            new_h = int(self.cap.x/w*h)
            delta_h = self.cap.y - new_h
            roi = cv2.resize(roi, (self.cap.x, new_h))
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = 0, 0
        else:
            new_w = int(self.cap.y/h*w)
            delta_w = self.cap.x - new_w
            roi = cv2.resize(roi, (new_w, self.cap.y))
            top, bottom = 0, 0
            left, right = delta_w // 2, delta_w - (delta_w // 2)
        roi = cv2.copyMakeBorder(roi, top, bottom, left, right, cv2.BORDER_CONSTANT)
        return roi

    def remove_background_avgbbox(self, img, base_img=10, th=30):
        """remove background by abs diff with a static background,
        calculated as a weighted sum of multiple images"""
        # grab current frame from area
        if self.bg['avg'][0] < base_img:
            # grab 10 images for background
            bg = cv2.GaussianBlur(img, (5,5), 0).astype('float')
            self.bg['avg'][1] += bg
            self.bg['avg'][0] += 1
            return np.zeros_like(bg).astype('uint8')
        elif self.bg['avg'][0] == base_img:
            self.bg['avg'][1] = (self.bg['avg'][1]/base_img).astype('uint8')
            self.bg['avg'][0] += 1
            # subtract background if exists
            no_bg_frame = cv2.absdiff(img, self.bg['avg'][1])
            ret, th_img = cv2.threshold(no_bg_frame, th, 255, cv2.THRESH_BINARY)
            return th_img
        else:
            # subtract background if exists
            no_bg_frame = cv2.absdiff(img, self.bg['avg'][1])
            ret, th_img = cv2.threshold(no_bg_frame, th, 255, cv2.THRESH_BINARY)
            return th_img

    def remove_background_histbp(self):
        pass

    def reset_background(self):
        """reset background"""
        self.bg = {'avg': [0, 0]}


    def detect_hand(self, img, th=8000):
        # find contours
        h,w = img.shape[:2]
        contours, hierarchy = cv2.findContours(img,
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
        # select best contour (largest)
        cntsSorted = sorted(contours, reverse=True, key=lambda x: cv2.contourArea(x))
        # calculate moments
        M = cv2.moments(cntsSorted[0])
        if M['m00'] < th:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            #retimg = np.zeros((h,w,3), dtype=('uint8'))
        else:
            palm = cntsSorted[0]
            area = cv2.contourArea(palm)
            rect = cv2.minAreaRect(palm)
            (x, y), (MA, ma), angle_ellipse = cv2.fitEllipse(palm)

            # calculate center of mass
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            # calculate convexHull
            hull = cv2.convexHull(palm, returnPoints=False)
            hull_points = palm[np.squeeze(hull),...]
            equi_diameter = np.sqrt(4 * area / np.pi)

            finger_points = hull_points[hull_points[:, :, 1] < center_y  , ...]
            distant_bool = np.linalg.norm(finger_points-(center_x, center_y), axis=1) > equi_diameter/2
            finger_points = finger_points[distant_bool,...]
            # filter countour points that are
            # find filter find local maxima
            # clustering
            # find direction



            # draw line cm to direction
            # draw
            retimg = np.zeros((h,w,3), dtype=('uint8'))
            cv2.drawContours(retimg, [palm], 0, (0, 255, 0), 1)
            cv2.drawContours(retimg, [hull_points], 0, (255, 0, 0), 1)
            cv2.circle(retimg, (center_x, center_y), 5, (0,0,255), 1)
            for finger_p in finger_points:
                cv2.line(retimg, (center_x, center_y), tuple(finger_p),
                         (0,0,255), 1)

            return retimg

        # find tips. select pointing forward
        # connect center to tips

