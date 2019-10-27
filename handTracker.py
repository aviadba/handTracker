"""Hand detection and basic finger counting.
Designed for image camera streamed from phone using the webip camera app"""


import cv2
import numpy as np
import imutils


class Handtracker:
    def __init__(self, capture=None):
        self.cap = capture
        self.vid_delay = 15  # may not be required
        self.debug = True
        self.img = None
        self.reset_tracking()
        self.reset_background()

    # self creation and resetting
    def reset_background(self):
        """reset background (and tracking)"""
        self.bg = {'avg': [0, 10, 0],  # counter, bg img#, bg value
                   'roihist': [0, 0, 10, None]}  # x_center, y_center, size, hist
        self.reset_tracking()

    def reset_tracking(self):
        """reset tracking bbox"""
        # define bounding box (x,y,w,h) 0.2 image size at the top right quadrant
        bboxh = int(self.cap.y * 0.45)
        bboxw = int(self.cap.x * 0.45)
        top = int(self.cap.y * 0.025)
        left = int(self.cap.x * 0.525)
        self.bbox = (left, top, bboxw, bboxh)
        self.tracker = None
        self.hand_detected_flag = False
        self.tracking_state = {'palm': None,
                               'area': None,
                               'centerxy':None,
                               'rect':None,
                               'finger_count':None}

    # single frame process wrapper
    def process_frame(self, img, bbox='static', background='avgbbox', th_method='binary', bin_th=30):
        """wrapper for all HandTracker outputs
        steps: 1. get image 2. detect roi using tracker and previous background 3. detect hand
            3. update background using 4. create images for display
                Parameters
                    bbox {'static', 'simple', 'mosse', 'kfc', 'csrt'} bounding box mode
                        static-as defined by defaults
                        mosse - Minimum Output Sum of Squared Error
                        kfc - Kernelized Correlation Filter
                        csrt - Discriminative correlation filter
                    background {'avgbbox'} - background removal mode:
                        avgbbox-average of bounding box
                    th_method {'binary', 'otsu'} - threshold method binary (must also give bin_th) or otsu method
                    bin_th: threshold value (for binary threshold)
                """
        self.img = img
        # first bbox updating
        if self.hand_detected_flag:
            if bbox in {'static', 'simple'}:
                (x, y, w, h) = self.bbox
            else:
                # update binding box
                self.updatebbox(img, tracker=bbox)
                (x, y, w, h) = self.bbox
        else:
            (x, y, w, h) = self.bbox
        if background == 'histbp' and self.hand_detected_flag:
            roi = self.remove_background_histbp(method=th_method, th=30)
        else:
            roi = self.remove_background_avgbbox(method=th_method, th=bin_th)
        if np.any(roi):  # bbox contains non background
            ret, roi = self.detect_hand(roi)
            # bbox refinment
            if ret:
                (x, y, w, h) = self.tracking_state['rect']
                x -= self.bbox[0]
                y -= self.bbox[1]
                roi = roi[y:y+h, x:x+w,...]
                if bbox == 'simple':  # update bbox for 2 'simple' mode
                    x = int(self.bbox[0] + x-w/2)
                    y = int(self.bbox[1] + y-h/2)
                    w = 2*w
                    h = 2*h
                    x = max(x,0)
                    y = max(y,0)
                    w = min(self.cap.x-x, w)
                    h = min(self.cap.y-y, h)
                    self.bbox = [x, y, w, h]
        else:
            (x, y, w, h) = self.bbox

        if w / h > self.cap.x / self.cap.y:
            new_h = int(self.cap.x / w * h)
            delta_h = self.cap.y - new_h
            try:
                roi = cv2.resize(roi, (self.cap.x, new_h))
            except:
                print('breakpoint')
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = 0, 0
        else:
            new_w = int(self.cap.y / h * w)
            delta_w = self.cap.x - new_w
            roi = cv2.resize(roi, (new_w, self.cap.y))
            top, bottom = 0, 0
            left, right = delta_w // 2, delta_w - (delta_w // 2)
        roi = cv2.copyMakeBorder(roi, top, bottom, left, right, cv2.BORDER_CONSTANT)
        return roi

    # tracking
    def updatebbox(self, img, tracker='mosse'):
        """Update bbox using selected tracker
            Parameters
                tracker {'mosse', 'kcf', 'csrt'} : tracker to initialize"""
        # if first run initialize tracker
        if self.tracker is None:
            OPENCV_OBJECT_TRACKER = {'mosse': cv2.TrackerMOSSE_create,
                                     'kcf': cv2.TrackerKCF_create,
                                     'csrt': cv2.TrackerCSRT_create}
            self.tracker = OPENCV_OBJECT_TRACKER[tracker]()
            # initialize the bounding box coordinates of the object we are going
            self.tracker.init(img, self.bbox)
        else:
            # to track
            (success, box) = self.tracker.update(img)
            if success:
                self.bbox = [int(v) for v in box]

    # background
    def remove_background_avgbbox(self, method='binary', th=30):
        """remove background by abs diff with a static background
            Parameters
            method {'binary', 'otsu'} threshold method
                th = Threshold value for binary threshold
        1. use val:self.bg['avg'][1] averaged first Gaussian blurred images in bbox for initial background
        2. Continuously update background outside current bbox by averaging values of current bg with
        Gaussian blurred current image
        """
        # grab current frame from area
        img = self.img.copy()
        (x, y, w, h) = self.bbox
        if self.bg['avg'][0] < self.bg['avg'][1]:  # continue collection of average (full) background images
            # grab 10 images for background
            bg = cv2.GaussianBlur(img, (5, 5), 0).astype('float')
            self.bg['avg'][2] += bg
            self.bg['avg'][0] += 1
            th_img = self.grab_bbox()
            return np.zeros((th_img.shape[1], th_img.shape[0]), dtype='uint8')
        elif self.bg['avg'][0] == self.bg['avg'][1]:  # background collected
            self.bg['avg'][2] = (self.bg['avg'][2]/self.bg['avg'][1]).astype('uint8')
            self.bg['avg'][0] += 1
            # subtract background if exists
            no_bg_frame = cv2.absdiff(img, self.bg['avg'][2])
            no_bg_frame = no_bg_frame[y:y+h,x:x+w]
            no_bg_frame = cv2.GaussianBlur(no_bg_frame, (5,5), 0)
            no_bg_frame_gw = cv2.cvtColor(no_bg_frame, cv2.COLOR_BGR2GRAY)
            # thresholding
            if method == 'binary':
                ret, mask = cv2.threshold(no_bg_frame_gw, th, 255, cv2.THRESH_BINARY)
            elif method == 'otsu':
                ret, mask = cv2.threshold(no_bg_frame_gw, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                if ret < th:
                    mask = np.zeros_like(mask)
            # process th_image to clean signal
            th_img = cv2.bitwise_and(no_bg_frame, no_bg_frame, mask=mask)
            return th_img
        else:
            # update background outside bbox
            bg = cv2.GaussianBlur(img, (5, 5), 0).astype('float')  #calculate 'full background'
            bg = ((bg + self.bg['avg'][2])/2).astype('uint8')
            bg[y:y+h,x:x+w] = self.bg['avg'][2][y:y+h,x:x+w]  # restore bbox image to
            self.bg['avg'][2] = bg
            # subtract background if exists
            no_bg_frame = cv2.absdiff(img, self.bg['avg'][2])
            no_bg_frame = no_bg_frame[y:y + h, x:x + w]
            no_bg_frame_gw = cv2.cvtColor(no_bg_frame, cv2.COLOR_BGR2GRAY)
            if method == 'binary':
                ret, mask = cv2.threshold(no_bg_frame_gw, th, 255, cv2.THRESH_BINARY)
            elif method == 'otsu':
                ret, mask = cv2.threshold(no_bg_frame_gw, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                if ret < th:
                    mask = np.zeros_like(mask)
            th_img = cv2.bitwise_and(no_bg_frame, no_bg_frame, mask=mask)
            return th_img

    def remove_background_histbp(self, method='binary', th=50):
        """remove background by calculating the 2dhist backprojection of the tracked palm
        Do not call before hand_detected_flag
        Parameters
            method {'binary', 'otsu'} threshold method
                th = Threshold value for binary threshold
        """
        if self.bg['roihist'][3] is None:  #[0, 0, 10, 0]}  # x_center, y_center, size, hist
            # get hand coordinates
            (x, y, w, h) = self.tracking_state['rect']
            # get 'offset' - size of sampeling square for reference histogram
            offset = self.bg['roihist'][2]
            # find center of
            sample_x = int(x + w/2 - offset)
            sample_y = int(y + h / 2 - offset)
            self.bg['roihist'][0] = sample_x
            self.bg['roihist'][1] = sample_y
            hsv = self.img[sample_y:sample_y+2*offset, sample_x:sample_x+2*offset].copy()
            hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)
            # calculating hand center and normalize
            roihist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
            # normalize the histogram
            cv2.normalize(roihist, roihist, 0, 255, cv2.NORM_MINMAX)
            self.bg['roihist'][3] = roihist
            # return blank image
            target = self.grab_bbox()
            return np.zeros_like(target).astype('uint8')
        else:
            target = self.grab_bbox()
            hsvt = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
            # calculating object histogram
            dst = cv2.calcBackProject([hsvt], [0, 1], self.bg['roihist'][3], [0, 180, 0, 256], 1)
            # Convolve with circular disc
            disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            cv2.filter2D(dst, -1, disc, dst)
            if method == 'binary':
                ret, mask = cv2.threshold(dst, 10, 255, cv2.THRESH_BINARY)
            elif method == 'otsu':
                ret, mask = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                if ret < th:
                    mask = np.zeros_like(mask)
            # TODO morphological operations to improve the image
            th_img = cv2.bitwise_and(target, target, mask=mask)
            return th_img

    # hand detection
    def detect_hand(self, img, th=10000):
        """Detect fingers
        Returns (bool, post_Detection image)

            """
        # find contours of palm
        img_gw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, img_th = cv2.threshold(img_gw, 1, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(img_th,
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
        # select best contour (largest)
        cntsSorted = sorted(contours, reverse=True, key=lambda x: cv2.contourArea(x))
        # calculate moments
        M = cv2.moments(cntsSorted[0])
        if M['m00'] < th:  #hand detection below th
            return False, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            # activate hand detected flag
            self.hand_detected_flag = True
            palm = cntsSorted[0]
            area = cv2.contourArea(palm)
            (x, y, w, h) = cv2.boundingRect(palm)
            x += self.bbox[0]
            y += self.bbox[1]
            # confine x and y to image borders
            x = max(x, 0)
            y = max(y, 0)
            # confine width and height to image borders
            w = min(w, self.img.shape[1]-x)
            h = min(h, self.img.shape[0]-y)
            # update to tracked area rect
            self.tracking_state['rect'] = [x, y, w, h]
            # find angle elipse (direction)
            # (x_ellipse, y_ellipse), (MA, ma), angle_ellipse = cv2.fitEllipse(palm)

            # calculate center of mass (center of hand)
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            self.tracking_state['centerxy'] = [center_x, center_y]  # if mode for hist background removal
            # calculate convexHull
            hull = cv2.convexHull(palm, returnPoints=False)
            hull_points = palm[np.squeeze(hull),...]
            equi_diameter = np.sqrt(4 * area / np.pi)

            # filter points pointing downwards
            finger_points = hull_points[hull_points[:, :, 1] < center_y  , ...]
            # filter points close to center
            distant_bool = np.linalg.norm(finger_points-(center_x, center_y), axis=1) > equi_diameter/2

            finger_points = finger_points[distant_bool,...]

            distance = finger_points[1:, :] - finger_points[:-1, :]
            distance = np.linalg.norm((distance), axis=1)
            dis_th = np.sqrt(M["m00"] / np.pi) / 2  # finger cluster distance th
            finger = np.split(finger_points, np.argwhere(distance > dis_th).ravel() + 1)
            finger = [np.max(a, axis=0) for a in finger]
            defects = cv2.convexityDefects(palm, hull)



            # find filter find local maxima
            # clustering for fingers
            # find direction (elipse AND fingers)
            # find tips. select pointing forward
            # connect center to tips
            # draw
            retimg = np.zeros((h,w,3), dtype=('uint8'))
            # draw binary outline of ahnd
            cv2.drawContours(retimg, [palm], 0, (0, 255, 0), 1)
            # draw convex hull
            cv2.drawContours(retimg, [hull_points], 0, (255, 0, 0), 1)
            cv2.circle(retimg, (center_x, center_y), 5, (0,0,255), 1)
            for finger_p in finger:  # draw lines to fingers
                cv2.line(retimg, (center_x, center_y), tuple(finger_p),
                         (0,0,255), 1)
            self.tracking_state['finger_count'] = len(finger)
            # TODO define summary_dict
            return True, retimg

    # drawing functions
    def draw_tracking(self, img):
        """draw bounding box on image and the number of tracked fingers
        Defaults to main image"""
        if self.hand_detected_flag:
            (x, y, w, h) = self.tracking_state['rect']
            color=(255, 0, 0)
            finger_count = self.tracking_state['finger_count']
        else:
            (x, y, w, h) = self.bbox
            color = (0, 255, 0)
            finger_count = 0
        cv2.putText(img=img, text='F: {:d}'.format(finger_count),
                    org=(20, 40),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=color,
                    thickness=2,
                    lineType=cv2.LINE_AA)
        cv2.rectangle(img, pt1=(x, y), pt2=(x + w, y + h), color=color, thickness=3)

    # auxillary
    def grab_bbox(self, img=None):
        """Return the bounded image. Defaults to main image"""
        (x, y, w, h) = self.bbox
        if img is None:
            bbox_img = self.img[y:y+h, x:x+w,...].copy()
        else:
            bbox_img = img[y:y + h, x:x + w, ...].copy()
        return bbox_img