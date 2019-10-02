"""Hand detection and basic finger counting.
Designed for image camera streamed from phone using the webip camera app"""


import cv2
import numpy as np
import tkinter as tk
import requests  # for accessing phone camera

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import ImageTk, Image  # to display video in tkinter app


class cellPhoneCapture:
    def __init__(self, camera_source=r'http://10.0.0.104:8080/shot.jpg'):
        self.camera_source = camera_source
        # read single image to create parameters
        ret, img = self.read()
        self.y, self.x = img.shape[:2]

    def read(self):
        try:
            # fetch frame from camera
            camera_response = requests.get(self.camera_source)
            # convert to numpy array
            np_image_array = np.array(bytearray(camera_response.content),
                                      dtype=np.uint8)
            img = cv2.imdecode(np_image_array, -1)
            cv2.flip(src=img, flipCode=0, dst=img)
            return True, img
        except:
            return None

    def release(self): # required for code compatability
        pass


class Handtracker:
    def __init__(self, capture=None):
        # use 'ip' for wifi camera (cellphone), native for builtin camera
        self.video_sources = {'ip', 'native'}
        self.default_ip = r'http://10.0.0.104:8080'  # add '/shot.jpg' for capture
        self.is_capture = False  # flag for video streaming (required for toggle)
        self.is_tracking = False  # Flag for tracking
        self.cap = capture
        self.vid_delay = 15  # may not be required
        self.debug = True

        # define bounding box (x,y,w,h) 0.2 image size at the top right quadrant
        if capture is not None:
            bboxh = int(self.cap.y * 0.45)
            bboxw = int(self.cap.x * 0.45)
            top = int(self.cap.y * 0.025)
            left = int(self.cap.x * 0.525)
            self.bbox = (left, top, bboxw, bboxh)
            self.bg = None

    def display_raw_stream(self):
        """this function creates a window and shows stream from cell phone front
        facing camera"""
        camera_source = r'http://10.0.0.104:8080/shot.jpg'
        while True:
            # fetch frame from camera
            ret, img = self.cap.read()
            processed = self.process_frame(img)
            # draw square on image
            self.drawbbox(img)
            vis = np.concatenate((img, processed), axis=1)
            cv2.imshow("AndroidCam", vis)
            # create an escape sequence
            if cv2.waitKey(1) & 0xFF == 27:
                cv2.destroyAllWindows()
                break

    def drawbbox(self, img):
        # draw bbox on img
        (x,y,w,h) = self.bbox
        cv2.rectangle(img, pt1=(x,y), pt2=(x+w,y+h), color=(255,0,0), thickness=3)

    def process_frame(self, img):
        # create copy if bbox
        # grab current frame from area
        (x,y,w,h) = self.bbox
        roi = img[y:y+h, x:x+w].copy()
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # remove background and threshold
        roi = self.remove_background(roi)
        if np.any(roi):
            roi = self.detect_hand(roi)
        else:
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        # draw lines from center of mass to fingertips
        roi = cv2.resize(roi, (self.cap.x, self.cap.y))
        return roi

    def remove_background(self, img, th=30):
        # grab current frame from area
        if self.bg is None:
            #self.bg = cv2.GaussianBlur(img (5,5), 0)
            self.bg = img
            return np.zeros_like(img)
        else:

            # TODO adapt background is too small
            # subtract background if exists
            no_bg_frame = cv2.absdiff(img, self.bg)
            ret, th_img = cv2.threshold(no_bg_frame, th, 255, cv2.THRESH_BINARY)
            return th_img

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

    # GUI

    def hand_tracker_gui(self):
        """Create a gui for hand tracker application"""
        self.root = tk.Tk()
        self.root.title("Hand tracking GUI")
        # main frame
        mainframe = tk.Frame(self.root)
        mainframe.grid(row=0, column=0, sticky='nsew')  # TODO add weights to make gui look nicer
        # control frame
        control_frame = tk.LabelFrame(mainframe, text="Controls")
        control_frame.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        # camera subframe
        camera_frame = tk.LabelFrame(control_frame, text="Video")
        camera_frame.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        source_label = tk.Label(camera_frame, text="Camera")
        source_label.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        # video source selection
        self.v_source_var = tk.StringVar(self.root)
        self.v_source_var.set(list(self.video_sources)[0])  # set the default option
        v_source_select_menu = tk.OptionMenu(camera_frame, self.v_source_var, *self.video_sources)
        v_source_select_menu.grid(row=0, column=1, sticky='nsew', pady=5)
        # video ip
        ip_label = tk.Label(camera_frame, text="ip:")
        ip_label.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)
        self.ip_source = tk.Entry(camera_frame)
        self.ip_source.grid(row=1, column=1, sticky='nsew', pady=5)
        self.ip_source.delete(0, tk.END)
        self.ip_source.insert(0, self.default_ip)
        # toggle capture
        toggle_capture_b = tk.Button(camera_frame, text="Start/Stop", command=self.gui_toggle_capture)
        toggle_capture_b.grid(row=2, column=0, sticky='nsw', padx=5, pady=5)
        # tracking subframe
        tracking_frame = tk.LabelFrame(control_frame, text="Tracking")
        tracking_frame.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)
        # toggle tracking
        toggle_tracking_b = tk.Button(tracking_frame, text="Start/Stop", command=self.gui_toggle_tracking)
        toggle_tracking_b.grid(row=0, column=0, sticky='nsw', padx=5, pady=5)

        # full image frame
        full_image_frame = tk.LabelFrame(mainframe, text="Full image")
        full_image_frame.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)

        full_frame_figure = plt.Figure(figsize=(2.5, 2.5), dpi=100)
        self.full_frame_axis = full_frame_figure.add_subplot(111)
        self.full_frame_axis.imshow(np.zeros((640,480,3), dtype='uint8'))
        self.full_frame_axis.axis('off')
        self.full_frame_canvas = FigureCanvasTkAgg(full_frame_figure, master=full_image_frame)  # A tk.DrawingArea.
        self.full_frame_canvas.draw()
        self.full_frame_canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')


        #self.full_canvas = tk.Canvas(self.full_image_frame,  width=480, height=640) #, width=self.vid.width, height=self.vid.height)
        #self.full_canvas.grid(row=0, column=0, sticky='nsew')

        # tracking view
        tracking_frame = tk.LabelFrame(mainframe, text="Tracking")
        tracking_frame.grid(row=0, column=2, sticky='nsew', padx=5, pady=5)
        self.tracking_canvas = tk.Canvas(tracking_frame, width=480, height=640)
        self.tracking_canvas.grid(row=0, column=0, sticky='nsew')

        self.root.mainloop()



    def gui_toggle_capture(self):
        """Toggle video capture and display in GUI"""
        if not self.is_capture:
            if self.v_source_var.get() == 'ip':
                self.cap = cellPhoneCapture(self.ip_source.get() +'/shot.jpg')
            elif self.v_source_var.get() == 'native':
                self.cap = cv2.VideoCapture(0)
            self.is_capture = True
            if self.debug:
                print('capture on')
            self.gui_display_stream()
        else:
            # toggle off
            self.cap.release()
            # set capture and tracking flags
            self.is_capture = False
            self.is_tracking = False
            if self.debug:
                print('capture off')

    def gui_display_stream(self):
        """Auxilary function to display stream in GUI"""
        if self.is_capture:
            ret, img = self.cap.read()
            if ret:
                # permute color channels if source is 'native'
                if self.v_source_var.get() == 'native':
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.full_frame_axis.imshow(img)
            self.root.after(self.vid_delay, self.gui_display_stream)


    def gui_toggle_tracking(self):
        """Toggle tracking in GUI"""
        print('TBD - toggle tracking')




def main():
    cap = cellPhoneCapture()
    tht = Handtracker(cap)
    tht.display_raw_stream()

if __name__ == "__main__":
    print('starting main')
    main()

