"""Utilities for running HandTracker and HandTrackerGUI
contains:
cellPhoneCapture - used to display cell phone camera images as defined by ip Android App"""

import cv2
import numpy as np
import requests  # for accessing phone camera


class cellPhoneCapture:
    """Use cellphone camera as video source (source: ip)"""
    def __init__(self, camera_source=r'http://10.0.0.104:8080/shot.jpg'):
        self.camera_source = camera_source
        self.y, self.x = self.get_size()

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

    def get_size(self):
        """Return the size of the image capture hXw"""
        ret, img = self.read()
        if ret:
            return tuple(img.shape[:2])
        else:
            return None


    def release(self):  # required for code compatibility
        pass

    def display_raw_stream(self):
        """this function creates a window and shows stream from cell phone front
        facing camera"""
        camera_source = r'http://10.0.0.104:8080/shot.jpg'
        while True:
            # fetch frame from camera
            ret, img = self.read()
            # processed = self.process_frame(img)
            # draw square on image
            # self.drawbbox(img)
            #vis = np.concatenate((img, processed), axis=1)
            cv2.imshow("AndroidCam", img)
            # create an escape sequence
            if cv2.waitKey(1) & 0xFF == 27:
                cv2.destroyAllWindows()
                break