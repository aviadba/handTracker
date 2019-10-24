"""handTracker GUI"""

from handTracker import Handtracker
import tkinter as tk
import threading
from PIL import Image, ImageTk
from handtracker_utils import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import ImageTk, Image  # to display video in tkinter app


class handTracker_gui():
    def __init__(self):
        # define settings for gui
        # use 'ip' for wifi camera (cellphone), native for builtin camera
        self.video_sources = {'ip', 'native'}
        self.default_ip = r'http://10.0.0.104:8080'  # add '/shot.jpg' for capture
        self.bg_methods = {'avgbbox', 'histbp'}  # background elimination methods
        self.bg_th = {'binary', 'otsu'}  # background TH methods
        self.tracking_methods = {'static', 'simple', 'mosse', 'kcf', 'csrt'}
        self.is_capture = False  # flag for video streaming (required for toggle)
        self.is_tracking = False  # Flag for tracking
        self.delay = 15
        self.cap = None
        self.debug = True
        # start gui
        self.initialize_gui()

    def initialize_gui(self):
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
        # background subframe
        background_frame = tk.LabelFrame(control_frame, text="Background")
        background_frame.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)
        reset_background = tk.Button(background_frame, text="Reset BG", command=self.reset_background)
        reset_background.grid(row=0, column=0, sticky='nsw', padx=5, pady=5)
        # background substraction methods
        bg_method_label = tk.Label(background_frame, text="BG method")
        bg_method_label.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)
        self.bg_method_var = tk.StringVar(self.root)
        self.bg_method_var.set(list(self.bg_methods)[0])  # set the default option
        bg_methods_select_menu = tk.OptionMenu(background_frame, self.bg_method_var, *self.bg_methods)
        bg_methods_select_menu.grid(row=1, column=1, sticky='nsew', pady=5)
        # background thresholding methods
        bg_TH_label = tk.Label(background_frame, text="BG TH")
        bg_TH_label.grid(row=2, column=0, sticky='nsew', padx=5, pady=5)
        self.bg_th_var = tk.StringVar(self.root)
        self.bg_th_var.set(list(self.bg_th)[0])  # set the default option
        bg_th_select_menu = tk.OptionMenu(background_frame, self.bg_th_var, *self.bg_th)
        bg_th_select_menu.grid(row=2, column=1, sticky='nsew', pady=5)

        # tracking subframe
        tracking_frame = tk.LabelFrame(control_frame, text="Tracking")
        tracking_frame.grid(row=2, column=0, sticky='nsew', padx=5, pady=5)
        # toggle tracking
        toggle_tracking_b = tk.Button(tracking_frame, text="Start/Stop", command=self.gui_toggle_tracking)
        toggle_tracking_b.grid(row=0, column=0, sticky='nsw', padx=5, pady=5)
        trk_method_label = tk.Label(tracking_frame, text="Tracking method")
        trk_method_label.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)
        self.tracking_method_var = tk.StringVar(self.root)
        self.tracking_method_var.set(list(self.tracking_methods)[0])  # set the default option
        tracking_methods_select_menu = tk.OptionMenu(tracking_frame,
                                                     self.tracking_method_var,
                                                     *self.tracking_methods)
        tracking_methods_select_menu.grid(row=1, column=1, sticky='nsew', pady=5)
        # show base frame
        self.show_bbox_var = tk.IntVar()
        toggle_bbox = tk.Checkbutton(tracking_frame, text="Show bbox", variable=self.show_bbox_var)
        toggle_bbox.grid(row=2, column=0, sticky='nsw', padx=5, pady=5)




        # full image frame
        self.full_image_frame = tk.LabelFrame(mainframe, text="Full image")
        self.full_image_frame.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)
        # tracking view
        self.tracking_frame = tk.LabelFrame(mainframe, text="Tracking")
        self.tracking_frame.grid(row=0, column=2, sticky='nsew', padx=5, pady=5)

        self.gui_display_stream()
        self.root.mainloop()


    def gui_toggle_capture(self):
        """Toggle video capture and display in GUI"""
        if not self.is_capture:  # initialize capture
            if self.v_source_var.get() == 'ip':
                self.cap = cellPhoneCapture(self.ip_source.get() + '/shot.jpg')
            elif self.v_source_var.get() == 'native':
                self.cap = cv2.VideoCapture(0)
            self.is_capture = True
            if self.debug:
                print('capture on')
            # initialize canvas
            self.full_frame_canvas = tk.Canvas(self.full_image_frame, width=self.cap.x, height=self.cap.y)
            self.full_frame_canvas.grid(row=0, column=0, sticky='nswe')
        else:
            # toggle off
            self.cap.release()
            # set capture and tracking flags
            self.is_capture = False
            self.is_tracking = False
            if self.debug:
                print('capture off')

    def gui_toggle_tracking(self):
        """Toggle tracking in GUI"""
        if not self.is_tracking:  # initialize capture
            self.is_tracking = True
            # initialize tracker
            self.tracker = Handtracker(capture=self.cap)
            if self.debug:
                print('tracking on')
            # initialize canvas
            self.tracking_canvas = tk.Canvas(self.tracking_frame,
                                             width=self.cap.x,
                                             height=self.cap.y)
            self.tracking_canvas.grid(row=0, column=0, sticky='nswe')
        else:
            # toggle off
            self.cap.release()
            # set capture and tracking flags
            self.is_tracking = False
            if self.debug:
                print('tracking off')

    def reset_background(self):
        """Reset HandTracker background
        """
        self.tracker.reset_background()


    def gui_display_stream(self):
        """Auxilary function to display stream in GUI
        """
        if self.is_tracking:
            ret, img = self.cap.read()
            if ret:
                processed = self.tracker.process_frame(img,
                                                       bbox=self.tracking_method_var.get(),
                                                       background=self.bg_method_var.get(),
                                                       th_method=self.bg_th_var.get())
                self.tracker.drawbbox(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.full_frame_image = ImageTk.PhotoImage(image=Image.fromarray(img),
                                                           master=self.full_frame_canvas)
                self.full_frame_canvas.create_image(0, 0, image=self.full_frame_image,
                                                    anchor=tk.NW)
                self.tracking_image = ImageTk.PhotoImage(image=Image.fromarray(processed),
                                                           master=self.tracking_canvas)
                self.tracking_canvas.create_image(0, 0, image=self.tracking_image,
                                                    anchor=tk.NW)
        elif self.is_capture:
            ret, img = self.cap.read()
            # permute color channels if source is 'native'
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.full_frame_image = ImageTk.PhotoImage(image=Image.fromarray(img),
                                                       master=self.full_frame_canvas)
            self.full_frame_canvas.create_image(0, 0, image=self.full_frame_image, anchor = tk.NW)

        self.root.after(self.delay, self.gui_display_stream)


if __name__ == "__main__":
    print('starting main')
    test = handTracker_gui()
    # cap = cellPhoneCapture()
    # tht = Handtracker(cap)
    # tht.display_raw_stream()