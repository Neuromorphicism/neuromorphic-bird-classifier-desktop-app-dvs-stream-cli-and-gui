# Joubert Damien, 03-02-2020 - updated by AvS 22-02-2024
"""
    Script converting a video into events. 
    The framerate of the video might not be the real framerate of the original video. 
    The user specifies this parameter at the beginning.
"""

# Code adjusted by Michał Antropik for live stream from webcam with opencv on 17th of October 2025
# GUI Design and Code by Michał Antropik
# Whole GUI finished on the 3rd of November 2025
import cv2
import os
import sys
import torch
import numpy as np

from norse.torch.module.lif import LIFCell, LIFRecurrentCell
from norse.torch import LIFParameters, LIFState
from norse.torch import LICell, LIState
from typing import NamedTuple
from aestream import FileInput
#import matplotlib.pyplot as plt

sys.path.append("../iebcs-src")
from event_buffer import EventBuffer
from dvs_sensor import DvsSensor
from dat_files import load_dat_event
from arbiter import SynchronousArbiter, BottleNeckArbiter, RowArbiter
from tqdm import tqdm

import tkinter as tk
import threading
import time as time_lib
from PIL import Image, ImageTk
from playsound3 import playsound

import customtkinter as ctk 
import copy
import datetime
import shutil


#os.environ["OMP_NUM_THREADS"] = "2"

def decode(x):
    x, _ = torch.max(x, 0)
    log_p_y = torch.nn.functional.log_softmax(x, dim=1)
    return log_p_y

class SNN(torch.nn.Module):
    def __init__(
        self,
        input_features,
        hidden_features,
        output_features,
        tau_syn_inv,
        tau_mem_inv,
        record=False,
        dt=1e-3,
    ):
        super(SNN, self).__init__()
        self.l1 = LIFRecurrentCell(
            input_features,
            hidden_features,
            p=LIFParameters(
                alpha=100,
                v_th=torch.as_tensor(0.3),
                tau_syn_inv=tau_syn_inv,
                tau_mem_inv=tau_mem_inv,
            ),
            dt=dt,
        )
        self.input_features = input_features
        self.fc_out = torch.nn.Linear(hidden_features, output_features, bias=False)
        self.out = LICell(dt=dt)

        self.hidden_features = hidden_features
        self.output_features = output_features
        self.record = record

    def forward(self, x):
        seq_length, batch_size, _, _, _ = x.shape
        s1 = so = None
        voltages = []

        if self.record:
            self.recording = SNNState(
                LIFState(
                    z=torch.zeros(seq_length, batch_size, self.hidden_features),
                    v=torch.zeros(seq_length, batch_size, self.hidden_features),
                    i=torch.zeros(seq_length, batch_size, self.hidden_features),
                ),
                LIState(
                    v=torch.zeros(seq_length, batch_size, self.output_features),
                    i=torch.zeros(seq_length, batch_size, self.output_features),
                ),
            )

        for ts in range(seq_length):
            z = x[ts, :, :, :].view(-1, self.input_features)
            z, s1 = self.l1(z, s1)
            z = self.fc_out(z)
            vo, so = self.out(z, so)
            if self.record:
                self.recording.lif0.z[ts, :] = s1.z
                self.recording.lif0.v[ts, :] = s1.v
                self.recording.lif0.i[ts, :] = s1.i
                self.recording.readout.v[ts, :] = so.v
                self.recording.readout.i[ts, :] = so.i
            voltages += [vo]

        return torch.stack(voltages)
        
class SNNState(NamedTuple):
    lif0: LIFState
    readout: LIState        

class SNNModel(torch.nn.Module):
    def __init__(self, snn, decoder):
        super(SNNModel, self).__init__()
        self.snn = snn
        self.decoder = decoder

    def forward(self, x):
        x = self.snn(x)
        log_p_y = self.decoder(x)
        return log_p_y

classes = {"chicken": 0, "eagle": 1, "owl": 2, "pigeon": 3, "raven": 4, "stork": 5}
sensor_size = (640, 360, 2,)
LR = 0.002
INPUT_FEATURES = np.prod(sensor_size)
HIDDEN_FEATURES = 100
OUTPUT_FEATURES = len(classes)

if torch.cuda.is_available():
    DEVICE = torch.device("cpu")
    # Neuromorphicism comment: uncomment the code below and comment the one above and test if this program works on your GPU
    #DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# Neuromorphicism comment: recreate the model architecture
model = SNNModel(
    snn=SNN(
        input_features=INPUT_FEATURES,
        hidden_features=HIDDEN_FEATURES,
        output_features=OUTPUT_FEATURES,
        tau_syn_inv=torch.tensor(1 / 1e-2),
        tau_mem_inv=torch.tensor(1 / 1e-2),
        record=True,
        dt=1000,
    ),
    decoder=decode,
).to(DEVICE)

# Neuromorphicism comment: load the trained weights
model.load_state_dict(torch.load("./ml-models/snn-birds-model.pth", weights_only=True))

# Neuromorphicism comment: set model to evaluation mode
model.eval()
        
        
# RUN GUI

cap = None
events_canvas_image_id = None
events_photo_img = None

# Neuromorphicism comment: MP4V -mp4 or XVID - avi format or VP90 - webm format
# Obviously XVID - avi is much faster and has better quality
rgb_live_recording_filename = "./outputs/output.avi"
codec = "XVID"

th_pos = 0.4        # ON threshold = 50% (ln(1.5) = 0.4)
th_neg = 0.4        # OFF threshold = 50%
th_noise = 0.01     # standard deviation of threshold noise
lat = 100           # latency in us
tau = 40            # front-end time constant at 1 klux in us
jit = 10            # temporal jitter standard deviation in us
bgnp = 0.1          # ON event noise rate in events / pixel / s
bgnn = 0.01         # OFF event noise rate in events / pixel / s
ref = 100           # refractory period in us
dt = 1000           # time between frames in us
time = 0

# Neuromorphicism comment: The ICNS EventDisplay class has to be modified to not to use cv2.imgshow that opens a separate window!
class EventDisplay():
    time = 0  # Internal counter of the display (us)
    last_frame = 0  # Time of the last frame
    frametime = 100000  # Time to refresh the display (us) = 100 ms
    time_surface = np.zeros((10, 10), dtype=np.uint64)  # Timestamp of the last event in the focal plane
    pol_surface = np.zeros((10, 10), dtype=np.uint8)  # Polarity of the last event in the focal plane
    im = np.zeros((10, 10, 3), dtype=np.uint8)  # Image to render
    render = 0  # 0: binary image, 1: ts
    render_tau = 40000  # tau decay of the time surface (us)
    display_time = False

    
    def __init__(self, dx, dy, frametime, render=1, events_canvas=None):
        self.time = 0
        self.last_frame = 0
        self.frametime = frametime
        self.time_surface = np.zeros((int(dy), int(dx)), dtype=np.uint64)
        self.pol_surface = np.zeros((int(dy), int(dx)), dtype=np.uint8)
        self.im = np.zeros((int(dy), int(dx), 3), dtype=np.uint8)
        self.render = render
        self.render_tau = 3 * frametime
        self.display_time = False

        # Neuromorphicism comment: CustomTkinter component for live preview events display
        self.events_canvas = events_canvas
        

    def reset(self):
        self.time = 0
        self.last_frame = 0
        self.time_surface[:] = 0
        self.pol_surface[:] = 0

    def update(self, pk, dt):
        global events_canvas_image_id
        global events_photo_img
        self.time_surface[pk.y[:pk.i], pk.x[:pk.i]] = pk.ts[:pk.i]
        self.pol_surface[pk.y[:pk.i], pk.x[:pk.i]] = pk.p[:pk.i]
        self.time += dt
        self.last_frame += dt

        if self.last_frame > self.frametime:
            self.last_frame = 0
            self.im[:] = 125
            
            if self.render == 0:
                ind = np.where((self.time_surface > self.time - self.frametime) & (self.time_surface <= self.time))
                self.im[:, :, 0][ind] = self.pol_surface[ind]*255
                self.im[:, :, 1][ind] = self.pol_surface[ind]*255
                self.im[:, :, 2][ind] = self.pol_surface[ind]*255
                
            elif self.render == 1:
                decay = np.exp(-(self.time - self.time_surface.astype(np.double)) / self.render_tau)
                self.im[:, :, 0] = ((self.pol_surface * 2 - 1) * 125 * decay).astype(np.uint8)
            
            # Neuromorphicism comment: convert image as BGR to Tkinter canvas colors
            #im_conv = cv2.cvtColor(self.im, cv2.COLOR_BGR2GRAY)
            #alpha = 3 # Contrast control (1.0-3.0)
            #beta = 0 # Brightness control (0-100)
            #contrast_img = cv2.convertScaleAbs(self.im, alpha=alpha, beta=beta)
            
            # Neuromorphicism comment: never convert to GRAY or from BGR because you will destroy contrast and proper events presentation! It also uses more CPU...
            im_pil = Image.fromarray(self.im)
            # Neuromorphicism comment: mirror swap the image (again more CPU but works better on discrete GPU)
            events_photo_img = ImageTk.PhotoImage(image=im_pil.transpose(Image.FLIP_LEFT_RIGHT))
            
            if events_canvas_image_id != None:
                # Update existing image on canvas
                self.events_canvas.itemconfig(events_canvas_image_id, image=events_photo_img)
            else:
                # Create image on canvas for the first time
                events_canvas_image_id = self.events_canvas.create_image(0, 0, anchor=tk.NW, image=events_photo_img)
    

            # Neuromorphicism comment: this would cause flicker!
            # This also would not allow for animation control and proper stopping of running background thread
            #if self.own_root:
                #self.root.update_idletasks()
                #self.root.update()






class App:
    canvas = None
    rgb_canvas = None
    
    def __init__(self, root):
        self.root = root
        self.root.title("Neuromorphic Bird Classifier")
        
        self.thread = None
        self.thread_event = None
        self.is_running = False
        
        
        # Neuromorphicism comment: CustomTKinter GUI components
        
        button_font = ctk.CTkFont(family="Helvetica", size=30, weight="bold")
        text_font = ctk.CTkFont(family="Helvetica", size=26)
        
        self.column_of_controls = 0
        self.predicted_bird_class = None
        
        # Neuromorphicism comment: label to show predicted bird class
        self.bird_desc = ctk.CTkLabel(root, text="Bird classification: ", font=text_font, text_color="white", fg_color="transparent")
        self.bird_desc.grid(row=4, column=self.column_of_controls, padx=(0, 160), pady=(0, 0))
        
        self.bird = ctk.CTkLabel(root, text=self.predicted_bird_class, font=text_font, text_color="white", fg_color="transparent")
        self.bird.grid(row=4, column=self.column_of_controls, padx=(190, 0), pady=(0, 0))
        
        # Neuromorphicism comment: RGB Live Preview checkbox
        self.rgb_checkbox_var = ctk.StringVar(value="off")
        self.checkbox = ctk.CTkCheckBox(root, text=" RGB Live Preview", command=self.rgb_checkbox_event, variable=self.rgb_checkbox_var, onvalue="on", offvalue="off", width=150, height=50, text_color='white', hover=False, font=text_font, checkbox_width=26, checkbox_height=26, fg_color="#DDDDDD", corner_radius=0)
        
        # Neuromorphicism comment: I decided not to use the CustomTkinter checkbox on purpose but those style descriptions in Tkinter are so horrible... It is like hell for web devs.
        #self.checkbox = tk.Checkbutton(root, text="RGB Live Preview", command=self.rgb_checkbox_event, variable=self.rgb_checkbox_var, onvalue="on", offvalue="off", relief="flat", highlightthickness=0, highlightbackground="white", bd=0)
        #self.checkbox.config(bg="#111111", fg="white", font=text_font, selectcolor="blue", relief="raised", padx=30, pady=30, activeforeground="white", activebackground="#111111")
        # Neuromorphicism comment: ^ I can't even increase the size of a checkbox itself, it is like going back 20 years in UI tech
        
        self.checkbox.grid(row=5, column=self.column_of_controls, padx=(0, 125), pady=(20, 0))

        # Neuromorphicism comment: start button
        # v1 styles: fg_color="#556b2f", hover_color="#005500", border_color="#222b13"
        self.start_button = ctk.CTkButton(root, text="Start", command=self.start, corner_radius=0, border_width=1, fg_color="#091707", hover_color="#132b0e", border_color="#558b63", text_color="white", font=button_font, state="normal")
        self.start_button.grid(row=8, column=self.column_of_controls, rowspan=2, ipadx=130, ipady=24, padx=100)

        # Neuromorphicism comment: stop button
        # v1 styles: fg_color="#5c0000", hover_color="#770000", border_color="#3d0000"
        self.stop_button = ctk.CTkButton(root, text="Stop", command=self.stop, corner_radius=0, border_width=1, fg_color="#170202", hover_color="#2b0a0a", border_color="#8b3c3c", text_color="white", font=button_font, state="disabled")
        self.stop_button.grid(row=10, column=self.column_of_controls, rowspan=3, ipadx=130, ipady=24, padx=100, pady=(50, 0))
        
        # Neuromorphicism comment: show elapsed time count
        self.time_counter = ctk.CTkLabel(root, text="0", font=text_font, text_color="white", fg_color="transparent")
        self.time_counter.grid(row=14, column=self.column_of_controls, rowspan=2, padx=(170, 0))
        self.counter = 0

        self.time_counter_label = ctk.CTkLabel(root, text="Uptime (seconds): ", font=text_font, text_color="white", fg_color="transparent")
        self.time_counter_label.grid(row=14, column=self.column_of_controls, rowspan=2, padx=(0, 155))
        
        
        
        
        # Neuromorphicism comment: second column
        self.column_of_live_previews = 1
        
        # Neuromorphicism comment: events Canvas
        self.events_canvas = ctk.CTkCanvas(self.root, width=640, height=360)
        self.events_canvas.grid(row=6, column=self.column_of_live_previews, rowspan=8, padx=10)
        
        # Neuromorphicism comment: optional RGB Canvas
        self.temp_rgb_img = None
        self.rgb_im_pil = None
        self.rgb_photo_img = None
        self.rgb_canvas_image_id = None
        
        
        
        # Neuromorphicism comment: third column
        self.column_of_logger = 2

        self.third_column_header = ctk.CTkLabel(self.root, text="Bird Spotter-Logger", fg_color="transparent", text_color="white", font=text_font)
        self.third_column_header.grid(row=4, column=self.column_of_logger)
        
        # Neuromorphicism comment: dropdown menu
        self.bird_optionmenu = ctk.CTkOptionMenu(self.root, values=["none", "chicken", "eagle", "owl", "pigeon", "raven", "stork"], command=self.optionmenu_callback, font=text_font, corner_radius=0, dynamic_resizing=False, dropdown_font=text_font, anchor="center", height=36, width=160, fg_color="#1c2e4a", button_color="#152238")
        self.bird_optionmenu.set("none")
        self.bird_optionmenu.grid(row=5, column=self.column_of_logger)
        
        # Neuromorphicism comment: log only once a minute if the same bird is still present but in events vision you will not know if it will sleep there or stay there for 1 hour only if it leaves the next log might be logged after 1 hour but it might also mean that it is a new bird arriving there so maybe it can be solved by some form of IDs per many birds
        self.log_interval = 60
        self.last_logged_timestamp = None
        # Neuromorphicism comment: Take just one video not 60 videos in one hour and this value below must be reset if new bird is chosen from the dropdown menu
        self.was_the_chosen_bird_video_saved = False

        
        # Neuromorphicism comment: in this widget padding means the padding inside the textbox...
        self.textbox = ctk.CTkTextbox(self.root, corner_radius=0, width=406, height=500, pady=5, padx=5, fg_color="transparent", border_width=2, border_color="#222222", text_color="#999999", font=text_font, wrap="word", state="disabled")
        # Neuromorphicism comment: you can not insert into a disabled textbox
        self.textbox.configure(state="normal")
        self.textbox.insert("1.0", "Click on a blue box above to pick a bird to spot and to log it here with 1 minute interval. \n\nIf the chosen bird will be spotted then only once its RGB photo and its events video will be automatically saved in this program's spotted-birds folder. \n\nYou can always change the bird to spot even if the program is already running!\n") 
        self.textbox.configure(state="disabled")
        self.textbox.grid(row=7, column=self.column_of_logger, rowspan=6)
        
        # Neuromorphicism comment: save log to file button
        self.save_log_button = ctk.CTkButton(root, text="Save Log", command=self.save_log, corner_radius=0, border_width=1, fg_color="#111111", hover_color="#222222", border_color="#333333", text_color="white", font=button_font, state="normal")
        self.save_log_button.grid(row=14, column=self.column_of_logger, rowspan=2, ipadx=130, ipady=24, padx=100)
        

        
    # Neuromorphicism comment: first column methods
    def rgb_checkbox_event(self):
        value = self.rgb_checkbox_var.get()
        # Neuromorphicism comment: always reset the ids before painting images on new ctk canvas (that has new id too)
        global events_canvas_image_id
        events_canvas_image_id = None
        self.rgb_canvas_image_id = None
            
        if value == "off":
            self.rgb_canvas.grid_remove()
            
            self.events_canvas.grid_forget()
            self.events_canvas = ctk.CTkCanvas(self.root, width=640, height=360)
            self.events_canvas.grid(row=6, column=self.column_of_live_previews, rowspan=8, padx=10)
        else:
            self.rgb_canvas = ctk.CTkCanvas(self.root, width=640, height=360)
            self.rgb_canvas.grid(row=10, column=self.column_of_live_previews, rowspan=8, padx=10, pady=(30, 0))
            
            self.events_canvas.grid_forget()
            self.events_canvas = ctk.CTkCanvas(self.root, width=640, height=360)
            self.events_canvas.grid(row=2, column=self.column_of_live_previews, rowspan=8, padx=10, pady=(0, 30))
                

    def start(self):
        if not self.is_running:
            self.is_running = True
            self.stop_button.configure(state='normal')
            self.start_button.configure(state='disabled')
            # Open the webcam (device 0)
            global cap
            cap = cv2.VideoCapture(0)
            
            # Neuromorphicism comment: clear the logger before running a new session
            self.textbox.configure(state="normal")
            self.textbox.delete(0.0, 'end')
            self.textbox.configure(state="disabled")
            
            self.last_logged_timestamp = None
            self.was_the_chosen_bird_video_saved = False
            
            # Neuromorphicism comment: begin the uptime counter from 0
            self.counter = 0
            self.run_count()
            self.start_thread()
        

    def run_count(self):
        if self.is_running:
            self.counter += 1
            self.time_counter.configure(text=str(self.counter))
            
            # Schedule this function to run after 1 second
            self.root.after(1000, self.run_count)

    def stop(self):
        self.stop_button.configure(state='disabled')
        self.start_button.configure(state='normal')
        self.is_running = False
        
        # Neuromorphicism comment: cap.read blocks the thread so if you will release it before shutting down the thread it will never really stop!
        if cap.isOpened():
            cap.release()
        
        while not self.thread_event.is_set():
            print("Trying to stop the Neuromorphic Engine thread...")
            self.thread_event.set()
            self.thread.join()      
            time_lib.sleep(1)
                
        print("\nNeuromorphic Engine thread stopped!\n\n")
        
        
    def background_task(self, event):
        # Neuromorphicism comment: GUI updates should be done via root.after or another thread-safe method inside the neuromorphicEngine
        self.neuromorphicEngine()

    def start_thread(self):
        self.thread_event = threading.Event()
        self.thread = threading.Thread(target=self.background_task, args=(self.thread_event,))
        self.thread.setDaemon(False)
        self.thread.start()
        
    # Neuromorphicism comment: third column methods
    def optionmenu_callback(self, choice):
        self.last_logged_timestamp = None
        self.was_the_chosen_bird_video_saved = False
    
    def save_log(self):
        now = datetime.datetime.now()
        date_time = now.strftime("%d-%m-%Y-%H:%M")
        initial_name = "bird_log_" + date_time
        file = ctk.filedialog.asksaveasfile(mode='w', defaultextension=".txt", initialfile=initial_name)
        if file is None:
            return
        text2save = str(self.textbox.get(1.0, 'end-1c'))
        file.write(text2save)
        file.close()
        
        
        
        
        
    # NEUROMORPHIC ML VISION ENGINE
        
    def neuromorphicEngine(self):
        # Define codec and create VideoWriter object for MP4V -mp4 or XVID - avi format
        fourcc = cv2.VideoWriter_fourcc(*codec)
        # Minimum 3 frames per second!
        # Maximum 5 frames per second!
        # Neuromorphicism comment: the best is 3 frames per second for true live preview and event generation!
        fps = 3
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        out = cv2.VideoWriter(rgb_live_recording_filename, fourcc, fps, (frame_width, 360))

        # Record for 1 second (fps * 1 second of frames)
        num_of_seconds_to_record = 1
        num_frames_to_record = fps * num_of_seconds_to_record

        for _ in range(num_frames_to_record):
            ret, frame = cap.read()
            if not ret:
                break
            resized_frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
            out.write(resized_frame)

        # Release resources
        out.release()
        
        
        
        
        cap2 = cv2.VideoCapture(rgb_live_recording_filename)
        
        # IEBCS comment: Initialise the DVS sensor
        
        dvs = DvsSensor("MySensor")
        dvs.initCamera(int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                           lat=lat, jit = jit, ref = ref, tau = tau, th_pos = th_pos, th_neg = th_neg, th_noise = th_noise,
                           bgnp=bgnp, bgnn=bgnn)

        # Drop one frame
        ret, im = cap2.read()

        # Convert the image from uint8, such that 255 = 1e4, representing 10 klux
        im = cv2.cvtColor(im, cv2.COLOR_RGB2LUV)[:, :, 0] / 255.0 * 1e4

        # Set as the initial condition of the sensor
        dvs.init_image(im)

        # Create the event buffer
        ev_full = EventBuffer(1)

        # Create the arbiter - optional, pick from one below
        ea = SynchronousArbiter(0.1, time, im.shape[0])  # DVS346-like arbiter

        # Neuromorphicism comment: create the display
        # Neuromorphicism comment: now render it in tkinter canvas and pass the root
        # Neuromorphicism comment: those are not commands to some GPT but to you tutorial follower... you should have done it on your own or you will never learn!
        
        render_timesurface = 1
        ed = EventDisplay(cap2.get(cv2.CAP_PROP_FRAME_WIDTH), 
                          cap2.get(cv2.CAP_PROP_FRAME_HEIGHT), 
                          dt, 
                          render_timesurface,
                          self.events_canvas)


        if cap2.isOpened():
            # Loop over num_frames frames
            num_frames = fps - 1
            
            for frame in tqdm(range(num_frames), desc="Converting webcam stream to events", position=0, leave=False, disable=True):
                # Get frame from the video
                ret, im = cap2.read()
                if not ret or im is None:
                    print("Warning: Empty frame received, stopping.")
                    break
                    
                # Neuromorphicism comment: copy Python variable and reference hell
                self.temp_rgb_img = copy.deepcopy(im)
                                    
                # Convert the image from uint8, such that 255 = 1e4, representing 10 klux
                im = cv2.cvtColor(im, cv2.COLOR_RGB2LUV)[:, :, 0] / 255.0 * 1e4
                # Calculate the events
                ev = dvs.update(im, dt)
                
                # Display the events
                ed.update(ev, dt)
                # Add the events to the buffer for the full video
                ev_full.increase_ev(ev)
                
                # Neuromorphicism comment: display RGB Live Preview   
                if self.rgb_checkbox_var.get() == "on":
                    self.rgb_im_pil = Image.fromarray(self.temp_rgb_img)
                    # Mirror swap the image to only display
                    self.rgb_photo_img = ImageTk.PhotoImage(image=self.rgb_im_pil.transpose(Image.FLIP_LEFT_RIGHT))
                    
                    if self.rgb_canvas_image_id != None:
                        self.rgb_canvas.itemconfig(self.rgb_canvas_image_id, image=self.rgb_photo_img)
                    else:
                        self.rgb_canvas_image_id = self.rgb_canvas.create_image(0, 0, anchor=tk.NW, image=self.rgb_photo_img)

        # Neuromorphicism comment: the live preview should be seen now inside GUI canvas
            
        
        cap2.release()
        
        # Neuromorphicism comment: fix the flicker in GUI, schedule next update after 100 ms (3 FPS)
        if self.is_running:
            # this has to be similar to the camera frametime variable
            self.root.after(110, self.neuromorphicEngine)
        
        
        # Save the events to a .dat file
        ev_full.write('outputs/events.dat'.format(lat, jit, ref, tau, th_pos, th_noise))
        
        
        # Neuromorphicism comment: before inference first test if there are any events on gray image - optimization
        # Neuromorphicism comment: I think it would be best to simply check the length of events (not the buffer) and if they are too short then do nothing to optimize the runtime
        
        events = FileInput('outputs/events.dat', (640, 360)).load()  
        
        # Neuromorphicism comment: from logged tests below 100 events means that the camera is empty of events
        if len(events) > 100:
        
            # Neuromorphicism comment: try to spawn another thread just to run the inference so it would not block the creation of new events frames for live preview! 
            # It would decrease lag! It would also not crash the live preview if there is an error from numpy or norse...
            # TODO
            
            # Run SNN Inference     
            processed = []
            model_snn = model.snn
            
            # Neuromorphicism comment: I spent 3 days on figuring this out...
            # events is a structured array with fields: 'timestamp', 'x', 'y', 'polarity'
            # Extract fields as separate arrays and stack them column-wise
            events_np = np.array(events.tolist()) 
                        
            events_struct = np.zeros(events_np.shape[0],
                     dtype=[("t", np.int64), ("x", np.int64),
                            ("y", np.int64), ("p", np.int8)])
            events_struct["t"] = events_np[:, 0]
            events_struct["x"] = events_np[:, 1]
            events_struct["y"] = events_np[:, 2]
            events_struct["p"] = events_np[:, 3]
            
            numpy_events = events_struct
            
            data_as_list = numpy_events.tolist()
            regular_array = np.array(data_as_list, dtype=np.int64)
            
            flat_array = regular_array.flatten()
            target_size = 460800
            
            padded_array = np.zeros(target_size, dtype=flat_array.dtype)
            padded_array[:flat_array.size] = flat_array
            
            tensor_input = torch.from_numpy(padded_array)
            tensor_input = tensor_input.view(1, -1)
            tensor_input = tensor_input.view(1, 1, 2, 640, 360)
            tensor_input = tensor_input.float()
            # DONE

            readout_voltages = model_snn(tensor_input)
            
            # Neuromorphicism comment: now the readout has to be transformed into a predicted class
            mean_voltages = readout_voltages.mean(dim=0)
            pobabilities = torch.softmax(mean_voltages, dim=-1)

            # Neuromorphicism comment: get predicted class index
            predicted_class = torch.argmax(pobabilities, dim=-1)
            
            def get_key_from_value(d, val):
                keys = [k for k, v in d.items() if v == val]
                if keys:
                    return keys[0]
                return None
            
            value = predicted_class.item()
            
            bird_class = get_key_from_value(classes, value)
            
            # Neuromorphicism comment: display predicted class now only in GUI
            self.predicted_bird_class = bird_class
            self.bird.configure(text=str(self.predicted_bird_class))
            
            # Neuromorphicism comment: log the spotted bird
            if self.bird_optionmenu.get() == bird_class:
                # Neuromorphicism comment: save the RGB and Events videos of a spotted bird
                if not self.was_the_chosen_bird_video_saved:
                
                    # Neuromorphicism comment: turn on the 5 seconds alarm because the bird was spotted for the first time but it blocks the whole thread on a single core! Always use block=False in playsound!
                    playsound("./sounds/alarm.mp3", block=False)
                    
                    # dd/mm/YY
                    now = datetime.datetime.now()
                    date_time = now.strftime("%d-%m-%Y_%H-%M-%S")
                    
                    self.was_the_chosen_bird_video_saved = True
                
                    # Neuromorphicism comment: if you also want 3 frames RGB video to be saved then uncomment the code below
                    #dest_video_name = "./spotted-birds/" + self.bird_optionmenu.get() + "_rgb_video_" + date_time + ".avi"
                    #shutil.copy2(rgb_live_recording_filename, dest_video_name)
                    
                    dest_events_name = "./spotted-birds/" + self.bird_optionmenu.get() + "_events_" + date_time + ".dat"
                    shutil.copy2("./outputs/events.dat", dest_events_name)
                    
                    # Neuromorphicism comment: whole code below is copied from the IEBCS repo to make the preview events video in a folder from a dat file

                    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
                    # Neuromorphicism comment: modified!
                    ts, x, y, p = load_dat_event(dest_events_name)
                    # Neuromorphicism comment: modified!
                    dest_events_video_name = "./spotted-birds/" + self.bird_optionmenu.get() + "_events_video_" + date_time + ".avi"
                    
                    
                    res = [640, 360]
                    out = cv2.VideoWriter('{}.avi'.format(dest_events_video_name[:-4]), fourcc, 20.0, (res[0], res[1]))
                    tw = 1000
                    img         = np.zeros((res[1], res[0]), dtype=float)
                    tsurface    = np.zeros((res[1], res[0]), dtype=np.int64)
                    indsurface  = np.zeros((res[1], res[0]), dtype=np.int8)

                    for t in range(ts[0], ts[-1], tw):
                        ind = np.where((ts > t) & (ts < t + tw))
                        tsurface[:, :] = 0
                        tsurface[y[ind], x[ind]] = t + tw
                        indsurface[y[ind], x[ind]] = 2.0 * p[ind] - 1
                        ind = np.where(tsurface > 0)
                        img[:, :] = 125
                        img[ind] = 125 + indsurface[ind] * np.exp(-(t + tw - tsurface[ind].astype(np.float32))/ (tw/30)) * 125

                        # Convert to color and display
                        img_c = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                        img_c = cv2.putText(img_c, '{} us'.format(t + tw), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                              (255, 255, 255))
                        img_c = cv2.applyColorMap(img_c, cv2.COLORMAP_VIRIDIS)
                        cv2.waitKey(1)
                        
                        # Write video to file
                        out.write(img_c)
                    out.release()
                    
                    # END of IEBCS code
                    
                    
                    # Neuromorphicism comment: save only 1 photo from the captured 3 frames of video
                    vidcap = cv2.VideoCapture(rgb_live_recording_filename)
                    success, image = vidcap.read()
                    
                    if not success:
                        while not success:
                          success, image = vidcap.read()
                    
                    if success:
                        cv2.imwrite("./spotted-birds/" + self.bird_optionmenu.get() + "_photo" + date_time + ".png", image)
                    
                

                # Neuromorphicism comment: log spotted bird only once a minute
                if not self.last_logged_timestamp or (self.last_logged_timestamp + datetime.timedelta(seconds=self.log_interval)) < datetime.datetime.now():
                    # dd/mm/YY
                    now = datetime.datetime.now()
                    date_time = now.strftime("%d/%m/%Y %H:%M:%S")
                    
                    self.last_logged_timestamp = now
                    
                    self.textbox.configure(state="normal")
                    self.textbox.insert("end", "\n* " + date_time + " -- " + self.predicted_bird_class) 
                    self.textbox.configure(state="disabled")
            
            # Neuromorphicism comment: if the root.after would be here then it would crash whole animation if any inference error appeared!
        


# Neuromorphicism comment: begin building GUI

root = ctk.CTk()

# Neuromorphicism comment: values below can be used for a dynamic geometry creation for different displays
#real_width = root.winfo_screenwidth()
#real_height = root.winfo_screenheight()

root.geometry("1920x1080")
root.configure(fg_color='#111111')

root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=2)
root.grid_columnconfigure(2, weight=1)

root.grid_rowconfigure((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20), weight=1, minsize=50)

ico = ImageTk.PhotoImage(file = os.path.join("icons", "neurobcda-icon.png"))
root.wm_iconphoto(False, ico)
root.iconphoto(True, ico)
root.iconbitmap("icons/neurobcda-favicon.ico")

app = App(root)
root.mainloop()
