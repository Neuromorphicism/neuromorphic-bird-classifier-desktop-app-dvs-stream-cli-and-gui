# Joubert Damien, 03-02-2020 - updated by AvS 22-02-2024
"""
    Script converting a video into events. 
    The framerate of the video might not be the real framerate of the original video. 
    The user specifies this parameter at the beginning.
"""
# ICNS code adjusted by Michał Antropik for live stream from webcam with opencv on 17th of October 2025

import cv2
import os
import sys
from tqdm import tqdm

from iebcs_src.event_buffer import EventBuffer
from iebcs_src.dvs_sensor import DvsSensor
from iebcs_src.event_display import EventDisplay
from iebcs_src.arbiter import SynchronousArbiter, BottleNeckArbiter, RowArbiter


script_dir = os.path.dirname(os.path.abspath(__file__))

filename = os.path.join(script_dir, "./outputs/output.mp4")
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

# Open the webcam (device 0)
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture(0, cv2.CAP_FFMPEG)
    
while True:
    # Define codec and create VideoWriter object for mp4 format
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Minimum 3 frames per second!
    fps = 5
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    out = cv2.VideoWriter(filename, fourcc, fps, (frame_width, frame_height))

    # Record for 1 second (fps * 1 second frames)
    num_frames_to_record = fps * 1

    for _ in range(num_frames_to_record):
        ret, frame = cap.read()
        
        if not ret:
            break
            
        # resized_frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
        # out.write(resized_frame)
                
        out.write(frame)

    # Release resources
    # cap.release()
    out.release()
    
    # IEBCS DVS Simulator transformed to live event stream

    cap2 = cv2.VideoCapture(filename)
    
    # Initialise the DVS sensor
    dvs = DvsSensor("MySensor")
    dvs.initCamera(int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                       lat=lat, jit = jit, ref = ref, tau = tau, th_pos = th_pos, th_neg = th_neg, th_noise = th_noise,
                       bgnp=bgnp, bgnn=bgnn)
    # To use the measured noise distributions, uncomment the following line
    # dvs.init_bgn_hist("../iebcs_data/noise_pos_161lux.npy", "../iebcs_data/noise_neg_161lux.npy")

    # Drop one frame
    ret, im = cap2.read()

    # Convert the image from uint8, such that 255 = 1e4, representing 10 klux
    im = cv2.cvtColor(im, cv2.COLOR_RGB2LUV)[:, :, 0] / 255.0 * 1e4

    # Set as the initial condition of the sensor
    dvs.init_image(im)

    # Create the event buffer
    ev_full = EventBuffer(1)

    # Create the arbiter - optional, pick from one below
    # ea = BottleNeckArbiter(0.01, time)                # This is a mock arbiter
    # ea = RowArbiter(0.01, time)                       # Old arbiter that handles rows in random order
    ea = SynchronousArbiter(0.1, time, im.shape[0])  # DVS346-like arbiter

    # Create the display
    render_timesurface = 2
    ed = EventDisplay("Events", 
                      cap2.get(cv2.CAP_PROP_FRAME_WIDTH), 
                      cap2.get(cv2.CAP_PROP_FRAME_HEIGHT), 
                      dt, 
                      render_timesurface)

    if cap2.isOpened():
        # Loop over num_frames frames
        num_frames = fps - 1
        
        for frame in tqdm(range(num_frames), desc="Converting webcam stream to events", position=0, leave=True):
            # Get frame from the video
            ret, im = cap2.read()
            if not ret or im is None:
                print("Warning: Empty frame received, stopping.")
                break

            # Convert the image from uint8, such that 255 = 1e4, representing 10 klux
            im = cv2.cvtColor(im, cv2.COLOR_RGB2LUV)[:, :, 0] / 255.0 * 1e4
            # Calculate the events
            ev = dvs.update(im, dt)
            
            # Display the events
            ed.update(ev, dt)
            # Add the events to the buffer for the full video
            ev_full.increase_ev(ev)

    cap2.release()
    
    # Save the events to a .dat file
    # ev_full.write(os.path.join(script_dir, "outputs/events.dat").format(lat, jit, ref, tau, th_pos, th_noise))
 
