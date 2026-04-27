# Joubert Damien, 03-02-2020 - updated by AvS 22-02-2024
"""
    Script converting a video into events. 
    The framerate of the video might not be the real framerate of the original video. 
    The user specifies this parameter at the beginning.
"""
# Code adjusted by Michał Antropik for live stream from webcam with opencv on 17th of October 2025
import cv2
import os
import sys
import torch
import numpy as np
from norse.torch.module.lif import LIFCell, LIFRecurrentCell
from norse.torch import LIFParameters, LIFState
from norse.torch import LICell, LIState
from typing import NamedTuple
#from aestream import FileInput
import matplotlib.pyplot as plt
from tqdm import tqdm

from iebcs_src.event_buffer import EventBuffer
from iebcs_src.dvs_sensor import DvsSensor
from iebcs_src.event_display import EventDisplay
from iebcs_src.arbiter import SynchronousArbiter, BottleNeckArbiter, RowArbiter


script_dir = os.path.dirname(os.path.abspath(__file__))

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
    #DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# Recreate the model architecture
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

# Load the trained weights
model_path = os.path.join(script_dir, "./ml-models/snn-birds-model.pth")
model.load_state_dict(torch.load(model_path, weights_only=True))

# Set to evaluation mode
model.eval()

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
    
while True:
    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Minimum 3 frames per second!
    fps = 3
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    out = cv2.VideoWriter(filename, fourcc, fps, (frame_width, 360))

    # Record for 1 second (fps * 1 second frames)
    num_frames_to_record = fps * 1

    for _ in range(num_frames_to_record):
        ret, frame = cap.read()
        
        if not ret:
            break
            
        resized_frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
                
        out.write(resized_frame)

    # Release resources
    #cap.release()
    out.release()
    
    
    # IEBCS

    cap2 = cv2.VideoCapture(filename)
    
    # Initialise the DVS sensor
    dvs = DvsSensor("MySensor")
    dvs.initCamera(int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                       lat=lat, jit = jit, ref = ref, tau = tau, th_pos = th_pos, th_neg = th_neg, th_noise = th_noise,
                       bgnp=bgnp, bgnn=bgnn)
    # To use the measured noise distributions, uncomment the following line
    # dvs.init_bgn_hist("../iebcs_data/noise_pos_161lux.npy", "../iebcs_data/noise_neg_161lux.npy")

    # drop one frame
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
    render_timesurface = 1
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
    ev_full.write(os.path.join(script_dir, "./outputs/events.dat").format(lat, jit, ref, tau, th_pos, th_noise))
    
  
    
    
    # Run SNN Inference
    
    # yh you have a dat file but still there is no official way how to do inference on a Norse model (aestream does not work on Windows yet)
    #events = FileInput('./outputs/events.dat', (640, 360)).load()
    events = np.zeros((640, 360))
    # ^ ofc remove those zeros if aestream will be fixed on Windows in the future
                    
    processed = []
    model_snn = model.snn
    
    # I spent 3 days on figuring this out...
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
    
    #print(events_struct)
    
    numpy_events = events_struct
    
    #reshaped_array = numpy_events.reshape((-1, 460800))
    
    data_as_list = numpy_events.tolist()
    regular_array = np.array(data_as_list, dtype=np.int64)
    
    flat_array = regular_array.flatten()
    target_size = 460800
    
    padded_array = np.zeros(target_size, dtype=flat_array.dtype)
    # TODO: The code below makes it impossible to cast outputs when they exceed target_size and gives an error
    #padded_array[:flat_array.size] = flat_array
    
    tensor_input = torch.from_numpy(padded_array)
    tensor_input = tensor_input.view(1, -1)
    tensor_input = tensor_input.view(1, 1, 2, 640, 360)
    tensor_input = tensor_input.float()
    # DONE

    # Yh sure I do need a tensor nor an output class, it looks like nobody truly used this framework for real inference
    readout_voltages = model_snn(tensor_input)
    
    # This code below would block the running engine
    #plt.plot(readout_voltages.squeeze(1).cpu().detach().numpy())
    #plt.show()
    
    print(readout_voltages)
    
    # Now the readout has to be transformed into a predicted class
    mean_voltages = readout_voltages.mean(dim=0)
    pobabilities = torch.softmax(mean_voltages, dim=-1)

    # Get predicted class index
    predicted_class = torch.argmax(pobabilities, dim=-1)
    
    def get_key_from_value(d, val):
        keys = [k for k, v in d.items() if v == val]
        if keys:
            return keys[0]
        return None
    
    value = predicted_class.item()

    print("Predicted bird on webcam:", get_key_from_value(classes, value))
