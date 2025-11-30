<p align="center">
<img src="https://raw.githubusercontent.com/Neuromorphicism/neuromorphic-bird-classifier-desktop-app-dvs-stream-cli-and-gui/main/neuromorphic-bird-classifier-desktop-app-banner.png">
</p>

<p align="center">
	<img src="https://img.shields.io/github/license/Neuromorphicism/neuromorphic-bird-classifier-desktop-app-dvs-stream-cli-and-gui" />
	<img src="https://img.shields.io/github/v/release/Neuromorphicism/neuromorphic-bird-classifier-desktop-app-dvs-stream-cli-and-gui" />
	<a href="https://github.com/Neuromorphicism/neuromorphic-bird-classifier-desktop-app-dvs-stream-cli-and-gui/pulse" alt="Activity">
        <img src="https://img.shields.io/github/last-commit/Neuromorphicism/neuromorphic-bird-classifier-desktop-app-dvs-stream-cli-and-gui" />
    </a>
    <a href="https://open-neuromorphic.org/neuromorphic-computing/">
	    <img src="https://img.shields.io/badge/Collaboration_Network-Open_Neuromorphic-blue">
	</a>
</p>
<br>

[Neuromorphic](https://en.wikipedia.org/wiki/Neuromorphic_computing) Bird Classifier Desktop App bundled with Live Event Camera Simulator.

This Python application uses Norse as its SNN inference engine and Tonic as the data loader. The GUI is created in CustomTkinter.

The Live Event Camera Simulator is a custom tool based on ICNS IEBCS that does not require CUDA to run unlike other available similar tools.

<br>

## Using NeuroBCDA

To use this software you must use **Python version 3.8+** and have **PyTorch version 1.9 or higher** installed. The AEStream 0.6.4 package requires the **gcc version to be higher than 10**.

It is best to use <a href="https://docs.anaconda.com/anaconda/install/"  title="Anaconda">Anaconda</a> or <a  href="https://docs.conda.io/en/latest/miniconda.html" title="Miniconda">Miniconda</a> but you can try installing the Python packages without it.

Pick your OS below and follow the steps that will enable you to use NeuroBCDA programs.

<br>

### Download the SNN Model first

Replace the file in `programs/ml-models/snn-birds-empty-model.pth.txt` with a proper model `snn-birds-model.pth` downloaded from here: https://huggingface.co/neuromorphicism/snn-birds-model-v1/tree/main

This `ml-models` directory should only contain: `programs/ml-models/snn-birds-model.pth`

<br>

### Linux

Download this repository and open the terminal window in its folder.

Create a conda environment first
```bash
conda create -n neurobcda python=3.9
```

Enable the created conda environment
```bash
conda activate neurobcda
```

Install NeuroBCDA packages from pip
```bash
pip install -r requirements-linux.txt
```

#### Run the GUI
```bash
python ./programs/neuromorphic-bird-classifier-gui-linux.py
```

#### Run the terminal CLI without GUI
```bash
python ./programs/neuromorphic-bird-classifier-cli-linux.py
```

#### Run the Event Camera Simulator
```bash
python ./programs/dvs-live-stream-simulator-linux.py
```
<br>


### Windows

Currently (November 2025) AEStream does not work in Windows. Read about the issues connected to installation of this library in the `windows.txt` file. You can still use the Event Camera Simulator and GUI without the inference engine on Windows.

Create a conda environment first
```bash
conda create -n neurobcda python=3.9
```

Enable the created conda environment
```bash
conda activate neurobcda
```

Install NeuroBCDA packages from pip
```bash
pip install -r requirements-windows.txt
```
  
#### Run the GUI
```
python .\programs\neuromorphic-bird-classifier-gui-windows.py
```

#### Run the terminal CLI without GUI
```
python .\programs\neuromorphic-bird-classifier-cli-windows.py
```

#### Run the Event Camera Simulator
```
python .\programs\dvs-live-stream-simulator-windows.py
```
<br>


### macOS

Currently only Live Event Camera Simulator is available on macOS. This time you do not need conda.

To use it run:

```
python ./programs/dvs-live-stream-simulator-macos.py
```
<br>

You must be sure to run it in a not dark environment outside with enough light for your webcam, otherwise cv2 on macOS will output an array with mostly zeros and it will stop events from generating as there is not enough light change spotted in the picture. 

This is a weird bug actually that comes from the IEBCS library, which does not provide the fallback for cv2 img as NoneType. It might also be caused by how Apple webcams work in their hardware.

<br>

## GUI Instructions

<p align="center">
<img src="https://raw.githubusercontent.com/Neuromorphicism/neuromorphic-bird-classifier-desktop-app-dvs-stream-cli-and-gui/main/neurobcda-gui-preview.png">
</p>

<br>

Click on the RGB checkbox to display both the events live preview and RGB live preview. Uncheck it to display only the events live preview.

Click on the "Start" button to turn on your webcam.

Click on the blue dropdown button on the right to pick a bird to spot.

Click on "Save Log" button to save the log of spotted chosen bird. You can find those logs in the `programs/logs` directory.

There will also be a photo and an events video taken once of a chosen bird. You can find it in the `programs/spotted-birds` directory.

You can click on the "Stop" button at any time. It will turn off the webcam and the log will still be there. If you will click on the "Start" button it will refresh all the GUI fields and erase the log.

<br>

## Development Tutorial

It is available here: https://www.linkedin.com/pulse/your-first-free-neuromorphic-desktop-app-neuromorphicism-8ixvf

<br>
  
## Contributing

To improve this software you can open an issue or create a pull request from your forked repository of NeuroBCDA to this main repository.

<br>

## Further Improvements

1. Try to generate frames in Live Event Camera Simulator without the use of a file system because RAM writes/reads are much faster than SSD writes/reads. I did not test this app on a HDD.
    
2.  Use Expelliarmus to fix the AEStream bugs on macOS and Windows. Rewrite this Application with use of Expelliarmus if possible.
    
3.  Improve the model inference. It might be better to squeeze images into squares and go below 200x200 dimensions. On the other hand, it might be a good practice only if you want to distinguish between a horse and a bird but to distinguish between different classes of birds you might need a greater resolution for better feature extraction.
    
4.  Neuromorphic MLOps pipelines can be created for this entire Desktop Application.
    
5.  Run manual tests of this Desktop Application in real life scenarios.
    
6.  Try to move this app to Raspbian OS and use it with the Raspberry Pi camera and this way make a mobile app/cam with built-in display that does not run on a PC/Laptop. Try to run it on a smartphone.

<br>

## Software Sources

IEBCS (GPL-3.0 license): [https://github.com/neuromorphicsystems/IEBCS](https://github.com/neuromorphicsystems/IEBCS)

TONIC (GPL-3.0 license): [https://github.com/neuromorphs/tonic](https://github.com/neuromorphs/tonic)

NORSE (LGPL-3.0 license): [https://github.com/norse/norse](https://github.com/norse/norse)

AESTREAM (MIT license): [https://github.com/aestream/aestream](https://github.com/aestream/aestream)

CUSTOMTKINTER (MIT license): [https://github.com/tomschimansky/customtkinter](https://github.com/tomschimansky/customtkinter)

<br>

## NeuroBCDA License

GNU General Public License (GPL) v3. See [LICENSE](LICENSE) for license details.

<br>

## No Warranty
I provide no warranty on the created Neuromorphic Bird Classifier Desktop Application nor on the changed source code of libraries used in this source code. Use it in real life scenarios at your own risk. The purpose of this open-source code is only educational. In derived works it has to have proper attributions and also be open-source on the same GPL-3.0 license.
