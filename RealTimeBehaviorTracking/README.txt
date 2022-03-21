README for Closed-loop TTLs with DeepLabCut

Overview of Falkner Lab closed-loop setup:

1) DLC-computer
	- Intel Xeon W-2133 6.GGHz 12-core CPU
	- running ubuntu 16.04 
	- GeForce RTX 2080 GPU
	- Cuda 10.0, Driver Version 410.104
	- Anaconda environement with packages listed in packages.txt
	- Run some version of rt_generic.py
2) Arduino Uno
	- Loaded with some version of pulse_chan12_20Hz_5pulses.ino
	- Connected with USB to DLC-computer
	- Connected from output channel (chan 12 for us) to a BNC to the laser (and split to synchronization computer)
3) Camera
	- FLIR Blackfly S
	- BNC connection to DLC-computer
	- Output signals sent to DAQ/sync computer via GPIO cable
4) Separate computer/system for DAQ and synchronization


Basic instructions:

1) Train and generate a deeplabcut network on your setup with the same camera settings as will be used for closed-loop experiments [code not included].
2) Gather hardware and initialize an anaconda environment for realtime-dlc (see packages.txt for packages we used).
3) Establish your desired stimulation criterion and edit rt_generic.py line 510 with your logic for when to trigger laser.
4) Load your desired stimulation parameters (pulse width, frequency etc.) by editing pulse_chan12_20Hz_5pulses.ino.