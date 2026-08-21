# Gimbal Teleoperation with HoloLens

This subdirectory's code is for handling communication from the HoloLens to the ground station computer (via UDP) and ultimately to the gimbal (via UART serial over wireless radio).

## Hardware Setup

TODO: Outline the techniques for setting up hardware, including:
* HoloLens connected to the Unity frontend
* Gimbal wired with the UART radio module and connected to power with a common ground, be it on the bench or the drone
* The other UART radio module connected to the computer and configured to communicate wirelessly

## Software Setup

TODO: Outline the techniques for setting up software, including:
* What the code files are and what they do
* Anything to install or setup
* How to run the teleoperation (NOTE: The other unity repo will be required for this.)

For now, to setup the project just run `pip install -e .` inside a Python virtual environment.