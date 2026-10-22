# Drone Inspection Ground Station

![repo-root](https://github.com/CCSU-Robotics-Research/Drone-Inspection-Ground-Station/actions/workflows/root-ci.yml/badge.svg)
![gimbal-teleoperation](https://github.com/CCSU-Robotics-Research/Drone-Inspection-Ground-Station/actions/workflows/gimbal-ci.yml/badge.svg)
![camera-vision](https://github.com/CCSU-Robotics-Research/Drone-Inspection-Ground-Station/actions/workflows/camera-ci.yml/badge.svg)

This repository houses all of the components for the ground station of the drone project. Different components include:
* Wireless teleoperation of the gimbal using HoloLens
* Wireless transmission of the camera feed from the drone to the HoloLens
* Live AI analysis of the video feed for culvert defect detection and inspection

## Repository Map

[gimbal_teleoperation/](gimbal_teleoperation/) - Contains all code for wireless teleoperation of the gimbal
[camera_vision/](camera_vision/) - Reads the drone camera feed, streams it to the HoloLens via the Unity repo, and runs AI defect detection.

## Setup

Each component of this repository is self-contained. Please refer to the corresponding README.md for setting up the different components.

## Running the Full System

Once each component is setup, run this `python launch.py` from the repository root. This starts gimbal teleoperation and camera vision together, each with its own venv, with live AI annotation on from the start. To stop everything, press Ctrl+C. If any component exits, the launcher kills all processes. To run each component on its own, see its corresponding README.
