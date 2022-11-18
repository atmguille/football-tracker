# Football Tracker

This repository implements a real-time and easy-to-use football tracker. As seen in the demo below, it is able to automatically detect the following key elements:
* Lines of the football field
* Players and ball
* Player's team
* Skeletons of a customizable number of players closer to the ball

We believe that this tracker can be used in many different applications, such as:
* Real-time offside detection
* Real-time statistical analysis of football matches
* Real-time strategy detection

## Demo

TODO

## Prerequisites

### Conda environment

The `enviroment.yml` file contains all the dependencies used during the development of this project. You can create the corresponding conda environment by running the following command:

```bash
conda env create -f environment.yml
```

### External repositories

This project uses two external repositories that have been added as submodules. To clone them, run the following command from the root of this repository:

```bash
git submodule update --init
```

Note that although `yolov7` is fully based on the original repository, `mmpose` is a fork of the original repository. This fork contains some optimizations for faster inference.

### Models checkpoints

Two pre-trained models are required to run this project: one to detect the players and the ball, and another one to detect the skeletons. Checkpoints for both models can be downloaded as follows:

```bash
mkdir models_checkpoints
cd models_checkpoints
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
wget https://download.openmmlab.com/mmpose/top_down/mobilenetv2/mobilenetv2_coco_256x192-d1e58e7b_20200727.pth
```

## Usage

First of all, one can get a list of all the available options by running the following command:

```bash
python preprocess_videos.py -h
```

The demo output can be generated with the default settings. Simply place your `demo.mp4` inside a directory (for instance `input_videos`) and run the following command:

```bash
python preprocess_videos.py --src_directory input_videos --dst_directory out_videos
```
