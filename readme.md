# Humanoid Robot Dance Control over Real-Time Musc Stimulation

## 1. Introduction to the Work

This work is the project implementation for AU332 Aritificial Intelligence Lectures, SJTU, 2019 Autumn. It mainly implements four jobs:

1) Establish dance motion base by extracting motions from dance videos with 3D human pose estimation and analytical mapping;
2) Extract music beat with beat tracking algorithms;
3) Implement a Markov Chain based motion selection policy;
4) Integrate individual parts and form the whole dance control system.

A high-level presentation of our work is shown in the figure below:

<img src="Final Report Latex Source Code\\overall_diagram.jpg" width = "50%" />

## 2. Code and File Arrangement

In this work, the codes and files are arranged as follow:

1) Folder **/3DMPPE_POSENET_RELEASE** stores the algorithm used for 3D human pose estimation, which is adapted from https://github.com/mks0601/3DMPPE_POSENET_RELEASE. We integrate Yolo v3 as dancer detector in the motion extrating process, and relevant codes are stored in folder **/3DMPPE_POSENET_RELEASE/common/detectors/yolo**. The yolo algorithm is adapted from https://github.com/eriklindernoren/PyTorch-YOLOv3. Utilizing these tools, we implement the code for extracting 3d dance poses and constructing the motion base in **/3DMPPE_POSENET_RELEASE/main/motion_extract.py**. Some testing scripts and outputs are also store in this folder;
2) Folder **/dance_videos** stores the videos we select for motion extraction. The extracted soundtracks and beat list are also stored here;
3) Folder **/MyNao** stores the main work we do for implementing our dance control system. Codes inside this folder are all implemented by ourselves;
4) Folder **/Final Report Latex Source Code** stores the latex source code for our final project report;
5) File **5_赵经纬_何正保.pptx** is the powerpoint for our presentation;
6) File **nao dance demo.mp4** is the demo of our work;
7) File **AI_Final_Report.pdf** is the final report of our project. It is 12 pages long (including references), and introduces our work in great detail.

## 3. Dependencies

1) **madmom**, package for music information retrieval. You should install it with 'pip install madmom'. Besides, **cython**, **mido**, **pytest**, **pyaudio**, **pyfftw** should be install along with it. For more information about this package, see https://github.com/CPJKU/madmom;
2) **pytorch 1.1** or higher, along with **opencv**;
3) **tqdm**;
4) **CoppeliaSim**, a simulation platform for robot control. Older versions (named Vrep) is also supported. For more information, see http://www.coppeliarobotics.com.

## 4. Run

If you wish to see Nao dance with our control, you should:

1) Start **CoppeliaSim**, drag Nao into the environment (Nao is in folder /robots/mobile), disable Nao's non-threaded child script for JointRecorder, and activate the environment by clicking the Start button. At this moment, you should see Nao stand still in the environment. If not, then the script for JointRecorder has not been disabled yet;
2) Play a piece of music with relatively strong beats. British and American pop music is recommended. Please make sure that the voice-recording function is not disabled for your laptop, and play the music piece in as loud volumn as possible.
3) Run **/MyNao/demo/demo_test_four-core-cpu.py**. You should see Nao in CoppeliaSim dance to the music soon.

Currently, the motion base consists of 10 motions, 5 of which are active. If you want to add your own motion, you should:
1) Find a proper solo dance video (multi-person dance is not implemented yet, but is feasible with yolo), and place it in folder **\dance videos**;
2) in **/3DMPPE_POSENET_RELEASE/main/motion_extract.py**, you should modify:

        1. video_dir (line 55) to your video's direcotory;
        2. interval (line 57) to the time interval between which you favoured motion lies;
        3. REDU = True (line 58). If you are not satisfied with the motion you've just set, you can reset this motion by setting REDU = False;
        4. Uncomment line 246 and line 247;
        5. You may modify motion['feature']['repeat'] (line 78) if you want this motion to be repeated with high chance;
        6. If you wish to add two symetric motions, you should set motion['feature']['symmetric'] (line 77) to each other's motion ID. Please also note that a left motion should be labeled with an odd ID, and its right counterpart's ID is 1 larger.
3) Run **/3DMPPE_POSENET_RELEASE/main/motion_extract.py**, and your motion will be added to the motion base.

## 5. Others

If you have any problem with our work, please refer to our report. This report introduces our work, motivations, backgrounds, contributions, and limitations, etc. in great detail. And if you have any problem in dealing with the codes, you may feel free to contact us.

Zhao Jingwei: zhaojw@sjtu.edu.cn

He  Zhengbao: lstefanie@sjtu.edu.cn

2019.12.31