from moviepy.editor import *
from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor
from madmom.io import write_beats
from tqdm import tqdm
import numpy as np
import cv2
from config import cfg
import torch
import time
from base import Tester
from utils.vis import vis_keypoints
from utils.pose_utils import flip
from detectors.YOLOv3 import YOLOv3
import torch.backends.cudnn as cudnn
from torch.nn.parallel.data_parallel import DataParallel
from PIL import Image
import torchvision.transforms as transforms
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.axes._axes import _log as matplotlib_axes_logger
mpl.use("TKAgg", warn=False, force=True)
matplotlib_axes_logger.setLevel('ERROR')
from dataset import generate_patch_image #BML Added
import json
import os
import sys
#from pyAudio import VideoFileClip

def show3D_pose(channels, ax,skeleton, radius=40, mpii=1, lcolor='#ff0000', rcolor='#0000ff'):
    vals = channels
    connections = skeleton

    for ind, (i,j) in enumerate(connections):
        if(i ==5.5):
            x, y, z = [np.array([(vals[5,c]+ vals[6,c])/2,vals[j,c]]) for c in range(3)]
        else:   
            x, y, z = [np.array([vals[i, c], vals[j, c]]) for c in range(3)]
        ax.plot(x, y, z, lw=2, c=lcolor )

    RADIUS = radius  # space around the subject

    xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.view_init(azim=-90, elev=-90)

def show_pose(input_patch, img_patch, coord_out, skeleton, save_path, frame):
    tmpimg = input_patch[0].cpu().numpy()
    tmpimg = tmpimg * np.array(cfg.pixel_std).reshape(3,1,1) + np.array(cfg.pixel_mean).reshape(3,1,1)
    tmpimg = (tmpimg).astype(np.uint8)
    tmpimg = tmpimg[::-1, :, :]
    tmpimg = np.transpose(tmpimg,(1,2,0)).copy()
    tmpkps = np.zeros((3,18))
    tmpkps[:2,:] = coord_out[0,:,:2].transpose(1,0) / cfg.output_shape[0] * cfg.input_shape[0]
    tmpkps[2,:] = 1
    tmpimg = vis_keypoints(tmpimg, tmpkps, skeleton)
    tmpimg = cv2.resize(tmpimg,(img_patch.shape[1],img_patch.shape[0]))
    file_name = save_path+'\\{0}.png'.format(str(frame).zfill(2))
    cv2.imwrite(file_name, tmpimg)

def main():
    video_list = ['Cant stop the feeling - Justin Timberlake - Easy Dance for Kids', 'Dance like yo daddy', 'Danny Ocean - Baby I Wont', 'Si una vez - If I Once', 'Vaiven - MegaMix']
    for video in video_list:
        video_dir = 'dance_videos\\' + video + '.mp4'
        beat_dir = video_dir.strip('mp4') + 'npy'

        cudnn.fastest = True
        cudnn.benchmark = True
        cudnn.deterministic = False
        cudnn.enabled = True

        time_0 = time.time()
        tester = Tester(24)

        ##loading 3D pose estimation model
        tester._make_model()

        time_1 = time.time()
        print('loading integral pose model elapse:',round(time_1-time_0,2),'s')

        ##loading yolo detector
        detector = YOLOv3( model_def="3DMPPE_POSENET_RELEASE\\common\\detectors\\yolo\\config\\yolov3.cfg",
                            class_path="3DMPPE_POSENET_RELEASE\\common\\detectors\\yolo\\data\\coco.names",
                            weights_path="3DMPPE_POSENET_RELEASE\\common\\detectors\\yolo\\weights\\yolov3.weights",
                            classes=('person',),
                            max_batch_size=16,
                            device=torch.device('cuda:{}'.format(cfg.gpu_ids[0])))
        print('loading yolo elapse:',round(time.time()-time_1,2),'s')
        skeleton = ( (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6) )
        #fig = plt.figure(figsize=(10,10)) 
        transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)]
                                )
        
        if not os.path.exists(video_dir.strip('mp4')+'wav'):
            videoclip = VideoFileClip(video_dir)
            audioclip = videoclip.audio
            audioclip.write_audiofile(video_dir.strip('mp4')+'wav')

        video = cv2.VideoCapture(video_dir)
        if not os.path.exists(beat_dir):
            time_2 = time.time()
            videoclip = VideoFileClip(video_dir)
            audioclip = videoclip.audio
            beat_activation = RNNBeatProcessor()(video_dir.strip('mp4')+'wav')
            processor = DBNBeatTrackingProcessor(fps=100)
            beats = processor(beat_activation)
            frames_at_beat = (beats/audioclip.duration*video.get(cv2.CAP_PROP_FRAME_COUNT)).astype(int)
            print('extracting beat sequence elapse:', round(time.time()-time_2, 2), 's')
            np.save(beat_dir, frames_at_beat)
        frames_at_beat = np.load(beat_dir).tolist()

        ##########################################
        dance_primitives_dir = '.\\danceprimitives_trial'
        if not os.path.exists(dance_primitives_dir):
            os.mkdir(dance_primitives_dir)
        motion_index = len(os.listdir(dance_primitives_dir))
        for i in range(len(frames_at_beat)-1):

            motion_dir = os.path.join(dance_primitives_dir, '{0}'.format(str(motion_index).zfill(5)))
            if not os.path.exists(motion_dir):
                os.mkdir(motion_dir)

            start = frames_at_beat[i]
            end =frames_at_beat[i+1]
            dance_primitive = np.empty((0, 17*3)) # for motion control
            #dance_primitive_norm = np.empty((0, 17*3)) # for motion clustering
            video.set(1, start)
            jump_flag = 0
            frame = 0
            with torch.no_grad():
                time_start = time.time()
                while True:
                    current_frame = video.get(cv2.CAP_PROP_POS_FRAMES)
                    ret_val, raw_image = video.read()
                    if current_frame == end:
                        break
                    ##using yolo to get human bounding box
                    input_img = raw_image.copy()
                    detections = detector.predict_single(input_img)
                    if detections is None or detections.size()[0] == 0:
                        jump_flag = 1
                        break
                    last_conf = 0
                    for i, (x1_pred, y1_pred, x2_pred, y2_pred, conf, cls_conf, cls_pred) in enumerate(detections):
                        if conf.item() > last_conf:
                            x1 = max(int(round(x1_pred.item())) - 40, 0)
                            x2 = min(int(round(x2_pred.item())) + 40, input_img.shape[1]-1)
                            y1 = max(int(round(y1_pred.item())) - 20, 0)
                            y2 = min(int(round(y2_pred.item())) + 20, input_img.shape[0]-1)   #for getting a larger bounding box to cover the full body, in order to get more accurate pose
                            last_conf = conf.item()
                    img_patch = (input_img[y1:y2, x1:x2, ::-1]).copy().astype(np.float32)
                    ##using ResPoseNet to get 3D human pose
                    input_patch = cv2.resize(img_patch,(cfg.input_shape))
                    input_patch = transform(input_patch).unsqueeze(0)
                    coord_out = tester.model(input_patch).cpu().numpy() #dimention: 1 X 18 X 3, where '3' refers to x, z, y in sequence.
                    #show_pose(input_patch, img_patch, coord_out, skeleton, motion_dir, frame)
                    coord_out_resize = coord_out * np.array([img_patch.shape[1]/cfg.input_shape[1], img_patch.shape[0]/cfg.input_shape[0], 1]) #transform to original scale
                    coord_out = coord_out_resize[:, :-1, :] # neglect the key point for "throx"
                    #coord_out_norm = (coord_out-np.mean(coord_out, axis=1))/np.std(coord_out, axis=1)
                    dance_primitive = np.vstack((dance_primitive, np.reshape(coord_out[0], -1)))
                    #dance_primitive_norm = np.vstack((dance_primitive_norm, np.reshape(coord_out_norm[0], -1)))
                    frame += 1
                print('Processing Time Elapse:', round(time.time()-time_start,2), 's')

            if jump_flag == 1:
                continue

            #norm_sample = np.empty((0, 17*3))
            #num_sample = 10
            #print(dance_primitive_norm.shape[0])
            #sample_step = (dance_primitive_norm.shape[0]-1)/(num_sample-1)
            #for i in range(num_sample):
            #    norm_sample = np.vstack((norm_sample, dance_primitive_norm[round(i * sample_step)]))
            
            #print(norm_sample.shape)
            print(dance_primitive.shape)
            #np.save(os.path.join(motion_dir, 'dance_motion_normlized_'+ str(motion_index)), norm_sample)
            np.save(os.path.join(motion_dir, 'dance_motion_'+ str(motion_index)), dance_primitive)

            motion_index+=1



    ###########################################
    sys.exit()
    video.set(1, interval[0])
    frame=0
    next_beat = 0
    last_beat = 0
    num_beat = 0
    num_frame_between_beats = []
    with torch.no_grad():
        while True:
            time_start = time.time()
            current_frame = video.get(cv2.CAP_PROP_POS_FRAMES)
            ret_val, raw_image = video.read()
            if current_frame == interval[1]:
                break
            input_img = raw_image.copy()
                    ##using yolo to get human bounding box
            detections = detector.predict_single(input_img)
            # if not detections.cpu().numpy().all():
            #     detections = (0,0,input_img.shape[1],input_img.shape[0],1,1)
            #     print('not detected')

            if detections is None:
                detections = np.array([[0,0,input_img.shape[1],input_img.shape[0],1,1,1]])
                print('not detected')
            elif detections.size()[0] == 0:
                detections = np.array([[0,0,input_img.shape[1],input_img.shape[0],1,1,1]])
                print('not detected')
            last_conf = 0
            last_last_conf = 0
            for i, (x1_pred, y1_pred, x2_pred, y2_pred, conf, cls_conf, cls_pred) in enumerate(detections):
                if conf.item() > last_conf:
                    x1 = int(round(x1_pred.item())) - 40
                    x2 = int(round(x2_pred.item())) + 40
                    y1 = int(round(y1_pred.item())) - 20
                    y2 = int(round(y2_pred.item())) + 20    #for getting a larger bounding box to cover the full body, in order to get more accurate pose
                    last_last_conf = last_conf
                    last_conf = conf.item()
                print(last_conf, last_last_conf)
                if last_last_conf != 0:
                    sys.exit()
            #print(x1, x2, y1, y2, last_conf)
            img_patch = (input_img[y1:y2, x1:x2, ::-1]).copy().astype(np.float32)
            input_patch = cv2.resize(img_patch,(cfg.input_shape))

            input_patch = transform(input_patch).unsqueeze(0)
            coord_out = tester.model(input_patch)
            print('Running model time:',round(time.time()-time_start,2),'s')

            motion['frame'][frame] = {}
            if frame+interval[0] in frames_at_beat:
                motion['frame'][frame]['next_beat'] = 0
                motion['frame'][frame]['last_beat'] = 0
                #frames_at_beat.remove(frame)
                next_beat = frames_at_beat.index(frame+interval[0]) + 1
                last_beat = frames_at_beat.index(frame+interval[0])
                num_beat += 1
                num_frame_between_beats.append(frames_at_beat[next_beat] - frames_at_beat[last_beat])
                print('Record key frame with beat:', current_frame)
            else:
                motion['frame'][frame]['next_beat'] = frames_at_beat[next_beat] - (frame+interval[0])
                motion['frame'][frame]['last_beat'] = (frame+interval[0]) - frames_at_beat[last_beat]

            coord_out = coord_out.cpu().numpy()
            coord_out_resize = coord_out * np.array([img_patch.shape[1]/cfg.input_shape[1], img_patch.shape[0]/cfg.input_shape[0], 1])

            for idx in range(coord_out_resize.shape[1]-1):
                motion['frame'][frame][idx]=(coord_out_resize[0][idx][0].item(), coord_out_resize[0][idx][2].item(), coord_out_resize[0][idx][1].item())
            
            vis = True
            vis_3d = False
            if vis:
                    tmpimg = input_patch[0].cpu().numpy()
                    tmpimg = tmpimg * np.array(cfg.pixel_std).reshape(3,1,1) + np.array(cfg.pixel_mean).reshape(3,1,1)
                    tmpimg = (tmpimg).astype(np.uint8)
                    tmpimg = tmpimg[::-1, :, :]
                    tmpimg = np.transpose(tmpimg,(1,2,0)).copy()
                    tmpkps = np.zeros((3,18))
                    tmpkps[:2,:] = coord_out[0,:,:2].transpose(1,0) / cfg.output_shape[0] * cfg.input_shape[0]
                    tmpkps[2,:] = 1
                    tmpimg = vis_keypoints(tmpimg, tmpkps, skeleton)
                    tmpimg = cv2.resize(tmpimg,(img_patch.shape[1],img_patch.shape[0]))
                    file_name = pose_save_dir+'\\{0}.png'.format(str(frame).zfill(4))
                    cv2.imwrite(file_name, tmpimg)
            if vis_3d:
                #coord_out = coord_out.cpu().numpy()
                #coord_out = coord_out * np.array([img_patch.shape[1]/cfg.input_shape[1], img_patch.shape[0]/cfg.input_shape[0], 1])
                pred=coord_out_resize.squeeze() #remove first batch dimension

                ax=plt.subplot('121',projection='3d')
                plt.axis('off')
                show3D_pose(pred,ax,skeleton,radius=40)
                file_name = pose_save_dir + '\\{0}.png'.format(str(frame).zfill(4))
                plt.savefig(file_name)
                # cv2.imwrite(file_name, tmpimg)

            frame+=1
            print('Processing Frame:',round(time.time()-time_start,2),'s')

        motion['feature']['fpb'] = np.mean(num_frame_between_beats)
        if REDU:
            motion_base[len(motion_base)-1] = motion
        else:
            motion_base[len(motion_base)] = motion
        #with open(motion_base_dir, 'w') as f:
        #    json.dump(motion_base, f)
    print('done with', num_beat + 1, 'beats! (This should be even for a normal dance)')
    print('num_frame between beats:')
    print(num_frame_between_beats)

if __name__ == "__main__":
    main()
