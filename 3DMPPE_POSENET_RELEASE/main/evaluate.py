import argparse
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--demo', type=str, help='running model with a video or an image')
    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        args.gpu_ids = str(np.argmin(mem_info()))

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = 0 if not gpus[0].isdigit() else int(gpus[0])
        gpus[1] = len(mem_info()) if not gpus[1].isdigit() else int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    
    return args

def prepare_input(imgpath, bbox):
    cvimg = cv2.imread(imgpath, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(cvimg, np.ndarray):
        raise IOError("Fail to read %s" % imgpath)
    img_height, img_width, img_channels = cvimg.shape

    scale, rot, do_flip, color_scale, do_occlusion = 1.0, 0.0, False, [1.0, 1.0, 1.0], False

    # 3. crop patch from img and perform data augmentation (flip, rot, color scale, synthetic occlusion)
    # Adds black padding and ensured 255x255 dimensions for input even by introducing stretching.
    img_patch, trans = generate_patch_image(cvimg, bbox, do_flip, scale, rot, do_occlusion)

    for i in range(img_channels):
        img_patch[:, :, i] = np.clip(img_patch[:, :, i] * color_scale[i], 0, 255)

    transform = transforms.Compose([ \
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)] \
        )

    img_patch = transform(img_patch)

    input_img = img_patch.unsqueeze(0)

    return input_img

def plot_3dkeypoints(pred, skeleton):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.axes._axes import _log as matplotlib_axes_logger
    mpl.use("TKAgg", warn=False, force=True)
    matplotlib_axes_logger.setLevel('ERROR')

    def set_axes_radius(ax, origin, radius):
        ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
        ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
        ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

    def set_axes_equal(ax):
        '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc..  This is one possible solution to Matplotlib's
        ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

        Call this function before plt.show()

        Input
          ax: a matplotlib axis, e.g., as output from plt.gca().
        '''

        limits = np.array([
            ax.get_xlim3d(),
            ax.get_ylim3d(),
            ax.get_zlim3d(),
        ])

        origin = np.mean(limits, axis=1)
        radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
        set_axes_radius(ax, origin, radius)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(pred) + 2)]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]

    for i, (x, y, z) in enumerate(pred):
        ax.scatter(x, y, z, c=np.array(colors[i]), marker='o')

    for (joint_a, joint_b) in skeleton:
        ax.plot([pred[joint_a][0], pred[joint_b][0]],
                [pred[joint_a][1], pred[joint_b][1]],
                [pred[joint_a][2], pred[joint_b][2]])

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')
    # The model by default predicts x and z to be the the width-wise and length-wise directions of the original image
    # With depth added into the y axis. So this rotation shows the original 2D perspective (i.e. the 2D keypoints)
    ax.view_init(azim=-90, elev=-90)
    ax.legend()
    set_axes_equal(ax)
    plt.show()

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
def main():

    args = parse_args()
    cfg.set_args(args.gpu_ids)
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
    detector = YOLOv3( model_def="D:\\Robot Dance\\AI-Project-Portfolio\\3DMPPE_POSENET_RELEASE\\common\\detectors\\yolo\\config\\yolov3.cfg",
                       class_path="D:\\Robot Dance\\AI-Project-Portfolio\\3DMPPE_POSENET_RELEASE\\common\\detectors\\yolo\\data\\coco.names",
                       weights_path="D:\\Robot Dance\\AI-Project-Portfolio\\3DMPPE_POSENET_RELEASE\\common\\detectors\\yolo\\weights\\yolov3.weights",
                       classes=('person',),
                       max_batch_size=16,
                       device=torch.device('cuda:{}'.format(cfg.gpu_ids[0])))
    print('loading yolo elapse:',round(time.time()-time_1,2),'s')
    skeleton = ( (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6) )
    fig = plt.figure(figsize=(10,10)) 
    transform = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)]
                            )
    ##load model

    preds = []
    if args.demo == 'image':

        image_path = 'D:\\Robot Dance\\AI-Project-Portfolio\\3DMPPE_POSENET_RELEASE\\image\\test_sample_20200711.jpg'
        raw_image = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        input_img = raw_image.copy()

        ##using yolo to get human bounding box
        detections = detector.predict_single(input_img)
        # if not detections.cpu().numpy():
        #     detections = (0,0,input_img.shape[1],input_img.shape[0],1,1)
        for i, (x1, y1, x2, y2, conf, cls_conf, cls_pred) in enumerate(detections):
            x1 = int(round(x1.item()))
            x2 = int(round(x2.item()))
            y1 = int(round(y1.item()))
            y2 = int(round(y2.item()))
        img_patch = (input_img[y1:y2, x1:x2, ::-1]).copy().astype(np.float32)
        input_patch = cv2.resize(img_patch,(cfg.input_shape))
       
        with torch.no_grad():          
            # forward
            input_patch = transform(input_patch).unsqueeze(0)
            coord_out = tester.model(input_patch)

            vis = True
            itr=3
            if vis:
                filename = str(itr)
                tmpimg = input_patch[0].cpu().numpy()
                tmpimg = tmpimg * np.array(cfg.pixel_std).reshape(3,1,1) + np.array(cfg.pixel_mean).reshape(3,1,1)
                tmpimg = (tmpimg).astype(np.uint8)
                tmpimg = tmpimg[::-1, :, :]
                tmpimg = np.transpose(tmpimg,(1,2,0)).copy()
                tmpkps = np.zeros((3,18))
                tmpkps[:2,:] = coord_out[0,:,:2].cpu().numpy().transpose(1,0) / cfg.output_shape[0] * cfg.input_shape[0]
                tmpkps[2,:] = 1
                tmpimg = vis_keypoints(tmpimg, tmpkps, skeleton)
                tmpimg = cv2.resize(tmpimg,(img_patch.shape[1],img_patch.shape[0]))
                cv2.imwrite(filename + '_output.jpg', tmpimg)
            # print(tmpkps)

            coord_out = coord_out.cpu().numpy()
            coord_out = coord_out * np.array([img_patch.shape[1]/cfg.input_shape[1], img_patch.shape[0]/cfg.input_shape[0], 1])
            #coord_out = (coord_out - np.mean(coord_out, axis = 1))/np.std(coord_out, axis = 1)
            coord_out = coord_out[:, :-1, :]
            print(coord_out)
            print(coord_out.shape)
            np.save('coord_out_0711.npy', coord_out)
            vis_3d=True
            if vis_3d:
                pred=coord_out.squeeze() #remove first batch dimension
                plot_3dkeypoints(pred, skeleton)
            motion_frame = {}
            for idx in range(coord_out.shape[1]-1):
                motion_frame[idx]=(coord_out[0][idx][0].item(), coord_out[0][idx][2].item(), coord_out[0][idx][1].item())
        with open('motionFrame.json', 'w') as f:
            json.dump(motion_frame, f)
    elif args.demo == 'video':
        video = cv2.VideoCapture('D:\\Robot Dance\\AI-Project-Portfolio\\dance_videos\\test_sample.mp4')
        ret_val, image = video.read()
        motion={}
        frame=0
        with torch.no_grad():
            while True:
                time_start = time.time()
                ret_val, raw_image = video.read()
                if not ret_val:
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
                for i, (x1, y1, x2, y2, conf, cls_conf, cls_pred) in enumerate(detections):
                    x1 = int(round(x1.item()))
                    x2 = int(round(x2.item()))
                    y1 = int(round(y1.item()))
                    y2 = int(round(y2.item()))
                img_patch = (input_img[y1:y2, x1:x2, ::-1]).copy().astype(np.float32)
                input_patch = cv2.resize(img_patch,(cfg.input_shape))

                input_patch = transform(input_patch).unsqueeze(0)
                coord_out = tester.model(input_patch)
                print('Running model time:',round(time.time()-time_start,2),'s')

                motion[frame] = {}
                for idx in range(coord_out.shape[1]-1):
                    motion[frame][idx]=(coord_out[0][idx][0].item(), coord_out[0][idx][2].item(), coord_out[0][idx][1].item())

                vis = True
                vis_3d = False

                if vis:
                    tmpimg = input_patch[0].cpu().numpy()
                    tmpimg = tmpimg * np.array(cfg.pixel_std).reshape(3,1,1) + np.array(cfg.pixel_mean).reshape(3,1,1)
                    tmpimg = (tmpimg).astype(np.uint8)
                    tmpimg = tmpimg[::-1, :, :]
                    tmpimg = np.transpose(tmpimg,(1,2,0)).copy()
                    tmpkps = np.zeros((3,18))
                    tmpkps[:2,:] = coord_out[0,:,:2].cpu().numpy().transpose(1,0) / cfg.output_shape[0] * cfg.input_shape[0]
                    tmpkps[2,:] = 1
                    tmpimg = vis_keypoints(tmpimg, tmpkps, skeleton)
                    tmpimg = cv2.resize(tmpimg,(img_patch.shape[1],img_patch.shape[0]))
                    file_name = '../demo_result/{0}.png'.format(str(frame).zfill(4))
                    cv2.imwrite(file_name, tmpimg)
                    frame+=1
                if vis_3d:
                    coord_out = coord_out.cpu().numpy()
                    coord_out = coord_out * np.array([img_patch.shape[1]/cfg.input_shape[1], img_patch.shape[0]/cfg.input_shape[0], 1])
                    pred=coord_out.squeeze() #remove first batch dimension

                    ax=plt.subplot('121',projection='3d')
                    plt.axis('off')
                    show3D_pose(pred,ax,skeleton,radius=40)

                    # plot_3dkeypoints(pred,skeleton)
                    file_name = '../test_sample_result/{0}.png'.format(str(frame).zfill(4))
                    plt.savefig(file_name)
                    # cv2.imwrite(file_name, tmpimg)
                    frame+=1
                print('Processing Frame:',round(time.time()-time_start,2),'s')

            with open('motionTest.json', 'w') as f:
                json.dump(motion, f)


if __name__ == "__main__":
    main()
