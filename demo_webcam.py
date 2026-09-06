import os
import sys
import argparse
import contextlib
import torch
import numpy as np
import cv2
from loguru import logger
from glob import glob
from train.core.tester import Tester
from train.utils.one_euro_filter import OneEuroFilter
from multi_person_tracker import MPT
from multi_person_tracker import Sort
#Dataloader
from torch.utils.data import DataLoader
#from train.core.tester_smpl import Tester
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')  # override with osmesa for software rendering
#os.environ["DISPLAY"] = ":0"e
sys.path.append('')

from tools import saveCSVFileFromListOfDicts, saveCSVFileFromListOfDictsFollowingSkeletonOrder, get_azure_kinect_skeleton, get_smpl_skeleton, get_colors

"""
Resolve --device (auto/cpu/cuda) to an actual torch device
"""
def resolveDevice(requested):
    if (requested == 'cuda') and (not torch.cuda.is_available()):
        logger.warning('CUDA requested but not available, falling back to CPU')
        requested = 'cpu'
    if (requested == 'auto'):
        requested = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(requested)

"""
Check if a path exists
"""
def checkIfPathIsDirectory(filename):
    if (filename is None):  
       return False
    return os.path.isdir(filename) 

"""
Easy way to switch inputs
"""
def getCaptureDeviceFromPath(videoFilePath,videoWidth,videoHeight,videoFramerate=30,useMjpg=False):
  #------------------------------------------
  def configureLiveCapture(cap):
     if useMjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
     cap.set(cv2.CAP_PROP_FPS,videoFramerate)
     cap.set(cv2.CAP_PROP_FRAME_WIDTH, videoWidth)
     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, videoHeight)
     return cap
  #------------------------------------------
  if (videoFilePath=="esp"):
     from espStream import ESP32CamStreamer
     cap = ESP32CamStreamer()
  if (videoFilePath=="screen"):
     from screenStream import ScreenGrabber
     cap =  ScreenGrabber(region=(0,0,videoWidth,videoHeight))
  elif (videoFilePath=="webcam"):
     cap = configureLiveCapture(cv2.VideoCapture(0))
  elif (videoFilePath=="/dev/video0"):
     cap = configureLiveCapture(cv2.VideoCapture(0))
  elif (videoFilePath=="/dev/video1"):
     cap = configureLiveCapture(cv2.VideoCapture(1))
  elif (videoFilePath=="/dev/video2"):
     cap = configureLiveCapture(cv2.VideoCapture(2))
  else:
     if (checkIfPathIsDirectory(videoFilePath) and (not "/dev/" in videoFilePath) ):
        from folderStream import FolderStreamer
        cap = FolderStreamer(path=videoFilePath,width=videoWidth,height=videoHeight)
     else:
        cap = cv2.VideoCapture(videoFilePath)
  return cap




def save_matlab_visualization(hmr_output,output_filename="skeleton.png"):
    from matplotlib import pyplot as plt
                       
    joints     = hmr_output['joints3d'].cpu().numpy()
    hmr_joints = hmr_output['joints3d'][:, 0:22, :].cpu().numpy()
    camera_translation = hmr_output['pred_cam_t'].cpu().numpy() * 0.5 
    ax = None

    radius =1
    skeleton = get_smpl_skeleton()
    #joints = joints[:len(skeleton)+1]
    if True:
                            fig = plt.figure(figsize=(12, 7))
                            ax = fig.add_subplot(111, projection='3d')
                            ax.set_aspect('auto')
    kp_3d = joints[0]
    for i, (j1, j2) in enumerate(skeleton):
                                if kp_3d[j1].shape[0] == 4:
                                    x, y, z, v = [np.array([kp_3d[j1, c], kp_3d[j2, c]]) for c in range(4)]
                                else:
                                    x, y, z = [np.array([kp_3d[j1, c], kp_3d[j2, c]]) for c in range(3)]
                                    v = [1, 1]
                                ax.plot(x, y, z, lw=2, c=get_colors()['purple'] / 255)
                                for j in range(2):
                                    if v[j] > 0: # if visible
                                        ax.plot(x[j], y[j], z[j], lw=2, c=get_colors()['blue'] / 255, marker='o')
                                    else: # nonvisible
                                        ax.plot(x[j], y[j], z[j], lw=2, c=get_colors()['red'] / 255, marker='x')

    pelvis_joint = 0
    RADIUS = radius  # space around the subject
    xroot, yroot, zroot = kp_3d[pelvis_joint, 0], kp_3d[pelvis_joint, 1], kp_3d[pelvis_joint, 2]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(-90, -90)
    if ax is None:
                            plt.show()
    plt.savefig(output_filename)



"""
This only encodes the body 3D points
""" 
def encode_smpl_skeleton_to_dict(hmr_output): 
    # Extract the first sample in the batch and take the first 22 SMPL joints
    joints_3d = hmr_output['joints3d'][0, :22, :]

    # Convert to numpy if needed
    if isinstance(joints_3d, torch.Tensor):
        joints_3d = joints_3d.cpu().numpy()

    SMPL_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"]

    # Encode as a dictionary
    joints_dict = {SMPL_JOINT_NAMES[i]: joints_3d[i].tolist() for i in range(joints_3d.shape[0])}

    return joints_dict

"""
This encodes both body AND hand AND face 3D points
""" 
def encode_smplx_skeleton_to_dict(hmr_output): 
    # --- Extract body joints ---
    joints_body = hmr_output['joints3d'][0]  # shape (22, 3)

    if isinstance(joints_body, torch.Tensor):
        joints_body = joints_body.cpu().numpy()
   
    #Documentation: https://github.com/open-mmlab/mmhuman3d/issues/149

    SMPL_JOINT_NAMES = [
        "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
        "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
        "neck", "left_collar", "right_collar", "head", "left_shoulder",
        "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"
    ]

    joints_dict = {SMPL_JOINT_NAMES[i]: joints_body[i].tolist() for i in range(joints_body.shape[0])}

    # --- Hand joint names based on MANO/SMPL-X ---
    HAND_JOINT_NAMES = [
        'wrist',
        'thumb1', 'thumb2', 'thumb3', 'thumb4',
        'index1', 'index2', 'index3', 'index4',
        'middle1', 'middle2', 'middle3', 'middle4',
        'ring1', 'ring2', 'ring3', 'ring4',
        'pinky1', 'pinky2', 'pinky3', 'pinky4'
    ]

    # --- Add hands ---
    for side in ['left', 'right']:
        hand_key_3d = f"{side}_hand_3d"
        if hand_key_3d in hmr_output:
            hand_joints = hmr_output[hand_key_3d][0]  # shape (15, 3) or 21?
            if isinstance(hand_joints, torch.Tensor):
                hand_joints = hand_joints.cpu().numpy()
            
            # If hand has fewer joints than HAND_JOINT_NAMES, truncate names
            names_to_use = HAND_JOINT_NAMES[:hand_joints.shape[0]]
            for name, joint in zip(names_to_use, hand_joints):
                joints_dict[f"{side}_hand_{name}"] = joint.tolist()

    # --- Head joint names (facial landmarks) ---
    HEAD_JOINT_NAMES = [
        'jaw', 'mouth_left', 'mouth_right', 'mouth_top', 'mouth_bottom',
        'nose', 'left_eye', 'right_eye', 'left_eyebrow', 'right_eyebrow'
    ]
    # If more joints are available, just number them
    if 'head_3d' in hmr_output:
        head_joints = hmr_output['head_3d'][0]
        if isinstance(head_joints, torch.Tensor):
            head_joints = head_joints.cpu().numpy()
        
        for i, joint in enumerate(head_joints):
            if i < len(HEAD_JOINT_NAMES):
                joints_dict[f"head_{HEAD_JOINT_NAMES[i]}"] = joint.tolist()
            elif i == 58:
                joints_dict[f"head_right_ear"] = joint.tolist()
            elif i == 59:
                joints_dict[f"head_left_ear"] = joint.tolist()
            else:
                joints_dict[f"head_extra_{i}"] = joint.tolist()

    return joints_dict



def save_skeleton_dict_to_json(skeleton_dict, output_filename="skeleton.json"):
    import json
    # Ensure the output directory exists
    directoryPath = os.path.dirname(output_filename)
  
    if (directoryPath!=""):
       os.makedirs(directoryPath, exist_ok=True)

    # Save as pretty-printed JSON
    with open(output_filename, "w") as f:
        json.dump(skeleton_dict, f, indent=4)

    print(f"[INFO] Skeleton saved to {output_filename}")


def to_json_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_json_serializable(v) for v in obj]
    else:
        return obj


def save_raw_dict_to_json(skeleton_dict, output_filename="skeleton.json"):
    import json


    skeleton_dict = to_json_serializable(skeleton_dict)
    # Ensure the output directory exists
    directoryPath = os.path.dirname(output_filename)
  
    if (directoryPath!=""):
       os.makedirs(directoryPath, exist_ok=True)

    # Save as pretty-printed JSON
    with open(output_filename, "w") as f:
        json.dump(skeleton_dict, f, indent=4)

    print(f"[INFO] Skeleton saved to {output_filename}")

def scale_and_embed_frame(frame, target_w=1280, target_h=720):
    """
    Resize + embed a frame into a fixed 1280x720 canvas
    while keeping aspect ratio (letterbox padding).

    Returns:
        embedded_rgb  - final 720p RGB frame
        scale_x, scale_y - scaling factors applied
        pad_x,   pad_y   - padding applied (useful for remapping detections)
    """
    h, w = frame.shape[:2]

    # Compute scale to fit inside (1280x720)
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize using that scale
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Compute padding amounts
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2

    # Create canvas and embed resized frame centered
    embedded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    embedded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    return embedded, scale, pad_x, pad_y

def rotate_frame_90(frame, clockwise=True):
    """
    Rotate an image by 90 degrees.

    Args:
        frame: input HxWxC image
        clockwise: True = rotate 90° CW, False = rotate 90° CCW

    Returns:
        rotated_frame: rotated image (HxWxC)
    """
    if clockwise:
        # 90° clockwise = transpose then flip horizontally
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    else:
        # 90° counter-clockwise
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)


def main(args):

    input_image_folder = args.image_folder
    output_path        = args.output_folder
    #os.makedirs(output_path, exist_ok=True)

    logger.add(
        os.path.join(output_path, 'demo.log'),
        level='INFO',
        colorize=False,
    )
    logger.info(f'Demo options: \n {args}')
    bbox_one_euro_filter = OneEuroFilter(
        np.zeros(4),
        np.zeros(4),
        min_cutoff=0.004,
        beta=0.4,
    )

    device = resolveDevice(args.device)
    logger.info(f'Running on {device}')

    tester  = Tester(args)
    history = list()

    if True:
        all_image_folder = [input_image_folder]
        torch.set_float32_matmul_precision('medium')
        # autocast only helps on CUDA, on CPU it hurts more than it helps
        autocast = torch.cuda.amp.autocast() if device.type == 'cuda' else contextlib.nullcontext()
        with autocast, torch.no_grad():
            mot = MPT(
                device=device,
                batch_size=4,
                display=False,
                detector_type='yolo',
                output_format='dict',
                yolo_img_size=416
            )
            videoWidth, videoHeight = args.size
            videoFramerate = args.fps
            cap = getCaptureDeviceFromPath(args.input,videoWidth,videoHeight,videoFramerate,useMjpg=args.mjpg)

            frameNumber = 0
            use_bbox_filter = False
            while True:
                frameNumber+=1
                if True:#frameNumber%2==0:
                    try:
                      ret, frame = cap.read()          
                      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    except Exception as e:
                      print("Error opening image",e)
                      break
  
                    #frame = rotate_frame_90(frame)
                    frame, scale, pad_x, pad_y = scale_and_embed_frame(frame)

                    input_tensor = torch.tensor(frame).permute(2, 0, 1).unsqueeze(0) / 255.0
                    detection    = mot.detector(input_tensor.to(device))

                    # Concatenate boxes and scores from all predictions at once
                    if detection:
                        #import ipdb; ipdb.set_trace()
                        t = torch.ones(4) * frameNumber
                        if use_bbox_filter:
                            boxes = torch.cat([bbox_one_euro_filter(t,pred['boxes']) for pred in detection], dim=0)
                        else:
                            boxes = torch.cat([pred['boxes'] for pred in detection], dim=0)
                        scores = torch.cat([pred['scores'] for pred in detection], dim=0)
                        # Apply threshold in a vectorized way
                        mask = scores > 0.7
                        # Filter and add scores as a new column
                        filtered_boxes = boxes[mask]
                        filtered_scores = scores[mask].unsqueeze(1)
                        # Merge boxes and scores using concatenation
                        dets = torch.cat([filtered_boxes, filtered_scores], dim=1).cpu().detach().numpy()
                    else:
                        dets = np.empty((0, 5))
                        #print("No detection")

                    detections = [dets]
                    detection = mot.prepare_output_detections(detections)
                    if len(detection[0]) > 0:

                        saveFilename = None
                        if (args.save):
                           saveFilename = 'colorFrame_0_%05d.jpg' % frameNumber

                        hmr_output=tester.run_on_single_image_tensor(frame, detection, save=saveFilename)
                        #---------------------------------------------------------------------------------------
                        #At this point hmr_output has the resolved pose data..!
                        #---------------------------------------------------------------------------------------
                        
                        #pose3DAsDictionary = encode_smpl_skeleton_to_dict(hmr_output)
                        pose3DAsDictionary = encode_smplx_skeleton_to_dict(hmr_output)

                        history.append(pose3DAsDictionary)
 
                        #We can dump the skeleton to disk as skeleton_00000.json etc.
                        if (args.save):
                           save_skeleton_dict_to_json(pose3DAsDictionary,output_filename="skeleton_%05u.json" % frameNumber)
                           del hmr_output['vertices'] # <- Remove this because it is really big
                           save_raw_dict_to_json(hmr_output,output_filename="raw_%05u.json" % frameNumber)

                        #Uncomment to also do a matlab visualization
                        #save_matlab_visualization(hmr_output,output_filename="skeleton_%05u.png" % frameNumber)
                    else:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        cv2.imshow('D-Pose', frame)
                        if (args.save):
                           saveFilename = 'colorFrame_0_%05d.jpg' % frameNumber
                           cv2.imwrite(saveFilename, frame)
                        

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

    saveCSVFileFromListOfDictsFollowingSkeletonOrder("3DPointsAzureKinect.csv",history,get_azure_kinect_skeleton())
  
    input_path = args.input_path
    csv_path = input_path + "_3DBody.csv"
    saveCSVFileFromListOfDicts(csv_path,history)
    if (args.save):
        os.system("ffmpeg -nostdin -framerate %u -start_number 1 -i colorFrame_0_%%05d.jpg -s %ux%u  -y -r %u -pix_fmt yuv420p -threads 8 livelastRun3DHiRes.mp4 && rm colorFrame_0_*.jpg " % (videoFramerate,videoWidth,videoHeight,videoFramerate)) # 
    del tester.model

    logger.info('================= END =================')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    parser.add_argument('--save', help='Save .json / visualization output', action=argparse.BooleanOptionalAction)

    parser.add_argument('--input', type=str, default='/dev/video0', dest='input',
                        help='From Device (path to files, videos , /dev/videoX or screen )')

    parser.add_argument('--from', type=str, default='/dev/video0', dest='input',
                        help='Alias for --input (path to files, videos, /dev/videoX or screen)')

    parser.add_argument('--size', type=int, nargs=2, default=[1280, 720], metavar=('WIDTH', 'HEIGHT'),
                        help='Webcam capture resolution (default: 1280 720)')

    parser.add_argument('--fps', type=int, default=30,
                        help='Webcam capture framerate (default: 30)')

    parser.add_argument('--mjpg', action='store_true',
                        help='Request MJPG streaming from the capture device (helps reach higher resolution/fps on USB webcams)')

    parser.add_argument('--cfg', type=str, default='configs/dpose_conf.yaml',
                        help='config file that defines model hyperparams')

    parser.add_argument('--ckpt', type=str, default='data/ckpt/paper_arxiv.ckpt',
                        help='checkpoint path')

    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'],
                        help='device to run inference on (auto picks cuda when available)')

    parser.add_argument('--image_folder', type=str, default='demo_images',
                        help='input image folder')

    parser.add_argument('--output_folder', type=str, default='demo_images/results',
                        help='output folder to write results')

    parser.add_argument('--tracker_batch_size', type=int, default=1,
                        help='batch size of object detector used for bbox tracking')
                        
    parser.add_argument('--display', action='store_true',
                        help='visualize the 3d body projection on image')

    parser.add_argument('--detector', type=str, default='yolo', choices=['yolo', 'maskrcnn'],
                        help='object detector to be used for bbox tracking')

    parser.add_argument('--yolo_img_size', type=int, default=416,
                        help='input image size for yolo detector')
    parser.add_argument('--eval_dataset', type=str, default=None)
    parser.add_argument('--dataframe_path', type=str, default='data/ssp_3d_test.npz')
    parser.add_argument('--data_split', type=str, default='test')

    args = parser.parse_args()
    main(args)
