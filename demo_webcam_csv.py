import os
import sys
import argparse
import torch
import numpy as np
import cv2
import csv
from loguru import logger
from glob import glob
from train.core.tester import Tester
from train.utils.one_euro_filter import OneEuroFilter
from multi_person_tracker import MPT
from multi_person_tracker import Sort
#Dataloader
from torch.utils.data import DataLoader
#from train.core.tester_smpl import Tester
os.environ['PYOPENGL_PLATFORM'] = 'egl'
#os.environ["DISPLAY"] = ":0"e
sys.path.append('')

from tools import saveCSVFileFromListOfDicts, saveCSVFileFromListOfDictsFollowingSkeletonOrder, get_azure_kinect_skeleton, get_smpl_skeleton, get_colors

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
def getCaptureDeviceFromPath(videoFilePath,videoWidth,videoHeight,videoFramerate=30):
  #------------------------------------------
  if (videoFilePath=="esp"):
     from espStream import ESP32CamStreamer
     cap = ESP32CamStreamer()
  if (videoFilePath=="screen"):
     from screenStream import ScreenGrabber
     cap =  ScreenGrabber(region=(0,0,videoWidth,videoHeight))
  elif (videoFilePath=="webcam"):
     cap = cv2.VideoCapture(0)
     cap.set(cv2.CAP_PROP_FPS,videoFramerate)
     cap.set(cv2.CAP_PROP_FRAME_WIDTH, videoWidth)
     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, videoHeight)
  elif (videoFilePath=="/dev/video0"):
     cap = cv2.VideoCapture(0)
     cap.set(cv2.CAP_PROP_FPS,videoFramerate)
     cap.set(cv2.CAP_PROP_FRAME_WIDTH, videoWidth)
     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, videoHeight)
  elif (videoFilePath=="/dev/video1"):
     cap = cv2.VideoCapture(1)
     cap.set(cv2.CAP_PROP_FPS,videoFramerate)
     cap.set(cv2.CAP_PROP_FRAME_WIDTH, videoWidth)
     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, videoHeight)
  elif (videoFilePath=="/dev/video2"):
     cap = cv2.VideoCapture(2)
     cap.set(cv2.CAP_PROP_FPS,videoFramerate)
     cap.set(cv2.CAP_PROP_FRAME_WIDTH, videoWidth)
     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, videoHeight)
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
def encode_smpl_skeleton_to_dict(hmr_output, person_idx=0): 
    # Extract the first sample in the batch and take the first 22 SMPL joints
    joints_3d = hmr_output['joints3d'][person_idx, :22, :]

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
def encode_smplx_skeleton_to_dict(hmr_output, person_idx=0): 
    # --- Extract body joints ---
    joints_body = hmr_output['joints3d'][person_idx]  # shape (22, 3)

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
            hand_joints = hmr_output[hand_key_3d][person_idx]  # shape (15, 3) or 21?
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
        head_joints = hmr_output['head_3d'][person_idx]
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
# --------------------------------------------------
# CSV export for 3D skeletons (one row per skeleton)
# Columns: frame_id, skeleton_id, <joint>_x, <joint>_y, <joint>_z
# --------------------------------------------------
def _skeleton_dict_to_csv_row(skeleton_dict, frame_id, skeleton_id):
    row = {
        "frame_id": int(frame_id),
        "skeleton_id": int(skeleton_id),
    }
    if not isinstance(skeleton_dict, dict):
        return row

    for joint_name, coords in skeleton_dict.items():
        if coords is None:
            continue
        # coords is expected to be [x,y,z] (list/tuple/np)
        try:
            x, y, z = coords[:3]
        except Exception:
            continue
        row[f"{joint_name}_x"] = float(x)
        row[f"{joint_name}_y"] = float(y)
        row[f"{joint_name}_z"] = float(z)
    return row


def write_skeleton_rows_to_csv(rows, csv_path):
    if rows is None:
        rows = []

    # Ensure output directory exists
    directoryPath = os.path.dirname(csv_path)
    if directoryPath != "":
        os.makedirs(directoryPath, exist_ok=True)

    # Build header = fixed cols + all joint coord cols (sorted for stability)
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())

    fixed = ["frame_id", "skeleton_id"]
    other = sorted(k for k in all_keys if k not in fixed)
    fieldnames = fixed + other

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[INFO] Skeleton CSV saved to {csv_path}")



def main(args):

    input_image_folder = args.image_folder
    output_path        = args.output_folder
    os.makedirs(output_path, exist_ok=True)

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

    tester  = Tester(args)
    history = list()
    skeleton_rows = []  # accumulated CSV rows (one per skeleton per frame)

    if True:
        all_image_folder = [input_image_folder]
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        torch.set_float32_matmul_precision('medium')
        with torch.cuda.amp.autocast(), torch.no_grad():
            mot = MPT(
                device=torch.device('cuda'),
                batch_size=4,
                display=False,
                detector_type='yolo',
                output_format='dict',
                yolo_img_size=416
            )
            videoWidth     = 1280
            videoHeight    = 720
            videoFramerate = 30 
            cap = getCaptureDeviceFromPath(args.input,videoWidth,videoHeight,videoFramerate)

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
                    detection    = mot.detector(input_tensor.cuda())

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
                        
                        
                        #---------------------------------------------------------------------------------------
                        # At this point hmr_output has the resolved pose data for (potentially) multiple people.
                        # We export ONE CSV row per skeleton, and include both frame_id and skeleton_id.
                        #---------------------------------------------------------------------------------------

                        # Determine how many skeletons are present in this frame
                        num_skeletons = 1
                        try:
                            num_skeletons = int(hmr_output['joints3d'].shape[0])
                        except Exception:
                            num_skeletons = 1

                        frame_rows = []

                        for skeleton_id in range(num_skeletons):
                            #pose3DAsDictionary = encode_smpl_skeleton_to_dict(hmr_output, person_idx=skeleton_id)
                            pose3DAsDictionary = encode_smplx_skeleton_to_dict(hmr_output, person_idx=skeleton_id)

                            # Keep the original history behavior (first skeleton only) for compatibility
                            if skeleton_id == 0:
                                history.append(pose3DAsDictionary)

                            row = _skeleton_dict_to_csv_row(pose3DAsDictionary, frame_id=frameNumber, skeleton_id=skeleton_id)
                            skeleton_rows.append(row)
                            frame_rows.append(row)

                        # Per-frame dump (replaces skeleton_XXXXX.json)
                        if (args.save):
                            write_skeleton_rows_to_csv(frame_rows, csv_path="skeleton_%05u.csv" % frameNumber)

                            # Keep raw dump in JSON (very large, but useful for debugging)
                            if 'vertices' in hmr_output:
                                del hmr_output['vertices']  # <- Remove this because it is really big
                            save_raw_dict_to_json(hmr_output, output_filename="raw_%05u.json" % frameNumber)

                        #Uncomment to also do a matlab visualization
                        #save_matlab_visualization(hmr_output,output_filename="skeleton_%05u.png" % frameNumber)
                    else:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        cv2.imshow('front', frame)
                        if (args.save):
                           saveFilename = 'colorFrame_0_%05d.jpg' % frameNumber
                           cv2.imwrite(saveFilename, frame)
                        

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
    # Write a single CSV that contains ALL skeletons across ALL frames.
    # One row per (frame_id, skeleton_id).

    input_path = args.input
    csv_path   = input_path + "_3DBody.csv"
    video_path = input_path + "_3DBody.mp4" 
    write_skeleton_rows_to_csv(skeleton_rows, csv_path)

    # (Optional) keep the legacy CSV exports (single skeleton per frame, no IDs)
    saveCSVFileFromListOfDictsFollowingSkeletonOrder(os.path.join(output_path, "3DPointsAzureKinect_legacy.csv"), history, get_azure_kinect_skeleton())
    saveCSVFileFromListOfDicts(os.path.join(output_path, "3DPoints_legacy.csv"), history)
    if (args.save):
        os.system("ffmpeg -nostdin -framerate %u -start_number 1 -i colorFrame_0_%%05d.jpg -s %ux%u  -y -r %u -pix_fmt yuv420p -threads 8 %s && rm colorFrame_0_*.jpg " % (videoFramerate,videoWidth,videoHeight,videoFramerate,video_path)) # 
    del tester.model

    logger.info('================= END =================')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    parser.add_argument('--save', help='Save .json / visualization output', action=argparse.BooleanOptionalAction)

    parser.add_argument('--input', type=str, default='/dev/video0',
                        help='From Device (path to files, videos , /dev/videoX or screen )')

    parser.add_argument('--cfg', type=str, default='configs/dpose_conf.yaml',
                        help='config file that defines model hyperparams')

    parser.add_argument('--ckpt', type=str, default='data/ckpt/paper_arxiv.ckpt',
                        help='checkpoint path')

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
