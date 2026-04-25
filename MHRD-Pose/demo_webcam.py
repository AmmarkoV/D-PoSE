import os
import sys
import argparse
import torch
import numpy as np
import cv2
from loguru import logger

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _THIS_DIR)

os.environ['PYOPENGL_PLATFORM'] = 'egl'

from multi_person_tracker import MPT
from torchvision.transforms import Normalize
from train.core import constants
from train.utils.image_utils import crop
from train.core.config import update_hparams
from mhr_hmr import MHRHMR


def checkIfPathIsDirectory(filename):
    if filename is None:
        return False
    return os.path.isdir(filename)


def getCaptureDeviceFromPath(videoFilePath, videoWidth, videoHeight, videoFramerate=30):
    if videoFilePath == "esp":
        from espStream import ESP32CamStreamer
        cap = ESP32CamStreamer()
    if videoFilePath == "screen":
        from screenStream import ScreenGrabber
        cap = ScreenGrabber(region=(0, 0, videoWidth, videoHeight))
    elif videoFilePath == "webcam":
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, videoFramerate)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, videoWidth)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, videoHeight)
    elif videoFilePath == "/dev/video0":
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, videoFramerate)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, videoWidth)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, videoHeight)
    elif videoFilePath == "/dev/video1":
        cap = cv2.VideoCapture(1)
        cap.set(cv2.CAP_PROP_FPS, videoFramerate)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, videoWidth)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, videoHeight)
    elif videoFilePath == "/dev/video2":
        cap = cv2.VideoCapture(2)
        cap.set(cv2.CAP_PROP_FPS, videoFramerate)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, videoWidth)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, videoHeight)
    else:
        if checkIfPathIsDirectory(videoFilePath) and "/dev/" not in videoFilePath:
            from folderStream import FolderStreamer
            cap = FolderStreamer(path=videoFilePath, width=videoWidth, height=videoHeight)
        else:
            cap = cv2.VideoCapture(videoFilePath)
    return cap


def scale_and_embed_frame(frame, target_w=1280, target_h=720):
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    embedded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    embedded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return embedded, scale, pad_x, pad_y


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


def encode_mhr_output_to_dict(mhr_output):
    """Encode MHRHMR model output to a JSON-serializable dictionary."""
    if mhr_output is None:
        return {}
    result = {}
    for src, dst in [('pred_identity', 'identity_coeffs'),
                     ('pred_pose', 'lbs_model_params'),
                     ('pred_expr', 'face_expr_coeffs'),
                     ('pred_cam_t', 'pred_cam_t')]:
        if src in mhr_output and mhr_output[src] is not None:
            result[dst] = to_json_serializable(mhr_output[src])
    if 'skel_state' in mhr_output and mhr_output['skel_state'] is not None:
        result['mhr_skel_state'] = to_json_serializable(mhr_output['skel_state'])
    if 'joints3d' in mhr_output and mhr_output['joints3d'] is not None:
        result['joints3d'] = to_json_serializable(mhr_output['joints3d'])
    if 'joints3d_smpl' in mhr_output and mhr_output['joints3d_smpl'] is not None:
        result['joints3d_smpl'] = to_json_serializable(mhr_output['joints3d_smpl'])
    return result


def save_skeleton_dict_to_json(skeleton_dict, output_filename="skeleton.json"):
    import json
    directory = os.path.dirname(output_filename)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(output_filename, "w") as f:
        json.dump(skeleton_dict, f, indent=4)
    print(f"[INFO] Skeleton saved to {output_filename}")


def save_raw_dict_to_json(data, output_filename="out.json"):
    import json
    data = to_json_serializable(data)
    directory = os.path.dirname(output_filename)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(output_filename, "w") as f:
        json.dump(data, f, indent=4)
    print(f"[INFO] Saved to {output_filename}")


class MHRTester:
    def __init__(self, args):
        self.args = args
        self.model_cfg = update_hparams(args.cfg)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)

        mhr_model_path = os.path.join(_PROJECT_ROOT, self.model_cfg.MHR.MODEL_PT)
        logger.info(f'Loading MHR model from {mhr_model_path}')
        mhr_model = torch.load(mhr_model_path, map_location='cpu', weights_only=False)
        mhr_model.eval()

        self.model = MHRHMR(
            backbone=self.model_cfg.MODEL.BACKBONE,
            img_res=self.model_cfg.DATASET.IMG_RES,
            pretrained_ckpt=self.model_cfg.TRAINING.PRETRAINED_CKPT,
            hparams=self.model_cfg,
            mhr_model=mhr_model,
        ).to(self.device)

        self._load_checkpoint()
        self.model.eval()

        self.renderer = self._build_renderer()

    def _load_checkpoint(self):
        if not self.args.ckpt:
            logger.warning('No checkpoint provided — running with random weights')
            return
        logger.info(f'Loading checkpoint from {self.args.ckpt}')
        ckpt = torch.load(self.args.ckpt, map_location='cpu', weights_only=False)
        state_dict = ckpt.get('state_dict', ckpt)
        # MHRTrainer stores MHRHMR as self.model, so keys are prefixed 'model.'
        stripped = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                stripped[k[len('model.'):]] = v
        result = self.model.load_state_dict(stripped, strict=False)
        logger.info(f'Checkpoint loaded — missing: {len(result.missing_keys)}, unexpected: {len(result.unexpected_keys)}')

    def _build_renderer(self):
        try:
            from train.utils.renderer_pyrd import Renderer
            faces_path = os.path.join(_PROJECT_ROOT, 'mhr_portable_dump', 'faces.npy')
            if os.path.isfile(faces_path):
                faces = np.load(faces_path)
            else:
                # Fall back to portable dump
                sys.path.insert(0, _PROJECT_ROOT)
                from load_mhr_portable import load_portable
                portable = load_portable(
                    os.path.join(_PROJECT_ROOT, 'mhr_portable_dump', 'mhr_dump_lod1.pt'))
                faces = portable['interesting']['character.mesh.faces']
                if isinstance(faces, torch.Tensor):
                    faces = faces.cpu().numpy()
                else:
                    faces = np.array(faces)
            return Renderer(focal_length=1468.6047, img_w=1280, img_h=720,
                            faces=faces, same_mesh_color=False)
        except Exception as e:
            logger.warning(f'Could not build renderer: {e} — display disabled')
            return None

    @torch.no_grad()
    def run_on_single_image_tensor(self, image_tensor, detections, save=None):
        img = image_tensor
        orig_height, orig_width = img.shape[:2]
        dets = detections

        if len(dets[0]) > 0:
            dets = dets[0]
        else:
            cv2.imshow('front', img[:, :, ::-1])
            if save is not None:
                cv2.imwrite(save, img[:, :, ::-1])
            return None

        batch_size = len(dets)
        bbox_scale = torch.tensor([d[2] / 200.0 for d in dets], device=self.device)
        bbox_center = torch.tensor([[d[0], d[1]] for d in dets], device=self.device)
        img_h = torch.tensor([orig_height] * batch_size, device=self.device, dtype=torch.float)
        img_w = torch.tensor([orig_width] * batch_size, device=self.device, dtype=torch.float)

        bbox_centers_np = bbox_center.cpu().numpy()
        bbox_scales_np = bbox_scale.cpu().numpy()

        cropped_imgs = [
            np.transpose(
                crop(img, bbox_centers_np[i], bbox_scales_np[i],
                     [self.model_cfg.DATASET.IMG_RES, self.model_cfg.DATASET.IMG_RES]
                     ).astype('float32'),
                (2, 0, 1)
            ) / 255.0
            for i in range(batch_size)
        ]

        inp_images = self.normalize_img(
            torch.from_numpy(np.stack(cropped_imgs)).to(self.device)
        )

        mhr_output, _, _, _, _ = self.model(
            inp_images,
            bbox_center=bbox_center,
            bbox_scale=bbox_scale,
            img_w=img_w,
            img_h=img_h,
        )

        if self.renderer is not None:
            vertices = (mhr_output['vertices'] + mhr_output['pred_cam_t'].unsqueeze(1))
            if isinstance(vertices, torch.Tensor):
                vertices = vertices.detach().cpu().numpy()
            front_view = self.renderer.render_front_view(vertices, bg_img_rgb=img.copy())
            cv2.imshow('front', front_view[:, :, ::-1])
            if save is not None:
                cv2.imwrite(save, front_view[:, :, ::-1])
        else:
            cv2.imshow('front', img[:, :, ::-1])
            if save is not None:
                cv2.imwrite(save, img[:, :, ::-1])

        return mhr_output


def main(args):
    input_image_folder = args.image_folder
    output_path = args.output_folder

    logger.add(
        os.path.join(output_path, 'demo.log'),
        level='INFO',
        colorize=False,
    )
    logger.info(f'Demo options: \n {args}')

    tester = MHRTester(args)
    history = []

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.set_float32_matmul_precision('medium')

    with torch.no_grad():
        mot = MPT(
            device=torch.device('cuda'),
            batch_size=4,
            display=False,
            detector_type='yolo',
            output_format='dict',
            yolo_img_size=416,
        )

        videoWidth = 1280
        videoHeight = 720
        videoFramerate = 30
        cap = getCaptureDeviceFromPath(args.input, videoWidth, videoHeight, videoFramerate)

        frameNumber = 0
        while True:
            frameNumber += 1
            try:
                ret, frame = cap.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except Exception as e:
                print("Error opening image", e)
                break

            frame, scale, pad_x, pad_y = scale_and_embed_frame(frame)

            input_tensor = torch.tensor(frame).permute(2, 0, 1).unsqueeze(0) / 255.0
            detection = mot.detector(input_tensor.cuda())

            if detection:
                boxes = torch.cat([pred['boxes'] for pred in detection], dim=0)
                scores = torch.cat([pred['scores'] for pred in detection], dim=0)
                mask = scores > 0.7
                filtered_boxes = boxes[mask]
                filtered_scores = scores[mask].unsqueeze(1)
                dets = torch.cat([filtered_boxes, filtered_scores], dim=1).cpu().detach().numpy()
            else:
                dets = np.empty((0, 5))

            detections = [dets]
            detection = mot.prepare_output_detections(detections)

            if len(detection[0]) > 0:
                saveFilename = None
                if args.save:
                    saveFilename = 'colorFrame_0_%05d.jpg' % frameNumber

                mhr_output = tester.run_on_single_image_tensor(frame, detection, save=saveFilename)

                pose3d = encode_mhr_output_to_dict(mhr_output)
                history.append(pose3d)

                if args.save:
                    save_skeleton_dict_to_json(
                        pose3d,
                        output_filename="mhr_%05u.json" % frameNumber,
                    )
            else:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow('front', frame_bgr)
                if args.save:
                    cv2.imwrite('colorFrame_0_%05d.jpg' % frameNumber, frame_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    save_raw_dict_to_json(history, output_filename="mhr_history.json")
    if args.save:
        os.system(
            "ffmpeg -framerate %u -start_number 1 -i colorFrame_0_%%05d.jpg "
            "-s %ux%u -y -r %u -pix_fmt yuv420p -threads 8 "
            "livelastRun3DHiRes.mp4 && rm colorFrame_0_*.jpg"
            % (videoFramerate, videoWidth, videoHeight, videoFramerate)
        )
    del tester.model
    logger.info('================= END =================')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--save', help='Save .json / visualization output',
                        action=argparse.BooleanOptionalAction)

    parser.add_argument('--input', type=str, default='/dev/video0',
                        help='From Device (path to files, videos, /dev/videoX or screen)')

    parser.add_argument('--cfg', type=str,
                        default=os.path.join(_THIS_DIR, 'config_mhr.yaml'),
                        help='config file that defines model hyperparams')

    parser.add_argument('--ckpt', type=str, default='',
                        help='MHRTrainer checkpoint path (.ckpt)')

    parser.add_argument('--image_folder', type=str, default='demo_images',
                        help='input image folder')

    parser.add_argument('--output_folder', type=str, default='demo_images/results',
                        help='output folder to write results')

    parser.add_argument('--tracker_batch_size', type=int, default=1,
                        help='batch size of object detector used for bbox tracking')

    parser.add_argument('--display', action='store_true',
                        help='visualize the 3d body projection on image')

    parser.add_argument('--detector', type=str, default='yolo',
                        choices=['yolo', 'maskrcnn'],
                        help='object detector to be used for bbox tracking')

    parser.add_argument('--yolo_img_size', type=int, default=416,
                        help='input image size for yolo detector')

    parser.add_argument('--eval_dataset', type=str, default=None)
    parser.add_argument('--dataframe_path', type=str, default='data/ssp_3d_test.npz')
    parser.add_argument('--data_split', type=str, default='test')

    args = parser.parse_args()
    main(args)
