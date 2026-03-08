
import os
import sys
import cv2
import torch
import tqdm
import matplotlib.pyplot as plt
from loguru import logger
import numpy as np
import time

from . import constants
from multi_person_tracker import MPT
from torchvision.transforms import Normalize
from train.utils.train_utils import load_pretrained_model

from .config import update_hparams
from ..models.hmr import HMR
from ..models.head.smplx_cam_head import SMPLXCamHead
from ..utils.renderer_pyrd import Renderer
from ..utils.image_utils import crop



sys.path.insert(0, "/home/ammar/Documents/3dParty/MHR")
sys.path.insert(0, "/home/ammar/Documents/3dParty/MHR/tools/mhr_smpl_conversion")
#MHR_ROOT = "/home/ammar/Documents/3dParty/MHR/tools/mhr_smpl_conversion"
#sys.path.append(MHR_ROOT)

from conversion import Conversion
from mhr.mhr import MHR



def load_mhr_faces():
    from pathlib import Path
    import trimesh
    candidates = [
        Path("/home/ammar/Documents/3dParty/D-PoSE/assets/lod1.fbx"),
        Path(__file__).resolve().parents[2] / "assets" / "lod1.fbx",
        Path.cwd() / "assets" / "lod1.fbx",
    ]

    errors = []

    for path in candidates:
        try:
            if not path.is_file():
                errors.append(f"{path} -> not found")
                continue

            obj = trimesh.load(str(path), file_type="fbx", force="scene")

            # FBX often loads as a Scene, not a Mesh
            if isinstance(obj, trimesh.Scene):
                for name, geom in obj.geometry.items():
                    if hasattr(geom, "faces") and geom.faces is not None and len(geom.faces) > 0:
                        faces = np.asarray(geom.faces, dtype=np.int64)
                        print(f"Loaded MHR faces from {path} geometry={name} faces={faces.shape}", flush=True)
                        return faces
                errors.append(f"{path} -> scene loaded but no geometry with faces")
                continue

            if hasattr(obj, "faces") and obj.faces is not None and len(obj.faces) > 0:
                faces = np.asarray(obj.faces, dtype=np.int64)
                print(f"Loaded MHR faces from {path} faces={faces.shape}", flush=True)
                return faces

            errors.append(f"{path} -> loaded object has no faces")

        except Exception as e:
            errors.append(f"{path} -> {repr(e)}")

    raise RuntimeError("Could not load MHR faces:\n" + "\n".join(errors))



class Tester:
    def __init__(self, args):

        self.args = args
        self.model_cfg = update_hparams(args.cfg)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
        self.bboxes_dict = {}

        self.model = self._build_model()
        self.smplx_cam_head = SMPLXCamHead(img_res=self.model_cfg.DATASET.IMG_RES).to(self.device)
        self._load_pretrained_model()
        self.model.eval()
        #self.renderer = Renderer(focal_length=1468.6047, img_w=1280, img_h=720, faces=self.smplx_cam_head.smplx.faces, same_mesh_color=False)

        print("A: pretrained model loaded", flush=True)

        print("B: before MHR.from_files", flush=True)
        #self.mhr_model = MHR.from_files(lod=1, device=torch.device("cpu"))
        self.mhr_model = torch.load("mhr_model.pt", map_location="cpu",weights_only=False)
        print("C: after MHR.from_files", flush=True)

        print("D: before Conversion", flush=True)
        self.converter = Conversion(mhr_model=self.mhr_model, smpl_model=self.smplx_cam_head.smplx, method="pytorch", batch_size=32, )
        print("E: after Conversion", flush=True)

        print("F: before Renderer", flush=True)
        #mhr_faces = self.mhr_model.character.mesh.faces
        mhr_faces = load_mhr_faces()
        self.renderer = Renderer(focal_length=1468.6047, img_w=1280, img_h=720, faces=mhr_faces, same_mesh_color=False, )
        print("G: after Renderer", flush=True)

    def _convert_smplx_output_to_mhr(self, hmr_output, single_identity=False, is_tracking=False):
        """
        Convert HMR/SMPL-X output to MHR parameters and MHR vertices.
        Returns a dictionary with MHR-only data.
        """
        # SMPL-X vertices in camera space
        smplx_vertices_cam = hmr_output["vertices"] + hmr_output["pred_cam_t"].unsqueeze(1)

        conversion_results = self.converter.convert_smpl2mhr(
        smpl_vertices=smplx_vertices_cam,
        smpl_parameters=None,
        single_identity=single_identity,
        exclude_expression=False,
        is_tracking=is_tracking,
        return_mhr_meshes=False,
        return_mhr_parameters=True,
        return_mhr_vertices=True,
        return_fitting_errors=True,
    )

        mhr_vertices_cm = conversion_results.result_vertices
        mhr_params = conversion_results.result_parameters
        mhr_errors = conversion_results.result_errors

        # MHR package uses centimeters
        if isinstance(mhr_vertices_cm, torch.Tensor):
            mhr_vertices_m = mhr_vertices_cm / 100.0
        else:
            mhr_vertices_m = mhr_vertices_cm / 100.0

        # Optional: reconstruct through the real MHR forward model for consistency
        identity_coeffs = mhr_params["identity_coeffs"]
        model_parameters = mhr_params["lbs_model_params"]
        face_expr_coeffs = mhr_params["face_expr_coeffs"]

        with torch.no_grad():
            mhr_forward_verts_cm, mhr_skel_state = self.mhr_model(
            identity_coeffs=identity_coeffs,
            model_parameters=model_parameters,
            face_expr_coeffs=face_expr_coeffs,
        )

        mhr_forward_verts_m = mhr_forward_verts_cm / 100.0

        return {
        "mhr_vertices_cm": mhr_vertices_cm,
        "mhr_vertices_m": mhr_vertices_m,
        "mhr_forward_vertices_cm": mhr_forward_verts_cm,
        "mhr_forward_vertices_m": mhr_forward_verts_m,
        "mhr_parameters": mhr_params,
        "mhr_skel_state": mhr_skel_state,
        "mhr_errors": mhr_errors,
       }

    def _build_model(self):
        self.hparams = self.model_cfg
        model = HMR(
            backbone=self.hparams.MODEL.BACKBONE,
            img_res=self.hparams.DATASET.IMG_RES,
            pretrained_ckpt=self.hparams.TRAINING.PRETRAINED_CKPT,
            hparams=self.hparams,
        ).to(self.device)
        return model

    def _load_pretrained_model(self):
        # ========= Load pretrained weights ========= #
        logger.info(f'Loading pretrained model from {self.args.ckpt}')
        ckpt = torch.load(self.args.ckpt,weights_only=False)['state_dict']
        load_pretrained_model(self.model, ckpt, overwrite_shape_mismatch=True, remove_lightning=True)
        logger.info(f'Loaded pretrained weights from \"{self.args.ckpt}\"')

    def run_detector(self, all_image_folder):
        # run multi object tracker
        mot = MPT(
            device=self.device,
            batch_size=self.args.tracker_batch_size,
            display=False,
            detector_type=self.args.detector,
            output_format='dict',
            yolo_img_size=self.args.yolo_img_size,
        )
        bboxes = []
        for fold_id, image_folder in enumerate(all_image_folder):
            bboxes.append(mot.detect(image_folder))

        return bboxes

    def load_yolov5_bboxes(self, all_bbox_folder):
        # run multi object tracker
        for fold_id, bbox_folder in enumerate(all_bbox_folder):
            for bbox_file in os.listdir(bbox_folder):
                bbox = np.loadtxt(os.path.join(bbox_folder, bbox_file))
                fname = os.path.join('/'.join(bbox_folder.split('/')[-3:-1]),bbox_file.replace('.txt','.png'))
                self.bboxes_dict[fname] = bbox

    @torch.no_grad()
    def run_on_image_folder(self, all_image_folder, detections, output_folder, visualize_proj=True):
        for fold_idx, image_folder in enumerate(all_image_folder):
            image_file_names = [
                os.path.join(image_folder, x)
                for x in os.listdir(image_folder)
                if x.endswith('.png') or x.endswith('.jpg') or x.endswith('.jpeg')
            ]
            image_file_names = (sorted(image_file_names))
            for img_idx, img_fname in tqdm.tqdm(enumerate(image_file_names)):
                
                dets = detections[fold_idx][img_idx]
                if len(dets) < 1:
                    continue

                img = cv2.cvtColor(cv2.imread(img_fname), cv2.COLOR_BGR2RGB)
                orig_height, orig_width = img.shape[:2]
                inp_images = torch.zeros(len(dets), 3, self.model_cfg.DATASET.IMG_RES,
                                         self.model_cfg.DATASET.IMG_RES, device=self.device, dtype=torch.float)

                batch_size = inp_images.shape[0]
                bbox_scale = []
                bbox_center = []

                for det_idx, det in enumerate(dets):
                    bbox = det
                    bbox_scale.append(bbox[2] / 200.)
                    bbox_center.append([bbox[0], bbox[1]])
                    rgb_img = crop(img, bbox_center[-1], bbox_scale[-1],[self.model_cfg.DATASET.IMG_RES, self.model_cfg.DATASET.IMG_RES])
                    rgb_img = np.transpose(rgb_img.astype('float32'), (2, 0, 1)) / 255.0
                    rgb_img = torch.from_numpy(rgb_img)
                    norm_img = self.normalize_img(rgb_img)
                    inp_images[det_idx] = norm_img.float().to(self.device)

                bbox_center = torch.tensor(bbox_center).cuda().float()
                bbox_scale = torch.tensor(bbox_scale).cuda().float()
                img_h = torch.tensor(orig_height).repeat(batch_size).cuda().float()
                img_w = torch.tensor(orig_width).repeat(batch_size).cuda().float()
                focal_length = ((img_w * img_w + img_h * img_h) ** 0.5).cuda().float()
                if not self.hparams.DATASET.USE_DEPTH:
                    hmr_output = self.model(inp_images, bbox_center=bbox_center, bbox_scale=bbox_scale, img_w=img_w, img_h=img_h)
                else:
                    hmr_output,orig_depth,_,_,segmentation = self.model(inp_images, bbox_center=bbox_center, bbox_scale=bbox_scale, img_w=img_w, img_h=img_h)

                focal_length = (img_w * img_w + img_h * img_h) ** 0.5
                pred_vertices_array = (hmr_output['vertices'] + hmr_output['pred_cam_t'].unsqueeze(1)).detach().cpu().numpy()
                #renderer = Renderer(focal_length=focal_length[0], img_w=img_w[0], img_h=img_h[0],
                #                    faces=self.smplx_cam_head.smplx.faces,
                #                    same_mesh_color=False)
                front_view = self.renderer.render_front_view(pred_vertices_array,
                                                        bg_img_rgb=img.copy())

                # save rendering results
                basename = img_fname.split('/')[-1]
                filename = basename + "pred_%s.jpg" % 'bedlam'
                filename_orig = basename + "orig_%s.jpg" % 'bedlam'
                front_view_path = os.path.join(output_folder, filename)
                orig_path = os.path.join(output_folder, filename_orig)
                cv2.imwrite(front_view_path, front_view[:, :, ::-1])
                logger.info(f'Writing output files to {output_folder}')
                '''
                cv2.imwrite(orig_path, img[:, :, ::-1])
                #import ipdb; ipdb.set_trace()
                cv2.imwrite(os.path.join(output_folder, filename + "depth.jpg"), orig_depth[0].detach().cpu().numpy().reshape(orig_depth.shape[2],orig_depth.shape[3])*255)
                segmentation = segmentation[0].detach().cpu().numpy()
                segmentation = np.argmax(segmentation, axis=0)
                plt.imshow(segmentation, cmap='viridis')
                plt.axis('off')
                plt.savefig(os.path.join(output_folder, filename + "segmentation.jpg"))
                #cv2.imwrite(os.path.join(output_folder, filename + "segmentation.jpg"), segmentation)
                #import ipdb; ipdb.set_trace()
                '''
                #renderer.delete()

    """
    @torch.no_grad()
    def _run_on_single_image_tensor(self, image_tensor,detections):
        dets = detections

        img = image_tensor#.transpose(1,0,2)
        orig_height, orig_width = img.shape[:2]
        if len(dets[0])>0:
            dets = dets[0]
        inp_images = torch.zeros(len(dets), 3, self.model_cfg.DATASET.IMG_RES,
                                         self.model_cfg.DATASET.IMG_RES, device=self.device, dtype=torch.float)

        batch_size = inp_images.shape[0]
       # import ipdb; ipdb.set_trace()
        bbox_scale  = []
        bbox_center = []
        #print(len(dets)," detections")
 
        for det_idx, det in enumerate(dets):
            #import ipdb; ipdb.set_trace()
            bbox = det#[0]
            bbox_scale.append(bbox[2] / 200.)
            bbox_center.append([bbox[0], bbox[1]])
            rgb_img  = crop(img, bbox_center[-1], bbox_scale[-1],[self.model_cfg.DATASET.IMG_RES, self.model_cfg.DATASET.IMG_RES])
            rgb_img  = np.transpose(rgb_img.astype('float32'), (2, 0, 1)) / 255.0
            rgb_img  = torch.from_numpy(rgb_img)
            norm_img = self.normalize_img(rgb_img)
            inp_images[det_idx] = norm_img.float().to(self.device)

        bbox_center = torch.tensor(bbox_center).cuda().float()
        bbox_scale  = torch.tensor(bbox_scale).cuda().float()
        img_h = torch.tensor(orig_height).repeat(batch_size).cuda().float()
        img_w = torch.tensor(orig_width).repeat(batch_size).cuda().float()
        focal_length = ((img_w * img_w + img_h * img_h) ** 0.5).cuda().float()
            
        hmr_output,orig_depth,_,_,segmentation = self.model(inp_images, bbox_center=bbox_center, bbox_scale=bbox_scale, img_w=img_w, img_h=img_h)
        focal_length = (img_w * img_w + img_h * img_h) ** 0.5
        pred_vertices_array = (hmr_output['vertices'] + hmr_output['pred_cam_t'].unsqueeze(1))#.detach().cpu().numpy()
        #import ipdb; ipdb.set_trace()
        #renderer = Renderer(focal_length=focal_length[0], img_w=img_w[0], img_h=img_h[0],
        #                            faces=self.smplx_cam_head.smplx.faces,
        #                            same_mesh_color=False)
        front_view = self.renderer.render_front_view(np.array(pred_vertices_array),
                                                        bg_img_rgb=img.copy())
        cv2.imshow('front', front_view[:, :, ::-1])
        #start = time.time()
        #side_view = renderer.render_side_view(pred_vertices_array)
        #end = time.time()
        #final_time = end-start
        #print("Time taken to render side view: ", final_time)
                # save rendering results
        #cv2.imshow('side', side_view[:, :, ::-1])
        '''
                cv2.imwrite(front_view_path, front_view[:, :, ::-1])
                cv2.imwrite(orig_path, img[:, :, ::-1])
                #import ipdb; ipdb.set_trace()
                cv2.imwrite(os.path.join(output_folder, filename + "depth.jpg"), orig_depth[0].detach().cpu().numpy().reshape(orig_depth.shape[2],orig_depth.shape[3])*255)
                segmentation = segmentation[0].detach().cpu().numpy()
                segmentation = np.argmax(segmentation, axis=0)
                plt.imshow(segmentation, cmap='viridis')
                plt.axis('off')
                plt.savefig(os.path.join(output_folder, filename + "segmentation.jpg"))
                #cv2.imwrite(os.path.join(output_folder, filename + "segmentation.jpg"), segmentation)
                #import ipdb; ipdb.set_trace()
        '''
        return hmr_output
        #renderer.delete()
        
    @torch.no_grad()
    def run_on_single_image_tensor(self, image_tensor, detections,render=True,save=None):
        dets = detections

        # Load the image tensor and get its dimensions
        img = image_tensor
        orig_height, orig_width = img.shape[:2]

        if len(dets[0]) > 0:
            dets = dets[0]

        # Create tensor for batch of images
        batch_size = len(dets)
        inp_images = torch.zeros(batch_size, 3, self.model_cfg.DATASET.IMG_RES, self.model_cfg.DATASET.IMG_RES, device=self.device, dtype=torch.float)

        # Precompute bbox scale and center for all detections
        bbox_scale = torch.tensor([det[2] / 200. for det in dets], device=self.device)
        bbox_center = torch.tensor([[det[0], det[1]] for det in dets], device=self.device)

        # Precompute image width and height tensors
        img_h = torch.tensor([orig_height] * batch_size, device=self.device, dtype=torch.float)
        img_w = torch.tensor([orig_width] * batch_size, device=self.device, dtype=torch.float)
        
        # Calculate focal length once
        #focal_length = torch.sqrt(img_w ** 2 + img_h ** 2).to(self.device)
        bbox_centers_np = bbox_center.cpu().numpy()
        bbox_scales_np = bbox_scale.cpu().numpy()

        # Crop images in a vectorized manner using list comprehension
        cropped_imgs = [
            np.transpose(
                crop(
                    img,
                    bbox_centers_np[i],
                    bbox_scales_np[i],
                    [self.model_cfg.DATASET.IMG_RES, self.model_cfg.DATASET.IMG_RES]
                ).astype('float32'),
                (2, 0, 1)
            ) / 255.0
            for i in range(len(dets))
        ]

        # Stack into one tensor and move to device
        inp_images = torch.from_numpy(np.stack(cropped_imgs)).to(self.device)

        # Apply normalization to the entire batch at once
        inp_images = self.normalize_img(inp_images)

        # Pass the batch through the model
        hmr_output, orig_depth, _, _, segmentation = self.model(
            inp_images, 
            bbox_center=bbox_center, 
            bbox_scale=bbox_scale, 
            img_w=img_w, 
            img_h=img_h
        )

        # Compute vertices array and render front view
        pred_vertices_array = (hmr_output['vertices'] + hmr_output['pred_cam_t'].unsqueeze(1)).cpu().numpy()
        #import ipdb; ipdb.set_trace()
        #renderer = Renderer(
        #    focal_length=focal_length[0], 
        #    img_w=img_w[0], 
        #    img_h=img_h[0],
        #    faces=self.smplx_cam_head.smplx.faces,
        #    same_mesh_color=False
        #)
        if render:
            front_view = self.renderer.render_front_view(pred_vertices_array, bg_img_rgb=img.copy())
            cv2.imshow('front', front_view[:, :, ::-1])
            if (save is not None):
                cv2.imwrite(save, front_view[:, :, ::-1])
        else:
            cv2.imshow('front', img[:, :, ::-1])
            if (save is not None):
                cv2.imwrite(save, img[:, :, ::-1])
            
        #cv2.imshow('front', img[:, :, ::-1])
        return hmr_output
        #side_view = self.renderer.render_side_view(pred_vertices_array)
        #cv2.imshow('side', side_view[:, :, ::-1])
    """
    @torch.no_grad()
    def run_on_single_image_tensor(self, image_tensor, detections, render=True, save=None):
        dets = detections
        img = image_tensor
        orig_height, orig_width = img.shape[:2]

        if len(dets[0]) > 0:
            dets = dets[0]
        else:
            if render:
                cv2.imshow('front', img[:, :, ::-1])
                if save is not None:
                    cv2.imwrite(save, img[:, :, ::-1])
            return None

        batch_size = len(dets)

        bbox_scale = torch.tensor([det[2] / 200.0 for det in dets], device=self.device)
        bbox_center = torch.tensor([[det[0], det[1]] for det in dets], device=self.device)

        img_h = torch.tensor([orig_height] * batch_size, device=self.device, dtype=torch.float)
        img_w = torch.tensor([orig_width] * batch_size, device=self.device, dtype=torch.float)

        bbox_centers_np = bbox_center.cpu().numpy()
        bbox_scales_np = bbox_scale.cpu().numpy()

        cropped_imgs = [
            np.transpose(
                crop(
                    img,
                    bbox_centers_np[i],
                    bbox_scales_np[i],
                    [self.model_cfg.DATASET.IMG_RES, self.model_cfg.DATASET.IMG_RES]
                ).astype('float32'),    
                (2, 0, 1)
            ) / 255.0
            for i in range(len(dets))
        ]

        inp_images = torch.from_numpy(np.stack(cropped_imgs)).to(self.device)
        inp_images = self.normalize_img(inp_images)

        # Existing HMR / SMPL-X inference
        hmr_output, orig_depth, _, _, segmentation = self.model(
            inp_images,
            bbox_center=bbox_center,
            bbox_scale=bbox_scale,
            img_w=img_w,
            img_h=img_h
        )

        # Intercept here: SMPL-X -> MHR
        mhr_output = self._convert_smplx_output_to_mhr(
            hmr_output,
            single_identity=False,
            is_tracking=True,
        )

        # Render only MHR
        if render:
            mhr_vertices_for_render = mhr_output["mhr_forward_vertices_m"]
            if isinstance(mhr_vertices_for_render, torch.Tensor):
                mhr_vertices_for_render = mhr_vertices_for_render.detach().cpu().numpy()
    
            front_view = self.renderer.render_front_view(
                mhr_vertices_for_render,
                bg_img_rgb=img.copy()
            )
            cv2.imshow('front', front_view[:, :, ::-1])

            if save is not None:
                cv2.imwrite(save, front_view[:, :, ::-1])
        else:
            cv2.imshow('front', img[:, :, ::-1])
            if save is not None:
                cv2.imwrite(save, img[:, :, ::-1])

        return mhr_output

    @torch.no_grad()
    def run_on_hbw_folder(self, all_image_folder, detections, output_folder, data_split='test', visualize_proj=True):
        img_names = []
        verts = []
        image_file_names = []
        for fold_idx, image_folder in enumerate(all_image_folder):
            image_file_names = [
                os.path.join(image_folder, x)
                for x in os.listdir(image_folder)
                if x.endswith('.png') or x.endswith('.jpg') or x.endswith('.jpeg')
            ]
            image_file_names = (sorted(image_file_names))
            print(image_folder, len(image_file_names))

            for img_idx, img_fname in tqdm.tqdm(enumerate(image_file_names)):
                if detections:
                    dets = detections[fold_idx][img_idx]
                    if len(dets) < 1:
                        img_names.append('/'.join(img_fname.split('/')[-4:]).replace(data_split + '_small_resolution', data_split))
                        template_verts = self.smplx_cam_head.smplx().vertices[0].detach().cpu().numpy()
                        verts.append(template_verts)
                        continue
                else:
                    match_fname = '/'.join(img_fname.split('/')[-3:])
                    if match_fname not in self.bboxes_dict.keys():
                        img_names.append('/'.join(img_fname.split('/')[-4:]).replace(data_split + '_small_resolution', data_split))
                        template_verts = self.smplx_cam_head.smplx().vertices[0].detach().cpu().numpy()
                        verts.append(template_verts)
                        continue
                    dets = self.bboxes_dict[match_fname]
                img = cv2.cvtColor(cv2.imread(img_fname), cv2.COLOR_BGR2RGB)
                orig_height, orig_width = img.shape[:2]
                inp_images = torch.zeros(1, 3, self.model_cfg.DATASET.IMG_RES,
                                         self.model_cfg.DATASET.IMG_RES, device=self.device, dtype=torch.float)
                batch_size = inp_images.shape[0]
                bbox_scale = []
                bbox_center = []
                if len(dets.shape)==1:
                    dets = np.expand_dims(dets, 0)
                for det_idx, det in enumerate(dets):
                    if det_idx>=1:
                        break
                    bbox = det
                    bbox_scale.append(bbox[2] / 200.)
                    bbox_center.append([bbox[0], bbox[1]])
                    rgb_img = crop(img, bbox_center[-1], bbox_scale[-1],[self.model_cfg.DATASET.IMG_RES, self.model_cfg.DATASET.IMG_RES])
                    rgb_img = np.transpose(rgb_img.astype('float32'), (2, 0, 1)) / 255.0
                    rgb_img = torch.from_numpy(rgb_img)
                    norm_img = self.normalize_img(rgb_img)
                    inp_images[det_idx] = norm_img.float().to(self.device)

                bbox_center = torch.tensor(bbox_center).cuda().float()
                bbox_scale = torch.tensor(bbox_scale).cuda().float()
                img_h = torch.tensor(orig_height).repeat(batch_size).cuda().float()
                img_w = torch.tensor(orig_width).repeat(batch_size).cuda().float()
                focal_length = ((img_w * img_w + img_h * img_h) ** 0.5).cuda().float()
                if self.hparams.DATASET.USE_DEPTH:
                    hmr_output,_,_,_ = self.model(inp_images, bbox_center=bbox_center, bbox_scale=bbox_scale, img_w=img_w, img_h=img_h)
                else:
                    hmr_output = self.model(inp_images, bbox_center=bbox_center, bbox_scale=bbox_scale, img_w=img_w, img_h=img_h)
                img_names.append('/'.join(img_fname.split('/')[-4:]).replace(data_split + '_small_resolution', data_split))
                template_verts = self.smplx_cam_head.smplx(betas=hmr_output['pred_shape'], pose2rot=False).vertices[0].detach().cpu().numpy()
                verts.append(template_verts)
                if visualize_proj:
                    focal_length = (img_w * img_w + img_h * img_h) ** 0.5

                    pred_vertices_array = (hmr_output['vertices'][0] + hmr_output['pred_cam_t']).unsqueeze(0).detach().cpu().numpy()
                    renderer = Renderer(focal_length=focal_length, img_w=img_w, img_h=img_h,
                                        faces=self.smplx_cam_head.smplx.faces,
                                        same_mesh_color=False)
                    front_view = renderer.render_front_view(pred_vertices_array,
                                                            bg_img_rgb=img.copy())

                    # save rendering results
                    basename = img_fname.split('/')[-3]+'_'+img_fname.split('/')[-2]+'_'+img_fname.split('/')[-1]
                    filename = basename + "pred_%s.jpg" % 'bedlam'
                    filename_orig = basename + "orig_%s.jpg" % 'bedlam'
                    front_view_path = os.path.join(output_folder, filename)
                    orig_path = os.path.join(output_folder, filename_orig)
                    logger.info(f'Writing output files to {output_folder}')
                    cv2.imwrite(front_view_path, front_view[:, :, ::-1])
                    cv2.imwrite(orig_path, img[:, :, ::-1])
                    renderer.delete()
        np.savez(os.path.join(output_folder, data_split + '_hbw_prediction.npz'), image_name=img_names, v_shaped=verts)
               
    def run_on_dataframe(self, dataframe_path, output_folder, visualize_proj=True):
        dataframe = np.load(dataframe_path)
        centers = dataframe['center']
        scales = dataframe['scale']
        image = dataframe['image']
        for ind, center in tqdm.tqdm(enumerate(centers)):
            center = centers[ind]
            scale = scales[ind]
            img = image[ind]
            orig_height, orig_width = img.shape[:2]
            rgb_img = crop(img, center, scale, [self.hparams.DATASET.IMG_RES, self.hparams.DATASET.IMG_RES])

            rgb_img = np.transpose(rgb_img.astype('float32'), (2, 0, 1)) / 255.0
            rgb_img = torch.from_numpy(rgb_img).float().cuda()
            rgb_img = self.normalize_img(rgb_img)

            img_h = torch.tensor(orig_height).repeat(1).cuda().float()
            img_w = torch.tensor(orig_width).repeat(1).cuda().float()
            center = torch.tensor(center).cuda().float()
            scale = torch.tensor(scale).cuda().float()
            if not self.hparams.DATASET.USE_DEPTH:
                hmr_output = self.model(rgb_img.unsqueeze(0), bbox_center=center.unsqueeze(0), bbox_scale=scale.unsqueeze(0), img_w=img_w, img_h=img_h)
            else:
                hmr_output,_,_,_ = self.model(rgb_img.unsqueeze(0), bbox_center=center.unsqueeze(0), bbox_scale=scale.unsqueeze(0), img_w=img_w, img_h=img_h)
            # Need to convert SMPL-X meshes to SMPL using conversion tool before calculating error
            import trimesh
            mesh = trimesh.Trimesh(vertices=hmr_output['vertices'][0].detach().cpu().numpy(),faces=self.smplx_cam_head.smplx.faces)
            output_mesh_path = os.path.join(output_folder, str(ind)+'.obj')
            mesh.export(output_mesh_path)

            if visualize_proj:
                focal_length = (img_w * img_w + img_h * img_h) ** 0.5

                pred_vertices_array = (hmr_output['vertices'][0] + hmr_output['pred_cam_t']).unsqueeze(0).detach().cpu().numpy()
                renderer = Renderer(focal_length=focal_length, img_w=img_w, img_h=img_h,
                                    faces=self.smplx_cam_head.smplx.faces,
                                    same_mesh_color=False)
                front_view = renderer.render_front_view(pred_vertices_array,
                                                        bg_img_rgb=img.copy())

                # save rendering results
                basename = str(ind)
                filename = basename + "pred_%s.jpg" % 'bedlam'
                filename_orig = basename + "orig_%s.jpg" % 'bedlam'
                front_view_path = os.path.join(output_folder, filename)
                orig_path = os.path.join(output_folder, filename_orig)
                logger.info(f'Writing output files to {output_folder}')
                cv2.imwrite(front_view_path, front_view[:, :, ::-1])
                cv2.imwrite(orig_path, img[:, :, ::-1])
                renderer.delete()
