"""
debug_vis.py — Visualize N samples from the training or validation dataset to
diagnose conversion correctness and loss alignment.

For each sample this script produces a row of panels:
  [input image] [pred mesh on crop] [GT mesh on crop]
  [pred 2D kps] [GT 2D kps (keypoints_orig)] [GT 2D kps (keypoints)]
  [3D joint scatter: pred vs GT, front view] [side view]

Usage (from the MHRD-Pose directory):
  python debug_vis.py --ckpt path/to/checkpoint.ckpt --n 4 --dataset agora-bfh
  python debug_vis.py --ckpt path/to/checkpoint.ckpt --n 4 --dataset 3dpw-val-cam --split val
  python debug_vis.py --n 4 --dataset agora-bfh   # no checkpoint: GT panels only

Output: debug_vis_out/sample_{i}.png  (one file per sample)
"""

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Make project root importable
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

#os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',    default=None,        help='Path to .ckpt checkpoint (omit for GT-only mode)')
    p.add_argument('--n',       type=int, default=4, help='Number of samples to visualise')
    p.add_argument('--dataset', default='agora-bfh', help='Dataset name passed to DatasetHMR')
    p.add_argument('--split',   default='train',     choices=['train', 'val'])
    p.add_argument('--out_dir', default='debug_vis_out')
    p.add_argument('--cfg',     default='config_mhr.yaml', help='Config yaml path')
    p.add_argument('--seed',    type=int, default=0)
    p.add_argument('--test',    action='store_true',
                   help='Run visual unit tests (saved to <out_dir>/tests/) instead of sample loop')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_numpy(x):
    """Convert a torch.Tensor or numpy array to a numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)


def unnorm_image(img_tensor):
    """Convert a [3,H,W] normalised image tensor to a uint8 HxWx3 numpy array."""
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)  # HWC
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def draw_keypoints_on_image(img_uint8, kps_xy, color, radius=3, conf=None):
    """Draw 2D keypoints on a copy of img_uint8 (HxWx3 uint8).

    kps_xy: [J, 2] in pixel coordinates
    conf:   [J] visibility/confidence or None
    """
    import cv2
    out = img_uint8.copy()
    for j, (x, y) in enumerate(kps_xy):
        if conf is not None and conf[j] < 0.1:
            continue
        cx, cy = int(round(float(x))), int(round(float(y)))
        cv2.circle(out, (cx, cy), radius, color, -1)
    return out


def render_mesh_on_image(renderer, verts_m, cam_t, img_uint8, focal_length):
    """Render a mesh (numpy [V,3] in metres) onto a uint8 crop image."""
    # renderer.__call__ expects float arrays; camera_translation is modified
    # in-place inside __call__ (flips x), so we copy.
    cam_t_copy = cam_t.copy()
    rendered = renderer(
        vertices=verts_m,
        camera_translation=cam_t_copy,
        image=img_uint8.astype(np.float32) / 255.0,
        focal_length=focal_length,
    )  # returns HxWx3 float [0,1] or [0,255] depending on renderer version
    if rendered.max() <= 1.0:
        rendered = (rendered * 255).astype(np.uint8)
    else:
        rendered = rendered.astype(np.uint8)
    return rendered



def render_mesh_stateless(
    vertices,
    faces,
    camera_translation,
    image,
    focal_length,
    camera_center=None,
    mesh_color=(0.9, 0.7, 0.7),
):
    """Render a mesh onto an image using an isolated EGL context.

    Coordinate-system contract
    ──────────────────────────
    Input vertices must be in SMPL / OpenCV camera-space convention:
      +X right,  +Y down,  +Z forward (depth increasing away from camera).

    The function converts to pyrender / OpenGL convention internally:
      +X right,  +Y up,    camera looks along −Z.

    camera_translation = [tx, ty, tz] in metres, pelvis-centred:
      tz — depth of the body root from the camera
      tx — rightward offset (positive → body right of principal point)
      ty — downward offset  (positive → body below principal point)

    Effective projection formula (after the Y-flip and tx-negation below):
      u = cx + fx * (x + tx) / (z + tz)
      v = cy + fy * (y + ty) / (z + tz)
    """
    import numpy as np
    import pyrender
    import trimesh

    H, W = image.shape[:2]

    if camera_center is None:
        camera_center = [W // 2, H // 2]

    # -- Step 1: build trimesh and flip Y-axis --------------------------------
    # Vertices arrive in camera-space Y-down (SMPL/OpenCV).
    # pyrender uses Y-up (OpenGL), so we rotate 180° around X:
    #   (x, y, z)  →  (x, −y, −z)
    # After this rotation:
    #   • Y is now up  (head projects above the principal point)
    #   • Z is negated, so positive-depth objects sit at negative pyrender Z,
    #     which is in front of the camera (pyrender looks along −Z). ✓
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    rot_180x = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot_180x)

    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',
        baseColorFactor=(*mesh_color, 1.0)
    )
    mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

    # -- Step 2: scene --------------------------------------------------------
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=(0.5, 0.5, 0.5))
    scene.add(mesh)

    # -- Step 3: camera pose --------------------------------------------------
    # pyrender's IntrinsicsCamera looks along −Z in its own frame.
    # We place the camera at world position (cam_t[0], cam_t[1], cam_t[2]).
    # The mesh root (pelvis) is at world origin; the camera is tz metres away.
    #
    # tx is negated here to reconcile two sign conventions:
    #   • In SMPL/PARE cam_t, positive tx means the person is to the RIGHT of
    #     the principal point, so u > cx.
    #   • After the 180°X mesh flip, a vertex at world x>0 still has x>0.
    #   • pyrender projection:  u = cx + fx * X_cam / (−Z_cam)
    #     where X_cam = x_world − cam_pos_x = x_world − (−tx) = x_world + tx. ✓
    cam_t = camera_translation.copy()
    cam_t[0] *= -1   # negate tx: camera is placed at (−tx, ty, tz)

    # 4×4 rigid camera pose: identity rotation (camera aligned with world axes),
    # translation column = cam_t.  pyrender interprets this as the camera-to-world
    # transform, so the camera origin is at position cam_t in world space.
    cam_pose = np.eye(4)
    cam_pose[:3, 3] = cam_t  # [−tx, ty, tz]

    camera = pyrender.IntrinsicsCamera(
        fx=focal_length[0],   # horizontal focal length in pixels
        fy=focal_length[1],   # vertical focal length in pixels
        cx=camera_center[0],  # principal point x (usually image-width / 2)
        cy=camera_center[1],  # principal point y (usually image-height / 2)
    )
    scene.add(camera, pose=cam_pose)

    # -- Step 4: lighting -----------------------------------------------------
    # Single directional light placed above-left-front; no pose rotation needed
    # because DirectionalLight direction is encoded in the pose matrix.
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    light_pose = np.eye(4)
    light_pose[:3, 3] = [0, 1, 1]   # light position hint (direction light ignores t)
    scene.add(light, pose=light_pose)

    # -- Step 5: render with an isolated EGL context --------------------------
    # Creating a second OffscreenRenderer while a first one is alive causes
    # EGL_BAD_ALLOC on some drivers.  We always create + delete here so no
    # context persists between calls.
    renderer = pyrender.OffscreenRenderer(W, H)
    try:
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    finally:
        renderer.delete()   # release EGL context immediately

    # -- Step 6: composite over background ------------------------------------
    color = color[:, :, :3]
    if image.max() > 1:
        image = image.astype(np.float32) / 255.0
    color = color.astype(np.float32) / 255.0
    mask = (color.sum(axis=2) > 0)[..., None]   # rendered pixels (depth > 0)
    out = color * mask + image * (1 - mask)

    return (out * 255).astype(np.uint8)

def plot_joints_3d(ax, joints_pred, joints_gt, title=''):
    """Scatter pred (red) and GT (blue) 3D joints on a matplotlib Axes3D."""
    if joints_pred is not None:
        j = joints_pred
        # Pelvis-centre
        pel = (j[1] + j[2]) / 2.0
        j = j - pel
        ax.scatter(j[:, 0], j[:, 2], j[:, 1], c='red',  s=20, label='pred', depthshade=False)

    if joints_gt is not None:
        j = joints_gt
        pel = (j[1] + j[2]) / 2.0
        j = j - pel
        ax.scatter(j[:, 0], j[:, 2], j[:, 1], c='blue', s=20, label='GT',   depthshade=False)

    ax.set_xlabel('X'); ax.set_ylabel('Z'); ax.set_zlabel('Y')
    ax.set_title(title, fontsize=7)
    ax.legend(fontsize=6)


# ---------------------------------------------------------------------------
# Rotation helpers (used by visual tests)
# ---------------------------------------------------------------------------

def _rot_x(v, deg):
    """Rotate [N,3] vertices around X axis."""
    rad = np.radians(deg)
    c, s = np.cos(rad), np.sin(rad)
    R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
    return v @ R.T

def _rot_y(v, deg):
    """Rotate [N,3] vertices around Y axis."""
    rad = np.radians(deg)
    c, s = np.cos(rad), np.sin(rad)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    return v @ R.T

def _rot_z(v, deg):
    """Rotate [N,3] vertices around Z axis."""
    rad = np.radians(deg)
    c, s = np.cos(rad), np.sin(rad)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    return v @ R.T


# ---------------------------------------------------------------------------
# Camera translation from PARE params  (mirrors mhr_head.py logic)
# ---------------------------------------------------------------------------

def convert_pare_to_full_img_cam(pare_cam, bbox_height, bbox_center, img_w, img_h,
                                  focal_length, crop_res=224):
    s, tx, ty = pare_cam[0], pare_cam[1], pare_cam[2]
    r = bbox_height / crop_res
    tz = 2 * focal_length / (r * crop_res * s)
    cx = 2 * (bbox_center[0] - img_w / 2.) / (s * bbox_height)
    cy = 2 * (bbox_center[1] - img_h / 2.) / (s * bbox_height)
    return np.array([tx + cx, ty + cy, tz], dtype=np.float32)


# ---------------------------------------------------------------------------
# Visual unit tests
# ---------------------------------------------------------------------------

def run_visual_tests(args, hparams, ds, renderer, faces, smpl_faces,
                     compute_smpl_joints, compute_smpl_pelvis, compute_smpl_verts):
    """
    Generate diagnostic PNGs that test individual operations and hypotheses
    about coordinate systems, vertex scales, camera placement, and rendering.

    Each test saves one PNG to <out_dir>/tests/.  Run with --test flag.

    Tests produced
    ──────────────
    01  Vertex scale          Is MHR vertex range ~1.7 m (metres) or ~170 (cm)?
    02  Raw verts centred     Centroid-centred verts: does a human shape appear?
    03  Pelvis subtraction    SMPL joint regressor pelvis offset; centred render.
    04  Y-flip hypothesis     Normal vs Y-flipped: which is right-side-up?
    05  Axis-flip matrix      Six axis/swap variants on one canvas.
    06  Full-image render     Mesh placed on the full (non-cropped) image.
    07  Multi-view            Front / left / right / back / top views.
    08  SMPLX canonical       SMPLX with zero global_orient rendered for reference.
    09  Joint overlay         SMPL joint regressor dots drawn on the mesh.
    10  Camera placement      Mesh aligned to GT 2D pelvis vs centred.
    11  Renderer axis frame   Coloured tetrahedron: which way does +Y point?
    """
    import cv2
    import trimesh
    import pyrender as _pyrender

    out_dir = os.path.join(args.out_dir, 'tests')
    os.makedirs(out_dir, exist_ok=True)
    sep = '=' * 60
    print(f'\n{sep}\nRunning visual unit tests  →  {out_dir}\n{sep}\n')

    # ── pick a fixed sample ────────────────────────────────────────────────
    np.random.seed(args.seed)
    ds_idx = int(np.random.randint(len(ds)))
    item = ds[ds_idx]
    img_crop = unnorm_image(item['img'])      # [224,224,3] uint8
    H, W = img_crop.shape[:2]

    gt_verts = item.get('vertices')
    if gt_verts is None:
        print('[SKIP] No vertices in dataset item. Run preconvert_mhr.sh first.')
        return

    gt_verts_np = to_numpy(gt_verts)         # [V, 3] metres

    # ── common camera params ───────────────────────────────────────────────
    img_h_full  = float(item['orig_shape'][0])
    img_w_full  = float(item['orig_shape'][1])
    bbox_scale  = float(item['scale'])
    center      = item['center']
    bbox_center = center.numpy() if isinstance(center, torch.Tensor) else np.array(center)
    fl_val      = (float(item['focal_length'][0]) if 'focal_length' in item
                   else float((img_w_full**2 + img_h_full**2)**0.5))
    bbox_height = bbox_scale * 200.0
    fl_crop     = fl_val * (224.0 / bbox_height)
    tz          = 2.0 * fl_val / bbox_height
    half        = bbox_scale * 100.0
    blank224    = lambda: (np.ones((224, 224, 3), dtype=np.float32) * 0.8 * 255).astype(np.uint8)

    def _render(verts, cam_t, bg=None):
        """Render verts on a 224×224 background using an isolated EGL context.

        Uses render_mesh_stateless so each call creates and destroys its own
        context. This avoids the EGL driver bug where creating a second context
        (e.g. for a full-res render) permanently invalidates the shared one.
        """
        bg = bg if bg is not None else blank224()
        return render_mesh_stateless(verts, faces, cam_t.copy(), bg.copy(),
                                     [fl_crop, fl_crop])

    def _save(fig, fname):
        path = os.path.join(out_dir, fname)
        fig.savefig(path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        return path

    print(f'Sample ds_idx={ds_idx}  verts={gt_verts_np.shape}  '
          f'fl_crop={fl_crop:.0f}  tz={tz:.3f}')

    # ── TEST 01: Vertex scale ──────────────────────────────────────────────
    def test_01():
        v = gt_verts_np
        vmin, vmax = v.min(0), v.max(0)
        ext = vmax - vmin
        y_range = ext[1]
        verdict = ('PASS — looks like metres' if 1.0 < y_range < 2.5
                   else 'FAIL — suspicious, check cm/m conversion')
        lines = [
            'TEST 01 — Vertex scale & bounding box',
            f'Shape:           {v.shape}',
            f'Min  (X,Y,Z):    {vmin.round(4)}',
            f'Max  (X,Y,Z):    {vmax.round(4)}',
            f'Range(X,Y,Z):    {ext.round(4)}',
            '',
            f'Hypothesis: Y-range ≈ 1.6–1.9 m for a standing person in metres.',
            f'            Y-range ≈ 160–190  would mean vertices are in cm.',
            '',
            f'Y-range = {y_range:.4f}   →   {verdict}',
        ]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.axis('off')
        ax.text(0.03, 0.97, '\n'.join(lines), transform=ax.transAxes,
                va='top', fontsize=11, family='monospace',
                color='darkgreen' if 'PASS' in verdict else 'red')
        path = _save(fig, 'test_01_vertex_scale.png')
        print(f'[01] Y-range={y_range:.4f}  {verdict}  →  {path}')

    # ── TEST 02: Raw verts centred at centroid ─────────────────────────────
    def test_02():
        centroid = gt_verts_np.mean(0)
        v = gt_verts_np - centroid
        cam_t = np.array([0., 0., tz], np.float32)
        rendered = _render(v, cam_t)

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        axes[0].imshow(img_crop);  axes[0].set_title('Input crop');  axes[0].axis('off')
        axes[1].imshow(rendered);  axes[1].set_title(f'Centroid-centred\ncam_t=[0,0,{tz:.2f}]');  axes[1].axis('off')

        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        step = max(1, len(v) // 600)
        sc = ax3.scatter(v[::step, 0], v[::step, 2], v[::step, 1],
                         s=1, c=v[::step, 1], cmap='RdYlGn')
        ax3.set_xlabel('X'); ax3.set_ylabel('Z'); ax3.set_zlabel('Y')
        ax3.set_title('3D scatter\n(colour = Y height)\nGreen = high, Red = low')
        plt.colorbar(sc, ax=ax3, fraction=0.03)

        fig.suptitle('TEST 02: Raw MHR verts centred at centroid\n'
                     'Rendered mesh should show a human silhouette.', fontsize=10)
        path = _save(fig, 'test_02_raw_verts_centred.png')
        print(f'[02] Raw centred  →  {path}')

    # ── TEST 03: Pelvis subtraction ────────────────────────────────────────
    def test_03():
        pelvis = compute_smpl_pelvis(gt_verts_np)
        v = gt_verts_np - pelvis
        cam_t = np.array([0., 0., tz], np.float32)
        rendered = _render(v, cam_t)
        # Green circle marks where pelvis should project (crop centre for tx=ty=0)
        cv2.circle(rendered, (112, 112), 7, (0, 255, 0), 2)

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        axes[0].imshow(img_crop);   axes[0].set_title('Input crop');  axes[0].axis('off')
        axes[1].imshow(rendered)
        axes[1].set_title(f'Pelvis-subtracted\ntx=ty=0, tz={tz:.2f}\n'
                          f'Green ○ = expected pelvis at (112,112)')
        axes[1].axis('off')

        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        step = max(1, len(v) // 600)
        ax3.scatter(v[::step, 0], v[::step, 2], v[::step, 1], s=1, c=v[::step, 1], cmap='RdYlGn')
        ax3.set_xlabel('X'); ax3.set_ylabel('Z'); ax3.set_zlabel('Y')
        ax3.set_title(f'3D scatter\nPelvis offset={pelvis.round(3)}')

        fig.suptitle('TEST 03: Pelvis subtraction\n'
                     'SMPL joint regressor applied to MHR verts → pelvis at origin.\n'
                     'Mesh should be centred; green circle marks its expected pixel.', fontsize=9)
        path = _save(fig, 'test_03_pelvis_subtracted.png')
        print(f'[03] Pelvis sub  pelvis={pelvis.round(3)}  →  {path}')

    # ── TEST 04: Y-flip hypothesis ─────────────────────────────────────────
    def test_04():
        pelvis = compute_smpl_pelvis(gt_verts_np)
        v = gt_verts_np - pelvis
        cam_t = np.array([0., 0., tz], np.float32)
        r_normal = _render(v,                      cam_t)
        r_yflip  = _render(v * [1, -1, 1],         cam_t)

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        axes[0].imshow(img_crop);  axes[0].set_title('Input crop');  axes[0].axis('off')
        axes[1].imshow(r_normal);  axes[1].set_title('Normal (current code)');  axes[1].axis('off')
        axes[2].imshow(r_yflip);   axes[2].set_title('Y-flipped\n(v[:,1] *= -1)');  axes[2].axis('off')

        fig.suptitle('TEST 04: Y-flip hypothesis\n'
                     'If mesh is upside-down, the Y-flipped panel should look correct.', fontsize=10)
        path = _save(fig, 'test_04_y_flip_hypothesis.png')
        print(f'[04] Y-flip hypothesis  →  {path}')

    # ── TEST 05: All axis-flip / swap variants ─────────────────────────────
    def test_05():
        pelvis = compute_smpl_pelvis(gt_verts_np)
        v = gt_verts_np - pelvis
        cam_t = np.array([0., 0., tz], np.float32)
        variants = [
            ('no flip\n(current)',    v * np.array([1,  1,  1])),
            ('flip Y\n(v[:,1]*=-1)',  v * np.array([1, -1,  1])),
            ('flip Z\n(v[:,2]*=-1)', v * np.array([1,  1, -1])),
            ('flip X\n(v[:,0]*=-1)', v * np.array([-1, 1,  1])),
            ('flip Y+Z',             v * np.array([1, -1, -1])),
            ('swap Y↔Z\n(v[:,[0,2,1]])', v[:, [0, 2, 1]]),
        ]
        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        axes = axes.flatten()
        for i, (lbl, vv) in enumerate(variants):
            axes[i].imshow(_render(vv, cam_t))
            axes[i].set_title(lbl, fontsize=9)
            axes[i].axis('off')
        fig.suptitle('TEST 05: Axis-flip / swap variants\n'
                     'Find the panel where the human stands upright.', fontsize=10)
        path = _save(fig, 'test_05_axis_flip_variants.png')
        print(f'[05] Axis-flip variants  →  {path}')

    # ── TEST 06: Full-image render ─────────────────────────────────────────
    def test_06():
        pelvis = compute_smpl_pelvis(gt_verts_np)
        v = gt_verts_np - pelvis
        fh, fw = int(img_h_full), int(img_w_full)

        tx_full = ty_full = 0.0
        if 'keypoints_orig' in item:
            kp = to_numpy(item['keypoints_orig'])
            if kp.shape[0] >= 3:
                pf = (kp[1, :2] + kp[2, :2]) / 2.0
                tx_full = (pf[0] - fw / 2) * tz / fl_val
                ty_full = (pf[1] - fh / 2) * tz / fl_val

        cam_t_full = np.array([tx_full, ty_full, tz], np.float32)
        canvas = np.full((fh, fw, 3), 200, dtype=np.uint8)

        # Use stateless renderer — avoids creating a second EGL context alongside
        # the shared renderer, which corrupts its context on some EGL drivers.
        full_render = render_mesh_stateless(
            v, faces, cam_t_full, canvas, [fl_val, fl_val],
            camera_center=[fw // 2, fh // 2],
        )

        # Also crop render for side-by-side comparison
        cam_t_crop = np.array([0., 0., tz], np.float32)
        crop_render = _render(v, cam_t_crop, img_crop.copy())

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].imshow(crop_render)
        axes[0].set_title(f'Crop render (224×224)\nfl_crop={fl_crop:.0f}  tz={tz:.3f}')
        axes[0].axis('off')
        axes[1].imshow(full_render)
        axes[1].set_title(f'Full-image render ({fw}×{fh})\nfl={fl_val:.0f}  cam_t={cam_t_full.round(3)}')
        axes[1].axis('off')

        fig.suptitle('TEST 06: Full-image render\n'
                     'Mesh should appear at the person\'s location in the full image.', fontsize=10)
        path = _save(fig, 'test_06_full_image_render.png')
        print(f'[06] Full-image render  cam_t_full={cam_t_full.round(3)}  →  {path}')

    # ── TEST 07: Multi-view ────────────────────────────────────────────────
    def test_07():
        pelvis = compute_smpl_pelvis(gt_verts_np)
        v = gt_verts_np - pelvis
        cam_t = np.array([0., 0., tz], np.float32)
        views = [
            ('Front\n(no rot)',          v),
            ('Y-flip\n(hypothesis)',      v * [1, -1, 1]),
            ('Left side\n(rot +90° Y)',   _rot_y(v,  90)),
            ('Right side\n(rot -90° Y)',  _rot_y(v, -90)),
            ('Back\n(rot 180° Y)',        _rot_y(v, 180)),
            ('Top-down\n(rot -90° X)',    _rot_x(v, -90)),
        ]
        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        for ax, (lbl, vv) in zip(axes.flatten(), views):
            ax.imshow(_render(vv, cam_t))
            ax.set_title(lbl, fontsize=9)
            ax.axis('off')
        fig.suptitle('TEST 07: Multi-view renders\n'
                     'Front should show a standing/sitting person face-on.', fontsize=10)
        path = _save(fig, 'test_07_multiview.png')
        print(f'[07] Multi-view  →  {path}')

    # ── TEST 08: SMPLX canonical mesh (ground-truth orientation) ──────────
    def test_08():
        try:
            import smplx as smplx_lib
        except ImportError:
            print('[08] SKIP — smplx not importable')
            return
        smplx_path = os.path.join(_ROOT, 'data/body_models/smplx')
        if not os.path.exists(smplx_path):
            print(f'[08] SKIP — smplx models not at {smplx_path}')
            return

        smplx_model = smplx_lib.create(
            smplx_path, model_type='smplx', gender='neutral',
            use_pca=False, num_pca_comps=45, flat_hand_mean=True).cpu()
        with torch.no_grad():
            out = smplx_model(
                betas=torch.zeros(1, 10), body_pose=torch.zeros(1, 63),
                global_orient=torch.zeros(1, 3), transl=torch.zeros(1, 3),
                jaw_pose=torch.zeros(1, 3), leye_pose=torch.zeros(1, 3),
                reye_pose=torch.zeros(1, 3),
                left_hand_pose=torch.zeros(1, 45), right_hand_pose=torch.zeros(1, 45),
                expression=torch.zeros(1, 10))
        sv = out.vertices[0].numpy()                     # [V, 3]
        sp = out.joints[0, 0].numpy()                    # pelvis (joint 0)
        sv_sub = sv - sp

        try:
            sx_faces = smplx_model.faces.astype(np.int32)
        except AttributeError:
            sx_faces = smplx_model.faces_tensor.cpu().numpy().astype(np.int32)

        cam_t = np.array([0., 0., tz], np.float32)

        def _rsx(vv):
            return render_mesh_stateless(vv, sx_faces, cam_t.copy(), blank224(),
                                         [fl_crop, fl_crop], mesh_color=(0.6, 0.8, 1.0))

        panels = [
            ('SMPLX canonical\n(no sub)',          sv),
            ('SMPLX canonical\n(pelvis-sub)',       sv_sub),
            ('SMPLX Y-flipped\n(pelvis-sub)',       sv_sub * [1, -1, 1]),
        ]
        vmin, vmax = sv.min(0), sv.max(0)
        fig, axes = plt.subplots(1, len(panels) + 1, figsize=(18, 5))
        axes[0].imshow(img_crop); axes[0].set_title('Input crop'); axes[0].axis('off')
        for ax, (lbl, vv) in zip(axes[1:], panels):
            ax.imshow(_rsx(vv)); ax.set_title(lbl, fontsize=9); ax.axis('off')
        fig.suptitle(f'TEST 08: SMPLX canonical mesh (ground-truth renderer orientation)\n'
                     f'pelvis={sp.round(3)}  Y∈[{vmin[1]:.3f},{vmax[1]:.3f}]', fontsize=9)
        path = _save(fig, 'test_08_smplx_canonical.png')
        del smplx_model
        print(f'[08] SMPLX canonical  →  {path}')

    # ── TEST 09: SMPL joint regressor overlay ─────────────────────────────
    def test_09():
        pelvis = compute_smpl_pelvis(gt_verts_np)
        v = gt_verts_np - pelvis
        joints = compute_smpl_joints(gt_verts_np) - pelvis   # [24,3] pelvis-sub
        cam_t = np.array([0., 0., tz], np.float32)
        rendered = _render(v, cam_t)

        joint_names = [
            'pelvis','L_hip','R_hip','spine1','L_knee','R_knee',
            'spine2','L_ankle','R_ankle','spine3','L_foot','R_foot',
            'neck','L_collar','R_collar','head',
            'L_shldr','R_shldr','L_elbow','R_elbow',
            'L_wrist','R_wrist','L_hand','R_hand']
        _tx, _ty, _tz = 0.0, 0.0, tz
        for ji, j3d in enumerate(joints):
            depth = j3d[2] + _tz
            if depth <= 0:
                continue
            jx = int(round(112 + fl_crop * (j3d[0] + _tx) / depth))
            jy = int(round(112 + fl_crop * (j3d[1] + _ty) / depth))
            if 0 <= jx < 224 and 0 <= jy < 224:
                cv2.circle(rendered, (jx, jy), 4, (255, 255, 0), -1)
                nm = joint_names[ji] if ji < len(joint_names) else str(ji)
                cv2.putText(rendered, nm, (jx + 4, jy), cv2.FONT_HERSHEY_PLAIN,
                            0.55, (255, 255, 0), 1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(rendered)
        axes[0].set_title('Mesh + SMPL joint regressor overlay\n(yellow = projected joints)')
        axes[0].axis('off')
        axes[1].imshow(img_crop)
        axes[1].set_title('Input crop (reference)')
        axes[1].axis('off')
        fig.suptitle('TEST 09: SMPL joint regressor overlay\n'
                     'Yellow dots should land on the correct anatomical locations.', fontsize=9)
        path = _save(fig, 'test_09_joint_overlay.png')
        print(f'[09] Joint overlay  →  {path}')

    # ── TEST 10: Camera placement (GT 2D pelvis alignment) ────────────────
    def test_10():
        pelvis = compute_smpl_pelvis(gt_verts_np)
        v = gt_verts_np - pelvis

        tx = ty = 0.0
        exp_px = exp_py = 112
        if 'keypoints_orig' in item:
            kp = to_numpy(item['keypoints_orig'])
            if kp.shape[0] >= 3:
                pf = (kp[1, :2] + kp[2, :2]) / 2.0
                px = (pf[0] - (bbox_center[0] - half)) / (2 * half) * 224.0
                py = (pf[1] - (bbox_center[1] - half)) / (2 * half) * 224.0
                tx = (px - 112.0) * tz / fl_crop
                ty = (py - 112.0) * tz / fl_crop
                exp_px = int(round(112 + fl_crop * tx / tz))
                exp_py = int(round(112 + fl_crop * ty / tz))

        cam_t_centred = np.array([0.,  0.,  tz], np.float32)
        cam_t_aligned = np.array([tx, ty, tz], np.float32)

        r_centred = _render(v, cam_t_centred, img_crop.copy())
        r_aligned = _render(v, cam_t_aligned, img_crop.copy())
        cv2.drawMarker(r_centred, (112, 112),   (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
        cv2.drawMarker(r_aligned, (exp_px, exp_py), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_crop);   axes[0].set_title('Input crop');  axes[0].axis('off')
        axes[1].imshow(r_centred);
        axes[1].set_title(f'Pelvis at crop centre\n(tx=0, ty=0, tz={tz:.2f})\nGreen ✚ = expected pixel')
        axes[1].axis('off')
        axes[2].imshow(r_aligned)
        axes[2].set_title(f'Pelvis at GT 2D pos\n(tx={tx:.3f}, ty={ty:.3f}, tz={tz:.2f})\nGreen ✚ = expected pixel')
        axes[2].axis('off')
        fig.suptitle('TEST 10: Camera placement\n'
                     'Green cross = where the pelvis should project in crop pixels.', fontsize=10)
        path = _save(fig, 'test_10_camera_placement.png')
        print(f'[10] Camera placement  tx={tx:.3f} ty={ty:.3f} tz={tz:.3f}  →  {path}')

    # ── TEST 11: Renderer axis frame ───────────────────────────────────────
    def test_11():
        """Render a coloured axis indicator to see which way +X/+Y/+Z point
        after the renderer's internal 180°X rotation.
        Red=+X, Green=+Y, Blue=+Z.  +Y should point UP in the image."""
        scale = 0.4
        verts = np.array([
            [0,      0,      0     ],   # 0 origin (white)
            [scale,  0,      0     ],   # 1 +X tip  (red)
            [0,      scale,  0     ],   # 2 +Y tip  (green)
            [0,      0,      scale ],   # 3 +Z tip  (blue)
        ], dtype=np.float32)
        faces = np.array([[0,1,2],[0,2,3],[0,1,3],[1,2,3]], dtype=np.int32)
        vcolors = np.array([
            [1., 1., 1., 1.],   # origin  white
            [1., 0., 0., 1.],   # +X      red
            [0., 1., 0., 1.],   # +Y      green
            [0., 0., 1., 1.],   # +Z      blue
        ], dtype=np.float32)

        mesh = trimesh.Trimesh(verts, faces, vertex_colors=vcolors, process=False)
        rot180x = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot180x)
        pymesh = _pyrender.Mesh.from_trimesh(mesh, smooth=False)

        cam_t = np.array([0., 0., tz], np.float32)
        cam_t_render = cam_t.copy(); cam_t_render[0] *= -1

        scene = _pyrender.Scene(ambient_light=(0.8, 0.8, 0.8))
        scene.add(pymesh)
        cam_pose = np.eye(4); cam_pose[:3, 3] = cam_t_render
        scene.add(_pyrender.IntrinsicsCamera(fx=fl_crop, fy=fl_crop, cx=112, cy=112),
                  pose=cam_pose)
        lp = np.eye(4); lp[:3, 3] = [0, 1, 1]
        scene.add(_pyrender.DirectionalLight([1,1,1], 1.5), pose=lp)

        r11 = _pyrender.OffscreenRenderer(224, 224)
        try:
            color, _ = r11.render(scene, flags=_pyrender.RenderFlags.RGBA)
        finally:
            r11.delete()
        img_render = (color[:, :, :3].astype(np.float32) / 255.0)

        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        axes[0].imshow(img_render)
        axes[0].set_title('Axis tetrahedron after 180°X rotation\n'
                          'Red=+X  Green=+Y  Blue=+Z  White=origin\n'
                          'Hypothesis: Green (+Y) should point UP in this image.')
        axes[0].axis('off')
        axes[1].imshow(img_crop)
        axes[1].set_title('Input crop (reference)')
        axes[1].axis('off')
        fig.suptitle('TEST 11: Renderer axis orientation\n'
                     'If +Y does NOT point up, the mesh will appear upside-down.', fontsize=10)
        path = _save(fig, 'test_11_renderer_axes.png')
        print(f'[11] Renderer axes  →  {path}')

    # ── TEST 12: SMPL vs MHR rendering parity ─────────────────────────────
    def test_12():
        """Render SMPL (barycentric-reconstructed) and MHR meshes on a black
        background, then quantify rendering discrepancy via mask IoU and a
        per-pixel difference map."""
        pelvis = compute_smpl_pelvis(gt_verts_np)
        v_mhr  = gt_verts_np - pelvis                      # [V_mhr, 3]
        v_smpl = compute_smpl_verts(gt_verts_np) - pelvis  # [6890, 3]

        cam_t = np.array([0., 0., tz], np.float32)
        black = np.zeros((224, 224, 3), dtype=np.uint8)

        r_mhr  = render_mesh_stateless(v_mhr,  faces,      cam_t, black.copy(),
                                       [fl_crop, fl_crop], mesh_color=(0.85, 0.45, 0.45))
        r_smpl = render_mesh_stateless(v_smpl, smpl_faces, cam_t, black.copy(),
                                       [fl_crop, fl_crop], mesh_color=(0.45, 0.60, 0.90))

        mask_mhr  = r_mhr.sum(2)  > 10
        mask_smpl = r_smpl.sum(2) > 10
        union     = mask_mhr | mask_smpl
        intersect = mask_mhr & mask_smpl
        iou       = intersect.sum() / max(union.sum(), 1)

        overlay = np.zeros((224, 224, 3), dtype=np.uint8)
        overlay[mask_mhr,  0] = 210   # red channel  → MHR (and overlap → purple)
        overlay[mask_smpl, 2] = 210   # blue channel → SMPL (and overlap → purple)

        diff = np.abs(r_mhr.astype(np.float32) - r_smpl.astype(np.float32)).mean(2)
        mean_diff = float(diff[union].mean()) if union.any() else 0.0

        fig, axes = plt.subplots(1, 5, figsize=(22, 5))
        axes[0].imshow(img_crop);  axes[0].set_title('Input crop');         axes[0].axis('off')
        axes[1].imshow(r_mhr);     axes[1].set_title('MHR mesh (red)');     axes[1].axis('off')
        axes[2].imshow(r_smpl);    axes[2].set_title('SMPL mesh (blue)');   axes[2].axis('off')
        axes[3].imshow(overlay)
        axes[3].set_title(f'Overlay  IoU={iou:.4f}\nRed=MHR  Blue=SMPL  Purple=both')
        axes[3].axis('off')
        im = axes[4].imshow(diff, cmap='hot', vmin=0)
        plt.colorbar(im, ax=axes[4], fraction=0.04)
        axes[4].set_title(f'|MHR − SMPL| pixel diff\nmean(union)={mean_diff:.1f}')
        axes[4].axis('off')

        fig.suptitle(f'TEST 12: SMPL vs MHR rendering parity\n'
                     f'IoU={iou:.4f}  union_px={union.sum()}  '
                     f'MHR_px={mask_mhr.sum()}  SMPL_px={mask_smpl.sum()}', fontsize=10)
        path = _save(fig, 'test_12_smpl_mhr_parity.png')
        print(f'[12] SMPL vs MHR parity  IoU={iou:.4f}  mean_diff={mean_diff:.1f}  →  {path}')

    # ── run all ───────────────────────────────────────────────────────────
    test_01()
    test_02()
    test_03()
    test_04()
    test_05()
    test_06()
    test_07()
    test_08()
    test_09()
    test_10()
    test_11()
    test_12()

    print(f'\n{sep}\nAll tests done  →  {out_dir}\n{sep}\n')


# ---------------------------------------------------------------------------
# Main ????
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- load config --------------------------------------------------------
    from train.utils.train_utils import update_hparams
    hparams = update_hparams(args.cfg)

    # ---- dataset ------------------------------------------------------------
    from dataset_wrapper import DatasetHMR
    is_train = (args.split == 'train')
    ds = DatasetHMR(hparams.DATASET, dataset=args.dataset, is_train=is_train)
    print(f'Dataset {args.dataset} ({args.split}): {len(ds)} samples')

    indices = np.random.choice(len(ds), size=min(args.n, len(ds)), replace=False)

    # ---- renderer -----------------------------------------------------------
    from load_mhr_portable import load_portable
    _portable = load_portable(os.path.join(_ROOT, 'mhr_portable_dump', 'mhr_dump_lod1.pt'))
    _faces = _portable['interesting']['character.mesh.faces']
    if isinstance(_faces, torch.Tensor):
        faces = _faces.cpu().numpy()
    else:
        faces = np.array(_faces)

    # ---- SMPL pelvis regressor (mirrors mhr_head.py) -------------------------
    # Used to compute the SMPL pelvis position from GT MHR vertices so we can
    # centre the mesh on the pelvis before rendering (MHR origin is at feet).
    import types, pickle, scipy.sparse
    _mapping = np.load(os.path.join(_ROOT, 'assets', 'mhr2smpl_mapping.npz'))
    _smpl_tri_vids = np.array(_faces)[_mapping['triangle_ids'].astype(np.int64)]  # [6890, 3]
    _smpl_baryc    = _mapping['baryc_coords'].astype(np.float32)                  # [6890, 3]
    def _stub_chumpy():
        if 'chumpy' not in sys.modules:
            _c = types.ModuleType('chumpy')
            _c.Ch = type('Ch', (), {})
            _c.array = lambda *a, **k: None
            sys.modules['chumpy'] = _c
            sys.modules['chumpy.ch'] = types.ModuleType('chumpy.ch')
            sys.modules['chumpy.ch'].Ch = _c.Ch
    _saved_mods = {k: sys.modules.get(k) for k in ('chumpy', 'chumpy.ch')}
    _stub_chumpy()
    try:
        with open(os.path.join(_ROOT, 'data/body_models/SMPL_python_v.1.1.0/smpl/models/SMPL_NEUTRAL.pkl'), 'rb') as _f:
            _smpl_data = pickle.load(_f, encoding='latin1')
    finally:
        for _k, _v in _saved_mods.items():
            if _v is None: sys.modules.pop(_k, None)
            else: sys.modules[_k] = _v
    _J_reg = _smpl_data['J_regressor']
    if scipy.sparse.issparse(_J_reg):
        _J_reg = np.array(_J_reg.todense()).astype(np.float32)
    else:
        _J_reg = np.array(_J_reg).astype(np.float32)  # [24, 6890]
    smpl_faces = _smpl_data['f'].astype(np.int32)     # [13776, 3]

    def compute_smpl_verts(verts_m):
        """Reconstruct 6890 SMPL vertex positions from MHR vertices [V,3] via barycentric mapping."""
        v0 = verts_m[_smpl_tri_vids[:, 0]]
        v1 = verts_m[_smpl_tri_vids[:, 1]]
        v2 = verts_m[_smpl_tri_vids[:, 2]]
        return (_smpl_baryc[:, 0:1]*v0 + _smpl_baryc[:, 1:2]*v1
                + _smpl_baryc[:, 2:3]*v2)  # [6890, 3]

    def compute_smpl_joints(verts_m):
        """Compute all 24 SMPL joints from MHR vertices [V,3] in metres."""
        return _J_reg @ compute_smpl_verts(verts_m)  # [24, 3]

    def compute_smpl_pelvis(verts_m):
        """Compute SMPL pelvis position (joint 0) from MHR vertices [V,3] in metres."""
        return compute_smpl_joints(verts_m)[0]

    from train.utils.renderer import Renderer
    renderer = Renderer(
        focal_length=hparams.DATASET.FOCAL_LENGTH,
        img_res=hparams.DATASET.IMG_RES,
        faces=faces,
        mesh_color='pinkish',
    )

    # ---- test mode ----------------------------------------------------------
    if args.test:
        run_visual_tests(args, hparams, ds, renderer, faces, smpl_faces,
                         compute_smpl_joints, compute_smpl_pelvis, compute_smpl_verts)
        return

    # ---- optional model -----------------------------------------------------
    model = None
    if args.ckpt:
        from mhr_trainer import MHRTrainer
        trainer_module = MHRTrainer.load_from_checkpoint(args.ckpt, hparams=hparams)
        trainer_module.eval().cuda()
        model = trainer_module
        print(f'Loaded checkpoint: {args.ckpt}')
    else:
        print('No checkpoint — GT panels only')

    # ---- per-sample loop ----------------------------------------------------
    for sample_idx, ds_idx in enumerate(indices):
        item = ds[int(ds_idx)]

        # Wrap single item into a batch-1 dict
        batch = {k: v.unsqueeze(0).cuda() if isinstance(v, torch.Tensor) else v
                 for k, v in item.items()}
        # scalar fields that need a batch dim
        for k in ('scale', 'center', 'orig_shape', 'focal_length'):
            if k in batch and isinstance(batch[k], torch.Tensor) and batch[k].dim() == 1:
                batch[k] = batch[k].unsqueeze(0)

        img_crop = unnorm_image(item['img'])   # HxWx3 uint8, the 224×224 crop
        H, W = img_crop.shape[:2]

        img_h_full = float(item['orig_shape'][0])
        img_w_full = float(item['orig_shape'][1])
        bbox_scale = float(item['scale'])
        center = item['center']
        bbox_center = center.numpy() if isinstance(center, torch.Tensor) else np.array(center)

        # Use per-sample focal length if available, else compute from image size
        if 'focal_length' in item:
            fl_val = float(item['focal_length'][0])
        else:
            fl_val = float((img_w_full**2 + img_h_full**2)**0.5)

        # Camera intrinsics for the CROP (renderer uses crop-space coordinates)
        # The renderer uses focal_length and image centre derived from img_res,
        # so we pass the full-image focal length and let renderer handle it.
        crop_focal = fl_val  # renderer will use this if we pass it explicitly

        # ---- GT mesh (full-image render) ----------------------------------------
        # Render onto the full-resolution image rather than the 224 crop.
        # The full-image path is reliable because it uses the raw camera intrinsics
        # (fl_val, principal point at image centre) with no crop rescaling, which is
        # what caused the upside-down / misaligned rendering on the crop path.
        gt_verts = item.get('vertices')   # [V, 3] metres, MHR body space
        gt_render = None   # uint8 image for the figure panel (set below)

        if gt_verts is not None:
            gt_verts_np = to_numpy(gt_verts)  # [V, 3] metres

            # -- Pelvis centering -------------------------------------------------
            # MHR stores vertices with the body origin at the feet.
            # Subtract SMPL pelvis (joint 0, via J_regressor → barycentric chain)
            # so the pelvis lands at world origin (0, 0, 0).
            # Every camera-translation formula below assumes root == origin.
            pelvis_3d = compute_smpl_pelvis(gt_verts_np)   # [3] metres, camera space
            gt_verts_np = gt_verts_np - pelvis_3d[np.newaxis]  # [V,3] pelvis-centred
            print(f'  SMPL pelvis offset: {pelvis_3d.round(3)}')

            # -- Full-image camera intrinsics ------------------------------------
            fh, fw = int(img_h_full), int(img_w_full)
            bbox_height = bbox_scale * 200.0   # bounding-box height in full-image px

            # Depth — weak-perspective formula:  tz = 2 * fl / bbox_height
            # Derivation: at depth tz a person of metric height H projects to
            #   H * fl / tz  pixels.  Setting H≈2 m and that equal to bbox_height
            # gives  tz = 2 * fl / bbox_height.
            # This places the body so it fills the annotated bounding box.
            tz = 2.0 * fl_val / bbox_height

            # -- Horizontal / vertical translation from 2D pelvis ---------------
            # keypoints_orig: [24, 3] full-image pixel coords, SMPL 24-joint order.
            #   j=0  Pelvis_SMPL   j=1  L_Hip_SMPL   j=2  R_Hip_SMPL   …
            # We average L_Hip and R_Hip to approximate the 2D pelvis pixel.
            #
            # Pinhole projection for the pelvis sitting at world origin (0,0,0):
            #   x_pix = cx + fx * cam_tx / tz   →   cam_tx = (x_pix − cx) * tz / fx
            #   y_pix = cy + fy * cam_ty / tz   →   cam_ty = (y_pix − cy) * tz / fy
            #
            # render_mesh_stateless negates cam_t[0] internally (cam_t_x *= −1) to
            # match pyrender's convention where the camera is oriented along −Z; we
            # keep the sign positive here so the formula above is self-consistent.
            tx_full, ty_full = 0.0, 0.0
            if 'keypoints_orig' in item:
                kp_orig = to_numpy(item['keypoints_orig'])   # [24, 3]: x_pix, y_pix, conf
                if kp_orig.shape[0] >= 3:
                    # Full-image pixel location of the estimated pelvis centre
                    pelvis_2d = (kp_orig[1, :2] + kp_orig[2, :2]) / 2.0  # [x, y]
                    # Principal point at the geometric image centre
                    tx_full = (pelvis_2d[0] - fw / 2.0) * tz / fl_val
                    ty_full = (pelvis_2d[1] - fh / 2.0) * tz / fl_val
                    print(f'  pelvis_2d=({pelvis_2d[0]:.0f},{pelvis_2d[1]:.0f}) px  '
                          f'tx={tx_full:.3f}  ty={ty_full:.3f}')

            # cam_t = [tx, ty, tz] in metres:
            #   tz — depth of the pelvis from the camera
            #   tx — rightward offset (positive → person to the right of image centre)
            #   ty — downward offset  (positive → person below image centre)
            cam_t_full = np.array([tx_full, ty_full, tz], dtype=np.float32)
            print(f'  cam_t_full={cam_t_full.round(3)}  fl={fl_val:.0f}  img={fw}x{fh}')

            # -- Load original full-resolution image ----------------------------
            # item['img'] is the 224×224 normalised crop; we need the raw full-
            # resolution frame so the mesh projects to the correct pixel location.
            import cv2 as _cv2
            try:
                _bgr = _cv2.imread(item['imgname'])
                if _bgr is not None:
                    full_img = _cv2.cvtColor(_bgr, _cv2.COLOR_BGR2RGB)
                else:
                    full_img = np.full((fh, fw, 3), 128, dtype=np.uint8)
            except Exception as _e:
                print(f'  Could not load full image ({_e}); using grey canvas')
                full_img = np.full((fh, fw, 3), 128, dtype=np.uint8)

            # -- Render mesh onto the full image --------------------------------
            # render_mesh_stateless pipeline:
            #   1. Creates a fresh pyrender EGL context for this call only, then
            #      destroys it — no shared-context state leaks between samples.
            #   2. Applies a 180° rotation around X to every mesh vertex:
            #          (x, y, z)  →  (x, −y, −z)
            #      This converts camera-space Y-down (SMPL / OpenCV convention)
            #      to Y-up (pyrender / OpenGL convention).
            #   3. Negates cam_t[0] before assembling the camera pose matrix:
            #          cam_pose = diag(1,1,1,1) with translation (−tx, ty, tz)
            #      so the camera sits at the correct world position looking along −Z.
            #   4. Uses IntrinsicsCamera(fx=fl_val, fy=fl_val, cx=fw/2, cy=fh/2).
            gt_render_full = render_mesh_stateless(
                gt_verts_np,                       # [V,3] metres, pelvis-centred, Y-down
                faces,                             # [F,3] MHR triangle indices
                cam_t_full,                        # [3] metres: [tx, ty, tz]
                full_img,                          # background: full-resolution RGB uint8
                [fl_val, fl_val],                  # [fx, fy]: full-image focal length (px)
                camera_center=[fw // 2, fh // 2],  # principal point at image centre
            )

            # -- Overlay SMPL joint dots for alignment verification -------------
            # Project every joint with the same pinhole formula pyrender uses.
            # After the 180°X mesh flip and tx-negation in render_mesh_stateless
            # the effective projection for a vertex at world (x, y, z) is:
            #   u = cx + fx * (x + tx) / (z + tz)
            #   v = cy + fy * (y + ty) / (z + tz)
            # Derivation:
            #   rotated vertex:  (x, −y, −z)
            #   camera at:       (−tx, ty, tz)
            #   vertex in cam:   (x+tx,  −y−ty,  −(z+tz))
            #   pyrender (Y-up, looks along −Z):
            #     u = cx + fx * (x+tx) / (z+tz)         ← depth = z+tz
            #     v = cy − fy * (−y−ty) / (z+tz)
            #       = cy + fy * (y+ty) / (z+tz)         ← verified: pelvis → pelvis_2d
            smpl_joints = compute_smpl_joints(gt_verts_np)   # [24,3] pelvis-centred
            for j3d in smpl_joints:
                depth = j3d[2] + tz   # ≈ tz for joints near z=0
                if depth <= 0:
                    continue
                jx_pix = int(round(fw / 2 + fl_val * (j3d[0] + tx_full) / depth))
                jy_pix = int(round(fh / 2 + fl_val * (j3d[1] + ty_full) / depth))
                if 0 <= jx_pix < fw and 0 <= jy_pix < fh:
                    _cv2.circle(gt_render_full, (jx_pix, jy_pix), 4, (255, 255, 0), -1)

            # Save full-resolution overlay as a separate diagnostic PNG
            _fi_path = os.path.join(args.out_dir,
                                    f'sample_{sample_idx:03d}_ds{ds_idx}_fullimg.png')
            _cv2.imwrite(_fi_path, _cv2.cvtColor(gt_render_full, _cv2.COLOR_RGB2BGR))
            print(f'  Full-image render → {_fi_path}')

            # Scale down for the figure panel: target 224 px tall, preserve aspect.
            # (Full images are typically wider than tall so this keeps panels uniform.)
            _disp_h = 224
            _disp_w = max(1, int(fw * _disp_h / fh))
            gt_render = _cv2.resize(gt_render_full, (_disp_w, _disp_h),
                                    interpolation=_cv2.INTER_LINEAR)

        # ---- prediction -----------------------------------------------------
        pred_render = None
        pred_kps2d_smpl = None
        pred_joints3d_np = None

        if model is not None:
            with torch.no_grad():
                out, *_ = model(
                    batch['img'],
                    bbox_scale=batch['scale'],
                    bbox_center=batch['center'],
                    img_w=batch['orig_shape'][:, 1],
                    img_h=batch['orig_shape'][:, 0],
                    fl=batch.get('focal_length'),
                )

            pred_verts_np = out['vertices'][0].cpu().numpy()      # [V, 3] metres
            pred_cam_t_np = out['pred_cam_t'][0].cpu().numpy()    # [3]
            pred_joints3d_np = out['joints3d_smpl'][0].cpu().numpy()  # [24, 3]

            # joints2d_smpl: full-image pixel coords [24, 2]
            if 'joints2d_smpl' in out:
                pred_kps2d_smpl = out['joints2d_smpl'][0].cpu().numpy()   # [24, 2]

            try:
                pred_render = render_mesh_on_image(
                    renderer, pred_verts_np, pred_cam_t_np, img_crop, [crop_focal, crop_focal]
                )
            except Exception as e:
                print(f'  Pred render failed: {e}')
                pred_render = img_crop.copy()

        # ---- GT 2D keypoints ------------------------------------------------
        # keypoints_orig: [J, 3] (x_full, y_full, conf)  — full image pixels
        # keypoints:      [J, 3] (x_norm, y_norm, conf)  — normalised [-1,1]
        gt_kps_orig = item.get('keypoints_orig')    # [J, 3] or None
        gt_kps_norm = item.get('keypoints')         # [J, 3] or None

        # Project keypoints_orig from full image onto the 224 crop for display.
        # The crop covers bbox_center ± bbox_scale*100 pixels in the full image.
        half = bbox_scale * 100.0   # half-width/height of crop in full-image pixels
        def full_to_crop(xy_full):
            """Map full-image [J,2] pixel coords to 224-crop pixel coords."""
            x = (xy_full[:, 0] - (bbox_center[0] - half)) / (2 * half) * W
            y = (xy_full[:, 1] - (bbox_center[1] - half)) / (2 * half) * H
            return np.stack([x, y], axis=1)

        # Unnormalise keypoints ([-1,1] → crop pixels)
        def unnorm_kps(kps_norm):
            xy = (kps_norm[:, :2] + 1) / 2.0 * W
            return xy

        # ---- build figure ---------------------------------------------------
        n_cols = 6 + (1 if model is not None else 0)  # extra pred column
        fig = plt.figure(figsize=(n_cols * 3, 7))
        fig.suptitle(
            f'Sample {ds_idx} | dataset={args.dataset} | '
            f'scale={bbox_scale:.2f} | fl={fl_val:.0f} | '
            f'img={int(img_w_full)}x{int(img_h_full)}',
            fontsize=9
        )

        col = 0
        axes = []

        def add_image_panel(title, img_uint8):
            nonlocal col
            ax = fig.add_subplot(1, n_cols, col + 1)
            ax.imshow(img_uint8)
            ax.set_title(title, fontsize=8)
            ax.axis('off')
            col += 1
            return ax

        # 1. Input crop
        add_image_panel('Input crop', img_crop)

        # 2. Pred mesh (if available)
        if pred_render is not None:
            add_image_panel('Pred mesh', pred_render)
        elif model is not None:
            add_image_panel('Pred mesh\n(failed)', img_crop)

        # 3. GT mesh on full image (scaled down)
        if gt_render is not None:
            add_image_panel('GT mesh\n(full img)', gt_render)
        else:
            add_image_panel('GT mesh\n(no verts)', img_crop)

        # 4. Pred 2D keypoints on crop
        if pred_kps2d_smpl is not None:
            kp_img = draw_keypoints_on_image(img_crop, full_to_crop(pred_kps2d_smpl), (255, 80, 80), radius=4)
            add_image_panel('Pred kps2d\n(SMPL, full-img px)', kp_img)
        elif model is not None:
            add_image_panel('Pred kps2d\n(none)', img_crop)

        # 5. GT keypoints_orig (full-image pixels → crop)
        if gt_kps_orig is not None:
            kp_np = to_numpy(gt_kps_orig)
            crop_xy = full_to_crop(kp_np[:, :2])
            conf = kp_np[:, 2]
            kp_img = draw_keypoints_on_image(img_crop, crop_xy, (80, 80, 255), radius=4, conf=conf)
            add_image_panel('GT kps_orig\n(full-img px → crop)', kp_img)
        else:
            add_image_panel('GT kps_orig\n(missing)', img_crop)

        # 6. GT keypoints_norm ([-1,1] → crop)
        if gt_kps_norm is not None:
            kp_np = to_numpy(gt_kps_norm)
            crop_xy = unnorm_kps(kp_np)
            conf = kp_np[:, 2] if kp_np.shape[1] > 2 else None
            kp_img = draw_keypoints_on_image(img_crop, crop_xy, (80, 200, 80), radius=4, conf=conf)
            add_image_panel('GT keypoints\n(norm [-1,1] → crop)', kp_img)
        else:
            add_image_panel('GT keypoints\n(missing)', img_crop)

        # 7. 3D joints scatter (pred red, GT blue).
        # Prefer joints3d_mhr (MHR body-local joints, same space as pred).
        # Fall back to joints3d (SMPL camera-space joints from original dataset).
        gt_joints3d_item = item.get('joints3d_mhr', item.get('joints3d'))
        gt_joints3d_np = to_numpy(gt_joints3d_item) if gt_joints3d_item is not None else None

        ax3d = fig.add_subplot(1, n_cols, col + 1, projection='3d')
        plot_joints_3d(ax3d, pred_joints3d_np, gt_joints3d_np, title='3D joints\n(pelvis-centred)')
        col += 1

        plt.tight_layout()
        out_path = os.path.join(args.out_dir, f'sample_{sample_idx:03d}_ds{ds_idx}.png')
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {out_path}')

        # ---- text summary ---------------------------------------------------
        print(f'  bbox_center={bbox_center}  bbox_scale={bbox_scale:.3f}')
        if gt_joints3d_np is not None:
            print(f'  GT joints3d range: {gt_joints3d_np.min():.3f} .. {gt_joints3d_np.max():.3f} m')
        if pred_joints3d_np is not None:
            print(f'  Pred joints3d range: {pred_joints3d_np.min():.3f} .. {pred_joints3d_np.max():.3f} m')
        if gt_verts is not None:
            vn = to_numpy(gt_verts)
            print(f'  GT verts range: {vn.min():.3f} .. {vn.max():.3f} m')
        if gt_kps_orig is not None:
            ko = to_numpy(gt_kps_orig)
            print(f'  keypoints_orig xy range: x=[{ko[:,0].min():.0f},{ko[:,0].max():.0f}]  '
                  f'y=[{ko[:,1].min():.0f},{ko[:,1].max():.0f}]  '
                  f'conf_min={ko[:,2].min():.2f}')
        if gt_kps_norm is not None:
            kn = to_numpy(gt_kps_norm)
            print(f'  keypoints (norm) xy range: [{kn[:,0].min():.2f},{kn[:,0].max():.2f}]')
        if pred_kps2d_smpl is not None:
            print(f'  Pred kps2d_smpl range: x=[{pred_kps2d_smpl[:,0].min():.0f},{pred_kps2d_smpl[:,0].max():.0f}]  '
                  f'y=[{pred_kps2d_smpl[:,1].min():.0f},{pred_kps2d_smpl[:,1].max():.0f}]')

    print(f'\nDone. Output in {os.path.abspath(args.out_dir)}')


if __name__ == '__main__':
    main()
