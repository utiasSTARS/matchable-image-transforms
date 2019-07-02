import torch.utils.data
from torchvision import transforms

import os.path
import glob
import numpy as np

from collections import namedtuple
from PIL import Image

import viso2

from .. import transforms as custom_transforms

CameraIntrinsics = namedtuple('CameraIntrinsics', 'fu, fv, cu, cv')
intrinsics_full = CameraIntrinsics(725.0, 725.0, 620.5, 187.0)
# 1242x375 --> 636x192
intrinsics_636x192 = CameraIntrinsics(371.2, 371.2, 317.5, 95.5)
# 1242x375 --> 256x192
intrinsics_centrecrop_256x192 = CameraIntrinsics(371.2, 371.2, 127.5, 95.5)


class Dataset:
    """Load and parse data from Virtual KITTI dataset."""

    def __init__(self, base_path, sequence, condition, **kwargs):
        self.base_path = base_path
        self.sequence = sequence
        self.condition = condition
        self.frames = kwargs.get('frames', None)
        self.rgb_dir = kwargs.get('rgb_dir', 'vkitti_1.3.1_rgb')
        self.depth_dir = kwargs.get('depth_dir', 'vkitti_1.3.1_depthgt')
        self.gt_dir = kwargs.get('gt_dir', 'vkitti_1.3.1_extrinsicsgt')

        self._load_timestamps_and_poses()

        self.num_frames = len(self.timestamps)

        self.rgb_files = sorted(glob.glob(
            os.path.join(self.base_path, self.rgb_dir,
                         self.sequence, self.condition, '*.png')))
        self.depth_files = sorted(glob.glob(
            os.path.join(self.base_path, self.depth_dir,
                         self.sequence, self.condition, '*.png')))

        if self.frames is not None:
            self.rgb_files = [self.rgb_files[i] for i in self.frames]
            self.depth_files = [self.depth_files[i] for i in self.frames]

    def __len__(self):
        return self.num_frames

    def get_rgb(self, idx, size=None):
        """Load RGB image from file."""
        return self._load_image(self.rgb_files[idx], size=size,
                                mode='RGB', dtype=np.uint8)

    def get_gray(self, idx, size=None):
        """Load grayscale image from file."""
        return self._load_image(self.rgb_files[idx], size=size,
                                mode='L', dtype=np.uint8)

    def get_depth(self, idx, size=None):
        """Load depth image from file."""
        return self._load_image(self.depth_files[idx], size=size,
                                mode='F', dtype=np.float, factor=100.)

    def _load_image(self, impath, size=None, mode='RGB',
                    dtype=np.float, factor=1):
        """Load image from file."""
        im = Image.open(impath).convert(mode)
        if size:
            im = im.resize(size, resample=Image.BILINEAR)
        return (np.array(im) / factor).astype(dtype)

    def _load_timestamps_and_poses(self):
        """Load ground truth poses (T_w_cam) and timestamps from file."""
        pose_file = os.path.join(self.base_path, self.gt_dir,
                                 '{}_{}.txt'.format(
                                     self.sequence, self.condition))

        self.timestamps = []
        self.poses = []

        # Read and parse the poses
        with open(pose_file, 'r') as f:
            for line in f.readlines():
                line = line.split()
                if line[0] == 'frame':  # this is the header
                    continue
                self.timestamps.append(float(line[0]))

                # from world to camera
                Tmatrix = np.array([float(x)
                                    for x in line[1:17]]).reshape((4, 4))
                # from camera to world
                self.poses.append(np.linalg.inv(Tmatrix))

        if self.frames is not None:
            self.timestamps = [self.timestamps[i] for i in self.frames]
            self.poses = [self.poses[i] for i in self.frames]


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, opts, sequence, cond1, cond2, random_crop, **kwargs):
        self.opts = opts
        self.random_crop = random_crop

        self.dataset1 = Dataset(self.opts.data_dir, sequence, cond1, **kwargs)
        self.dataset2 = Dataset(self.opts.data_dir, sequence, cond2, **kwargs)

        self.vo_params = viso2.Mono_parameters()  # Use ransac
        self.vo_params.ransac_iters = 400
        self.vo = viso2.VisualOdometryMono(self.vo_params)

    def __len__(self):
        return len(self.dataset1)

    def __getitem__(self, idx1):
        interval = np.random.randint(
            low=-self.opts.max_interval, high=self.opts.max_interval+1)

        idx2 = idx1 + interval
        if idx2 >= len(self.dataset2):
            idx2 = len(self.dataset2) - 1
        elif idx2 < 0:
            idx2 = 0

        # Get images
        rgb1 = Image.fromarray(self.dataset1.get_rgb(idx1))
        rgb2 = Image.fromarray(self.dataset2.get_rgb(idx2))

        resize_scale = min(self.opts.image_load_size) / min(rgb1.size)
        resize_offset = 0.5 * (max(rgb1.size) * resize_scale -
                               max(self.opts.image_load_size))

        resize = transforms.Compose([
            transforms.Resize(min(self.opts.image_load_size)),
            transforms.CenterCrop(self.opts.image_load_size),
            custom_transforms.StatefulRandomCrop(
                self.opts.image_final_size) if self.random_crop else transforms.Resize(self.opts.image_final_size)
        ])
        make_grayscale = transforms.Grayscale()
        make_normalized_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.opts.image_mean, self.opts.image_std)
        ])
        make_normalized_gray_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (self.opts.image_mean[0],), (self.opts.image_std[0],))
        ])
        # Clamp to at the minimum to avoid computing log(0) = -inf
        make_log_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda tensor: tensor.clamp(1e-3, 1.)),
            transforms.Lambda(lambda tensor: tensor.log())
        ])

        rgb1 = resize(rgb1)
        rgb2 = resize(rgb2)
        gray1 = make_grayscale(rgb1)
        gray2 = make_grayscale(rgb2)

        if self.opts.compute_matches:
            matches11 = self._get_match_count(gray1, gray1)
            matches12 = self._get_match_count(gray1, gray2)
            matches22 = self._get_match_count(gray2, gray2)

        logrgb1 = make_log_tensor(rgb1)
        logrgb2 = make_log_tensor(rgb2)
        rgb1 = make_normalized_tensor(rgb1)
        rgb2 = make_normalized_tensor(rgb2)
        gray1 = make_normalized_gray_tensor(gray1)
        gray2 = make_normalized_gray_tensor(gray2)

        data = {'rgb1': rgb1, 'rgb2': rgb2,
                'gray1': gray1, 'gray2': gray2,
                'logrgb1': logrgb1, 'logrgb2': logrgb2}
        if self.opts.compute_matches:
            data.update({'matches11': matches11,
                         'matches12': matches12,
                         'matches22': matches22})

        return data

    def _get_match_count(self, im1, im2):
        self.vo.process_frame(np.array(im1), np.array(im2))
        return np.float32(self.vo.getNumberOfInliers())
