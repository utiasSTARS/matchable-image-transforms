import torch.utils.data
from torchvision import transforms

import os.path
import glob
import numpy as np

from collections import namedtuple
from PIL import Image

import viso2

from .. import transforms as custom_transforms

CameraIntrinsics = namedtuple('CameraIntrinsics', 'fu, fv, cu, cv, b')
intrinsics_full = CameraIntrinsics(
    387.777, 387.777, 257.446, 197.718, 0.239965)
# 512x384 --> 256x192
intrinsics_256x192 = CameraIntrinsics(
    193.8885, 193.8885, 128.723, 98.859, 0.239965)

Timestamp = namedtuple('Timestamp', 'vertex_id_current, timestamp')
TemporalTransform = namedtuple('TemporalTransform',
                               'run_id_curr, vertex_id_curr, ' +
                               'run_id_prev, vertex_id_prev, ' +
                               'transformation')
SpatialTransform = namedtuple('SpatialTransform',
                              'run_id_man, vertex_id_man, ' +
                              'run_id_auto, vertex_id_auto, ' +
                              'transformation')


class Dataset:
    """Load and parse data from processed robochunk dataset."""

    def __init__(self, base_path, sequence, **kwargs):
        self.base_path = base_path
        self.sequence = sequence
        self.frames = kwargs.get('frames', None)
        self.camera = kwargs.get('camera', 'left')

        self._load_timestamps_and_transforms()

        self.num_frames = len(self.transforms_spatial)

        self.rgb_left_files = sorted(glob.glob(
            os.path.join(self.base_path, self.sequence, 'images', 'left', '*.png')))
        self.rgb_right_files = sorted(glob.glob(
            os.path.join(self.base_path, self.sequence, 'images', 'right', '*.png')))
        self.rgb_mono_files = sorted(glob.glob(
            os.path.join(self.base_path, self.sequence, 'images', 'mono', '*.png')))

        if self.frames is not None:
            self.rgb_left_files = [self.rgb_left_files[i] for i in self.frames]
            self.rgb_right_files = [self.rgb_right_files[i]
                                    for i in self.frames]
            self.rgb_mono_files = [self.mono_files[i] for i in self.frames]

    def __len__(self):
        return self.num_frames

    def get_rgb_left(self, idx):
        """Load RGB image from file."""
        return self._load_image(self.rgb_left_files[idx], mode='RGB', dtype=np.uint8)

    def get_rgb_right(self, idx):
        """Load RGB image from file."""
        return self._load_image(self.rgb_right_files[idx], mode='RGB', dtype=np.uint8)

    def get_rgb_mono(self, idx):
        """Load RGB image from file."""
        return self._load_image(self.rgb_mono_files[idx], mode='RGB', dtype=np.uint8)

    def get_gray_left(self, idx):
        """Load grayscale image from file."""
        return self._load_image(self.rgb_left_files[idx], mode='L', dtype=np.uint8)

    def get_gray_right(self, idx):
        """Load grayscale image from file."""
        return self._load_image(self.rgb_right_files[idx], mode='L', dtype=np.uint8)

    def get_gray_mono(self, idx):
        """Load grayscale image from file."""
        return self._load_image(self.rgb_mono_files[idx], mode='L', dtype=np.uint8)

    def get_rgb_spatial_pair(self, idx):
        return self._get_spatial_pair(idx, 'RGB')

    def get_gray_spatial_pair(self, idx):
        return self._get_spatial_pair(idx, 'L')

    def _get_spatial_pair(self, idx, mode):
        tf = self.transforms_spatial[idx]
        image_man_file = self._get_image_filename(tf.run_id_man,
                                                  tf.vertex_id_man)
        image_auto_file = self._get_image_filename(tf.run_id_auto,
                                                   tf.vertex_id_auto)

        return (self._load_image(image_man_file, mode=mode, dtype=np.uint8),
                self._load_image(image_auto_file, mode=mode, dtype=np.uint8))

    def _get_image_filename(self, run_id, vertex_id):
        return os.path.join(self.base_path, 'run_{:06d}'.format(
            run_id), 'images', self.camera, '{:06d}.png'.format(vertex_id))

    def _vertex_has_image(self, run_id, vertex_id):
        image_file = self._get_image_filename(run_id, vertex_id)
        return os.path.exists(image_file)

    def _load_image(self, impath, mode='RGB', dtype=np.float, factor=1):
        """Load image from file."""
        im = Image.open(impath).convert(mode)
        return (np.array(im) / factor).astype(dtype)

    def _load_timestamps_and_transforms(self):
        """Load timestamps and spatial/temporal transforms from file."""
        timestamp_file = os.path.join(
            self.base_path, self.sequence, 'images', 'timestamps.txt')
        temporal_file = os.path.join(
            self.base_path, self.sequence, 'transforms', 'transforms_temporal.txt')
        spatial_file = os.path.join(
            self.base_path, self.sequence, 'transforms', 'transforms_spatial.txt')

        self.timestamps = []
        self.transforms_temporal = []  # within-sequence (VO)
        self.transforms_spatial = []   # sequence-to-sequence

        # Read and parse the timestamps
        with open(timestamp_file, 'r') as f:
            for line in f.readlines():
                line = line.split(',')
                line[0] = int(line[0])
                line[1] = float(line[1])
                self.timestamps.append(Timestamp(*line))

        # Read and parse the temporal transforms
        with open(temporal_file, 'r') as f:
            for line in f.readlines():
                line = line.split(',')
                vertex_ids = [int(x) for x in line[:4]]

                if self._vertex_has_image(*vertex_ids[0:2]) \
                        and self._vertex_has_image(*vertex_ids[2:4]):
                    Tmatrix = np.array([float(x)
                                        for x in line[4:]]).reshape(4, 4)
                    self.transforms_temporal.append(
                        TemporalTransform(*vertex_ids, Tmatrix))

        # Read and parse the spatial transforms (don't exist for manual runs)
        try:
            with open(spatial_file, 'r') as f:
                for line in f.readlines():
                    line = line.split(',')
                    vertex_ids = [int(x) for x in line[:4]]
                    if self._vertex_has_image(*vertex_ids[0:2]) \
                            and self._vertex_has_image(*vertex_ids[2:4]):
                        Tmatrix = np.array([float(x)
                                            for x in line[4:]]).reshape(4, 4)
                        self.transforms_spatial.append(
                            SpatialTransform(*vertex_ids, Tmatrix))
        except FileNotFoundError as e:
            print(e)


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, opts, sequence, random_crop, cache_matches=True, **kwargs):
        self.opts = opts
        self.random_crop = random_crop
        self.cache_matches = cache_matches

        self.dataset = Dataset(self.opts.data_dir, sequence, **kwargs)

        self.vo_params = viso2.Mono_parameters()  # Use ransac
        self.vo_params.ransac_iters = 400
        self.vo = viso2.VisualOdometryMono(self.vo_params)

        if self.cache_matches:
            self.matches11 = [None for _ in range(len(self.dataset))]
            self.matches12 = [None for _ in range(len(self.dataset))]
            self.matches22 = [None for _ in range(len(self.dataset))]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get images
        rgb1, rgb2 = self.dataset.get_rgb_spatial_pair(idx)
        rgb1 = Image.fromarray(rgb1)
        rgb2 = Image.fromarray(rgb2)

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
            if self.cache_matches and self.matches12[idx] is not None:
                matches11 = self.matches11[idx]
                matches12 = self.matches12[idx]
                matches22 = self.matches22[idx]
            else:
                matches11 = self._get_match_count(gray1, gray1)
                matches12 = self._get_match_count(gray1, gray2)
                matches22 = self._get_match_count(gray2, gray2)
                if self.cache_matches:
                    self.matches11[idx] = matches11
                    self.matches12[idx] = matches12
                    self.matches22[idx] = matches22
            # matchability_score = matches12 / matches11

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
