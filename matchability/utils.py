from . import transforms as custom_transforms
import torch
from torch import nn
from torchvision import transforms

import glob
import os.path
import viso2
from PIL import Image

import pickle
import progress.bar

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Agg')


def print_network(net):
    num_params = sum([param.numel() for param in net.parameters()])
    print(net)
    print('Total number of parameters: {}'.format(num_params))


def image_from_tensor(tensor, image_mean=0., image_std=1., size=None):
    tensor = tensor.cpu()

    if tensor.size(0) == 1:
        # need to make a copy here for unnormalization to work right
        tensor = tensor.repeat(3, 1, 1)

    tf_list = [custom_transforms.UnNormalize(image_mean, image_std),
               custom_transforms.Clamp(0, 1),
               transforms.ToPILImage()]  # multiplication by 255 happens here

    if size is not None:
        tf_list.append(transforms.Resize(size, interpolation=Image.NEAREST))

    transform = transforms.Compose(tf_list)

    return transform(tensor)


def concatenate_dicts(*dicts):
    concat_dict = {}
    for key in dicts[0]:
        concat_dict[key] = []
        for d in dicts:
            val = d[key]
            if isinstance(val, list):
                concat_dict[key] = concat_dict[key] + val
            else:
                concat_dict[key].append(val)

    return concat_dict


def compute_dict_avg(dict):
    avg_dict = {}
    for key, val in dict.items():
        avg_dict[key] = np.mean(np.array(val))
    return avg_dict


def tag_dict_keys(dict, tag):
    new_dict = {}
    for key, val in dict.items():
        new_key = key + '/' + tag
        new_dict[new_key] = val
    return new_dict


def save_matches(logdir, model, which_epoch='best'):
    imgdir = logdir + '/test_{}'.format(which_epoch)
    rgb1files = sorted(glob.glob(os.path.join(imgdir, 'rgb1', '*.png')))
    rgb2files = sorted(glob.glob(os.path.join(imgdir, 'rgb2', '*.png')))
    out1files = sorted(glob.glob(os.path.join(imgdir, 'out1', '*.png')))
    out2files = sorted(glob.glob(os.path.join(imgdir, 'out2', '*.png')))

    vo_params = viso2.Mono_parameters()  # Use ransac
    vo_params.ransac_iters = 400
    vo = viso2.VisualOdometryMono(vo_params)

    model.set_mode('eval')

    make_grayscale = transforms.Grayscale()
    make_normalized_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda tensor: tensor.unsqueeze(dim=0)),
        transforms.Lambda(lambda tensor: tensor.to(model.device))
    ])

    matches_rgb11 = []
    matches_rgb12 = []
    matches_out12 = []
    inliers_rgb11 = []
    inliers_rgb12 = []
    inliers_out12 = []
    match_count_rgb11_est = []
    match_count_rgb12_est = []
    match_count_out12_est = []

    bar = progress.bar.Bar('Computing feature matches', max=len(rgb1files))
    for rgb1file, rgb2file, out1file, out2file in zip(
            rgb1files, rgb2files, out1files, out2files):
        rgb1 = make_grayscale(Image.open(rgb1file))
        rgb2 = make_grayscale(Image.open(rgb2file))
        out1 = make_grayscale(Image.open(out1file))
        out2 = make_grayscale(Image.open(out2file))

        rgb1_ten = make_normalized_tensor(rgb1)
        rgb2_ten = make_normalized_tensor(rgb2)
        out1_ten = make_normalized_tensor(out1)
        out2_ten = make_normalized_tensor(out2)

        rgb1 = np.array(rgb1)
        rgb2 = np.array(rgb2)
        out1 = np.array(out1)
        out2 = np.array(out2)

        vo.process_frame(rgb1, rgb1)
        matches = vo.getMatches()
        inliers = vo.getInlierMatches()
        matches_rgb11.append(
            np.array([[m.u1p, m.v1p, m.u1c, m.v1c] for m in matches]))
        inliers_rgb11.append(
            np.array([[m.u1p, m.v1p, m.u1c, m.v1c] for m in inliers]))

        vo.process_frame(rgb1, rgb2)
        matches = vo.getMatches()
        inliers = vo.getInlierMatches()
        matches_rgb12.append(
            np.array([[m.u1p, m.v1p, m.u1c, m.v1c] for m in matches]))
        inliers_rgb12.append(
            np.array([[m.u1p, m.v1p, m.u1c, m.v1c] for m in inliers]))

        vo.process_frame(out1, out2)
        matches = vo.getMatches()
        inliers = vo.getInlierMatches()
        matches_out12.append(
            np.array([[m.u1p, m.v1p, m.u1c, m.v1c] for m in matches]))
        inliers_out12.append(
            np.array([[m.u1p, m.v1p, m.u1c, m.v1c] for m in inliers]))

        with torch.no_grad():
            model.forward(rgb1_ten, rgb2_ten, compute_loss=False)
            match_count_rgb12_est.append(
                model.matches_est.detach().cpu().squeeze().numpy().tolist())

            model.forward(rgb1_ten, rgb1_ten, compute_loss=False)
            match_count_rgb11_est.append(
                model.matches_est.detach().cpu().squeeze().numpy().tolist())

            model.forward(out1_ten, out2_ten, compute_loss=False)
            match_count_out12_est.append(
                model.matches_est.detach().cpu().squeeze().numpy().tolist())

        bar.next()
    bar.finish()

    mdict = {'matches_rgb11': matches_rgb11,
             'matches_rgb12': matches_rgb12,
             'matches_out12': matches_out12,
             'inliers_rgb11': inliers_rgb11,
             'inliers_rgb12': inliers_rgb12,
             'inliers_out12': inliers_out12,
             'match_count_rgb11_est': match_count_rgb11_est,
             'match_count_rgb12_est': match_count_rgb12_est,
             'match_count_out12_est': match_count_out12_est}
    savefile = os.path.join(logdir, 'matches.pickle')
    print('Saving matches to {}'.format(savefile))
    pickle.dump(mdict, open(savefile, 'wb'))
