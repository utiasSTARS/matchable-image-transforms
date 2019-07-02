import skimage.draw
import ipdb
import glob
import os.path
import argparse
import pickle

import numpy as np

import progress.bar

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as manimation
matplotlib.use('Agg')


def to_gray(rgb):
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    gray = np.repeat(gray[:, :, np.newaxis], 3, axis=2)
    return gray


def draw_matches(im1, im2, matches, axis=1):
    impair = np.concatenate((im1, im2), axis=axis)

    if matches.size == 0:
        return impair

    if axis == 0:
        matches[:, 3] += im2.shape[0]
    elif axis == 1:
        matches[:, 2] += im2.shape[1]

    for m in matches.astype(np.int):
        rr, cc = skimage.draw.line(m[0], m[1], m[2], m[3])
        impair[cc, rr, :] = [1, 1, 0]

    return impair


def make_grid(imfiles, fidx, dir='h', matches=None):
    axes = {'v': [1, 0],
            'h': [0, 1]}

    impairs = []
    for key, val in imfiles.items():
        try:
            im1 = mpimg.imread(val[0][fidx])
            im2 = mpimg.imread(val[1][fidx])

            # hoffset = int(0.5 * (im1.shape[1] - 256))
            # im1 = im1[:, hoffset:-hoffset, :]
            # im2 = im2[:, hoffset:-hoffset, :]

            if key == 'gray':
                im1 = to_gray(im1)
                im2 = to_gray(im2)

            if matches and key != 'rgb':
                impairs.append(draw_matches(
                    im1, im2, matches[key][fidx], axis=axes[dir][0]))
            else:
                impairs.append(np.concatenate((im1, im2), axis=axes[dir][0]))
        except IndexError:
            pass

    return np.concatenate(impairs, axis=axes[dir][1])


parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str)
parser.add_argument('seq', type=str)
parser.add_argument('--subdir', type=str)
parser.add_argument('--saveframe', type=int)
parser.add_argument('--plot_matches', action='store_true')
args = parser.parse_args()

expdir = '/media/raid5-array/experiments/matchability/{}/'.format(args.dataset)
seqdir = '{}-test'.format(args.seq)
matcherdir = 'matcher'
modeldirs = ['logrgb-noenc', 'logrgb-enc', 'learned-noenc', 'learned-enc']
subdir = args.subdir if args.subdir else ''

os.makedirs(os.path.join(expdir, seqdir, subdir), exist_ok=True)

imfiles = {}
matches = {}
for mdir in modeldirs:
    imgdir = os.path.join(expdir, seqdir, mdir, subdir, 'test_best')
    matchfile = os.path.join(expdir, seqdir, mdir, subdir, 'matches.pickle')
    matchdict = pickle.load(open(matchfile, 'rb'))

    if not imfiles or len(imfiles['rgb'][0]) == 0:
        imfiles['rgb'] = [sorted(glob.glob(os.path.join(imgdir, 'rgb1', '*.png'))),
                          sorted(glob.glob(os.path.join(imgdir, 'rgb2', '*.png')))]
        imfiles['gray'] = imfiles['rgb']
        matches['gray'] = matchdict['inliers_rgb12']

    imfiles[mdir] = [sorted(glob.glob(os.path.join(imgdir, 'out1', '*.png'))),
                     sorted(glob.glob(os.path.join(imgdir, 'out2', '*.png')))]
    matches[mdir] = matchdict['inliers_out12']

video = manimation.writers['ffmpeg'](fps=20)
dpi = 384
grid_dir = 'v' if 'virtual-kitti' in args.dataset else 'h'
# grid_dir = 'h'

test_frame = make_grid(imfiles, 0, grid_dir)

subplotpars = matplotlib.figure.SubplotParams(
    left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
fig = plt.figure(figsize=(test_frame.shape[1] / dpi,
                          test_frame.shape[0] / dpi),
                 dpi=dpi, frameon=False, subplotpars=subplotpars)
ax = fig.add_subplot(111, frame_on=False)
ax.set_aspect('equal')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

im = ax.imshow(test_frame)

if args.plot_matches:
    videofile = os.path.join(expdir, seqdir, subdir,
                             args.seq + '_grid_matches.mp4')
else:
    videofile = os.path.join(expdir, seqdir, subdir, args.seq + '_grid.mp4')


with video.saving(fig, videofile, dpi=dpi):
    bar = progress.bar.Bar('Video grid', max=len(imfiles['rgb'][0]))
    for fidx in range(len(imfiles['rgb'][0])):
        if args.plot_matches:
            frame = make_grid(imfiles, fidx, grid_dir, matches)
        else:
            frame = make_grid(imfiles, fidx, grid_dir, None)
        im.set_data(frame)
        video.grab_frame()

        if fidx == args.saveframe:
            if args.plot_matches:
                outfile = os.path.join(
                    expdir, seqdir, subdir, '{}_grid_{:05}_matches.png'.format(args.seq, fidx))
            else:
                outfile = os.path.join(
                    expdir, seqdir, subdir, '{}_grid_{:05}.png'.format(args.seq, fidx))
            mpimg.imsave(outfile, frame)

        bar.next()
    bar.finish()
