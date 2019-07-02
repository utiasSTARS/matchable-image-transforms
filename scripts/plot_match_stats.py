import ipdb
import glob
import os
import argparse

import numpy as np
from PIL import Image
import viso2

import pickle

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str)
parser.add_argument('--subdir', type=str)
args = parser.parse_args()

expdir = '/media/raid5-array/experiments/matchability/{}/'.format(args.dataset)
seqdirs = sorted([d for d in next(os.walk(expdir))[1] if '-test' in d])
matcherdir = 'matcher'
modeldirs = ['logrgb-noenc', 'logrgb-enc', 'learned-noenc', 'learned-enc']
subdir = args.subdir if args.subdir else ''

# Matcher net
all_matcher_data = {}
all_match_counts = {}
all_inlier_counts = {}
raster_scatter = True
dpi = 600

for seqdir in seqdirs:
    seqid = seqdir.split('-test')[0]

    os.makedirs(os.path.join(expdir, seqdir, subdir), exist_ok=True)
    lossfile = os.path.join(expdir, seqdir, matcherdir,
                            subdir, 'test_best', 'loss.csv')
    matcher_data = {}
    with open(lossfile, 'r') as f:
        for line in f.readlines():
            line = line.rstrip().split(',')
            try:
                line = [float(entry) for entry in line]
                for key, entry in zip(keys, line):
                    try:
                        matcher_data[key].append(entry)
                    except KeyError:
                        matcher_data[key] = [entry]
            except ValueError:
                print('Got header line')
                keys = [entry for entry in line]

    for key in matcher_data.keys():
        matcher_data[key] = np.array(matcher_data[key]).squeeze()

    all_matcher_data[seqdir] = matcher_data
    # squared error
    # matcher_data['matcher/loss'] = np.sqrt(matcher_data['matcher/loss'])

    fig, ax = plt.subplots(figsize=(3, 1.5), tight_layout=True, dpi=dpi)
    ax.hist(matcher_data['matcher/loss'], bins=21, density=True)
    ax.grid(which='both', linestyle=':', linewidth=0.5)
    ax.set_title('mean: {:.2f}, std: {:.2f} | avg {:.0f}'.format(np.mean(
        matcher_data['matcher/loss']), np.std(matcher_data['matcher/loss']), np.mean(matcher_data['matcher/target'])), fontsize=10)
    # ax.set_xlabel('Error', fontsize=10)
    ax.set_ylabel('PDF', fontsize=10)
    outfile = os.path.join(expdir, seqdir, subdir,
                           '{}_matcher_loss.pdf'.format(seqid))
    print('Saving to {}'.format(outfile))
    fig.savefig(outfile, bbox_inches='tight', transparent=True)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 4), tight_layout=True, dpi=dpi)
    ax.scatter(matcher_data['matcher/target'],
               matcher_data['matcher/est'], s=2, rasterized=raster_scatter)
    xline = np.linspace(0, np.max(matcher_data['matcher/est']))
    ax.plot(xline, xline, 'r--')
    ax.grid(which='both', linestyle=':', linewidth=0.5)
    # ax.set_title('{}'.format(seqid), fontsize=10)
    ax.set_xlabel('Actual', fontsize=16)
    ax.set_ylabel('Estimated', fontsize=16)
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    ax.set_xticks(ax.get_yticks())
    ax.set_xlim([0, np.max(matcher_data['matcher/est'])])
    ax.set_ylim(ax.get_xlim())
    ax.set_aspect('equal')
    outfile = os.path.join(
        expdir, seqdir, subdir, '{}_matcher_target_est.pdf'.format(seqid))
    print('Saving to {}'.format(outfile))
    fig.savefig(outfile, bbox_inches='tight', transparent=True)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3, 3), tight_layout=True, dpi=dpi)
    ax.scatter(matcher_data['matcher/target'],
               matcher_data['matcher/est'], s=2, rasterized=raster_scatter)
    xline = np.linspace(0, np.max(matcher_data['matcher/est']))
    ax.plot(xline, xline, 'r--')
    ax.grid(which='both', linestyle=':', linewidth=0.5)
    # ax.set_title('{}'.format(seqid), fontsize=10)
    ax.set_xlabel('Actual', fontsize=16)
    ax.set_ylabel('Estimated', fontsize=16)
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    ax.set_xticks(ax.get_yticks())
    ax.set_xlim([0, np.max(matcher_data['matcher/est'])])
    ax.set_ylim(ax.get_xlim())
    ax.set_aspect('equal')
    outfile = os.path.join(
        expdir, seqdir, subdir, '{}_matcher_target_est_small.pdf'.format(seqid))
    print('Saving to {}'.format(outfile))
    fig.savefig(outfile, bbox_inches='tight', transparent=True)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3, 1.5), tight_layout=True, dpi=dpi)
    ax.scatter(matcher_data['matcher/target'],
               matcher_data['matcher/loss'], s=2, rasterized=raster_scatter)
    ax.grid(which='both', linestyle=':', linewidth=0.5)
    # ax.set_title('{}'.format(seqid), fontsize=10)
    ax.set_xlabel('Target', fontsize=10)
    ax.set_ylabel('Loss', fontsize=10)
    outfile = os.path.join(
        expdir, seqdir, subdir, '{}_matcher_target_loss.pdf'.format(seqid))
    print('Saving to {}'.format(outfile))
    fig.savefig(outfile, bbox_inches='tight', transparent=True)
    plt.close(fig)

    # Feature matching
    match_counts = []
    inlier_counts = []
    match_counts_est = []
    got_rgb_matches = False
    for mdir in modeldirs:
        matchfile = os.path.join(expdir, seqdir, mdir,
                                 subdir, 'matches.pickle')
        try:
            matches = pickle.load(open(matchfile, 'rb'))
            if not got_rgb_matches:
                match_counts = [[m.shape[0]
                                 for m in matches['matches_rgb12']]] + match_counts
                inlier_counts = [[m.shape[0]
                                  for m in matches['inliers_rgb12']]] + inlier_counts
                match_counts_est = [
                    matches['match_count_rgb12_est']] + match_counts_est
                got_rgb_matches = True

            match_counts.append([m.shape[0] for m in matches['matches_out12']])
            inlier_counts.append([m.shape[0]
                                  for m in matches['inliers_out12']])
            match_counts_est.append(matches['match_count_out12_est'])
        except FileNotFoundError:
            match_counts.append([0])

    all_match_counts[seqdir] = match_counts
    all_inlier_counts[seqdir] = inlier_counts

    legend = ['Gray', 'SumLog', 'SumLog-E', 'MLP', 'MLP-E']
    # legend = ['Gray', 'SumLog', 'SumLog-E', 'MLP', 'MLP-E', 'Conv-E', 'UNet']
    header = ['model', 'mean', 'std', 'med', 'min', 'max']
    outfile = outfile = os.path.join(
        expdir, seqdir, subdir, '{}_matchstats.csv'.format(seqid))
    print('Saving to {}'.format(outfile))
    with open(outfile, 'wt') as file:
        file.write(','.join(header) + '\n')
        for model, counts in zip(legend, inlier_counts):
            try:
                counts = np.array(counts)
                entries = [np.mean(counts), np.std(counts), np.median(
                    counts), np.min(counts), np.max(counts)]
                entries = [model] + ['{:.0f}'.format(e) for e in entries]
                line = ','.join(entries) + '\n'
                file.write(line)
            except ValueError:
                print('Missing data for {}'.format(model))

    fig, ax = plt.subplots(figsize=(4.5, 2.5), tight_layout=True, dpi=dpi)
    ax.grid(True, which='both', axis='y', linestyle=':')
    ax.boxplot(inlier_counts, labels=legend)
    # ax.set_ylabel('Match Count (Inliers)', fontsize=10)
    ax.set_title('Match Counts (Inliers)')
    ylim = ax.get_ylim()
    ax.set_ylim((0, ylim[1]))

    outfile = os.path.join(expdir, seqdir, subdir,
                           '{}_matches.pdf'.format(seqid))
    print('Saving to {}'.format(outfile))
    fig.savefig(outfile, bbox_inches='tight', transparent=True)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(legend),
                             figsize=(5*len(legend), 4), tight_layout=True, dpi=dpi)
    for ax, target, est, label in zip(axes, match_counts, match_counts_est, legend):
        ax.scatter(target, est, s=2, rasterized=raster_scatter)
        xline = np.linspace(0, np.max(est))
        ax.plot(xline, xline, 'r--')
        ax.grid(which='both', linestyle=':', linewidth=0.5)
        ax.set_title('{}'.format(label), fontsize=16)
        ax.set_xlabel('Actual', fontsize=16)
        ax.set_ylabel('Estimated', fontsize=16)
        ax.xaxis.set_tick_params(labelsize=16)
        ax.yaxis.set_tick_params(labelsize=16)
        ax.set_xticks(ax.get_yticks())
        ax.set_xlim([0, np.max(est)])
        ax.set_ylim(ax.get_xlim())
        ax.set_aspect('equal')
    outfile = os.path.join(
        expdir, seqdir, subdir, '{}_out_matcher_target_est.pdf'.format(seqid))
    print('Saving to {}'.format(outfile))
    fig.savefig(outfile, bbox_inches='tight', transparent=True)
    plt.close(fig)

all_matcher_target = np.concatenate(
    [all_matcher_data[seq]['matcher/target'] for seq in seqdirs], axis=0)
all_matcher_est = np.concatenate(
    [all_matcher_data[seq]['matcher/est'] for seq in seqdirs], axis=0)

fig, ax = plt.subplots(figsize=(4, 4), tight_layout=True, dpi=dpi)
ax.scatter(all_matcher_target,
           all_matcher_est, s=2, rasterized=raster_scatter)
xline = np.linspace(0, np.max(all_matcher_est))
ax.plot(xline, xline, 'r--')
ax.grid(which='both', linestyle=':', linewidth=0.5)
# ax.set_title('{}'.format(seqid), fontsize=10)
ax.set_xlabel('Actual', fontsize=16)
ax.set_ylabel('Estimated', fontsize=16)
ax.set_title(r'$r = {:.3f}$'.format(
    np.corrcoef(all_matcher_target, all_matcher_est)[0, 1]), fontsize=16)
ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.set_tick_params(labelsize=16)
ax.set_xticks(ax.get_yticks())
ax.set_xlim([0, np.max(all_matcher_est)])
ax.set_ylim(ax.get_xlim())
ax.set_aspect('equal')
outfile = os.path.join(
    expdir, '{}{}_matcher_target_est.pdf'.format(args.dataset, subdir))
print('Saving to {}'.format(outfile))
fig.savefig(outfile, bbox_inches='tight', transparent=True)
plt.close(fig)

for loc_thresh in [10, 20, 30, 50, 100]:
    fig, ax = plt.subplots(figsize=(8, 2), dpi=dpi)
    ax.grid(True, which='both', axis='y', linestyle=':')

    all_loc_succ = {}
    for seq, match_counts in all_inlier_counts.items():
        all_loc_succ[seq] = [100 * (np.array(counts) >= loc_thresh).sum() / len(counts)
                             for counts in match_counts]

    barwidth = 0.12
    xticks = ['{:04d}'.format(int(''.join(filter(str.isdigit, seq))))
              for seq in all_loc_succ.keys()]
    xdata = np.arange(len(xticks))
    handles = []
    for idx, label in enumerate(legend):
        ydata = [all_loc_succ[seq][idx] for seq in all_loc_succ.keys()]
        # ax.plot(xdata, ydata, linestyle='--',
        #         marker='s', linewidth=1, markersize=4)
        h = ax.bar(xdata + idx*barwidth, ydata,
                   width=barwidth, edgecolor='white', label=label)
        # ax.scatter(xdata, ydata, marker='s', s=2)
        handles.append(h)
    ax.set_xticks(xdata + 0.5*barwidth*(len(legend)-1))
    ax.set_xticklabels(xticks)
    ax.tick_params(axis='x', rotation=90)
    ax.set_ylim([0, 110])
    ax.set_title(
        r'Localization Success \% ($\geq {}$ inliers)'.format(loc_thresh))
    fig.legend(handles, legend, loc='lower center', ncol=len(legend))
    # fig.tight_layout()
    fig.subplots_adjust(bottom=0.35)

    outfile = os.path.join(
        expdir, '{}{}_loc_succ_{}.pdf'.format(args.dataset, subdir, loc_thresh))
    print('Saving to {}'.format(outfile))
    fig.savefig(outfile, bbox_inches='tight', transparent=True)
    plt.close(fig)
