from datetime import datetime
import argparse
import os.path

from torch.utils.data import ConcatDataset

from tensorboardX import SummaryWriter

from matchability.options import Options
from matchability.models import MatcherModel, TransformModel
from matchability.datasets import vkitti
from matchability.utils import save_matches
from matchability import experiment


### COMMAND LINE ARGUMENTS ###
parser = argparse.ArgumentParser()
parser.add_argument('stage', type=str, choices=['train', 'test', 'both'])
parser.add_argument('model', type=str, choices=[
                    'matcher',
                    'logrgb-noenc', 'logrgb-enc',
                    'learned-noenc', 'learned-enc'])
parser.add_argument('--resume', action='store_true')
parser.add_argument('--use_param_avg', action='store_true')
parser.add_argument('--matches_only', action='store_true')
args = parser.parse_args()

resume_from_epoch = 'latest' if args.resume else None

### CONFIGURATION ###
opts = Options()
opts.data_dir = '/path/to/dataset'
opts.results_dir = '/path/to/results'

opts.model = args.model

opts.image_load_size = (375, 1242)
opts.image_final_size = (192, 636)
opts.random_crop = False

opts.dataloader_workers = 3

opts.train_epochs = 10
opts.batch_size = 8

opts.lr = 1e-4

opts.matcher_source_channels = 1
opts.matcher_output_channels = 1
opts.matcher_init_channels = 8
opts.matcher_feature_blocks = 3
opts.matcher_concat_blocks = 3
opts.matcher_max_channels = 64
opts.matcher_final_kernel_size = (3, 10)

opts.enc_source_channels = 3
opts.enc_output_channels = 3
opts.enc_init_channels = 32
opts.enc_feature_blocks = 3
opts.enc_concat_blocks = 3
opts.enc_max_channels = 256
opts.enc_final_kernel_size = (3, 10)

opts.gen_source_channels = 3 + \
    opts.enc_output_channels if opts.model == 'learned-enc' else 3
opts.gen_output_channels = 1
opts.gen_init_channels = 16
opts.gen_layers = 5
opts.gen_max_channels = 16

opts.lambda_matcher = 1.
opts.lambda_gen_match = 1.

opts.max_interval = 3

### SET TRAINING, VALIDATION AND TEST SETS ###
seqs = ['0001', '0002', '0006', '0018', '0020']
conds = ['clone', 'morning', 'overcast', 'sunset']

train_conds = conds
val_conds = train_conds

test_conds = [conds[1], conds[3]]  # Morning, Sunset
# test_conds = [conds[0], conds[2]] # Clone, Overcast

for test_seq in seqs:
    train_seqs = seqs.copy()
    train_seqs.remove(test_seq)
    val_seqs = [test_seq]

    train_data = []
    for seq in train_seqs:
        for i in range(len(train_conds)):
            for j in range(i, len(train_conds)):
                cond1 = train_conds[i]
                cond2 = train_conds[j]
                print('Train {}: {} --> {}'.format(seq, cond1, cond2))
                data = vkitti.TorchDataset(opts, seq, cond1, cond2,
                                           opts.random_crop)
                train_data.append(data)
    train_data = ConcatDataset(train_data)

    val_data = []
    for seq in val_seqs:
        for i in range(len(val_conds)):
            for j in range(i, len(val_conds)):
                cond1 = val_conds[i]
                cond2 = val_conds[j]
                print('Val {}: {} --> {}'.format(seq, cond1, cond2))
                data = vkitti.TorchDataset(opts, seq, cond1, cond2, False)
                val_data.append(data)
    val_data = ConcatDataset(val_data)

    test_data = vkitti.TorchDataset(
        opts, test_seq, test_conds[0], test_conds[1], False)

    ### TRAIN  / TEST ###
    matcher_name = '{}-test/matcher'.format(test_seq)
    gen_name = '{}-test/{}'.format(test_seq, opts.model)

    if args.model == 'matcher':
        opts.compute_matches = True
        opts.experiment_name = matcher_name
        matcher = MatcherModel(opts)

        if args.stage == 'train' or args.stage == 'both':
            print(opts)
            opts.save_txt()
            experiment.train(opts, matcher, train_data,
                             val_data, opts.train_epochs, resume_from_epoch=resume_from_epoch)

        if args.stage == 'test' or args.stage == 'both':
            expdir = matcher_name + \
                '/{}-{}-test'.format(test_conds[0], test_conds[1])
            experiment.test(opts, matcher, test_data, expdir=expdir,
                            save_loss=True, save_images=False)

    else:
        opts.compute_matches = True
        opts.experiment_name = matcher_name
        matcher = MatcherModel(opts)
        matcher.load_checkpoint('best')

        opts.experiment_name = gen_name
        gen = TransformModel(opts, matcher)

        if args.stage == 'train' or args.stage == 'both':
            print(opts)
            opts.save_txt()
            experiment.train(opts, gen, train_data, val_data,
                             opts.train_epochs,
                             resume_from_epoch=resume_from_epoch)

        if args.stage == 'test' or args.stage == 'both':
            opts.max_interval = 0
            experiment_name = gen_name
            expdir = gen_name + \
                '/{}-{}-test'.format(test_conds[0], test_conds[1])

            if not args.matches_only:
                experiment.test(opts, gen, test_data, expdir=expdir)

            logdir = os.path.join(opts.results_dir, expdir)
            save_matches(logdir, matcher)
