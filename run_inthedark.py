from datetime import datetime
import argparse
import os.path

from torch.utils.data import ConcatDataset

from tensorboardX import SummaryWriter

from matchability.options import Options
from matchability.models import MatcherModel, TransformModel
from matchability.datasets import robochunk
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
parser.add_argument('--matches_only', action='store_true')
args = parser.parse_args()

resume_from_epoch = 'latest' if args.resume else None

### CONFIGURATION ###
opts = Options()
opts.data_dir = '/path/to/dataset'
opts.results_dir = '/path/to/results'

opts.model = args.model

opts.image_load_size = (384, 512)
opts.image_final_size = (192, 256)
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
opts.matcher_final_kernel_size = (3, 4)

opts.enc_source_channels = 3
opts.enc_output_channels = 3
opts.enc_init_channels = 32
opts.enc_feature_blocks = 3
opts.enc_concat_blocks = 3
opts.enc_max_channels = 256
opts.enc_final_kernel_size = (3, 4)

opts.gen_source_channels = 3 + \
    opts.enc_output_channels if opts.model == 'learned-enc' else 3
opts.gen_output_channels = 1
opts.gen_init_channels = 16
opts.gen_layers = 5
opts.gen_max_channels = 16

opts.lambda_matcher = 1.
opts.lambda_gen_match = 1.

### SET TRAINING, VALIDATION AND TEST SETS ###
with open(os.path.join(opts.data_dir, 'metadata', 'runs_automatic.txt')) as f:
    seqs = f.readlines()[0].split(',')
    seqs = [int(entry) for entry in seqs]

seqs = ['run_{:06d}'.format(i) for i in seqs if i < 91]
val_seqs = ['run_{:06d}'.format(i) for i in [6, 27, 41, 58, 71, 83, 89]]
train_seqs = [seq for seq in seqs if seq not in val_seqs]
test_seqs = val_seqs
assert all([test_seq not in train_seqs for test_seq in test_seqs])

train_data = []
for seq in train_seqs:
    print('Train {}'.format(seq))
    data = robochunk.TorchDataset(opts, seq, opts.random_crop)
    train_data.append(data)
train_data = ConcatDataset(train_data)

val_data = []
for seq in val_seqs:
    print('Val {}'.format(seq))
    data = robochunk.TorchDataset(opts, seq, False)
    val_data.append(data)
val_data = ConcatDataset(val_data)

### TRAIN  / TEST ###
matcher_name = 'matcher'
gen_name = opts.model

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
        for seq in test_seqs:
            print('Test {}'.format(seq))
            test_data = robochunk.TorchDataset(opts, seq, False)

            expdir = os.path.join('{}-test'.format(seq), matcher_name)
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
        experiment_name = gen_name

        for seq in test_seqs:
            print('Test {}'.format(seq))
            test_data = robochunk.TorchDataset(opts, seq, False)
            expdir = os.path.join('{}-test'.format(seq), experiment_name)

            if not args.matches_only:
                experiment.test(opts, gen, test_data, expdir=expdir)

            logdir = os.path.join(opts.results_dir, expdir)
            save_matches(logdir, matcher)
