import torch
from torch import nn
from torchvision import transforms

import os
import numpy as np
import viso2

from . import utils
from . import networks
from . import transforms as custom_transforms


class MatcherModel:
    def __init__(self, opts):
        self.opts = opts

        self.device = torch.device(self.opts.device)

        # Initialize networks
        self.net = (networks.SiameseNet(
            source_channels=self.opts.matcher_source_channels,
            output_channels=self.opts.matcher_output_channels,
            init_channels=self.opts.matcher_init_channels,
            feature_blocks=self.opts.matcher_feature_blocks,
            concat_blocks=self.opts.matcher_concat_blocks,
            max_channels=self.opts.matcher_max_channels,
            final_kernel_size=self.opts.matcher_final_kernel_size
        )).to(self.device)

        print('\n{:-^50}'.format(' Network initialized '))
        utils.print_network(self.net)
        print('-' * 50 + '\n')

        # Set loss function
        self.loss_function = nn.MSELoss()

        print('\n{:-^50}'.format(' Loss functions '))
        print('Match count loss: {}'.format(self.loss_function))
        print('-' * 50 + '\n')

        # Set optimizer
        self.opt = torch.optim.Adam(self.net.parameters(),
                                    lr=self.opts.lr)

    def set_mode(self, mode):
        """Set the network to train/eval mode. Affects dropout and batchnorm."""
        if mode == 'train':
            self.net.train()
        elif mode == 'eval':
            self.net.eval()
        else:
            raise ValueError(
                "Got invalid mode '{}'. Valid options are 'train' and 'eval'.".format(mode))

    def set_data(self, data):
        """Set the input tensors"""
        self.gray1 = data['gray1']
        self.gray2 = data['gray2']
        self.matches11 = data['matches11']
        self.matches12 = data['matches12']
        self.matches22 = data['matches22']

        self.gray112 = torch.cat(
            [self.gray1, self.gray1, self.gray2], dim=0).to(self.device)
        self.gray122 = torch.cat(
            [self.gray1, self.gray2, self.gray2], dim=0).to(self.device)
        self.matches = torch.cat(
            [self.matches11, self.matches12, self.matches22], dim=0).to(self.device)

        self.matches = self.matches.unsqueeze(
            dim=1).unsqueeze(
            dim=2).unsqueeze(
            dim=3)

    def forward(self, image1, image2, compute_loss=True):
        """Evaluate the forward pass of the matcher proxy model."""
        self.matches_est = self.net.forward(image1, image2)

        if compute_loss:
            self.loss = self.loss_function(self.matches_est,
                                           self.matches)
            self.loss *= self.opts.lambda_matcher

    def optimize(self):
        """Do one step of training with the current input tensors"""
        self.forward(self.gray112, self.gray122)
        self.opt.zero_grad()
        self.loss.backward()
        self.opt.step()

    def test(self, compute_loss=True):
        """Evaluate the model and test loss without optimizing"""
        with torch.no_grad():
            self.forward(self.gray112, self.gray122, compute_loss)

    def get_images(self, select_images=None):
        """Return a dictionary of the current images"""
        return {}

    def get_errors(self):
        """Return a dictionary of the current errors"""
        error_dict = {
            'matcher/loss': self.loss.item(),
            'matcher/eval_loss': self.loss.sqrt().item()
        }

        return error_dict

    def get_data(self):
        target = self.matches
        est = self.matches_est.detach()
        err = est - target
        data_dict = {
            'matcher/target': target.cpu().squeeze().numpy().tolist(),
            'matcher/est': est.cpu().squeeze().numpy().tolist(),
            'matcher/loss': err.cpu().squeeze().numpy().tolist()
        }

        return data_dict

    def save_checkpoint(self, epoch, label):
        """Save the model to file"""
        model_dir = os.path.join(
            self.opts.results_dir, self.opts.experiment_name, 'checkpoints')
        os.makedirs(model_dir, exist_ok=True)
        model_file = os.path.join(model_dir, '{}_net.pth.tar'.format(label))

        model_dict = {'epoch': epoch,
                      'label': label,
                      'net_state_dict': self.net.state_dict()}

        print("Saving model to {}".format(model_file))
        torch.save(model_dict, model_file)

    def load_checkpoint(self, label):
        """Load a model from file"""
        model_dir = os.path.join(
            self.opts.results_dir, self.opts.experiment_name, 'checkpoints')
        model_file = os.path.join(model_dir, '{}_net.pth.tar'.format(label))

        print("Loading model from {}".format(model_file))
        model_dict = torch.load(model_file, map_location=self.opts.device)
        self.net.to(self.device)

        self.net.load_state_dict(model_dict['net_state_dict'])

        return model_dict['epoch']


class TransformModel:
    def __init__(self, opts, matcher=None):
        self.opts = opts

        self.device = torch.device(self.opts.device)

        # Initialize ground truth matcher (non-differentiable)
        self.vo_params = viso2.Mono_parameters()  # Use ransac
        self.vo_params.ransac_iters = 400
        self.vo = viso2.VisualOdometryMono(self.vo_params)

        # Initialize networks
        self.matcher = matcher
        if self.matcher is None:
            self.matcher = MatcherModel(self.opts)

        if self.opts.model == 'logrgb-noenc':
            self.enc = (networks.Constant(
                self.opts.enc_output_channels)).to(self.device)
        else:
            self.enc = (networks.SiameseNet(
                source_channels=self.opts.enc_source_channels,
                output_channels=self.opts.enc_output_channels,
                init_channels=self.opts.enc_init_channels,
                feature_blocks=self.opts.enc_feature_blocks,
                concat_blocks=self.opts.enc_concat_blocks,
                max_channels=self.opts.enc_max_channels,
                final_kernel_size=self.opts.enc_final_kernel_size
            )).to(self.device)

        self.gen = (networks.TransformerNet(
            source_channels=self.opts.gen_source_channels,
            output_channels=self.opts.gen_output_channels,
            init_channels=self.opts.gen_init_channels,
            max_channels=self.opts.gen_max_channels,
            num_layers=self.opts.gen_layers,
        )).to(self.device)

        print('\n{:-^50}'.format(' Encoder network initialized '))
        utils.print_network(self.enc)
        print('-' * 50 + '\n')

        print('\n{:-^50}'.format(' Generator network initialized '))
        utils.print_network(self.gen)
        print('-' * 50 + '\n')

        # Set optimizers
        self.enc_opt = torch.optim.Adam(self.enc.parameters(),
                                        lr=self.opts.lr)
        self.gen_opt = torch.optim.Adam(self.gen.parameters(),
                                        lr=self.opts.lr)

    def set_mode(self, mode):
        """Set the network to train/eval mode. Affects dropout and batchnorm."""
        if mode == 'train':
            self.enc.train()
            self.gen.train()
        elif mode == 'eval':
            self.enc.eval()
            self.gen.eval()
            self.matcher.set_mode('eval')
        else:
            raise ValueError(
                "Got invalid mode '{}'. Valid options are 'train' and 'eval'.".format(mode))

    def set_data(self, data):
        """Set the input tensors"""
        self.rgb1 = data['rgb1'].to(self.device)
        self.rgb2 = data['rgb2'].to(self.device)
        self.gray1 = data['gray1'].to(self.device)
        self.gray2 = data['gray2'].to(self.device)
        self.logrgb1 = data['logrgb1'].to(self.device)
        self.logrgb2 = data['logrgb2'].to(self.device)
        self.matches_gray11 = data['matches11'].to(self.device)
        self.matches_gray12 = data['matches12'].to(self.device)
        self.matches_gray22 = data['matches22'].to(self.device)

    def forward_gen(self, compute_loss=True):
        """Evaluate the forward pass of the generator model."""
        if self.opts.model == 'logrgb-enc' or self.opts.model == 'learned-enc':
            self.params = self.enc.forward(self.rgb1, self.rgb2)
        elif self.opts.model == 'logrgb-noenc':
            self.params = self.enc.forward(self.rgb1.shape[0])
        else:
            self.params = None

        if self.params is not None:
            self.params = nn.functional.normalize(self.params, p=1, dim=1)

        if self.opts.model == 'logrgb-noenc' or self.opts.model == 'logrgb-enc':
            logrgb12 = torch.cat([self.logrgb1, self.logrgb2], dim=2)
            out12 = self._make_sumlog_image(logrgb12, self.params)
            self.out1, self.out2 = torch.split(
                out12, self.logrgb1.shape[2], dim=2)
        elif self.opts.model == 'learned-noenc' or self.opts.model == 'learned-enc':
            rgb12 = torch.cat([self.rgb1, self.rgb2], dim=2)
            out12 = self._make_gen_image(rgb12, self.params)
            self.out1, self.out2 = torch.split(
                out12, self.rgb1.shape[2], dim=2)

        if compute_loss:
            self.matcher.forward(self.out1, self.out2, compute_loss=False)
            self.matches_out12_est = self.matcher.matches_est

            self.gen_match_loss = -self.matches_out12_est.mean()
            self.gen_match_loss *= self.opts.lambda_gen_match

            self.gen_loss = self.gen_match_loss

            self.matches_out11, self.matches_out12, self.matches_out22 = self._get_match_counts(
                self.out1.detach(), self.out2.detach())

    def _make_sumlog_image(self, logrgb, params):
        params = params.expand([-1, -1, logrgb.shape[2], logrgb.shape[3]])
        out = logrgb.mul(params).sum(dim=1, keepdim=True)

        post = nn.InstanceNorm2d(1)
        out = post(out).div(3).clamp(-1, 1)  # saturate outside 3 sigma

        return out

    def _make_gen_image(self, rgb, params=None):
        if params is not None:
            params = params.expand([-1, -1, rgb.shape[2], rgb.shape[3]])
            x = torch.cat([rgb, params], dim=1)
        else:
            x = rgb
        out = self.gen.forward(x)

        post = nn.InstanceNorm2d(1)
        out = post(out).div(3).clamp(-1, 1)  # saturate outside 3 sigma

        return out

    def set_matcher_data(self):
        matches11 = torch.cat([self.matches_gray11, self.matches_out11], dim=0)
        matches12 = torch.cat([self.matches_gray12, self.matches_out12], dim=0)
        matches22 = torch.cat([self.matches_gray22, self.matches_out22], dim=0)

        gray1 = torch.cat([self.gray1, self.out1.detach()], dim=0)
        gray2 = torch.cat([self.gray2, self.out2.detach()], dim=0)

        data = {'gray1': gray1, 'gray2': gray2,
                'matches11': matches11, 'matches12': matches12, 'matches22': matches22}

        self.matcher.set_data(data)

    def _get_matches(self, image1, image2):
        self.vo.process_frame(image1, image2)  # use ransac
        matches = self.vo.getNumberOfInliers()
        return matches

    def _get_match_counts(self, batch1, batch2):
        matches11 = torch.FloatTensor(batch1.shape[0])
        matches12 = torch.FloatTensor(batch1.shape[0])
        matches22 = torch.FloatTensor(batch1.shape[0])

        to_image = transforms.Compose(
            [custom_transforms.UnNormalize(self.opts.image_mean,
                                           self.opts.image_std),
             custom_transforms.Clamp(0, 1),
             transforms.ToPILImage()])

        for idx, (image1, image2) in enumerate(zip(batch1.cpu(), batch2.cpu())):
            image1 = np.array(to_image(image1))
            image2 = np.array(to_image(image2))

            matches11[idx] = self._get_matches(image1, image1)
            matches12[idx] = self._get_matches(image1, image2)
            matches22[idx] = self._get_matches(image2, image2)

        matches11 = matches11.to(self.device)
        matches12 = matches12.to(self.device)
        matches22 = matches22.to(self.device)

        return matches11, matches12, matches22

    def optimize(self):
        """Do one step of training with the current input tensors"""
        # Update the generator using the current matcher
        self.matcher.set_mode('eval')
        self.forward_gen(compute_loss=True)
        if self.opts.model == 'logrgb-noenc' or self.opts.model == 'logrgb-enc':
            self.enc_opt.zero_grad()
            self.gen_loss.backward()
            self.enc_opt.step()
        elif self.opts.model == 'learned-enc':
            self.enc_opt.zero_grad()
            self.gen_opt.zero_grad()
            self.gen_loss.backward()
            self.enc_opt.step()
            self.gen_opt.step()
        elif self.opts.model == 'learned-noenc':
            self.gen_opt.zero_grad()
            self.gen_loss.backward()
            self.gen_opt.step()

        # Update the matcher using the current generator outputs
        self.matcher.set_mode('train')
        self.set_matcher_data()
        self.matcher.optimize()

    def test(self, compute_loss=True):
        """Evaluate the model and test loss without optimizing"""
        with torch.no_grad():
            self.forward_gen(compute_loss=compute_loss)
            if compute_loss:
                self.set_matcher_data()
                self.matcher.forward(self.matcher.gray112, self.matcher.gray122,
                                     compute_loss=compute_loss)

    def get_images(self):
        """Return a dictionary of the current source/static/target images"""
        image_dict = {
            'rgb1': utils.image_from_tensor(self.rgb1[0].detach(),
                                            self.opts.image_mean,
                                            self.opts.image_std),
            'rgb2': utils.image_from_tensor(self.rgb2[0].detach(),
                                            self.opts.image_mean,
                                            self.opts.image_std),
            'out1': utils.image_from_tensor(self.out1[0].detach(),
                                            self.opts.image_mean,
                                            self.opts.image_std),
            'out2': utils.image_from_tensor(self.out2[0].detach(),
                                            self.opts.image_mean,
                                            self.opts.image_std)
        }

        return image_dict

    def get_errors(self):
        """Return a dictionary of the current errors"""
        error_dict = {
            'matcher/loss': self.matcher.loss.item(),
            'gen/match_loss': self.gen_match_loss.item(),
            'gen/eval_loss': -self.matches_out12.mean().item()
        }

        return error_dict

    def save_checkpoint(self, epoch, label):
        """Save the model to file"""
        model_dir = os.path.join(
            self.opts.results_dir, self.opts.experiment_name, 'checkpoints')
        os.makedirs(model_dir, exist_ok=True)
        model_file = os.path.join(model_dir, '{}_net.pth.tar'.format(label))

        model_dict = {'epoch': epoch,
                      'label': label,
                      'matcher_state_dict': self.matcher.net.state_dict(),
                      'enc_state_dict': self.enc.state_dict(),
                      'gen_state_dict': self.gen.state_dict()}

        print("Saving model to {}".format(model_file))
        torch.save(model_dict, model_file)

    def load_checkpoint(self, label):
        """Load a model from file"""
        model_dir = os.path.join(
            self.opts.results_dir, self.opts.experiment_name, 'checkpoints')
        model_file = os.path.join(model_dir, '{}_net.pth.tar'.format(label))

        print("Loading model from {}".format(model_file))
        model_dict = torch.load(model_file, map_location=self.opts.device)
        self.enc.to(self.device)
        self.gen.to(self.device)

        self.matcher.net.load_state_dict(model_dict['matcher_state_dict'])
        try:
            self.enc.load_state_dict(model_dict['enc_state_dict'])
        except RuntimeError as e:
            print(e)
        try:
            self.gen.load_state_dict(model_dict['gen_state_dict'])
        except RuntimeError as e:
            print(e)

        return model_dict['epoch']
