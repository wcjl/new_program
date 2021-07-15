import argparse
from datetime import datetime
import os
import shutil
import cv2

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
from tensorboardX import SummaryWriter

import models
from models.readout_nets.center_bias_layer import CenterBiasLayer
from utils import np_transforms
from utils.progress_bar import progress_bar
from utils.dataset import SaliencyDataset, get_dataset_config
from utils.loss import KLD, CC, SIM, NSS, KLD_adjust_gtmap_size

#torch.manual_seed(123)
torch.cuda.manual_seed(123)

model_names = sorted(name for name in models.__dict__
    if not name.startswith("_") and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='Train the model that predict saliency-map in PyTorch')
parser.add_argument('--dataset-name', default='salicon',
                    choices=['salicon', 'cat2000', 'mit1003', 'osie', 'osie2', 'osie_antonio'],
                    help='name of dataset (default: salicon)')

parser.add_argument('--root-dataset-dir', default='/home/wuchenjunlin/new_program/datasets', type=str,
                    help='root dataset directory')

parser.add_argument('--arch', '-a', metavar='ARCH', default='densesal_v2',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: densesal_v2)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')


parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 1)')


parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--center-bias-lr', default=1e-4, type=float,
                    metavar='LR of center bias layer', help='initial learning rate')
parser.add_argument('--alpha', default=0.995, type=float, metavar='M',
                    help='alpha of RMSProp')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 1)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--use-center-bias-layer', dest='use_center_bias_layer', action='store_true',
                    help='using center bias layer')
parser.add_argument('--use-fixed-center-bias', dest='use_fixed_center_bias', action='store_true',
                    help='using fixed center bias layer')
parser.add_argument('--fix-main-net', dest='fix_main_net', action='store_true',
                    help='fix model parameters of main net')
parser.add_argument('--fix-model-param', dest='fix_model_param', action='store_true',
                    help='fix model parameters other than center bias layer')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--from-scratch', dest='from_scratch', action='store_true',
                    help='training from scratch')
parser.add_argument('--root-log-dir', default='logs', type=str,
                    help='output directory')
parser.add_argument('--sub-log-dir', default='experiment1', type=str,
                    help='output directory under root-log-dir')

n_iter = 0

def main():
    best_val_loss = 1e5
    args = parser.parse_args()

    """
    dt = datetime.now().strftime('%B%d %H:%M:%S')
    train_log_dir = os.path.join(args.root_log_dir, args.sub_log_dir, 'train ' + dt)
    train_writer = SummaryWriter(train_log_dir) 
    val_log_dir = os.path.join(args.root_log_dir, args.sub_log_dir, 'val ' + dt)
    val_writer = SummaryWriter(val_log_dir) 
    """
    if args.use_center_bias_layer:
        train_log_dir = os.path.join(args.root_log_dir, args.sub_log_dir, 'train ' + args.arch + ' with CBL')
        train_writer = SummaryWriter(train_log_dir) 
        val_log_dir = os.path.join(args.root_log_dir, args.sub_log_dir, 'val ' + args.arch + ' with CBL')
        val_writer = SummaryWriter(val_log_dir) 
    if args.fix_main_net:
        train_log_dir = os.path.join(args.root_log_dir, args.sub_log_dir, 'train ' + args.arch + ' fixed main net')
        train_writer = SummaryWriter(train_log_dir) 
        val_log_dir = os.path.join(args.root_log_dir, args.sub_log_dir, 'val ' + args.arch + ' fixed main net')
        val_writer = SummaryWriter(val_log_dir) 
    else:
        train_log_dir = os.path.join(args.root_log_dir, args.sub_log_dir, 'train ' + args.arch)
        train_writer = SummaryWriter(train_log_dir) 
        val_log_dir = os.path.join(args.root_log_dir, args.sub_log_dir, 'val ' + args.arch)
        val_writer = SummaryWriter(val_log_dir) 

    # write parameters
    train_writer.add_text('Model', args.arch)
    train_writer.add_text('Pretrained on ImageNet1k', '{}'.format(not args.from_scratch))
    train_writer.add_text('Center bias layer', str(args.use_center_bias_layer))
    train_writer.add_text('Fixed center bias', str(args.use_fixed_center_bias))
    train_writer.add_text('Loss function', 'KL-Divergence')
    train_writer.add_text('Optimizer', 'algorithm:RMSprop, initial learning rate:{}, alpha:{}, weight_decay:{}'.format(args.lr, args.alpha, args.weight_decay))
    train_writer.add_text('Dataset', args.dataset_name)
    train_writer.add_text('Batch size', str(args.batch_size))
    train_writer.add_text('Resume', args.resume)

    # create model
    if args.from_scratch:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=False)
    else:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)

    model = torch.nn.DataParallel(model).cuda()
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    

    if args.use_center_bias_layer:
        print('model_lr', args.lr)
        print('center_bias_lr', args.center_bias_lr)
        center_bias_layer = CenterBiasLayer().cuda()
        if args.fix_model_param:
            parameters = [
                {'params': center_bias_layer.parameters(), 'lr': args.center_bias_lr}
            ]
        elif args.fix_main_net:
            parameters = [
                {'params': model.module.readout_net.parameters()},
                {'params': center_bias_layer.parameters(), 'lr': args.center_bias_lr}
            ]
        else:
            parameters = [
                {'params': model.parameters()},
                {'params': center_bias_layer.parameters(), 'lr': args.center_bias_lr}
            ]
    else:
        print('model_lr', args.lr)
        center_bias_layer = None
        if args.fix_main_net:
            parameters = model.module.readout_net.parameters()
        else:
            parameters = model.parameters()

    # define loss function (criterion) and optimizer
    criterion = KLD().cuda()
    metrics = {'CC': CC().cuda(),'SIM': SIM().cuda(),'KLD': KLD_adjust_gtmap_size().cuda(), 'NSS': NSS().cuda()}
    optimizer = torch.optim.RMSprop(parameters, args.lr,
                                alpha=args.alpha,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            if args.use_center_bias_layer and 'center_bias_state_dict' in checkpoint:
                center_bias_layer.load_state_dict(checkpoint['center_bias_state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            raise OSError("No checkpoint found at '{}'".format(args.resume))

    print('Dataset:', args.dataset_name)
    train_loader = torch.utils.data.DataLoader(
        SaliencyDataset(
            args.dataset_name,
            args.root_dataset_dir,
            data_type='train',
            target_type=['location', 'distribution'],
            transform=np_transforms.ResizeToInnerRectangle(rec_long_side=640, rec_short_side=480),#salicon size
            #transform=np_transforms.Resize((640, 480)),
            post_simul_transform=np_transforms.ToTensor(),
        ),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None)
    
    val_loader = torch.utils.data.DataLoader(
        SaliencyDataset(
            args.dataset_name,
            args.root_dataset_dir,
            data_type='val',
            target_type=['location', 'distribution'],
            transform=np_transforms.ResizeToInnerRectangle(rec_long_side=640, rec_short_side=480),
            #transform=np_transforms.Resize((640, 480)),
            post_simul_transform=np_transforms.ToTensor(),
        ),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.use_fixed_center_bias:
        fixed_center_bias = torch.autograd.Variable(torch.FloatTensor(np.load(get_dataset_config(args.dataset_name).fixed_center_bias)).cuda())
        fixed_center_bias = torch.unsqueeze(torch.unsqueeze(fixed_center_bias, 0), 0)
    else:
        fixed_center_bias = None

    if args.evaluate:
        results = validate(args, val_loader, model, center_bias_layer, fixed_center_bias, criterion, metrics, writer=None, only_validate=True)
        for name, val in results:
            print('{}:{}'.format(name, val))
        return

    train_writer.add_scalar('Partial Epoch', 0, 0)
    #validate(args, val_loader, model, center_bias_layer, fixed_center_bias, criterion, metrics, val_writer)
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(args, train_loader, model, center_bias_layer, fixed_center_bias, criterion, metrics, optimizer, epoch, train_writer)

        # evaluate on validation set
        val_loss = validate(args, val_loader, model, center_bias_layer, fixed_center_bias, criterion, metrics, val_writer)

        # remember best prec@1 and save checkpoint
        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)
        save_dict = {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_val_loss': best_val_loss,
            'optimizer' : optimizer.state_dict(),
        }
        if args.use_center_bias_layer:
            save_dict['center_bias_state_dict'] = center_bias_layer.state_dict()
        save_checkpoint(
            save_dict,
            is_best,
            out_dir=train_log_dir)

    train_writer.close()    
    val_writer.close()    


def train(args, train_loader, model, center_bias_layer, fixed_center_bias, criterion, metrics, optimizer, epoch, writer):
    global n_iter
    meters = MeterStorage()
    meters.add('Loss')
    for name in metrics.keys():
        meters.add(name)

    # switch to train mode
    model.train()
    if args.use_center_bias_layer:
        center_bias_layer.train()

    #adjust_learning_rate(args, optimizer, epoch)
    for i, (input, gt_binary_map, gt_map) in enumerate(train_loader):
        n_iter += 1
        partial_epoch = epoch + ((i+1) * input.size(0)) / len(train_loader)
        
        gt_binary_map = gt_binary_map.cuda(non_blocking=True)
        gt_map = gt_map.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input)
        gt_map_var = torch.autograd.Variable(gt_map)
        gt_binary_map_var = torch.autograd.Variable(gt_binary_map)

        # compute output
        if args.use_center_bias_layer:
            output = center_bias_layer(model(input_var))
        elif args.use_fixed_center_bias:
            fixed_center_bias = nn.functional.interpolate(fixed_center_bias, size=(output.size(2), output.size(3)), mode='bilinear')
            output = output * fixed_center_bias
        else:
            #print("input_var:",input_var.size())
            output = model(input_var)

        loss = criterion(output, gt_map_var)
        meters.update('Loss', loss.data, input.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for name, module in metrics.items():
            if name=='NSS':
                metric_value = module(output, gt_binary_map_var).mul(-1)
                meters.update(name, metric_value.data, input.size(0))
            else:
                metric_value = module(output, gt_map_var)
                if name!='KLD':
                    metric_value.mul_(-1)
                meters.update(name, metric_value.data, input.size(0))
            writer.add_scalar(name, getattr(meters, name).val, n_iter)

        writer.add_scalar('Loss', meters.Loss.val, n_iter)
        writer.add_scalar('Partial Epoch', partial_epoch, n_iter)
        if i % args.print_freq == 0:
            progress_bar(i, len(train_loader),
                'Phase: Training, Epoch: %d, Loss: %.3f, Average loss: %.3f (num of the image:%d)'
                % (epoch, meters.Loss.val, meters.Loss.avg, meters.Loss.count))


def validate(args, val_loader, model, center_bias_layer, fixed_center_bias, criterion, metrics, writer, only_validate=False):
    global n_iter
    meters = MeterStorage()
    meters.add('Loss')
    for name in metrics.keys():
        meters.add(name)

    # switch to evaluate mode
    model.eval()
    if args.use_center_bias_layer:
        center_bias_layer.eval()

    for i, (input, gt_binary_map, gt_map) in enumerate(val_loader):
        gt_binary_map = gt_binary_map.cuda(non_blocking=True)
        gt_map = gt_map.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input)
        gt_map_var = torch.autograd.Variable(gt_map)
        gt_binary_map_var = torch.autograd.Variable(gt_binary_map)

        # compute output
        if args.use_center_bias_layer:
            output = center_bias_layer(model(input_var))
        elif args.use_fixed_center_bias:
            fixed_center_bias =  nn.functional.interpolate(fixed_center_bias, size=(output.size(2), output.size(3)), mode='bilinear')
            output = output * fixed_center_bias
        else:
            output = model(input_var)

        if n_iter==0 and i==0:
            print('output size:', output.size())

        loss = criterion(output, gt_map_var)
        meters.update('Loss', loss.data, input.size(0))

        for name, module in metrics.items():
            if name=='NSS':
                metric_value = module(output, gt_binary_map_var).mul(-1)
                meters.update(name, metric_value.data, input.size(0))
            else:
                metric_value = module(output, gt_map_var)
                if name!='KLD':
                    metric_value.mul_(-1)
                meters.update(name, metric_value.data, input.size(0))

        if i % args.print_freq == 0:
            progress_bar(i, len(val_loader),
                'Phase: validation, Loss: %.3f, Average loss: %.3f (num of the image:%d)'
                % (meters.Loss.val, meters.Loss.avg, meters.Loss.count))


    if only_validate:
        return ((name, getattr(meters, name).avg) for name in metrics.keys())
    else:
        writer.add_scalar('Loss', meters.Loss.avg, n_iter)
        for name in metrics.keys():
            writer.add_scalar(name, getattr(meters, name).avg, n_iter)
        return meters.Loss.avg


def save_checkpoint(state, is_best, out_dir, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(out_dir, filename))
    if is_best:
        shutil.copyfile(os.path.join(out_dir, filename), os.path.join(out_dir, 'model_best.pth.tar'))


def adjust_learning_rate(args, optimizer, epoch, writer=None):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    
    if 'NI' in args.arch:
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr * (0.1 ** (epoch // 1))
            print('lr =', param_group['lr'])
    """
    if args.dataset_name=='salicon':
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr * (0.1 ** (epoch // 1))
            if args.fix_main_net:
                param_group['lr'] = args.lr * (0.1 ** (epoch // 10))
            else:
                param_group['lr'] = args.lr * (0.1 ** (epoch // 10))
            if epoch % 10 == 0:
                print('lr =', param_group['lr'])

    if writer is not None:
        global n_iter
        writer.add_scalar('Learning Rate', lr, n_iter)
    """


class MeterStorage(object):
    def add(self, name):
        setattr(self, name, AverageMeter())

    def reset(self, name):
        getattr(self, name).reset()

    def update(self, name, val, n=1):
        getattr(self, name).update(val, n)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
