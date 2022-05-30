from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
import torch.nn as nn
from torch_geometric.nn import DataParallel
from progress.bar import Bar
from utils.utils import AverageMeter

_loss_factory = {
    'l1': nn.L1Loss,
    'sl1': nn.SmoothL1Loss,
    'l2': nn.MSELoss,
}

class ModelWithLoss(torch.nn.Module):
    def __init__(self, model, loss, gpus):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.loss = loss
        self.gpus = gpus

    def forward(self, batch):
        outputs = self.model(batch)
        if len(self.gpus) > 1:
            y = torch.cat([data.y for data in batch])
        else:
            y = batch.y
        loss = self.loss(outputs, y)
        loss_stats = {'loss': loss}
        return outputs, loss, loss_stats


class DeepSATTrainer(object):
    def __init__(
            self, args, model, optimizer=None):
        self.args = args
        self.optimizer = optimizer
        self.loss_stats, self.loss = self._get_losses(args.reg_loss)
        self.model_with_loss = ModelWithLoss(model, self.loss, args.gpus)

    # def set_device(self, gpus, chunk_sizes, device):
    def set_device(self, device, gpus):
        if len(gpus) > 1:
          self.model_with_loss = DataParallel(
            self.model_with_loss, device_ids=gpus).to(device)
        else:
            self.model_with_loss = self.model_with_loss.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def run_epoch(self, phase, epoch, dataset):
        model_with_loss = self.model_with_loss
        if phase == 'train':
            model_with_loss.train()
        else:
            if len(self.args.gpus) > 1:
                model_with_loss = self.model_with_loss.module
            model_with_loss.eval()
            torch.cuda.empty_cache()

        args = self.args
        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(dataset) if args.num_iters < 0 else args.num_iters
        bar = Bar('{}/{}'.format(args.task, args.exp_id), max=num_iters)
        end = time.time()
        for iter_id, batch in enumerate(dataset):
            if iter_id >= num_iters:
                break
            if len(self.args.gpus) == 1:
                batch = batch.to(self.args.device)
            data_time.update(time.time() - end)
            output, loss, loss_stats = model_with_loss(batch)
            loss = loss.mean()
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model_with_loss.parameters(), args.grad_clip)
                self.optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()
            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                epoch, iter_id, num_iters, phase=phase,
                total=bar.elapsed_td, eta=bar.eta_td)
            for l in avg_loss_stats:
                avg_loss_stats[l].update(
                    loss_stats[l].mean().item(), batch.num_graphs * len(output))
                Bar.suffix = Bar.suffix + \
                    '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
            if not args.hide_data_time:
                Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                    '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
            if args.print_iter > 0:
                if iter_id % args.print_iter == 0:
                    print('{}/{}| {}'.format(args.task, args.exp_id, Bar.suffix))
            else:
                bar.next()
            # if args.debug > 0:
            #     self.debug(batch, output, iter_id)
            # if args.test:
            #     self.save_result(output, batch, results)
            del output, loss, loss_stats


        bar.finish()
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = bar.elapsed_td.total_seconds() / 60.
        return ret, results

    def debug(self, batch, output, iter_id):
        raise NotImplementedError

    def save_result(self, output, batch, results):
        raise NotImplementedError

    def _get_losses(self, loss):
        if loss in _loss_factory.keys():
            loss = _loss_factory[loss]()
        else:
            raise KeyError
        loss_states = ['loss']
        return loss_states, loss

    def val(self, epoch, data_loader):
        return self.run_epoch('val', epoch, data_loader)

    def train(self, epoch, data_loader):
        return self.run_epoch('train', epoch, data_loader)
