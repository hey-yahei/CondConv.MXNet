#-*- coding: utf-8 -*-
# MIT License
#
# Copyright (c) 2019 hey-yahei
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from .CondConv import CondConv2D

from mxnet import nd, init, cpu
from mxnet.lr_scheduler import LRScheduler
from math import cos, pi

__all__ = ['HalfCosineScheduler', 'se_initialize', 'se_initialize_v2']
__author__ = 'YaHei'


class HalfCosineScheduler(LRScheduler):
    def __init__(self, cycle, max_steps=None, base_lr=0.01, final_lr=0,
                 warmup_steps=0, warmup_begin_lr=0, warmup_mode='linear'):
        super(HalfCosineScheduler, self).__init__(base_lr, warmup_steps, warmup_begin_lr, warmup_mode)
        self.base_lr_orig = base_lr
        self.cycle = cycle
        self.final_lr = final_lr
        self.max_steps = max_steps

    def __call__(self, num_update):
        if num_update < self.warmup_steps:
            return self.get_warmup_lr(num_update)
        if self.max_steps is None or num_update <= self.max_steps:
            self.base_lr = self.final_lr + (self.base_lr_orig - self.final_lr) * \
                           (1 + cos(pi * ((num_update - self.warmup_steps) % self.cycle) / self.cycle)) / 2
        return self.base_lr


def _get_param_by_name(net, param_name):
    cur = net
    for name in param_name.split('.'):
        if name.isdigit():
            cur = cur[int(name)]
        else:
            cur = getattr(cur, name)
    return cur


def se_initialize(net, param_files, ctx=cpu()):
    finished_params = []
    """ Load parameters """
    origin_params = []
    for fname in param_files:
        origin_params.append(nd.load(fname))

    """ Initialize parameters with snapshot-ensembled models """
    for pname in origin_params[0].keys():
        origins = nd.stack(*[pdict[pname] for pdict in origin_params])
        dest = _get_param_by_name(net, pname)
        if len(dest.shape) == len(origins.shape):
            # print("Stack params:", dest.name)
            dest.initialize(init.Constant(origins), ctx=ctx)
        elif len(dest.shape) == len(origins.shape) - 1:
            # print("Reduced params:", dest.name)
            dest.initialize(init.Constant(origins.mean(axis=0)), ctx=ctx)
        else:
            raise ValueError()
        finished_params.append(dest.name)

    """ Initialize other parameters """
    for p in net.collect_params().values():
        if p.name not in finished_params:
            # print("Random params:", p.name)
            p.initialize()


def se_initialize_v2(net, param_files, conv_name='condconv2d', bn_name='batchnorm', ctx=cpu()):
    finished_params = []
    """ Load parameters """
    origin_params = []
    for fname in param_files:
        origin_params.append(nd.load(fname))

    """ Initialize parameters with snapshot-ensembled models """
    bn_collections = {}
    for pname in origin_params[0].keys():
        origins = nd.stack(*[pdict[pname] for pdict in origin_params])
        dest = _get_param_by_name(net, pname)
        if len(dest.shape) == len(origins.shape):
            print("Stack params:", dest.name)
            dest.initialize()
            dest.set_data(origins)
            dest._finish_deferred_init()
            finished_params.append(dest.name)
        elif len(dest.shape) == len(origins.shape) - 1 and bn_name in dest.name:
            print("BatchNorm params:", dest.name)
            bn_collections[dest.name] = origins
        else:
            print("Ignore params:", dest.name)

    """ Merge BatchNorm into Convolution """
    def _merge_bn_to_condconv2d(m):
        if isinstance(m, CondConv2D):
            base_name = m.name.replace(conv_name, bn_name)
            print(f"Merge {base_name} to {m.name}")
            gamma = bn_collections[base_name + "_gamma"]
            beta = bn_collections[base_name + "_beta"]
            mean = bn_collections[base_name + "_running_mean"]
            var = bn_collections[base_name + "_running_var"]

            weight = m.weight.data()
            w_shape = m.weight.shape
            m.weight.set_data((weight.reshape(0, 0, -1) * gamma.reshape(0, 0, 1) \
                                  / nd.sqrt(var + 1e-10).reshape(0, 0, 1)).reshape(w_shape))
            if m.bias is None:
                m._kwargs['no_bias'] = False
                m.bias = m.params.get('bias',
                                      shape=w_shape[:2], init="zeros",
                                      allow_deferred_init=True)
                m.bias.initialize()
                finished_params.append(m.bias.name)
            bias = m.bias.data()
            m.bias.set_data(gamma * (bias - mean) / nd.sqrt(var + 1e-10) + beta)
    _ = net.apply(_merge_bn_to_condconv2d)

    """ Initialize other parameters """
    random_params = {}
    for p in net.collect_params().values():
        if p.name not in finished_params:
            print("Random params:", p.name)
            p.initialize()
            random_params[p.name] = p
    return random_params
