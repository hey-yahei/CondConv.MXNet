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

from mxnet.gluon.nn import HybridBlock, Activation, GlobalAvgPool2D, Dense, HybridSequential
from mxnet.base import numeric_types

__all__ = ['DefaultRouter', 'CondConv2D']
__author__ = 'YaHei'


class DefaultRouter(HybridBlock):
    def __init__(self, num_hidden):
        super(DefaultRouter, self).__init__()
        with self.name_scope():
            self.body = HybridSequential(prefix='')
            self.body.add(GlobalAvgPool2D())
            self.body.add(Dense(num_hidden, activation='sigmoid'))

    def hybrid_forward(self, F, x):
        return self.body(x)


class CondConv2D(HybridBlock):
    def __init__(self, channels, kernel_size, strides=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, layout='NCHW',
                 activation=None, use_bias=True, weight_initializer=None,
                 bias_initializer='zeros', in_channels=0,
                 router=None, num_experts=1, compute_mode='auto'):
        super(CondConv2D, self).__init__()
        assert dilation in (1, (1, 1)) and groups == 1

        with self.name_scope():
            if compute_mode == 'auto':
                self._combine_kernels = (num_experts > 4)
            else:
                self._combine_kernels = (compute_mode == 'combine_kernels')
            self._channels = channels
            self._in_channels = in_channels

            if isinstance(kernel_size, numeric_types):
                kernel_size = (kernel_size,) * 2
            if isinstance(strides, numeric_types):
                strides = (strides,) * 2
            if isinstance(padding, numeric_types):
                padding = (padding,) * 2
            if isinstance(dilation, numeric_types):
                dilation = (dilation,) * 2

            self._kwargs = {
                'kernel': kernel_size, 'stride': strides, 'dilate': dilation,
                'pad': padding, 'num_filter': channels, 'num_group': groups,
                'no_bias': not use_bias, 'layout': layout}

            self.weight = self.params.get('weight', shape=(num_experts, channels, in_channels, *kernel_size),
                                          init=weight_initializer,
                                          allow_deferred_init=True)
            if use_bias:
                self.bias = self.params.get('bias', shape=(num_experts, channels),
                                            init=bias_initializer,
                                            allow_deferred_init=True)
            else:
                self.bias = None

            if activation is not None:
                self.act = Activation(activation, prefix=activation + '_')
            else:
                self.act = None

            self.router = router(num_experts) or DefaultRouter(num_experts)

    def hybrid_forward(self, F, x, weight, bias=None):
        routing_weights = self.router(x)
        if self._combine_kernels:
            # x_split = x.split(axis=0, num_outputs=x.shape[0])
            # new_weight = (weight.expand_dims(0) * routing_weights.reshape(0, 0, 1, 1, 1, 1)).sum(axis=1)
            # if bias is not None:
            #     new_bias = (bias.expand_dims(0) * routing_weights.reshape(0, 0, 1)).sum(axis=1)
            #     act = F.concat(*[F.Convolution(x, w, b, name='fwd', **self._kwargs)
            #                      for x, w, b in zip(x_split, new_weight, new_bias)], dim=0)
            # else:
            #     act = F.concat(*[F.Convolution(x, w, name='fwd', **self._kwargs)
            #                      for x, w in zip(x_split, new_weight)], dim=0)
            assert x.shape[0] == 1
            new_weight = (weight * routing_weights.reshape(-1, 1, 1, 1, 1)).sum(axis=0)
            if bias is not None:
                new_bias = (bias * routing_weights.reshape(-1, 1)).sum(axis=0)
                act = F.Convolution(x, new_weight, new_bias, name='fwd', **self._kwargs)
            else:
                act = F.Convolution(x, new_weight, name='fwd', **self._kwargs)
        else:
            if bias is not None:
                act = sum([
                    routing_weights[:, i].reshape(0, 1, 1, 1) * F.Convolution(x, weight[i], bias[i], name='fwd', **self._kwargs)
                    for i in range(weight.shape[0])
                ])
            else:
                act = sum([
                    routing_weights[:, i].reshape(0, 1, 1, 1) * F.Convolution(x, weight[i], name='fwd', **self._kwargs)
                    for i in range(weight.shape[0])
                ])
        if self.act is not None:
            act = self.act(act)
        return act
