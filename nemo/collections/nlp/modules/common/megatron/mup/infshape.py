# coding=utf-8
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# MIT License
#
# Copyright (c) Microsoft Corporation.
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
# SOFTWARE

from copy import copy


class InfDim:
    '''A dimension with a base dimension, used for calculating μP scaling.

    An `InfDim` object is made up of 2 numbers: a dimension and a base
    dimension. If the base dimension is None, then this object represents a
    "finite", or "non-width" dimension. Otherwise, it represents an "infinite",
    or "width" dimension.
    '''

    def __init__(self, base_dim, dim):
        self.base_dim = base_dim
        self.dim = dim

    def isinf(self):
        return self.base_dim is not None

    def width_mult(self):
        '''Width multiplier used for calculating μP scaling.

        If finite, return 1.
        If infinite, return dim / base_dim.
        '''
        if self.isinf():
            return self.dim / self.base_dim
        return 1

    def __repr__(self):
        return f'InfDim({self.base_dim}, {self.dim})'

    def __str__(self):
        if self.isinf():
            return repr(self)
        return f'FinDim({self.dim})'

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, InfDim):
            return False
        return self.base_dim == other.base_dim and self.dim == other.dim


class InfShape(tuple):
    '''A tuple of `InfDim`s.

    This is intended to be attached to each parameter tensor `p` as `p.infshape`.
    '''

    def __init__(self, *args, **kwargs):
        tuple.__init__(*args, **kwargs)
        for dim in self:
            if not isinstance(dim, InfDim):
                raise ValueError('Elements of InfShape needs to be of class InfDim')
        # set main to be the last dimension that is infinite
        # for inf x inf this is fanin
        # for inf x fin or fin x inf it's the unique inf dim
        # user can set this manually if necessary
        self.main_idx = self.main = None
        for i, dim in list(enumerate(self))[::-1]:
            if dim.isinf():
                self.main_idx = i
                self.main = dim
                break

    def fanin_fanout(self):
        assert len(self) >= 2, 'fanin, fanout undefined for 1-dimensional weights'
        return self[1], self[0]

    def fanin_fanout_mult_ratio(self):
        fanin, fanout = self.fanin_fanout()
        return fanin.width_mult() / fanout.width_mult()

    def ninf(self):
        return sum(1 for dim in self if dim.isinf())

    def width_mult(self):
        if self.main is not None:
            return self.main.width_mult()
        return 1

    def base_shape(self):
        return [d.base_dim for d in self]

    def shape(self):
        return [d.dim for d in self]

    def __repr__(self):
        r = tuple.__repr__(self)[1:-1]
        return f'InfShape([{r}])'

    def serialize(self):
        d = {'base_shape': [], 'shape': []}
        for infdim in self:
            d['shape'].append(infdim.dim)
            d['base_shape'].append(infdim.base_dim)
        return d

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, InfShape):
            return False
        return all(d == dd for d, dd in zip(self, other))

    @classmethod
    def deserialize(cls, d):
        infshape = []
        for base_dim, dim in zip(d['base_shape'], d['shape']):
            infshape.append(InfDim(base_dim, dim))
        return InfShape(infshape)

    @classmethod
    def from_base_shape(cls, bsh):
        return InfShape([InfDim(bd, None) for bd in bsh])


def zip_infshape(base_dims, dims, fin_if_same=True):
    infshape = []
    for bd, d in zip(base_dims, dims):
        if isinstance(bd, InfDim):
            # retain bd's base_dim but overwrite dim
            infdim = copy(bd)
            infdim.dim = d
            infshape.append(infdim)
        elif isinstance(bd, int):
            if bd == d and fin_if_same:
                infshape.append(InfDim(None, d))
            else:
                infshape.append(InfDim(bd, d))
        else:
            raise ValueError(f'unhandled base_dim type: {type(bd)}')
    return InfShape(infshape)


if __name__ == '__main__':
    infshape = InfShape([InfDim(None, 100), InfDim(128, 1024), InfDim(128, 128)])
    print(infshape)
    print(f'{infshape.ninf()} dims are inf')
    print(f'width_mult {infshape.width_mult()}')

    print(zip_infshape([64, 128, 1024], [32, 128, 2048]))
