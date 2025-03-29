# -*- coding: utf-8 -*-
# pylint: disable=E1101,W0221,R0901,R0902,R0913,R0914,E1136,C0301,C0411,C0412
"""
Wrap Encoding
=============
*Created on 29 by bari_is*
*Copyright (C) 2025*
*For COPYING and LICENSE details, please refer to the LICENSE file*

This module encodes geographical coordinates (longitude and latitude)
into a 4-dimensional representation using trigonometric functions.
The encoding ensures that the coordinates are represented in a
continuous and periodic manner, suitable for neural network inputs.
"""


import torch
from torch import nn


class Wrap(nn.Module):
    """
    Wrap encoding module, as used by Mac Aodha et al.
    This module encodes geographical coordinates (longitude and latitude)
    into a 4-dimensional representation using trigonometric functions.
    The encoding ensures that the coordinates are represented in a
    continuous and periodic manner, suitable for neural network inputs.

    Attributes
    ----------
    embedding_dim : int
        The dimensionality of the encoded representation, fixed at 4.

    Methods
    -------
    forward(coords)
        Encodes the input longitude and latitude coordinates into a
        4-dimensional representation.

    """

    def __init__(self):
        super(Wrap, self).__init__()

        # adding this class variable is important to determine
        # the dimension of the follow-up neural network
        self.embedding_dim = 4

    def forward(self, coords):
        # place lon lat coordinates in a -pi, pi range
        coords = torch.deg2rad(coords)

        cos_lon = torch.cos(coords[:, 0]).unsqueeze(-1)
        sin_lon = torch.sin(coords[:, 0]).unsqueeze(-1)
        cos_lat = torch.cos(coords[:, 1]).unsqueeze(-1)
        sin_lat = torch.sin(coords[:, 1]).unsqueeze(-1)

        return torch.cat((cos_lon, sin_lon, cos_lat, sin_lat), 1)
