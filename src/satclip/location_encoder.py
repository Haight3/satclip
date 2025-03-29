# -*- coding: utf-8 -*-
# pylint: disable=E1101,W0221,R0901,R0902,R0913,R0914,E1136,C0301,C0411,C0412,E1102
"""
Location Encoder
================
*Created on 25 by bari_is*
*Copyright (C) 2025*
*For COPYING and LICENSE details, please refer to the LICENSE file*

"""


import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from satclip import positional_encoding as PE


class ResLayer(nn.Module):
    """
    A residual layer module for neural networks.
    This module implements a residual connection with two linear layers,
    ReLU activations, and dropout. The input is passed through the layers
    and added to the original input to form the output.

    Parameters
    ----------
    linear_size : int
        The size of the input and output features for the linear layers.

    Attributes
    ----------
    l_size : int
        The size of the input and output features for the linear layers.
    nonlin1 : nn.ReLU
        The first ReLU activation function.
    nonlin2 : nn.ReLU
        The second ReLU activation function.
    dropout1 : nn.Dropout
        The dropout layer applied after the first activation.
    w1 : nn.Linear
        The first linear transformation layer.
    w2 : nn.Linear
        The second linear transformation layer.

    Methods
    -------
    forward(x)
        Defines the forward pass of the residual layer. Applies the
        transformations and adds the input to the output.
    """

    def __init__(self, linear_size):
        """
        Initialize the ResLayer class.

        Parameters
        ----------
        linear_size : int
            The size of the linear layer, which determines the input and output dimensions
            of the fully connected layers in the residual block.
        """

        super(ResLayer, self).__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout()
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.dropout1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        out = x + y

        return out


class FCNet(nn.Module):
    """
    A fully connected neural network (FCNet) for encoding input features and
    predicting class probabilities.

    Parameters
    ----------
    num_inputs : int
        The number of input features.
    num_classes : int
        The number of output classes.
    dim_hidden : int
        The dimensionality of the hidden layers.

    Attributes
    ----------
    inc_bias : bool
        Whether to include a bias term in the final classification layer.
        Defaults to False.
    class_emb : nn.Linear
        A linear layer mapping the hidden representation to class logits.
    feats : nn.Sequential
        A sequential model consisting of a linear layer, ReLU activation,
        and multiple residual layers for feature extraction.

    Methods
    -------
    forward(x)
        Forward pass through the network. Encodes the input features and
        predicts class logits.
    Notes

    -----
    This model uses residual layers (`ResLayer`) to enhance feature extraction
    in the hidden layers.
    """

    def __init__(self, num_inputs, num_classes, dim_hidden):
        super(FCNet, self).__init__()
        self.inc_bias = False
        self.class_emb = nn.Linear(dim_hidden, num_classes, bias=self.inc_bias)

        self.feats = nn.Sequential(
            nn.Linear(num_inputs, dim_hidden),
            nn.ReLU(inplace=True),
            ResLayer(dim_hidden),
            ResLayer(dim_hidden),
            ResLayer(dim_hidden),
            ResLayer(dim_hidden),
        )

    def forward(self, x):
        loc_emb = self.feats(x)
        class_pred = self.class_emb(loc_emb)
        return class_pred


class MLP(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) implementation using PyTorch.

    Parameters
    ----------
    input_dim : int
        The dimensionality of the input features.
    dim_hidden : int
        The number of units in each hidden layer.
    num_layers : int
        The number of hidden layers in the MLP.
    out_dims : int
        The dimensionality of the output features.

    Attributes
    ----------
    features : nn.Sequential
        A sequential container of the MLP layers, including input, hidden, and output layers.

    Methods
    -------
    forward(x)
        Performs a forward pass through the MLP.

    """

    def __init__(self, input_dim, dim_hidden, num_layers, out_dims):
        super(MLP, self).__init__()

        layers = []
        layers += [
            nn.Linear(input_dim, dim_hidden, bias=True),
            nn.ReLU(),
        ]  # input layer
        layers += [
            nn.Linear(dim_hidden, dim_hidden, bias=True),
            nn.ReLU(),
        ] * num_layers  # hidden layers
        layers += [nn.Linear(dim_hidden, out_dims, bias=True)]  # output layer

        self.features = nn.Sequential(*layers)

    def forward(self, x):
        return self.features(x)


def exists(val):
    """
    Check if a value is not None.

    Parameters
    ----------
    val : any
        The value to check.

    Returns
    -------
    bool
        True if the value is not None, False otherwise.
    """

    return val is not None


def cast_tuple(val, repeat=1):
    """
    Converts a value into a tuple. If the input value is not already a tuple,
    it repeats the value a specified number of times to create a tuple.

    Parameters
    ----------
    val : Any
        The value to be converted into a tuple.
    repeat : int, optional
        The number of times to repeat the value if it is not a tuple.
        Default is 1.

    Returns
    -------
    tuple
        A tuple containing the input value repeated `repeat` times,
        or the input value itself if it is already a tuple.
    """

    return val if isinstance(val, tuple) else ((val,) * repeat)


class SirenNet(nn.Module):
    """
    A neural network implementation of a SIREN (Sinusoidal Representation Network)
    with multiple layers, designed for tasks requiring high-frequency function fitting.

    Parameters
    ----------
    dim_in : int
        The dimensionality of the input features.
    dim_hidden : int
        The dimensionality of the hidden layers.
    dim_out : int
        The dimensionality of the output features.
    num_layers : int
        The number of layers in the network.
    w0 : float, optional
        The frequency scaling factor for layers other than the first. Default is 1.0.
    w0_initial : float, optional
        The frequency scaling factor for the first layer. Default is 30.0.
    use_bias : bool, optional
        Whether to include a bias term in the layers. Default is True.
    final_activation : callable or None, optional
        The activation function to apply in the final layer. If None, an identity function is used. Default is None.
    degreeinput : bool, optional
        If True, the input is assumed to be in degrees and will be normalized to the range [-π, π]. Default is False.
    dropout : bool, optional
        Whether to apply dropout in the layers. Default is True.

    Attributes
    ----------
    num_layers : int
        The number of layers in the network.
    dim_hidden : int
        The dimensionality of the hidden layers.
    degreeinput : bool
        Indicates whether the input is normalized from degrees to radians.
    layers : nn.ModuleList
        A list of SIREN layers in the network.
    last_layer : Siren
        The final SIREN layer with optional activation.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=1.0,
        w0_initial=30.0,
        use_bias=True,
        final_activation=None,
        degreeinput=False,
        dropout=True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.degreeinput = degreeinput

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            self.layers.append(
                Siren(
                    dim_in=layer_dim_in,
                    dim_out=dim_hidden,
                    w0=layer_w0,
                    use_bias=use_bias,
                    is_first=is_first,
                    dropout=dropout,
                )
            )

        final_activation = nn.Identity() if not exists(final_activation) else final_activation
        self.last_layer = Siren(
            dim_in=dim_hidden,
            dim_out=dim_out,
            w0=w0,
            use_bias=use_bias,
            activation=final_activation,
            dropout=False,
        )

    def forward(self, x, mods=None):

        # do some normalization to bring degrees in a -pi to pi range
        if self.degreeinput:
            x = torch.deg2rad(x) - torch.pi

        mods = cast_tuple(mods, self.num_layers)

        for layer, mod in zip(self.layers, mods):
            x = layer(x)

            if exists(mod):
                x *= rearrange(mod, "d -> () d")

        return self.last_layer(x)


class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class Siren(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        w0=1.0,
        c=6.0,
        is_first=False,
        use_bias=True,
        activation=None,
        dropout=False,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first
        self.dim_out = dim_out
        self.dropout = dropout

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c=c, w0=w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if exists(bias):
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        if self.dropout:
            out = F.dropout(out, training=self.training)
        out = self.activation(out)
        return out


class Modulator(nn.Module):
    def __init__(self, dim_in, dim_hidden, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([])

        for ind in range(num_layers):
            is_first = ind == 0
            dim = dim_in if is_first else (dim_hidden + dim_in)

            self.layers.append(nn.Sequential(nn.Linear(dim, dim_hidden), nn.ReLU()))

    def forward(self, z):
        x = z
        hiddens = []

        for layer in self.layers:
            x = layer(x)
            hiddens.append(x)
            x = torch.cat((x, z))

        return tuple(hiddens)


class SirenWrapper(nn.Module):
    def __init__(self, net, image_width, image_height, latent_dim=None):
        super().__init__()
        assert isinstance(net, SirenNet), "SirenWrapper must receive a Siren network"

        self.net = net
        self.image_width = image_width
        self.image_height = image_height

        self.modulator = None
        if exists(latent_dim):
            self.modulator = Modulator(dim_in=latent_dim, dim_hidden=net.dim_hidden, num_layers=net.num_layers)

        tensors = [
            torch.linspace(-1, 1, steps=image_height),
            torch.linspace(-1, 1, steps=image_width),
        ]
        mgrid = torch.stack(torch.meshgrid(*tensors, indexing="ij"), dim=-1)
        mgrid = rearrange(mgrid, "h w c -> (h w) c")
        self.register_buffer("grid", mgrid)

    def forward(self, img=None, *, latent=None):
        modulate = exists(self.modulator)
        assert not (modulate ^ exists(latent)), "latent vector must be only supplied if `latent_dim` was passed in on instantiation"

        mods = self.modulator(latent) if modulate else None

        coords = self.grid.clone().detach().requires_grad_()
        out = self.net(coords, mods)
        out = rearrange(out, "(h w) c -> () c h w", h=self.image_height, w=self.image_width)

        if exists(img):
            return F.mse_loss(img, out)

        return out


def get_positional_encoding(
    name,
    legendre_polys=10,
    harmonics_calculation="analytic",
    min_radius=1,
    max_radius=360,
    frequency_num=10,
):

    if name == "direct":
        return PE.Direct()
    elif name == "cartesian3d":
        return PE.Cartesian3D()
    elif name == "sphericalharmonics":
        if harmonics_calculation == "discretized":
            return PE.DiscretizedSphericalHarmonics(legendre_polys=legendre_polys)
        else:
            return PE.SphericalHarmonics(
                legendre_polys=legendre_polys,
                harmonics_calculation=harmonics_calculation,
            )
    elif name == "theory":
        return PE.Theory(min_radius=min_radius, max_radius=max_radius, frequency_num=frequency_num)
    elif name == "wrap":
        return PE.Wrap()
    elif name in ["grid", "spherec", "spherecplus", "spherem", "spheremplus"]:
        return PE.GridAndSphere(
            min_radius=min_radius,
            max_radius=max_radius,
            frequency_num=frequency_num,
            name=name,
        )
    else:
        raise ValueError(f"{name} not a known positional encoding.")


def get_neural_network(name, input_dim, num_classes=256, dim_hidden=256, num_layers=2):
    if name == "linear":
        return nn.Linear(input_dim, num_classes)
    elif name == "mlp":
        return MLP(
            input_dim=input_dim,
            dim_hidden=dim_hidden,
            num_layers=num_layers,
            out_dims=num_classes,
        )
    elif name == "siren":
        return SirenNet(
            dim_in=input_dim,
            dim_hidden=dim_hidden,
            num_layers=num_layers,
            dim_out=num_classes,
        )
    elif name == "fcnet":
        return FCNet(num_inputs=input_dim, num_classes=num_classes, dim_hidden=dim_hidden)
    else:
        raise ValueError(f"{name} not a known neural networks.")


class LocationEncoder(nn.Module):
    """
    A neural network module for encoding and processing location data.
    This module combines a positional encoding function/module with a neural
    network to process location-based inputs, such as latitude and longitude.

    Parameters
    ----------
    posenc : callable or nn.Module
        A positional encoding function or module that encodes raw coordinates
        (e.g., latitude and longitude) into a higher-dimensional representation.

    nnet : nn.Module
        A neural network module that processes the encoded features.

    Methods
    -------
    forward(x)
        Encodes the input coordinates using the positional encoding function
        and processes the encoded features using the neural network.

    Examples
    --------
    >>> posenc = SomePositionalEncodingModule()
    >>> nnet = SomeNeuralNetwork()
    >>> encoder = LocationEncoder(posenc, nnet)
    >>> raw_coordinates = torch.tensor([[37.7749, -122.4194]])  # Example: San Francisco
    >>> output = encoder(raw_coordinates)
    """

    def __init__(self, posenc, nnet):
        super().__init__()
        self.posenc = posenc  # a positional encoding function/module
        self.nnet = nnet  # a neural network that takes encoded inputs

    def forward(self, x):
        x = self.posenc(x)  # encode raw coordinates (lat/lon)
        return self.nnet(x)  # process the encoded features
