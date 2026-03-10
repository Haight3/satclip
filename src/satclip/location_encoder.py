# -*- coding: utf-8 -*-
# pylint: disable=E1101,W0221,R0901,R0902,R0913,R0914,E1136,C0301,C0411,C0412,E1102
"""
Location Encoding Components
============================
*Created on 25 by bari_is*
*Copyright (C) 2025*
*For COPYING and LICENSE details, please refer to the LICENSE file*

This module defines positional-encoding driven neural components used to
encode geographic coordinates. It includes selectable encoding backends,
network factories, and wrapper modules used by SatCLIP location models.
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

    def __init__(self, linear_size: int) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the residual block to an input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input features with shape ``(..., linear_size)``.

        Returns
        -------
        torch.Tensor
            Features after two linear layers, non-linearities, and residual
            addition.
        """
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

    def __init__(self, num_inputs: int, num_classes: int, dim_hidden: int) -> None:
        """
        Initialize the fully connected location network.

        Parameters
        ----------
        num_inputs : int
            Dimension of encoded input features.
        num_classes : int
            Output feature dimension.
        dim_hidden : int
            Hidden feature dimension for internal residual blocks.
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass through the feature extractor and output head.

        Parameters
        ----------
        x : torch.Tensor
            Input encoded coordinates.

        Returns
        -------
        torch.Tensor
            Output logits or embeddings with shape ``(..., num_classes)``.
        """
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

    def __init__(
        self,
        input_dim: int,
        dim_hidden: int,
        num_layers: int,
        out_dims: int,
    ) -> None:
        """
        Initialize the multilayer perceptron.

        Parameters
        ----------
        input_dim : int
            Number of input features.
        dim_hidden : int
            Width of hidden layers.
        num_layers : int
            Number of repeated hidden blocks.
        out_dims : int
            Number of output features.
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute output features for an input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input features.

        Returns
        -------
        torch.Tensor
            Network outputs.
        """
        return self.features(x)


def exists(val: object) -> bool:
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


def cast_tuple(val: object, repeat: int = 1) -> tuple:
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
        dim_in: int,
        dim_hidden: int,
        dim_out: int,
        num_layers: int,
        w0: float = 1.0,
        w0_initial: float = 30.0,
        use_bias: bool = True,
        final_activation: nn.Module | None = None,
        degreeinput: bool = False,
        dropout: bool = True,
    ) -> None:
        """
        Initialize a SIREN stack with configurable first-layer frequency.

        Parameters
        ----------
        dim_in : int
            Input feature dimension.
        dim_hidden : int
            Hidden layer width.
        dim_out : int
            Output feature dimension.
        num_layers : int
            Number of hidden SIREN layers.
        w0 : float, optional
            Frequency factor for non-first layers.
        w0_initial : float, optional
            Frequency factor for the first layer.
        use_bias : bool, optional
            Whether to use bias terms in linear transforms.
        final_activation : nn.Module or None, optional
            Activation applied after the final linear transform.
        degreeinput : bool, optional
            Whether to convert degree inputs to radians in ``[-pi, pi]``.
        dropout : bool, optional
            Whether to enable dropout in hidden SIREN layers.
        """
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

    def forward(self, x: torch.Tensor, mods=None) -> torch.Tensor:
        """
        Run the full SIREN forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input coordinates or encoded features.
        mods : torch.Tensor or tuple or None, optional
            Optional per-layer modulation vectors broadcast onto hidden states.

        Returns
        -------
        torch.Tensor
            Final network outputs.
        """

        # do some normalization to bring degrees in a -pi to pi range
        if self.degreeinput:
            x = torch.deg2rad(x) - torch.pi

        mods = cast_tuple(mods, self.num_layers)

        for layer, mod in zip(self.layers, mods):
            x = layer(x)

            if exists(mod):
                x *= rearrange(mod, "d -> () d")

        return self.last_layer(x)

    def forward_features(self, x: torch.Tensor, mods=None) -> torch.Tensor:
        """
        Compute hidden features before the final output layer.

        Parameters
        ----------
        x : torch.Tensor
            Input coordinates or encoded features.
        mods : torch.Tensor or tuple or None, optional
            Optional per-layer modulation vectors.

        Returns
        -------
        torch.Tensor
            Hidden representation produced by the last hidden SIREN block.
        """
        # do some normalization to bring degrees in a -pi to pi range
        if self.degreeinput:
            x = torch.deg2rad(x) - torch.pi

        mods = cast_tuple(mods, self.num_layers)

        for layer, mod in zip(self.layers, mods):
            x = layer(x)

            if exists(mod):
                x *= rearrange(mod, "d -> () d")

        return x


class Sine(nn.Module):
    """
    Sine activation module with configurable frequency multiplier.

    Parameters
    ----------
    w0 : float, optional
        Frequency multiplier applied before ``sin``.
    """

    def __init__(self, w0: float = 1.0) -> None:
        """
        Initialize the sine activation.

        Parameters
        ----------
        w0 : float, optional
            Frequency multiplier applied to the input tensor.
        """
        super().__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply sine activation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            ``sin(w0 * x)``.
        """
        return torch.sin(self.w0 * x)


class Siren(nn.Module):
    """
    Single SIREN layer with custom initialization and activation.

    Parameters
    ----------
    dim_in : int
        Input feature dimension.
    dim_out : int
        Output feature dimension.
    w0 : float, optional
        Frequency multiplier for the default sine activation.
    c : float, optional
        Constant used by SIREN initialization.
    is_first : bool, optional
        Whether this layer is the first layer in the network.
    use_bias : bool, optional
        Whether to include a learnable bias.
    activation : nn.Module or None, optional
        Activation module; defaults to :class:`Sine`.
    dropout : bool, optional
        Whether to apply dropout to the linear output.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        w0: float = 1.0,
        c: float = 6.0,
        is_first: bool = False,
        use_bias: bool = True,
        activation: nn.Module | None = None,
        dropout: bool = False,
    ) -> None:
        """
        Initialize the SIREN layer.

        Parameters
        ----------
        dim_in : int
            Input feature dimension.
        dim_out : int
            Output feature dimension.
        w0 : float, optional
            Frequency multiplier for the default sine activation.
        c : float, optional
            Initialization constant from the SIREN formulation.
        is_first : bool, optional
            Whether this is the first layer.
        use_bias : bool, optional
            Whether to allocate a learnable bias.
        activation : nn.Module or None, optional
            Activation module to apply after the linear transform.
        dropout : bool, optional
            Whether to apply dropout during training.
        """
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

    def init_(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        c: float,
        w0: float,
    ) -> None:
        """
        Initialize layer parameters with SIREN-compatible bounds.

        Parameters
        ----------
        weight : torch.Tensor
            Weight tensor to initialize in-place.
        bias : torch.Tensor or None
            Optional bias tensor to initialize in-place.
        c : float
            Initialization constant.
        w0 : float
            Frequency scaling factor.
        """
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if exists(bias):
            bias.uniform_(-w_std, w_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply linear projection, optional dropout, and activation.

        Parameters
        ----------
        x : torch.Tensor
            Input features.

        Returns
        -------
        torch.Tensor
            Activated outputs.
        """
        out = F.linear(x, self.weight, self.bias)
        if self.dropout:
            out = F.dropout(out, training=self.training)
        out = self.activation(out)
        return out


class Modulator(nn.Module):
    """
    Layer-wise modulator that conditions SIREN hidden states on a latent vector.

    Parameters
    ----------
    dim_in : int
        Latent vector dimension.
    dim_hidden : int
        Hidden modulation dimension per layer.
    num_layers : int
        Number of modulation layers.
    """

    def __init__(self, dim_in: int, dim_hidden: int, num_layers: int) -> None:
        """
        Initialize the latent modulator network.

        Parameters
        ----------
        dim_in : int
            Latent vector dimension.
        dim_hidden : int
            Hidden modulation dimension.
        num_layers : int
            Number of generated modulation tensors.
        """
        super().__init__()
        self.layers = nn.ModuleList([])

        for ind in range(num_layers):
            is_first = ind == 0
            dim = dim_in if is_first else (dim_hidden + dim_in)

            self.layers.append(nn.Sequential(nn.Linear(dim, dim_hidden), nn.ReLU()))

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Produce per-layer modulation features.

        Parameters
        ----------
        z : torch.Tensor
            Latent conditioning vector.

        Returns
        -------
        tuple[torch.Tensor, ...]
            One modulation tensor per layer.
        """
        x = z
        hiddens = []

        for layer in self.layers:
            x = layer(x)
            hiddens.append(x)
            x = torch.cat((x, z))

        return tuple(hiddens)


class SirenWrapper(nn.Module):
    """
    Image-grid wrapper around :class:`SirenNet` for coordinate-based rendering.

    Parameters
    ----------
    net : SirenNet
        SIREN network evaluated on generated image coordinates.
    image_width : int
        Output image width.
    image_height : int
        Output image height.
    latent_dim : int or None, optional
        Optional latent dimension for modulation.
    """

    def __init__(
        self,
        net: SirenNet,
        image_width: int,
        image_height: int,
        latent_dim: int | None = None,
    ) -> None:
        """
        Initialize the wrapper and precompute the coordinate grid.

        Parameters
        ----------
        net : SirenNet
            SIREN network to evaluate over the image grid.
        image_width : int
            Width of the output image.
        image_height : int
            Height of the output image.
        latent_dim : int or None, optional
            Latent dimension used for modulation. If omitted, modulation is
            disabled.
        """
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

    def forward(
        self,
        img: torch.Tensor | None = None,
        *,
        latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Evaluate the wrapped SIREN on the coordinate grid.

        Parameters
        ----------
        img : torch.Tensor or None, optional
            Target image tensor. When provided, returns MSE loss against model
            output.
        latent : torch.Tensor or None, optional
            Latent vector used when modulation is enabled.

        Returns
        -------
        torch.Tensor
            Rendered image tensor or scalar MSE loss when ``img`` is provided.

        Raises
        ------
        AssertionError
            If latent input is inconsistently provided with modulation setup.
        """
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
    name: str,
    legendre_polys: int = 10,
    harmonics_calculation: str = "analytic",
    min_radius: int = 1,
    max_radius: int = 360,
    frequency_num: int = 10,
) -> nn.Module:
    """
    Create a positional encoding module by name.

    Parameters
    ----------
    name : str
        Encoding identifier.
    legendre_polys : int, optional
        Number of Legendre polynomials for spherical harmonics variants.
    harmonics_calculation : str, optional
        Strategy used by spherical harmonics encoders.
    min_radius : int, optional
        Minimum radius for radial encodings.
    max_radius : int, optional
        Maximum radius for radial encodings.
    frequency_num : int, optional
        Number of frequencies for radial or grid-based encodings.

    Returns
    -------
    nn.Module
        Instantiated positional encoding module.

    Raises
    ------
    ValueError
        If ``name`` is not supported.
    """
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


def get_neural_network(
    name: str,
    input_dim: int,
    num_classes: int = 256,
    dim_hidden: int = 256,
    num_layers: int = 2,
) -> nn.Module:
    """
    Create a neural network backend by name.

    Parameters
    ----------
    name : str
        Network identifier.
    input_dim : int
        Input feature dimension.
    num_classes : int, optional
        Output feature dimension.
    dim_hidden : int, optional
        Hidden dimension for multi-layer models.
    num_layers : int, optional
        Number of hidden layers for configurable models.

    Returns
    -------
    nn.Module
        Instantiated neural network module.

    Raises
    ------
    ValueError
        If ``name`` is not supported.
    """
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

    def __init__(self, posenc: nn.Module, nnet: nn.Module) -> None:
        """
        Initialize the location encoder.

        Parameters
        ----------
        posenc : nn.Module
            Positional encoding module that maps raw coordinates to encoded
            features.
        nnet : nn.Module
            Downstream network operating on encoded features.
        """
        super().__init__()
        self.posenc = posenc  # a positional encoding function/module
        self.nnet = nnet  # a neural network that takes encoded inputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode coordinates and run prediction.

        Parameters
        ----------
        x : torch.Tensor
            Raw coordinate tensor, typically latitude/longitude pairs.

        Returns
        -------
        torch.Tensor
            Network output after positional encoding.
        """
        x = self.posenc(x)  # encode raw coordinates (lat/lon)
        return self.nnet(x)  # process the encoded features

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode coordinates and return intermediate model features.

        Parameters
        ----------
        x : torch.Tensor
            Raw coordinate tensor.

        Returns
        -------
        torch.Tensor
            Feature representation from ``self.nnet.forward_features``.
        """
        x = self.posenc(x)  # encode raw coordinates (lat/lon)
        return self.nnet.forward_features(x)  # process the encoded features

    def forward_decomp(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode coordinates and return decomposed outputs from the backend.

        Parameters
        ----------
        x : torch.Tensor
            Raw coordinate tensor.

        Returns
        -------
        torch.Tensor
            Decomposed output from ``self.nnet.forward_decomp``.
        """
        x = self.posenc(x)  # encode raw coordinates (lat/lon)
        return self.nnet.forward_decomp(x)  # process the encoded features
