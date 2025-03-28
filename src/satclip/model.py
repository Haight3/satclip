# -*- coding: utf-8 -*-
# pylint: disable=E1101,W0221,R0901,R0902,R0913,R0914,E1136,C0301,C0411,C0412
"""
Model
=====
*Created on 26 by bari_is*
*Copyright (C) 2025*
*For COPYING and LICENSE details, please refer to the LICENSE file*

"""


from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch import nn
from torchgeo.models import ResNet18_Weights, ResNet50_Weights, ViTSmall16_Weights

from satclip.location_encoder import LocationEncoder, get_neural_network, get_positional_encoding


class Bottleneck(nn.Module):
    """
    A bottleneck block used in deep residual networks, such as ResNet. This block
    implements a sequence of convolutional layers with batch normalization and
    ReLU activations, along with an optional downsampling path for the residual
    connection. The bottleneck design reduces the number of parameters while
    maintaining the representational power of the network.

    Attributes
    ----------
    expansion : int
        The expansion factor for the number of output channels in the third
        convolutional layer.
    conv1 : nn.Conv2d
        The first convolutional layer with a kernel size of 1x1.
    bn1 : nn.BatchNorm2d
        Batch normalization layer for the first convolutional layer.
    relu1 : nn.ReLU
        ReLU activation function applied after the first batch normalization.
    conv2 : nn.Conv2d
        The second convolutional layer with a kernel size of 3x3.
    bn2 : nn.BatchNorm2d
        Batch normalization layer for the second convolutional layer.
    relu2 : nn.ReLU
        ReLU activation function applied after the second batch normalization.
    avgpool : nn.Module
        An average pooling layer applied after the second convolutional layer
        when stride > 1, otherwise an identity layer.
    conv3 : nn.Conv2d
        The third convolutional layer with a kernel size of 1x1.
    bn3 : nn.BatchNorm2d
        Batch normalization layer for the third convolutional layer.
    relu3 : nn.ReLU
        ReLU activation function applied after the third batch normalization.
    downsample : nn.Sequential or None
        An optional downsampling layer for the residual connection, consisting
        of an average pooling layer followed by a 1x1 convolution and batch
        normalization, used when the input and output dimensions do not match.
    stride : int
        The stride value for the block, which determines whether downsampling
        is applied.

    Parameters
    ----------
    inplanes : int
        The number of input channels.
    planes : int
        The number of output channels for the first and second convolutional
        layers. The third convolutional layer outputs `planes * expansion`
        channels.
    stride : int, optional
        The stride value for the block. If greater than 1, downsampling is
        applied. Default is 1.

    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor
        Defines the forward pass of the bottleneck block. Applies the sequence
        of convolutional layers, batch normalization, ReLU activations, and
        optional downsampling. Combines the output with the residual connection
        and applies a final ReLU activation.

    Notes
    -----
    - The bottleneck block is designed to reduce the computational cost and
      number of parameters in deep residual networks by using 1x1 convolutions
      to compress and expand the feature dimensions.
    - The residual connection ensures gradient flow during backpropagation,
      mitigating the vanishing gradient problem in deep networks.
    """

    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(
                OrderedDict(
                    [
                        ("-1", nn.AvgPool2d(stride)),
                        (
                            "0",
                            nn.Conv2d(
                                inplanes,
                                planes * self.expansion,
                                1,
                                stride=1,
                                bias=False,
                            ),
                        ),
                        ("1", nn.BatchNorm2d(planes * self.expansion)),
                    ]
                )
            )

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    class AttentionPool2d:
        """
        A PyTorch module that implements a 2D attention pooling mechanism. This module is designed to
        aggregate spatial information from a 2D input tensor using multi-head attention, which allows
        the model to focus on different parts of the input simultaneously.

        Parameters
        ----------
        spacial_dim : int
            The spatial dimension of the input tensor (assumes square input, i.e., height = width).
        embed_dim : int
            The dimensionality of the embedding space.
        num_heads : int
            The number of attention heads to use in the multi-head attention mechanism.
        output_dim : int, optional
            The dimensionality of the output embedding. If not provided, it defaults to `embed_dim`.

        Attributes
        ----------
        positional_embedding : torch.nn.Parameter
            A learnable positional embedding tensor of shape `(spacial_dim**2 + 1, embed_dim)`.
        k_proj : torch.nn.Linear
            A linear layer for projecting the input to the key space.
        q_proj : torch.nn.Linear
            A linear layer for projecting the input to the query space.
        v_proj : torch.nn.Linear
            A linear layer for projecting the input to the value space.
        c_proj : torch.nn.Linear
            A linear layer for projecting the output of the attention mechanism to the desired output dimension.
        num_heads : int
            The number of attention heads used in the multi-head attention mechanism.

        Methods
        -------
        forward(x)
            Applies the attention pooling mechanism to the input tensor.

        Notes
        -----
        - The input tensor is expected to have the shape `(N, C, H, W)`, where `N` is the batch size,
          `C` is the number of channels, and `H` and `W` are the spatial dimensions.
        - The module flattens the spatial dimensions of the input, adds a learnable positional embedding,
          and applies multi-head attention to compute a pooled representation.
        - The first token in the sequence (representing the global average) is used as the query for
          the attention mechanism.
        """

    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim**2 + 1, embed_dim) / embed_dim**0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1],
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """ModifiedResNet

    A modified version of the ResNet architecture with customizations for enhanced feature extraction and attention-based pooling.
    This class introduces the following changes to the standard ResNet:
    - A 3-layer "stem" convolutional block instead of a single convolution, with an average pooling layer replacing the max pooling layer.
    - Anti-aliasing strided convolutions, where an average pooling layer is prepended to convolutions with stride > 1.
    - A QKV attention mechanism replaces the final average pooling layer for better feature aggregation.

    Parameters
    ----------
    layers : list of int
        A list specifying the number of residual blocks in each of the four layers of the network.
    output_dim : int
        The dimensionality of the output feature vector.
    heads : int
        The number of attention heads in the QKV attention pooling layer.
    input_resolution : int, optional
        The input image resolution (default is 224).
    width : int, optional
        The base width of the network (default is 64).
    in_channels : int, optional
        The number of input channels in the input image (default is 3).

    Attributes
    ----------
    output_dim : int
        The dimensionality of the output feature vector.
    input_resolution : int
        The input image resolution.
    conv1, conv2, conv3 : nn.Conv2d
        Convolutional layers in the 3-layer "stem" block.
    bn1, bn2, bn3 : nn.BatchNorm2d
        Batch normalization layers corresponding to the "stem" convolutions.
    relu1, relu2, relu3 : nn.ReLU
        ReLU activation functions for the "stem" convolutions.
    avgpool : nn.AvgPool2d
        Average pooling layer in the "stem" block.
    layer1, layer2, layer3, layer4 : nn.Sequential
        Residual layers of the network, each containing a sequence of Bottleneck blocks.
    attnpool : AttentionPool2d
        Attention-based pooling layer for aggregating features.

    Methods
    -------
    forward(x)
        Defines the forward pass of the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_dim).
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64, in_channels=3):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(in_channels, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        in_channels: int,
        output_dim: int,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class SatCLIP(nn.Module):
    """
    SatCLIP: A model for satellite image and location embedding alignment.
    This class implements a neural network model designed to align embeddings of satellite images
    and their corresponding geospatial locations. It combines a vision backbone for image feature
    extraction and a location encoder for geospatial feature extraction. The model computes cosine
    similarity between the two embeddings to measure their alignment.

    Parameters
    ----------
    embed_dim : int
        The dimensionality of the embedding space for both image and location features.
    image_resolution : int
        The resolution of input images.
    vision_layers : Union[Tuple[int, int, int, int], int, str]
        The architecture of the vision backbone. Can be a tuple for ResNet, an integer for Vision Transformer,
        or a string for specific pretrained models (e.g., "moco_resnet18").
    vision_width : int
        The width of the vision backbone.
    vision_patch_size : int
        The patch size for Vision Transformer models.
    in_channels : int
        The number of input channels for the vision backbone.
    le_type : str
        The type of location encoding to use.
    pe_type : str
        The type of positional encoding to use.
    frequency_num : int
        The number of frequency components for positional encoding.
    max_radius : int
        The maximum radius for positional encoding.
    min_radius : int
        The minimum radius for positional encoding.
    harmonics_calculation : str
        The method used for harmonics calculation in positional encoding.
    legendre_polys : int, optional
        The number of Legendre polynomials to use in positional encoding (default is 10).
    sh_embedding_dims : int, optional
        The dimensionality of spherical harmonics embeddings (default is 16).
    ffn : bool, optional
        Whether to use a feed-forward network in the location encoder (default is True).
    num_hidden_layers : int, optional
        The number of hidden layers in the location encoder (default is 2).
    capacity : int, optional
        The hidden layer size in the location encoder (default is 256).
    *args : tuple
        Additional positional arguments.
    **kwargs : dict
        Additional keyword arguments.

    Attributes
    ----------
    visual : nn.Module
        The vision backbone for image feature extraction.
    posenc : nn.Module
        The positional encoding module for geospatial data.
    nnet : nn.Module
        The neural network for processing positional encodings.
    location : nn.Module
        The location encoder combining positional encoding and neural network.
    logit_scale : nn.Parameter
        A learnable parameter for scaling logits in cosine similarity computation.

    Methods
    -------
    initialize_parameters()
        Initializes the parameters of the model, particularly for the ResNet backbone.
    dtype
        Returns the data type of the model's parameters.
    encode_image(image)
        Encodes an input image into its feature representation.
    encode_location(coords)
        Encodes geospatial coordinates into their feature representation.
    forward(image, coords)
        Computes the cosine similarity logits between image and location embeddings.

    Notes
    -----
    The model supports multiple vision backbones, including ResNet, Vision Transformer, and pretrained
    models such as MoCo ResNet18/ResNet50/ViT. It also supports various positional encoding techniques
    for geospatial data, making it flexible for different satellite image and location alignment tasks.
    """

    def __init__(
        self,
        embed_dim: int,
        # vision
        image_resolution: int,
        vision_layers: Union[Tuple[int, int, int, int], int, str],
        vision_width: int,
        vision_patch_size: int,
        in_channels: int,
        # location
        le_type: str,
        pe_type: str,
        frequency_num: int,
        max_radius: int,
        min_radius: int,
        harmonics_calculation: str,
        legendre_polys: int = 10,
        sh_embedding_dims: int = 16,
        ffn: bool = True,
        num_hidden_layers: int = 2,
        capacity: int = 256,
        *args,
        **kwargs,
    ):
        super().__init__()

        if isinstance(vision_layers, (tuple, list)):
            print("using modified resnet")
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width,
                in_channels=in_channels,
            )

        elif vision_layers == "moco_resnet18":
            print("using pretrained moco resnet18")
            weights = ResNet18_Weights.SENTINEL2_ALL_MOCO
            in_chans = weights.meta["in_chans"]
            self.visual = timm.create_model("resnet18", in_chans=in_chans, num_classes=embed_dim)
            self.visual.load_state_dict(weights.get_state_dict(progress=True), strict=False)
            self.visual.requires_grad_(False)
            self.visual.fc.requires_grad_(True)

        elif vision_layers == "moco_resnet50":
            print("using pretrained moco resnet50")
            weights = ResNet50_Weights.SENTINEL2_ALL_MOCO
            in_chans = weights.meta["in_chans"]
            self.visual = timm.create_model("resnet50", in_chans=in_chans, num_classes=embed_dim)
            self.visual.load_state_dict(weights.get_state_dict(progress=True), strict=False)
            self.visual.requires_grad_(False)
            self.visual.fc.requires_grad_(True)

        elif vision_layers == "moco_vit16":
            print("using pretrained moco vit16")
            weights = ViTSmall16_Weights.SENTINEL2_ALL_MOCO
            in_chans = weights.meta["in_chans"]
            self.visual = timm.create_model("vit_small_patch16_224", in_chans=in_chans, num_classes=embed_dim)
            self.visual.load_state_dict(weights.get_state_dict(progress=True), strict=False)
            self.visual.requires_grad_(False)
            self.visual.head.requires_grad_(True)

        else:
            print("using vision transformer")
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                in_channels=in_channels,
            )

        self.posenc = get_positional_encoding(
            name=le_type,
            harmonics_calculation=harmonics_calculation,
            legendre_polys=legendre_polys,
            min_radius=min_radius,
            max_radius=max_radius,
            frequency_num=frequency_num,
        ).double()
        self.nnet = get_neural_network(
            name=pe_type,
            input_dim=self.posenc.embedding_dim,
            num_classes=embed_dim,
            dim_hidden=capacity,
            num_layers=num_hidden_layers,
        ).double()
        self.location = LocationEncoder(self.posenc, self.nnet).double()

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features**-0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [
                self.visual.layer1,
                self.visual.layer2,
                self.visual.layer3,
                self.visual.layer4,
            ]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

    @property
    def dtype(self):
        if isinstance(self.visual, timm.models.vision_transformer.VisionTransformer):
            return self.visual.patch_embed.proj.weight.dtype
        else:
            return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_location(self, coords):
        return self.location(coords.double())

    def forward(self, image, coords):

        image_features = self.encode_image(image)
        location_features = self.encode_location(coords).float()
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        location_features = location_features / location_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ location_features.t()
        logits_per_location = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_location


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(val):
        if isinstance(val, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            val.weight.data = val.weight.data.half()
            if val.bias is not None:
                val.bias.data = val.bias.data.half()

        if isinstance(val, nn.MultiheadAttention):
            for attr in [
                *[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]],
                "in_proj_bias",
                "bias_k",
                "bias_v",
            ]:
                tensor = getattr(val, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(val, name):
                attr = getattr(val, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)
