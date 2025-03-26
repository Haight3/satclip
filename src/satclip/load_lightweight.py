# -*- coding: utf-8 -*-
# pylint: disable=E1101,W0221,R0901,R0902,R0913,R0914,E1136,C0301,C0411,C0412
"""
Load Encoder
============
*Created on 25 by bari_is*
*Copyright (C) 2025*
*For COPYING and LICENSE details, please refer to the LICENSE file*

This module provides functionality for loading a lightweight location encoder for the SatClip project.
The primary purpose of this file is to define a utility function `get_satclip_loc_encoder` that initializes
and returns a `LocationEncoder` instance using a checkpoint file containing the model's state dictionary
and hyperparameters.
The `LocationEncoder` is a key component of the SatClip system, responsible for encoding spatial
information using positional encoding and a neural network.

NOTES
-----
- The checkpoint file must contain:
  - `state_dict`: The parameters of the neural network.
  - `hyper_parameters`: Configuration details for positional encoding and the neural network.
- The module assumes that only the neural network parameters (`nnet`) are stored in the state dictionary.

"""


import torch

from satclip.location_encoder import LocationEncoder, get_neural_network, get_positional_encoding


def get_satclip_loc_encoder(ckpt_path: str, device: str) -> LocationEncoder:
    """
    Loads and returns a location encoder for SatClip from a checkpoint file.

    Parameters
    ----------
    ckpt_path : str
        Path to the checkpoint file containing the model's state dictionary and hyperparameters.
    device : torch.device
        The device on which to load the model (e.g., 'cpu' or 'cuda').

    Returns
    -------
    LocationEncoder
        A location encoder instance initialized with the loaded parameters and set to evaluation mode.

    Notes
    -----
    - The checkpoint file is expected to contain a 'state_dict' with the neural network parameters
      and 'hyper_parameters' with the configuration for positional encoding and the neural network.
    - Only the parameters related to the neural network (`nnet`) are loaded from the state dictionary.
    """

    ckpt = torch.load(ckpt_path, map_location=device)
    hp = ckpt['hyper_parameters']

    posenc = get_positional_encoding(
        hp['le_type'],
        hp['legendre_polys'],
        hp['harmonics_calculation'],
        hp['min_radius'],
        hp['max_radius'],
        hp['frequency_num'],
    )

    nnet = get_neural_network(
        hp['pe_type'],
        posenc.embedding_dim,
        hp['embed_dim'],
        hp['capacity'],
        hp['num_hidden_layers'],
    )

    # Only load nnet params from state dict
    state_dict = ckpt['state_dict']
    state_dict = {k[k.index('nnet') :]: state_dict[k] for k in state_dict.keys() if 'nnet' in k}

    loc_encoder = LocationEncoder(posenc, nnet).double()
    loc_encoder.load_state_dict(state_dict)
    loc_encoder.eval()

    return loc_encoder
