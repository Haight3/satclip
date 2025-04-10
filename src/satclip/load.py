# -*- coding: utf-8 -*-
# pylint: disable=E1101,W0221,R0901,R0902,R0913,R0914,E1136,C0301,C0411,C0412
"""
Load SatClip
============
*Created on 25 by bari_is*
*Copyright (C) 2025*
*For COPYING and LICENSE details, please refer to the LICENSE file*

This module provides functionality to load and initialize the SatCLIP model
and its components, such as the location encoder. The SatCLIP model is a
specialized machine learning model designed for geospatial tasks.

Notes
-----
- Ensure that the checkpoint file paths provided are valid and accessible.
- The checkpoint file should contain the necessary hyperparameters and
  state dictionary for the SatCLIP model.
- Some hyperparameters are removed during loading to avoid conflicts.

"""


import torch

from satclip import SatCLIPLightningModule
from satclip.location_encoder import LocationEncoder
from satclip.model import SatCLIP

__all__ = ["get_satclip", "load_location_encoder"]


def get_satclip(ckpt_path: str | None = None) -> SatCLIP:
    """
    Loads and returns a SatCLIP model.

    Parameters
    ----------
    ckpt_path : str or None, optional
        Path to the checkpoint file. If provided, the model will be loaded
        from the checkpoint. If None, a new instance of the SatCLIP model
        will be created. Default is None.

    Returns
    -------
    SatCLIP
        The loaded or newly created SatCLIP model instance.
    """

    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        ckpt["hyper_parameters"].pop("eval_downstream")
        ckpt["hyper_parameters"].pop("air_temp_data_path")
        ckpt["hyper_parameters"].pop("election_data_path")
        lightning_model = SatCLIPLightningModule(**ckpt["hyper_parameters"])

        lightning_model.load_state_dict(ckpt["state_dict"])
        lightning_model.eval()

    else:
        lightning_model = SatCLIPLightningModule()

    geo_model = lightning_model.model

    return geo_model


def load_location_encoder(ckpt_path: str | None = None) -> LocationEncoder:
    """
    Load the location encoder from a SATCLIP model checkpoint.

    Parameters
    ----------
    ckpt_path : str or None, optional
        Path to the checkpoint file for the SATCLIP model. If None, the default
        checkpoint will be used.

    Returns
    -------
    LocationEncoder
        The location encoder component of the SATCLIP model.
    """

    model = get_satclip(ckpt_path=ckpt_path)

    return model.location
