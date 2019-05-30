#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Observation model for scientific imaging.

This is a simple implementation of the generative model of many scientific imaging methods such as CT, or microscopy.

Karen Ullrich, May 2019
"""

import numpy as np
import math
import torch
import torch.nn as nn
from torch import distributions

from operators import base_grid_generator3d, translate, SliceExctractor


class ScientificImagingObservationModel(nn.Module):
    """This class implements the scientific imaging generative model.
    """

    def __init__(self, D=128, batch_size=1, std_noise=0.):
        r"""Module init.

            Some elements of the model are best shared across serveral forward passes. That is why we put them into the
            __init__ method, for example the set of coordinates to generate the slice base_slice.

            Parameters
            ----------
            D : int
                This is the dimensionality of the observation. For a 128 x 128 pixel image, D=128.
            batch_size : int
                The expected batch size. This is useful for optimization purposes.
            std_noise : float
                We assume a Gaussian observation model in real space. This parameter represents the corresponding standard deviation.
        """
        super(ScientificImagingObservationModel, self).__init__()
        self.D = D
        self.batch_size = batch_size
        self.std_noise = std_noise

        self.base_slice = nn.Parameter(base_grid_generator3d((self.batch_size, 2, self.D, self.D, self.D)),
                                       requires_grad=False)
        self.extract_slice = SliceExctractor(limit=self.D, batch_size=self.batch_size)
        self.translate = translate(self.batch_size, self.D)

    def forward(self, protein_samples, rotation_samples, translation_samples):
        r"""Module forward method.

            This function implements the backprojection method.

            Parameters
            ----------
            protein_samples : torch.Tensor
                Protein samples in Fourier space, sampled from latent distribution.
            rotation_samples : torch.Tensor
                Rotation samples are either coming from the latent rotation distribution or from data labels.
            translation_samples : torch.Tensor
                Translation samples are either coming from the latent rotation distribution or from data labels.
        """
        # compute inverse rotation and translation
        R = rotation_samples.permute(0, 2, 1).view(-1, 3, 3)
        t = - translation_samples.view(-1, 2)
        t = torch.stack([t[:, 1], t[:, 0]], dim=1)

        # generate slice coordinates
        slice = torch.bmm(self.base_slice.view(self.batch_size, -1, 3), R)
        slice = slice.view(self.batch_size, self.D, self.D, 3)
        slice += self.D / 2

        # extract interpolated projection slice
        projection = self.extract_slice(protein_samples, slice).permute(0, 1, 3, 2)

        # unitary fourier transform
        projection *= np.sqrt(self.D)

        # translation in fourier space
        t = t * self.D / 2  # scale by size of image
        projection = self.translate(projection, t)

        # return the projection and the observation models distribution, this is a Gaussian pytorch distribution
        if self.std_noise == 0.:
            return projection, distributions.Normal(projection, 1.)
        else:
            return projection, distributions.Normal(projection, self.std_noise / math.sqrt(2))
        # Why is there a factor math.sqrt(2)? This relates to the question how white noise in the real domain
        # relates to the Fourier domain. This post explains it in detail:
        # https://dsp.stackexchange.com/questions/24170/what-is-the-statistics-of-the-discrete-fourier-transform-of-white-gaussian-noise
        # WARNING: You should not sample from this ddistribution since the noise would not be restircted to a real signal!
