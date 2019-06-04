#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Geometric operators for the Fourier Domain.


Karen Ullrich, May 2019
"""

import numpy as np

import torch
import torch.nn as nn


def base_grid_generator3d(size):
    """Compute grid for the center slice
    """
    N, C, H, W, D = size
    x = np.linspace(-H / 2, H / 2 - 1, H)
    y = np.linspace(-H / 2, H / 2 - 1, H)
    base_grid = np.vstack(np.meshgrid(x, y)).reshape(2, -1).T
    base_grid = np.hstack([base_grid, np.zeros((H * W, 1))])
    base_grid = np.expand_dims(base_grid.reshape(H, W, 1, 3), 0)
    base_grid = base_grid.repeat(N, 0)
    return nn.Parameter(torch.Tensor(base_grid), requires_grad=False)


def base_grid_generator2d(size):
    """Compute grid for the center slice
    """
    N, C, H, W = size
    x = np.linspace(-H / 2, H / 2 - 1, H) / (H / 2)
    y = np.linspace(-H / 2, H / 2 - 1, H) / (H / 2)
    base_grid = np.vstack(np.meshgrid(x, y)).reshape(2, -1).T
    base_grid = np.expand_dims(base_grid.reshape(H, W, 2), 0)
    base_grid = base_grid.repeat(N, 0)
    return nn.Parameter(torch.Tensor(base_grid), requires_grad=False)


class Translate(nn.Module):
    def __init__(self, batch_size, N):
        super(Translate, self).__init__()

        self.image_base_grid = base_grid_generator2d((batch_size, 2, N, N))

        self.realidx = nn.Parameter(
            torch.LongTensor([0]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(batch_size, 1, N, N),
            requires_grad=False)
        self.imagidx = nn.Parameter(
            torch.LongTensor([1]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(batch_size, 1, N, N),
            requires_grad=False)

    def __call__(self, projection, x):
        device = projection.device
        preal = torch.gather(projection, dim=1, index=self.realidx)
        pimag = torch.gather(projection, dim=1, index=self.imagidx)

        x = x.unsqueeze(1).unsqueeze(1)
        kx = torch.sum(self.image_base_grid * x, dim=-1).unsqueeze(1)

        coskx = torch.cos(2 * np.pi * kx).to(device)
        sinkx = torch.sin(2 * np.pi * kx).to(device)

        outreal = coskx * preal + sinkx * pimag
        outimag = coskx * pimag - sinkx * preal

        return torch.cat([outreal, outimag], dim=1)


class SliceExctractor(nn.Module):
    def __init__(self, limit, batch_size):
        super(SliceExctractor, self).__init__()

        self.limit = limit
        self.batch_size = batch_size
        batch_idx = torch.Tensor(np.arange(self.batch_size)).repeat(self.limit ** 2, 1, 2, 1).permute(0, 3, 2, 1).long()
        c0_idx = torch.Tensor(np.zeros(self.batch_size)).repeat(self.limit ** 2, 1, 1, 1).permute(0, 3, 2, 1).long()
        c1_idx = torch.Tensor(np.ones(self.batch_size)).repeat(self.limit ** 2, 1, 1, 1).permute(0, 3, 2, 1).long()
        self.idxer = nn.Parameter(torch.cat([batch_idx, torch.cat([c0_idx, c1_idx], dim=-2)], dim=-1),
                                  requires_grad=False)

    def save_get(self, volume, idx, boundary_mode="periodic"):

        if boundary_mode == "periodic":
            idx = (idx % (self.limit - 1))
        elif boundary_mode == "continious":
            idx = torch.clamp(idx, 0, self.limit - 1)

        idx = idx.permute(1, 2, 0, 3).view(self.limit * self.limit, self.batch_size, 3)
        idx = idx.unsqueeze(-2).repeat(1, 1, 2, 1)
        idx = torch.cat([self.idxer, idx.long()], dim=-1).view(self.limit * self.limit * self.batch_size * 2, 5)

        return volume[torch.unbind(idx, dim=-1)].view(self.limit, self.limit, self.batch_size, 2).permute(2, 3, 0, 1)

    def forward(self, volume, grid):
        ix = grid[:, :, :, 0]
        iy = grid[:, :, :, 1]
        iz = grid[:, :, :, 2]

        px_0 = torch.floor(ix)
        py_0 = torch.floor(iy)
        pz_0 = torch.floor(iz)
        px_1 = torch.ceil(ix)
        py_1 = torch.ceil(iy)
        pz_1 = torch.ceil(iz)

        dx = (ix - px_0).unsqueeze(1)
        dy = (iy - py_0).unsqueeze(1)
        dz = (iz - pz_0).unsqueeze(1)

        c_000 = self.save_get(volume, idx=torch.stack([py_0, px_0, pz_0], dim=-1))
        c_100 = self.save_get(volume, idx=torch.stack([py_0, px_1, pz_0], dim=-1))
        c_00 = c_000 * (1. - dx) + c_100 * (dx)
        del c_000, c_100

        c_010 = self.save_get(volume, idx=torch.stack([py_1, px_0, pz_0], dim=-1))
        c_110 = self.save_get(volume, idx=torch.stack([py_1, px_1, pz_0], dim=-1))
        c_10 = c_010 * (1. - dx) + c_110 * (dx)
        del c_010, c_110

        c_0 = c_00 * (1. - dy) + c_10 * (dy)
        del c_00, c_10

        c_001 = self.save_get(volume, idx=torch.stack([py_0, px_0, pz_1], dim=-1))
        c_101 = self.save_get(volume, idx=torch.stack([py_0, px_1, pz_1], dim=-1))
        c_01 = c_001 * (1. - dx) + c_101 * (dx)
        del c_001, c_101

        c_011 = self.save_get(volume, idx=torch.stack([py_1, px_0, pz_1], dim=-1))
        c_111 = self.save_get(volume, idx=torch.stack([py_1, px_1, pz_1], dim=-1))
        c_11 = c_011 * (1. - dx) + c_111 * (dx)
        del c_011, c_111

        c_1 = c_01 * (1. - dy) + c_11 * (dy)
        del c_11, c_01

        return c_0 * (1. - dz) + c_1 * (dz)


# compute Euler Angles based rotation matrix

component_1_x = torch.FloatTensor([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
component_cos_x = torch.FloatTensor([[0, 0, 0, 0, 1, 0, 0, 0, 1]])
component_sin_x = torch.FloatTensor([[0, 0, 0, 0, 0, -1, 0, 1, 0]])

component_1_z = torch.FloatTensor([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
component_cos_z = torch.FloatTensor([[1, 0, 0, 0, 1, 0, 0, 0, 0]])
component_sin_z = torch.FloatTensor([[0, -1, 0, 1, 0, 0, 0, 0, 0]])

component_1_y = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
component_cos_y = torch.FloatTensor([[1, 0, 0, 0, 0, 0, 0, 0, 1]])
component_sin_y = torch.FloatTensor([[0, 0, 1, 0, 0, 0, -1, 0, 0]])


def cosinefy_x(x, device):
    batch_size = len(x)
    y = torch.mm(x, component_cos_x.to(device))
    y = y.resize(batch_size * 3, 3)
    return y


def sinefy_x(x, device):
    batch_size = len(x)
    y = torch.mm(x, component_sin_x.to(device))
    y = y.resize(batch_size * 3, 3)
    return y


def cosinefy_z(x, device):
    batch_size = len(x)
    y = torch.mm(x, component_cos_z.to(device))
    y = y.resize(batch_size * 3, 3)
    return y


def sinefy_z(x, device):
    batch_size = len(x)
    y = torch.mm(x, component_sin_z.to(device))
    y = y.resize(batch_size * 3, 3)
    return y


def cosinefy_y(x, device):
    batch_size = len(x)
    y = torch.mm(x, component_cos_y.to(device))
    y = y.resize(batch_size * 3, 3)
    return y


def sinefy_y(x, device):
    batch_size = len(x)
    y = torch.mm(x, component_sin_y.to(device))
    y = y.resize(batch_size * 3, 3)
    return y


def R_x(g, device):
    """
    Compute the Euler Angles R_x
    """
    cos_angles = cosinefy_x(torch.cos(g), device)
    sin_angles = sinefy_x(torch.sin(g), device)

    out = sin_angles + cos_angles
    return out.resize(len(g), 3, 3) + component_1_x.to(device)


def R_z(g, device):
    """
    Compute the Euler Angles R_z
    """
    cos_angles = cosinefy_z(torch.cos(g), device)
    sin_angles = sinefy_z(torch.sin(g), device)

    out = sin_angles + cos_angles
    return out.resize(len(g), 3, 3) + component_1_z.to(device)


def R_y(g, device):
    """
    Compute the Euler Angles R_y
    """
    cos_angles = cosinefy_y(torch.cos(g), device)
    sin_angles = sinefy_y(torch.sin(g), device)

    out = sin_angles + cos_angles
    return out.resize(len(g), 3, 3) + component_1_y.to(device)


def rotmat3D_EA(g):
    """
    Generates a rotation matrix from Z-Y-Z Euler angles. This rotation matrix
    maps from image coordinates (x,y,0) to view coordinates.
    """
    device = g.device
    R_phi = R_z(g[:, 0].view(-1, 1), device)
    R_theta = R_y(g[:, 1].view(-1, 1), device)
    R_psi = R_z(g[:, 2].view(-1, 1), device)

    R = torch.bmm(R_phi, R_theta)
    R = torch.bmm(R, R_psi)

    return R.resize(len(g), 3, 3)
