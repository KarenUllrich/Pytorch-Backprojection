#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Tools for visualising microscopy data.


Karen Ullrich, May 2019
"""

import gc
import matplotlib.pyplot as plt
import numpy as np


def preprocess_fourier_projection(cuda_array, logsclae=False, centercrop=64, N=128):
    x = cuda_array.cpu().data.numpy()

    if logsclae:
        x = np.sign(x) * np.log(np.abs(x) + 1e-8)

    x = x[int(N / 2) - centercrop:int(N / 2) + centercrop, int(N / 2) - centercrop:int(N / 2) + centercrop]
    return x


def plot_fourier(cuda_array, name, vmin=None, vmax=None, logsclae=False, centercrop=64, N=128, ticks=False, save=True):
    x = preprocess_fourier_projection(cuda_array, logsclae=logsclae, centercrop=centercrop, N=N)

    plt.figure()
    ax = plt.gca()

    if ticks:
        # Major ticks
        ax.set_xticks(np.arange(0, 2. * centercrop, 1))
        ax.set_yticks(np.arange(0, 2. * centercrop, 1))

        # Labels for major ticks
        ax.set_xticklabels(np.arange(-centercrop, centercrop, 1))
        ax.set_yticklabels(np.arange(-centercrop, centercrop, 1))
        # Minor ticks
        ax.set_xticks(np.arange(-.5, 2. * centercrop, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 2. * centercrop, 1), minor=True)
    else:
        plt.xticks([])
        plt.yticks([])

    # Gridlines based on minor ticks
    ax.grid(which='minor', color='w', linestyle='-', linewidth=0.1)
    plt.imshow(x, interpolation='none', vmin=vmin, vmax=vmax, aspect='equal', cmap='bone')
    plt.colorbar()
    if save:
        plt.savefig(name, bbox_inches='tight',
                    transparent=True,
                    pad_inches=0)
    else:
        plt.show()

    plt.close('all')
    gc.collect()


def plot_fourier2real(I, name, instance=0, save=True):
    projection = I.cpu().data.numpy()

    complex_projection = projection[instance, 0] + 1j * projection[instance, 1]

    reconstruction = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(complex_projection)))

    fig, ax = plt.subplots(1, 1)
    plt.imshow(reconstruction.real, interpolation="none", cmap='Greys', vmin=None, vmax=None)
    plt.colorbar()
    ax.axis('off')
    if save:
        plt.savefig(name + '_real',
                    bbox_inches='tight',
                    transparent=True,
                    pad_inches=0)
    else:
        print("Real part (signal):")
        plt.show()
    plt.close('all')

    plt.imshow(reconstruction.imag, interpolation="none", cmap='Greys', vmin=None, vmax=None)
    plt.colorbar()
    ax.axis('off')
    if save:
        plt.savefig(name + '_imag',
                    bbox_inches='tight',
                    transparent=True,
                    pad_inches=0)
    else:
        print("Imaginary part (This should be zero):")
        plt.show()
    plt.close('all')
    plt.show()
    gc.collect()
