#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" A collection of functions useful for loading and preprocessing microscope input.

This code are excerpts from Marcus Marcus Brubaker's github:
https://github.com/mbrubake/cryoem-cvpr2015

Please consider other licensing conditions.
Author: Marcus Brubaker, 2015
"""
import numpy as n

try:
    import pyfftw

    fftmod = pyfftw.interfaces.numpy_fft
    pyfftw.interfaces.cache.enable()

    # install like so: https://dranek.com/blog/2014/Feb/conda-binstar-and-fftw/
    #     print "LOADED FFTW"
    USINGFFTW = True

    import multiprocessing

    fft_threads = multiprocessing.cpu_count()
except:
    fftmod = n.fft
    USINGFFTW = False
    print("ERROR LOADING FFTW! USING NUMPY")
    fft_threads = None

real_t = n.float32
complex_t = n.complex64


def readMRCheader(fname):
    hdr = None
    with open(fname) as f:
        hdr = {}
        header = n.fromfile(f, dtype=n.int32, count=256)
        header_f = header.view(n.float32)
        [hdr['nx'], hdr['ny'], hdr['nz'], hdr['datatype']] = header[:4]
        [hdr['xlen'], hdr['ylen'], hdr['zlen']] = header_f[10:13]
        # print "Nx %d Ny %d Nz %d Type %d" % (nx, ny, nz, datatype)
    return hdr


def readMRC(fname, inc_header=False):
    hdr = readMRCheader(fname)
    nx = hdr['nx']
    ny = hdr['ny']
    nz = hdr['nz']
    datatype = hdr['datatype']
    with open(fname) as f:
        f.seek(1024)  # seek to start of data
        if datatype == 0:
            data = n.reshape(n.fromfile(f, dtype='int8', count=nx * ny * nz), (nx, ny, nz), order='F')
        elif datatype == 1:
            data = n.reshape(n.fromfile(f, dtype='int16', count=nx * ny * nz), (nx, ny, nz), order='F')
        elif datatype == 2:
            data = n.reshape(n.fromfile(f, dtype='float32'), (nx, ny, nz), order='F')
        else:
            assert False, 'Unsupported MRC datatype: {0}'.format(datatype)
    if inc_header:
        return data, hdr
    else:
        return data


def compute_premultiplier(N, kernel, kernsize, scale=512):
    krange = N / 2
    koffset = (N / 2) * scale

    x = n.arange(-scale * krange, scale * krange) / float(scale)

    if kernel == 'lanczos':
        a = kernsize / 2
        k = n.sinc(x) * n.sinc(x / a) * (n.abs(x) <= a)
    elif kernel == 'sinc':
        a = kernsize / 2.0
        k = n.sinc(x) * (n.abs(x) <= a)
    elif kernel == 'linear':
        assert kernsize == 2
        k = n.maximum(0.0, 1 - n.abs(x))
    elif kernel == 'quad':
        assert kernsize == 3
        k = (n.abs(x) <= 0.5) * (1 - 2 * x ** 2) + ((n.abs(x) < 1) * (n.abs(x) > 0.5)) * 2 * (1 - n.abs(x)) ** 2
    else:
        assert False, 'Unknown kernel type'

    sk = n.fft.fftshift(n.fft.ifft(n.fft.ifftshift(k))).real
    premult = 1.0 / (N * sk[int(koffset - krange):int(koffset + krange)])

    return premult


""" Convert real-space M to (unitary) Fourier space """


def real_to_fspace(M, axes=None, threads=None):
    if USINGFFTW:
        if threads is None:
            threads = fft_threads
        ret = n.require(n.fft.fftshift(fftmod.fftn(n.fft.fftshift(M, axes=axes), \
                                                   axes=axes, threads=threads), \
                                       axes=axes), \
                        dtype=complex_t)
    else:
        ret = n.require(n.fft.fftshift(fftmod.fftn(n.fft.fftshift(M, axes=axes), \
                                                   axes=axes), \
                                       axes=axes), \
                        dtype=complex_t)
        ret = n.require(n.fft.fftshift(fftmod.fftn(n.fft.fftshift(M))),
                        dtype=complex_t)
    # nrm is the scaling factor needed to make an unnormalized FFT a
    # unitary transform
    if axes is None:
        nrm = 1.0 / n.sqrt(n.prod(M.shape))
    else:
        nrm = 1.0 / n.sqrt(n.prod(n.array(M.shape)[n.array(axes)]))
    ret *= nrm
    return ret


""" Convert unitary Fourier space fM to real space """


def fspace_to_real(fM, axes=None, threads=None):
    if USINGFFTW:
        if threads is None:
            threads = fft_threads
        ret = n.require(n.fft.ifftshift(fftmod.ifftn(n.fft.ifftshift(fM, axes=axes), \
                                                     axes=axes, threads=threads), \
                                        axes=axes).real, \
                        dtype=real_t)
    else:
        ret = n.require(n.fft.ifftshift(fftmod.ifftn(n.fft.ifftshift(fM, axes=axes), \
                                                     axes=axes), \
                                        axes=axes).real, \
                        dtype=real_t)
    # nrm is the scaling factor needed to make an unnormalized FFT a
    # unitary transform
    if axes is None:
        nrm = n.sqrt(n.prod(fM.shape))
    else:
        nrm = n.sqrt(n.prod(n.array(fM.shape)[n.array(axes)]))
    ret *= nrm
    return ret


def gencoords_base(N, d):
    x = n.arange(-N / 2, N / 2, dtype=n.float32)
    c = x.copy()
    for i in range(1, d):
        c = n.column_stack([n.repeat(c, N, axis=0), n.tile(x, N ** i)])

    return c


def gencoords(N, d, rad=None, truncmask=False, trunctype='circ'):
    """ generate coordinates of all points in an NxN..xN grid with d dimensions
    coords in each dimension are [-N/2, N/2)
    N should be even"""

    if not truncmask:
        _, truncc, _ = gencoords(N, d, rad, True)
        return truncc

    c = gencoords_base(N, d)

    if rad is not None:
        if trunctype == 'circ':
            r2 = n.sum(c ** 2, axis=1)
            trunkmask = r2 < (rad * N / 2.0) ** 2
        elif trunctype == 'square':
            r = n.max(n.abs(c), axis=1)
            trunkmask = r < (rad * N / 2.0)

        truncc = c[trunkmask, :]
    else:
        trunkmask = n.ones((c.shape[0],), dtype=n.bool8)
        truncc = c

    return c, truncc, trunkmask


def window(v, func='hanning', params=None):
    """ applies a windowing function to the 3D volume v (inplace, as reference) """

    N = v.shape[0]
    D = v.ndim
    if any([d != N for d in list(v.shape)]) or D != 3:
        raise Exception("Error: Volume is not Cube.")

    def apply_seperable_window(v, w):
        v *= n.reshape(w, (-1, 1, 1))
        v *= n.reshape(w, (1, -1, 1))
        v *= n.reshape(w, (1, 1, -1))

    if func == "hanning":
        w = n.hanning(N)
        apply_seperable_window(v, w)
    elif func == 'hamming':
        w = n.hamming(N)
        apply_seperable_window(v, w)
    elif func == 'gaussian':
        raise Exception('Unimplimented')
    elif func == 'circle':
        c = gencoords(N, 3)
        if params == None:
            r = N / 2 - 1
        else:
            r = params[0] * (N / 2 * 1)
        v *= (n.sum(c ** 2, 1) < (r ** 2)).reshape((N, N, N))
    elif func == 'box':
        v[:, 0, 0] = 0.0
        v[0, :, 0] = 0.0
        v[0, 0, :] = 0.0
    else:
        raise Exception("Error: Window Type Not Supported")
