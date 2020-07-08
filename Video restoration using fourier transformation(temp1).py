# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 15:35:58 2020

@author: Kushani Ranaweera
"""
import numpy as np
import argparse
import cv2
import imutils
import time
import scipy
from PIL import Image, ImageFilter
from scipy import fftpack

vs = cv2.VideoCapture('C:/Users/Kushani Ranaweera/Documents/tharka/6th semester/CO543/project/Implenentations/case3_2.mp4')

Lower = (110, 50, 50)
Upper = (130, 255, 255)



    
def get_rho_sigma(sigma=2.55/255, iter_num=15):
    '''
    Kai Zhang (github: https://github.com/cszn)
    03/03/2019
    '''
    modelSigma1 = 49.0
    modelSigma2 = 2.55
    modelSigmaS = np.logspace(np.log10(modelSigma1), np.log10(modelSigma2), iter_num)
    sigmas = modelSigmaS/255.
    mus = list(map(lambda x: (sigma**2)/(x**2)/3, sigmas))
    rhos = mus
    return rhos, sigmas


# --------------------------------
# HWC, get uperleft and denominator
# --------------------------------
def get_uperleft_denominator(Vid, kernel):
    '''
    Kai Zhang (github: https://github.com/cszn)
    03/03/2019
    '''
    V = psf2otf(kernel, Vid.shape[:2])   # discrete fourier transform of kernel
    denominator = np.expand_dims(np.abs(V)**2, axis=2)  # Fourier transform of K transpose * K
    upperleft = np.expand_dims(np.conj(V), axis=2) * np.fft.fft2(Vid, axes=[0, 1])
    return upperleft, denominator


# otf2psf: not sure where I got this one from. Maybe translated from Octave source code or whatever. It's just math.
def otf2psf(otf, outsize=None):
    insize = np.array(otf.shape)
    psf = np.fft.ifftn(otf, axes=(0, 1))
    for axis, axis_size in enumerate(insize):
        psf = np.roll(psf, np.floor(axis_size / 2).astype(int), axis=axis)
    if type(outsize) != type(None):
        insize = np.array(otf.shape)
        outsize = np.array(outsize)
        n = max(np.size(outsize), np.size(insize))
        # outsize = postpad(outsize(:), n, 1);
        # insize = postpad(insize(:) , n, 1);
        colvec_out = outsize.flatten().reshape((np.size(outsize), 1))
        colvec_in = insize.flatten().reshape((np.size(insize), 1))
        outsize = np.pad(colvec_out, ((0, max(0, n - np.size(colvec_out))), (0, 0)), mode="constant")
        insize = np.pad(colvec_in, ((0, max(0, n - np.size(colvec_in))), (0, 0)), mode="constant")

        pad = (insize - outsize) / 2
        if np.any(pad < 0):
            print("otf2psf error: OUTSIZE must be smaller than or equal than OTF size")
        prepad = np.floor(pad)
        postpad = np.ceil(pad)
        dims_start = prepad.astype(int)
        dims_end = (insize - postpad).astype(int)
        for i in range(len(dims_start.shape)):
            psf = np.take(psf, range(dims_start[i][0], dims_end[i][0]), axis=i)
    n_ops = np.sum(otf.size * np.log2(otf.shape))
    psf = np.real_if_close(psf, tol=n_ops)
    return psf


# psf2otf copied/modified from https://github.com/aboucaud/pypher/blob/master/pypher/pypher.py
def psf2otf(psf, shape=None):

    if type(shape) == type(None):
        shape = psf.shape
    shape = np.array(shape)
    if np.all(psf == 0):
        # return np.zeros_like(psf)
        return np.zeros(shape)
    if len(psf.shape) == 1:
        psf = psf.reshape((1, psf.shape[0]))
    inshape = psf.shape
    psf = zero_pad(psf, shape, position='corner')
    for axis, axis_size in enumerate(inshape):
        psf = np.roll(psf, -int(axis_size / 2), axis=axis)
    # Compute the OTF
    otf = np.fft.fft2(psf, axes=(0, 1))
    # Estimate the rough number of operations involved in the FFT
    # and discard the PSF imaginary part if within roundoff error
    # roundoff error  = machine epsilon = sys.float_info.epsilon
    # or np.finfo().eps
    n_ops = np.sum(psf.size * np.log2(psf.shape))
    otf = np.real_if_close(otf, tol=n_ops)
    return otf


def zero_pad(frame, shape, position='corner'):

    shape = np.asarray(shape, dtype=int)
    Videoshape = np.asarray(frame.shape, dtype=int)
    if np.alltrue(Videoshape == shape):
        return frame
    if np.any(shape <= 0):
        raise ValueError("ZERO_PAD: null or negative shape given")
    dshape = shape - shape
    if np.any(dshape < 0):
        raise ValueError("ZERO_PAD: target size smaller than source one")
    pad_frame = np.zeros(shape, dtype=image.dtype)
    idx, idy = np.indices(frame.shape)
    if position == 'center':
        if np.any(dshape % 2 != 0):
            raise ValueError("ZERO_PAD: source and target shapes "
                             "have different parity.")
        offx, offy = dshape // 2
    else:
        offx, offy = (0, 0)
    pad_frame[idx + offx, idy + offy] = frame
    return pad_frame


'''
Reducing boundary artifacts
'''


def opt_fft_size(n):
    '''
    Kai Zhang (github: https://github.com/cszn)
    03/03/2019
    #  opt_fft_size.m
    # compute an optimal data length for Fourier transforms
    # written by Sunghyun Cho (sodomau@postech.ac.kr)
    # persistent opt_fft_size_LUT;
    '''

    LUT_size = 2048
    # print("generate opt_fft_size_LUT")
    opt_fft_size_LUT = np.zeros(LUT_size)

    e2 = 1
    while e2 <= LUT_size:
        e3 = e2
        while e3 <= LUT_size:
            e5 = e3
            while e5 <= LUT_size:
                e7 = e5
                while e7 <= LUT_size:
                    if e7 <= LUT_size:
                        opt_fft_size_LUT[e7-1] = e7
                    if e7*11 <= LUT_size:
                        opt_fft_size_LUT[e7*11-1] = e7*11
                    if e7*13 <= LUT_size:
                        opt_fft_size_LUT[e7*13-1] = e7*13
                    e7 = e7 * 7
                e5 = e5 * 5
            e3 = e3 * 3
        e2 = e2 * 2

    nn = 0
    for i in range(LUT_size, 0, -1):
        if opt_fft_size_LUT[i-1] != 0:
            nn = i-1
        else:
            opt_fft_size_LUT[i-1] = nn+1

    m = np.zeros(len(n))
    for c in range(len(n)):
        nn = n[c]
        if nn <= LUT_size:
            m[c] = opt_fft_size_LUT[nn-1]
        else:
            m[c] = -1
    return m


def wrap_boundary_liu(frame, frame_size):

    """
    Reducing boundary artifacts in image deconvolution
    Renting Liu, Jiaya Jia
    ICIP 2008
    """
    if frame.ndim == 2:
        ret = wrap_boundary(frame, frame_size)
    elif frame.ndim == 3:
        ret = [wrap_boundary(frame[:, :, i], frame_size) for i in range(3)]
        ret = np.stack(ret, 2)
    return ret


def wrap_boundary(frame, frame_size):

    """
    python code from:
    https://github.com/ys-koshelev/nla_deblur/blob/90fe0ab98c26c791dcbdf231fe6f938fca80e2a0/boundaries.py
    Reducing boundary artifacts in image deconvolution
    Renting Liu, Jiaya Jia
    ICIP 2008
    """
    (H, W) = np.shape(frame)
    H_w = int(frame_size[0]) - H
    W_w = int(frame_size[1]) - W

    # ret = np.zeros((img_size[0], img_size[1]));
    alpha = 1
    HG = frame[:, :]

    r_A = np.zeros((alpha*2+H_w, W))
    r_A[:alpha, :] = HG[-alpha:, :]
    r_A[-alpha:, :] = HG[:alpha, :]
    a = np.arange(H_w)/(H_w-1)
    # r_A(alpha+1:end-alpha, 1) = (1-a)*r_A(alpha,1) + a*r_A(end-alpha+1,1)
    r_A[alpha:-alpha, 0] = (1-a)*r_A[alpha-1, 0] + a*r_A[-alpha, 0]
    # r_A(alpha+1:end-alpha, end) = (1-a)*r_A(alpha,end) + a*r_A(end-alpha+1,end)
    r_A[alpha:-alpha, -1] = (1-a)*r_A[alpha-1, -1] + a*r_A[-alpha, -1]

    r_B = np.zeros((H, alpha*2+W_w))
    r_B[:, :alpha] = HG[:, -alpha:]
    r_B[:, -alpha:] = HG[:, :alpha]
    a = np.arange(W_w)/(W_w-1)
    r_B[0, alpha:-alpha] = (1-a)*r_B[0, alpha-1] + a*r_B[0, -alpha]
    r_B[-1, alpha:-alpha] = (1-a)*r_B[-1, alpha-1] + a*r_B[-1, -alpha]

    if alpha == 1:
        A2 = solve_min_laplacian(r_A[alpha-1:, :])
        B2 = solve_min_laplacian(r_B[:, alpha-1:])
        r_A[alpha-1:, :] = A2
        r_B[:, alpha-1:] = B2
    else:
        A2 = solve_min_laplacian(r_A[alpha-1:-alpha+1, :])
        r_A[alpha-1:-alpha+1, :] = A2
        B2 = solve_min_laplacian(r_B[:, alpha-1:-alpha+1])
        r_B[:, alpha-1:-alpha+1] = B2
    A = r_A
    B = r_B

    r_C = np.zeros((alpha*2+H_w, alpha*2+W_w))
    r_C[:alpha, :] = B[-alpha:, :]
    r_C[-alpha:, :] = B[:alpha, :]
    r_C[:, :alpha] = A[:, -alpha:]
    r_C[:, -alpha:] = A[:, :alpha]

    if alpha == 1:
        C2 = C2 = solve_min_laplacian(r_C[alpha-1:, alpha-1:])
        r_C[alpha-1:, alpha-1:] = C2
    else:
        C2 = solve_min_laplacian(r_C[alpha-1:-alpha+1, alpha-1:-alpha+1])
        r_C[alpha-1:-alpha+1, alpha-1:-alpha+1] = C2
    C = r_C
    # return C
    A = A[alpha-1:-alpha-1, :]
    B = B[:, alpha:-alpha]
    C = C[alpha:-alpha, alpha:-alpha]
    ret = np.vstack((np.hstack((frame, B)), np.hstack((A, C))))
    return ret


def solve_min_laplacian(boundary_video):
    (H, W) = np.shape(boundary_video)

    # Laplacian
    f = np.zeros((H, W))
    # boundary image contains image intensities at boundaries
    boundary_video[1:-1, 1:-1] = 0
    j = np.arange(2, H)-1
    k = np.arange(2, W)-1
    f_bp = np.zeros((H, W))
    f_bp[np.ix_(j, k)] = -4*boundary_video[np.ix_(j, k)] + boundary_video[np.ix_(j, k+1)] + boundary_video[np.ix_(j, k-1)] + boundary_video[np.ix_(j-1, k)] + boundary_video[np.ix_(j+1, k)]
    
    del(j, k)
    f1 = f - f_bp  # subtract boundary points contribution
    del(f_bp, f)

    # DST Sine Transform algo starts here
    f2 = f1[1:-1,1:-1]
    del(f1)

    # compute sine tranform
    if f2.shape[1] == 1:
        tt = fftpack.dst(f2, type=1, axis=0)/2
    else:
        tt = fftpack.dst(f2, type=1)/2

    if tt.shape[0] == 1:
        f2sin = np.transpose(fftpack.dst(np.transpose(tt), type=1, axis=0)/2)
    else:
        f2sin = np.transpose(fftpack.dst(np.transpose(tt), type=1)/2) 
    del(f2)

    # compute Eigen Values
    [x, y] = np.meshgrid(np.arange(1, W-1), np.arange(1, H-1))
    denom = (2*np.cos(np.pi*x/(W-1))-2) + (2*np.cos(np.pi*y/(H-1)) - 2)

    # divide
    f3 = f2sin/denom
    del(f2sin, x, y)

    # compute Inverse Sine Transform
    if f3.shape[0] == 1:
        tt = fftpack.idst(f3*2, type=1, axis=1)/(2*(f3.shape[1]+1))
    else:
        tt = fftpack.idst(f3*2, type=1, axis=0)/(2*(f3.shape[0]+1))
    del(f3)
    if tt.shape[1] == 1:
        frame_tt = np.transpose(fftpack.idst(np.transpose(tt)*2, type=1)/(2*(tt.shape[0]+1)))
    else:
        frame_tt = np.transpose(fftpack.idst(np.transpose(tt)*2, type=1, axis=0)/(2*(tt.shape[1]+1)))
    del(tt)

    # put solution in inner points; outer points obtained from boundary image
    frame_direct = boundary_video
    frame_direct[1:-1, 1:-1] = 0
    frame_direct[1:-1, 1:-1] = frame_tt
    return frame_direct




def fspecial_average(hsize=3):
    """Smoothing filter"""
    return np.ones((hsize, hsize))/hsize**2


def fspecial_disk(radius):
    """Disk filter"""
    raise(NotImplemented)
    rad = 0.6
    crad = np.ceil(rad-0.5)
    [x, y] = np.meshgrid(np.arange(-crad, crad+1), np.arange(-crad, crad+1))
    maxxy = np.zeros(x.shape)
    maxxy[abs(x) >= abs(y)] = abs(x)[abs(x) >= abs(y)]
    maxxy[abs(y) >= abs(x)] = abs(y)[abs(y) >= abs(x)]
    minxy = np.zeros(x.shape)
    minxy[abs(x) <= abs(y)] = abs(x)[abs(x) <= abs(y)]
    minxy[abs(y) <= abs(x)] = abs(y)[abs(y) <= abs(x)]
    m1 = (rad**2 <  (maxxy+0.5)**2 + (minxy-0.5)**2)*(minxy-0.5) +\
         (rad**2 >= (maxxy+0.5)**2 + (minxy-0.5)**2)*\
         np.sqrt((rad**2 + 0j) - (maxxy + 0.5)**2)
    m2 = (rad**2 >  (maxxy-0.5)**2 + (minxy+0.5)**2)*(minxy+0.5) +\
         (rad**2 <= (maxxy-0.5)**2 + (minxy+0.5)**2)*\
         np.sqrt((rad**2 + 0j) - (maxxy - 0.5)**2)
    h = None
    return h


def fspecial_gaussian(hsize, sigma):
    hsize = [hsize, hsize]
    siz = [(hsize[0]-1.0)/2.0, (hsize[1]-1.0)/2.0]
    std = sigma
    [x, y] = np.meshgrid(np.arange(-siz[1], siz[1]+1), np.arange(-siz[0], siz[0]+1))
    arg = -(x*x + y*y)/(2*std*std)
    h = np.exp(arg)
    h[h < scipy.finfo(float).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h = h/sumh
    return h


def fspecial_laplacian(alpha):
    alpha = max([0, min([alpha,1])])
    h1 = alpha/(alpha+1)
    h2 = (1-alpha)/(alpha+1)
    h = [[h1, h2, h1], [h2, -4/(alpha+1), h2], [h1, h2, h1]]
    h = np.array(h)
    return h


def fspecial_log(hsize, sigma):
    raise(NotImplemented)


def fspecial_motion(motion_len, theta):
    raise(NotImplemented)


def fspecial_prewitt():
    return np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])


def fspecial_sobel():
    return np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])







def fspecial(filter_type, *args, **kwargs):
    '''
    python code from:
    https://github.com/ronaldosena/imagens-medicas-2/blob/40171a6c259edec7827a6693a93955de2bd39e76/Aulas/aula_2_-_uniform_filter/matlab_fspecial.py
    '''
    if filter_type == 'average':
        output=fspecial_average(*args, **kwargs)
        inverse = np.fft.ifft2(output)
        absolute = np.absolute(inverse)
        result= absolute.astype(np.uint8)
        return result
        
        
    if filter_type == 'disk':
        output=fspecial_disk(*args, **kwargs)
        inverse = np.fft.ifft2(output)
        absolute = np.absolute(inverse)
        result= absolute.astype(np.uint8)
        return result
    
    
    if filter_type == 'gaussian':
         output=fspecial_gaussian(*args, **kwargs)
         inverse = np.fft.ifft2(output)
         absolute = np.absolute(inverse)
         result= absolute.astype(np.uint8)
         return result
     
        
    if filter_type == 'laplacian':
         output=fspecial_laplacian(*args, **kwargs)
         inverse = np.fft.ifft2(output)
         absolute = np.absolute(inverse)
         result= absolute.astype(np.uint8)
         return result
     
        
    if filter_type == 'log':
         output=fspecial_log(*args, **kwargs)
         inverse = np.fft.ifft2(output)
         absolute = np.absolute(inverse)
         result= absolute.astype(np.uint8)
         return result
     
        
    if filter_type == 'motion':
         output=fspecial_motion(*args, **kwargs)
         inverse = np.fft.ifft2(output)
         absolute = np.absolute(inverse)
         result= absolute.astype(np.uint8)
         return result
     
        
    if filter_type == 'prewitt':
         output=fspecial_prewitt(*args, **kwargs)
         inverse = np.fft.ifft2(output)
         absolute = np.absolute(inverse)
         result= absolute.astype(np.uint8)
         return result
    if filter_type == 'sobel':
         output=fspecial_sobel(*args, **kwargs)
         inverse = np.fft.ifft2(output)
         absolute = np.absolute(inverse)
         result= absolute.astype(np.uint8)
         return result



frame=fspecial('gaussian',5,1)


if __name__ == '__main__':
    a = opt_fft_size([111])
    print(a)

    print(fspecial('gaussian', 5, 1))


    
while True :
	# grab the current frame
	(grabbed, image) = vs.read()
	frame=fspecial('gaussian',5,1)
    
    
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break
	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break    

