import numpy as np
from nndl.layers import *
import pdb

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""

def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  pad = conv_param['pad']
  stride = conv_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the forward pass of a convolutional neural network.
  #   Store the output as 'out'.
  #   Hint: to pad the array, you can use the function np.pad.
  # ================================================================ #

  #quadruple for loops - iterate over each sample, filter, and height/width axes (numpy multiple axes)
  #print("w shape", w.shape)

  #using https://cs231n.github.io/convolutional-networks/#conv as a resource, although it's still not very clear 


  N, C, H, W = x.shape
  F, C, HH, WW = w.shape 

  #pad, stride = conv_param['pad'], conv_param['stride'] 

  #use formulas given above 
  HPrime = 1 + (H + 2 * pad - HH) // stride # `//`to force int 
  WPrime = 1 + (W + 2 * pad - WW) // stride 

  #output shape -- fill w/ 0s 
  out = np.zeros((N, F, HPrime, WPrime)) 

  #pad x here, to fit the new dimensions 
  x = np.pad(x, pad_width = [(0, 0), (0, 0), (pad, pad), (pad, pad)], mode = "constant")


  for i in np.arange(N): #samples
    for j in np.arange(F): #filters 
      for h in np.arange(HPrime): #height 
        for w_i in np.arange(WPrime): #width 

          xx = x[i, :, h*stride : h*stride + HH, w_i*stride : w_i*stride + WW]  

          #filter specific terms 
          ww = w[j] #get the data in the first dimension (should be a 3d tensor) 
          
          
          #print("ww shape", ww.shape) 

          beta = b[j] 

          out[i, j, h, w_i] = np.sum(xx*ww) + beta #note * is element wise here 

  
    
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #
    
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None

  N, F, out_height, out_width = dout.shape
  x, w, b, conv_param = cache
  
  stride, pad = [conv_param['stride'], conv_param['pad']]
  xpad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
  num_filts, _, f_height, f_width = w.shape

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the backward pass of a convolutional neural network.
  #   Calculate the gradients: dx, dw, and db.
  # ================================================================ #
    
  #we also use convolution here!
  #using https://medium.com/@mayank.utexas/backpropagation-for-convolution-with-strides-8137e4fc2710 from Piazza @437
    
  dx = np.zeros(shape=x.shape) 
  dw = np.zeros_like(w) #np.zeros_like(x) = np.zeros(shape=x.shape)
  db = np.zeros_like(b) 
  
  N, C, H, W = x.shape
  HPrime = 1 + (H - f_height) // stride 
  WPrime = 1 + (W - f_width) // stride 

  
  for i in np.arange(N): 
    for j in np.arange(num_filts): 
      for k in np.arange(HPrime): 
        for l in np.arange(WPrime): 
          
          sh, eh = k*stride, k*stride + f_height #start_height and end_height, abbreviated 
          sw, ew = l*stride, l*stride + f_width
          
          xx = x[i, :, sh:eh, sw:ew]
          ww = w[j]  
          cur_dout = dout[i, j, k, l]
          
          dx[i, :, sh:eh, sw:ew] += ww * cur_dout
          dw[j] += xx * cur_dout 
  
  #handle db 
  for i in np.arange(N): 
    for j in np.arange(num_filts): 
      if i == 0: 
        db[j] += np.sum(dout[:, j, :, :])
  
  dx = dx[:, :, pad:-pad, pad:-pad] #just guessing at this point why are these assignments so stupidly hard


  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #

  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  
  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling forward pass.
  # ================================================================ #
  pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
  
  N, C, H, W = x.shape
  HPrime = 1 + (H - pool_height) // stride 
  WPrime = 1 + (W - pool_width) // stride 
  out = np.zeros((N, C, HPrime, WPrime)) 

  
  for i in np.arange(N): 
    for j in np.arange(C): 
      for h in np.arange(HPrime): 
        for w in np.arange(WPrime): 
          sh, eh = h*stride, h*stride + pool_height #start_height and end_height, abbreviated 
          sw, ew = w*stride, w*stride + pool_width
          
          out[i, j, h, w] = np.max(x[i, j, sh:eh, sw:ew])
          
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 
  cache = (x, pool_param)
  return out, cache

def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  x, pool_param = cache
  pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling backward pass.
  # ================================================================ #
  N, C, H, W = x.shape
  HPrime = 1 + (H - pool_height) // stride 
  WPrime = 1 + (W - pool_width) // stride 
  
  dx = np.zeros_like(x) 
  
  for i in np.arange(N): 
      for j in np.arange(C): 
        for h in np.arange(HPrime): 
          for w in np.arange(WPrime): 
            sh, eh = h*stride, h*stride + pool_height
            sw, ew = w*stride, w*stride + pool_width

            xx = x[i, j, sh:eh, sw:ew]
            ddout = dout[i, j, h, w]
                        
            dx[i, j, sh:eh, sw:ew] += ddout * (xx == np.max(xx)) #last term is the mask
  
  

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm forward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  
  N, C, H, W = x.shape
  
  x = x.reshape(N*H*W, C)
  
  out, cache = batchnorm_forward(x, gamma, beta, bn_param) 
  
  out = out.T.reshape(C, N, H, W).transpose(1, 0, 2, 3) #swap second axis with first, and vice versa 
  
#   print(out.shape)
#   print(out.T.shape)

#couldn't figure out how to get rid of the out.T, pretty sure there's a way but transpose with multiple parameters is mega confusing sad
  

    

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm backward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  
  N, C, H, W = dout.shape 
  
  dout = dout.transpose(1, 0, 2, 3) #equivalent to swapaxes(1, 0) I think 
  dout = dout.reshape(C, N*H*W).T
  
  #https://numpy.org/doc/stable/reference/generated/numpy.transpose.html
  
  #not really sure why the bottom doesn't work .. maybe it's b/c I'm swapping columns when I should be swapping rows 
  
#   dout = dout.transpose(3, 2, 0, 1)
#   dout = dout.reshape(N*H*W, C)
  
  dx, dgamma, dbeta = batchnorm_backward(dout, cache) 
  
  dx = dx.reshape(N, C, H, W)

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx, dgamma, dbeta