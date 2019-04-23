from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # initialization
        self.params['W1'] = weight_scale*np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = weight_scale*np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        A1, cache_A1 = affine_relu_forward(X, self.params['W1'], self.params['b1'])
        Z2, cache_Z2 = affine_forward(A1, self.params['W2'], self.params['b2'])
        scores = Z2

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        if y is not None:
            # compute the loss
            loss, dZ2 = softmax_loss(scores, y) 
            # add regularization
            loss += 0.5*self.reg*(np.sum(self.params['W1']**2) + np.sum(self.params['W2']**2))
            
            # compute the backward pass
            dA1, dW2, db2 = affine_backward(dZ2, cache_Z2)
            dX, dW1, db1 = affine_relu_backward(dA1, cache_A1)
            
            # store the gradients (with regularization)
            grads['W1'] = dW1 + self.reg*self.params['W1']
            grads['b1'] = db1
            grads['W2'] = dW2 + self.reg*self.params['W2']
            grads['b2'] = db2
             
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # initialization
        hidden_dim_prev = input_dim
        
        for i in range(1, self.num_layers):
            # linear layer parameters
            self.params['W' + str(i)] = weight_scale*np.random.randn(hidden_dim_prev, hidden_dims[i-1])
            self.params['b' + str(i)] = np.zeros(hidden_dims[i-1])
                        
            # normalization layer parameters
            if self.normalization:
                self.params['gamma' + str(i)] = np.ones(hidden_dims[i-1])
                self.params['beta' + str(i)] = np.zeros(hidden_dims[i-1])
                
            hidden_dim_prev = hidden_dims[i-1]
            
        # last layer
        i += 1
        self.params['W' + str(i)] = weight_scale*np.random.randn(hidden_dim_prev, num_classes)
        self.params['b' + str(i)] = np.zeros(num_classes)
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # store the caches
        caches = {}
        
        # initialization
        A = X
        
        for i in range(1, self.num_layers):
            # use batch normalization
            if self.normalization:
                A, cache = affine_bn_relu_forward(A, self.params['W' + str(i)], self.params['b' + str(i)], self.params['gamma' + str(i)], self.params['beta' + str(i)], self.bn_params[i-1], self.normalization)  
                
            else:
                A, cache = affine_relu_forward(A, self.params['W' + str(i)], self.params['b' + str(i)]) 
            
            # use dropout
            if self.use_dropout:
                A, do_cache = dropout_forward(A, self.dropout_param)
                # add dropout cache to the cache
                cache = (do_cache, cache)
            
            # store the cache
            caches[i] = cache
        
        # last layer: compute the scores
        i += 1
        scores, cache = affine_forward(A, self.params['W' + str(i)], self.params['b' + str(i)])
        caches[i] = cache
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        if y is not None:
            # compute the loss
            loss, dZ = softmax_loss(scores, y) 
            
            # last layer: backward step
            dA, dW, db = affine_backward(dZ, caches[i])
            
            # store the gradients (with regularization)
            grads['W' + str(i)] = dW + self.reg*self.params['W' + str(i)]
            grads['b' + str(i)] = db

            # add regularization
            loss += 0.5*self.reg*np.sum(self.params['W' + str(i)]**2)
                
            # backward pass
            for i in range(self.num_layers-1, 0, -1):
                
                cache = caches[i]
                
                # use dropout
                if self.use_dropout:
                    # unpack the cache
                    (do_cache, cache) = cache
                    dA = dropout_backward(dA, do_cache)
                
                # use batch normalization
                if self.normalization:
                    dA, dW, db, dgamma, dbeta = affine_bn_relu_backward(dA, cache, self.normalization)   
                    # store the gradients
                    grads['gamma' + str(i)] = dgamma
                    grads['beta' + str(i)] = dbeta
                else:   
                    # compute the gradients
                    dA, dW, db = affine_relu_backward(dA, cache) 
            
                # store the gradients (with regularization)
                grads['W' + str(i)] = dW + self.reg*self.params['W' + str(i)]
                grads['b' + str(i)] = db
                
                # add regularization
                loss += 0.5*self.reg*np.sum(self.params['W' + str(i)]**2)
            
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

    
def affine_bn_relu_forward(x, w, b, gamma, beta, bn_params, normalization):
    """
    Convenience layer that performs an affine transform followed by Batch Normalization followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    - gamma, beta: Weights for the batch norm layer
    - bn_params: dictionnary storing the running averages for batch normalization
    - normalization: What type of normalization the network should use. Valid values
      are "batchnorm" or  "layernorm"
          
    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    z, fc_cache = affine_forward(x, w, b)
        
    if normalization == 'batchnorm':
        a, bn_cache = batchnorm_forward(z, gamma, beta, bn_params)
    elif normalization == 'layernorm':
        a, bn_cache = layernorm_forward(z, gamma, beta, bn_params)

    out, relu_cache = relu_forward(a)
    cache = (fc_cache, bn_cache, relu_cache)
    return out, cache


def affine_bn_relu_backward(dout, cache, normalization):
    """
    Backward pass for the affine-bn-relu convenience layer
    """
    fc_cache, bn_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
            
    if normalization == 'batchnorm':
        dz, dgamma, dbeta = batchnorm_backward(da, bn_cache)
    elif normalization == 'layernorm':
        dz, dgamma, dbeta = layernorm_backward(da, bn_cache)

    dx, dw, db = affine_backward(dz, fc_cache)
    return dx, dw, db, dgamma, dbeta