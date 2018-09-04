from keras.engine import Layer, InputSpec
from keras.initializers import Identity
from keras import initializers
from keras import regularizers
from keras import constraints
from keras import backend as K
from keras.backend import tf as ktf

import numpy as np

from keras.utils.generic_utils import get_custom_objects


class DecorrelatedBatchNorm(Layer):
    """Instance normalization layer (Lei Huang et al. 2018).
    Normalize the activations of the previous layer at each step,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1 and whitens activations.
    # Arguments
        groups: Integer, the number of groups for Group Normalization.
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `InstanceNormalization`.
            Setting `axis=None` will normalize all values in each instance of the batch.
            Axis 0 is the batch dimension. `axis` cannot be set to 0 to avoid errors.
        epsilon: Small float added to variance to avoid dividing by zero.
        momentum: momentum in the computation of the
            exponential average of the mean and covariance
            of the data, for normalization.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
        moving_mean_initializer: Initializer for the moving mean.
        moving_covariance_initializer: Initializer for the moving covariance.
        batch_size: Required paramter for batch size.
        group_size: Requried parameter for group whitening.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as input.
    # References
        - [Decorrelated Batch Normalization](https://arxiv.org/abs/1804.08450)
    """

    def __init__(self,
                  momentum=0.99,
                  epsilon=1e-3,
                  moving_mean_initializer='zeros',
                  decomposition='cholesky',
                  group=16,
                  renorm=False,
                  center=True,
                  scale=True,
                  moving_cov_initializer=Identity(),
                  beta_initializer='zeros',
                  gamma_initializer='ones',
                  beta_regularizer=None,
                  gamma_regularizer=None,
                  beta_constraint=None,
                  gamma_constraint=None,
                  **kwargs):
        assert decomposition in ['cholesky', 'zca', 'zca-cor', 'pca', 'pca-cor']
        super(DecorrelatedBatchNorm, self).__init__(**kwargs)
        self.supports_masking = True
        self.momentum = momentum
        self.epsilon = epsilon
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_cov_initializer = initializers.get(moving_cov_initializer)
        self.axis = -1
        self.renorm = renorm
        self.group = group
        self.decomposition = decomposition
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')
        shape = (dim, )
        self.moving_mean = self.add_weight(
            (dim, 1),
            name='moving_mean',
            initializer=self.moving_mean_initializer,
            trainable=False)
        self.moving_cov = self.add_weight(
            (dim, dim),
            name='moving_variance',
            initializer=self.moving_cov_initializer,
            trainable=False)
        
        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
            
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        _, w, h, c = input_shape
        bs = K.shape(inputs)[0]
        
        #if c < self.group:
        #    raise ValueError('Input channels should be larger than group size' +
        #                     '; Received input channels: ' + str(c) +
        #                     '; Group size: ' + str(self.group)
        #                    )
        #x = K.reshape(inputs, (batch_size, h, w, self.group, c // self.group))
        
        x_t = ktf.transpose(inputs, (3, 0, 1, 2))
        #x_t = ktf.transpose(x, (3, 4, 0, 1, 2))

        # BxCxHxW -> CxB*H*W
        x_flat = ktf.reshape(x_t, (c, -1))

        # Covariance
        m = ktf.reduce_mean(x_flat, axis=1, keepdims=True)
        m = K.in_train_phase(m, self.moving_mean)
        f = x_flat - m


        if self.decomposition == 'cholesky':
            def get_inv_sqrt(ff):
                sqrt = ktf.cholesky(ff)
                inv_sqrt = ktf.matrix_triangular_solve(sqrt, ktf.eye(c))
                return sqrt, inv_sqrt
        elif self.decomposition in ['zca','zca-cor']:
            def get_inv_sqrt(ff):
                with ktf.device('/cpu:0'):
                    S, U, _ = ktf.svd(ff + ktf.eye(c)*self.epsilon, full_matrices=True)
                D = ktf.diag(ktf.pow(S, -0.5))
                inv_sqrt = ktf.matmul(ktf.matmul(U, D), U, transpose_b=True)
                D = ktf.diag(ktf.pow(S, 0.5))
                sqrt =  ktf.matmul(ktf.matmul(U, D), U, transpose_b=True)
                return sqrt, inv_sqrt
        elif self.decomposition in ['pca','pca-cor']:
             def get_inv_sqrt(ff):
                with ktf.device('/cpu:0'):
                    S, U, _ = ktf.svd(ff + ktf.eye(c)*self.epsilon, full_matrices=True)
                D = ktf.diag(ktf.pow(S, -0.5))
                inv_sqrt = ktf.matmul(D, U, transpose_b=True)
                D = ktf.diag(ktf.pow(S, 0.5))
                sqrt =  ktf.matmul(D, U, transpose_b=True)
                return sqrt, inv_sqrt
        else:
            assert False

        def train():
            ff_apr = ktf.matmul(f, f, transpose_b=True) / (ktf.cast(bs*w*h, ktf.float32) - 1.)
            if self.decomposition in ['pca-cor','zca-cor']:
              dinv = ktf.diag(ktf.sqrt(ktf.diag_part(ff_apr)))
              ff_apr = ktf.matmul(ktf.matmul(dinv,ff_apr),ktf.matrix_inverse(dinv), transpose_b=True)
            self.add_update([K.moving_average_update(self.moving_mean,
                                                     m,
                                                     self.momentum),
                             K.moving_average_update(self.moving_cov,
                                                     ff_apr,
                                                     self.momentum)],
                             inputs) 
            ff_apr_shrinked = (1 - self.epsilon) * ff_apr + ktf.eye(c) * self.epsilon
            
            if self.renorm:
                l, l_inv = get_inv_sqrt(ff_apr_shrinked)
                ff_mov =  (1 - self.epsilon) * self.moving_cov + ktf.eye(c) * self.epsilon
                _, l_mov_inverse = get_inv_sqrt(ff_mov)
                l_ndiff = K.stop_gradient(l)
                return ktf.matmul(ktf.matmul(l_mov_inverse, l_ndiff), l_inv)
               
            return get_inv_sqrt(ff_apr_shrinked)[1]

        def test():
            ff_mov = (1 - self.epsilon) * self.moving_cov + ktf.eye(c) * self.epsilon
            return get_inv_sqrt(ff_mov)[1]
        
        inv_sqrt = K.in_train_phase(train, test)
        f_hat = ktf.matmul(inv_sqrt, f)

        decorelated = K.reshape(f_hat, [c, bs, w, h])
        decorelated = ktf.transpose(decorelated, [1, 2, 3, 0])
        
        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]
        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            decorelated = decorelated * broadcast_gamma
        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            decorelated = decorelated + broadcast_beta

        return decorelated

    def get_config(self):
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'moving_mean_initializer': initializers.serialize(self.moving_mean_initializer),
            'moving_variance_initializer': initializers.serialize(self.moving_cov_initializer),
            'center': self.center,
            'scale': self.scale,
            'group': self.group,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(DecorrelatedBatchNorm, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
