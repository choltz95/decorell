# keras-decorrelated-batch-norm
keras-decorrelated-batch-norm

A Keras layer which implements batch whitening for neural networks as described in [1] I extend the base zca implementation with other whitening methods described in [1]. ZCA and cholesky-based approaches are recommended over pca-based approaches due to the stochastic axis swapping issue. ZCA-Cor is recommended by [2] over cholesky whitening as it is guarunteed to produce whitened variables that are maximally similar to the input. I also include an experimental framework taken from [shaoanlu's repo](https://github.com/shaoanlu/GroupNormalization-keras).

SVD computation done on cpu due to slow computation on gpu.

todo: fix renormalization component, implement group whitening, report timing experiments.

[1] [Huang et al., Decorrelated Batch Normalization, 2018](https://arxiv.org/abs/1804.08450)  
[2] [Kessy et al., Optimal whitening and decorrelation, 2015](https://arxiv.org/abs/1512.00809)  
[3] [Santurkar et al., How Does Batch Normalization Help Optimization? (No, It Is Not About Internal Covariate Shift), 2018](https://arxiv.org/abs/1805.11604)
