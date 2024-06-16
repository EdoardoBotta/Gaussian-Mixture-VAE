# Gaussian-Mixture-VAE
Implementation of a Variational Autoencoder with Categorical latent variables.

## Model
We consider a time series $(x)_t$ whose transitions are described by a mixture of $K$ Gaussians, 
```math 
z_t \sim Categorical(p_1, p_2, \dots, p_K)
```
```math 
x_t | x_{t-1} \sim Normal \left( \Pi_{z_t}x_{t-1}, \Sigma \right)
```
Intuitively, each of the $K$ classes defined by the is associated a matrix $Pi_{k}, \ k \leq K$ and the mean of $x_t$ evolves linearly according to the chosen class. 

## Variational inference
We use to Variational Inference to estimate the parameters $(p_1, p_2, \dots, p_K), (\Pi_1, \Pi_2, \dots, Pi_k), \Sigma$ of the above model. In particular, we maximize the variational lower bound (ELBO) on the log-likelihood, which takes the form
```math 
\mathbb E_{q_{\theta}(z_t|x_t, x_{t-1})} \left[ \log p_{\theta}(x_t | x_{t-1}, z_t) + \log p_{\theta}(z_t) - \log q_{\phi} (z_t | x_t, x_{t-1}) \right],
```
where $p_{\theta}$ and $q_{\phi}$ are parametrized by neural networks.

## Tackling non-differentiability: Gumbel approximation
One of the main challenges in using discrete latents in VAEs is the inherent non-differentiability of the resulting PMF. We solve this issue by approximating the one-hot encoded vectors $z_t$ by a Gumbel distribution, whose temperature is decreased over time during training. 
