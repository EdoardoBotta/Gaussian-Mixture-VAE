# Gaussian-Mixture-VAE
This repository contains a minimal PyTorch implementation of a Variational Autoencoder with Categorical Latent variables.

## Model
We consider data distributed as a time series $(x)_t$ whose transitions are described by an auto-regressive mixture of $K$ Gaussians. The generative process can be described hierarchically as follows
```math 
z_t \sim Categorical(p_1, p_2, \dots, p_K)
```
```math 
x_t | x_{t-1} \sim Normal \left( \Pi_{z_t}x_{t-1}, \Sigma \right)
```
Intuitively, at time $t$ the series evolves as a linear autoregressive model according to one of $K$ choices of parameter matrix $(\Pi)_k$. 

## Variational inference
We leverage Variational Inference to estimate the parameters $(p_1, p_2, \dots, p_K), (\Pi_1, \Pi_2, \dots, Pi_k), \Sigma$ of the above model. 

Specifically, this is achieved by maximizing the variational lower bound (ELBO) on the log-likelihood of the observed data. 

The resulting optimization problem takes the form
```math 
\max_{\theta, \phi} \mathbb E_{q_{\theta}(z_t|x_t, x_{t-1})} \left[ \log p_{\theta}(x_t | x_{t-1}, z_t) + \log p_{\theta}(z_t) - \log q_{\phi} (z_t | x_t, x_{t-1}) \right],
```
where $p_{\theta}$ is as described in the generative model and $q_{\phi}$ is a variational distribution parametrized by a feed-forward neural network.

## Gumbel approximation
One of the main challenges in using discrete latents in VAEs is the inherent non-differentiability of the resulting PMF that appear in the objective. We solve this issue by approximating the one-hot encoded vectors $z_t$ by a Gumbel distribution. As training proceeds, the temperature of the Gumbel distributino is slowly annalead to 0 and the approximation becomes progressively more accurate. 

## References
* [Tutorial: Categorical Variational Autoencoders using Gumbel-Softmax](https://blog.evjang.com/2016/11/tutorial-categorical-variational.html) by Eric Jang
* [Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144) by Eric Jang, Shixiang Gu, Ben Poole
* [Deep Unsupervised Clustering with Gaussian Mixture Variational Autoencoders](https://arxiv.org/abs/1611.02648) by Nat Dilokthanakul, Pedro A.M. Mediano, Marta Garnelo, Matthew C.H. Lee, Hugh Salimbeni, Kai Arulkumaran, Murray Shanahan
