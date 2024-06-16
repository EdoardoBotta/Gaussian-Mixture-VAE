# Gaussian-Mixture-VAE
Implementation of a Variational Autoencoder with Categorical Latent variables.

## Model
We consider a time series $(x)_t$ whose transitions are described by an auto-regressive mixture of $K$ Gaussians. The generative process can be described hierarchically as follows
```math 
z_t \sim Categorical(p_1, p_2, \dots, p_K)
```
```math 
x_t | x_{t-1} \sim Normal \left( \Pi_{z_t}x_{t-1}, \Sigma \right)
```
Intuitively, at time $t$ the series evolves as a linear autoregressively according to one of $K$ choices of parameter matrices $(Pi)_k$. 

## Variational inference
We leverage Variational Inference to estimate the parameters $(p_1, p_2, \dots, p_K), (\Pi_1, \Pi_2, \dots, Pi_k), \Sigma$ of the above model. In particular, we maximize the variational lower bound (ELBO) on the log-likelihood, which takes the form
```math 
\mathbb E_{q_{\theta}(z_t|x_t, x_{t-1})} \left[ \log p_{\theta}(x_t | x_{t-1}, z_t) + \log p_{\theta}(z_t) - \log q_{\phi} (z_t | x_t, x_{t-1}) \right],
```
where $p_{\theta}$ and $q_{\phi}$ are parametrized by feed-forward neural networks.

## Gumbel approximation
One of the main challenges in using discrete latents in VAEs is the inherent non-differentiability of the resulting PMF that appear in the objective. We solve this issue by approximating the one-hot encoded vectors $z_t$ by a Gumbel distribution. As training proceeds, the temperature of the Gumbel distributino is aslowly annalead to 0 and the approximation becomes progressively more accurate. 
