import torch
import torch.nn as nn
import numpy as np
from torch.distributions.normal import Normal
from mixture_linear import MixtureLinear
from distributions.gumbel import gumbel_softmax_sample

class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_classes, learnable_prior=False):
        super(VAE, self).__init__()

        self.K = n_classes
        self.is_prior_learnable = learnable_prior
        self.p_mean_linear = MixtureLinear(input_size, input_size, n_classes)
        self.p_var_linear = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.Softplus(),
            nn.Linear(input_size, input_size),
            nn.Softplus()
        )

        self.q_logits_net = nn.Sequential(
            nn.Linear(2*input_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, n_classes)
        )

        self.p_logits = nn.Parameter(torch.ones(self.K)) if self.is_prior_learnable else torch.ones(self.K)

    def forward(self, x_t, x_t_lag, temperature):
        z_t, q = self.q_sample(x_t, x_t_lag, temperature)
        device = x_t.device

        log_p = self.p_log_proba(x_t, x_t_lag, z_t)
        log_q = torch.log(q+1e-20)
        log_prior = self.prior_probas(log=True).to(device)
        KL = (q*(log_q - log_prior)).sum(axis=1)
        elbo = (log_p - KL).mean()
        return elbo

    def _independent_normal_log_prob(self, x, mu, var):
        n,m = x.shape
        normal = Normal(loc=mu.flatten(), scale=torch.sqrt(var).flatten())
        log_proba = normal.log_prob(x.flatten()).reshape((n,m)).sum(axis=1)
        return log_proba

    def prior_probas(self, log=False):
        prior_p = nn.functional.softmax(self.p_logits)
        return torch.log(prior_p) if log else prior_p

    def p_sample(self, x_t_lag):
        n,m = x_t_lag.shape
        device = x_t_lag.device

        prior_p = self.prior_probas().cpu().numpy()
        z_t = torch.tensor(np.random.choice(np.arange(self.K), size=(n,), p=prior_p), device=device)
        z_t = torch.nn.functional.one_hot(z_t, num_classes=self.K).float()

        (mu, Pi), var = self.p_mean_linear(x_t_lag, z_t, return_Pi=True), self.p_var_linear(x_t_lag) + 1e-10
        
        normal = Normal(loc=mu.flatten(), scale=torch.sqrt(var).flatten())
        x_t = normal.sample().reshape((n,m))

        return x_t, z_t, mu, Pi

    def p_log_proba(self, x_t, x_t_lag, z_t):
        mu = self.p_mean_linear(x_t_lag, z_t)
        var = self.p_var_linear(x_t_lag)

        x_t_log_proba = self._independent_normal_log_prob(x_t, mu, var)
        return x_t_log_proba

    def q_sample(self, x_t, x_t_lag, temperature):
        x = torch.cat([x_t, x_t_lag], axis=1)
        q_logits = self.q_logits_net(x)
        sample_z = gumbel_softmax_sample(q_logits, temperature, x.device)
        q = nn.functional.softmax(q_logits, dim=-1)
        return sample_z, q