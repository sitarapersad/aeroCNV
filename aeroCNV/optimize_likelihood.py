import scipy
import torch
import time

import numpy as np
import pandas as pd
import torch

from .logging_config import log

def compute_Poisson_summaries(observations, gamma, modifiers):
    '''
    Compute summaries for Poisson distribution for computing the likelihood of the observations.
    For a given gene and cell, c, the negative log likelihood of the observation, x, is given by:
        NLL^(k)(μ; x) = -∑(c=1 to N) [γ^(k)_c (x_c log(μ^(k)) + x_c log(modifier_c) - μ modifier_c - log(x_c!))]
    where:
        k is the state of the HMM
        γ^(k)_c is the probability of cell c being in state k
        x_c is the observation in cell c
        μ^(k) is the mean of the Poisson distribution in state k
        modifier_c is the modifier for cell c
        N is the number of cells

    Summaries computed are as follows:
    Summary[0][k] = 1 * t array containing sum_{cells} gamma[cell,k,t]*x_t*log(mods)
    Summary[1][k] = 1 * t array containing sum_{cells} gamma[cell,k,t]*log(x_t!)
    Summary[2][k] = 1 * t array containing sum_{cells} gamma[cell,k,t]*mods
    Summary[3][k] = 1 * t array containing sum_{cells} gamma[cell,k,t]*x_t

    :param observations: (torch.Tensor) - N x T tensor of observations
    :param gamma: (torch.Tensor) - N x K x T tensor of probabilities of cells being in states
    :param modifiers: (torch.Tensor) - N x 1 list of modifiers for each cell
    :param n_states: (int) - Number of states in the HMM

    :return: (list) - List of summaries for the Poisson distribution
    '''

    # We are not interested in the first state; this is the Reset state
    gamma = gamma[:, 1:, :]
    n_states = gamma.shape[1]
    x = observations.clone().double()
    modifiers = torch.DoubleTensor(modifiers).view(-1,1,1).repeat(1,n_states,1)

    # --- Summary[0][k] = k * t array containing sum_{cells} gamma[cell,k,t]*x_t*log(mods)
    summ_0 = torch.mul(torch.mul(x, gamma), torch.log(modifiers)).sum(0)
    assert not torch.isnan(summ_0).any(), 'summ_0 contains nans'

    # --- Summary[1][k] = k * t array containing sum_{cells} gamma[cell,k,t]*log(x_t!)
    summ_1 = torch.mul(torch.lgamma(x+1), gamma)
    summ_1[(x==-1).repeat(1, n_states, 1)] = 0
    summ_1	= summ_1.sum(0)
    assert not torch.isnan(summ_1).any(), 'summ_1 contains nans'

    # --- Summary[2][k] = k * t array containing sum_{cells} gamma[cell,k,t]*mods
    summ_2 = torch.mul(gamma, modifiers).sum(0)
    assert not torch.isnan(summ_2).any(), 'summ_2 contains nans'

    # --- Summary[3][k] = k * t array containing sum_{cells} gamma[cell,k,t]*x_t
    summ_3 = torch.mul(x, gamma).sum(0)
    assert not torch.isnan(summ_3).any(), 'summ_3 contains nans'

    return [summ_0.numpy(), summ_1.numpy(), summ_2.numpy(), summ_3.numpy()]


def poisson_NLL(alphas, diploid_means, summaries, prior_distributions):
    """
    Compute the negative log likelihood of the observations given the Poisson distribution parameters.
    For a given gene and cell, c, the negative log likelihood of the observation, x, is given by:
        NLL^(k)(μ; x) = -∑(c=1 to N) [γ^(k)_c (x_c log(μ^(k)) + x_c log(modifier_c) - μ modifier_c - log(x_c!))]
    where:
        k is the state of the HMM
        γ^(k)_c is the probability of cell c being in state k
        x_c is the observation in cell c
        μ^(k) is the mean of the Poisson distribution in state k
        modifier_c is the modifier for cell c
        N is the number of cells

    We also incorporate prior distributions on the normalized diploid means, from which the Poisson means are derived
    and on the alpha weights.

    Summaries computed are as follows:
    Summary[0][k] = 1 * t array containing sum_{cells} gamma[cell,k,t]*x_t*log(mods)
    Summary[1][k] = 1 * t array containing sum_{cells} gamma[cell,k,t]*log(x_t!)
    Summary[2][k] = 1 * t array containing sum_{cells} gamma[cell,k,t]*mods
    Summary[3][k] = 1 * t array containing sum_{cells} gamma[cell,k,t]*x_t

    :param alphas: (np.ndarray) - per gene alpha weights, shape (n_genes,1)
    :param diploid_means: (np.ndarray) - per gene diploid means, shape (n_genes,1)
    :param summaries: (list) - List of summaries for the Poisson distribution as computed by compute_Poisson_summaries
    :param prior_distributions: (pd.DataFrame) - DataFrame containing the prior distributions parameters for
            the diploid means (diploid_mean, diploid_std) and alpha weights (alpha_mean, alpha_std). These are used to
            construct the torch Normal distributions for the Poisson means and alpha weights.
    :return: (float) - Negative log likelihood of the observations given the Poisson distribution parameters
    """

    if isinstance(alphas, pd.Series):
        alphas = alphas.values
    if isinstance(diploid_means, pd.Series):
        diploid_means = diploid_means.values

    # Compute the Poisson means for each state
    n_states = summaries[0].shape[0]
    mean = np.arange(n_states) / 2
    mean[0] = 1e-2
    # Compute the mean for each gene
    mean = np.repeat(mean.reshape(-1, 1), summaries[0].shape[1], axis=1)
    mean = diploid_means * (mean ** alphas)

    LL = summaries[0] + summaries[3] * np.log(mean) - summaries[1] - summaries[2] * mean
    NLL = -np.sum(LL)

    # Incorporate prior distributions on the normalized diploid means and alpha weights
    # Compute the log probability of the prior distributions
    prior_alpha_mean = torch.tensor(prior_distributions['alpha_mean'].values)
    prior_alpha_std = torch.tensor(prior_distributions['alpha_std'].values)
    alpha_dist = torch.distributions.Normal(prior_alpha_mean, prior_alpha_std)

    alpha_prior_NLL = -torch.sum(alpha_dist.log_prob(torch.tensor(alphas)))
    NLL += alpha_prior_NLL

    # print('alpha_NLL', -torch.sum(alpha_dist.log_prob(torch.tensor(alphas))))
    return NLL


def poisson_NLL_single_alpha(alpha, diploid_mean, summaries, prior_distributions):
    """
    Compute the negative log likelihood for a single alpha while keeping others fixed.
    :param alpha: (float) - Alpha weight to optimize
    :param diploid_mean (float) - Diploid mean for the gene
    :param summaries: (list) - List of summaries for the Poisson distribution as computed by compute_Poisson_summaries
    :param prior_distributions: (pd.DataFrame) - DataFrame containing the prior distributions parameters for
            the diploid means (diploid_mean, diploid_std) and alpha weights (alpha_mean, alpha_std). These are used to
            construct the torch Normal distributions for the Poisson means and alpha weights.
    :return: (float) - Negative log likelihood of the observations given the Poisson distribution parameters
    """
    # Compute the Poisson means for each state
    n_states = summaries[0].shape[0]
    mean = np.arange(n_states) / 2
    mean[0] = 1e-2
    mean = np.repeat(mean.reshape(-1, 1), summaries[0].shape[1], axis=1)
    mean = diploid_mean * (mean ** alpha)

    if np.any(mean <= 0):
        raise ValueError("Mean values must be positive to compute log")

    LL = summaries[0] + summaries[3] * np.log(mean) - summaries[1] - summaries[2] * mean
    NLL = -np.sum(LL)

    # Incorporate prior distributions on the normalized diploid means and alpha weights
    # empirical_diploid_mean = torch.tensor(prior_distributions['diploid_mean'])
    # empirical_diploid_std = torch.tensor(prior_distributions['diploid_std'])

    prior_alpha_mean = torch.tensor(prior_distributions['alpha_mean'])
    prior_alpha_std = torch.tensor(prior_distributions['alpha_std'])

    # diploid_mean_dist = torch.distributions.Normal(empirical_diploid_mean, empirical_diploid_std)
    alpha_dist = torch.distributions.Normal(prior_alpha_mean, prior_alpha_std)

    # NLL += -torch.sum(diploid_mean_dist.log_prob(torch.tensor(diploid_mean, dtype=torch.float32)))
    alpha_prior_NLL = -torch.sum(alpha_dist.log_prob(torch.tensor(alpha, dtype=torch.float32)))
    NLL += alpha_prior_NLL

    return NLL.item()  # Ensure the return type is float


def update_single_alpha(alpha, diploid_mean, summaries, prior_distribution, bounds):
    """
    Wrapper to optimize a single alpha while keeping others fixed.
    """
    args = (diploid_mean, summaries, prior_distribution)
    result = scipy.optimize.minimize(poisson_NLL_single_alpha, alpha, args=args, method='L-BFGS-B', bounds=[bounds])
    return result.x


def update_alphas(initial_alphas, diploid_means, summaries, prior_distributions, bounds=(0.1, 3)):
    """
    Use scipy.optimize.minimize to update each alpha weight independently for the Poisson distribution.
    """
    log.debug('Optimizing alpha weights separately...')
    optimized_alphas = initial_alphas.copy()
    start_time = time.time()
    for i in range(len(initial_alphas)):
        optimized_alphas.iloc[i] = update_single_alpha(initial_alphas.iloc[i],
                                                       diploid_means.iloc[i],
                                                       summaries,
                                                       prior_distributions.iloc[i],
                                                       bounds)
    log.debug(f'Optimized alpha weights in {(time.time() - start_time):.2f} seconds')
    return optimized_alphas


def update_diploid_means(initial_means, summaries, prior_distributions):
    """
    Compute the direct update of the diploid means given the Poisson distribution parameters.
    :param initial_means:
    :param summaries:
    :param prior_distributions:
    :return:
    """
    raise NotImplementedError