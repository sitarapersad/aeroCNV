import torch
from torch.distributions.distribution import Distribution
from torch.distributions import Poisson
from numbers import Number
import numpy as np

class State:
    """
    State object for Hidden Markov Model. Contains a distribution object (e.g. Poisson, Negative Binomial, Reset, etc.)
    and a name for the state. Distribution object should be a subclass of torch.distributions.Distribution or None for
    states that don't emit anything.
    """

    def __init__(self, distribution, name):
        """
        Initialize a state object.
        :param distribution: (torch.distributions.Distribution) Distribution object for the state or None
        :param name: (str) Name of the state
        """
        self.name = name
        # Check if distribution is an instance of Distribution class
        if distribution is not None:
            assert isinstance(distribution, Distribution), "Distribution must be an instance of torch.distributions.Distribution"
        self.distribution = distribution

    def __repr__(self):
        """
        Return a string representation of the state object.
        :return:
        """
        return "State: {0} \n Distribution: {1}".format(self.name, type(self.distribution))


class Reset(Distribution):
    """
    Creates a 'reset' distribution which emits reset character with probability 1, and anything else with probability 0.
    Default reset character is -1. This distribution is used to model the reset state in the Hidden Markov Model, so that
    transitions are reset between chromosomes/non-contiguous regions of the genome.
    Example::

            >>> m = Reset(-1)
            >>> m.rsample()
                    -1
    Args:
            reset (int/Tensor): character which is emitted with prob 1.
    """

    def __init__(self, reset=-1):
        """
        Initialize a reset distribution.
        :param reset: (int/Tensor) Character which is emitted with probability 1
        """
        self.reset = reset
        self.length = 1

    def log_prob(self, value):
        """
        Compute the log probability of the given value under the reset distribution.
        :param value: (Tensor) Value to compute log probability for
        :return: (Tensor) Log probability of the value under the reset distribution
        """
        prob = torch.zeros(value.shape, dtype=torch.float64)
        prob[value == -1] = 1

        return torch.log(prob)


import torch
from torch.distributions import Poisson
import numpy as np
#
#
# class OurPoisson(Distribution):
#     """
#     Initialize an instance of a Poisson distribution. Supports multivariate Poisson distribution
#     by passing in 1D arrays for means.
#
#     :param mean: (float or Tensor) Mean parameter of the distribution
#     """
#
#     def __init__(self, mean):
#         self.length = len(mean)
#
#         # Define associated torch Poisson distribution
#         if isinstance(mean, np.ndarray) or isinstance(mean, list):
#             mean = torch.DoubleTensor(mean)
#         self.pois = Poisson(rate=mean)
#
#     @property
#     def mean(self):
#         """
#         :return: (Tensor) Mean of the Poisson distribution
#         """
#         return self.pois.mean
#
#     @property
#     def variance(self):
#         """
#         :return: (Tensor) Variance of the Poisson distribution
#         """
#         return self.pois.variance
#
#     def rsample(self, n_samples=1, dim=None, modifier=1):
#         """
#         Return n_samples random samples. With probability dropout_prob, sample 0.
#         Otherwise, sample from underlying Gamma distribution.
#
#         :param n_samples: (int) Number of samples to draw
#         :param dim: (int or None) Dimension at which samples should be drawn
#         :param modifier: (float or Tensor) Modifier for sequencing depth
#         :return: (Tensor) Random samples
#         """
#         if dim is None:
#             sample = self.pois.sample(torch.Size([n_samples]))
#         else:
#             corr_mean = self.pois.mean[dim].float() * modifier
#             pois_slice = Poisson(rate=corr_mean)
#             sample = pois_slice.sample(torch.Size([n_samples]))
#         return sample
#
#     def log_prob(self, value, modifiers=None):
#         """
#         Compute the log probability of the given values under the Poisson distribution, incorporating modifiers.
#
#         :param value: (Tensor) Observed values with shape (n_cells, n_features)
#         :param modifiers: (Tensor or float) Modifiers for sequencing depth. Length should be n_cells or a single value.
#         :return: (Tensor) Log probabilities of the observed values
#         """
#         if modifiers is not None:
#             if isinstance(modifiers, (np.ndarray, list, float, int)):
#                 modifiers = torch.DoubleTensor(np.broadcast_to(modifiers, value.shape[0]))
#             elif isinstance(modifiers, torch.Tensor) and modifiers.dim() == 0:
#                 modifiers = modifiers.expand(value.shape[0])
#             modified_mean = self.pois.mean * modifiers[:, None]
#         else:
#             modified_mean = self.pois.mean
#
#         # Compute log probability using the Poisson log probability formula
#         pois_lp = value * torch.log(modified_mean) - torch.lgamma(value + 1.0) - modified_mean
#         # Check there are no NaNs in the log probability
#         assert not torch.isnan(pois_lp).any(), "NaNs in log probability"
#
#         return pois_lp


import torch
import numpy as np
from torch.distributions import Poisson, Bernoulli


class OurPoisson(Distribution):
    """
    Initialize an instance of a Zero-Inflated Poisson distribution. Supports multivariate Zero-Inflated Poisson distribution
    by passing in 1D arrays for means and zero-inflation probabilities.

    :param mean: (float or Tensor) Mean parameter of the Poisson distribution
    :param zero_inflation_prob: (float or Tensor) Probability of zero inflation
    """

    def __init__(self, mean, zero_inflation_prob=0.01):
        self.length = len(mean)

        if isinstance(mean, np.ndarray) or isinstance(mean, list):
            mean = torch.DoubleTensor(mean)
        if isinstance(zero_inflation_prob, float) or isinstance(zero_inflation_prob, int):
            zero_inflation_prob = [zero_inflation_prob] * self.length

        if isinstance(zero_inflation_prob, np.ndarray) or isinstance(zero_inflation_prob, list):
            zero_inflation_prob = torch.DoubleTensor(zero_inflation_prob)
        assert len(mean) == len(zero_inflation_prob), "Mean and zero-inflation probability should have the same length"
        # zero inflation probability should be between 0 and 1
        assert (zero_inflation_prob >= 0).all() and (zero_inflation_prob <1).all(), ("Zero-inflation probability should "
                                                                                      "be between 0 and 1")

        self.pois = Poisson(rate=mean)
        self.zero_inflation_prob = zero_inflation_prob
        self.bernoulli = Bernoulli(probs=1 - zero_inflation_prob)

    @property
    def mean(self):
        """
        :return: (Tensor) Mean of the Zero-Inflated Poisson distribution
        """
        return (1 - self.zero_inflation_prob) * self.pois.mean

    @property
    def variance(self):
        """
        :return: (Tensor) Variance of the Zero-Inflated Poisson distribution
        """
        return self.zero_inflation_prob * (1 - self.zero_inflation_prob) * self.pois.mean ** 2 + (
                    1 - self.zero_inflation_prob) * self.pois.variance

    def rsample(self, n_samples=1, dim=None, modifier=1):
        """
        Return n_samples random samples from the Zero-Inflated Poisson distribution.

        :param n_samples: (int) Number of samples to draw
        :param dim: (int or None) Dimension at which samples should be drawn
        :param modifier: (float or Tensor) Modifier for sequencing depth
        :return: (Tensor) Random samples
        """
        if dim is None:
            sample = self.pois.sample(torch.Size([n_samples]))
        else:
            corr_mean = self.pois.mean[dim].float() * modifier
            pois_slice = Poisson(rate=corr_mean)
            sample = pois_slice.sample(torch.Size([n_samples]))

        bernoulli_sample = self.bernoulli.sample(torch.Size([n_samples, self.length]))
        zero_inflated_sample = bernoulli_sample * sample
        return zero_inflated_sample

    def log_prob(self, value, modifiers=None):
        """
        Compute the log probability of the given values under the Zero-Inflated Poisson distribution, incorporating modifiers.

        :param value: (Tensor) Observed values with shape (n_cells, n_features)
        :param modifiers: (Tensor or float) Modifiers for sequencing depth. Length should be n_cells or a single value.
        :return: (Tensor) Log probabilities of the observed values
        """
        if modifiers is not None:
            if isinstance(modifiers, (np.ndarray, list, float, int)):
                modifiers = torch.DoubleTensor(np.broadcast_to(modifiers, value.shape[0]))
            elif isinstance(modifiers, torch.Tensor) and modifiers.dim() == 0:
                modifiers = modifiers.expand(value.shape[0])
            modified_mean = self.pois.mean * modifiers[:, None]
        else:
            modified_mean = self.pois.mean

        # Poisson log probability
        pois_lp = value * torch.log(modified_mean) - torch.lgamma(value + 1.0) - modified_mean

        # Zero-inflated log probability
        zero_lp = torch.log(self.zero_inflation_prob + (1 - self.zero_inflation_prob) * torch.exp(-modified_mean))
        non_zero_lp = torch.log(1 - self.zero_inflation_prob) + pois_lp

        zero_mask = (value == 0).float()
        log_prob = zero_mask * zero_lp + (1 - zero_mask) * non_zero_lp

        # Check there are no NaNs in the log probability
        assert not torch.isnan(log_prob).any(), "NaNs in log probability"

        return log_prob



