"""
Distribution test

Test functions and methods of the base distribution class.
"""

import numpy as np
import pytest

from frg.distributions.distributions import Distribution


class TestBaseDistribution:
    """Test the base distribution"""

    dist = Distribution()

    def test_init(self):
        """Test the initialization of the class."""
        assert self.dist.m2 == np.inf
        assert self.dist.lminus == -np.inf
        assert self.dist.lplus == np.inf

    def test_pdf(self):
        """Test the PDF of the distribution."""
        with pytest.raises(NotImplementedError):
            self.dist.pdf(0.0)

    def test_cdf(self):
        """Test the CDF of the distribution."""
        with pytest.raises(NotImplementedError):
            self.dist.cdf(0.0)

    def test_dpdf(self):
        """Test the derivative of the PDF of the distribution."""
        with pytest.raises(NotImplementedError):
            self.dist.dpdf(0.0)

    def test_ipdf(self):
        """Test the PDF of the inverse distribution."""
        with pytest.raises(NotImplementedError):
            self.dist.ipdf(0.0)

    def test_icdf(self):
        """Test the CDF of the inverse distribution."""
        assert self.dist.icdf(0.0) == 0.0
        with pytest.raises(NotImplementedError):
            self.dist.icdf(10.0)

    def test_dipdf(self):
        """Test the derivative of the PDF of the inverse distribution."""
        with pytest.raises(NotImplementedError):
            self.dist.dipdf(0.0)
