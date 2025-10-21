"""
Test the empirical distribution

Test functions and methods of the empirical distribution data class.
"""

import numpy as np
import pytest

from frg.distributions.distributions import EmpiricalDistribution
from frg.utils.utils import get_cfg_defaults


class TestEmpiricalDistribution:
    """Test the empirical distribution"""

    def test_init(self):
        """Test the constructor of the class"""
        # Assert raise if n_samples < 2
        with pytest.raises(ValueError):
            EmpiricalDistribution(n_samples=1)

        # Assert raise if sigma <= 0
        with pytest.raises(ValueError):
            EmpiricalDistribution(n_samples=2000, sigma=0.0)
            EmpiricalDistribution(n_samples=2000, sigma=-1.0)

        # Assert raise if ratio <= 0
        with pytest.raises(ValueError):
            EmpiricalDistribution(n_samples=2000, ratio=0.0)
            EmpiricalDistribution(n_samples=2000, ratio=-1.0)

        # Warning if 2 < n_samples < 1000
        with pytest.warns(UserWarning):
            EmpiricalDistribution(n_samples=100)

        # Assert values
        emp = EmpiricalDistribution(
            n_samples=1532,
            sigma=1.3,
            ratio=0.1,
            seed=123,
        )
        assert emp.n_samples == 1532
        assert emp.sigma == 1.3
        assert emp.ratio == 0.1
        assert emp.n_vars == int(1532 * 0.1)
        assert emp.seed == 123
        assert hasattr(emp, "data")
        assert isinstance(emp.data, np.ndarray)
        assert emp.data.shape == (1532, int(1532 * 0.1))
        assert emp.data.dtype == np.float64
        assert emp.data.mean() == pytest.approx(0.0, abs=1.0e-2)
        assert emp.data.std() == pytest.approx(1.3, abs=1.0e-2)

    def test_from_config(self):
        """Test the from_config method of the class"""
        cfg = get_cfg_defaults()
        emp = EmpiricalDistribution.from_config(cfg)
        assert emp.n_samples == 1000
        assert emp.sigma == 1.0
        assert emp.ratio == 0.5
        assert emp.n_vars == int(1000 * 0.5)
        assert emp.seed == 42

    def test_from_covariance(self):
        """Test the from_covariance method of the class"""
        X = np.random.randn(100, 50)
        cov = np.cov(X, rowvar=False)
        assert cov.shape == (50, 50)
        emp = EmpiricalDistribution.from_covariance(cov)
        assert emp._iscov
        assert emp.n_samples == 9999
        assert emp.sigma == 1.0
        assert emp.ratio == 1.0
        assert emp.seed == 9999
        assert (emp.data == cov).all()

    def test_add_signal(self):
        """Test the add_signal method of the class"""
        emp = EmpiricalDistribution(n_samples=1024, sigma=1.0, ratio=0.5)
        X = np.random.randn(1024, 512)
        emp2 = emp.add_signal(X)
        assert isinstance(emp2, EmpiricalDistribution)
        assert emp2.n_samples == 1024
        assert emp2.sigma == 1.0
        assert emp2.ratio == 0.5
        assert emp2.n_vars == int(1024 * 0.5)
        assert emp2.seed == 42
        assert (emp2.data == emp.data).all()
        emp3 = emp2.add_signal(X, snr=1.0)
        assert isinstance(emp3, EmpiricalDistribution)
        assert emp3.n_samples == 1024
        assert emp3.sigma == 1.0
        assert emp3.ratio == 0.5
        assert emp3.n_vars == int(1024 * 0.5)
        assert emp3.seed == 42
        assert (emp3.data == emp2.data).all()

        with pytest.raises(ValueError):
            emp2.add_signal(X, snr=-1.0)

    def test_fit(self):
        """Test the fit method of the class"""
        emp = EmpiricalDistribution(n_samples=1024, sigma=1.0, ratio=0.5)
        X = np.random.randn(1024, 512)
        emp.fit(X, snr=0.5)
        assert hasattr(emp, "eigenvalues")
        assert isinstance(emp.eigenvalues, np.ndarray)
        assert hasattr(emp, "ipdf")
        assert hasattr(emp, "dipdf")
        assert hasattr(emp, "icdf")

        # Test from the covariance matrix
        cov = np.cov(X, rowvar=False)
        emp2 = EmpiricalDistribution.from_covariance(cov)
        emp2.fit()
