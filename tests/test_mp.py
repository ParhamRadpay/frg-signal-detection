"""
Test the  distributions

Test the Marchenko-Pastur distribution.
"""

import numpy as np
import pytest
from scipy.integrate import quad

from frg.distributions.distributions import MarchenkoPastur


class TestMarchenkoPastur:
    """Test the Marchenko-Pastur distribution"""

    def test_init(self):
        """Test the constructor of the class"""
        # Assert raise if ratio <= 0
        with pytest.raises(ValueError):
            MarchenkoPastur(0.0, 1.0)
            MarchenkoPastur(-1.0, 1.0)

        # Assert warning if ratio >= 1
        with pytest.warns(UserWarning):
            MarchenkoPastur(1.0, 1.0)
            MarchenkoPastur(2.0, 1.0)

        # Assert values
        mp = MarchenkoPastur(0.5, 1.0)
        assert mp.ratio == 0.5
        assert mp.sigma == 1.0
        assert mp.lplus == 1.0**2 * (1.0 + np.sqrt(0.5)) ** 2
        assert mp.lminus == 1.0**2 * (1.0 - np.sqrt(0.5)) ** 2

    def test_pdf(self):
        """Test the computation of the PDF"""
        mp = MarchenkoPastur(0.5, 1.0)
        assert mp.pdf(0.0) == 0.0
        assert mp.pdf(50.0) == 0.0
        assert isinstance(mp.pdf([0.0, 50.0]), np.ndarray)
        assert (mp.pdf([0.0, 50.0]) == np.array([0.0, 0.0])).all()
        assert mp.pdf(mp.lminus) == 0.0
        assert mp.pdf(mp.lplus) == 0.0
        assert quad(mp.pdf, mp.lminus, mp.lplus)[0] == pytest.approx(1.0)

    def test_cdf(self):
        """Test the computation of the CDF"""
        mp = MarchenkoPastur(0.5, 1.0)
        assert mp.cdf(0.0) == 0.0
        assert mp.cdf(50.0) == 1.0
        assert isinstance(mp.cdf([0.0, 50.0]), np.ndarray)
        assert (mp.cdf([0.0, 50.0]) == np.array([0.0, 1.0])).all()
        assert mp.cdf(mp.lminus) == 0.0
        assert mp.cdf(mp.lplus) == 1.0
        x = np.linspace(mp.lminus, mp.lplus, 100)
        y = mp.cdf(x)
        assert all((y[1:] - y[:-1]) >= 0.0)  # test monotonic increasing

        # Test with different arguments
        assert mp.cdf(2.0, x0=0.0) == mp.cdf(2.0)

    def test_dpdf(self):
        """Test the computation of the derivative of the PDF"""
        mp = MarchenkoPastur(0.5, 1.0)
        assert mp.dpdf(0.0) == 0.0
        assert mp.dpdf(50.0) == 0.0
        assert isinstance(mp.dpdf([0.0, 50.0]), np.ndarray)
        assert (mp.dpdf([0.0, 50.0]) == np.array([0.0, 0.0])).all()

    def test_ipdf(self):
        """Test the computation of the inverse PDF (momenta)"""
        mp = MarchenkoPastur(0.5, 1.0)
        assert mp.ipdf(0.0) == 0.0
        assert mp.ipdf(0.000001) > 0.0
        assert mp.ipdf(50.0) > 0.0
        assert isinstance(mp.ipdf([0.0, 50.0]), np.ndarray)

    def test_icdf(self):
        """Test the computation of the inverse CDF (momenta)"""
        mp = MarchenkoPastur(0.5, 1.0)
        assert mp.icdf(0.0) == 0.0
        assert 0.0 < mp.icdf(50.0) < 1.0
        assert isinstance(mp.icdf([0.0, 50.0]), np.ndarray)
        x = np.linspace(0.0, 1.0, 100)
        y = mp.icdf(x)
        assert all((y[1:] - y[:-1]) >= 0.0)  # test monotonic increasing

    def test_dipdf(self):
        """Test the computation of the derivative of the inverse PDF"""
        mp = MarchenkoPastur(0.5, 1.0)
        x = np.linspace(1.0, 10.0, 100)
        y = mp.dipdf(x)
        assert all((y[1:] - y[:-1]) >= 0.0)  # test monotonic increasing

    def test_canonical_dimensions(self):
        """Test the computation of the canonical dimensions"""
        mp = MarchenkoPastur(0.5, 1.0)
        x = np.linspace(0.0, 3.0, 100)
        dimu2, dimu4, dimu6, dimchi = mp.canonical_dimensions(x).T
        assert isinstance(dimu2, np.ndarray)
        assert isinstance(dimu4, np.ndarray)
        assert isinstance(dimu6, np.ndarray)
        assert isinstance(dimchi, np.ndarray)
        dimu2, dimu4, dimu6, dimchi = mp.canonical_dimensions(1.0).T
        assert isinstance(dimu2, float)
        assert isinstance(dimu4, float)
        assert isinstance(dimu6, float)
        assert isinstance(dimchi, float)

    def test_frg_equations(self):
        """Test the computation of the FRG equations"""
        mp = MarchenkoPastur(0.5, 1.0)
        x = 1.0e-5
        u2, u4, u6 = mp._frg_equations_single(
            x,
            u2=1.0e-5,
            u4=1.0e-5,
            u6=1.0e-5,
        ).T
        assert isinstance(u2, float)
        assert isinstance(u4, float)
        assert isinstance(u6, float)
        k2, u2, u4, u6 = mp.frg_equations(
            x,
            u2_init=1.0e-5,
            u4_init=1.0e-5,
            u6_init=1.0e-5,
        ).T
        assert isinstance(k2, np.ndarray)
        assert isinstance(u2, np.ndarray)
        assert isinstance(u4, np.ndarray)
        assert isinstance(u6, np.ndarray)

    def test_frg_equations_lpa(self):
        """
        Test the computations of the FRG equations in Local Potential Approximation
        """
        mp = MarchenkoPastur(0.5, 1.0)
        x = 1.0e-5
        kappa, u4, u6 = mp._frg_equations_lpa_single(
            x,
            kappa=1.0e-5,
            u4=1.0e-5,
            u6=1.0e-5,
        ).T
        assert isinstance(kappa, float)
        assert isinstance(u4, float)
        assert isinstance(u6, float)
        k2, kappa, u4, u6 = mp.frg_equations_lpa(
            x,
            kappa_init=1.0e-5,
            u4_init=1.0e-5,
            u6_init=1.0e-5,
        ).T
        assert isinstance(k2, np.ndarray)
        assert isinstance(kappa, np.ndarray)
        assert isinstance(u4, np.ndarray)
        assert isinstance(u6, np.ndarray)
