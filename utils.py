"""
Utility functions and helpers to handle configuration files and logging.

Author: Riccardo Finotello <riccardo.finotello@cea.fr>
Maintainers (name.surname@cea.fr) :

- Riccardo Finotello
"""

import logging
import os
from pathlib import Path
from PIL import Image


import numpy as np
from yacs.config import CfgNode as CN

from frg.distributions.distributions import EmpiricalDistribution
from frg.distributions.distributions import PoissonEmpiricalDistribution


def get_cfg_defaults() -> CN:
    """
    Get the default configuration.

    Returns
    -------
    CN
        The default configuration (YACS CfgNode)
    """

    cfg = CN()

    # Distribution parameters
    cfg.DIST = CN()
    cfg.DIST.NUM_SAMPLES = 1000
    cfg.DIST.SIGMA = 1.0
    cfg.DIST.RATIO = 0.5
    cfg.DIST.SEED = 42

    # Signal parameters
    cfg.SIG = CN()
    cfg.SIG.INPUT = None
    cfg.SIG.SNR = 0.0

    # Potential parameters
    cfg.POT = CN()
    cfg.POT.UV_SCALE = 1.0e-5
    cfg.POT.KAPPA_INIT = 1.0e-5
    cfg.POT.U2_INIT = 1.0e-5
    cfg.POT.U4_INIT = 1.0e-5
    cfg.POT.U6_INIT = 1.0e-5

    # Data parameters
    cfg.DATA = CN()
    cfg.DATA.OUTPUT_DIR = "results"
    # Poisson parameters
    cfg.DIST.NOISE_TYPE = "gaussian"  # 'gaussian' or 'poisson'
    cfg.DIST.POISSON_CENTERING = ""
    cfg.DIST.POISSON_LAMBDA = 100.0  # mean \lambda for Poisson background

    return cfg.clone()


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """
    Get the logger.

    Parameters
    ----------
    name : str
        The name of the logger (logging session)
    level : int
        The logging level:

            - logging.DEBUG = 10
            - logging.INFO = 20
            - logging.WARNING = 30
            - logging.ERROR = 40
            - logging.CRITICAL = 50

    Returns
    -------
    logging.Logger
        The logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Set up the format
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="{asctime} | [{levelname:^8s}] : {message}",
        datefmt="%Y-%m-%d %H:%M:%S",
        style="{",
    )

    # Set the format
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def load_data(cfg: CN) -> EmpiricalDistribution:
    """
    Load the data from file.

    Parameters
    ----------
    cfg : CN
        The configuration file.

    .. warning:

        The image must be a B/W image (single channel) and present pixels in the range :math:`[0, 255]`.

    Returns
    -------
    EmpiricalDistribution
        The distribution
    """
    data = Path(os.path.expandvars(cfg.SIG.INPUT))
    if not data.exists():
        raise FileNotFoundError("Input data file %s does not exist!" % data)

        # Create the distribution
    if "npy" in data.suffix.lower():
        # Load covariance matrix
        data = np.load(data)
        dist = EmpiricalDistribution.from_covariance(data).fit()
    elif ("png" in data.suffix.lower()) or ("jpg" in data.suffix.lower()):
        # Load image
        img = np.array(Image.open(data)) / 255.0
        img -= img.mean()  # centre the image
        img /= img.std()  # scale the image
        dist = EmpiricalDistribution.from_config(cfg).fit(
            X=img, snr=cfg.SIG.SNR, fac=0.3
        )

    return dist


def load_poisson_data(
    cfg: CN, poisson_centering: str | None = None
) -> PoissonEmpiricalDistribution:
    """
    Poissonian version of load_data:
    - .npy ->  as covariance
    - .png/.jpg -> load grayscale, standardize, and fit from config
    Resizing (to NUM_SAMPLES x NUM_SAMPLES*RATIO) is done inside the class .fit(),
    """
    # --- resolve mode & lambda like load_data resolves config ---
    mode = (
        poisson_centering
        or getattr(cfg.DIST, "POISSON_CENTERING", "")
        or "non-centered"
    ).lower()
    if mode not in {"non-centered", "centered", "mirrored"}:
        raise ValueError(
            f"Invalid poisson_centering='{mode}'. "
            "Use one of {'non-centered','centered','mirrored'}."
        )

    lam = float(getattr(cfg.DIST, "POISSON_LAMBDA", 100.0))

    # --- path & existence check (same pattern as load_data) ---
    data = Path(os.path.expandvars(cfg.SIG.INPUT))
    if not data.exists():
        raise FileNotFoundError(f"Input data file {data} does not exist!")

    if "npy" in data.suffix.lower():
        # Load covariance matrix
        cov = np.load(data)
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise ValueError(
                f"Cov shape must be square, got shape {cov.shape}."
            )
        return PoissonEmpiricalDistribution.from_covariance(
            cov, lam=lam, poisson_centering=mode
        ).fit()

    elif data.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        img = np.asarray(Image.open(data).convert("L"), dtype=np.float64)
        # renormalization and standardization
        img = img / 255.0
        img -= img.mean()
        std = float(img.std())
        img = img / (std if std > 1e-12 else 1.0)

        return PoissonEmpiricalDistribution.from_config(
            cfg, lam=lam, poisson_centering=mode
        ).fit(X=img, snr=float(cfg.SIG.SNR), fac=0.3)
    else:
        raise ValueError(f"Unsupported input type: {data.suffix}")


def load_mixed_data(
    cfg: CN,
    poisson_centering: str | None = None,
    poisson_psd: bool = False,  # make Poisson term PSD if True
    resize: bool = True,  # To assure right size according to config. DO NOT SET FALSE!
) -> EmpiricalDistribution:
    """
    resize: Needs to be True otherwise the function can only be used with .fit.
    Builds a mixed covariance:
        C_mixed = ( (beta * I_tilde + Z)(beta * I_tilde + Z)^T )/(n-1)  +  P_lambda
    where Z ~ N(0,1). P_lambda is a Poisson-based symmetric matrix;
    poisson_psd=True constructs a PSD covariance-like term.
    """
    # --- resolve config ---
    data_path = Path(os.path.expandvars(cfg.SIG.INPUT))
    if not data_path.exists():
        raise FileNotFoundError(f"Input data file {data_path} does not exist!")

    n_samples = int(cfg.DIST.NUM_SAMPLES)
    ratio = float(cfg.DIST.RATIO)
    n_vars = int(n_samples * ratio)
    beta = float(cfg.SIG.SNR)
    lam = float(getattr(cfg.DIST, "POISSON_LAMBDA", 100.0))
    if lam <= 0:
        raise ValueError("POISSON_LAMBDA must be > 0.")
    mode = (
        poisson_centering
        or getattr(cfg.DIST, "POISSON_CENTERING", "non-centered")
    ).lower()
    if mode not in {"non-centered", "centered", "mirrored"}:
        raise ValueError(
            "poisson_centering must be one of {'centered','non-centered','mirrored'}."
        )

    # Only in case the size parameters are not chosen properly.
    def _fit_to_shape(A: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
        H, W = A.shape
        Oh, Ow = out_shape
        # crop if larger
        A = A[: min(H, Oh), : min(W, Ow)]
        H, W = A.shape
        if H == Oh and W == Ow:
            return A

        # reflect-pad if smaller
        def _reflect_pad(x, target, axis):
            if x.shape[axis] >= target:
                return x
            k = target - x.shape[axis]
            pads = [(0, 0), (0, 0)]
            pads[axis] = (0, k)
            # reflect along the axis
            refl = np.flip(x, axis=axis)
            # tile as needed then cut
            tiles = int(np.ceil(k / x.shape[axis])) + 1
            ext = np.concatenate([x, *(refl for _ in range(tiles))], axis=axis)
            return (
                np.pad(x, [(0, 0), (0, 0)], mode="constant")
                if False
                else ext.take(range(target), axis=axis)
            )

        A = _reflect_pad(A, Oh, axis=0)
        A = _reflect_pad(A, Ow, axis=1)
        return A

    if "npy" in data_path.suffix.lower():
        # Treat as covariance; size Poisson term from C itself.
        C = np.load(data_path)
        if C.ndim != 2 or C.shape[0] != C.shape[1]:
            raise ValueError(
                f"Expected square covariance, got shape {C.shape}."
            )
        p = C.shape[0]
        P = _make_poisson_cov(
            p, lam=lam, mode=mode, seed=int(cfg.DIST.SEED), psd=poisson_psd
        )
        C_mixed = C + P
        return EmpiricalDistribution.from_covariance(C_mixed).fit()

    elif data_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        # renormalize and standardize
        img = np.asarray(Image.open(data_path).convert("L"), dtype=np.float64)
        img = img / 255.0
        img = img - img.mean()
        std = float(img.std())
        img = img / (std if std > 1e-12 else 1.0)

        # --- Make signal template
        if resize:
            # Fit the size of the image to the config size
            from skimage.transform import resize as _resize

            Irs = _resize(
                img,
                (n_samples, n_vars),
                mode="reflect",
                anti_aliasing=True,
                preserve_range=True,
            )
        else:
            # no resampling: crop/reflect-pad to match target shape
            Irs = _fit_to_shape(img, (n_samples, n_vars))

        # --- Gaussian background & covariance ---
        rng = np.random.default_rng(int(cfg.DIST.SEED))
        Z = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_vars))
        X = Z + beta * Irs
        C = (X.T @ X) / max(1, n_samples - 1)

        # --- Add Poisson term sized to p=n_vars ---
        P = _make_poisson_cov(
            n_vars, lam=lam, mode=mode, seed=int(cfg.DIST.SEED), psd=poisson_psd
        )
        C_mixed = C + P
        return EmpiricalDistribution.from_covariance(C_mixed).fit()

    else:
        raise ValueError(f"Unsupported input type: {data_path.suffix}")


def _make_poisson_cov(
    n_vars: int, lam: float, mode: str, seed: int, psd: bool = False
) -> np.ndarray:
    """
    Build a random Poissonian distributed  symmetric matrix of shape (n_vars, n_vars).
    If psd=True, returns a PSD 'covariance-like' term via (A @ A.T)/n_vars otherwise regular symmetrization
    Centering modes:
      - 'non-centered':   A ~ Pois(λ)
      - 'centered':       A ~ Pois(λ) - λ
      - 'mirrored':       A ~ Pois(λ) - Pois(λ)
    In all cases entries are scaled by 1/255 to correspond the image renormalization.
    """
    if lam <= 0:
        raise ValueError("POISSON_LAMBDA must be > 0.")
    rng = np.random.default_rng(int(seed))
    mode = mode.lower()

    if mode == "non-centered":
        A = rng.poisson(lam=lam, size=(n_vars, n_vars)).astype(np.float64)
    elif mode == "centered":
        A = rng.poisson(lam=lam, size=(n_vars, n_vars)).astype(np.float64) - lam
    elif mode == "mirrored":
        A = rng.poisson(lam=lam, size=(n_vars, n_vars)).astype(
            np.float64
        ) - rng.poisson(lam=lam, size=(n_vars, n_vars)).astype(np.float64)
    else:
        raise ValueError(
            "poisson_centering must be one of {'centered','non-centered','mirrored'}"
        )

    A /= 255.0

    if psd:
        # PSD by construction; Wishart-like term
        return (A @ A.T) / max(1, n_vars)

    # Symmetric GOE-like (not necessarily PSD)
    return 0.5 * (A + A.T)


def load_no_noise(cfg: CN) -> EmpiricalDistribution:
    """
    Signal-only loader, WITHOUT adding any synthetic noise.
    - .npy  -> interpret as covariance matrix or spectrum depending on the dimensions
    - .png/.jpg/.jpeg -> load grayscale image, standardize, build covariance
    Returns
    -------
    EmpiricalDistribution
        The empirical spectral distribution of the signal-only covariance.
    """
    # --- resolve path ---
    data = Path(os.path.expandvars(cfg.SIG.INPUT))
    if not data.exists():
        raise FileNotFoundError(f"Input data file {data} does not exist!")

    # ---- case 1: .npy -> covariance or spectrum ----
    if "npy" in data.suffix.lower():
        arr = np.load(data)
        if arr.ndim == 1:
            # direct spectrum mode
            return EmpiricalDistribution.from_spectrum(arr).fit()
        elif arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
            # covariance
            return EmpiricalDistribution.from_covariance(arr).fit()
        else:
            raise ValueError(
                f".npy input in load_no_noise must be 1D (spectrum) or 2D "
                f"square (covariance); got shape {arr.shape}."
            )

    # ---- case 2: image file -> build covariance from standardized pixels ----
    elif data.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        # make sure B/W, rescale, then center & standardize
        img = np.asarray(Image.open(data).convert("L"), dtype=np.float64)
        img = img / 255.0
        img -= img.mean()
        std = float(img.std())
        img = img / (std if std > 1e-12 else 1.0)

        # interpret rows as samples, cols as features
        X = img
        n_samples = X.shape[0]
        cov = (X.T @ X) / max(1, n_samples - 1)
        return EmpiricalDistribution.from_covariance(cov).fit()

    # ---- unsupported type ----
    else:
        raise ValueError(
            f"Unsupported input type for load_no_noise: {data.suffix}"
        )
