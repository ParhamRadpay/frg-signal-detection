#! /usr/bin/env python3
"""
Compute the canonical dimensions

Compute the canonical dimensions of the couplings in a theory with given momenta distribution.

Author: Riccardo Finotello <riccardo.finotello@cea.fr>
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from frg import MarchenkoPastur, get_cfg_defaults, get_logger
from frg.utils.utils import (
    load_data,
    load_poisson_data,
    load_mixed_data,
    load_no_noise,
)

__author__ = "Riccardo Finotello"
__email__ = "riccardo.finotello@cea.fr"
__description__ = "Compute the canonical dimensions of the couplings in a theory with given momenta distribution."
__epilog__ = "For bug reports and info: " + __author__ + " <" + __email__ + ">"


def main(a: argparse.Namespace) -> int | str:
    # Get the logger
    logger_level = 10 * (4 - a.verb)
    logger = get_logger(__name__, level=logger_level)
    logger.info("Starting...")
    cfg = get_cfg_defaults()

    # Open the configuration file
    if a.config is None:
        logger.debug("No configuration file specified")
    else:
        logger.debug("Configuration file: %s" % a.config)
        cfg_file = Path(a.config)
        if cfg_file.exists():
            logger.debug("Configuration file exists!")
            cfg.merge_from_file(cfg_file)
    cfg.merge_from_list(a.args)
    cfg.freeze()

    if a.print_config:
        print(cfg.dump())
        return 0

    # Run the simulation
    logger.info("Computing the canonical dimensions...")

    # Distribution parameters
    x_max = cfg.POT.UV_SCALE
    n_vars = int(cfg.DIST.NUM_SAMPLES * cfg.DIST.RATIO)
    x_min = 1.0 / np.sqrt(n_vars)  # the smallest bin

    # Define the distribution
    if a.analytic:
        x_min = 0.0  # analytic can go to zero
        dist = MarchenkoPastur(ratio=cfg.DIST.RATIO, sigma=cfg.DIST.SIGMA)
    else:
        noise_type = (
            a.noise_type or getattr(cfg.DIST, "NOISE_TYPE", "gaussian")
        ).lower()

        if noise_type == "no_noise":
            logger.info("Signal-only mode; using cfg.SIG.INPUT")
            dist = load_no_noise(cfg)

        elif noise_type == "poisson":
            dist = load_poisson_data(
                cfg, poisson_centering=getattr(a, "PoissonCentering", None)
            )

        elif noise_type == "mixed":
            dist = load_mixed_data(
                cfg, poisson_centering=getattr(a, "PoissonCentering", None)
            )

        else:
            dist = load_data(cfg)
    # Compute the canonical dimensions
    x = np.linspace(x_min, x_max, num=5000)
    dimu2, dimu4, dimu6, _ = dist.canonical_dimensions(x).T

    # Save data
    output_dir = Path(cfg.DATA.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"snr={cfg.SIG.SNR}"
    if a.suffix.lower() == "var":
        suffix = f"sigma={cfg.DIST.SIGMA}"
    if a.suffix.lower() == "ratio":
        suffix = f"ratio={cfg.DIST.RATIO}_seed={cfg.DIST.SEED}"
    if a.suffix.lower() == "seed":
        suffix = f"seed={cfg.DIST.SEED}"
    if a.suffix.lower() == "snrlambda":
        # SNR + lambda (mixed case)
        if noise_type == "mixed":
            lam = getattr(cfg.DIST, "POISSON_LAMBDA", None)
            lam_str = "NA" if lam is None else f"{lam}"
            suffix = f"snr={cfg.SIG.SNR}_lam={lam_str}"
        else:
            suffix = f"snr={cfg.SIG.SNR}"
    if a.suffix.lower() == "input":
        img_stem = Path(str(getattr(cfg.SIG, "INPUT", ""))).stem
        suffix = f"snr={cfg.SIG.SNR}_img={img_stem}"
    if a.analytic:
        suffix = f"analytic_sigma={cfg.DIST.SIGMA}_ratio={cfg.DIST.RATIO}_seed={cfg.DIST.SEED}"
    output_file = output_dir / f"mp_canonical_dimensions_{suffix}.json"
    payload = {
        "k2": x.tolist(),
        "dimu2": dimu2.tolist(),
        "dimu4": dimu4.tolist(),
        "dimu6": dimu6.tolist(),
        "dist": dist.ipdf(x).tolist(),
        "m2": dist.m2,
    }
    if hasattr(dist, "m2_mp"):
        payload["m2_mp"] = dist.m2_mp
    with open(output_file, "w") as f:
        json.dump(payload, f)
    logger.info("Results saved in %s" % output_file)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__description__, epilog=__epilog__
    )
    parser.add_argument("--config", required=False, help="Configuration file")
    parser.add_argument(
        "--analytic", action="store_true", help="Run an analytic simulation"
    )
    parser.add_argument(
        "--print_config", action="store_true", help="Print configuration"
    )
    parser.add_argument(
        "--suffix",
        default="snr",
        choices=["snr", "var", "ratio", "seed", "snrlambda", "input"],
        help="Type of suffix used in the output files",
    )
    parser.add_argument(
        "--args",
        nargs="+",
        default=[],
        help="Additional configuration arguments (see YACS documentation)",
    )
    parser.add_argument(
        "--noise_type",
        choices=["gaussian", "poisson", "mixed", "no_noise"],
        default=None,
        help="Override noise model. 'no_noise' uses the pure-signal covariance.",
    )
    parser.add_argument(
        "-v", dest="verb", action="count", default=0, help="Verbosity level"
    )
    parser.add_argument(
        "--PoissonCentering",
        choices=["centered", "non-centered", "mirrored"],
        default="centered",
        help="Poisson background centering mode (overrides cfg.DIST.POISSON_CENTERING).",
    )
    args = parser.parse_args()

    code = main(args)

    sys.exit(code)
