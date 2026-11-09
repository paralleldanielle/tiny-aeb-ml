"""Process setup and cost reporting shared by every script entry point.

Every script sets the torch thread count to the number of physical cores and reports package
versions and peak RAM, so CPU cost is comparable across runs.
"""

import importlib.metadata
import logging
import os
import platform
import resource
from pathlib import Path

import torch

__all__ = [
    "REPORTED_PACKAGES",
    "configure_logging",
    "configure_threads",
    "count_physical_cores",
    "package_versions",
    "peak_rss_mb",
    "physical_core_count",
]

REPORTED_PACKAGES = ("torch", "transformers", "trl", "datasets", "accelerate")
# Hugging Face Hub logs every HTTP request at INFO, which buries the script's own output.
QUIET_LOGGERS = ("httpx", "httpx2", "httpcore")
CPUINFO_PATH = Path("/proc/cpuinfo")

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure INFO logging for a script entry point and quiet chatty library loggers."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for name in QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def count_physical_cores(cpuinfo: str) -> int:
    """Count distinct physical cores in the text of ``/proc/cpuinfo``.

    Hyper-threading lists each core twice under the same ``core id``, so counting distinct
    ``(physical id, core id)`` pairs gives physical cores, not logical CPUs.

    Args:
        cpuinfo: Contents of ``/proc/cpuinfo``.

    Returns:
        Number of physical cores, or 0 if the text has no ``core id`` entries.
    """
    cores: set[tuple[str, str]] = set()
    physical_id = "0"
    for line in cpuinfo.splitlines():
        key, _, value = line.partition(":")
        key = key.strip()
        if key == "physical id":
            physical_id = value.strip()
        elif key == "core id":
            cores.add((physical_id, value.strip()))
    return len(cores)


def physical_core_count() -> int:
    """Return the number of physical CPU cores, falling back to logical CPUs if unknown."""
    try:
        cpuinfo = CPUINFO_PATH.read_text()
    except FileNotFoundError:
        num_cores = 0
    else:
        num_cores = count_physical_cores(cpuinfo)
    if num_cores == 0:
        num_cores = os.cpu_count() or 1
        logger.warning("Physical core count unknown; using %d logical CPUs", num_cores)
    return num_cores


def configure_threads() -> int:
    """Set torch's intra-op thread count to the physical core count and log it.

    Hyper-threads share a core's vector units, so more threads than physical cores usually
    slows dense matrix math down rather than speeding it up.

    Returns:
        The thread count that was set.
    """
    num_threads = physical_core_count()
    torch.set_num_threads(num_threads)
    logger.info("torch threads: %d (physical cores)", num_threads)
    return num_threads


def package_versions() -> dict[str, str]:
    """Return the Python version and the installed versions of the core ML packages."""
    versions = {"python": platform.python_version()}
    for package in REPORTED_PACKAGES:
        versions[package] = importlib.metadata.version(package)
    return versions


def peak_rss_mb() -> float:
    """Return this process's peak resident memory in MiB so far.

    Linux only: ``ru_maxrss`` is reported in KiB there (macOS uses bytes).
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
