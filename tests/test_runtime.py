import torch

from tiny_aeb_ml import runtime

# Two physical cores with hyper-threading: four logical CPUs.
CPUINFO_2_CORES_HT = """\
processor\t: 0
physical id\t: 0
core id\t\t: 0

processor\t: 1
physical id\t: 0
core id\t\t: 0

processor\t: 2
physical id\t: 0
core id\t\t: 1

processor\t: 3
physical id\t: 0
core id\t\t: 1
"""

# Two sockets reuse the same core ids, so the socket must be part of the key.
CPUINFO_2_SOCKETS = """\
processor\t: 0
physical id\t: 0
core id\t\t: 0

processor\t: 1
physical id\t: 1
core id\t\t: 0
"""


def test_count_physical_cores_ignores_hyperthreads() -> None:
    assert runtime.count_physical_cores(CPUINFO_2_CORES_HT) == 2


def test_count_physical_cores_counts_each_socket() -> None:
    assert runtime.count_physical_cores(CPUINFO_2_SOCKETS) == 2


def test_count_physical_cores_without_core_ids() -> None:
    assert runtime.count_physical_cores("processor\t: 0\n") == 0


def test_configure_threads_sets_torch() -> None:
    num_threads = runtime.configure_threads()
    assert num_threads >= 1
    assert torch.get_num_threads() == num_threads


def test_package_versions_reports_core_stack() -> None:
    versions = runtime.package_versions()
    assert set(versions) == {"python", *runtime.REPORTED_PACKAGES}


def test_peak_rss_is_positive() -> None:
    assert runtime.peak_rss_mb() > 0
