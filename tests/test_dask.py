import logging
import sys

import msprime
import pytest
from test_sgkit import make_ts_and_zarr

import tsinfer

logger = logging.getLogger(__name__)


def test_dask_match_ancestors(tmpdir):
    from dask.distributed import Client
    from dask.distributed import LocalCluster

    cluster = LocalCluster(processes=True, threads_per_worker=1, n_workers=2)
    client = Client(cluster)  # noqa F841
    ts = msprime.sim_ancestry(
        100, recombination_rate=3e-5, sequence_length=1e6, random_seed=42
    )
    ts = msprime.sim_mutations(ts, rate=3e-5, random_seed=42)
    sd = tsinfer.SampleData.from_tree_sequence(ts)
    anc = tsinfer.generate_ancestors(sd, path=str(tmpdir / "anc"), num_threads=2)
    anc_ts_dask = tsinfer.match_ancestors(
        sd,
        anc,
        recombination_rate=2e-8,
        precision=13,
        path_compression=True,
        num_threads=2,
        use_dask=True,
    )
    anc_ts = tsinfer.match_ancestors(
        sd,
        anc,
        recombination_rate=2e-8,
        precision=13,
        path_compression=True,
        num_threads=2,
    )
    anc_ts.tables.assert_equals(anc_ts_dask.tables, ignore_provenance=True)


@pytest.mark.skipif(sys.platform == "win32", reason="No cyvcf2 on Windows")
def test_dask_match_samples(tmp_path):
    from dask.distributed import Client
    from dask.distributed import LocalCluster

    cluster = LocalCluster(processes=False, threads_per_worker=1, n_workers=2)
    client = Client(cluster)  # noqa F841
    ts, zarr_path = make_ts_and_zarr(tmp_path)
    sd = tsinfer.SgkitSampleData(zarr_path)
    anc = tsinfer.generate_ancestors(sd, path=str(tmp_path / "anc"), num_threads=2)
    anc_ts_dask = tsinfer.match_ancestors(
        sd,
        anc,
        recombination_rate=2e-8,
        precision=13,
        path_compression=True,
        num_threads=1,
        use_dask=True,
    )
    anc_ts = tsinfer.match_ancestors(
        sd,
        anc,
        recombination_rate=2e-8,
        precision=13,
        path_compression=True,
        num_threads=1,
    )
    anc_ts.tables.assert_equals(anc_ts_dask.tables, ignore_provenance=True)
    inf_ts_dask = tsinfer.match_samples(
        sd,
        anc_ts,
        recombination_rate=2e-8,
        precision=13,
        path_compression=True,
        num_threads=1,
        use_dask=True,
    )
    inf_ts = tsinfer.match_samples(
        sd,
        anc_ts,
        recombination_rate=2e-8,
        precision=13,
        path_compression=True,
        num_threads=1,
    )
    inf_ts.tables.assert_equals(inf_ts_dask.tables, ignore_provenance=True)
