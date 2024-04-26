import logging
import sys

import msprime
import pytest
import tsutil

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
def test_dask_match_with_mask(tmp_path, tmpdir):
    from dask.distributed import Client
    from dask.distributed import LocalCluster

    cluster = LocalCluster(processes=False, threads_per_worker=1, n_workers=2)
    client = Client(cluster)  # noqa F841
    (
        mat_sd,
        mask_sd,
        samples_mask,
        variant_mask,
    ) = tsutil.make_materialized_and_masked_sampledata(tmp_path, tmpdir)
    mat_anc = tsinfer.generate_ancestors(mat_sd, path=str(tmp_path / "mat_anc"))
    mask_anc = tsinfer.generate_ancestors(mask_sd, path=str(tmp_path / "mask_anc"))
    mat_anc_ts = tsinfer.match_ancestors(mat_sd, mat_anc, use_dask=True)
    mask_anc_ts = tsinfer.match_ancestors(mask_sd, mask_anc, use_dask=True)
    mat_ts = tsinfer.match_samples(mat_sd, mat_anc_ts)
    mask_ts = tsinfer.match_samples(mask_sd, mask_anc_ts)

    mat_ts.tables.assert_equals(mask_ts.tables, ignore_provenance=True)
