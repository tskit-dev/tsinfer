import logging

import msprime

import tsinfer

logger = logging.getLogger(__name__)


def test_dask_match_ancestors(tmpdir):
    from dask.distributed import Client
    from dask.distributed import LocalCluster

    cluster = LocalCluster(processes=True, threads_per_worker=1, n_workers=8)
    client = Client(cluster)  # noqa F841
    ts = msprime.sim_ancestry(
        100, recombination_rate=3e-5, sequence_length=1e6, random_seed=42
    )
    ts = msprime.sim_mutations(ts, rate=3e-5, random_seed=42)
    sd = tsinfer.SampleData.from_tree_sequence(ts)
    anc = tsinfer.generate_ancestors(sd, path=str(tmpdir / "anc"), num_threads=8)
    anc_ts_dask = tsinfer.match_ancestors(
        sd,
        anc,
        recombination_rate=2e-8,
        precision=13,
        path_compression=True,
        num_threads=8,
        use_dask=True,
    )
    anc_ts = tsinfer.match_ancestors(
        sd,
        anc,
        recombination_rate=2e-8,
        precision=13,
        path_compression=True,
        num_threads=8,
    )
    anc_ts.tables.assert_equals(anc_ts_dask.tables, ignore_provenance=True)
