#
# Copyright (C) 2018-2026 University of Oxford
#
# This file is part of tsinfer.
#
# tsinfer is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tsinfer is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with tsinfer.  If not, see <http://www.gnu.org/licenses/>.
#
"""
Tests for perfect inference: verify that tsinfer can exactly recover a known
tree sequence topology when given perfect ancestor information.

The test tree sequences in tests/data/perfect_inference/ were pre-selected
from msprime simulations (with ``insert_perfect_mutations``) so that every
tree-transition involves a genuine topology change (no adjacent trees with
the same branching structure but different internal node IDs).  This means
the simplified inferred tree sequence should have exactly the same edges as
the simplified original.

See ``scripts/generate_perfect_inference_data.py`` (or the seed search notes
in this file) for how the seeds were found.
"""

from __future__ import annotations

import bisect
import pathlib

import numpy as np
import tskit
from helpers import make_ancestor_vcz, ts_to_sample_vcz

from tsinfer.config import (
    AncestorsConfig,
    Config,
    MatchConfig,
    Source,
)
from tsinfer.pipeline import match

UNKNOWN_ALLELE = 255

DATA_DIR = pathlib.Path(__file__).parent / "data" / "perfect_inference"


# ---------------------------------------------------------------------------
# Ported helpers from master's evaluation.py
# ---------------------------------------------------------------------------


def insert_perfect_mutations(ts, delta=None):
    """
    Returns a copy of the specified tree sequence where the left and right
    coordinates of all edgesets are marked by mutations. This *should* be
    sufficient information to recover the tree sequence exactly.
    """
    tables = ts.dump_tables()
    tables.sites.clear()
    tables.mutations.clear()

    num_children = np.zeros(ts.num_nodes, dtype=int)
    parent = np.zeros(ts.num_nodes, dtype=int) - 1

    current_delta = 0
    if delta is not None:
        current_delta = delta

    for (left, right), edges_out, edges_in in ts.edge_diffs():
        last_num_children = list(num_children)
        children_in = set()
        children_out = set()
        parents_in = set()
        parents_out = set()
        for e in edges_out:
            parent[e.child] = -1
            num_children[e.parent] -= 1
            children_out.add(e.child)
            parents_out.add(e.parent)
        for e in edges_in:
            parent[e.child] = e.parent
            num_children[e.parent] += 1
            children_in.add(e.child)
            parents_in.add(e.parent)
        root = 0
        while parent[root] != -1:
            root = parent[root]
        if len(edges_out) > 4 or (len(edges_out) == 2 and root not in parents_in):
            raise ValueError("Multiple recombination detected")
        x = left - current_delta
        for u in list(children_out - children_in) + list(children_in & children_out):
            if last_num_children[u] > 0:
                site_id = tables.sites.add_row(position=x, ancestral_state="0")
                tables.mutations.add_row(site=site_id, node=u, derived_state="1")
                x -= current_delta

        if delta is None:
            max_nodes = 2 * (len(children_out) + len(children_in)) + len(parents_in) + 1
            current_delta = (right - left) / max_nodes
        x = left
        for c in list(children_in - children_out) + list(children_in & children_out):
            if num_children[c] > 0:
                site_id = tables.sites.add_row(position=x, ancestral_state="0")
                tables.mutations.add_row(site=site_id, node=c, derived_state="1")
                x += current_delta

        for u in parents_in:
            if parent[u] != -1:
                site_id = tables.sites.add_row(position=x, ancestral_state="0")
                tables.mutations.add_row(site=site_id, node=u, derived_state="1")
                x += current_delta

    tables.sort()
    return tables.tree_sequence()


def get_ancestral_haplotypes(ts):
    """
    Returns a numpy array of the haplotypes of the ancestors in the
    specified tree sequence.
    """
    tables = ts.dump_tables()
    flags = tables.nodes.flags[:]
    flags[:] = 1
    tables.nodes.flags = flags

    sites = [site.position for site in ts.sites()]
    tsp = tables.tree_sequence()
    B = tsp.genotype_matrix().T

    A = np.zeros((ts.num_nodes, ts.num_sites), dtype=np.uint8)
    A[:] = UNKNOWN_ALLELE
    for edge in ts.edges():
        start = bisect.bisect_left(sites, edge.left)
        end = bisect.bisect_right(sites, edge.right)
        if sites[end - 1] == edge.right:
            end -= 1
        A[edge.parent, start:end] = B[edge.parent, start:end]
    A[: ts.num_samples] = B[: ts.num_samples]
    return A


def get_ancestor_descriptors(A):
    """
    Given an array of ancestral haplotypes A in forwards time-order,
    return the descriptors for each ancestor.
    """
    L = A.shape[1]
    ancestors = [np.zeros(L, dtype=np.uint8)]
    focal_sites = [[]]
    start = [0]
    end = [L]
    mask = np.ones(L)
    for a in A:
        masked = np.logical_and(a == 1, mask).astype(int)
        new_sites = np.where(masked)[0]
        mask[new_sites] = 0
        segment = np.where(a != UNKNOWN_ALLELE)[0]
        if segment.shape[0] > 0:
            s = segment[0]
            e = segment[-1] + 1
            assert np.all(a[s:e] != UNKNOWN_ALLELE)
            assert np.all(a[:s] == UNKNOWN_ALLELE)
            assert np.all(a[e:] == UNKNOWN_ALLELE)
            ancestors.append(a)
            focal_sites.append(new_sites)
            start.append(s)
            end.append(e)
    return np.array(ancestors, dtype=np.uint8), start, end, focal_sites


# ---------------------------------------------------------------------------
# Bridge: convert perfect ancestors into VCZ stores
# ---------------------------------------------------------------------------


def build_perfect_ancestors(ts):
    """
    Build sample and ancestor VCZ stores from a tree sequence with perfect
    mutations. Returns (sample_vcz, ancestor_vcz).
    """
    A = get_ancestral_haplotypes(ts)
    # Non-sample node IDs, reversed to forward time order
    nonsample_ids = np.arange(ts.num_nodes - 1, ts.num_samples - 1, -1)
    A = A[ts.num_samples :][::-1]
    ancestors, start, end, focal_sites = get_ancestor_descriptors(A)

    # Skip ancestors[0] — the all-zeros root — make_root_ts() creates one.
    # get_ancestor_descriptors prepends a synthetic root that doesn't
    # correspond to any original node, so ancestor indices 1..N map to the
    # rows of A that were kept.  We track which original rows those are
    # so we can look up the real coalescent times.
    kept_indices = []
    idx = 0
    for a in A:
        segment = np.where(a != UNKNOWN_ALLELE)[0]
        if segment.shape[0] > 0:
            kept_indices.append(idx)
        idx += 1
    # kept_indices[i] corresponds to ancestors[i+1] (after the synthetic root)

    ancestors = ancestors[1:]
    start = start[1:]
    end = end[1:]
    focal_sites = focal_sites[1:]
    N = len(ancestors)

    positions = np.array([int(site.position) for site in ts.sites()], dtype=np.int32)
    num_sites = len(positions)

    # Build genotype array (num_sites, N, 1) int8; map 255 -> -1
    genotypes = np.full((num_sites, N, 1), -1, dtype=np.int8)
    for i in range(N):
        s, e = start[i], end[i]
        hap = ancestors[i][s:e].astype(np.int8)
        genotypes[s:e, i, 0] = hap

    # Use real coalescent times from the original tree sequence
    node_times = np.array([ts.node(int(n)).time for n in nonsample_ids])
    times = np.array([node_times[ki] for ki in kept_indices], dtype=float)

    # Alleles
    alleles = np.array([["0", "1"]] * num_sites)

    # Focal positions: map site indices to genomic positions, pad with -2
    max_focal = max((len(f) for f in focal_sites), default=0)
    if max_focal == 0:
        max_focal = 1
    focal_positions = np.full((N, max_focal), -2, dtype=np.int32)
    for i, fs in enumerate(focal_sites):
        for j, site_idx in enumerate(fs):
            focal_positions[i, j] = positions[site_idx]

    # Sequence intervals from site positions
    sequence_length = int(ts.sequence_length)
    if num_sites > 0:
        sequence_intervals = np.array(
            [[int(positions[0]), int(positions[-1])]], dtype=np.int32
        )
    else:
        sequence_intervals = np.array([[0, sequence_length]], dtype=np.int32)

    ancestor_vcz = make_ancestor_vcz(
        genotypes=genotypes,
        positions=positions,
        alleles=alleles,
        times=times,
        focal_positions=focal_positions,
        sequence_intervals=sequence_intervals,
        contig_length=sequence_length,
    )
    sample_vcz = ts_to_sample_vcz(ts, ancestral_allele="ANCESTRAL")
    return sample_vcz, ancestor_vcz


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------


def _make_config(
    sample_store,
    ancestor_store,
    path_compression=False,
):
    src = Source(path=sample_store, name="test")
    return Config(
        sources={"test": src},
        ancestors=AncestorsConfig(path=ancestor_store, sources=["test"]),
        match=MatchConfig(
            sources=["test"],
            output="output.trees",
            path_compression=path_compression,
        ),
    )


# ---------------------------------------------------------------------------
# Topology verification
# ---------------------------------------------------------------------------


def assert_edges_equal(original_ts, inferred_ts):
    """
    Verify exact topology recovery: after simplifying both tree sequences
    to their leaf sample nodes, the edge tables must be identical.

    Node times are allowed to differ (the root time in the inferred TS
    is offset by the virtual-root construction), but the parent/child
    relationships and genomic intervals must match exactly.

    This check is only valid for tree sequences where every breakpoint
    corresponds to a genuine topology change (no adjacent trees that
    differ only in internal node identity while preserving the same
    branching structure for all samples).  The test fixtures in
    ``tests/data/perfect_inference/`` are pre-selected to satisfy this.
    """
    # Identify real sample nodes in the inferred TS (time=0, flagged)
    inf_samples = [
        n.id
        for n in inferred_ts.nodes()
        if n.time == 0.0 and n.flags & tskit.NODE_IS_SAMPLE
    ]
    assert len(inf_samples) == original_ts.num_samples

    inferred_simple = inferred_ts.simplify(samples=inf_samples)
    original_simple = original_ts.simplify()

    orig_edges = set()
    for e in original_simple.edges():
        orig_edges.add((e.left, e.right, e.parent, e.child))
    inf_edges = set()
    for e in inferred_simple.edges():
        inf_edges.add((e.left, e.right, e.parent, e.child))

    assert orig_edges == inf_edges, (
        f"Edge sets differ.\n"
        f"Original: {original_simple.num_trees} trees, "
        f"{original_simple.num_edges} edges\n"
        f"Inferred: {inferred_simple.num_trees} trees, "
        f"{inferred_simple.num_edges} edges\n"
        f"Only in original ({len(orig_edges - inf_edges)}): "
        f"{sorted(orig_edges - inf_edges)[:5]}\n"
        f"Only in inferred ({len(inf_edges - orig_edges)}): "
        f"{sorted(inf_edges - orig_edges)[:5]}"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# Seeds were found by searching msprime simulations for tree sequences
# where perfect inference recovers exact edges after simplification.
# The .trees files in tests/data/perfect_inference/ are the output of
# insert_perfect_mutations(base_ts, delta=1) for each seed.
#
# single_tree.trees:
#   msprime.sim_ancestry(5, sequence_length=1_000_000,
#       recombination_rate=0, random_seed=234)
#   → 1 tree, 10 samples, 16 sites
#
# small_smc.trees:
#   msprime.sim_ancestry(5, sequence_length=1_000_000,
#       recombination_rate=1e-6, model="smc_prime", random_seed=35)
#   → 12 trees, 10 samples, 80 sites
#
# larger_smc.trees:
#   msprime.sim_ancestry(8, sequence_length=1_000_000,
#       recombination_rate=1e-6, model="smc_prime", random_seed=86)
#   → 15 trees, 16 samples, 142 sites


class TestPerfectInference:
    def _run_perfect_inference(self, ts):
        sample_vcz, ancestor_vcz = build_perfect_ancestors(ts)
        cfg = _make_config(sample_vcz, ancestor_vcz)
        inferred_ts = match(cfg)
        assert_edges_equal(ts, inferred_ts)

    def test_single_tree(self):
        ts = tskit.load(DATA_DIR / "single_tree.trees")
        assert ts.num_trees == 1
        self._run_perfect_inference(ts)

    def test_small_smc(self):
        ts = tskit.load(DATA_DIR / "small_smc.trees")
        assert ts.num_trees > 1
        self._run_perfect_inference(ts)

    def test_larger_smc(self):
        ts = tskit.load(DATA_DIR / "larger_smc.trees")
        assert ts.num_trees > 1
        self._run_perfect_inference(ts)
