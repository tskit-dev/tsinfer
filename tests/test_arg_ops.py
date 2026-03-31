import pytest
import tskit

from tsinfer import arg_ops


class TestAddVestigialRoot:
    def test_non_discrete_genome(self):
        tables = tskit.TableCollection(sequence_length=1.5)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        ts = tables.tree_sequence()
        with pytest.raises(ValueError, match="discrete genome"):
            arg_ops.add_vestigial_root(ts)

    def test_empty_tree_sequence(self):
        tables = tskit.TableCollection(sequence_length=1)
        ts = tables.tree_sequence()
        with pytest.raises(ValueError, match="Emtpy trees"):
            arg_ops.add_vestigial_root(ts)


def _make_ts(nodes, edges, sites=None, mutations=None, sequence_length=100):
    """Helper to build a tree sequence from lists of (flags, time) and
    (left, right, parent, child) tuples."""
    tables = tskit.TableCollection(sequence_length=sequence_length)
    for flags, time in nodes:
        tables.nodes.add_row(flags=flags, time=time)
    for left, right, parent, child in edges:
        tables.edges.add_row(left=left, right=right, parent=parent, child=child)
    if sites is not None:
        for pos, ancestral in sites:
            tables.sites.add_row(position=pos, ancestral_state=ancestral)
    if mutations is not None:
        for site, node, derived in mutations:
            tables.mutations.add_row(site=site, node=node, derived_state=derived)
    tables.sort()
    return tables.tree_sequence()


class TestIsPcAncestor:
    def test_zero_flags(self):
        assert not arg_ops.is_pc_ancestor(0)

    def test_sample_flag(self):
        assert not arg_ops.is_pc_ancestor(tskit.NODE_IS_SAMPLE)

    def test_pc_flag(self):
        assert arg_ops.is_pc_ancestor(arg_ops.NODE_IS_PC_ANCESTOR)

    def test_pc_flag_combined(self):
        assert arg_ops.is_pc_ancestor(arg_ops.NODE_IS_PC_ANCESTOR | tskit.NODE_IS_SAMPLE)

    def test_other_bits(self):
        for bit in range(32):
            flags = 1 << bit
            if bit == 16:
                assert arg_ops.is_pc_ancestor(flags)
            else:
                assert not arg_ops.is_pc_ancestor(flags)


class TestCompressPaths:
    def test_identical_single_edge_path(self):
        """Two children with identical single-edge path — compressed."""
        ts = _make_ts(
            nodes=[(1, 0), (1, 0), (0, 1.0)],
            edges=[(0, 100, 2, 0), (0, 100, 2, 1)],
        )
        result = arg_ops.compress_paths(ts)
        assert result.num_nodes == 4  # 3 + 1 PC
        # 3 edges: parent->PC, PC->0, PC->1
        assert result.num_edges == 3
        pc_node = result.node(3)
        assert arg_ops.is_pc_ancestor(pc_node.flags)
        assert pc_node.time == 1.0 - arg_ops.PC_ANCESTOR_INCREMENT

    def test_identical_two_edge_path(self):
        """Two children with identical two-edge path — one PC node."""
        ts = _make_ts(
            nodes=[(1, 0), (1, 0), (0, 1.0), (0, 2.0), (0, 2.0)],
            edges=[
                (0, 50, 3, 0),
                (50, 100, 4, 0),
                (0, 50, 3, 1),
                (50, 100, 4, 1),
            ],
        )
        result = arg_ops.compress_paths(ts)
        assert result.num_nodes == 6  # 5 + 1 PC
        # 6 edges: 3->PC, 4->PC, PC->0 x2, PC->1 x2
        assert result.num_edges == 6
        pc_node = result.node(5)
        assert arg_ops.is_pc_ancestor(pc_node.flags)
        assert pc_node.time == 2.0 - arg_ops.PC_ANCESTOR_INCREMENT

    def test_three_children_identical_path(self):
        """Three children with identical path — one PC node."""
        ts = _make_ts(
            nodes=[(1, 0), (1, 0), (1, 0), (0, 2.0)],
            edges=[(0, 100, 3, 0), (0, 100, 3, 1), (0, 100, 3, 2)],
        )
        result = arg_ops.compress_paths(ts)
        assert result.num_nodes == 5  # 4 + 1 PC
        # 4 edges: parent->PC, PC->0, PC->1, PC->2
        assert result.num_edges == 4

    def test_different_paths_not_compressed(self):
        """Children with different edge sets are not compressed."""
        ts = _make_ts(
            nodes=[(1, 0), (1, 0), (0, 1.0), (0, 2.0), (0, 2.0), (0, 2.0)],
            edges=[
                (0, 50, 3, 0),
                (50, 100, 5, 0),  # parent 5 for child 0
                (0, 50, 3, 1),
                (50, 100, 4, 1),  # parent 4 for child 1
            ],
        )
        result = arg_ops.compress_paths(ts)
        assert result.num_nodes == ts.num_nodes
        assert result.num_edges == ts.num_edges

    def test_no_shared_edges(self):
        """Each child has unique edges — nothing to compress."""
        ts = _make_ts(
            nodes=[(1, 0), (1, 0), (0, 1.0)],
            edges=[(0, 100, 2, 0), (0, 50, 2, 1)],  # different intervals
        )
        result = arg_ops.compress_paths(ts)
        assert result.num_nodes == ts.num_nodes
        assert result.num_edges == ts.num_edges

    def test_partial_path_overlap_not_compressed(self):
        """Children share some edges but not all — not compressed."""
        # Child 0 has edges: (0,50,P3) and (50,100,P4)
        # Child 1 has edges: (0,50,P3) and (50,100,P5)  ← differs
        ts = _make_ts(
            nodes=[(1, 0), (1, 0), (0, 1.0), (0, 2.0), (0, 2.0), (0, 2.0)],
            edges=[
                (0, 50, 3, 0),
                (50, 100, 4, 0),
                (0, 50, 3, 1),
                (50, 100, 5, 1),
            ],
        )
        result = arg_ops.compress_paths(ts)
        assert result.num_nodes == ts.num_nodes
        assert result.num_edges == ts.num_edges

    def test_preserves_other_edges(self):
        """Edges for non-compressed children are preserved."""
        ts = _make_ts(
            nodes=[
                (1, 0),
                (1, 0),
                (1, 0),
                (0, 1.0),
                (0, 2.0),
                (0, 2.0),
            ],
            edges=[
                (0, 50, 4, 0),
                (50, 100, 5, 0),
                (0, 50, 4, 1),
                (50, 100, 5, 1),
                (0, 100, 3, 2),  # different path, not shared
            ],
        )
        result = arg_ops.compress_paths(ts)
        assert result.num_nodes == 7  # 6 + 1 PC
        # 6 compressed edges + 1 preserved = 7
        assert result.num_edges == 7

    def test_multiple_groups(self):
        """Two groups of children with different identical paths."""
        # Children 0,1 share path through parent 4
        # Children 2,3 share path through parent 5
        ts = _make_ts(
            nodes=[(1, 0), (1, 0), (1, 0), (1, 0), (0, 2.0), (0, 2.0)],
            edges=[
                (0, 100, 4, 0),
                (0, 100, 4, 1),
                (0, 100, 5, 2),
                (0, 100, 5, 3),
            ],
        )
        result = arg_ops.compress_paths(ts)
        assert result.num_nodes == 8  # 6 + 2 PC
        # 6 edges: 2 * (parent->PC, PC->child, PC->child)
        assert result.num_edges == 6

    def test_time_validation(self):
        """Raises ValueError when PC ancestor time would not be valid."""
        inc = arg_ops.PC_ANCESTOR_INCREMENT
        ts = _make_ts(
            nodes=[(1, 1.0), (1, 1.0), (0, 1.0 + inc / 2)],
            edges=[(0, 100, 2, 0), (0, 100, 2, 1)],
        )
        with pytest.raises(ValueError, match="path compression ancestor"):
            arg_ops.compress_paths(ts)

    def test_with_mutations(self):
        """Mutations on compressed edges still reference correct nodes."""
        ts = _make_ts(
            nodes=[(1, 0), (1, 0), (0, 1.0)],
            edges=[(0, 100, 2, 0), (0, 100, 2, 1)],
            sites=[(50, "A")],
            mutations=[(0, 0, "T")],
        )
        result = arg_ops.compress_paths(ts)
        assert result.mutation(0).node == 0
        assert result.mutation(0).derived_state == "T"

    def test_no_edges(self):
        """Tree sequence with no edges — nothing to compress."""
        ts = _make_ts(nodes=[(1, 0)], edges=[])
        result = arg_ops.compress_paths(ts)
        assert result.num_nodes == 1
        assert result.num_edges == 0

    def test_min_parent_time_used(self):
        """PC ancestor time uses minimum parent time across the path."""
        ts = _make_ts(
            nodes=[(1, 0), (1, 0), (0, 1.0), (0, 3.0), (0, 5.0)],
            edges=[
                (0, 50, 3, 0),
                (50, 100, 4, 0),
                (0, 50, 3, 1),
                (50, 100, 4, 1),
            ],
        )
        result = arg_ops.compress_paths(ts)
        pc_node = result.node(5)
        assert pc_node.time == 3.0 - arg_ops.PC_ANCESTOR_INCREMENT

    def test_pc_ancestor_flags(self):
        """New PC ancestor nodes have NODE_IS_PC_ANCESTOR flag set."""
        ts = _make_ts(
            nodes=[(1, 0), (1, 0), (0, 1.0)],
            edges=[(0, 100, 2, 0), (0, 100, 2, 1)],
        )
        result = arg_ops.compress_paths(ts)
        for i in range(3):
            assert not arg_ops.is_pc_ancestor(result.node(i).flags)
        assert arg_ops.is_pc_ancestor(result.node(3).flags)
