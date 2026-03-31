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
