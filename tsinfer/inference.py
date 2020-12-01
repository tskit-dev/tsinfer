#
# Copyright (C) 2018-2020 University of Oxford
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
Central module for high-level inference. The actual implementation of
of the core tasks like ancestor generation and matching are delegated
to other modules.
"""
import collections
import heapq
import json
import logging
import queue
import threading
import time

import humanize
import numpy as np
import tskit

import _tsinfer
import tsinfer.algorithm as algorithm
import tsinfer.constants as constants
import tsinfer.formats as formats
import tsinfer.progress as progress
import tsinfer.provenance as provenance
import tsinfer.threads as threads

logger = logging.getLogger(__name__)


def is_pc_ancestor(flags):
    """
    Returns True if the path compression ancestor flag is set on the specified
    flags value.
    """
    return (flags & constants.NODE_IS_PC_ANCESTOR) != 0


def is_srb_ancestor(flags):
    """
    Returns True if the shared recombination breakpoint flag is set on the
    specified flags value.
    """
    return (flags & constants.NODE_IS_SRB_ANCESTOR) != 0


def count_pc_ancestors(flags):
    """
    Returns the number of values in the specified array which have the
    NODE_IS_PC_ANCESTOR set.
    """
    flags = np.array(flags, dtype=np.uint32, copy=False)
    return np.sum(is_pc_ancestor(flags))


def count_srb_ancestors(flags):
    """
    Returns the number of values in the specified array which have the
    NODE_IS_SRB_ANCESTOR set.
    """
    flags = np.array(flags, dtype=np.uint32, copy=False)
    return np.sum(np.bitwise_and(flags, constants.NODE_IS_SRB_ANCESTOR) != 0)


AlleleCounts = collections.namedtuple("AlleleCounts", "known ancestral derived")


def allele_counts(genotypes):
    """
    Return summary counts of the number of different allele types for a genotypes array
    """
    n_known = np.sum(genotypes != tskit.MISSING_DATA)
    n_ancestral = np.sum(genotypes == 0)
    return AlleleCounts(
        known=n_known, ancestral=n_ancestral, derived=n_known - n_ancestral
    )


def _get_progress_monitor(progress_monitor, **kwargs):
    """
    Check if this really is a ProgressMonitor, if not, return something usable as one
    """
    if isinstance(progress_monitor, progress.ProgressMonitor):
        return progress_monitor
    if progress_monitor:
        return progress.ProgressMonitor(**kwargs)
    return progress.DummyProgressMonitor()


def _encode_metadata(obj):
    return json.dumps(obj).encode()


def _update_site_metadata(current_metadata, inference_type):
    assert inference_type in {
        constants.INFERENCE_NONE,
        constants.INFERENCE_FULL,
        constants.INFERENCE_PARSIMONY,
    }
    return {"inference_type": inference_type, **current_metadata}


def verify(sample_data, tree_sequence, progress_monitor=None):
    """
    verify(samples, tree_sequence)

    Verifies that the specified sample data and tree sequence files encode the
    same data.

    :param SampleData samples: The input :class:`SampleData` instance
        representing the observed data that we wish to compare to.
    :param TreeSequence tree_sequence: The input :class:`tskit.TreeSequence`
        instance an encoding of the specified samples that we wish to verify.
    """
    progress_monitor = _get_progress_monitor(progress_monitor, verify=True)
    if sample_data.num_sites != tree_sequence.num_sites:
        raise ValueError("numbers of sites not equal")
    if sample_data.num_samples != tree_sequence.num_samples:
        raise ValueError("numbers of samples not equal")
    if sample_data.sequence_length != tree_sequence.sequence_length:
        raise ValueError("Sequence lengths not equal")
    progress = progress_monitor.get("verify", tree_sequence.num_sites)
    for var1, var2 in zip(sample_data.variants(), tree_sequence.variants()):
        if var1.site.position != var2.site.position:
            raise ValueError(
                "site positions not equal: {} != {}".format(
                    var1.site.position, var2.site.position
                )
            )
        if var1.alleles != var2.alleles:
            raise ValueError(f"alleles not equal: {var1.alleles} != {var2.alleles}")
        if not np.array_equal(var1.genotypes, var2.genotypes):
            raise ValueError(f"Genotypes not equal at site {var1.site.id}")
        progress.update()
    progress.close()


def check_sample_indexes(sample_data, indexes):
    """
    Checks that the specified sample indexes are valid for the specified
    sample data file.
    """
    if indexes is None:
        return np.arange(sample_data.num_samples, dtype=np.int32)
    indexes = np.array(indexes)
    if len(indexes) == 0:
        raise ValueError("Must supply at least one sample to match")
    if np.any(indexes < 0) or np.any(indexes >= sample_data.num_samples):
        raise ValueError("Sample index out of bounds")
    return indexes


def infer(
    sample_data,
    *,
    num_threads=0,
    path_compression=True,
    simplify=True,
    recombination_rate=None,
    mismatch_rate=None,
    precision=None,
    exclude_positions=None,
    engine=constants.C_ENGINE,
    progress_monitor=None,
):
    """
    infer(sample_data, *, num_threads=0, path_compression=True, simplify=True,\
            exclude_positions=None)

    Runs the full :ref:`inference pipeline <sec_inference>` on the specified
    :class:`SampleData` instance and returns the inferred
    :class:`tskit.TreeSequence`.

    :param SampleData sample_data: The input :class:`SampleData` instance
        representing the observed data that we wish to make inferences from.
    :param int num_threads: The number of worker threads to use in parallelised
        sections of the algorithm. If <= 0, do not spawn any threads and
        use simpler sequential algorithms (default).
    :param bool path_compression: Whether to merge edges that share identical
        paths (essentially taking advantage of shared recombination breakpoints).
    :param bool simplify: Whether to remove extra tree nodes and edges that are not
        on a path between the root and any of the samples. To do so, the final tree
        sequence is simplified by appling the :meth:`tskit.TreeSequence.simplify`
        method with ``filter_sites``, ``filter_populations`` and
        ``filter_individuals`` set to False, and ``keep_unary`` set to True
        (default = ``True``).
    :param array_like exclude_positions: A list of site positions to exclude
        for full inference. Sites with these positions will not be used to generate
        ancestors, and not used during the copying process. Any such sites that
        exist in the sample data file will be included in the trees after the
        main inference process using parsimony. The list does not need to be
        in to be in any particular order, and can include site positions that
        are not present in the sample data file.
    :return: The :class:`tskit.TreeSequence` object inferred from the
        input sample data.
    :rtype: tskit.TreeSequence
    """
    progress_monitor = _get_progress_monitor(
        progress_monitor,
        generate_ancestors=True,
        match_ancestors=True,
        match_samples=True,
    )
    ancestor_data = generate_ancestors(
        sample_data,
        num_threads=num_threads,
        exclude_positions=exclude_positions,
        engine=engine,
        progress_monitor=progress_monitor,
    )
    ancestors_ts = match_ancestors(
        sample_data,
        ancestor_data,
        engine=engine,
        num_threads=num_threads,
        recombination_rate=recombination_rate,
        mismatch_rate=mismatch_rate,
        precision=precision,
        path_compression=path_compression,
        progress_monitor=progress_monitor,
    )
    inferred_ts = match_samples(
        sample_data,
        ancestors_ts,
        engine=engine,
        num_threads=num_threads,
        recombination_rate=recombination_rate,
        mismatch_rate=mismatch_rate,
        precision=precision,
        path_compression=path_compression,
        simplify=simplify,
        progress_monitor=progress_monitor,
    )
    return inferred_ts


def generate_ancestors(
    sample_data,
    *,
    num_threads=0,
    path=None,
    exclude_positions=None,
    engine=constants.C_ENGINE,
    progress_monitor=None,
    **kwargs,
):
    """
    generate_ancestors(sample_data, *, num_threads=0, path=None, \
            exclude_positions=None, **kwargs)

    Runs the ancestor generation :ref:`algorithm <sec_inference_generate_ancestors>`
    on the specified :class:`SampleData` instance and returns the resulting
    :class:`AncestorData` instance. If you wish to store generated ancestors
    persistently on file you must pass the ``path`` keyword argument to this
    function. For example,

    .. code-block:: python

        ancestor_data = tsinfer.generate_ancestors(sample_data, path="mydata.ancestors")

    Other keyword arguments are passed to the :class:`AncestorData` constructor,
    which may be used to control the storage properties of the generated file.

    :param SampleData sample_data: The :class:`SampleData` instance that we are
        genering putative ancestors from.
    :param int num_threads: The number of worker threads to use. If < 1, use a
        simpler synchronous algorithm.
    :param str path: The path of the file to store the sample data. If None,
        the information is stored in memory and not persistent.
    :param array_like exclude_positions: A list of site positions to exclude
        for full inference. Sites with these positions will not be used to generate
        ancestors, and not used during the copying process. The list does not
        need be in any particular order.
    :return: The inferred ancestors stored in an :class:`AncestorData` instance.
    :rtype: AncestorData
    """
    sample_data._check_finalised()
    if np.any(np.isfinite(sample_data.sites_time[:])) and np.any(
        tskit.is_unknown_time(sample_data.sites_time[:])
    ):
        raise ValueError(
            "Cannot generate ancestors from a sample_data instance that mixes user-"
            "specified times with times-as-frequencies. To explicitly set an undefined"
            "time for a site, permanently excluding it from inference, set it to np.nan."
        )
    with formats.AncestorData(sample_data, path=path, **kwargs) as ancestor_data:
        generator = AncestorsGenerator(
            sample_data,
            ancestor_data,
            num_threads=num_threads,
            engine=engine,
            progress_monitor=progress_monitor,
        )
        generator.add_sites(exclude_positions)
        generator.run()
        ancestor_data.record_provenance("generate-ancestors")
    return ancestor_data


def match_ancestors(
    sample_data,
    ancestor_data,
    *,
    num_threads=0,
    path_compression=True,
    recombination_rate=None,
    mismatch_rate=None,
    precision=None,
    extended_checks=False,
    engine=constants.C_ENGINE,
    progress_monitor=None,
):
    """
    match_ancestors(sample_data, ancestor_data, *, num_threads=0, path_compression=True)

    Runs the ancestor matching :ref:`algorithm <sec_inference_match_ancestors>`
    on the specified :class:`SampleData` and :class:`AncestorData` instances,
    returning the resulting :class:`tskit.TreeSequence` representing the
    complete ancestry of the putative ancestors.

    :param SampleData sample_data: The :class:`SampleData` instance
        representing the input data.
    :param AncestorData ancestor_data: The :class:`AncestorData` instance
        representing the set of ancestral haplotypes that we are finding
        a history for.
    :param int num_threads: The number of match worker threads to use. If
        this is <= 0 then a simpler sequential algorithm is used (default).
    :param bool path_compression: Whether to merge edges that share identical
        paths (essentially taking advantage of shared recombination breakpoints).
    :return: The ancestors tree sequence representing the inferred history
        of the set of ancestors.
    :rtype: tskit.TreeSequence
    """
    progress_monitor = _get_progress_monitor(progress_monitor, match_ancestors=True)
    sample_data._check_finalised()
    ancestor_data._check_finalised()
    matcher = AncestorMatcher(
        sample_data,
        ancestor_data,
        num_threads=num_threads,
        recombination_rate=recombination_rate,
        mismatch_rate=mismatch_rate,
        precision=precision,
        path_compression=path_compression,
        extended_checks=extended_checks,
        engine=engine,
        progress_monitor=progress_monitor,
    )
    return matcher.match_ancestors()


def augment_ancestors(
    sample_data,
    ancestors_ts,
    indexes,
    *,
    num_threads=0,
    path_compression=True,
    recombination_rate=None,
    mismatch_rate=None,
    precision=None,
    extended_checks=False,
    engine=constants.C_ENGINE,
    progress_monitor=None,
):
    """
    augment_ancestors(sample_data, ancestors_ts, indexes, *, num_threads=0,\
        path_compression=True)

    Runs the sample matching :ref:`algorithm <sec_inference_match_samples>`
    on the specified :class:`SampleData` instance and ancestors tree sequence,
    for the specified subset of sample indexes, returning the
    :class:`tskit.TreeSequence` instance including these samples. This
    tree sequence can then be used as an ancestors tree sequence for subsequent
    matching against all samples.

    :param SampleData sample_data: The :class:`SampleData` instance
        representing the input data.
    :param tskit.TreeSequence ancestors_ts: The
        :class:`tskit.TreeSequence` instance representing the inferred
        history among ancestral ancestral haplotypes.
    :param array indexes: The sample indexes to insert into the ancestors
        tree sequence.
    :param int num_threads: The number of match worker threads to use. If
        this is <= 0 then a simpler sequential algorithm is used (default).
    :param bool path_compression: Whether to merge edges that share identical
        paths (essentially taking advantage of shared recombination breakpoints).
    :return: The specified ancestors tree sequence augmented with copying
        paths for the specified sample.
    :rtype: tskit.TreeSequence
    """
    sample_data._check_finalised()
    progress_monitor = _get_progress_monitor(progress_monitor, augment_ancestors=True)
    manager = SampleMatcher(
        sample_data,
        ancestors_ts,
        num_threads=num_threads,
        recombination_rate=recombination_rate,
        mismatch_rate=mismatch_rate,
        precision=precision,
        path_compression=path_compression,
        extended_checks=extended_checks,
        engine=engine,
        progress_monitor=progress_monitor,
    )
    sample_indexes = check_sample_indexes(sample_data, indexes)
    sample_times = np.zeros(
        len(sample_indexes), dtype=sample_data.individuals_time.dtype
    )
    manager.match_samples(sample_indexes, sample_times)
    ts = manager.get_augmented_ancestors_tree_sequence(sample_indexes)
    return ts


def match_samples(
    sample_data,
    ancestors_ts,
    *,
    num_threads=0,
    path_compression=True,
    simplify=True,
    recombination_rate=None,
    mismatch_rate=None,
    precision=None,
    extended_checks=False,
    stabilise_node_ordering=False,
    engine=constants.C_ENGINE,
    progress_monitor=None,
    indexes=None,
    force_sample_times=False,
):
    """
    match_samples(sample_data, ancestors_ts, *, num_threads=0, path_compression=True,\
        simplify=True, indexes=None, force_sample_times=False)

    Runs the sample matching :ref:`algorithm <sec_inference_match_samples>`
    on the specified :class:`SampleData` instance and ancestors tree sequence,
    returning the final :class:`tskit.TreeSequence` instance containing
    the full inferred history for all samples and sites.

    :param SampleData sample_data: The :class:`SampleData` instance
        representing the input data.
    :param tskit.TreeSequence ancestors_ts: The
        :class:`tskit.TreeSequence` instance representing the inferred
        history among ancestral ancestral haplotypes.
    :param int num_threads: The number of match worker threads to use. If
        this is <= 0 then a simpler sequential algorithm is used (default).
    :param bool path_compression: Whether to merge edges that share identical
        paths (essentially taking advantage of shared recombination breakpoints).
    :param bool simplify: Whether to remove extra tree nodes and edges that are not
        on a path between the root and any of the samples. To do so, the final tree
        sequence is simplified by appling the :meth:`tskit.TreeSequence.simplify`
        method with ``filter_sites``, ``filter_populations`` and
        ``filter_individuals`` set to False and ``keep_unary`` set to True
        (default = ``True``).
    :param array_like indexes: An array of indexes into the sample_data file of
        the samples to match, or None for all samples.
    :param bool force_sample_times: After matching, should an attempt be made to
        adjust the time of "historical samples" (those associated with an individual
        having a non-zero time) such that the sample nodes in the tree sequence
        appear at the time of the individual with which they are associated.

    :return: The tree sequence representing the inferred history
        of the sample.
    :rtype: tskit.TreeSequence
    """
    sample_data._check_finalised()
    progress_monitor = _get_progress_monitor(progress_monitor, match_samples=True)
    manager = SampleMatcher(
        sample_data,
        ancestors_ts,
        num_threads=num_threads,
        recombination_rate=recombination_rate,
        mismatch_rate=mismatch_rate,
        precision=precision,
        path_compression=path_compression,
        extended_checks=extended_checks,
        engine=engine,
        progress_monitor=progress_monitor,
    )
    sample_indexes = check_sample_indexes(sample_data, indexes)
    sample_times = np.zeros(
        len(sample_indexes), dtype=sample_data.individuals_time.dtype
    )
    if force_sample_times:
        individuals = sample_data.samples_individual[:][sample_indexes]
        # By construction all samples in an sd file have an individual: but check anyway
        assert np.all(individuals >= 0)
        sample_times = sample_data.individuals_time[:][individuals]

        # Here we might want to re-order sample_indexes and sample_times
        # so that any historical ones come first, any we bomb out early if they conflict
        # but that would mean re-ordering the sample nodes in the final ts, and
        # we sometimes assume they are in the same order as in the file

    manager.match_samples(sample_indexes, sample_times)
    ts = manager.finalise(
        simplify=simplify, stabilise_node_ordering=stabilise_node_ordering
    )
    return ts


def insert_missing_sites(
    sample_data, tree_sequence, *, sample_id_map=None, progress_monitor=None
):
    """
    Return a new tree sequence containing extra sites that are present in a
    :class:`SampleData` instance but are missing from a corresponding tree sequence.
    At each newly inserted site, mutations are overlaid parsimoneously, using
    :meth:`tskit.Tree.map_mutations`,
    such that the realised variation at that site corresponds to the allelic
    distribution seen in the sample_data file. Sites that have mutations overlaid
    in this way can be identified in the output tree sequence as their
    :ref:`metadata<tskit.sec_metadata_definition>` will contain a key named
    ``inference`` set to ``tsinfer.INFERENCE_PARSIMONY``. Newly inserted sites
    that do not require mutations will have this set to `tsinfer.INFERENCE_NONE`
    instead. Sites in ``sample_data`` that already exist in the tree sequence are
    left untouched.

    By default, sample 0 in ``sample_data`` is assumed to correspond to the first
    sample node in the input tree sequence (i.e. ``tree_sequence.samples()[0]``),
    sample 1 to the second sample node, and so on. If this is not the case, a map
    can be provided, which specifies the sample ids in ``sample_data`` that
    correspond to the sample nodes in the tree sequence. This also allows the use
    of :class:`SampleData` instances that contain samples in addition to those
    in the original tree sequence.

    .. note::
        Sample states observed as missing in the input ``sample_data`` need
        not correspond to samples whose nodes are actually "missing" (i.e.
        :ref:`isolated<tskit.sec_data_model_missing_data>`) in the input tree
        sequence. In this case, the allelic state of the sample in the returned
        tree sequence will be imputed to the most parsimonious state.

    :param SampleData sample_data: The :class:`SampleData` instance
        containing some sites that are not in the input tree sequence.
    :param tskit.TreeSequence tree_sequence: The input :class:`tskit.TreeSequence`
        whose sample nodes correspond to a set of samples in the sample_data.
    :param sample_id_map array: An array of length `tree_sequence.num_samples`
        specifying the indexes of samples in the sample_data file that correspond
        to sample nodes ``0..(num_samples-1)`` in the tree sequence. If None,
        assume that all the samples in sample_data correspond to the sample nodes
        in the tree sequence, and are in the same order.
    :return: The input tree sequence with additional sites and mutations.
    :rtype: tskit.TreeSequence

    """
    if sample_data.sequence_length != tree_sequence.sequence_length:
        raise ValueError(
            "sample_data and tree_sequence must have the same sequence length"
        )
    if sample_id_map is None:
        sample_id_map = np.arange(sample_data.num_samples)
    if len(sample_id_map) != tree_sequence.num_samples:
        raise ValueError(
            "You must specify the same number of samples in sample_data "
            "as in the tree_sequence"
        )
    progress_monitor = _get_progress_monitor(progress_monitor)
    tables = tree_sequence.dump_tables()
    trees = tree_sequence.trees()
    tree = next(trees)
    positions = sample_data.sites_position[:]
    new_sd_sites = np.where(np.isin(positions, tables.sites.position) == 0)[0]
    # Create new sites and add the mutations

    progress = progress_monitor.get("ms_extra_sites", len(new_sd_sites))
    for variant in sample_data.variants(sites=new_sd_sites):
        site = variant.site
        pos = site.position
        anc_state = site.ancestral_state
        anc_value = 0  # sample_data files always have 0 as the ancestral allele idx
        G = variant.genotypes[sample_id_map]
        # We can't perform parsimony inference if all sites are missing, and there's no
        # point if all non-missing sites are the ancestral state, so skip these cases
        if np.all(np.logical_or(G == tskit.MISSING_DATA, G == anc_value)):
            tables.sites.add_row(
                position=pos,
                ancestral_state=anc_state,
                metadata=_encode_metadata(
                    _update_site_metadata(
                        site.metadata, inference_type=constants.INFERENCE_NONE
                    )
                ),
            )
        else:
            while tree.interval[1] <= pos:
                tree = next(trees)
            inferred_anc_state, mapped_mutations = tree.map_mutations(
                G, variant.alleles
            )
            if anc_state is None:
                anc_state = inferred_anc_state
            new_site_id = tables.sites.add_row(
                position=pos,
                ancestral_state=anc_state,
                metadata=_encode_metadata(
                    _update_site_metadata(
                        site.metadata, inference_type=constants.INFERENCE_PARSIMONY
                    )
                ),
            )
            mut_map = {tskit.NULL: tskit.NULL}
            if anc_state != inferred_anc_state:
                # Need to add an extra mutation above the root to switch ancestral state
                for root in tree.roots:
                    mut_map[tskit.NULL] = tables.mutations.add_row(
                        site=new_site_id,
                        node=root,
                        derived_state=inferred_anc_state,
                        parent=tskit.NULL,
                    )
            for i, mutation in enumerate(mapped_mutations):
                mut_map[i] = tables.mutations.add_row(
                    site=new_site_id,
                    node=mutation.node,
                    derived_state=mutation.derived_state,
                    parent=mut_map[mutation.parent],
                )
        progress.update()
    progress.close()

    tables.sort()
    return tables.tree_sequence()


class AncestorsGenerator:
    """
    Manages the process of building ancestors.
    """

    def __init__(
        self,
        sample_data,
        ancestor_data,
        num_threads=0,
        engine=constants.C_ENGINE,
        progress_monitor=None,
    ):
        self.sample_data = sample_data
        self.ancestor_data = ancestor_data
        self.progress_monitor = _get_progress_monitor(
            progress_monitor, generate_ancestors=True
        )
        self.max_sites = sample_data.num_sites
        self.num_sites = 0
        self.num_samples = sample_data.num_samples
        self.num_threads = num_threads
        if engine == constants.C_ENGINE:
            logger.debug("Using C AncestorBuilder implementation")
            self.ancestor_builder = _tsinfer.AncestorBuilder(
                self.num_samples, self.max_sites
            )
        elif engine == constants.PY_ENGINE:
            logger.debug("Using Python AncestorBuilder implementation")
            self.ancestor_builder = algorithm.AncestorBuilder(
                self.num_samples, self.max_sites
            )
        else:
            raise ValueError(f"Unknown engine:{engine}")

    def add_sites(self, exclude_positions=None):
        """
        Add all sites that are suitable for inference into the
        ancestor builder (and subsequent inference), unless they
        are held in the specified list of excluded site positions.
        Suitable sites have only 2 listed alleles, with at least two
        samples carrying the derived allele and at least one sample
        carrying the ancestral allele.

        Suitable sites will be added at the time given by site.time, unless
        site.time is  ``np.nan`` or ``tskit.UNKNOWN_TIME``. In the first case,
        the site will simply excluded as if it were in the list of
        ``excluded_positions``. In the second case, then the time associated with
        the site will be the frequency of the derived allele (i.e. the number
        of samples with the derived allele divided by the total number of samples
        with non-missing alleles).
        """
        if exclude_positions is None:
            exclude_positions = set()
        else:
            exclude_positions = np.array(exclude_positions, dtype=np.float64)
            if len(exclude_positions.shape) != 1:
                raise ValueError("exclude_positions must be a 1D array of numbers")
        exclude_positions = set(exclude_positions)

        logger.info(f"Starting addition of {self.max_sites} sites")
        progress = self.progress_monitor.get("ga_add_sites", self.max_sites)
        inference_site_id = []
        for variant in self.sample_data.variants():
            # If there's missing data the last allele is None
            num_alleles = len(variant.alleles) - int(variant.alleles[-1] is None)
            counts = allele_counts(variant.genotypes)
            use_site = False
            site = variant.site
            if (
                site.position not in exclude_positions
                and num_alleles == 2
                and 1 < counts.derived < counts.known
            ):
                use_site = True
                time = site.time
                if tskit.is_unknown_time(time):
                    # Non-variable sites have no obvious freq-as-time values
                    assert counts.known != counts.derived
                    assert counts.known != counts.ancestral
                    time = counts.derived / counts.known
                if np.isnan(time):
                    use_site = False  # Site with meaningless time value: skip inference
            if use_site:
                self.ancestor_builder.add_site(time, variant.genotypes)
                inference_site_id.append(site.id)
                self.num_sites += 1
            progress.update()
        progress.close()
        self.ancestor_data.set_inference_sites(inference_site_id)
        logger.info("Finished adding sites")

    def _run_synchronous(self, progress):
        a = np.zeros(self.num_sites, dtype=np.int8)
        for t, focal_sites in self.descriptors:
            before = time.perf_counter()
            start, end = self.ancestor_builder.make_ancestor(focal_sites, a)
            duration = time.perf_counter() - before
            logger.debug(
                "Made ancestor in {:.2f}s at timepoint {} (epoch {}) "
                "from {} to {} (len={}) with {} focal sites ({})".format(
                    duration,
                    t,
                    self.timepoint_to_epoch[t],
                    start,
                    end,
                    end - start,
                    focal_sites.shape[0],
                    focal_sites,
                )
            )
            self.ancestor_data.add_ancestor(
                start=start,
                end=end,
                time=t,
                focal_sites=focal_sites,
                haplotype=a[start:end],
            )
            progress.update()

    def _run_threaded(self, progress):
        # This works by pushing the ancestor descriptors onto the build_queue,
        # which the worker threads pop off and process. We need to add ancestors
        # in the the ancestor_data object in the correct order, so we maintain
        # a priority queue (add_queue) which allows us to track the next smallest
        # index of the generated ancestor. We add build ancestors to this queue
        # as they are built, and drain it when we can.
        queue_depth = 8 * self.num_threads  # Seems like a reasonable limit
        build_queue = queue.Queue(queue_depth)
        add_lock = threading.Lock()
        next_add_index = 0
        add_queue = []

        def drain_add_queue():
            nonlocal next_add_index
            num_drained = 0
            while len(add_queue) > 0 and add_queue[0][0] == next_add_index:
                _, t, focal_sites, s, e, haplotype = heapq.heappop(add_queue)
                self.ancestor_data.add_ancestor(
                    start=s, end=e, time=t, focal_sites=focal_sites, haplotype=haplotype
                )
                progress.update()
                next_add_index += 1
                num_drained += 1
            logger.debug(f"Drained {num_drained} ancestors from add queue")

        def build_worker(thread_index):
            a = np.zeros(self.num_sites, dtype=np.int8)
            while True:
                work = build_queue.get()
                if work is None:
                    break
                index, t, focal_sites = work
                start, end = self.ancestor_builder.make_ancestor(focal_sites, a)
                with add_lock:
                    haplotype = a[start:end].copy()
                    heapq.heappush(
                        add_queue, (index, t, focal_sites, start, end, haplotype)
                    )
                    drain_add_queue()
                build_queue.task_done()
            build_queue.task_done()

        build_threads = [
            threads.queue_consumer_thread(
                build_worker, build_queue, name=f"build-worker-{j}", index=j
            )
            for j in range(self.num_threads)
        ]
        logger.debug(f"Started {self.num_threads} build worker threads")

        for index, (t, focal_sites) in enumerate(self.descriptors):
            build_queue.put((index, t, focal_sites))

        # Stop the the worker threads.
        for _ in range(self.num_threads):
            build_queue.put(None)
        for j in range(self.num_threads):
            build_threads[j].join()
        drain_add_queue()

    def run(self):
        self.descriptors = self.ancestor_builder.ancestor_descriptors()
        self.num_ancestors = len(self.descriptors)
        # Maps epoch numbers to their corresponding ancestor times.
        self.timepoint_to_epoch = {}
        for t, _ in reversed(self.descriptors):
            if t not in self.timepoint_to_epoch:
                self.timepoint_to_epoch[t] = len(self.timepoint_to_epoch) + 1
        if self.num_ancestors > 0:
            logger.info(f"Starting build for {self.num_ancestors} ancestors")
            progress = self.progress_monitor.get("ga_generate", self.num_ancestors)
            a = np.zeros(self.num_sites, dtype=np.int8)
            root_time = max(self.timepoint_to_epoch.keys()) + 1
            ultimate_ancestor_time = root_time + 1
            # Add the ultimate ancestor. This is an awkward hack really; we don't
            # ever insert this ancestor. The only reason to add it here is that
            # it makes sure that the ancestor IDs we have in the ancestor file are
            # the same as in the ancestor tree sequence. This seems worthwhile.
            self.ancestor_data.add_ancestor(
                start=0,
                end=self.num_sites,
                time=ultimate_ancestor_time,
                focal_sites=[],
                haplotype=a,
            )
            # Hack to ensure we always have a root with zeros at every position.
            self.ancestor_data.add_ancestor(
                start=0,
                end=self.num_sites,
                time=root_time,
                focal_sites=np.array([], dtype=np.int32),
                haplotype=a,
            )
            if self.num_threads <= 0:
                self._run_synchronous(progress)
            else:
                self._run_threaded(progress)
            progress.close()
            logger.info("Finished building ancestors")


class Matcher:
    def __init__(
        self,
        sample_data,
        inference_site_position,
        num_threads=1,
        path_compression=True,
        recombination_rate=None,
        mismatch_rate=None,
        precision=None,
        extended_checks=False,
        engine=constants.C_ENGINE,
        progress_monitor=None,
    ):
        self.sample_data = sample_data
        self.num_threads = num_threads
        self.path_compression = path_compression
        self.num_samples = self.sample_data.num_samples
        self.num_sites = len(inference_site_position)
        self.progress_monitor = _get_progress_monitor(progress_monitor)
        self.match_progress = None  # Allocated by subclass
        self.extended_checks = extended_checks
        # Map of site index to tree sequence position. Bracketing
        # values of 0 and L are used for simplicity.
        self.position_map = np.hstack(
            [inference_site_position, [sample_data.sequence_length]]
        )
        self.position_map[0] = 0

        if precision is None:
            # TODO Is this a good default? Need to investigate the effects.
            precision = 2

        if recombination_rate is None:
            # TODO is this a good value? Will need to tune
            recombination_rate = 1e-8

        self.recombination_rate = np.zeros(self.num_sites)
        # FIXME not quite right: we should check the rho[0] = 0
        self.recombination_rate[:] = recombination_rate
        if mismatch_rate is None:
            # Setting a very small value for now.
            mismatch_rate = 1e-20
        self.mismatch_rate = np.zeros(self.num_sites)
        self.mismatch_rate[:] = mismatch_rate
        self.precision = precision

        if engine == constants.C_ENGINE:
            logger.debug("Using C matcher implementation")
            self.tree_sequence_builder_class = _tsinfer.TreeSequenceBuilder
            self.ancestor_matcher_class = _tsinfer.AncestorMatcher
        elif engine == constants.PY_ENGINE:
            logger.debug("Using Python matcher implementation")
            self.tree_sequence_builder_class = algorithm.TreeSequenceBuilder
            self.ancestor_matcher_class = algorithm.AncestorMatcher
        else:
            raise ValueError(f"Unknown engine:{engine}")
        self.tree_sequence_builder = None

        all_sites = self.sample_data.sites_position[:]
        index = np.searchsorted(all_sites, inference_site_position)
        if not np.all(all_sites[index] == inference_site_position):
            raise ValueError(
                "Site positions for inference must be a subset of those in "
                "the sample data file."
            )
        num_alleles = sample_data.num_alleles()[index]
        self.inference_site_id = index

        # Allocate 64K nodes and edges initially. This will double as needed and will
        # quickly be big enough even for very large instances.
        max_edges = 64 * 1024
        max_nodes = 64 * 1024
        self.tree_sequence_builder = self.tree_sequence_builder_class(
            num_alleles=num_alleles, max_nodes=max_nodes, max_edges=max_edges
        )
        logger.debug(f"Allocated tree sequence builder with max_nodes={max_nodes}")

        # Allocate the matchers and statistics arrays.
        num_threads = max(1, self.num_threads)
        self.match = [
            np.full(self.num_sites, tskit.MISSING_DATA, np.int8)
            for _ in range(num_threads)
        ]
        self.results = ResultBuffer()
        self.mean_traceback_size = np.zeros(num_threads)
        self.num_matches = np.zeros(num_threads)
        self.matcher = [
            self.ancestor_matcher_class(
                self.tree_sequence_builder,
                recombination_rate=self.recombination_rate,
                mismatch_rate=self.mismatch_rate,
                precision=precision,
                extended_checks=self.extended_checks,
            )
            for _ in range(num_threads)
        ]

    def _find_path(self, child_id, haplotype, start, end, thread_index=0):
        """
        Finds the path of the specified haplotype and upates the results
        for the specified thread_index.
        """
        matcher = self.matcher[thread_index]
        match = self.match[thread_index]
        missing = haplotype == tskit.MISSING_DATA

        left, right, parent = matcher.find_path(haplotype, start, end, match)
        self.results.set_path(child_id, left, right, parent)
        match[missing] = tskit.MISSING_DATA
        diffs = start + np.where(haplotype[start:end] != match[start:end])[0]
        derived_state = haplotype[diffs]
        self.results.set_mutations(child_id, diffs.astype(np.int32), derived_state)

        self.match_progress.update()
        self.mean_traceback_size[thread_index] += matcher.mean_traceback_size
        self.num_matches[thread_index] += 1
        logger.debug(
            "matched node {}; num_edges={} tb_size={:.2f} match_mem={}".format(
                child_id,
                left.shape[0],
                matcher.mean_traceback_size,
                humanize.naturalsize(matcher.total_memory, binary=True),
            )
        )

    def convert_inference_mutations(self, tables):
        """
        Convert the mutations stored in the tree sequence builder into the output
        format.
        """
        mut_site, node, derived_state, _ = self.tree_sequence_builder.dump_mutations()
        mutation_id = 0
        num_mutations = len(mut_site)
        progress = self.progress_monitor.get(
            "ms_full_mutations", len(self.inference_site_id)
        )
        for site in self.sample_data.sites(self.inference_site_id):
            metadata = _update_site_metadata(site.metadata, constants.INFERENCE_FULL)
            site_id = tables.sites.add_row(
                site.position,
                ancestral_state=site.alleles[0],
                metadata=_encode_metadata(metadata),
            )
            while mutation_id < num_mutations and mut_site[mutation_id] == site_id:
                tables.mutations.add_row(
                    site_id,
                    node=node[mutation_id],
                    derived_state=site.alleles[derived_state[mutation_id]],
                )
                mutation_id += 1
            progress.update()
        progress.close()


class AncestorMatcher(Matcher):
    def __init__(self, sample_data, ancestor_data, **kwargs):
        super().__init__(sample_data, ancestor_data.sites_position[:], **kwargs)
        self.ancestor_data = ancestor_data
        self.num_ancestors = self.ancestor_data.num_ancestors
        self.epoch = self.ancestor_data.ancestors_time[:]

        # Add nodes for all the ancestors so that the ancestor IDs are equal
        # to the node IDs.
        for ancestor_id in range(self.num_ancestors):
            self.tree_sequence_builder.add_node(self.epoch[ancestor_id])
        self.ancestors = self.ancestor_data.ancestors()
        # Consume the first ancestor.
        a = next(self.ancestors, None)
        self.num_epochs = 0
        if a is not None:
            # assert np.array_equal(a.haplotype, np.zeros(self.num_sites, dtype=np.int8))
            # Create a list of all ID ranges in each epoch.
            breaks = np.where(self.epoch[1:] != self.epoch[:-1])[0]
            start = np.hstack([[0], breaks + 1])
            end = np.hstack([breaks + 1, [self.num_ancestors]])
            self.epoch_slices = np.vstack([start, end]).T
            self.num_epochs = self.epoch_slices.shape[0]
        self.start_epoch = 1

    def __epoch_info_dict(self, epoch_index):
        start, end = self.epoch_slices[epoch_index]
        return collections.OrderedDict(
            [("epoch", str(self.epoch[start])), ("nanc", str(end - start))]
        )

    def __ancestor_find_path(self, ancestor, thread_index=0):
        # NOTE we're no longer using the ancestor's focal sites as a way
        # of knowing where mutations happen but instead having a non-zero
        # mutation rate and letting the mismatches do the work. We might
        # want to have a version with a zero mutation rate.
        haplotype = np.full(self.num_sites, tskit.MISSING_DATA, dtype=np.int8)
        start = ancestor.start
        end = ancestor.end
        assert ancestor.haplotype.shape[0] == (end - start)
        haplotype[start:end] = ancestor.haplotype
        self._find_path(ancestor.id, haplotype, start, end, thread_index)

    def __start_epoch(self, epoch_index):
        start, end = self.epoch_slices[epoch_index]
        info = collections.OrderedDict(
            [("epoch", str(self.epoch[start])), ("nanc", str(end - start))]
        )
        self.progress_monitor.set_detail(info)
        self.tree_sequence_builder.freeze_indexes()

    def __complete_epoch(self, epoch_index):
        start, end = map(int, self.epoch_slices[epoch_index])
        num_ancestors_in_epoch = end - start
        current_time = self.epoch[start]
        nodes_before = self.tree_sequence_builder.num_nodes

        for child_id in range(start, end):
            left, right, parent = self.results.get_path(child_id)
            self.tree_sequence_builder.add_path(
                child_id,
                left,
                right,
                parent,
                compress=self.path_compression,
                extended_checks=self.extended_checks,
            )
            site, derived_state = self.results.get_mutations(child_id)
            self.tree_sequence_builder.add_mutations(child_id, site, derived_state)

        extra_nodes = self.tree_sequence_builder.num_nodes - nodes_before
        mean_memory = np.mean([matcher.total_memory for matcher in self.matcher])
        logger.debug(
            "Finished epoch {} with {} ancestors; {} extra nodes inserted; "
            "mean_tb_size={:.2f} edges={}; mean_matcher_mem={}".format(
                current_time,
                num_ancestors_in_epoch,
                extra_nodes,
                np.sum(self.mean_traceback_size) / np.sum(self.num_matches),
                self.tree_sequence_builder.num_edges,
                humanize.naturalsize(mean_memory, binary=True),
            )
        )
        self.mean_traceback_size[:] = 0
        self.num_matches[:] = 0
        self.results.clear()

    def __match_ancestors_single_threaded(self):
        for j in range(self.start_epoch, self.num_epochs):
            self.__start_epoch(j)
            start, end = map(int, self.epoch_slices[j])
            for ancestor_id in range(start, end):
                a = next(self.ancestors)
                assert ancestor_id == a.id
                self.__ancestor_find_path(a)
            self.__complete_epoch(j)

    def __match_ancestors_multi_threaded(self, start_epoch=1):
        # See note on match samples multithreaded below. Should combine these
        # into a single function. Possibly when trying to make the thread
        # error handling more robust.
        queue_depth = 8 * self.num_threads  # Seems like a reasonable limit
        match_queue = queue.Queue(queue_depth)

        def match_worker(thread_index):
            while True:
                work = match_queue.get()
                if work is None:
                    break
                self.__ancestor_find_path(work, thread_index)
                match_queue.task_done()
            match_queue.task_done()

        match_threads = [
            threads.queue_consumer_thread(
                match_worker, match_queue, name=f"match-worker-{j}", index=j
            )
            for j in range(self.num_threads)
        ]
        logger.debug(f"Started {self.num_threads} match worker threads")

        for j in range(self.start_epoch, self.num_epochs):
            self.__start_epoch(j)
            start, end = map(int, self.epoch_slices[j])
            for ancestor_id in range(start, end):
                a = next(self.ancestors)
                assert a.id == ancestor_id
                match_queue.put(a)
            # Block until all matches have completed.
            match_queue.join()
            self.__complete_epoch(j)

        # Stop the the worker threads.
        for _ in range(self.num_threads):
            match_queue.put(None)
        for j in range(self.num_threads):
            match_threads[j].join()

    def match_ancestors(self):
        logger.info(f"Starting ancestor matching for {self.num_epochs} epochs")
        self.match_progress = self.progress_monitor.get("ma_match", self.num_ancestors)
        if self.num_threads <= 0:
            self.__match_ancestors_single_threaded()
        else:
            self.__match_ancestors_multi_threaded()
        ts = self.store_output()
        self.match_progress.close()
        logger.info("Finished ancestor matching")
        return ts

    def get_ancestors_tree_sequence(self):
        """
        Return the ancestors tree sequence. Only inference sites are included in this
        tree sequence. All nodes have the sample flag bit set, and if a node
        corresponds to an ancestor in the ancestors file, it is indicated via metadata.
        """
        logger.debug("Building ancestors tree sequence")
        tsb = self.tree_sequence_builder

        tables = tskit.TableCollection(
            sequence_length=self.ancestor_data.sequence_length
        )

        flags, times = tsb.dump_nodes()
        pc_ancestors = is_pc_ancestor(flags)
        tables.nodes.set_columns(flags=flags, time=times)

        # Add metadata for any non-PC node, pointing to the original ancestor
        metadata = []
        ancestor = 0
        for is_pc in pc_ancestors:
            if is_pc:
                metadata.append(b"")
            else:
                metadata.append(_encode_metadata({"ancestor_data_id": ancestor}))
                ancestor += 1
        tables.nodes.packset_metadata(metadata)
        left, right, parent, child = tsb.dump_edges()
        tables.edges.set_columns(
            left=self.position_map[left],
            right=self.position_map[right],
            parent=parent,
            child=child,
        )

        self.convert_inference_mutations(tables)

        logger.debug("Sorting ancestors tree sequence")
        tables.sort()
        # Note: it's probably possible to compute the mutation parents from the
        # tsb data structures but we're not doing it for now.
        tables.build_index()
        tables.compute_mutation_parents()
        logger.debug("Sorting ancestors tree sequence done")
        for timestamp, record in self.ancestor_data.provenances():
            tables.provenances.add_row(timestamp=timestamp, record=json.dumps(record))
        record = provenance.get_provenance_dict(
            command="match_ancestors", source={"uuid": self.ancestor_data.uuid}
        )
        tables.provenances.add_row(record=json.dumps(record))
        logger.info(
            "Built ancestors tree sequence: {} nodes ({} pc ancestors); {} edges; "
            "{} sites; {} mutations".format(
                len(tables.nodes),
                np.sum(pc_ancestors),
                len(tables.edges),
                len(tables.mutations),
                len(tables.sites),
            )
        )
        return tables.tree_sequence()

    def store_output(self):
        if self.num_ancestors > 0:
            ts = self.get_ancestors_tree_sequence()
        else:
            # Allocate an empty tree sequence.
            tables = tskit.TableCollection(
                sequence_length=self.ancestor_data.sequence_length
            )
            ts = tables.tree_sequence()
        return ts


class SampleMatcher(Matcher):
    def __init__(self, sample_data, ancestors_ts, **kwargs):
        self.ancestors_ts_tables = ancestors_ts.dump_tables()
        super().__init__(sample_data, self.ancestors_ts_tables.sites.position, **kwargs)
        self.restore_tree_sequence_builder()
        # Map from input sample indexes (IDs in the SampleData file) to the
        # node ID in the tree sequence.
        self.sample_id_map = {}

    def restore_tree_sequence_builder(self):
        tables = self.ancestors_ts_tables
        if self.sample_data.sequence_length != tables.sequence_length:
            raise ValueError(
                "Ancestors tree sequence not compatible: sequence length is different to"
                " sample data file."
            )
        if np.any(tables.nodes.time <= 0):
            raise ValueError("All nodes must have time > 0")

        edges = tables.edges
        # Get the indexes into the position array.
        left = np.searchsorted(self.position_map, edges.left)
        if np.any(self.position_map[left] != edges.left):
            raise ValueError("Invalid left coordinates")
        right = np.searchsorted(self.position_map, edges.right)
        if np.any(self.position_map[right] != edges.right):
            raise ValueError("Invalid right coordinates")

        # Need to sort by child ID here and left so that we can efficiently
        # insert the child paths.
        index = np.lexsort((left, edges.child))
        nodes = tables.nodes
        self.tree_sequence_builder.restore_nodes(nodes.time, nodes.flags)
        self.tree_sequence_builder.restore_edges(
            left[index].astype(np.int32),
            right[index].astype(np.int32),
            edges.parent[index],
            edges.child[index],
        )

        mutations = tables.mutations
        derived_state = np.zeros(len(mutations), dtype=np.int8)
        mutation_site = mutations.site
        site_id = 0
        mutation_id = 0
        for site in self.sample_data.sites(self.inference_site_id):
            while (
                mutation_id < len(mutations) and mutation_site[mutation_id] == site_id
            ):
                allele = mutations[mutation_id].derived_state
                derived_state[mutation_id] = site.alleles.index(allele)
                mutation_id += 1
            site_id += 1
        self.tree_sequence_builder.restore_mutations(
            mutation_site, mutations.node, derived_state, mutations.parent
        )
        logger.info(
            "Loaded {} samples {} nodes; {} edges; {} sites; {} mutations".format(
                self.num_samples,
                len(nodes),
                len(edges),
                self.num_sites,
                len(mutations),
            )
        )

    def __process_sample(self, sample_id, haplotype, thread_index=0):
        self._find_path(sample_id, haplotype, 0, self.num_sites, thread_index)

    def __match_samples_single_threaded(self, indexes):
        sample_haplotypes = self.sample_data.haplotypes(
            indexes, sites=self.inference_site_id
        )
        for j, a in sample_haplotypes:
            assert len(a) == self.num_sites
            self.__process_sample(self.sample_id_map[j], a)

    def __match_samples_multi_threaded(self, indexes):
        # Note that this function is not almost identical to the match_ancestors
        # multithreaded function above. All we need to do is provide a function
        # to do the matching and some producer for the actual items and we
        # can bring this into a single function.

        queue_depth = 8 * self.num_threads  # Seems like a reasonable limit
        match_queue = queue.Queue(queue_depth)

        def match_worker(thread_index):
            while True:
                work = match_queue.get()
                if work is None:
                    break
                sample_id, a = work
                self.__process_sample(sample_id, a, thread_index)
                match_queue.task_done()
            match_queue.task_done()

        match_threads = [
            threads.queue_consumer_thread(
                match_worker, match_queue, name=f"match-worker-{j}", index=j
            )
            for j in range(self.num_threads)
        ]
        logger.debug(f"Started {self.num_threads} match worker threads")

        sample_haplotypes = self.sample_data.haplotypes(
            indexes, sites=self.inference_site_id
        )
        for j, a in sample_haplotypes:
            match_queue.put((self.sample_id_map[j], a))

        # Stop the the worker threads.
        for _ in range(self.num_threads):
            match_queue.put(None)
        for j in range(self.num_threads):
            match_threads[j].join()

    def match_samples(self, sample_indexes, sample_times):
        num_samples = len(sample_indexes)
        for j, t in zip(sample_indexes, sample_times):
            self.sample_id_map[j] = self.tree_sequence_builder.add_node(t)
        flags, times = self.tree_sequence_builder.dump_nodes()
        logger.info(f"Started matching for {num_samples} samples")
        if self.num_sites > 0:
            self.match_progress = self.progress_monitor.get("ms_match", num_samples)
            if self.num_threads <= 0:
                self.__match_samples_single_threaded(sample_indexes)
            else:
                self.__match_samples_multi_threaded(sample_indexes)
            self.match_progress.close()
            logger.info(
                "Inserting sample paths: {} edges in total".format(
                    self.results.total_edges
                )
            )
            progress_monitor = self.progress_monitor.get("ms_paths", num_samples)
            for j in sample_indexes:
                node_id = int(self.sample_id_map[j])
                left, right, parent = self.results.get_path(node_id)
                if np.any(times[node_id] > times[parent]):
                    p = parent[np.argmin(times[parent])]
                    raise ValueError(
                        f"Failed to put sample {j} (node {node_id}) at time "
                        f"{times[node_id]} as it has a younger parent (node {p})."
                    )
                self.tree_sequence_builder.add_path(
                    node_id, left, right, parent, compress=self.path_compression
                )
                site, derived_state = self.results.get_mutations(node_id)
                self.tree_sequence_builder.add_mutations(node_id, site, derived_state)
                progress_monitor.update()
            progress_monitor.close()

    def finalise(self, simplify, stabilise_node_ordering):
        logger.info("Finalising tree sequence")
        ts = self.get_samples_tree_sequence()
        if simplify:
            logger.info(
                "Running simplify(filter_sites=False, filter_populations=False, "
                "filter_individuals=False, keep_unary=True) on "
                f"{ts.num_nodes} nodes and {ts.num_edges} edges"
            )
            if stabilise_node_ordering:
                # Ensure all the node times are distinct so that they will have
                # stable IDs after simplifying. This could possibly also be done
                # by reversing the IDs within a time slice. This is used for comparing
                # tree sequences produced by perfect inference.
                tables = ts.dump_tables()
                times = tables.nodes.time
                for t in range(1, int(times[0])):
                    index = np.where(times == t)[0]
                    k = index.shape[0]
                    times[index] += np.arange(k)[::-1] / k
                tables.nodes.time = times
                tables.sort()
                ts = tables.tree_sequence()
            ts = ts.simplify(
                samples=list(self.sample_id_map.values()),
                filter_sites=False,
                filter_populations=False,
                filter_individuals=False,
                keep_unary=True,
            )
            logger.info(
                "Finished simplify; now have {} nodes and {} edges".format(
                    ts.num_nodes, ts.num_edges
                )
            )
        return ts

    def get_samples_tree_sequence(self, map_additional_sites=True):
        """
        Returns the current state of the build tree sequence. All samples and
        ancestors will have the sample node flag set. For correct sample reconstruction,
        the non-inference sites also need to be placed into the resulting tree sequence.
        """
        tsb = self.tree_sequence_builder

        tables = self.ancestors_ts_tables.copy()
        num_ancestral_individuals = len(tables.individuals)

        # Currently there's no information about populations etc stored in the
        # ancestors ts.
        for metadata in self.sample_data.populations_metadata[:]:
            tables.populations.add_row(_encode_metadata(metadata))
        for ind in self.sample_data.individuals():
            if ind.time != 0:
                ind.metadata["sample_data_time"] = ind.time
            tables.individuals.add_row(
                location=ind.location,
                metadata=_encode_metadata(ind.metadata),
                flags=ind.flags,
            )

        logger.debug("Adding tree sequence nodes")
        flags, times = tsb.dump_nodes()
        num_pc_ancestors = count_pc_ancestors(flags)

        # All true ancestors are samples in the ancestors tree sequence. We unset
        # the SAMPLE flag but keep other flags intact.
        new_flags = np.bitwise_and(tables.nodes.flags, ~tskit.NODE_IS_SAMPLE)
        tables.nodes.flags = new_flags.astype(np.uint32)
        sample_ids = list(self.sample_id_map.values())
        assert len(tables.nodes) == sample_ids[0]
        samples_metadata = self.sample_data.samples_metadata[:]
        individuals_population = self.sample_data.individuals_population[:]
        samples_individual = self.sample_data.samples_individual[:]
        individuals_time = self.sample_data.individuals_time[:]
        for index, sample_id in self.sample_id_map.items():
            individual = samples_individual[index]
            if individuals_time[individual] != 0:
                flags[sample_id] = np.bitwise_or(
                    flags[sample_id], constants.NODE_IS_HISTORICAL_SAMPLE
                )
            population = individuals_population[individual]
            tables.nodes.add_row(
                flags=flags[sample_id],
                time=times[sample_id],
                population=population,
                individual=num_ancestral_individuals + individual,
                metadata=_encode_metadata(samples_metadata[index]),
            )
        # Add in the remaining non-sample nodes.
        for u in range(sample_ids[-1] + 1, tsb.num_nodes):
            tables.nodes.add_row(flags=flags[u], time=times[u])

        logger.debug("Adding tree sequence edges")
        tables.edges.clear()
        left, right, parent, child = tsb.dump_edges()
        if self.num_sites == 0:
            # We have no inference sites, so no edges have been estimated. To ensure
            # we have a rooted tree, we add in edges for each sample to an artificial
            # root.
            assert left.shape[0] == 0
            root = tables.nodes.add_row(flags=0, time=tables.nodes.time.max() + 1)
            for sample_id in sample_ids:
                tables.edges.add_row(0, tables.sequence_length, root, sample_id)
        else:
            tables.edges.set_columns(
                left=self.position_map[left],
                right=self.position_map[right],
                parent=parent,
                child=child,
            )

        logger.debug("Sorting and building intermediate tree sequence.")
        tables.sites.clear()
        tables.mutations.clear()
        tables.sort()
        self.convert_inference_mutations(tables)

        # FIXME this is a shortcut. We should be computing the mutation parent above
        # during insertion (probably)
        tables.build_index()
        tables.compute_mutation_parents()

        # We don't have a source here because tree sequence files don't have a
        # UUID yet.
        record = provenance.get_provenance_dict(command="match-samples")
        tables.provenances.add_row(record=json.dumps(record))

        logger.info(
            "Built samples tree sequence: {} nodes ({} pc); {} edges; "
            "{} sites; {} mutations".format(
                len(tables.nodes),
                num_pc_ancestors,
                len(tables.edges),
                len(tables.sites),
                len(tables.mutations),
            )
        )
        ts = tables.tree_sequence()
        num_additional_sites = self.sample_data.num_sites - self.num_sites
        if map_additional_sites and num_additional_sites > 0:
            logger.info("Mapping additional sites")
            assert np.array_equal(ts.samples(), list(self.sample_id_map.values()))
            ts = insert_missing_sites(
                self.sample_data,
                ts,
                sample_id_map=np.array(list(self.sample_id_map.keys())),
                progress_monitor=self.progress_monitor,
            )
        else:
            logger.info("Skipping additional site mapping")

        return ts

    def get_augmented_ancestors_tree_sequence(self, sample_indexes):
        """
        Return the ancestors tree sequence augmented with samples as extra ancestors.
        """
        logger.debug("Building augmented ancestors tree sequence")
        tsb = self.tree_sequence_builder
        tables = self.ancestors_ts_tables.copy()

        flags, times = tsb.dump_nodes()
        s = 0
        num_pc_ancestors = 0
        for j in range(len(tables.nodes), len(flags)):
            if times[j] == 0.0:
                # This is an augmented ancestor node.
                tables.nodes.add_row(
                    flags=constants.NODE_IS_SAMPLE_ANCESTOR,
                    time=times[j],
                    metadata=_encode_metadata(
                        {"sample_data_id": int(sample_indexes[s])}
                    ),
                )
                s += 1
            else:
                # This is a path compressed node
                tables.nodes.add_row(flags=flags[j], time=times[j])
                assert is_pc_ancestor(flags[j])
                num_pc_ancestors += 1
        assert s == len(sample_indexes)
        assert len(tables.nodes) == len(flags)

        # Increment the time for all nodes so the augmented samples are no longer
        # at timepoint 0.
        tables.nodes.time = tables.nodes.time + 1

        # TODO - check this works for augmented ancestors with missing data
        left, right, parent, child = tsb.dump_edges()
        tables.edges.set_columns(
            left=self.position_map[left],
            right=self.position_map[right],
            parent=parent,
            child=child,
        )

        tables.sites.clear()
        tables.mutations.clear()
        self.convert_inference_mutations(tables)

        record = provenance.get_provenance_dict(command="augment_ancestors")
        tables.provenances.add_row(record=json.dumps(record))
        logger.debug("Sorting ancestors tree sequence")
        tables.sort()
        logger.debug("Sorting ancestors tree sequence done")
        logger.info(
            "Augmented ancestors tree sequence: {} nodes ({} extra pc ancestors); "
            "{} edges; {} sites; {} mutations".format(
                len(tables.nodes),
                num_pc_ancestors,
                len(tables.edges),
                len(tables.mutations),
                len(tables.sites),
            )
        )
        return tables.tree_sequence()


class ResultBuffer:
    """
    A wrapper for numpy arrays representing the results of a copying operations.
    """

    def __init__(self):
        self.paths = {}
        self.mutations = {}
        self.lock = threading.Lock()
        self.total_edges = 0

    def clear(self):
        """
        Clears this result buffer.
        """
        self.paths.clear()
        self.mutations.clear()
        self.total_edges = 0

    def set_path(self, node_id, left, right, parent):
        with self.lock:
            assert node_id not in self.paths
            self.paths[node_id] = left, right, parent
            self.total_edges += len(left)

    def set_mutations(self, node_id, site, derived_state):
        with self.lock:
            self.mutations[node_id] = site, derived_state

    def get_path(self, node_id):
        return self.paths[node_id]

    def get_mutations(self, node_id):
        return self.mutations[node_id]


def minimise(ts):
    """
    Returns a tree sequence with the minimal information required to represent
    the tree topologies at its sites.

    This is a convenience function used when we wish to use a subset of the
    sites in a tree sequence for ancestor matching. It is a thin-wrapper
    over the simplify method.
    """
    return ts.simplify(
        reduce_to_site_topology=True,
        filter_sites=False,
        filter_individuals=False,
        filter_populations=False,
    )
