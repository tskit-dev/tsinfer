/*
** Copyright (C) 2020-2023 University of Oxford
**
** This file is part of tsinfer.
**
** tsinfer is free software: you can redistribute it and/or modify
** it under the terms of the GNU General Public License as published by
** the Free Software Foundation, either version 3 of the License, or
** (at your option) any later version.
**
** tsinfer is distributed in the hope that it will be useful,
** but WITHOUT ANY WARRANTY; without even the implied warranty of
** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
** GNU General Public License for more details.
**
** You should have received a copy of the GNU General Public License
** along with tsinfer.  If not, see <http://www.gnu.org/licenses/>.
*/

/*
 * Unit tests for the low-level tsinfer API.
 */

#define _XOPEN_SOURCE 600 /* needed for fileno */

#include "tsinfer.h"
#include "tskit.h"

#include <float.h>
#include <limits.h>
#include <stdio.h>
#include <unistd.h>

#include <CUnit/Basic.h>

/* FIXME this needs to be updated somehow to allow the tests to be run from
 * different directories, i.e., with ninja -C build test
 */
#define TEST_DATA_DIR "test_data"

/* Global variables used for test in state in the test suite */

char *_tmp_file_name;
FILE *_devnull;

tsk_treeseq_t _single_tree_ex_ts;
/* 3.00┊    0    ┊ */
/*     ┊    ┃    ┊ */
/* 2.00┊    7    ┊ */
/*     ┊  ┏━┻━┓  ┊ */
/* 1.00┊  5   6  ┊ */
/*     ┊ ┏┻┓ ┏┻┓ ┊ */
/* 0.00┊ 1 2 3 4 ┊ */
/*     0         4 */
tsk_treeseq_t _multi_tree_ex_ts;
/* 1.84┊     0   ┊    0    ┊ */
/*     ┊     ┃   ┊    ┃    ┊ */
/* 0.84┊     8   ┊    8    ┊ */
/*     ┊   ┏━┻━┓ ┊  ┏━┻━┓  ┊ */
/* 0.42┊   ┃   ┃ ┊  7   ┃  ┊ */
/*     ┊   ┃   ┃ ┊ ┏┻┓  ┃  ┊ */
/* 0.05┊   6   ┃ ┊ ┃ ┃  ┃  ┊ */
/*     ┊ ┏━┻┓  ┃ ┊ ┃ ┃  ┃  ┊ */
/* 0.04┊ ┃  5  ┃ ┊ ┃ ┃  5  ┊ */
/*     ┊ ┃ ┏┻┓ ┃ ┊ ┃ ┃ ┏┻┓ ┊ */
/* 0.00┊ 1 2 3 4 ┊ 1 4 2 3 ┊ */
/*     0         2         4 */

static void
test_ancestor_builder_errors(void)
{
    int ret = 0;
    ancestor_builder_t ancestor_builder;
    allele_t genotypes_ones[4] = { 1, 1, 1, 1 };
    allele_t genotypes_zeros[4] = { 0, 0, 0, 0 };
    tsk_id_t start, end;
    allele_t haplotype[4];
    FILE *mmap_file;

    /* Bad file descriptor for mmap FD */
    ret = ancestor_builder_alloc(&ancestor_builder, 2, 1, -2, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSI_ERR_IO);
    ancestor_builder_free(&ancestor_builder);

    /* File is opened in the wrong mode */
    mmap_file = fopen(_tmp_file_name, "w");
    CU_ASSERT_FATAL(mmap_file != NULL);
    ret = ancestor_builder_alloc(&ancestor_builder, 2, 1, fileno(mmap_file), 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSI_ERR_IO);
    ancestor_builder_free(&ancestor_builder);
    fclose(mmap_file);

    ret = ancestor_builder_alloc(&ancestor_builder, 0, 1, -1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSI_ERR_BAD_NUM_SAMPLES);
    ancestor_builder_free(&ancestor_builder);

    ret = ancestor_builder_alloc(&ancestor_builder, 1, 1, -1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSI_ERR_BAD_NUM_SAMPLES);
    ancestor_builder_free(&ancestor_builder);

    ret = ancestor_builder_alloc(&ancestor_builder, 2, 0, -1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(ancestor_builder.num_sites, 0);
    ret = ancestor_builder_add_site(&ancestor_builder, 4, genotypes_ones);
    CU_ASSERT_EQUAL_FATAL(ret, TSI_ERR_TOO_MANY_SITES);
    ret = ancestor_builder_add_terminal_site(&ancestor_builder);
    CU_ASSERT_EQUAL_FATAL(ret, TSI_ERR_TOO_MANY_SITES);
    ancestor_builder_free(&ancestor_builder);

    ret = ancestor_builder_alloc(&ancestor_builder, 4, 2, -1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = ancestor_builder_add_site(&ancestor_builder, 4, genotypes_zeros);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = ancestor_builder_add_site(&ancestor_builder, 4, genotypes_ones);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL_FATAL(ancestor_builder.num_sites, 2);
    ret = ancestor_builder_finalise(&ancestor_builder);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = ancestor_builder_make_ancestor(&ancestor_builder,
        ancestor_builder.descriptors[0].num_focal_sites,
        ancestor_builder.descriptors[0].focal_sites, &start, &end, haplotype);
    CU_ASSERT_EQUAL_FATAL(ret, TSI_ERR_BAD_FOCAL_SITE);
    ancestor_builder_free(&ancestor_builder);
}

static void
test_ancestor_builder_one_site(void)
{
    int ret = 0;
    ancestor_builder_t ancestor_builder;
    allele_t genotypes[4] = { 1, 1, 1, 1 };
    allele_t ancestor[1];
    tsk_id_t start, end, focal_sites[1];

    ret = ancestor_builder_alloc(&ancestor_builder, 4, 1, -1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = ancestor_builder_add_site(&ancestor_builder, 4, genotypes);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = ancestor_builder_finalise(&ancestor_builder);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    CU_ASSERT_EQUAL_FATAL(ancestor_builder.num_ancestors, 1);
    CU_ASSERT_EQUAL(ancestor_builder.descriptors[0].time, 4);
    CU_ASSERT_EQUAL(ancestor_builder.descriptors[0].num_focal_sites, 1);
    CU_ASSERT_EQUAL(ancestor_builder.descriptors[0].focal_sites[0], 0);

    ancestor_builder_print_state(&ancestor_builder, _devnull);

    focal_sites[0] = 0;
    ret = ancestor_builder_make_ancestor(
        &ancestor_builder, 1, focal_sites, &start, &end, ancestor);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL(start, 0);
    CU_ASSERT_EQUAL(end, 1);
    CU_ASSERT_EQUAL(ancestor[0], 1);

    ancestor_builder_free(&ancestor_builder);
}

static void
test_ancestor_builder_terminal_site(void)
{
    int ret = 0;
    ancestor_builder_t ancestor_builder;
    allele_t genotypes[4] = { 1, 0, 1, 0 };
    allele_t ancestor[2];
    tsk_id_t start, end, focal_sites[1];

    ret = ancestor_builder_alloc(&ancestor_builder, 4, 2, -1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = ancestor_builder_add_site(&ancestor_builder, 4, genotypes);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = ancestor_builder_add_terminal_site(&ancestor_builder);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    focal_sites[0] = 0;
    ret = ancestor_builder_make_ancestor(
        &ancestor_builder, 1, focal_sites, &start, &end, ancestor);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL(start, 0);
    CU_ASSERT_EQUAL(end, 1);

    ancestor_builder_free(&ancestor_builder);
}

static void
test_matcher_indexes_errors(void)
{
    int ret;
    matcher_indexes_t mi;
    tsk_table_collection_t tables;

    memset(&mi, 0, sizeof(mi));
    ret = tsk_table_collection_init(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tables.sequence_length = 100;

    tsk_node_table_add_row(&tables.nodes, 0, 2.0, TSK_NULL, TSK_NULL, NULL, 0);
    tsk_node_table_add_row(&tables.nodes, 0, 1.0, TSK_NULL, TSK_NULL, NULL, 0);
    tsk_node_table_add_row(&tables.nodes, 0, 0.5, TSK_NULL, TSK_NULL, NULL, 0);

    tsk_edge_table_add_row(&tables.edges, 0, 100, 0, 1, NULL, 0);
    tsk_edge_table_add_row(&tables.edges, 0, 100, 1, 2, NULL, 0);

    tsk_site_table_add_row(&tables.sites, 10, "A", 1, NULL, 0);

    /* Two mutations at the same site */
    tsk_mutation_table_add_row(
        &tables.mutations, 0, 1, TSK_NULL, TSK_UNKNOWN_TIME, "T", 1, NULL, 0);
    tsk_mutation_table_add_row(
        &tables.mutations, 0, 2, 0, TSK_UNKNOWN_TIME, "C", 1, NULL, 0);

    ret = tsk_table_collection_sort(&tables, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_table_collection_build_index(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = matcher_indexes_alloc(&mi, &tables, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, TSI_ERR_MULTIPLE_MUTATIONS_AT_SITE);
    matcher_indexes_free(&mi);

    tsk_table_collection_free(&tables);
}

static void
test_packbits_1(void)
{
    int ret = 0;
    allele_t a[] = { 0, 1, 0, 1, 0, 1, 0, 0, 1 };
    uint8_t b[] = { 42, 1 };
    uint8_t bitpacked[100];
    allele_t bitunpacked[100];

    ret = packbits(a, sizeof(a), bitpacked);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL(memcmp(b, bitpacked, sizeof(b)), 0);
    unpackbits(b, sizeof(b), bitunpacked);
    CU_ASSERT_EQUAL(memcmp(a, bitunpacked, sizeof(a)), 0);
}

static void
test_packbits_2(void)
{
    int ret = 0;
    allele_t a[] = { 0, 1, 0, 1, 0, 1, 0, 0 };
    uint8_t b[] = { 42 };
    uint8_t bitpacked[100];
    allele_t bitunpacked[100];

    ret = packbits(a, sizeof(a), bitpacked);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL(memcmp(b, bitpacked, sizeof(b)), 0);
    unpackbits(b, sizeof(b), bitunpacked);
    CU_ASSERT_EQUAL(memcmp(a, bitunpacked, sizeof(a)), 0);
}

static void
test_packbits_3(void)
{
    int ret = 0;
    allele_t a[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    uint8_t b[] = { 255, 127 };
    uint8_t bitpacked[100];
    allele_t bitunpacked[100];

    ret = packbits(a, sizeof(a), bitpacked);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL(memcmp(b, bitpacked, sizeof(b)), 0);
    unpackbits(b, sizeof(b), bitunpacked);
    CU_ASSERT_EQUAL(memcmp(a, bitunpacked, sizeof(a)), 0);
}

static void
test_packbits_4(void)
{
    int ret = 0;
    allele_t a[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    uint8_t b[] = { 0, 0, 0 };
    uint8_t bitpacked[100];
    allele_t bitunpacked[100];

    ret = packbits(a, sizeof(a), bitpacked);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL(memcmp(b, bitpacked, sizeof(b)), 0);
    unpackbits(b, sizeof(b), bitunpacked);
    CU_ASSERT_EQUAL(memcmp(a, bitunpacked, sizeof(a)), 0);
}

static void
test_packbits_errors(void)
{
    int ret = 0;
    allele_t a[] = { 0 };
    uint8_t b[] = { 0 };

    a[0] = -1;
    ret = packbits(a, sizeof(a), b);
    CU_ASSERT_EQUAL_FATAL(ret, TSI_ERR_ONE_BIT_NON_BINARY);

    a[0] = 2;
    ret = packbits(a, sizeof(a), b);
    CU_ASSERT_EQUAL_FATAL(ret, TSI_ERR_ONE_BIT_NON_BINARY);
}

static int
run_match(const tsk_treeseq_t *ts, double rho, double mu, const allele_t *h,
    allele_t *match, tsk_size_t *path_length, tsk_id_t *left, tsk_id_t *right,
    tsk_id_t *parent)
{
    int ret;
    ancestor_matcher_t am;
    matcher_indexes_t mi;
    const size_t m = tsk_treeseq_get_num_sites(ts);
    double *recombination_rate = calloc(m, sizeof(*recombination_rate));
    double *mutation_rate = calloc(m, sizeof(*mutation_rate));
    size_t j;

    CU_ASSERT_FATAL(recombination_rate != NULL);
    CU_ASSERT_FATAL(mutation_rate != NULL);
    for (j = 0; j < m; j++) {
        mutation_rate[j] = mu;
        recombination_rate[j] = rho;
    }

    ret = matcher_indexes_alloc(&mi, ts->tables, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    /* matcher_indexes_print_state(&mi, stdout); */
    ret = ancestor_matcher_alloc(
        &am, &mi, recombination_rate, mutation_rate, DBL_MIN, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = ancestor_matcher_find_path(
        &am, 0, (tsk_id_t) m, h, match, path_length, left, right, parent);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    /* ancestor_matcher_print_state(&am, stdout); */

    ancestor_matcher_free(&am);
    matcher_indexes_free(&mi);
    free(recombination_rate);
    free(mutation_rate);

    return 0;
}

static void
check_matching_single_site_match(const tsk_treeseq_t *ts)
{
    allele_t h[] = { 0, 0, 0, 0 };
    allele_t match[4];
    tsk_id_t j, left[4], right[4], parent[4];
    tsk_size_t path_length;

    CU_ASSERT_EQUAL_FATAL(tsk_treeseq_get_num_sites(ts), 4);

    for (j = 0; j < 4; j++) {
        memset(h, 0, sizeof(h));
        h[j] = 1;
        run_match(ts, 1e-8, 0, h, match, &path_length, left, right, parent);
        CU_ASSERT_EQUAL_FATAL(path_length, 1);
        CU_ASSERT_EQUAL_FATAL(left[0], 0);
        CU_ASSERT_EQUAL_FATAL(right[0], 4);
        CU_ASSERT_EQUAL_FATAL(parent[0], j + 1);
    }
}

static void
test_matching_single_tree_single_site_match(void)
{
    check_matching_single_site_match(&_single_tree_ex_ts);
}

static void
test_matching_multi_tree_single_site_match(void)
{
    check_matching_single_site_match(&_multi_tree_ex_ts);
}

static void
check_matching_multi_switch(const tsk_treeseq_t *ts)
{
    allele_t h[] = { 1, 1, 1, 1 };
    allele_t match[4];
    tsk_id_t left[4], right[4], parent[4];
    tsk_size_t path_length;

    CU_ASSERT_EQUAL_FATAL(tsk_treeseq_get_num_sites(ts), 4);
    CU_ASSERT_EQUAL_FATAL(tsk_treeseq_get_sequence_length(ts), 4);

    run_match(ts, 1e-8, 0, h, match, &path_length, left, right, parent);
    CU_ASSERT_EQUAL_FATAL(path_length, 4);
    CU_ASSERT_EQUAL_FATAL(left[3], 0);
    CU_ASSERT_EQUAL_FATAL(right[3], 1);
    CU_ASSERT_EQUAL_FATAL(parent[3], 1);
    CU_ASSERT_EQUAL_FATAL(left[2], 1);
    CU_ASSERT_EQUAL_FATAL(right[2], 2);
    CU_ASSERT_EQUAL_FATAL(parent[2], 2);
    CU_ASSERT_EQUAL_FATAL(left[1], 2);
    CU_ASSERT_EQUAL_FATAL(right[1], 3);
    CU_ASSERT_EQUAL_FATAL(parent[1], 3);
    CU_ASSERT_EQUAL_FATAL(left[0], 3);
    CU_ASSERT_EQUAL_FATAL(right[0], 4);
    CU_ASSERT_EQUAL_FATAL(parent[0], 4);
}

static void
test_matching_single_tree_multi_switch(void)
{
    check_matching_multi_switch(&_single_tree_ex_ts);
}

static void
test_matching_multi_tree_multi_switch(void)
{
    check_matching_multi_switch(&_multi_tree_ex_ts);
}

/* ===================================================================
 * tsinfer-topology fixture builders
 *
 * These mirror the Python fixtures in test_matcher_fixtures.py.
 * Each builds a tsk_treeseq_t with the exact same nodes, edges,
 * sites and mutations.
 * =================================================================== */

static void
build_star_ts(tsk_treeseq_t *ts)
{
    int ret;
    tsk_table_collection_t tables;

    ret = tsk_table_collection_init(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tables.sequence_length = 100;
    /* Nodes: 0=ultimate root, 1=virtual root, 2,3=leaves */
    tsk_node_table_add_row(&tables.nodes, 0, 2.0, TSK_NULL, TSK_NULL, NULL, 0);
    tsk_node_table_add_row(&tables.nodes, 0, 1.0, TSK_NULL, TSK_NULL, NULL, 0);
    tsk_node_table_add_row(&tables.nodes, 0, 0.5, TSK_NULL, TSK_NULL, NULL, 0);
    tsk_node_table_add_row(&tables.nodes, 0, 0.3, TSK_NULL, TSK_NULL, NULL, 0);
    /* Edges: all span [0,100) for valid tree sequence coverage */
    tsk_edge_table_add_row(&tables.edges, 0, 100, 1, 2, NULL, 0);
    tsk_edge_table_add_row(&tables.edges, 0, 100, 1, 3, NULL, 0);
    tsk_edge_table_add_row(&tables.edges, 0, 100, 0, 1, NULL, 0);
    /* Sites at positions 10, 20, 30 */
    tsk_site_table_add_row(&tables.sites, 10, "A", 1, NULL, 0);
    tsk_site_table_add_row(&tables.sites, 20, "A", 1, NULL, 0);
    tsk_site_table_add_row(&tables.sites, 30, "A", 1, NULL, 0);
    /* Mutations: node 3 at site 0, node 2 at site 1 */
    tsk_mutation_table_add_row(
        &tables.mutations, 0, 3, TSK_NULL, TSK_UNKNOWN_TIME, "T", 1, NULL, 0);
    tsk_mutation_table_add_row(
        &tables.mutations, 1, 2, TSK_NULL, TSK_UNKNOWN_TIME, "T", 1, NULL, 0);

    ret = tsk_table_collection_sort(&tables, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_treeseq_init(ts, &tables, TSK_TS_INIT_BUILD_INDEXES);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tsk_table_collection_free(&tables);
}

static void
build_binary_ts(tsk_treeseq_t *ts)
{
    int ret;
    tsk_table_collection_t tables;

    ret = tsk_table_collection_init(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tables.sequence_length = 100;
    /* Nodes */
    tsk_node_table_add_row(&tables.nodes, 0, 2.0, TSK_NULL, TSK_NULL, NULL, 0);
    tsk_node_table_add_row(&tables.nodes, 0, 1.0, TSK_NULL, TSK_NULL, NULL, 0);
    tsk_node_table_add_row(&tables.nodes, 0, 0.7, TSK_NULL, TSK_NULL, NULL, 0);
    tsk_node_table_add_row(&tables.nodes, 0, 0.4, TSK_NULL, TSK_NULL, NULL, 0);
    tsk_node_table_add_row(&tables.nodes, 0, 0.2, TSK_NULL, TSK_NULL, NULL, 0);
    /* Edges */
    tsk_edge_table_add_row(&tables.edges, 0, 100, 2, 3, NULL, 0);
    tsk_edge_table_add_row(&tables.edges, 0, 100, 2, 4, NULL, 0);
    tsk_edge_table_add_row(&tables.edges, 0, 100, 1, 2, NULL, 0);
    tsk_edge_table_add_row(&tables.edges, 0, 100, 0, 1, NULL, 0);
    /* Sites at 10, 20, 30, 40 */
    tsk_site_table_add_row(&tables.sites, 10, "A", 1, NULL, 0);
    tsk_site_table_add_row(&tables.sites, 20, "A", 1, NULL, 0);
    tsk_site_table_add_row(&tables.sites, 30, "A", 1, NULL, 0);
    tsk_site_table_add_row(&tables.sites, 40, "A", 1, NULL, 0);
    /* Mutations: node 2 at site 0, node 3 at site 2, node 4 at site 3 */
    tsk_mutation_table_add_row(
        &tables.mutations, 0, 2, TSK_NULL, TSK_UNKNOWN_TIME, "T", 1, NULL, 0);
    tsk_mutation_table_add_row(
        &tables.mutations, 2, 3, TSK_NULL, TSK_UNKNOWN_TIME, "T", 1, NULL, 0);
    tsk_mutation_table_add_row(
        &tables.mutations, 3, 4, TSK_NULL, TSK_UNKNOWN_TIME, "T", 1, NULL, 0);

    ret = tsk_table_collection_sort(&tables, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_treeseq_init(ts, &tables, TSK_TS_INIT_BUILD_INDEXES);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tsk_table_collection_free(&tables);
}

static void
build_two_tree_ts(tsk_treeseq_t *ts)
{
    int ret;
    tsk_table_collection_t tables;

    ret = tsk_table_collection_init(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tables.sequence_length = 100;
    /* Nodes */
    tsk_node_table_add_row(&tables.nodes, 0, 2.0, TSK_NULL, TSK_NULL, NULL, 0);
    tsk_node_table_add_row(&tables.nodes, 0, 1.0, TSK_NULL, TSK_NULL, NULL, 0);
    tsk_node_table_add_row(&tables.nodes, 0, 0.6, TSK_NULL, TSK_NULL, NULL, 0);
    tsk_node_table_add_row(&tables.nodes, 0, 0.3, TSK_NULL, TSK_NULL, NULL, 0);
    /* Edges: breakpoint at 30 */
    tsk_edge_table_add_row(&tables.edges, 30, 100, 2, 3, NULL, 0);
    tsk_edge_table_add_row(&tables.edges, 0, 100, 1, 2, NULL, 0);
    tsk_edge_table_add_row(&tables.edges, 0, 30, 1, 3, NULL, 0);
    tsk_edge_table_add_row(&tables.edges, 0, 100, 0, 1, NULL, 0);
    /* Sites at 10, 20, 30, 40 */
    tsk_site_table_add_row(&tables.sites, 10, "A", 1, NULL, 0);
    tsk_site_table_add_row(&tables.sites, 20, "A", 1, NULL, 0);
    tsk_site_table_add_row(&tables.sites, 30, "A", 1, NULL, 0);
    tsk_site_table_add_row(&tables.sites, 40, "A", 1, NULL, 0);
    /* Mutations: node 2 at site 0, node 3 at site 2 */
    tsk_mutation_table_add_row(
        &tables.mutations, 0, 2, TSK_NULL, TSK_UNKNOWN_TIME, "T", 1, NULL, 0);
    tsk_mutation_table_add_row(
        &tables.mutations, 2, 3, TSK_NULL, TSK_UNKNOWN_TIME, "T", 1, NULL, 0);

    ret = tsk_table_collection_sort(&tables, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_treeseq_init(ts, &tables, TSK_TS_INIT_BUILD_INDEXES);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tsk_table_collection_free(&tables);
}

static void
build_deep_chain_ts(tsk_treeseq_t *ts)
{
    int ret;
    tsk_table_collection_t tables;

    ret = tsk_table_collection_init(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tables.sequence_length = 100;
    /* Nodes: root -> A -> B -> C */
    tsk_node_table_add_row(&tables.nodes, 0, 2.0, TSK_NULL, TSK_NULL, NULL, 0);
    tsk_node_table_add_row(&tables.nodes, 0, 1.0, TSK_NULL, TSK_NULL, NULL, 0);
    tsk_node_table_add_row(&tables.nodes, 0, 0.8, TSK_NULL, TSK_NULL, NULL, 0);
    tsk_node_table_add_row(&tables.nodes, 0, 0.5, TSK_NULL, TSK_NULL, NULL, 0);
    tsk_node_table_add_row(&tables.nodes, 0, 0.2, TSK_NULL, TSK_NULL, NULL, 0);
    /* Edges */
    tsk_edge_table_add_row(&tables.edges, 0, 100, 3, 4, NULL, 0);
    tsk_edge_table_add_row(&tables.edges, 0, 100, 2, 3, NULL, 0);
    tsk_edge_table_add_row(&tables.edges, 0, 100, 1, 2, NULL, 0);
    tsk_edge_table_add_row(&tables.edges, 0, 100, 0, 1, NULL, 0);
    /* Sites at 10, 20, 30, 40 */
    tsk_site_table_add_row(&tables.sites, 10, "A", 1, NULL, 0);
    tsk_site_table_add_row(&tables.sites, 20, "A", 1, NULL, 0);
    tsk_site_table_add_row(&tables.sites, 30, "A", 1, NULL, 0);
    tsk_site_table_add_row(&tables.sites, 40, "A", 1, NULL, 0);
    /* Mutations: node 2 at site 0, node 3 at site 1, node 4 at site 2 */
    tsk_mutation_table_add_row(
        &tables.mutations, 0, 2, TSK_NULL, TSK_UNKNOWN_TIME, "T", 1, NULL, 0);
    tsk_mutation_table_add_row(
        &tables.mutations, 1, 3, TSK_NULL, TSK_UNKNOWN_TIME, "T", 1, NULL, 0);
    tsk_mutation_table_add_row(
        &tables.mutations, 2, 4, TSK_NULL, TSK_UNKNOWN_TIME, "T", 1, NULL, 0);

    ret = tsk_table_collection_sort(&tables, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_treeseq_init(ts, &tables, TSK_TS_INIT_BUILD_INDEXES);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tsk_table_collection_free(&tables);
}

/* ===================================================================
 * tsinfer-topology matching tests
 * =================================================================== */

static void
test_matching_star_ts(void)
{
    tsk_treeseq_t ts;
    allele_t match[3];
    tsk_id_t left[3], right[3], parent[3];
    tsk_size_t path_length;

    build_star_ts(&ts);
    /* [0,1,0] copies node 2 */
    allele_t h1[] = { 0, 1, 0 };
    run_match(&ts, 1e-9, 0, h1, match, &path_length, left, right, parent);
    CU_ASSERT_EQUAL(path_length, 1);
    CU_ASSERT_EQUAL(left[0], 0);
    CU_ASSERT_EQUAL(right[0], 100);
    CU_ASSERT_EQUAL(parent[0], 2);

    /* [1,0,0] copies node 3 */
    allele_t h2[] = { 1, 0, 0 };
    run_match(&ts, 1e-9, 0, h2, match, &path_length, left, right, parent);
    CU_ASSERT_EQUAL(path_length, 1);
    CU_ASSERT_EQUAL(left[0], 0);
    CU_ASSERT_EQUAL(right[0], 100);
    CU_ASSERT_EQUAL(parent[0], 3);

    tsk_treeseq_free(&ts);
}

static void
test_matching_binary_ts(void)
{
    tsk_treeseq_t ts;
    allele_t match[4];
    tsk_id_t left[4], right[4], parent[4];
    tsk_size_t path_length;

    build_binary_ts(&ts);
    /* [1,0,1,0] copies node 3 */
    allele_t h1[] = { 1, 0, 1, 0 };
    run_match(&ts, 1e-9, 0, h1, match, &path_length, left, right, parent);
    CU_ASSERT_EQUAL(path_length, 1);
    CU_ASSERT_EQUAL(left[0], 0);
    CU_ASSERT_EQUAL(right[0], 100);
    CU_ASSERT_EQUAL(parent[0], 3);

    /* [1,0,0,0] copies internal node 2 */
    allele_t h2[] = { 1, 0, 0, 0 };
    run_match(&ts, 1e-9, 0, h2, match, &path_length, left, right, parent);
    CU_ASSERT_EQUAL(path_length, 1);
    CU_ASSERT_EQUAL(left[0], 0);
    CU_ASSERT_EQUAL(right[0], 100);
    CU_ASSERT_EQUAL(parent[0], 2);

    tsk_treeseq_free(&ts);
}

static void
test_matching_two_tree_ts(void)
{
    tsk_treeseq_t ts;
    allele_t match[4];
    tsk_id_t left[4], right[4], parent[4];
    tsk_size_t path_length;

    build_two_tree_ts(&ts);
    /* [1,0,0,0] copies node 2 */
    allele_t h1[] = { 1, 0, 0, 0 };
    run_match(&ts, 1e-9, 0, h1, match, &path_length, left, right, parent);
    CU_ASSERT_EQUAL(path_length, 1);
    CU_ASSERT_EQUAL(left[0], 0);
    CU_ASSERT_EQUAL(right[0], 100);
    CU_ASSERT_EQUAL(parent[0], 2);

    tsk_treeseq_free(&ts);
}

static void
test_matching_deep_chain_ts(void)
{
    tsk_treeseq_t ts;
    allele_t match[4];
    tsk_id_t left[4], right[4], parent[4];
    tsk_size_t path_length;

    build_deep_chain_ts(&ts);
    /* [1,1,1,0] copies node 4 (C) */
    allele_t h1[] = { 1, 1, 1, 0 };
    run_match(&ts, 1e-9, 0, h1, match, &path_length, left, right, parent);
    CU_ASSERT_EQUAL(path_length, 1);
    CU_ASSERT_EQUAL(left[0], 0);
    CU_ASSERT_EQUAL(right[0], 100);
    CU_ASSERT_EQUAL(parent[0], 4);

    /* [1,0,0,0] copies node 2 (A) */
    allele_t h2[] = { 1, 0, 0, 0 };
    run_match(&ts, 1e-9, 0, h2, match, &path_length, left, right, parent);
    CU_ASSERT_EQUAL(path_length, 1);
    CU_ASSERT_EQUAL(left[0], 0);
    CU_ASSERT_EQUAL(right[0], 100);
    CU_ASSERT_EQUAL(parent[0], 2);

    tsk_treeseq_free(&ts);
}

static void
test_matching_triallelic_ts(void)
{
    /*
     * Test that num_alleles > 2 is accepted.
     * Use the star topology but tell the matcher that site 0 has 3 alleles.
     * Query with allele 2 at site 0 should not trigger BAD_HAPLOTYPE_ALLELE.
     */
    tsk_treeseq_t ts;
    ancestor_matcher_t am;
    matcher_indexes_t mi;
    allele_t match[3];
    tsk_id_t left[3], right[3], parent[3];
    tsk_size_t path_length;
    tsk_size_t num_alleles[] = { 3, 2, 2 };
    double rho[] = { 1e-9, 1e-9, 1e-9 };
    double mu[] = { 1e-20, 1e-20, 1e-20 };
    int ret;

    build_star_ts(&ts);

    ret = matcher_indexes_alloc(&mi, ts.tables, num_alleles, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = ancestor_matcher_alloc(&am, &mi, rho, mu, 14, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    /* Query [2,0,0]: allele 2 at site 0 needs num_alleles >= 3 */
    allele_t h[] = { 2, 0, 0 };
    ret = ancestor_matcher_find_path(
        &am, 0, 3, h, match, &path_length, left, right, parent);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL(path_length, 1);

    ancestor_matcher_free(&am);
    matcher_indexes_free(&mi);
    tsk_treeseq_free(&ts);
}

static void
test_strerror(void)
{
    int j;
    const char *msg;
    int max_error_code = 8192; /* totally arbitrary */

    for (j = 0; j < max_error_code; j++) {
        msg = tsi_strerror(-j);
        CU_ASSERT_FATAL(msg != NULL);
        CU_ASSERT(strlen(msg) > 0);
    }
    CU_ASSERT_STRING_EQUAL(
        tsi_strerror(0), "Normal exit condition. This is not an error!");
}

static int
tsinfer_suite_init(void)
{
    int ret, fd;
    static char template[] = "/tmp/tsi_c_test_XXXXXX";

    _tmp_file_name = NULL;
    _devnull = NULL;
    memset(&_single_tree_ex_ts, 0, sizeof(_single_tree_ex_ts));
    memset(&_multi_tree_ex_ts, 0, sizeof(_multi_tree_ex_ts));

    _tmp_file_name = malloc(sizeof(template));
    if (_tmp_file_name == NULL) {
        return CUE_NOMEMORY;
    }
    strcpy(_tmp_file_name, template);
    fd = mkstemp(_tmp_file_name);
    if (fd == -1) {
        return CUE_SINIT_FAILED;
    }
    close(fd);
    _devnull = fopen("/dev/null", "w");
    if (_devnull == NULL) {
        return CUE_SINIT_FAILED;
    }

    ret = tsk_treeseq_load(
        &_single_tree_ex_ts, TEST_DATA_DIR "/single_tree_example.trees", 0);
    if (ret != 0) {
        return CUE_SINIT_FAILED;
    }
    ret = tsk_treeseq_load(
        &_multi_tree_ex_ts, TEST_DATA_DIR "/multi_tree_example.trees", 0);
    if (ret != 0) {
        return CUE_SINIT_FAILED;
    }

    return CUE_SUCCESS;
}

static int
tsinfer_suite_cleanup(void)
{
    if (_tmp_file_name != NULL) {
        unlink(_tmp_file_name);
        free(_tmp_file_name);
    }
    if (_devnull != NULL) {
        fclose(_devnull);
    }
    tsk_treeseq_free(&_single_tree_ex_ts);
    tsk_treeseq_free(&_multi_tree_ex_ts);
    return CUE_SUCCESS;
}

static void
handle_cunit_error()
{
    fprintf(stderr, "CUnit error occured: %d: %s\n", CU_get_error(), CU_get_error_msg());
    exit(EXIT_FAILURE);
}

int
main(int argc, char **argv)
{
    int ret;
    CU_pTest test;
    CU_pSuite suite;
    CU_TestInfo tests[] = {
        { "test_ancestor_builder_errors", test_ancestor_builder_errors },
        { "test_ancestor_builder_one_site", test_ancestor_builder_one_site },
        { "test_ancestor_builder_terminal_site", test_ancestor_builder_terminal_site },

        { "test_matcher_indexes_errors", test_matcher_indexes_errors },

        { "test_packbits_1", test_packbits_1 },
        { "test_packbits_2", test_packbits_2 },
        { "test_packbits_3", test_packbits_3 },
        { "test_packbits_4", test_packbits_4 },
        { "test_packbits_errors", test_packbits_errors },

        { "test_matching_single_tree_single_site_match",
            test_matching_single_tree_single_site_match },
        { "test_matching_multi_tree_single_site_match",
            test_matching_multi_tree_single_site_match },
        { "test_matching_single_tree_multi_switch",
            test_matching_single_tree_multi_switch },
        { "test_matching_multi_tree_multi_switch",
            test_matching_multi_tree_multi_switch },

        { "test_matching_star_ts", test_matching_star_ts },
        { "test_matching_binary_ts", test_matching_binary_ts },
        { "test_matching_two_tree_ts", test_matching_two_tree_ts },
        { "test_matching_deep_chain_ts", test_matching_deep_chain_ts },
        { "test_matching_triallelic_ts", test_matching_triallelic_ts },

        { "test_strerror", test_strerror },

        CU_TEST_INFO_NULL,
    };

    /* We use initialisers here as the struct definitions change between
     * versions of CUnit */
    CU_SuiteInfo suites[] = {
        { .pName = "tsinfer",
            .pInitFunc = tsinfer_suite_init,
            .pCleanupFunc = tsinfer_suite_cleanup,
            .pTests = tests },
        CU_SUITE_INFO_NULL,
    };
    if (CUE_SUCCESS != CU_initialize_registry()) {
        handle_cunit_error();
    }
    if (CUE_SUCCESS != CU_register_suites(suites)) {
        handle_cunit_error();
    }
    CU_basic_set_mode(CU_BRM_VERBOSE);

    if (argc == 1) {
        CU_basic_run_tests();
    } else if (argc == 2) {
        suite = CU_get_suite_by_name("tsinfer", CU_get_registry());
        if (suite == NULL) {
            printf("Suite not found\n");
            return EXIT_FAILURE;
        }
        test = CU_get_test_by_name(argv[1], suite);
        if (test == NULL) {
            printf("Test '%s' not found\n", argv[1]);
            return EXIT_FAILURE;
        }
        CU_basic_run_test(suite, test);
    } else {
        printf("usage: ./test_ancestor_builder <test_name>\n");
        return EXIT_FAILURE;
    }

    ret = EXIT_SUCCESS;
    if (CU_get_number_of_tests_failed() != 0) {
        printf("Test failed!\n");
        ret = EXIT_FAILURE;
    }
    CU_cleanup_registry();
    return ret;
}
