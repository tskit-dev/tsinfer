/*
** Copyright (C) 2020 University of Oxford
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

#include "tsinfer.h"

#include <float.h>
#include <limits.h>
#include <stdio.h>
#include <unistd.h>

#include <CUnit/Basic.h>

/* Global variables used for test in state in the test suite */

char * _tmp_file_name;
FILE * _devnull;

/* Verifies the tree sequence encodes the specified set of sample haplotypes. */
static void
verify_round_trip(tsk_table_collection_t *tables, size_t num_samples,
        size_t num_sites, allele_t **samples)
{
    int ret;
    tsk_treeseq_t ts;
    tsk_vargen_t vargen;
    tsk_variant_t *var;
    size_t j, k;

    ret = tsk_table_collection_sort(tables, 0, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = tsk_treeseq_init(&ts, tables, TSK_BUILD_INDEXES);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    /* tsk_treeseq_print_state(&ts, stdout); */

    CU_ASSERT_EQUAL_FATAL(num_samples, tsk_treeseq_get_num_samples(&ts));
    CU_ASSERT_EQUAL_FATAL(num_sites, tsk_treeseq_get_num_sites(&ts));
    ret = tsk_vargen_init(&vargen, &ts, NULL, 0, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    for (j = 0; j < num_sites; j++) {
        ret = tsk_vargen_next(&vargen, &var);
        /* printf("ret = %s\n", tsk_strerror(ret)); */
        CU_ASSERT_FATAL(ret >= 0);
        CU_ASSERT_EQUAL(var->site->id, j);
        CU_ASSERT_EQUAL(var->site->position, (double) j);
        for (k = 0; k < num_samples; k++) {
            CU_ASSERT_EQUAL(var->genotypes.u8[k], samples[k][j]);
        }
    }

    tsk_treeseq_free(&ts);
    tsk_vargen_free(&vargen);
}

static void
add_haplotype(tree_sequence_builder_t *tsb, ancestor_matcher_t *ancestor_matcher,
        tsk_id_t child, tsk_id_t start, tsk_id_t end, allele_t *haplotype)
{
    int ret;
    size_t num_sites = tsb->num_sites;
    size_t num_samples = tsb->num_sites;
    tsk_id_t *left, *right, *parent;
    size_t k, num_edges, num_mutations;
    allele_t *match = malloc(num_sites * sizeof(*match));
    allele_t *mutation_derived_state = malloc(num_samples * sizeof(*mutation_derived_state));
    tsk_id_t *mutation_site = malloc(num_samples * sizeof(*mutation_site));

    CU_ASSERT_FATAL(match != NULL);
    CU_ASSERT_FATAL(mutation_derived_state != NULL);
    CU_ASSERT_FATAL(mutation_site != NULL);

    /* printf("FIND PATH\n"); */
    /* ancestor_matcher_print_state(ancestor_matcher, stdout); */

    ret = ancestor_matcher_find_path(ancestor_matcher,
            start, end, haplotype, match,
            &num_edges, &left, &right, &parent);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    /* Add the edges for this match */
    ret = tree_sequence_builder_add_path(tsb,
            child, num_edges, left, right, parent, TSI_EXTENDED_CHECKS|TSI_COMPRESS_PATH);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    /* printf("start = %d end = %d num_edges = %d\n", (int) start, (int) end, (int) num_edges); */
    /* printf("hap = ["); */
    /* for (k = start; k < end; k++) { */
    /*     printf("%d, ", haplotype[k]); */
    /* } */
    /* printf("]\nmat = ["); */
    /* for (k = start; k < end; k++) { */
    /*     printf("%d, ", match[k]); */
    /* } */
    /* printf("]\n"); */

    num_mutations = 0;
    for (k = start; k < end; k++) {
        if (haplotype[k] != match[k]) {
            mutation_site[num_mutations] = k;
            CU_ASSERT_EQUAL_FATAL(haplotype[k], 1)
            mutation_derived_state[num_mutations] = haplotype[k];
            num_mutations++;
        }
    }
    ret = tree_sequence_builder_add_mutations(tsb,
            child, num_mutations, mutation_site, mutation_derived_state);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    free(match);
    free(mutation_derived_state);
    free(mutation_site);
}

static allele_t **
generate_random_haplotypes(size_t num_samples, size_t num_sites, size_t num_alleles,
        int seed)
{
    size_t j, k;
    allele_t **haplotypes = malloc(num_samples * sizeof(*haplotypes));
    allele_t *haplotype;

    srand(seed);
    CU_ASSERT_FATAL(haplotypes != NULL);
    for (j = 0; j < num_samples; j++) {
        haplotype = malloc(num_sites * sizeof(*haplotype));
        CU_ASSERT_FATAL(haplotype != NULL);
        haplotypes[j] = haplotype;

        for (k = 0; k < num_sites; k++) {
            haplotype[k] = (allele_t) abs(rand()) % num_alleles;
        }
    }
    return haplotypes;
}

static void
initialise_builder(tree_sequence_builder_t *tsb, double oldest_time)
{
    int ret;
    tsk_id_t left, right, parent, child;

    /* FIXME the tree generation algorithm currently assumes that we always have
     * this root edge. We should get rid of this when updating
     */

    /* Add the zero and root nodes */
    ret = tree_sequence_builder_add_node(tsb, oldest_time + 2, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tree_sequence_builder_add_node(tsb, oldest_time + 1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 1);

    /* Add the root edge */
    child = 1;
    parent = 0;
    left = 0;
    right = tsb->num_sites;
    ret = tree_sequence_builder_add_path(tsb,
        child, 1, &left, &right, &parent, TSI_EXTENDED_CHECKS|TSI_COMPRESS_PATH);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
}

static void
run_random_data(size_t num_samples, size_t num_sites, int seed)
{
    tsk_table_collection_t tables;
    ancestor_builder_t ancestor_builder;
    ancestor_matcher_t ancestor_matcher;
    tree_sequence_builder_t tsb;
    ancestor_descriptor_t ad;
    double *recombination_rate = calloc(num_sites, sizeof(double));
    double *mutation_rate = calloc(num_sites, sizeof(double));
    allele_t **samples = generate_random_haplotypes(num_samples, num_sites, 2, seed);
    allele_t *genotypes = malloc(num_samples * sizeof(*genotypes));
    allele_t *haplotype = malloc(num_sites * sizeof(*haplotype));
    double time;
    tsk_id_t child, start, end;
    size_t j, k;
    int ret;

    CU_ASSERT_FATAL(genotypes != NULL);
    CU_ASSERT_FATAL(haplotype != NULL);
    CU_ASSERT_FATAL(recombination_rate != NULL);
    CU_ASSERT_FATAL(mutation_rate != NULL);

    for (j = 0; j < num_sites; j++) {
        recombination_rate[j] = 1e-3;
        /* We need a nonzero mutation rate here */
        mutation_rate[j] = 1e-20;
    }

    CU_ASSERT_FATAL(num_samples >= 2);
    ret = ancestor_builder_alloc(&ancestor_builder, num_samples, num_sites, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tree_sequence_builder_alloc(&tsb, num_sites, 1, 1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = ancestor_matcher_alloc(&ancestor_matcher, &tsb, recombination_rate,
            mutation_rate, TSI_EXTENDED_CHECKS);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    for (j = 0; j < num_sites; j++) {
        time = 0;
        for (k = 0; k < num_samples; k++) {
            genotypes[k] = samples[k][j];
            time += genotypes[k];
        }
        ret = ancestor_builder_add_site(&ancestor_builder, j, time, genotypes);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
    }
    ret = ancestor_builder_finalise(&ancestor_builder);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    initialise_builder(&tsb, ancestor_builder.descriptors[0].time);

    time = -1;
    for (j = 0; j < ancestor_builder.num_ancestors; j++) {
        ad = ancestor_builder.descriptors[j];
        if (ad.time != time) {
            /* Finish the previous epoch */
            /* printf("NEW EPOCH: %f\n", ad.time); */
            ret = tree_sequence_builder_freeze_indexes(&tsb);
            CU_ASSERT_EQUAL_FATAL(ret, 0);
            time = ad.time;
        }
        ret = tree_sequence_builder_add_node(&tsb, ad.time, 0);
        CU_ASSERT_FATAL(ret >= 0);
        child = ret;
        /* Build the ancestral haplotype */
        ret = ancestor_builder_make_ancestor(&ancestor_builder,
                ad.num_focal_sites, ad.focal_sites, &start, &end,
                haplotype);
        CU_ASSERT_EQUAL_FATAL(ret, 0);

        add_haplotype(&tsb, &ancestor_matcher, child, start, end, haplotype);
        /* printf("\n"); */
        /* tree_sequence_builder_print_state(&tsb, stdout); */
        /* printf("\n-----------------\n"); */
    }

    /* Add the samples */
    ret = tree_sequence_builder_freeze_indexes(&tsb);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    for (j = 0; j < num_samples; j++) {
        ret = tree_sequence_builder_add_node(&tsb, 0, TSK_NODE_IS_SAMPLE);
        CU_ASSERT_FATAL(ret >= 0);
        child = ret;
        add_haplotype(&tsb, &ancestor_matcher, child, 0, num_sites, samples[j]);
    }
    ancestor_matcher_print_state(&ancestor_matcher, _devnull);
    tree_sequence_builder_print_state(&tsb, _devnull);

    ret = tree_sequence_builder_dump(&tsb, &tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    verify_round_trip(&tables, num_samples, num_sites, samples);

    ancestor_builder_free(&ancestor_builder);
    tree_sequence_builder_free(&tsb);
    ancestor_matcher_free(&ancestor_matcher);
    tsk_table_collection_free(&tables);

    for (j = 0; j < num_samples; j++) {
        free(samples[j]);
    }
    free(samples);
    free(haplotype);
    free(genotypes);
    free(recombination_rate);
    free(mutation_rate);
}

static void
test_ancestor_builder_one_site(void)
{
    int ret = 0;
    ancestor_builder_t ancestor_builder;
    allele_t genotypes[4] = {1, 1, 1, 1};
    allele_t ancestor[1];
    tsk_id_t start, end, focal_sites[1];

    ret = ancestor_builder_alloc(&ancestor_builder, 4, 1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = ancestor_builder_add_site(&ancestor_builder, 0, 4, genotypes);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = ancestor_builder_finalise(&ancestor_builder);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    CU_ASSERT_EQUAL_FATAL(ancestor_builder.num_ancestors, 1);
    CU_ASSERT_EQUAL(ancestor_builder.descriptors[0].time, 4);
    CU_ASSERT_EQUAL(ancestor_builder.descriptors[0].num_focal_sites, 1);
    CU_ASSERT_EQUAL(ancestor_builder.descriptors[0].focal_sites[0], 0);

    ancestor_builder_print_state(&ancestor_builder, _devnull);

    focal_sites[0] = 0;
    ret = ancestor_builder_make_ancestor(&ancestor_builder, 1,
        focal_sites, &start, &end, ancestor);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL(start, 0);
    CU_ASSERT_EQUAL(end, 1);
    CU_ASSERT_EQUAL(ancestor[0], 1);

    ancestor_builder_free(&ancestor_builder);
}

static void
test_matching_one_site(void)
{
    int ret;
    tsk_table_collection_t tables;
    ancestor_matcher_t ancestor_matcher;
    tree_sequence_builder_t tsb;
    allele_t haplotype[1] = {0};
    allele_t match[1];
    double recombination_rate = 0;
    double mutation_rate = 0;
    size_t num_edges;
    tsk_id_t *left, *right, *parent;

    ret = tree_sequence_builder_alloc(&tsb, 1, 1, 1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tree_sequence_builder_add_node(&tsb, 2.0, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tree_sequence_builder_freeze_indexes(&tsb);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = ancestor_matcher_alloc(&ancestor_matcher, &tsb,
            &recombination_rate, &mutation_rate, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = ancestor_matcher_find_path(&ancestor_matcher, 0, 1,
            haplotype, match, &num_edges, &left, &right, &parent);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL(num_edges, 1);
    CU_ASSERT_EQUAL(left[0], 0);
    CU_ASSERT_EQUAL(right[0], 1);
    CU_ASSERT_EQUAL(parent[0], 0);
    CU_ASSERT_EQUAL(match[0], 0);

    ret = tree_sequence_builder_add_node(&tsb, 1.0, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 1);
    ret = tree_sequence_builder_add_path(&tsb, 1, 1, left, right, parent, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tree_sequence_builder_freeze_indexes(&tsb);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = ancestor_matcher_find_path(&ancestor_matcher, 0, 1,
            haplotype, match, &num_edges, &left, &right, &parent);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL(num_edges, 1);
    CU_ASSERT_EQUAL(left[0], 0);
    CU_ASSERT_EQUAL(right[0], 1);
    CU_ASSERT_EQUAL(parent[0], 1);
    CU_ASSERT_EQUAL(match[0], 0);

    ret = tsk_table_collection_init(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tree_sequence_builder_dump(&tsb, &tables, TSK_NO_INIT);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL(tables.sequence_length, 1);
    CU_ASSERT_EQUAL(tables.nodes.num_rows, 2);
    CU_ASSERT_EQUAL(tables.edges.num_rows, 1);
    CU_ASSERT_EQUAL(tables.sites.num_rows, 1);
    CU_ASSERT_EQUAL(tables.mutations.num_rows, 0);

    ancestor_matcher_free(&ancestor_matcher);
    tree_sequence_builder_free(&tsb);
    tsk_table_collection_free(&tables);
}

static void
test_random_data_n5_m3(void)
{
    int seed;

    for (seed = 1; seed < 100; seed++) {
        /* printf("seed = %d\n", seed); */
        run_random_data(5, 3, seed);
    }
}


static void
test_random_data_n5_m20(void)
{
    int seed;

    for (seed = 1; seed < 10; seed++) {
        run_random_data(5, 20, seed);
    }
}

static void
test_random_data_n10_m10(void)
{
    run_random_data(10, 10, 43);
}

static void
test_random_data_n10_m100(void)
{
    run_random_data(10, 100, 43);
}

static void
test_random_data_n100_m10(void)
{
    run_random_data(10, 10, 1243);
}

static void
test_random_data_n100_m100(void)
{
    run_random_data(100, 100, 42);
}

static int
tsinfer_suite_init(void)
{
    int fd;
    static char template[] = "/tmp/tsi_c_test_XXXXXX";

    _tmp_file_name = NULL;
    _devnull = NULL;

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
    return CUE_SUCCESS;
}

static void
handle_cunit_error()
{
    fprintf(stderr, "CUnit error occured: %d: %s\n",
            CU_get_error(), CU_get_error_msg());
    exit(EXIT_FAILURE);
}

int
main(int argc, char **argv)
{
    int ret;
    CU_pTest test;
    CU_pSuite suite;
    CU_TestInfo tests[] = {
        {"test_ancestor_builder_one_site", test_ancestor_builder_one_site},
        /* TODO more ancestor builder tests */
        {"test_matching_one_site", test_matching_one_site},

        {"test_random_data_n5_m3", test_random_data_n5_m3},
        {"test_random_data_n5_m20", test_random_data_n5_m20},
        {"test_random_data_n10_m10", test_random_data_n10_m10},
        {"test_random_data_n10_m100", test_random_data_n10_m100},
        {"test_random_data_n100_m10", test_random_data_n100_m10},
        {"test_random_data_n100_m100", test_random_data_n100_m100},

        CU_TEST_INFO_NULL,
    };

    /* We use initialisers here as the struct definitions change between
     * versions of CUnit */
    CU_SuiteInfo suites[] = {
        {
            .pName = "tsinfer",
            .pInitFunc = tsinfer_suite_init,
            .pCleanupFunc = tsinfer_suite_cleanup,
            .pTests = tests
        },
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