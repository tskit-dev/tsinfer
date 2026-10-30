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

/* Global variables used for test in state in the test suite */

char *_tmp_file_name;
FILE *_devnull;

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
test_ancestor_builder_multi_site(void)
{
    /*
     * 8 samples, 6 sites at two time epochs.
     * Sites 0,2,4 at time 2.0 (younger focal sites)
     * Sites 1,3,5 at time 3.0 (older sites that appear "between" in time)
     *
     * This exercises compute_between_focal_sites, compute_ancestral_states,
     * get_site_genotypes_subset, get_consistent_samples, and the consensus/
     * disagree logic.
     */
    int ret = 0;
    ancestor_builder_t ab;
    size_t num_samples = 8;
    size_t max_sites = 6;
    size_t j;

    /* Sites at time 2.0: three different genotype patterns */
    allele_t g0[8] = { 1, 1, 0, 0, 0, 0, 0, 0 }; /* site 0, time 2 */
    allele_t g1[8] = { 1, 1, 1, 1, 0, 0, 0, 0 }; /* site 1, time 3 (older) */
    allele_t g2[8]
        = { 1, 1, 0, 0, 0, 0, 0, 0 }; /* site 2, time 2 (same pattern as g0) */
    allele_t g3[8] = { 1, 1, 1, 1, 0, 0, 0, 0 }; /* site 3, time 3 (older) */
    allele_t g4[8] = { 0, 0, 1, 1, 0, 0, 0, 0 }; /* site 4, time 2 (different pattern) */
    allele_t g5[8] = { 1, 1, 1, 1, 1, 1, 0, 0 }; /* site 5, time 3 (older) */

    allele_t ancestor[6];
    tsk_id_t start, end;

    ret = ancestor_builder_alloc(&ab, num_samples, max_sites, -1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = ancestor_builder_add_site(&ab, 2.0, g0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = ancestor_builder_add_site(&ab, 3.0, g1);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = ancestor_builder_add_site(&ab, 2.0, g2);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = ancestor_builder_add_site(&ab, 3.0, g3);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = ancestor_builder_add_site(&ab, 2.0, g4);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = ancestor_builder_add_site(&ab, 3.0, g5);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL(ab.num_sites, 6);

    /* Call print_state before finalise to cover check_state and inner loops */
    ancestor_builder_print_state(&ab, _devnull);

    ret = ancestor_builder_finalise(&ab);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_TRUE(ab.num_ancestors > 0);

    /* Build each ancestor and verify start <= end */
    for (j = 0; j < ab.num_ancestors; j++) {
        ret = ancestor_builder_make_ancestor(&ab, ab.descriptors[j].num_focal_sites,
            ab.descriptors[j].focal_sites, &start, &end, ancestor);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        CU_ASSERT_TRUE(start >= 0);
        CU_ASSERT_TRUE(end > start);
        CU_ASSERT_TRUE(end <= (tsk_id_t) max_sites);
    }

    ancestor_builder_free(&ab);
}

static void
test_ancestor_builder_stopping_condition(void)
{
    /*
     * 8 samples, 5 sites. Site 0 is focal at time 2.0; the rest are older.
     * The focal carriers are samples {0, 1}. Sample 1 disagrees with the
     * consensus at sites 1 and 3. Site 2 is missing for both carriers and
     * must preserve the disagreement. Removing sample 1 at site 3 halves
     * the sample set, so extension stops before site 4.
     */
    int ret = 0;
    ancestor_builder_t ab;
    size_t num_samples = 8;
    size_t max_sites = 5;
    allele_t g0[8] = { 1, 1, 0, 0, 0, 0, 0, 0 };   /* focal */
    allele_t g1[8] = { 1, 0, 1, 1, 0, 0, 0, 0 };   /* first disagreement */
    allele_t g2[8] = { -1, -1, 1, 1, 1, 0, 0, 0 }; /* missing for focal carriers */
    allele_t g3[8] = { 1, 0, 1, 1, 0, 0, 0, 0 };   /* second disagreement */
    allele_t g4[8] = { 1, 1, 1, 1, 0, 0, 0, 0 };   /* beyond the stopping point */
    allele_t ancestor[5];
    tsk_id_t focal_sites[] = { 0 };
    tsk_id_t start, end;

    ret = ancestor_builder_alloc(&ab, num_samples, max_sites, -1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = ancestor_builder_add_site(&ab, 2.0, g0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = ancestor_builder_add_site(&ab, 3.0, g1);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = ancestor_builder_add_site(&ab, 3.0, g2);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = ancestor_builder_add_site(&ab, 3.0, g3);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = ancestor_builder_add_site(&ab, 3.0, g4);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = ancestor_builder_finalise(&ab);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = ancestor_builder_make_ancestor(&ab, 1, focal_sites, &start, &end, ancestor);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL(start, 0);
    CU_ASSERT_EQUAL(end, 4);

    ancestor_builder_free(&ab);
}

static void
test_ancestor_builder_break_ancestor(void)
{
    /*
     * Construct a scenario where two focal sites share the same genotype
     * pattern but an intervening older site causes the ancestor to be split.
     *
     * Layout (8 samples, 5 sites):
     *   site 0: time 3.0  genotypes {1,1,1,1,0,0,0,0}  (older)
     *   site 1: time 2.0  genotypes {1,1,0,0,0,0,0,0}  (focal, pattern A)
     *   site 2: time 3.0  genotypes {1,0,1,0,0,0,0,0}  (older, disagreeing)
     *   site 3: time 2.0  genotypes {1,1,0,0,0,0,0,0}  (focal, pattern A)
     *   site 4: time 3.0  genotypes {1,1,1,1,0,0,0,0}  (older)
     *
     * Sites 1 and 3 have the same genotype pattern at the same time,
     * so they group into one pattern_map entry with 2 focal sites.
     * Site 2 is older (time 3.0 > 2.0) and sits between them.
     * The consistent samples for pattern A are {0, 1}.
     * At site 2 these samples have genotypes {1, 0} — not unanimous —
     * so break_ancestor should return true and split the ancestor.
     */
    int ret = 0;
    ancestor_builder_t ab;
    size_t num_samples = 8;
    size_t max_sites = 5;
    size_t j;

    allele_t g0[8] = { 1, 1, 1, 1, 0, 0, 0, 0 }; /* site 0, time 3 */
    allele_t g1[8] = { 1, 1, 0, 0, 0, 0, 0, 0 }; /* site 1, time 2 */
    allele_t g2[8] = { 1, 0, 1, 0, 0, 0, 0, 0 }; /* site 2, time 3 */
    allele_t g3[8] = { 1, 1, 0, 0, 0, 0, 0, 0 }; /* site 3, time 2 */
    allele_t g4[8] = { 1, 1, 1, 1, 0, 0, 0, 0 }; /* site 4, time 3 */

    allele_t ancestor[5];
    tsk_id_t start, end;
    size_t num_single_focal = 0;

    ret = ancestor_builder_alloc(&ab, num_samples, max_sites, -1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = ancestor_builder_add_site(&ab, 3.0, g0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = ancestor_builder_add_site(&ab, 2.0, g1);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = ancestor_builder_add_site(&ab, 3.0, g2);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = ancestor_builder_add_site(&ab, 2.0, g3);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = ancestor_builder_add_site(&ab, 3.0, g4);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = ancestor_builder_finalise(&ab);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    /* If break_ancestor split the time-2 pattern (sites 1,3), we should
     * see individual descriptors with 1 focal site each instead of a
     * single descriptor with 2 focal sites. Count how many descriptors
     * at time 2 have exactly 1 focal site. */
    for (j = 0; j < ab.num_ancestors; j++) {
        if (ab.descriptors[j].time == 2.0 && ab.descriptors[j].num_focal_sites == 1) {
            num_single_focal++;
        }
    }
    /* The two focal sites (1, 3) should be split into separate ancestors */
    CU_ASSERT_EQUAL(num_single_focal, 2);

    for (j = 0; j < ab.num_ancestors; j++) {
        ret = ancestor_builder_make_ancestor(&ab, ab.descriptors[j].num_focal_sites,
            ab.descriptors[j].focal_sites, &start, &end, ancestor);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        CU_ASSERT_TRUE(start >= 0);
        CU_ASSERT_TRUE(end > start);
    }

    ancestor_builder_free(&ab);
}

static void
test_ancestor_builder_one_bit_encoding(void)
{
    /*
     * Exercise the TSI_GENOTYPE_ENCODING_ONE_BIT code paths:
     *   - alloc sets encoded_genotypes_size = ceil(num_samples/8)
     *   - add_site calls packbits
     *   - make_ancestor calls unpackbits / one-bit subset extraction
     */
    int ret = 0;
    ancestor_builder_t ab;
    size_t num_samples = 10; /* not a multiple of 8 */
    size_t max_sites = 4;
    size_t j;

    allele_t g0[10] = { 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 }; /* time 3 (older) */
    allele_t g1[10] = { 1, 1, 0, 0, 0, 0, 0, 0, 0, 0 }; /* time 2 (focal) */
    allele_t g2[10] = { 1, 1, 1, 0, 0, 0, 0, 0, 0, 0 }; /* time 3 (older) */
    allele_t g3[10] = { 0, 0, 0, 0, 1, 1, 0, 0, 0, 0 }; /* time 2 (focal) */

    allele_t ancestor[4];
    tsk_id_t start, end;

    ret = ancestor_builder_alloc(
        &ab, num_samples, max_sites, -1, TSI_GENOTYPE_ENCODING_ONE_BIT);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    /* ceil(10/8) = 2 */
    CU_ASSERT_EQUAL(ab.encoded_genotypes_size, 2);

    ret = ancestor_builder_add_site(&ab, 3.0, g0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = ancestor_builder_add_site(&ab, 2.0, g1);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = ancestor_builder_add_site(&ab, 3.0, g2);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = ancestor_builder_add_site(&ab, 2.0, g3);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ancestor_builder_print_state(&ab, _devnull);

    ret = ancestor_builder_finalise(&ab);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_TRUE(ab.num_ancestors > 0);

    for (j = 0; j < ab.num_ancestors; j++) {
        ret = ancestor_builder_make_ancestor(&ab, ab.descriptors[j].num_focal_sites,
            ab.descriptors[j].focal_sites, &start, &end, ancestor);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        CU_ASSERT_TRUE(start >= 0);
        CU_ASSERT_TRUE(end > start);
    }

    ancestor_builder_free(&ab);
}

static void
test_ancestor_builder_mmap(void)
{
    /*
     * Exercise the mmap genotype storage path by providing a valid
     * read-write file descriptor.
     */
    int ret = 0;
    ancestor_builder_t ab;
    size_t num_samples = 4;
    size_t max_sites = 3;
    FILE *f;
    int fd;
    size_t j;

    allele_t g0[4] = { 1, 1, 0, 0 };
    allele_t g1[4] = { 1, 1, 1, 0 };
    allele_t g2[4] = { 1, 0, 1, 0 };

    allele_t ancestor[3];
    tsk_id_t start, end;

    f = fopen(_tmp_file_name, "w+");
    CU_ASSERT_FATAL(f != NULL);
    fd = fileno(f);

    ret = ancestor_builder_alloc(&ab, num_samples, max_sites, fd, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_TRUE(ab.mmap_buffer != NULL);

    ret = ancestor_builder_add_site(&ab, 3.0, g0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = ancestor_builder_add_site(&ab, 2.0, g1);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = ancestor_builder_add_site(&ab, 3.0, g2);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = ancestor_builder_finalise(&ab);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_TRUE(ab.num_ancestors > 0);

    for (j = 0; j < ab.num_ancestors; j++) {
        ret = ancestor_builder_make_ancestor(&ab, ab.descriptors[j].num_focal_sites,
            ab.descriptors[j].focal_sites, &start, &end, ancestor);
        CU_ASSERT_EQUAL_FATAL(ret, 0);
        CU_ASSERT_TRUE(end > start);
    }

    ancestor_builder_free(&ab);
    fclose(f);
}

static void
test_ancestor_builder_between_focal_sites(void)
{
    /*
     * One ancestor with focal sites 0 and 3 (the same pattern at time 2).
     * Between them, the older site 1 is ancestral in every carrier, and the
     * older site 2 is missing in every carrier, so neither breaks the
     * ancestor: it is 0 at site 1 and missing at site 2.
     */
    int ret = 0;
    ancestor_builder_t ab;
    allele_t g0[6] = { 1, 1, 1, 0, 0, 0 };
    allele_t g1[6] = { 0, 0, 0, 1, 1, 1 };
    allele_t g2[6] = { -1, -1, -1, 1, 1, 0 };
    allele_t g3[6] = { 1, 1, 1, 0, 0, 0 };
    allele_t ancestor[4];
    tsk_id_t start, end;
    size_t j;
    bool found = false;

    ret = ancestor_builder_alloc(&ab, 6, 4, -1, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = ancestor_builder_add_site(&ab, 2.0, g0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = ancestor_builder_add_site(&ab, 3.0, g1);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = ancestor_builder_add_site(&ab, 3.0, g2);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = ancestor_builder_add_site(&ab, 2.0, g3);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = ancestor_builder_finalise(&ab);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    for (j = 0; j < ab.num_ancestors; j++) {
        if (ab.descriptors[j].time == 2.0) {
            found = true;
            CU_ASSERT_EQUAL_FATAL(ab.descriptors[j].num_focal_sites, 2);
            CU_ASSERT_EQUAL(ab.descriptors[j].focal_sites[0], 0);
            CU_ASSERT_EQUAL(ab.descriptors[j].focal_sites[1], 3);
            ret = ancestor_builder_make_ancestor(
                &ab, 2, ab.descriptors[j].focal_sites, &start, &end, ancestor);
            CU_ASSERT_EQUAL_FATAL(ret, 0);
            CU_ASSERT_EQUAL(start, 0);
            CU_ASSERT_EQUAL(end, 4);
            CU_ASSERT_EQUAL(ancestor[0], 1);
            CU_ASSERT_EQUAL(ancestor[1], 0);
            CU_ASSERT_EQUAL(ancestor[2], TSK_MISSING_DATA);
            CU_ASSERT_EQUAL(ancestor[3], 1);
        }
    }
    CU_ASSERT_TRUE(found);
    ancestor_builder_free(&ab);
}

static void
test_ancestor_builder_one_bit_non_binary(void)
{
    int ret = 0;
    ancestor_builder_t ab;
    allele_t genotypes[4] = { 0, 1, 2, 0 };

    ret = ancestor_builder_alloc(&ab, 4, 2, -1, TSI_GENOTYPE_ENCODING_ONE_BIT);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = ancestor_builder_add_site(&ab, 1.0, genotypes);
    CU_ASSERT_EQUAL(ret, TSI_ERR_ONE_BIT_NON_BINARY);
    ancestor_builder_free(&ab);
}

static void
test_blkalloc(void)
{
    int ret;
    int j;
    tsi_blkalloc_t alloc;
    char *p;
    char *first[10];

    ret = tsi_blkalloc_init(&alloc, 0);
    CU_ASSERT_EQUAL(ret, TSK_ERR_BAD_PARAM_VALUE);
    tsi_blkalloc_free(&alloc);

    ret = tsi_blkalloc_init(&alloc, 64);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    /* Larger than a chunk */
    CU_ASSERT_PTR_NULL(tsi_blkalloc_get(&alloc, 65));
    /* Each block of 40 bytes needs a new chunk */
    for (j = 0; j < 10; j++) {
        p = tsi_blkalloc_get(&alloc, 40);
        CU_ASSERT_PTR_NOT_NULL_FATAL(p);
        memset(p, j, 40);
        first[j] = p;
    }
    CU_ASSERT_EQUAL(alloc.num_chunks, 10);
    CU_ASSERT_EQUAL(alloc.total_allocated, 400);
    CU_ASSERT_EQUAL(alloc.total_size, 10 * (64 + sizeof(void *)));
    for (j = 0; j < 10; j++) {
        CU_ASSERT_EQUAL(first[j][39], j);
    }
    tsi_blkalloc_print_state(&alloc, _devnull);

    /* After a reset, the chunks are reused */
    ret = tsi_blkalloc_reset(&alloc);
    CU_ASSERT_EQUAL(ret, 0);
    CU_ASSERT_EQUAL(alloc.total_allocated, 0);
    for (j = 0; j < 10; j++) {
        p = tsi_blkalloc_get(&alloc, 40);
        CU_ASSERT_EQUAL(p, first[j]);
    }
    CU_ASSERT_EQUAL(alloc.num_chunks, 10);
    tsi_blkalloc_free(&alloc);
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
        { "test_ancestor_builder_multi_site", test_ancestor_builder_multi_site },
        { "test_ancestor_builder_stopping_condition",
            test_ancestor_builder_stopping_condition },
        { "test_ancestor_builder_break_ancestor", test_ancestor_builder_break_ancestor },
        { "test_ancestor_builder_one_bit_encoding",
            test_ancestor_builder_one_bit_encoding },
        { "test_ancestor_builder_mmap", test_ancestor_builder_mmap },
        { "test_ancestor_builder_between_focal_sites",
            test_ancestor_builder_between_focal_sites },
        { "test_ancestor_builder_one_bit_non_binary",
            test_ancestor_builder_one_bit_non_binary },
        { "test_blkalloc", test_blkalloc },

        { "test_packbits_1", test_packbits_1 },
        { "test_packbits_2", test_packbits_2 },
        { "test_packbits_3", test_packbits_3 },
        { "test_packbits_4", test_packbits_4 },
        { "test_packbits_errors", test_packbits_errors },

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
