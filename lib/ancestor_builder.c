/*
** Copyright (C) 2018-2023 University of Oxford
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
/* It's not worth trying to get mmap'd genotypes working on windows,
 * and is just a silent no-op if it's tried.
 */
#if defined(_WIN32)
#else
/* Needed for ftruncate */
#define _XOPEN_SOURCE 700
#define MMAP_GENOTYPES 1
#endif

#include "tsinfer.h"
#include "err.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>

#ifdef MMAP_GENOTYPES
#include <sys/mman.h>
#include <unistd.h>
#include <sys/types.h>
#endif

#include "avl.h"

/* Note: using an unoptimised version of bit packing here because we're
 * only doing it once per site, and so it shouldn't matter much how fast
 * the implementation is. */
int
packbits(const allele_t *restrict source, size_t len, uint8_t *restrict dest)
{
    int ret = 0;
    size_t j, k, i;
    int x = 0;

    k = 0;
    i = 0;
    for (j = 0; j < len; j++) {
        if (source[j] < 0 || source[j] > 1) {
            ret = TSI_ERR_ONE_BIT_NON_BINARY;
            goto out;
        }
        if (j % 8 == 0 && j > 0) {
            dest[k] = (uint8_t) x;
            k++;
            i = 0;
            x = 0;
        }
        x += source[j] << i;
        i++;
    }
    dest[k] = (uint8_t) x;
out:
    return ret;
}

/* NOTE: this is a simple initial version, it will probably be worth having
 * more highly tuned versions of this, using e.g., AVX registeres */
void
unpackbits(const uint8_t *restrict source, size_t len, allele_t *restrict dest)
{
    size_t j, k, i;
    int v;

    k = 0;
    for (j = 0; j < len; j++) {
        /* I'm assuming any compiler will unroll this? */
        for (i = 0; i < 8; i++) {
            v = source[j] & (1 << i);
            dest[k + i] = (allele_t) v != 0;
        }
        k += 8;
    }
}

static int
cmp_time_map(const void *a, const void *b)
{
    const time_map_t *ia = (time_map_t const *) a;
    const time_map_t *ib = (time_map_t const *) b;
    return (ia->time > ib->time) - (ia->time < ib->time);
}

static int
cmp_pattern_map(const void *a, const void *b)
{
    const pattern_map_t *ia = (pattern_map_t const *) a;
    const pattern_map_t *ib = (pattern_map_t const *) b;
    int ret = memcmp(
        ia->encoded_genotypes, ib->encoded_genotypes, ia->encoded_genotypes_size);
    return ret;
}

static void
ancestor_builder_check_state(const ancestor_builder_t *self)
{
    size_t count;
    avl_node_t *a, *b;
    pattern_map_t *pattern_map;
    time_map_t *time_map;
    site_list_t *s;

    assert(self->decoded_genotypes_size >= self->num_samples);

    for (a = self->time_map.head; a != NULL; a = a->next) {
        time_map = (time_map_t *) a->item;
        for (b = time_map->pattern_map.head; b != NULL; b = b->next) {
            pattern_map = (pattern_map_t *) b->item;
            assert(pattern_map->encoded_genotypes_size == self->encoded_genotypes_size);
            count = 0;
            for (s = pattern_map->sites; s != NULL; s = s->next) {
                /* printf("HIT\n"); */
                assert(self->sites[s->site].time == time_map->time);
                assert(self->sites[s->site].encoded_genotypes
                       == pattern_map->encoded_genotypes);
                count++;
            }
            assert(pattern_map->num_sites == count);
        }
    }
}

int
ancestor_builder_print_state(ancestor_builder_t *self, FILE *out)
{
    size_t j, k;
    avl_node_t *a, *b;
    pattern_map_t *pattern_map;
    time_map_t *time_map;
    site_list_t *s;

    fprintf(out, "Ancestor builder\n");
    fprintf(out, "flags = %d\n", (int) self->flags);
    fprintf(out, "mmap_fd = %d\n", self->mmap_fd);
    fprintf(out, "num_samples = %d\n", (int) self->num_samples);
    fprintf(out, "num_sites = %d\n", (int) self->num_sites);
    fprintf(out, "num_ancestors = %d\n", (int) self->num_ancestors);
    fprintf(out, "mem_size = %d\n", (int) ancestor_builder_get_memsize(self));
    fprintf(out, "encoded_genotypes_size = %d\n", (int) self->encoded_genotypes_size);
    fprintf(out, "decoded_genotypes_size = %d\n", (int) self->decoded_genotypes_size);

    fprintf(out, "Sites:\n");
    for (j = 0; j < self->num_sites; j++) {
        fprintf(out, "%d\t%d\n", (int) j, (int) self->sites[j].time);
    }
    fprintf(out, "Time map:\n");

    for (a = self->time_map.head; a != NULL; a = a->next) {
        time_map = (time_map_t *) a->item;
        fprintf(out, "Epoch: time = %f: %d ancestors\n", time_map->time,
            avl_count(&time_map->pattern_map));
        for (b = time_map->pattern_map.head; b != NULL; b = b->next) {
            pattern_map = (pattern_map_t *) b->item;
            fprintf(out, "\t%p\t[", (void *) pattern_map->encoded_genotypes);
            for (k = 0; k < self->encoded_genotypes_size; k++) {
                fprintf(out, "%d,", pattern_map->encoded_genotypes[k]);
            }
            fprintf(out, "]\t");
            for (s = pattern_map->sites; s != NULL; s = s->next) {
                fprintf(out, "%d (%p)", s->site,
                    (void *) self->sites[s->site].encoded_genotypes);
            }
            fprintf(out, "\n");
        }
    }
    fprintf(out, "Descriptors:\n");
    for (j = 0; j < self->num_ancestors; j++) {
        fprintf(out, "%f\t%d: ", self->descriptors[j].time,
            (int) self->descriptors[j].num_focal_sites);
        for (k = 0; k < self->descriptors[j].num_focal_sites; k++) {
            fprintf(out, "%d, ", self->descriptors[j].focal_sites[k]);
        }
        fprintf(out, "\n");
    }
    tsk_blkalloc_print_state(&self->main_allocator, out);
    tsk_blkalloc_print_state(&self->indexing_allocator, out);
    ancestor_builder_check_state(self);
    return 0;
}

#ifdef MMAP_GENOTYPES

static int
ancestor_builder_make_genotype_mmap(ancestor_builder_t *self)
{

    int ret = 0;

    self->mmap_size = self->max_sites * self->encoded_genotypes_size;
    if (ftruncate(self->mmap_fd, (off_t) self->mmap_size) != 0) {
        ret = TSI_ERR_IO;
        goto out;
    }
    self->mmap_buffer = mmap(
        NULL, self->mmap_size, PROT_READ | PROT_WRITE, MAP_SHARED, self->mmap_fd, 0);
    if (self->mmap_buffer == MAP_FAILED) {
        self->mmap_buffer = NULL;
        ret = TSI_ERR_IO;
        goto out;
    }
    self->mmap_offset = 0;
out:
    return ret;
}

static int
ancestor_builder_free_genotype_mmap(ancestor_builder_t *self)
{
    if (self->mmap_buffer != NULL) {
        /* There's nothing we can do about it here, so don't check errors. */
        munmap(self->mmap_buffer, self->mmap_size);
    }
    /* Try to truncate to zero so we don't flush out all the data */
    ftruncate(self->mmap_fd, 0);
    return 0;
}
#endif

int
ancestor_builder_alloc(ancestor_builder_t *self, size_t num_samples, size_t max_sites,
    int mmap_fd, int flags)
{
    int ret = 0;
    unsigned long max_size = 1024 * 1024;

    memset(self, 0, sizeof(ancestor_builder_t));
    self->num_samples = num_samples;
    self->max_sites = max_sites;
    self->mmap_fd = mmap_fd;
    self->num_sites = 0;
    self->flags = flags;

    if (num_samples <= 1) {
        ret = TSI_ERR_BAD_NUM_SAMPLES;
        goto out;
    }
    if (self->flags & TSI_GENOTYPE_ENCODING_ONE_BIT) {
        self->encoded_genotypes_size = (num_samples / 8) + ((num_samples % 8) != 0);
        self->decoded_genotypes_size = self->encoded_genotypes_size * 8;
    } else {
        self->encoded_genotypes_size = num_samples * sizeof(allele_t);
        self->decoded_genotypes_size = self->encoded_genotypes_size;
    }
    self->sites = calloc(max_sites, sizeof(site_t));
    self->descriptors = calloc(max_sites, sizeof(ancestor_descriptor_t));
    self->genotype_encode_buffer = calloc(self->encoded_genotypes_size, 1);
    if (self->sites == NULL || self->descriptors == NULL
        || self->genotype_encode_buffer == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }

    /* Pre-calculate the maximum sizes asked for in other methods when calling
     * tsk_blkalloc_get(&self->allocator, ...)  */
    max_size = TSK_MAX(128 * self->num_samples * sizeof(allele_t), max_size);
    /* NB: using self->max_sites below is probably overkill: the real number should be
     * the maximum number of focal sites in a single ancestor, usually << max_sites */
    max_size = TSK_MAX(self->max_sites * sizeof(tsk_id_t), max_size);
    ret = tsk_blkalloc_init(&self->main_allocator, max_size);
    if (ret != 0) {
        goto out;
    }
    ret = tsk_blkalloc_init(&self->indexing_allocator, max_size);
    if (ret != 0) {
        goto out;
    }
#if MMAP_GENOTYPES
    if (self->mmap_fd != -1) {
        ret = ancestor_builder_make_genotype_mmap(self);
        if (ret != 0) {
            goto out;
        }
    }
#endif
    avl_init_tree(&self->time_map, cmp_time_map, NULL);
out:
    return ret;
}

size_t
ancestor_builder_get_memsize(const ancestor_builder_t *self)
{
    /* Ignore the other allocs as insignificant, and don't report the
     * size of the mmap'd region */
    return self->main_allocator.total_size + self->indexing_allocator.total_size;
}

int
ancestor_builder_free(ancestor_builder_t *self)
{
#if MMAP_GENOTYPES
    if (self->mmap_fd != -1) {
        ancestor_builder_free_genotype_mmap(self);
    }
#endif
    tsi_safe_free(self->sites);
    tsi_safe_free(self->descriptors);
    tsk_safe_free(self->genotype_encode_buffer);
    tsk_blkalloc_free(&self->main_allocator);
    tsk_blkalloc_free(&self->indexing_allocator);
    return 0;
}

static time_map_t *
ancestor_builder_get_time_map(ancestor_builder_t *self, double time)
{
    time_map_t *ret = NULL;
    time_map_t search, *time_map;
    avl_node_t *avl_node;

    search.time = time;
    avl_node = avl_search(&self->time_map, &search);
    if (avl_node == NULL) {
        avl_node = tsk_blkalloc_get(&self->indexing_allocator, sizeof(*avl_node));
        time_map = tsk_blkalloc_get(&self->indexing_allocator, sizeof(*time_map));
        if (avl_node == NULL || time_map == NULL) {
            goto out;
        }
        time_map->time = time;

        avl_init_tree(&time_map->pattern_map, cmp_pattern_map, NULL);
        avl_init_node(avl_node, time_map);
        avl_node = avl_insert_node(&self->time_map, avl_node);
        assert(avl_node != NULL);
        ret = time_map;
    } else {
        ret = (time_map_t *) avl_node->item;
    }
out:
    return ret;
}

static void
ancestor_builder_get_site_genotypes_subset(const ancestor_builder_t *self, tsk_id_t site,
    const tsk_id_t *samples, size_t num_samples, allele_t *restrict dest)
{
    size_t j;
    const uint8_t *restrict encoded = self->sites[site].encoded_genotypes;
    tsk_id_t u;
    uint8_t byte;
    int v, bit_index;
    allele_t *g = dest;

    if (self->flags & TSI_GENOTYPE_ENCODING_ONE_BIT) {
        for (j = 0; j < num_samples; j++) {
            u = samples[j];
            byte = encoded[u / 8];
            bit_index = u % 8;
            v = byte & (1 << bit_index);
            g[j] = (allele_t) v != 0;
        }
    } else {
        for (j = 0; j < num_samples; j++) {
            g[j] = (allele_t) encoded[samples[j]];
        }
    }
}

static void
ancestor_builder_get_site_genotypes(
    const ancestor_builder_t *self, tsk_id_t site, allele_t *restrict dest)
{
    uint8_t *restrict encoded = self->sites[site].encoded_genotypes;

    if (self->flags & TSI_GENOTYPE_ENCODING_ONE_BIT) {
        unpackbits(encoded, self->encoded_genotypes_size, dest);
    } else {
        memcpy(dest, self->sites[site].encoded_genotypes, self->num_samples);
    }
}

static inline void
ancestor_builder_get_consistent_samples(const ancestor_builder_t *self, tsk_id_t site,
    tsk_id_t *samples, size_t *num_samples, allele_t *restrict genotypes)
{
    tsk_id_t j, k;
    ancestor_builder_get_site_genotypes(self, site, genotypes);

    k = 0;
    for (j = 0; j < (tsk_id_t) self->num_samples; j++) {
        if (genotypes[j] == 1) {
            samples[k] = j;
            k++;
        }
    }
    *num_samples = (size_t) k;
}

static int
ancestor_builder_compute_ancestral_states(const ancestor_builder_t *self, int direction,
    tsk_id_t focal_site, allele_t *ancestor, tsk_id_t *restrict sample_set,
    bool *restrict disagree, tsk_id_t *last_site_ret, allele_t *restrict genotypes)
{
    int ret = 0;
    tsk_id_t last_site = focal_site;
    int64_t l;
    tsk_id_t u;
    size_t j, ones, zeros, tmp_size, sample_set_size, min_sample_set_size;
    double focal_site_time = self->sites[focal_site].time;
    const site_t *restrict sites = self->sites;
    const size_t num_sites = self->num_sites;
    allele_t consensus;

    ancestor_builder_get_consistent_samples(
        self, focal_site, sample_set, &sample_set_size, genotypes);
    /* This can't happen because we've already tested for it in
     * ancestor_builder_compute_between_focal_sites */
    assert(sample_set_size > 0);
    memset(disagree, 0, self->num_samples * sizeof(*disagree));
    min_sample_set_size = sample_set_size / 2;

    /* printf("site=%d, direction=%d min_sample_size=%d\n", (int) focal_site, direction,
     */
    /*         (int) min_sample_set_size); */
    for (l = focal_site + direction; l >= 0 && l < (int64_t) num_sites; l += direction) {
        /* printf("\tl = %d\n", (int) l); */
        ancestor[l] = 0;
        last_site = (tsk_id_t) l;
        if (sites[l].time > focal_site_time) {

            /* printf("\t%d\t%d:", (int) l, (int) sample_set_size); */
            /* for (j = 0; j < sample_set_size; j++) { */
            /*     printf("%d, ", sample_set[j]); */
            /* } */
            /* printf("\n"); */

            ancestor_builder_get_site_genotypes_subset(
                self, (tsk_id_t) l, sample_set, sample_set_size, genotypes);
            ones = 0;
            zeros = 0;
            for (j = 0; j < sample_set_size; j++) {
                switch (genotypes[j]) {
                    case 0:
                        zeros++;
                        break;
                    case 1:
                        ones++;
                        break;
                }
            }
            if (ones + zeros == 0) {
                ancestor[l] = TSK_MISSING_DATA;
            } else {
                if (ones >= zeros) {
                    consensus = 1;
                } else {
                    consensus = 0;
                }
                /* printf("\t:ones=%d, consensus=%d\n", (int) ones, consensus); */
                /* fflush(stdout); */
                for (j = 0; j < sample_set_size; j++) {
                    u = sample_set[j];
                    if (disagree[u] && (genotypes[j] != consensus)
                        && (genotypes[j] != TSK_MISSING_DATA)) {
                        /* This sample has disagreed with consensus twice in a row,
                         * so remove it */
                        /* printf("\t\tremoving %d\n", sample_set[j]); */
                        sample_set[j] = -1;
                    }
                }
                ancestor[l] = consensus;
                /* For the remaining samples, set the disagree flags based
                 * on whether they agree with the consensus for this site. */
                for (j = 0; j < sample_set_size; j++) {
                    u = sample_set[j];
                    if (u != -1) {
                        disagree[u] = ((genotypes[j] != consensus)
                                       && (genotypes[j] != TSK_MISSING_DATA));
                    }
                }
                /* Repack the sample set */
                tmp_size = 0;
                for (j = 0; j < sample_set_size; j++) {
                    if (sample_set[j] != -1) {
                        sample_set[tmp_size] = sample_set[j];
                        tmp_size++;
                    }
                }
                sample_set_size = tmp_size;
                if (sample_set_size <= min_sample_set_size) {
                    /* printf("BREAK\n"); */
                    break;
                }
            }
        }
    }
    *last_site_ret = last_site;
    return ret;
}

static int
ancestor_builder_compute_between_focal_sites(const ancestor_builder_t *self,
    size_t num_focal_sites, const tsk_id_t *focal_sites, allele_t *ancestor,
    tsk_id_t *sample_set, allele_t *restrict genotypes)
{
    int ret = 0;
    tsk_id_t l;
    size_t j, k, ones, zeros, sample_set_size;
    double focal_site_time;
    const site_t *restrict sites = self->sites;

    assert(num_focal_sites > 0);
    ancestor_builder_get_consistent_samples(
        self, focal_sites[0], sample_set, &sample_set_size, genotypes);
    if (sample_set_size == 0) {
        ret = TSI_ERR_BAD_FOCAL_SITE;
        goto out;
    }
    focal_site_time = self->sites[focal_sites[0]].time;

    ancestor[focal_sites[0]] = 1;
    for (j = 1; j < num_focal_sites; j++) {
        ancestor[focal_sites[j]] = 1;
        for (l = focal_sites[j - 1] + 1; l < focal_sites[j]; l++) {
            ancestor[l] = 0;
            if (sites[l].time > focal_site_time) {
                /* printf("\t%d\t%d:", l, (int) sample_set_size); */
                /* for (k = 0; k < sample_set_size; k++) { */
                /*     printf("%d, ", sample_set[k]); */
                /* } */

                ancestor_builder_get_site_genotypes_subset(
                    self, (tsk_id_t) l, sample_set, sample_set_size, genotypes);
                ones = 0;
                zeros = 0;
                for (k = 0; k < sample_set_size; k++) {
                    switch (genotypes[k]) {
                        case 0:
                            zeros++;
                            break;
                        case 1:
                            ones++;
                            break;
                    }
                }
                if (ones + zeros == 0) {
                    ancestor[l] = TSK_MISSING_DATA;
                } else if (ones >= zeros) {
                    ancestor[l] = 1;
                }
            }
        }
    }
out:
    return ret;
}

/* Build the ancestors for sites in the specified focal sites */
int
ancestor_builder_make_ancestor(const ancestor_builder_t *self, size_t num_focal_sites,
    const tsk_id_t *focal_sites, tsk_id_t *ret_start, tsk_id_t *ret_end,
    allele_t *ancestor)
{
    int ret = 0;
    tsk_id_t focal_site, last_site;
    tsk_id_t *sample_set = malloc(self->num_samples * sizeof(tsk_id_t));
    bool *restrict disagree = calloc(self->num_samples, sizeof(*disagree));
    allele_t *restrict genotypes = malloc(self->decoded_genotypes_size);

    if (sample_set == NULL || disagree == NULL || genotypes == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    memset(ancestor, 0xff, self->num_sites * sizeof(*ancestor));

    ret = ancestor_builder_compute_between_focal_sites(
        self, num_focal_sites, focal_sites, ancestor, sample_set, genotypes);
    if (ret != 0) {
        goto out;
    }

    focal_site = focal_sites[num_focal_sites - 1];
    ret = ancestor_builder_compute_ancestral_states(
        self, +1, focal_site, ancestor, sample_set, disagree, &last_site, genotypes);
    if (ret != 0) {
        goto out;
    }
    *ret_end = last_site + 1;

    focal_site = focal_sites[0];
    ret = ancestor_builder_compute_ancestral_states(
        self, -1, focal_site, ancestor, sample_set, disagree, &last_site, genotypes);
    if (ret != 0) {
        goto out;
    }
    *ret_start = last_site;
out:
    tsi_safe_free(sample_set);
    tsi_safe_free(disagree);
    tsi_safe_free(genotypes);
    return ret;
}

static int WARN_UNUSED
ancestor_builder_encode_genotypes(
    const ancestor_builder_t *self, const allele_t *genotypes, uint8_t *dest)
{
    int ret = 0;

    if (self->flags & TSI_GENOTYPE_ENCODING_ONE_BIT) {
        ret = packbits(genotypes, self->num_samples, dest);
    } else {
        memcpy(dest, genotypes, self->num_samples * sizeof(allele_t));
    }
    return ret;
}

static uint8_t *
ancestor_builder_allocate_genotypes(ancestor_builder_t *self)
{
    uint8_t *ret = NULL;
    void *p;

    if (self->mmap_buffer == NULL) {
        ret = tsk_blkalloc_get(&self->main_allocator, self->encoded_genotypes_size);
    } else {
        p = (char *) self->mmap_buffer + self->mmap_offset;
        self->mmap_offset += self->encoded_genotypes_size;
        assert(self->mmap_offset <= self->mmap_size);
        ret = (uint8_t *) p;
    }
    return ret;
}

int WARN_UNUSED
ancestor_builder_add_site(ancestor_builder_t *self, double time, allele_t *genotypes)
{
    int ret = 0;
    site_t *site;
    avl_node_t *avl_node;
    site_list_t *list_node;
    pattern_map_t search, *map_elem;
    uint8_t *encoded_genotypes = self->genotype_encode_buffer;
    uint8_t *stored_genotypes = NULL;
    avl_tree_t *pattern_map;
    tsk_id_t site_id = (tsk_id_t) self->num_sites;
    time_map_t *time_map = ancestor_builder_get_time_map(self, time);

    if (time_map == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    if (self->num_sites == self->max_sites) {
        ret = TSI_ERR_TOO_MANY_SITES;
        goto out;
    }
    ret = ancestor_builder_encode_genotypes(self, genotypes, encoded_genotypes);
    if (ret != 0) {
        goto out;
    }
    self->num_sites++;
    pattern_map = &time_map->pattern_map;
    site = &self->sites[site_id];
    site->time = time;

    search.encoded_genotypes = encoded_genotypes;
    search.encoded_genotypes_size = self->encoded_genotypes_size;
    avl_node = avl_search(pattern_map, &search);
    if (avl_node == NULL) {
        stored_genotypes = ancestor_builder_allocate_genotypes(self);
        avl_node = tsk_blkalloc_get(&self->indexing_allocator, sizeof(avl_node_t));
        map_elem = tsk_blkalloc_get(&self->indexing_allocator, sizeof(pattern_map_t));
        if (stored_genotypes == NULL || avl_node == NULL || map_elem == NULL) {
            ret = TSI_ERR_NO_MEMORY;
            goto out;
        }
        memcpy(stored_genotypes, encoded_genotypes, self->encoded_genotypes_size);
        avl_init_node(avl_node, map_elem);
        map_elem->encoded_genotypes = stored_genotypes;
        map_elem->encoded_genotypes_size = self->encoded_genotypes_size;
        map_elem->sites = NULL;
        map_elem->num_sites = 0;
        avl_node = avl_insert_node(pattern_map, avl_node);
        assert(avl_node != NULL);
    } else {
        map_elem = (pattern_map_t *) avl_node->item;
    }
    map_elem->num_sites++;
    self->sites[site_id].encoded_genotypes = map_elem->encoded_genotypes;

    list_node = tsk_blkalloc_get(&self->indexing_allocator, sizeof(site_list_t));
    if (list_node == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    list_node->site = site_id;
    list_node->next = map_elem->sites;
    map_elem->sites = list_node;

out:
    return ret;
}

/* Returns true if we should break the an ancestor that spans from focal
 * site a to focal site b */
static bool
ancestor_builder_break_ancestor(ancestor_builder_t *self, tsk_id_t a, tsk_id_t b,
    const tsk_id_t *restrict samples, size_t num_samples, allele_t *restrict genotypes)
{
    bool ret = false;
    tsk_id_t j, k;
    size_t ones, missing;

    for (j = a + 1; j < b && !ret; j++) {
        if (self->sites[j].time > self->sites[a].time) {
            ancestor_builder_get_site_genotypes_subset(
                self, j, samples, num_samples, genotypes);
            ones = 0;
            missing = 0;
            for (k = 0; k < (tsk_id_t) num_samples; k++) {
                switch (genotypes[k]) {
                    case TSK_MISSING_DATA:
                        missing++;
                        break;
                    case 1:
                        ones++;
                        break;
                }
            }
            if (ones != (num_samples - missing) && ones != 0) {
                ret = true;
            }
        }
    }
    return ret;
}

int
ancestor_builder_finalise(ancestor_builder_t *self)
{
    int ret = 0;
    size_t j, num_consistent_samples;
    avl_node_t *a, *b;
    pattern_map_t *pattern_map;
    time_map_t *time_map;
    site_list_t *s;
    ancestor_descriptor_t *descriptor;
    tsk_id_t *focal_sites = NULL;
    tsk_id_t *p;
    tsk_id_t *consistent_samples = malloc(self->num_samples * sizeof(tsk_id_t));
    allele_t *genotypes = malloc(self->decoded_genotypes_size);

    if (consistent_samples == NULL || genotypes == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    num_consistent_samples = 0; /* Keep the compiler happy */
    self->num_ancestors = 0;

    /* Return the descriptors in *reverse* order */
    for (a = self->time_map.tail; a != NULL; a = a->prev) {
        time_map = (time_map_t *) a->item;
        for (b = time_map->pattern_map.head; b != NULL; b = b->next) {
            pattern_map = (pattern_map_t *) b->item;
            descriptor = self->descriptors + self->num_ancestors;
            self->num_ancestors++;
            descriptor->time = time_map->time;
            focal_sites = tsk_blkalloc_get(
                &self->main_allocator, pattern_map->num_sites * sizeof(tsk_id_t));
            if (focal_sites == NULL) {
                ret = TSI_ERR_NO_MEMORY;
                goto out;
            }
            descriptor->focal_sites = focal_sites;
            descriptor->num_focal_sites = pattern_map->num_sites;
            j = pattern_map->num_sites - 1;
            for (s = pattern_map->sites; s != NULL; s = s->next) {
                focal_sites[j] = s->site;
                j--;
            }
            /* Now check to see if we need to split this ancestor up
             * further */
            if (pattern_map->num_sites > 1) {
                ancestor_builder_get_consistent_samples(self, focal_sites[0],
                    consistent_samples, &num_consistent_samples, genotypes);
            }
            for (j = 0; j < pattern_map->num_sites - 1; j++) {
                if (ancestor_builder_break_ancestor(self, focal_sites[j],
                        focal_sites[j + 1], consistent_samples, num_consistent_samples,
                        genotypes)) {
                    p = focal_sites + j + 1;
                    descriptor->num_focal_sites = (size_t)(p - descriptor->focal_sites);
                    descriptor = self->descriptors + self->num_ancestors;
                    self->num_ancestors++;
                    descriptor->time = time_map->time;
                    descriptor->num_focal_sites = pattern_map->num_sites - j - 1;
                    descriptor->focal_sites = p;
                }
            }
        }
    }

    /* After we've finalised, free up the large chunks of memory we're no longer using */
    self->time_map.head = NULL;
    self->time_map.tail = NULL;
    tsk_blkalloc_free(&self->indexing_allocator);
    memset(&self->indexing_allocator, 0, sizeof(self->indexing_allocator));
out:
    tsi_safe_free(consistent_samples);
    tsi_safe_free(genotypes);
    return ret;
}
