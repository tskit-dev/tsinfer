/*
** Copyright (C) 2018-2026 University of Oxford
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

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

/* No atomics on Windows for now */
#if defined(_WIN32)
#define TSI_NO_ATOMICS 1
#endif

#if !defined(TSI_NO_ATOMICS)
#include <stdatomic.h>
#define TSI_ATOMIC(T) _Atomic T
#else
#define TSI_ATOMIC(T) T
#endif

#include "tskit.h"
#include "err.h"
#include "avl.h"

/* TODO remove this when we update tskit version. */
#define TSK_MISSING_DATA (-1)

#define TSI_GENOTYPE_ENCODING_ONE_BIT 1

typedef int8_t allele_t;

/* A block allocator, as tsk_blkalloc, with total_size atomic so that it can
 * be read safely while other threads may be allocating. */
typedef struct {
    size_t chunk_size; /* number of bytes per chunk */
    size_t top;        /* the offset of the next available byte in the current chunk */
    size_t current_chunk;          /* the index of the chunk currently being used */
    TSI_ATOMIC(size_t) total_size; /* the total number of bytes allocated + overhead. */
    size_t total_allocated;        /* the total number of bytes allocated. */
    size_t num_chunks;             /* the number of memory chunks. */
    char **mem_chunks;             /* the memory chunks */
} tsi_blkalloc_t;

extern void tsi_blkalloc_print_state(tsi_blkalloc_t *self, FILE *out);
extern int tsi_blkalloc_reset(tsi_blkalloc_t *self);
extern int tsi_blkalloc_init(tsi_blkalloc_t *self, size_t chunk_size);
extern void *tsi_blkalloc_get(tsi_blkalloc_t *self, size_t size);
extern void tsi_blkalloc_free(tsi_blkalloc_t *self);

typedef struct {
    double time;
    uint8_t *encoded_genotypes;
    tsk_size_t derived_count;
} site_t;

typedef struct _site_list_t {
    tsk_id_t site;
    struct _site_list_t *next;
} site_list_t;

typedef struct {
    uint8_t *encoded_genotypes;
    size_t encoded_genotypes_size;
    size_t num_sites;
    site_list_t *sites;
} pattern_map_t;

typedef struct {
    double time;
    size_t num_focal_sites;
    tsk_id_t *focal_sites;
} ancestor_descriptor_t;

/* Maps all ancestors with a specific time to their genotype patterns  */
typedef struct {
    double time;
    avl_tree_t pattern_map;
} time_map_t;

typedef struct {
    size_t num_sites;
    size_t max_sites;
    size_t num_samples;
    size_t num_ancestors;
    int flags;
    site_t *sites;
    avl_tree_t time_map;
    /* Thread-safe block allocators for bulk allocations */
    tsi_blkalloc_t main_allocator;
    tsi_blkalloc_t indexing_allocator;
    ancestor_descriptor_t *descriptors;
    size_t encoded_genotypes_size;
    size_t decoded_genotypes_size;
    uint8_t *genotype_encode_buffer;
    /* Optional file-descriptor for the file to be used as the mmap memory
     * store for encoded genotypes */
    int mmap_fd;
    void *mmap_buffer;
    size_t mmap_offset;
    size_t mmap_size;
} ancestor_builder_t;

int ancestor_builder_alloc(ancestor_builder_t *self, size_t num_samples,
    size_t num_sites, int mmap_fd, int flags);
int ancestor_builder_free(ancestor_builder_t *self);
int ancestor_builder_print_state(ancestor_builder_t *self, FILE *out);
int ancestor_builder_add_site(
    ancestor_builder_t *self, double time, allele_t *genotypes);
int ancestor_builder_finalise(ancestor_builder_t *self);
int ancestor_builder_make_ancestor(const ancestor_builder_t *self,
    size_t num_focal_sites, const tsk_id_t *focal_sites, tsk_id_t *start, tsk_id_t *end,
    allele_t *haplotype);
size_t ancestor_builder_get_memsize(const ancestor_builder_t *self);

int packbits(const allele_t *restrict source, size_t len, uint8_t *restrict dest);
void unpackbits(const uint8_t *restrict source, size_t len, allele_t *restrict dest);

#define tsi_safe_free(pointer)                                                          \
    do {                                                                                \
        if (pointer != NULL) {                                                          \
            free(pointer);                                                              \
        }                                                                               \
    } while (0)
