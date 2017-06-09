#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "object_heap.h"

typedef int32_t ancestor_id_t;
typedef int8_t allele_t;
typedef uint32_t site_id_t;

typedef struct _segment_t {
    ancestor_id_t start;
    ancestor_id_t end;
    double value;
    struct _segment_t *next;
} segment_t;

typedef struct {
    ancestor_id_t *start;
    ancestor_id_t *end;
    allele_t *state;
    size_t num_segments;
    size_t max_num_segments;
    /* TODO: position, etc ? */
} site_t;

typedef struct {
    size_t num_sites;
    size_t num_ancestors;
    object_heap_t segment_heap;
    size_t segment_block_size;
    site_t *sites;
} ancestor_matcher_t;

int ancestor_matcher_alloc(ancestor_matcher_t *self, size_t num_sites,
        size_t segment_block_size);
int ancestor_matcher_free(ancestor_matcher_t *self);
int ancestor_matcher_add(ancestor_matcher_t *self, allele_t *haplotype);
int ancestor_matcher_best_path(ancestor_matcher_t *self,
        allele_t *haplotype, double recombination_rate,
        double mutation_rate, ancestor_id_t *path,
        size_t *num_mutations, site_id_t *mutation_sites);
int ancestor_matcher_print_state(ancestor_matcher_t *self, FILE *out);

void __tsi_safe_free(void **ptr);

#define tsi_safe_free(pointer) __tsi_safe_free((void **) &(pointer))
