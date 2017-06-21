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
    site_id_t id;
    double position;
    size_t frequency;
} site_t;

typedef struct {
    size_t frequency;
    size_t num_sites;
    site_t **sites;
} frequency_class_t;

typedef struct {
    ancestor_id_t *start;
    ancestor_id_t *end;
    allele_t *state;
    size_t num_segments;
    /* TODO: position, etc ? */
} site_state_t;

typedef struct {
    size_t num_sites;
    size_t num_ancestors;
    size_t total_segments;
    size_t max_num_site_segments;
    size_t total_memory;
    site_state_t *sites;
} ancestor_store_t;

typedef struct {
    size_t num_sites;
    size_t num_ancestors;
    size_t total_segments;
    size_t segment_block_size;
    segment_t **sites_head;
    segment_t **sites_tail;
    object_heap_t segment_heap;
} ancestor_store_builder_t;

typedef struct {
    ancestor_store_t *store;
    segment_t **sites_head;
    segment_t **sites_tail;
    object_heap_t segment_heap;
    size_t segment_block_size;
} traceback_t;

typedef struct {
    double recombination_rate;
    ancestor_store_t *store;
} ancestor_matcher_t;

typedef struct {
    size_t num_sites;
    size_t num_samples;
    size_t num_ancestors;
    size_t num_frequency_classes;
    allele_t *haplotypes;
    site_t *sites;
    site_t **sorted_sites;
    frequency_class_t *frequency_classes;
} ancestor_builder_t;

typedef struct {
    site_id_t index;
    site_id_t current_position;
    allele_t state;
} permutation_sort_t;

typedef struct {
    size_t num_sites;
    size_t num_ancestors;
    allele_t *ancestors;
    site_id_t *permutation;
    permutation_sort_t *sort_buffer;
} ancestor_sorter_t;

int ancestor_store_builder_alloc(ancestor_store_builder_t *self, size_t num_sites,
        size_t segment_block_size);
int ancestor_store_builder_free(ancestor_store_builder_t *self);
int ancestor_store_builder_print_state(ancestor_store_builder_t *self, FILE *out);
int ancestor_store_builder_add(ancestor_store_builder_t *self, allele_t *ancestor);
int ancestor_store_builder_dump(ancestor_store_builder_t *self,
        site_id_t *site, ancestor_id_t *start, ancestor_id_t *end, allele_t *state);

int ancestor_store_alloc(ancestor_store_t *self, size_t num_sites, size_t num_segments,
        site_id_t *site, ancestor_id_t *start, ancestor_id_t *end, allele_t *state);
int ancestor_store_free(ancestor_store_t *self);
int ancestor_store_print_state(ancestor_store_t *self, FILE *out);
int ancestor_store_init_build(ancestor_store_t *self, size_t segment_block_size);
int ancestor_store_get_state(ancestor_store_t *self, site_id_t site_id,
        ancestor_id_t ancestor_id, allele_t *state);
int ancestor_store_get_ancestor(ancestor_store_t *self, ancestor_id_t ancestor_id,
        allele_t *ancestor, site_id_t *start_site, site_id_t *end_site);

int ancestor_builder_alloc(ancestor_builder_t *self, size_t num_samples,
        size_t num_sites, double *positions, allele_t *haplotypes);
int ancestor_builder_free(ancestor_builder_t *self);
int ancestor_builder_print_state(ancestor_builder_t *self, FILE *out);
int ancestor_builder_make_ancestor(ancestor_builder_t *self,
        site_id_t focal_site_id, allele_t *haplotype);

int ancestor_sorter_alloc(ancestor_sorter_t *self, size_t num_ancestors,
        size_t num_sites, allele_t *ancestors, site_id_t *permutation);
int ancestor_sorter_free(ancestor_sorter_t *self);
int ancestor_sorter_print_state(ancestor_sorter_t *self, FILE *out);
int ancestor_sorter_sort(ancestor_sorter_t *self);

int ancestor_matcher_alloc(ancestor_matcher_t *self, ancestor_store_t *store,
        double recombination_rate);
int ancestor_matcher_free(ancestor_matcher_t *self);
int ancestor_matcher_best_path(ancestor_matcher_t *self, size_t num_ancestors,
        allele_t *haplotype, site_id_t start_site, site_id_t end_site,
        double mutation_rate, traceback_t *traceback, ancestor_id_t *end_site_value);
int ancestor_matcher_print_state(ancestor_matcher_t *self, FILE *out);

int traceback_alloc(traceback_t *self, ancestor_store_t *store, size_t segment_block_size);
int traceback_free(traceback_t *self);
int traceback_reset(traceback_t *self);
int traceback_add_recombination(traceback_t *self, site_id_t site,
        ancestor_id_t start, ancestor_id_t end, ancestor_id_t ancestor);
int traceback_run(traceback_t *self, allele_t *haplotype,
        site_id_t start_site, site_id_t end_site, ancestor_id_t end_site_value,
        ancestor_id_t *path, size_t *num_mutations, site_id_t *mutation_sites);
int traceback_print_state(traceback_t *self, FILE *out);


void __tsi_safe_free(void **ptr);

#define tsi_safe_free(pointer) __tsi_safe_free((void **) &(pointer))
