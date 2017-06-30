#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "object_heap.h"

/* TODO change this to node_id_t */
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
    double position;
} site_state_t;

typedef struct {
    size_t num_sites;
    size_t num_ancestors;
    size_t total_segments;
    size_t max_num_site_segments;
    size_t total_memory;
    site_state_t *sites;
    struct {
        /* TODO add start_site and end_site to improve get_ancestor performance. */
        site_id_t *focal_site;
        size_t *focal_site_frequency;
        size_t *num_older_ancestors;
    } ancestors;
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
    size_t num_sites;
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

typedef struct child_list_node_t_t {
    ancestor_id_t node;
    struct child_list_node_t_t *next;
} child_list_node_t;

typedef struct mutation_list_node_t_t {
    ancestor_id_t node;
    allele_t derived_state;
    struct mutation_list_node_t_t *next;
} mutation_list_node_t;

typedef struct list_segment_t_t {
    site_id_t start;
    site_id_t end;
    child_list_node_t *head;
    child_list_node_t *tail;
    struct list_segment_t_t *next;
} list_segment_t;

typedef struct {
    size_t num_sites;
    size_t num_samples;
    size_t num_ancestors;
    size_t num_edgesets;
    size_t num_children;
    size_t num_mutations;
    ancestor_store_t *store;
    size_t segment_block_size;
    size_t child_list_node_block_size;
    size_t mutation_list_node_block_size;
    list_segment_t **children;
    mutation_list_node_t **mutations;
    object_heap_t segment_heap;
    object_heap_t child_list_node_heap;
    object_heap_t mutation_list_node_heap;
} tree_sequence_builder_t;

int ancestor_store_builder_alloc(ancestor_store_builder_t *self, size_t num_sites,
        size_t segment_block_size);
int ancestor_store_builder_free(ancestor_store_builder_t *self);
int ancestor_store_builder_print_state(ancestor_store_builder_t *self, FILE *out);
int ancestor_store_builder_add(ancestor_store_builder_t *self, allele_t *ancestor);
int ancestor_store_builder_dump(ancestor_store_builder_t *self,
        site_id_t *site, ancestor_id_t *start, ancestor_id_t *end, allele_t *state);

int ancestor_store_alloc(ancestor_store_t *self,
        size_t num_sites, double *position,
        size_t num_ancestors, site_id_t *focal_site, size_t *focal_site_frequency,
        size_t num_segments, site_id_t *site, ancestor_id_t *start, ancestor_id_t *end,
        allele_t *state);
int ancestor_store_free(ancestor_store_t *self);
int ancestor_store_print_state(ancestor_store_t *self, FILE *out);
int ancestor_store_init_build(ancestor_store_t *self, size_t segment_block_size);
int ancestor_store_get_state(ancestor_store_t *self, site_id_t site_id,
        ancestor_id_t ancestor_id, allele_t *state);
int ancestor_store_get_ancestor(ancestor_store_t *self, ancestor_id_t ancestor_id,
        allele_t *ancestor, site_id_t *start_site, site_id_t *focal_site,
        site_id_t *end_site, size_t *num_older_ancestors);

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
        site_id_t focal_site, double error_rate,
        traceback_t *traceback, ancestor_id_t *end_site_value);
int ancestor_matcher_print_state(ancestor_matcher_t *self, FILE *out);

int traceback_alloc(traceback_t *self, size_t num_sites, size_t segment_block_size);
int traceback_free(traceback_t *self);
int traceback_reset(traceback_t *self);
int traceback_add_recombination(traceback_t *self, site_id_t site,
        ancestor_id_t start, ancestor_id_t end, ancestor_id_t ancestor);
int traceback_print_state(traceback_t *self, FILE *out);

int tree_sequence_builder_alloc(tree_sequence_builder_t *self,
        ancestor_store_t *store, size_t num_samples,
        size_t segment_block_size, size_t child_list_node_block_size,
        size_t mutation_list_node_block_size);
int tree_sequence_builder_print_state(tree_sequence_builder_t *self, FILE *out);
int tree_sequence_builder_free(tree_sequence_builder_t *self);
int tree_sequence_builder_update(tree_sequence_builder_t *self, ancestor_id_t child_id,
        allele_t *haplotype, site_id_t start_site, site_id_t end_site,
        ancestor_id_t end_site_parent, traceback_t *traceback);
int tree_sequence_builder_dump_edgesets(tree_sequence_builder_t *self,
        double *left, double *right, ancestor_id_t *parent, ancestor_id_t *children,
        uint32_t *children_length);
int tree_sequence_builder_dump_mutations(tree_sequence_builder_t *self,
        site_id_t *site, ancestor_id_t *node, allele_t *derived_state);

void __tsi_safe_free(void **ptr);

#define tsi_safe_free(pointer) __tsi_safe_free((void **) &(pointer))
