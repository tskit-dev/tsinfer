#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "block_allocator.h"
#include "object_heap.h"
#include "avl.h"

/* TODO change all instances of this to node_id_t */
typedef int32_t ancestor_id_t;
typedef int32_t node_id_t;
typedef int8_t allele_t;
/* TODO change site_id_t to int for compatability with msprime. */
typedef uint32_t site_id_t;

typedef struct {
    site_id_t left;
    site_id_t right;
    site_id_t end;
    node_id_t parent;
    node_id_t child;
} edge_t;

typedef struct _node_segment_list_node_t {
    ancestor_id_t start;
    ancestor_id_t end;
    struct _node_segment_list_node_t *next;
} node_segment_list_node_t;

/* TODO rename this struct and see where we're actually using it.*/
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
    size_t num_ancestors;
    site_id_t **ancestor_focal_sites;
    size_t *num_ancestor_focal_sites;
    site_id_t *ancestor_focal_site_mem;
} frequency_class_t;

typedef struct {
    ancestor_id_t *start;
    ancestor_id_t *end;
    size_t num_segments;
    double position;
} site_state_t;

typedef struct {
    size_t num_sites;
    size_t num_ancestors;
    size_t total_segments;
    size_t max_site_segments;
    double mean_site_segments;
    size_t total_memory;
    site_state_t *sites;
    struct {
        /* TODO add start_site and end_site to improve get_ancestor performance. */
        site_id_t **focal_sites;
        uint32_t *num_focal_sites;
        uint32_t *age;
        uint32_t *num_older_ancestors;
        site_id_t *focal_sites_mem;
    } ancestors;
    size_t num_epochs;
    struct {
        ancestor_id_t *first_ancestor;
        size_t *num_ancestors;
    } epochs;
} ancestor_store_t;

typedef struct {
    size_t num_sites;
    size_t num_ancestors;
    size_t total_segments;
    size_t segment_block_size;
    node_segment_list_node_t **sites_head;
    node_segment_list_node_t **sites_tail;
    block_allocator_t allocator;
} ancestor_store_builder_t;

typedef struct {
    size_t num_sites;
    ancestor_id_t *best_match;
    node_segment_list_node_t **sites_head;
    node_segment_list_node_t **sites_tail;
    block_allocator_t allocator;
    size_t segment_block_size;
} traceback_t;

typedef struct {
    double recombination_rate;
    ancestor_store_t *store;
    /* The mean number of likelihood segments in the last performed match */
    double mean_likelihood_segments;
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

typedef struct _mutation_list_node_t {
    ancestor_id_t node;
    allele_t derived_state;
    struct _mutation_list_node_t *next;
} mutation_list_node_t;

typedef struct {
    site_id_t left;
    site_id_t right;
    ancestor_id_t parent;
    uint32_t num_children;
    ancestor_id_t *children;
    double time; /* Used for sorting */
} edgeset_t;

typedef struct _node_mapping_t {
    site_id_t left;
    site_id_t right;
    ancestor_id_t node;
    struct _node_mapping_t *next;
} node_mapping_t;

typedef struct _segment_list_node_t {
    site_id_t start;
    site_id_t end;
    struct _segment_list_node_t *next;
} segment_list_node_t;

typedef struct {
    segment_list_node_t *head;
    segment_list_node_t *tail;
    size_t length;
    size_t block_size;
    object_heap_t heap;
} segment_list_t;


typedef struct {
    node_id_t index;
    site_id_t position;
    double time;
} index_sort_t;

typedef struct _likelihood_list_t {
    node_id_t node;
    double likelihood;
    struct _likelihood_list_t *next;
} likelihood_list_t;

typedef struct {
    site_id_t site;
    node_id_t node;
} site_mutation_t;

typedef struct {
    double recombination_rate;
    size_t num_sites;
    size_t max_nodes;
    size_t max_edges;
    size_t max_output_edges;
    size_t num_nodes;
    size_t num_edges;
    size_t num_mutations;
    edge_t *edges;
    double *time;
    node_id_t *mutations;
    node_id_t *parent;
    index_sort_t *sort_buffer;
    node_id_t *insertion_order;
    node_id_t *removal_order;
    double *likelihood;
    avl_tree_t likelihood_nodes;
    likelihood_list_t **traceback;
    object_heap_t avl_node_heap;
    block_allocator_t likelihood_list_allocator;
    edge_t *output_edge_buffer;
    size_t total_traceback_size;
} tree_sequence_builder_t;

int ancestor_store_builder_alloc(ancestor_store_builder_t *self, size_t num_sites,
        size_t segment_block_size);
int ancestor_store_builder_free(ancestor_store_builder_t *self);
int ancestor_store_builder_print_state(ancestor_store_builder_t *self, FILE *out);
int ancestor_store_builder_add(ancestor_store_builder_t *self, allele_t *ancestor);
int ancestor_store_builder_dump(ancestor_store_builder_t *self,
        site_id_t *site, ancestor_id_t *start, ancestor_id_t *end);
int ancestor_store_alloc(ancestor_store_t *self,
        size_t num_sites, double *position,
        size_t num_ancestors, uint32_t *ancestor_age,
        size_t num_focal_sites, ancestor_id_t *focal_site_ancestor, site_id_t *focal_site,
        size_t num_segments, site_id_t *site, ancestor_id_t *start, ancestor_id_t *end);
int ancestor_store_free(ancestor_store_t *self);
int ancestor_store_print_state(ancestor_store_t *self, FILE *out);
int ancestor_store_init_build(ancestor_store_t *self, size_t segment_block_size);
int ancestor_store_get_state(ancestor_store_t *self, site_id_t site_id,
        ancestor_id_t ancestor_id, allele_t *state);
int ancestor_store_get_ancestor(ancestor_store_t *self, ancestor_id_t ancestor_id,
        allele_t *ancestor, site_id_t *start_site, site_id_t *end_site,
        size_t *num_older_ancestors, size_t *num_focal_sites, site_id_t **focal_sites);
int ancestor_store_get_epoch_ancestors(ancestor_store_t *self, int epoch,
        ancestor_id_t *epoch_ancestors, size_t *num_epoch_ancestors);

int ancestor_builder_alloc(ancestor_builder_t *self, size_t num_samples,
        size_t num_sites, double *positions, allele_t *haplotypes);
int ancestor_builder_free(ancestor_builder_t *self);
int ancestor_builder_print_state(ancestor_builder_t *self, FILE *out);
int ancestor_builder_make_ancestor(ancestor_builder_t *self,
        size_t num_focal_sites, site_id_t *focal_sites, allele_t *haplotype);

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
        size_t num_focal_sites, site_id_t *focal_sites, double error_rate,
        traceback_t *traceback);
int ancestor_matcher_print_state(ancestor_matcher_t *self, FILE *out);

int traceback_alloc(traceback_t *self, size_t num_sites, size_t segment_block_size);
int traceback_free(traceback_t *self);
int traceback_reset(traceback_t *self);
int traceback_add_recombination(traceback_t *self, site_id_t site,
        ancestor_id_t start, ancestor_id_t end);
int traceback_set_best_match(traceback_t *self, site_id_t site, ancestor_id_t best_match);
int traceback_print_state(traceback_t *self, FILE *out);

int tree_sequence_builder_alloc(tree_sequence_builder_t *self,
        size_t num_sites, size_t max_nodes, size_t max_edges);
int tree_sequence_builder_print_state(tree_sequence_builder_t *self, FILE *out);
int tree_sequence_builder_free(tree_sequence_builder_t *self);
int tree_sequence_builder_find_path(tree_sequence_builder_t *self, allele_t *haplotype,
        node_id_t child, size_t *num_outout_edges, edge_t **output_edges);
int tree_sequence_builder_update(tree_sequence_builder_t *self, size_t num_nodes,
        double time, size_t num_edges, edge_t *edges, size_t num_site_mutations,
        site_mutation_t *site_mutations);

/* int tree_sequence_builder_get_live_segments(tree_sequence_builder_t *self, */
/*         ancestor_id_t parent, segment_list_t *list); */
/* int tree_sequence_builder_update(tree_sequence_builder_t *self, ancestor_id_t child_id, */
/*         allele_t *haplotype, site_id_t start_site, site_id_t end_site, */
/*         traceback_t *traceback); */
int tree_sequence_builder_dump_nodes(tree_sequence_builder_t *self,
        uint32_t *flags, double *time);
int tree_sequence_builder_dump_edges(tree_sequence_builder_t *self,
        double *left, double *right, ancestor_id_t *parent, ancestor_id_t *children);
int tree_sequence_builder_dump_mutations(tree_sequence_builder_t *self,
        site_id_t *site, ancestor_id_t *node, allele_t *derived_state);
/* int tree_sequence_builder_resolve(tree_sequence_builder_t *self, */
/*         int epoch, ancestor_id_t *ancestors, size_t num_ancestors); */

int segment_list_alloc(segment_list_t *self, size_t block_size);
int segment_list_free(segment_list_t *self);
int segment_list_append(segment_list_t *self, site_id_t start, site_id_t end);
int segment_list_clear(segment_list_t *self);
int segment_list_print_state(segment_list_t *self, FILE *out);

void __tsi_safe_free(void **ptr);

#define tsi_safe_free(pointer) __tsi_safe_free((void **) &(pointer))
