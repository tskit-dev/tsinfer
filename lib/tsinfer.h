#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#include "block_allocator.h"
#include "object_heap.h"
#include "avl.h"

#define NULL_LIKELIHOOD (-1)
#define NONZERO_ROOT_LIKELIHOOD (-2)
#define NULL_NODE (-1)

#define TSI_RESOLVE_SHARED_RECOMBS  1
#define TSI_RESOLVE_POLYTOMIES      2

/* TODO change all instances of this to node_id_t */
typedef int32_t ancestor_id_t;
typedef int32_t node_id_t;
typedef int8_t allele_t;
/* TODO change site_id_t to int for compatability with msprime. */
typedef uint32_t site_id_t;
typedef int32_t mutation_id_t;

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
    size_t num_samples;
    size_t num_ancestors;
    size_t num_frequency_classes;
    allele_t *haplotypes;
    site_t *sites;
    site_t **sorted_sites;
    frequency_class_t *frequency_classes;
} ancestor_builder_t;

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
    int flags;
    double sequence_length;
    size_t num_sites;
    struct {
        double *position;
        double *recombination_rate;
        mutation_list_node_t **mutations;
    } sites;
    size_t max_nodes;
    size_t max_edges;
    size_t num_nodes;
    size_t num_edges;
    size_t num_mutations;
    edge_t *edges;
    double *time;
    uint32_t *node_flags;
    index_sort_t *sort_buffer;
    node_id_t *insertion_order;
    node_id_t *removal_order;
    block_allocator_t block_allocator;
} tree_sequence_builder_t;

typedef struct {
    tree_sequence_builder_t *tree_sequence_builder;
    double *recombination_rate;
    double observation_error;
    size_t num_nodes;
    size_t num_sites;
    size_t max_nodes;
    /* The quintuply linked tree */
    node_id_t *parent;
    node_id_t *left_child;
    node_id_t *right_child;
    node_id_t *left_sib;
    node_id_t *right_sib;
    double *likelihood;
    avl_tree_t likelihood_nodes;
    likelihood_list_t **traceback;
    object_heap_t avl_node_heap;
    block_allocator_t likelihood_list_allocator;
    size_t total_traceback_size;
    /* Some better nameing is needed here. The 'output' struct here
     * is really the 'path', and mismatches are also output. Perhaps
     * we should put both into the output struct? */
    struct {
        site_id_t *left;
        site_id_t *right;
        node_id_t *parent;
        size_t size;
        size_t max_size;
    } output;
    size_t num_mismatches;
    size_t max_num_mismatches;
    site_id_t *mismatches;
} ancestor_matcher_t;

int ancestor_builder_alloc(ancestor_builder_t *self, size_t num_samples,
        size_t num_sites, double *positions, allele_t *haplotypes);
int ancestor_builder_free(ancestor_builder_t *self);
int ancestor_builder_print_state(ancestor_builder_t *self, FILE *out);
int ancestor_builder_make_ancestor(ancestor_builder_t *self,
        size_t num_focal_sites, site_id_t *focal_sites,
        site_id_t *start, site_id_t *end, allele_t *haplotype);

int ancestor_matcher_alloc(ancestor_matcher_t *self,
        tree_sequence_builder_t *tree_sequence_builder,
        double observation_error);
int ancestor_matcher_free(ancestor_matcher_t *self);
int ancestor_matcher_find_path(ancestor_matcher_t *self,
        site_id_t start, site_id_t end, allele_t *haplotype,
        allele_t *matched_haplotype, size_t *num_output_edges,
        site_id_t **left_output, site_id_t **right_output, node_id_t **parent_output);
int ancestor_matcher_print_state(ancestor_matcher_t *self, FILE *out);
double ancestor_matcher_get_mean_traceback_size(ancestor_matcher_t *self);
size_t ancestor_matcher_get_total_memory(ancestor_matcher_t *self);

int tree_sequence_builder_alloc(tree_sequence_builder_t *self,
        double sequence_length, size_t num_sites, double *position,
        double *recombination_rate, size_t max_nodes, size_t max_edges, int flags);
int tree_sequence_builder_print_state(tree_sequence_builder_t *self, FILE *out);
int tree_sequence_builder_free(tree_sequence_builder_t *self);
int tree_sequence_builder_update(tree_sequence_builder_t *self,
        size_t num_nodes, double time,
        size_t num_edges, site_id_t *left, site_id_t *right, node_id_t *parent,
        node_id_t *child, size_t num_mutations, site_id_t *site, node_id_t *node,
        allele_t *derived_state);
int tree_sequence_builder_dump_nodes(tree_sequence_builder_t *self,
        uint32_t *flags, double *time);
int tree_sequence_builder_dump_edges(tree_sequence_builder_t *self,
        double *left, double *right, ancestor_id_t *parent, ancestor_id_t *children);
int tree_sequence_builder_dump_mutations(tree_sequence_builder_t *self,
        site_id_t *site, ancestor_id_t *node, allele_t *derived_state,
        mutation_id_t *parent);

void __tsi_safe_free(void **ptr);

#define tsi_safe_free(pointer) __tsi_safe_free((void **) &(pointer))
