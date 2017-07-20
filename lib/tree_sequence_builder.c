#include "tsinfer.h"
#include "err.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

/* TODO remove */
#include <gsl/gsl_math.h>

static int
cmp_node_id(const void *a, const void *b) {
    const ancestor_id_t ia = *(const ancestor_id_t *) a;
    const ancestor_id_t ib = *(const ancestor_id_t *) b;
    return (ia > ib) - (ia < ib);
}


/* Sort node mappings by increasing left coordinate. When there is a
 * tie, sort by decreasing right coordinate */
static int
cmp_node_mapping(const void *a, const void *b) {
    const node_mapping_t ia = *(const node_mapping_t *) a;
    const node_mapping_t ib = *(const node_mapping_t *) b;
    int ret = (ia.left > ib.left) - (ia.left < ib.left);
    if (ret == 0) {
        ret = (ia.right < ib.right) - (ia.right > ib.right);
    }
    return ret;
}

static int
cmp_edgeset(const void *a, const void *b) {
    const edgeset_t ia = *(const edgeset_t *) a;
    const edgeset_t ib = *(const edgeset_t *) b;
    return (ia.time > ib.time) - (ia.time < ib.time);
}

static void
tree_sequence_builder_check_state(tree_sequence_builder_t *self)
{
    size_t j;
    node_mapping_t *u;
    size_t num_child_mappings;

    for (j = 0; j < self->num_ancestors; j++) {
        num_child_mappings = 0;
        for (u = self->child_mappings[j]; u != NULL; u = u->next) {
            assert(u->left < u->right);
            num_child_mappings++;
        }
        assert(num_child_mappings == self->num_child_mappings[j]);
    }
    for (j = 0; j < self->num_ancestors; j++) {
        for (u = self->live_segments_head[j]; u != NULL; u = u->next) {
            assert(u->left < u->right);
        }
    }
}

int
tree_sequence_builder_print_state(tree_sequence_builder_t *self, FILE *out)
{
    size_t j, l;
    node_mapping_t *u;
    mutation_list_node_t *v;

    fprintf(out, "Tree sequence builder state\n");
    fprintf(out, "num_samples = %d\n", (int) self->num_samples);
    fprintf(out, "num_sites = %d\n", (int) self->num_sites);
    fprintf(out, "num_ancestors = %d\n", (int) self->num_ancestors);
    fprintf(out, "num_nodes = %d\n", (int) self->num_nodes);
    fprintf(out, "num_edgesets = %d\n", (int) self->num_edgesets);
    fprintf(out, "num_children = %d\n", (int) self->num_children);
    fprintf(out, "max_num_edgesets = %d\n", (int) self->max_num_edgesets);
    fprintf(out, "max_num_nodes = %d\n", (int) self->max_num_nodes);
    fprintf(out, "parent_mapping heap:\n");
    object_heap_print_state(&self->node_mapping_heap, out);
    fprintf(out, "child_mappings:\n");
    for (j = 0; j < self->num_ancestors; j++) {
        if (self->child_mappings[j] != NULL) {
            fprintf(out, "%d\t:", (int) j);
            for (u = self->child_mappings[j]; u != NULL; u = u->next) {
                fprintf(out, "(%d, %d, %d)", u->left, u->right, u->node);
                if (u->next != NULL) {
                    fprintf(out, ",");
                }
            }
            fprintf(out, "\n");
        }
    }
    fprintf(out, "node times\n");
    fprintf(out, "**ancestors**\n");
    for (j = 0; j < self->num_ancestors; j++) {
        fprintf(out, "%d\t%f\n", (int) j, self->node_time[j]);
    }
    fprintf(out, "**samples**\n");
    for (j = self->num_ancestors; j < self->num_ancestors + self->num_samples; j++) {
        fprintf(out, "%d\t%f\n", (int) j, self->node_time[j]);
    }
    fprintf(out, "**minor nodes**\n");
    for (j = self->num_ancestors + self->num_samples; j < self->num_nodes; j++) {
        fprintf(out, "%d\t%f\n", (int) j, self->node_time[j]);
    }
    fprintf(out, "live segments\n");
    for (j = 0; j < self->num_ancestors; j++) {
        if (self->live_segments_head[j] != NULL) {
            fprintf(out, "%d\t:", (int) j);
            for (u = self->live_segments_head[j]; u != NULL; u = u->next) {
                fprintf(out, "(%d, %d)", u->left, u->right);
                if (u->next != NULL) {
                    fprintf(out, ",");
                }
            }
            fprintf(out, "\n");
        }
    }
    fprintf(out, "edgesets:\n");
    for (j = 0; j < self->num_edgesets; j++) {
        fprintf(out, "%d\t%d\t%d\t", self->edgesets[j].left, self->edgesets[j].right,
                self->edgesets[j].parent);
        for (l = 0; l < self->edgesets[j].num_children; l++) {
            fprintf(out, "%d", self->edgesets[j].children[l]);
            if (l < self->edgesets[j].num_children - 1) {
                fprintf(out, ",");
            }
        }
        fprintf(out, "\n");
    }
    fprintf(out, "mutations:\n");
    for (l = 0; l < self->num_sites; l++) {
        if (self->mutations[l] != NULL) {
            fprintf(out, "%d\t:", (int) l);
            for (v = self->mutations[l]; v != NULL; v = v->next) {
                fprintf(out, "(%d, %d)", v->node, v->derived_state);
                if (v->next != NULL) {
                    fprintf(out, ",");
                }
            }
            fprintf(out, "\n");
        }
    }
    tree_sequence_builder_check_state(self);
    return 0;
}

/* #define RESOLVE_DEBUG */
#ifdef RESOLVE_DEBUG
static void
tree_sequence_builder_draw_segments(tree_sequence_builder_t *self,
        size_t num_segments, node_mapping_t *segments)
{
    size_t j, k;

    printf("    ");
    for (j = 0; j < self->num_sites; j++) {
        printf("-");
    }
    printf("\n");
    for (j = 0; j < num_segments; j++) {
        printf("%.4d:", segments[j].node);
        for (k = 0; k < segments[j].left; k++)  {
            printf(" ");
        }
        for (k = segments[j].left; k < segments[j].right; k++)  {
            printf("=");
        }
        printf("\n");
    }
    printf("    ");
    for (j = 0; j < self->num_sites; j++) {
        printf("-");
    }
    printf("\n");
}
#endif

static inline void
tree_sequence_builder_free_node_mapping(tree_sequence_builder_t *self, node_mapping_t *u)
{
    object_heap_free_object(&self->node_mapping_heap, u);
}

static inline mutation_list_node_t * WARN_UNUSED
tree_sequence_builder_alloc_mutation_list_node(tree_sequence_builder_t *self,
        ancestor_id_t node, allele_t derived_state)
{
    mutation_list_node_t *ret = NULL;

    if (object_heap_empty(&self->mutation_list_node_heap)) {
        if (object_heap_expand(&self->mutation_list_node_heap) != 0) {
            goto out;
        }
    }
    ret = (mutation_list_node_t *) object_heap_alloc_object(
            &self->mutation_list_node_heap);
    if (ret == NULL) {
        goto out;
    }
    ret->node = node;
    ret->derived_state = derived_state;
    ret->next = NULL;
    self->num_mutations++;
out:
    return ret;
}

static inline node_mapping_t * WARN_UNUSED
tree_sequence_builder_alloc_node_mapping(tree_sequence_builder_t *self,
        site_id_t left, site_id_t right, ancestor_id_t node)
{
    node_mapping_t *ret = NULL;

    if (object_heap_empty(&self->node_mapping_heap)) {
        if (object_heap_expand(&self->node_mapping_heap) != 0) {
            goto out;
        }
    }
    ret = (node_mapping_t *) object_heap_alloc_object(&self->node_mapping_heap);
    if (ret == NULL) {
        goto out;
    }
    ret->left = left;
    ret->right = right;
    ret->node = node;
    ret->next = NULL;
out:
    return ret;
}

int
tree_sequence_builder_alloc(tree_sequence_builder_t *self,
        ancestor_store_t *store, size_t num_samples,
        size_t node_mapping_block_size, size_t edgeset_block_size,
        size_t mutation_list_node_block_size)
{
    int ret = 0;
    assert(edgeset_block_size > 0);
    assert(mutation_list_node_block_size > 0);
    assert(node_mapping_block_size > 0);

    memset(self, 0, sizeof(tree_sequence_builder_t));
    self->store = store;
    self->num_sites = store->num_sites;
    self->num_ancestors = store->num_ancestors;
    self->num_samples = num_samples;
    self->num_nodes = self->num_samples + self->num_ancestors;
    self->node_mapping_block_size = node_mapping_block_size;
    self->edgeset_block_size = edgeset_block_size;
    self->num_edgesets = 0;
    self->max_num_edgesets = edgeset_block_size;
    /* For the internal node time map, we start with 2 * original nodes as a
     * reasonable guess */
    self->max_num_nodes = 2 * (self->num_samples + self->num_ancestors);
    self->mutation_list_node_block_size = mutation_list_node_block_size;
    ret = object_heap_init(&self->node_mapping_heap,
            sizeof(node_mapping_t), self->node_mapping_block_size, NULL);
    if (ret != 0) {
        goto out;
    }
    ret = object_heap_init(&self->mutation_list_node_heap, sizeof(mutation_list_node_t),
           self->mutation_list_node_block_size, NULL);
    if (ret != 0) {
        goto out;
    }
    self->num_child_mappings = calloc(self->num_ancestors, sizeof(uint32_t));
    self->child_mappings = calloc(self->num_ancestors, sizeof(node_mapping_t *));
    self->live_segments_head = calloc(self->num_ancestors, sizeof(node_mapping_t *));
    self->live_segments_tail = calloc(self->num_ancestors, sizeof(node_mapping_t *));
    self->edgesets = malloc(self->edgeset_block_size * sizeof(edgeset_t));
    self->mutations = calloc(self->num_sites, sizeof(mutation_list_node_t *));
    self->node_time = calloc(self->max_num_nodes, sizeof(double));
    if (self->num_child_mappings == NULL || self->live_segments_head == NULL
            || self->live_segments_tail == NULL || self->child_mappings == NULL
            || self->edgesets == NULL || self->mutations == NULL
            || self->node_time == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
out:
    return ret;
}

int
tree_sequence_builder_free(tree_sequence_builder_t *self)
{
    size_t j;
    for (j = 0; j < self->num_edgesets; j++) {
        tsi_safe_free(self->edgesets[j].children);
    }
    object_heap_free(&self->node_mapping_heap);
    object_heap_free(&self->mutation_list_node_heap);
    tsi_safe_free(self->child_mappings);
    tsi_safe_free(self->num_child_mappings);
    tsi_safe_free(self->live_segments_head);
    tsi_safe_free(self->live_segments_tail);
    tsi_safe_free(self->node_time);
    tsi_safe_free(self->edgesets);
    tsi_safe_free(self->mutations);
    return 0;
}

static inline int
tree_sequence_builder_add_mapping(tree_sequence_builder_t *self, site_id_t left,
        site_id_t right, ancestor_id_t parent, ancestor_id_t child)
{
    int ret = 0;
    node_mapping_t *u;

    /* printf("add mapping %d -> %d\n", child, parent); */
    assert(parent < (ancestor_id_t) (self->num_ancestors));
    assert(child < (ancestor_id_t) (self->num_ancestors + self->num_samples));
    /* Map the ancestor ID into a node index */
    u = tree_sequence_builder_alloc_node_mapping(self, left, right, child);
    if (u == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    u->next = self->child_mappings[parent];
    self->child_mappings[parent] = u;
    self->num_child_mappings[parent]++;
out:
    return ret;
}

static inline int
tree_sequence_builder_add_mutation(tree_sequence_builder_t *self, site_id_t site,
        ancestor_id_t node, allele_t derived_state)
{
    int ret = 0;
    mutation_list_node_t *u, *v;

    /* We can't handle zero derived states here yet as we're not handling the
     * logic of traversing upwards correctly.
     */
    assert(derived_state == 1);

    v = tree_sequence_builder_alloc_mutation_list_node(self, node, derived_state);
    if (v == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    if (self->mutations[site] == NULL) {
        self->mutations[site] = v;
    } else {
        /* It's not worth keeping head and tail pointers for these lists because
         * we should have small numbers of mutations at each site */
        u = self->mutations[site];
        while (u->next != NULL) {
            u = u->next;
        }
        u->next = v;
    }
out:
    return ret;
}

int
tree_sequence_builder_update(tree_sequence_builder_t *self, ancestor_id_t child,
        allele_t *haplotype, site_id_t start_site, site_id_t end_site,
        traceback_t *traceback)
{
    int ret = 0;
    site_id_t l;
    site_id_t end = end_site;
    ancestor_id_t parent;
    bool switch_parent;
    allele_t state;
    node_segment_list_node_t *u;

    /* traceback_print_state(traceback, stdout); */
    parent = traceback->best_match[end - 1];
    for (l = end_site - 1; l > start_site; l--) {
        /* printf("Tracing back at site %d: parent = %d\n", l, parent); */
        /* print_segment_chain(T_head[l], 1, stdout); */
        /* printf("\n"); */
        ret = ancestor_store_get_state(self->store, l, parent, &state);
        if (ret != 0) {
            goto out;
        }
        if (state != haplotype[l]) {
            ret = tree_sequence_builder_add_mutation(self, l, child, haplotype[l]);
            if (ret != 0) {
                goto out;
            }
        }
        u = traceback->sites_head[l];
        switch_parent = false;
        while (u != NULL) {
            if (u->start <= parent && parent < u->end) {
                switch_parent = true;
                break;
            }
            if (u->start > parent) {
                break;
            }
            u = u->next;
        }
        if (switch_parent) {
            /* Complete a segment at this site */
            assert(l < end);
            ret = tree_sequence_builder_add_mapping(self, l, end, parent, child);
            if (ret != 0) {
                goto out;
            }
            end = l;
            parent = traceback->best_match[l - 1];
        }
    }
    assert(start_site < end);
    ret = tree_sequence_builder_add_mapping(self, start_site, end, parent, child);
    if (ret != 0) {
        goto out;
    }
    l = start_site;
    ret = ancestor_store_get_state(self->store, l, parent, &state);
    if (ret != 0) {
        goto out;
    }
    if (state != haplotype[l]) {
        ret = tree_sequence_builder_add_mutation(self, l, child, haplotype[l]);
        if (ret != 0) {
            goto out;
        }
    }
out:
    return ret;
}

int
tree_sequence_builder_get_live_segments(tree_sequence_builder_t *self,
        ancestor_id_t parent, segment_list_t *list)
{
    int ret = 0;
    node_mapping_t *u, *tmp;

    assert(parent < (ancestor_id_t) self->num_ancestors);
    u = self->live_segments_head[parent];
    while (u != NULL) {
        ret = segment_list_append(list, u->left, u->right);
        if (ret != 0) {
            goto out;
        }
        tmp = u;
        u = u->next;
        tree_sequence_builder_free_node_mapping(self, tmp);
    }
out:
    self->live_segments_head[parent] = NULL;
    self->live_segments_tail[parent] = NULL;
    return ret;
}


int
tree_sequence_builder_dump_nodes(tree_sequence_builder_t *self, uint32_t *flags,
        double *time)
{
    int ret = 0;
    size_t j;

    memset(flags, 0, self->num_nodes * sizeof(uint32_t));
    memcpy(time, self->node_time, self->num_nodes * sizeof(double));
    for (j = 0; j < self->num_samples; j++) {
        flags[self->num_ancestors + j] = 1;
    }
    return ret;
}

int
tree_sequence_builder_dump_edgesets(tree_sequence_builder_t *self,
        double *left, double *right, ancestor_id_t *parent, ancestor_id_t *children,
        uint32_t *children_length)
{
    int ret = 0;
    size_t j, k, offset;
    edgeset_t *e;

    /* Go through the edgesets and assign times */
    for (j = 0; j < self->num_edgesets; j++) {
        e = self->edgesets + j;
        e->time = self->node_time[e->parent];
    }
    qsort(self->edgesets, self->num_edgesets, sizeof(edgeset_t), cmp_edgeset);

    offset = 0;
    for (j = 0; j < self->num_edgesets; j++) {
        e = self->edgesets + j;
        left[j] = e->left;
        right[j] = e->right;
        parent[j] = e->parent;
        children_length[j] = e->num_children;
        for (k = 0; k < e->num_children; k++) {
            children[offset] = e->children[k];
            offset++;
        }
    }
    assert(offset == self->num_children);
    return ret;
}

int
tree_sequence_builder_dump_mutations(tree_sequence_builder_t *self,
        site_id_t *site, ancestor_id_t *node, allele_t *derived_state)
{
    int ret = 0;
    site_id_t l;
    size_t offset = 0;
    mutation_list_node_t *u;

    for (l = 0; l < self->num_sites; l++) {
        for (u = self->mutations[l]; u != NULL; u = u->next) {
            assert(offset < self->num_mutations);
            site[offset] = l;
            node[offset] = u->node;
            derived_state[offset] = u->derived_state;
            offset++;
        }
    }
    assert(offset == self->num_mutations);
    return ret;
}

static int
tree_sequence_builder_alloc_minor_node(tree_sequence_builder_t *self,
        ancestor_id_t *new_node)
{
    int ret = 0;
    double *p;

    if (self->num_nodes == self->max_num_nodes) {
        self->max_num_nodes += self->max_num_nodes;
        p = realloc(self->node_time, self->max_num_nodes * sizeof(double));
        if (p == NULL) {
            ret = TSI_ERR_NO_MEMORY;
            goto out;
        }
        self->node_time = p;
    }
    /* printf("ALLOC NODE %d\n", (int) self->num_nodes); */
    *new_node = (ancestor_id_t) self->num_nodes;
    self->num_nodes++;
out:
    return ret;
}

static edgeset_t *
tree_sequence_builder_alloc_edgeset(tree_sequence_builder_t *self, site_id_t left,
        site_id_t right, ancestor_id_t parent, uint32_t num_children)
{
    edgeset_t *ret = NULL;
    ancestor_id_t *children = NULL;
    edgeset_t *p;

    if (self->num_edgesets == self->max_num_edgesets) {
        /* Grow the array */
        self->max_num_edgesets += self->edgeset_block_size;
        p = realloc(self->edgesets, self->max_num_edgesets * sizeof(edgeset_t));
        if (p == NULL) {
            goto out;
        }
        self->edgesets = p;
    }
    /* NOOoooooooooooooooooooooooo!!!!!!!!!!! This must be replaced with a sensible
     * alloc strategy */
    children = malloc(num_children * sizeof(ancestor_id_t));
    if (children == NULL) {
        goto out;
    }
    ret = self->edgesets + self->num_edgesets;
    self->num_edgesets++;
    self->num_children += num_children;
    ret->left = left;
    ret->right = right;
    ret->parent = parent;
    ret->num_children = num_children;
    ret->children = children;
    children = NULL;
out:
    if (children != NULL) {
        free(children);
    }
    return ret;
}

static int
tree_sequence_builder_record_non_overlapping(tree_sequence_builder_t *self,
        site_id_t left, site_id_t right, ancestor_id_t parent, ancestor_id_t child)
{
    int ret = 0;
    node_mapping_t *u;
    edgeset_t *e;

    /* printf("RECORD NON_OVERLAP (%d, %d), %d %d\n", left, right, parent, child); */
    /* We shouldn't really be using a node_mapping here as it's wasteful of space. */
    u = tree_sequence_builder_alloc_node_mapping(self, left, right, 0);
    if (u == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    if (self->live_segments_head[parent] == NULL) {
        self->live_segments_head[parent] = u;
        self->live_segments_tail[parent] = u;
    } else {
        /* TODO: check for contiguous segments here */
        self->live_segments_tail[parent]->next = u;
        self->live_segments_tail[parent] = u;
    }

    e = tree_sequence_builder_alloc_edgeset(self, left, right, parent, 1);
    if (e == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    e->children[0] = child;
out:
    return ret;
}


static int
tree_sequence_builder_record_pairwise_coalescence(tree_sequence_builder_t *self,
        site_id_t left, site_id_t right, ancestor_id_t parent, ancestor_id_t child1,
        ancestor_id_t child2)
{
    int ret = 0;
    edgeset_t *e;

    e = tree_sequence_builder_alloc_edgeset(self, left, right, parent, 2);
    if (e == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    assert(left < right);
    /* printf("RECORD PAIRWISE COALESCENCE: (%d, %d): %d -> (%d, %d)\n", */
    /*         left, right, parent, child1, child2); */
    assert(child1 != child2);
    if (child1 < child2) {
        e->children[0] = child1;
        e->children[1] = child2;
    } else {
        e->children[0] = child2;
        e->children[1] = child1;
    }
out:

    return ret;
}

static int
tree_sequence_builder_record_coalescence(tree_sequence_builder_t *self,
        ancestor_id_t parent, size_t num_segments, node_mapping_t *segments)
{
    int ret = 0;
    size_t j;
    edgeset_t *e;

    assert(num_segments > 1);
    /* printf("RECORD COALESCENCE: parent = %d\n", parent); */
    e = tree_sequence_builder_alloc_edgeset(self, segments[0].left, segments[0].right,
            parent, num_segments);
    if (e == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    for (j = 0; j < num_segments; j++) {
        /* printf("\t(%d, %d, %d)\n", segments[j].left, segments[j].right, segments[j].node); */
        assert(segments[j].left == e->left);
        assert(segments[j].right == e->right);
        e->children[j] = segments[j].node;
    }
    qsort(e->children, e->num_children, sizeof(ancestor_id_t), cmp_node_id);
out:
    return ret;
}

inline static void
update_segment(node_mapping_t *seg, site_id_t left, site_id_t right, ancestor_id_t node)
{
    seg->left = left;
    seg->right = right;
    seg->node = node;
}

static int
tree_sequence_builder_resolve_identical(tree_sequence_builder_t *self,
        size_t num_segments, node_mapping_t *segments,
        size_t *num_result_segments, node_mapping_t *result_segments)
{
    int ret = 0;
    node_mapping_t *S = segments;
    ancestor_id_t parent;
    size_t j, k, num_returned;

    j = 0;
    k = 1;
    num_returned = 0;
    while (j < num_segments) {
        while (k < num_segments && S[j].left == S[k].left && S[j].right == S[k].right) {
            k++;
        }
        if (k > j + 1) {
            ret = tree_sequence_builder_alloc_minor_node(self, &parent);
            if (ret != 0) {
                goto out;
            }
            ret = tree_sequence_builder_record_coalescence(self, parent, k - j, S + j);
            if (ret != 0) {
                goto out;
            }
            S[j].node = parent;
        }
        assert(num_returned < num_segments);
        update_segment(result_segments + num_returned, S[j].left, S[j].right, S[j].node);
        num_returned++;
        j = k;
        k++;
    }
    assert(num_returned <= num_segments);
    *num_result_segments = num_returned;
out:
    return ret;
}

static int
tree_sequence_builder_resolve_largest_overlap(tree_sequence_builder_t *self,
        size_t num_segments, size_t max_num_segments, node_mapping_t *segments,
        size_t *num_result_segments, node_mapping_t *result_segments)
{
    int ret = 0;
    node_mapping_t *S = segments;
    node_mapping_t *R = result_segments;
    ancestor_id_t parent;
    size_t j, k, num_returned, intersection_j, intersection_k;
    int intersection, max_intersection;
    site_id_t right;

    /* Keep the compiler happy */
    intersection_j = 0;
    intersection_k = 0;

    max_intersection = 0;
    j = 0;
    k = 1;
    while (1) {
        while (k < num_segments && S[k].right <= S[j].right) {
            intersection = S[k].right - S[k].left;
            if (intersection > max_intersection) {
                max_intersection = intersection;
                intersection_j = j;
                intersection_k = k;
            }
            k++;
        }
        if (k == num_segments) {
            break;
        }
        intersection = S[j].right - S[k].left;
        if (intersection > max_intersection) {
            max_intersection = intersection;
            intersection_j = j;
            intersection_k = k;
        }
        j = k;
        k++;
    }

    /* for (j = 0; j < num_segments; j++) { */
    /*     printf("SEG: %d = (%d, %d) -> %d\n", (int) j, S[j].left, S[j].right, S[j].node); */
    /* } */
    if (max_intersection == 0) {
        memcpy(result_segments, S, num_segments * sizeof(node_mapping_t));
        num_returned = num_segments;
    } else {
        /* printf("max_intersection = %d (%d, %d)\n", max_intersection, */
        /*         (int) intersection_j, (int) intersection_k); */
        num_returned = 0;
        for (j = 0; j < intersection_j; j++) {
            assert(num_returned < max_num_segments);
            update_segment(R + num_returned, S[j].left, S[j].right, S[j].node);
            num_returned++;
        }
        j = intersection_j;
        k = intersection_k;
        if (S[j].left != S[k].left) {
            /* Trim off a new segment for the leading overhang. */
            assert(num_returned < max_num_segments);
            update_segment(R + num_returned, S[j].left, S[k].left, S[j].node);
            num_returned++;
            S[j].left = S[k].left;
        }
        /* Create a new segment for the coalesced region and record the coalescence */
        ret = tree_sequence_builder_alloc_minor_node(self, &parent);
        if (ret != 0) {
            goto out;
        }
        right = GSL_MIN(S[j].right, S[k].right);
        /* printf("left = %d right = %d\n~", S[j].left, right); */
        ret = tree_sequence_builder_record_pairwise_coalescence(self, S[j].left,
                right, parent, S[j].node, S[k].node);
        if (ret != 0) {
            goto out;
        }
        assert(num_returned < max_num_segments);
        update_segment(R + num_returned, S[j].left, right, parent);
        num_returned++;
        /* Create segments for any overhang on the right hand side */
        if (S[j].right > S[k].right) {
            assert(num_returned < max_num_segments);
            update_segment(R + num_returned, S[k].right, S[j].right, S[j].node);
            num_returned++;
        } else if (S[k].right > S[j].right) {
            assert(num_returned < max_num_segments);
            update_segment(R + num_returned, S[j].right, S[k].right, S[k].node);
            num_returned++;
        }
        /* Fill in any missing segments between j and k */
        for (j = intersection_j + 1; j < intersection_k; j++) {
            assert(num_returned < max_num_segments);
            update_segment(R + num_returned, S[j].left, S[j].right, S[j].node);
            num_returned++;
        }
        /* Fill out the remaining segments after k */
        for (j = intersection_k + 1; j < num_segments; j++) {
            assert(num_returned < max_num_segments);
            update_segment(R + num_returned, S[j].left, S[j].right, S[j].node);
            num_returned++;
        }
        /* TODO This is cheating!!!! We shouldn't need to sort the returned segments
         * here as we can maintain the sorted order with a bit of care.
         */
        qsort(R, num_returned, sizeof(node_mapping_t), cmp_node_mapping);
    }
    assert(num_returned <= max_num_segments);
    *num_result_segments = num_returned;
out:
    return ret;
}

static int
tree_sequence_builder_resolve_non_overlapping(tree_sequence_builder_t *self,
        ancestor_id_t ancestor, size_t num_segments, node_mapping_t *segments,
        size_t *num_result_segments, node_mapping_t *result_segments)
{
    int ret = 0;
    node_mapping_t *S = segments;
    node_mapping_t *R = result_segments;
    site_id_t next_left, max_right;
    size_t j, k, num_returned;

    j = 0;
    k = 1;
    max_right = 0;
    num_returned = 0;
    while (j < num_segments) {
        next_left = self->num_sites;
        if (j < num_segments - 1) {
            next_left = S[j + 1].left;
        }
        if (max_right <= S[j].left && S[j].right <= next_left) {
            ret = tree_sequence_builder_record_non_overlapping(self,
                    S[j].left, S[j].right, ancestor, S[j].node);
            if (ret != 0) {
                goto out;
            }
        } else {
            assert(num_returned < num_segments);
            update_segment(R + num_returned, S[j].left, S[j].right, S[j].node);
            num_returned++;
        }
        max_right = GSL_MAX(max_right, S[j].right);
        while (k < num_segments && S[k].right <= S[j].right) {
            assert(num_returned < num_segments);
            update_segment(R + num_returned, S[k].left, S[k].right, S[k].node);
            num_returned++;
            k++;
        }
        j = k;
        k++;
    }
    assert(num_returned <= num_segments);
    *num_result_segments = num_returned;
out:
    return ret;
}

static int
tree_sequence_builder_resolve_ancestor(tree_sequence_builder_t *self,
        ancestor_id_t ancestor, size_t num_segments, size_t max_num_segments,
        node_mapping_t *segments, node_mapping_t *results_buffer)
{
    int ret = 0;
    node_mapping_t *S[2];
    size_t N[2];
    int input_buffer, output_buffer;
    /* size_t last_iteration_size; */

    qsort(segments, num_segments, sizeof(node_mapping_t), cmp_node_mapping);

#ifdef RESOLVE_DEBUG
    tree_sequence_builder_draw_segments(self, num_segments, segments);
#endif

    S[0] = segments;
    N[0] = num_segments;
    S[1] = results_buffer;
    N[1] = 0;
    input_buffer = 0;
    output_buffer = 1;
    /* last_iteration_size = num_segments + 1; */

    /* printf("START %d\n", ancestor); */
    while (N[input_buffer] > 0) {
        /* assert(N[input_buffer] < last_iteration_size); */
        /* last_iteration_size = N[input_buffer]; */

        ret = tree_sequence_builder_resolve_identical(self,
                N[input_buffer], S[input_buffer], &N[output_buffer], S[output_buffer]);
        if (ret != 0) {
            goto out;
        }
        /* printf("AFTER IDENTICAL: %d\n", (int) N[output_buffer]); */
        /* tree_sequence_builder_draw_segments(self, N[output_buffer], S[output_buffer]); */
        input_buffer = output_buffer;
        output_buffer = (output_buffer + 1) % 2;

        ret = tree_sequence_builder_resolve_largest_overlap(self,
                N[input_buffer], max_num_segments,
                S[input_buffer], &N[output_buffer], S[output_buffer]);
        if (ret != 0) {
            goto out;
        }
        /* printf("AFTER MAX_OVERLAP : %d\n", (int) N[output_buffer]); */
        /* tree_sequence_builder_draw_segments(self, N[output_buffer], S[output_buffer]); */
        input_buffer = output_buffer;
        output_buffer = (output_buffer + 1) % 2;

        ret = tree_sequence_builder_resolve_non_overlapping(self, ancestor,
                N[input_buffer], S[input_buffer], &N[output_buffer], S[output_buffer]);
        if (ret != 0) {
            goto out;
        }
        /* printf("AFTER NON_OVERLAP: %d\n", (int) N[output_buffer]); */
        /* tree_sequence_builder_draw_segments(self, N[output_buffer], S[output_buffer]); */
        input_buffer = output_buffer;
        output_buffer = (output_buffer + 1) % 2;
    }
    /* printf("END %d\n", ancestor); */
out:
    return ret;
}

int
tree_sequence_builder_resolve(tree_sequence_builder_t *self, int epoch,
        ancestor_id_t *ancestors, size_t num_ancestors)
{
    int ret = 0;
    size_t j, k, max_num_segments, num_segments, num_nodes_before, used_nodes;
    double interval;
    node_mapping_t *u, *tmp;
    node_mapping_t *segments = NULL;
    node_mapping_t *segments_buffer = NULL;

    max_num_segments = 0;
    for (j = 0; j < num_ancestors; j++) {
        max_num_segments = GSL_MAX(max_num_segments, self->num_child_mappings[ancestors[j]]);
    }
    /* We can have some extra segments during coalescence, so allow for them */
    /* TODO this number is actually unbounded in pathological cases, so this is
     * approach is definitely bad. CHANGE!!*/
    max_num_segments *= 8;
    segments = malloc(max_num_segments * sizeof(node_mapping_t));
    segments_buffer = malloc(max_num_segments * sizeof(node_mapping_t));
    if (segments == NULL || segments_buffer == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }

    /* tree_sequence_builder_print_state(self, stdout); */
    /* printf("Resolving for epoch %d\n", epoch); */

    /* Search through the parent mappings for all referring to
    for (j = 0 j < self->num_major_nodes; j++) {

    }
    */
    for (j = 0; j < num_ancestors; j++) {
        /* printf("ancestor %d: num_child_mappings = %d\n", ancestors[j], */
        /*         self->num_child_mappings[ancestors[j]]); */
        num_segments = 0;
        u = self->child_mappings[ancestors[j]];
        while  (u != NULL) {
            segments[num_segments].left = u->left;
            segments[num_segments].right = u->right;
            segments[num_segments].node = u->node;
            num_segments++;
            tmp = u;
            u = u->next;
            tree_sequence_builder_free_node_mapping(self, tmp);
        }
        assert(num_segments == self->num_child_mappings[ancestors[j]]);
        self->child_mappings[ancestors[j]] = NULL;
        self->num_child_mappings[ancestors[j]] = 0;
        num_nodes_before = self->num_nodes;
        ret = tree_sequence_builder_resolve_ancestor(self, ancestors[j], num_segments,
                max_num_segments, segments, segments_buffer);
        if (ret != 0) {
            goto out;
        }
        /* Assign node times */
        self->node_time[ancestors[j]] = epoch;
        used_nodes = self->num_nodes - num_nodes_before;
        for (k = 0; k < used_nodes; k++) {
            interval = 1.0 / ((double) used_nodes + 1);
            self->node_time[num_nodes_before + k] = epoch - 1 + (k + 1) * interval;
        }
    }
    /* printf("AFTER resolve\n"); */
    /* tree_sequence_builder_print_state(self, stdout); */
out:
    tsi_safe_free(segments);
    tsi_safe_free(segments_buffer);
    return ret;
}
