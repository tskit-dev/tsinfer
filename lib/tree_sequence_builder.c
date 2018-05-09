/*
** Copyright (C) 2018 University of Oxford
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

#include "tsinfer.h"
#include "err.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#include "avl.h"


static int
cmp_edge_left_increasing_time(const void *a, const void *b) {
    const edge_t *ca = (const edge_t *) a;
    const edge_t *cb = (const edge_t *) b;
    int ret = (ca->left > cb->left) - (ca->left < cb->left);
    if (ret == 0) {
        ret = (ca->time > cb->time) - (ca->time < cb->time);
        if (ret == 0) {
            ret = (ca->child > cb->child) - (ca->child < cb->child);
        }
    }
    return ret;
}

static int
cmp_edge_right_decreasing_time(const void *a, const void *b) {
    const edge_t *ca = (const edge_t *) a;
    const edge_t *cb = (const edge_t *) b;
    int ret = (ca->right > cb->right) - (ca->right < cb->right);
    if (ret == 0) {
        ret = (ca->time < cb->time) - (ca->time > cb->time);
        if (ret == 0) {
            ret = (ca->child > cb->child) - (ca->child < cb->child);
        }
    }
    return ret;
}

static int
cmp_edge_path(const void *a, const void *b) {
    const edge_t *ca = (const edge_t *) a;
    const edge_t *cb = (const edge_t *) b;
    int ret = (ca->left > cb->left) - (ca->left < cb->left);
    if (ret == 0) {
        ret = (ca->right > cb->right) - (ca->right < cb->right);
        if (ret == 0) {
            ret = (ca->parent > cb->parent) - (ca->parent < cb->parent);
            if (ret == 0) {
                ret = (ca->child > cb->child) - (ca->child < cb->child);
            }
        }
    }
    return ret;
}

static void
print_edge_path(edge_t *head, FILE *out)
{
    edge_t *edge;

    for (edge = head; edge != NULL; edge = edge->next) {
        fprintf(out, "(%d, %d, %d, %d)", edge->left, edge->right, edge->parent,
                edge->child);
        if (edge->next != NULL) {
            fprintf(out, "->");
        }
    }
    fprintf(out, "\n");
}

static void
tree_sequence_builder_check_index_integrity(tree_sequence_builder_t *self)
{
    avl_node_t *avl_node;
    edge_t *edge;
    size_t j;

    for (j = 0; j < self->num_nodes; j++) {
        for (edge = self->path[j]; edge != NULL; edge = edge->next) {
            avl_node = avl_search(&self->left_index, edge);
            assert(avl_node != NULL);
            assert(avl_node->item == (void *) edge);

            avl_node = avl_search(&self->right_index, edge);
            assert(avl_node != NULL);
            assert(avl_node->item == (void *) edge);

            avl_node = avl_search(&self->path_index, edge);
            assert(avl_node != NULL);
            assert(avl_node->item == (void *) edge);
        }
    }
}

static void
tree_sequence_builder_check_state(tree_sequence_builder_t *self)
{
    node_id_t child;
    edge_t *edge;
    size_t total_edges = 0;

    for (child = 0; child < (node_id_t) self->num_nodes; child++) {
        for (edge = self->path[child]; edge != NULL; edge = edge->next) {
            total_edges++;
            assert(edge->child == child);
            if (edge->next != NULL) {
                /* contiguity can be violated for synthetic nodes */
                if (self->node_flags[edge->child] != 0) {
                    assert(edge->next->left == edge->right);
                }
            }
        }
    }
    assert(avl_count(&self->left_index) == total_edges);
    assert(avl_count(&self->right_index) == total_edges);
    assert(avl_count(&self->path_index) == total_edges);
    assert(total_edges == object_heap_get_num_allocated(&self->edge_heap));
    assert(3 * total_edges == object_heap_get_num_allocated(&self->avl_node_heap));
    tree_sequence_builder_check_index_integrity(self);
}

int
tree_sequence_builder_print_state(tree_sequence_builder_t *self, FILE *out)
{
    size_t j;
    mutation_list_node_t *u;
    avl_node_t *a;
    edge_t *edge;

    fprintf(out, "Tree sequence builder state\n");
    fprintf(out, "flags = %d\n", (int) self->flags);
    fprintf(out, "num_sites = %d\n", (int) self->num_sites);
    fprintf(out, "num_nodes = %d\n", (int) self->num_nodes);
    fprintf(out, "num_edges = %d\n", (int) tree_sequence_builder_get_num_edges(self));
    fprintf(out, "max_nodes = %d\n", (int) self->max_nodes);
    fprintf(out, "nodes_chunk_size = %d\n", (int) self->nodes_chunk_size);
    fprintf(out, "edges_chunk_size = %d\n", (int) self->edges_chunk_size);

    fprintf(out, "nodes = \n");
    fprintf(out, "id\tflags\ttime\tpath\n");
    for (j = 0; j < self->num_nodes; j++) {
        fprintf(out, "%d\t%d\t%f ", (int) j, self->node_flags[j], self->time[j]);
        print_edge_path(self->path[j], out);
    }

    fprintf(out, "mutations = \n");
    fprintf(out, "site\t(node, derived_state),...\n");
    for (j = 0; j < self->num_sites; j++) {
        if (self->sites.mutations[j] != NULL) {
            fprintf(out, "%d\t", (int) j);
            for (u = self->sites.mutations[j]; u != NULL; u = u->next) {
                fprintf(out, "(%d, %d) ", u->node, u->derived_state);

            }
            fprintf(out, "\n");
        }
    }
    fprintf(out, "path index \n");
    for (a = self->path_index.head; a != NULL; a = a->next) {
        edge = (edge_t *) a->item;
        fprintf(out, "%d\t%d\t%d\t%d\n", edge->left, edge->right,
                edge->parent, edge->child);
    }

    fprintf(out, "block_allocator = \n");
    block_allocator_print_state(&self->block_allocator, out);
    fprintf(out, "avl_node_heap = \n");
    object_heap_print_state(&self->avl_node_heap, out);
    fprintf(out, "edge_heap = \n");
    object_heap_print_state(&self->edge_heap, out);

    tree_sequence_builder_check_state(self);
    return 0;
}

int
tree_sequence_builder_alloc(tree_sequence_builder_t *self,
        size_t num_sites, size_t nodes_chunk_size, size_t edges_chunk_size, int flags)
{
    int ret = 0;
    memset(self, 0, sizeof(tree_sequence_builder_t));

    assert(num_sites < INT32_MAX);

    self->num_sites = num_sites;
    self->nodes_chunk_size = nodes_chunk_size;
    self->edges_chunk_size = edges_chunk_size;
    self->flags = flags;
    self->num_nodes = 0;
    self->max_nodes = nodes_chunk_size;

    self->time = malloc(self->max_nodes * sizeof(double));
    self->node_flags = malloc(self->max_nodes * sizeof(uint32_t));
    self->path = calloc(self->max_nodes, sizeof(edge_t *));
    self->sites.mutations = calloc(self->num_sites, sizeof(mutation_list_node_t));
    if (self->time == NULL || self->node_flags == NULL || self->path == NULL
            || self->sites.mutations == NULL)  {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    ret = object_heap_init(&self->avl_node_heap, sizeof(avl_node_t),
            self->edges_chunk_size, NULL);
    if (ret != 0) {
        goto out;
    }
    ret = object_heap_init(&self->edge_heap, sizeof(edge_t),
            self->edges_chunk_size, NULL);
    if (ret != 0) {
        goto out;
    }
    ret = block_allocator_alloc(&self->block_allocator,
            TSI_MAX(8192, num_sites * sizeof(mutation_list_node_t) / 4));
    if (ret != 0) {
        goto out;
    }
    avl_init_tree(&self->left_index, cmp_edge_left_increasing_time, NULL);
    avl_init_tree(&self->right_index, cmp_edge_right_decreasing_time, NULL);
    avl_init_tree(&self->path_index, cmp_edge_path, NULL);
out:
    return ret;
}

int
tree_sequence_builder_free(tree_sequence_builder_t *self)
{
    tsi_safe_free(self->time);
    tsi_safe_free(self->path);
    tsi_safe_free(self->node_flags);
    tsi_safe_free(self->sites.mutations);
    block_allocator_free(&self->block_allocator);
    object_heap_free(&self->avl_node_heap);
    object_heap_free(&self->edge_heap);
    return 0;
}

static inline avl_node_t * WARN_UNUSED
tree_sequence_builder_alloc_avl_node(tree_sequence_builder_t *self, edge_t *edge)
{
    avl_node_t *ret = NULL;

    if (object_heap_empty(&self->avl_node_heap)) {
        if (object_heap_expand(&self->avl_node_heap) != 0) {
            goto out;
        }
    }
    ret = (avl_node_t *) object_heap_alloc_object(&self->avl_node_heap);
    avl_init_node(ret, edge);
out:
    return ret;
}

static inline void
tree_sequence_builder_free_avl_node(tree_sequence_builder_t *self, avl_node_t *node)
{
    object_heap_free_object(&self->avl_node_heap, node);
}

static inline edge_t * WARN_UNUSED
tree_sequence_builder_alloc_edge(tree_sequence_builder_t *self,
        site_id_t left, site_id_t right, node_id_t parent, node_id_t child,
        edge_t *next)
{
    edge_t *ret = NULL;

    if (object_heap_empty(&self->edge_heap)) {
        if (object_heap_expand(&self->edge_heap) != 0) {
            goto out;
        }
    }
    assert(parent < (node_id_t) self->num_nodes);
    assert(child < (node_id_t) self->num_nodes);
    assert(self->time[parent] > self->time[child]);
    ret = (edge_t *) object_heap_alloc_object(&self->edge_heap);
    ret->left = left;
    ret->right = right;
    ret->parent = parent;
    ret->child = child;
    ret->time = self->time[child];
    ret->next = next;
out:
    return ret;
}

static inline void
tree_sequence_builder_free_edge(tree_sequence_builder_t *self, edge_t *edge)
{
    object_heap_free_object(&self->edge_heap, edge);
}

static int WARN_UNUSED
tree_sequence_builder_expand_nodes(tree_sequence_builder_t *self)
{
    int ret = 0;
    void *tmp;

    self->max_nodes += self->nodes_chunk_size;
    tmp = realloc(self->time, self->max_nodes * sizeof(double));
    if (tmp == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    self->time = tmp;
    tmp = realloc(self->node_flags, self->max_nodes * sizeof(uint32_t));
    if (tmp == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    self->node_flags = tmp;
    tmp = realloc(self->path, self->max_nodes * sizeof(edge_t *));
    if (tmp == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    self->path = tmp;
    /* Zero out the extra nodes. */
    memset(self->path + self->num_nodes, 0,
            (self->max_nodes - self->num_nodes) * sizeof(edge_t *));
out:
    return ret;
}

node_id_t WARN_UNUSED
tree_sequence_builder_add_node(tree_sequence_builder_t *self, double time, bool is_sample)
{
    int ret = 0;
    uint32_t flags = 0;

    if (self->num_nodes == self->max_nodes) {
        ret = tree_sequence_builder_expand_nodes(self);
        if (ret != 0) {
            goto out;
        }
    }
    assert(self->num_nodes < self->max_nodes);
    if (is_sample) {
        flags = 1;
    }
    ret = self->num_nodes;
    self->time[ret] = time;
    self->node_flags[ret] = flags;
    self->num_nodes++;
out:
    return ret;
}


static int WARN_UNUSED
tree_sequence_builder_add_mutation(tree_sequence_builder_t *self, site_id_t site,
        node_id_t node, allele_t derived_state)
{
    int ret = 0;
    mutation_list_node_t *list_node, *tail;

    assert(node < (node_id_t) self->num_nodes);
    assert(node >= 0);
    assert(site < (site_id_t) self->num_sites);
    assert(site >= 0);
    assert(derived_state == 0 || derived_state == 1);
    list_node = block_allocator_get(&self->block_allocator, sizeof(mutation_list_node_t));
    if (list_node == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    list_node->node = node;
    list_node->derived_state = derived_state;
    list_node->next = NULL;
    if (self->sites.mutations[site] == NULL) {
        self->sites.mutations[site] = list_node;
        assert(list_node->derived_state == 1);
    } else {
        tail = self->sites.mutations[site];
        while (tail->next != NULL) {
            tail = tail->next;
        }
        tail->next = list_node;
    }
    self->num_mutations++;
out:
    return ret;
}

static int WARN_UNUSED
tree_sequence_builder_unindex_edge(tree_sequence_builder_t *self, edge_t *edge)
{
    int ret = 0;
    avl_node_t *avl_node;

    avl_node = avl_search(&self->left_index, edge);
    assert(avl_node != NULL);
    avl_unlink_node(&self->left_index, avl_node);
    tree_sequence_builder_free_avl_node(self, avl_node);

    avl_node = avl_search(&self->right_index, edge);
    assert(avl_node != NULL);
    avl_unlink_node(&self->right_index, avl_node);
    tree_sequence_builder_free_avl_node(self, avl_node);

    avl_node = avl_search(&self->path_index, edge);
    assert(avl_node != NULL);
    avl_unlink_node(&self->path_index, avl_node);
    tree_sequence_builder_free_avl_node(self, avl_node);
    return ret;
}

static int WARN_UNUSED
tree_sequence_builder_index_edge(tree_sequence_builder_t *self, edge_t *edge)
{
    int ret = 0;
    avl_node_t *avl_node;

    avl_node = tree_sequence_builder_alloc_avl_node(self, edge);
    if (avl_node == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    avl_node = avl_insert_node(&self->left_index, avl_node);
    assert(avl_node != NULL);

    avl_node = tree_sequence_builder_alloc_avl_node(self, edge);
    if (avl_node == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    avl_node = avl_insert_node(&self->right_index, avl_node);
    assert(avl_node != NULL);

    avl_node = tree_sequence_builder_alloc_avl_node(self, edge);
    if (avl_node == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    avl_node = avl_insert_node(&self->path_index, avl_node);
    assert(avl_node != NULL);
out:
    return ret;
}

static int WARN_UNUSED
tree_sequence_builder_index_edges(tree_sequence_builder_t *self, node_id_t node)
{
    int ret = 0;
    edge_t *edge;

    for (edge = self->path[node]; edge != NULL; edge = edge->next) {
        ret = tree_sequence_builder_index_edge(self, edge);
        if (ret != 0) {
            goto out;
        }
    }
out:
    return ret;
}

/* Looks up the path index to find a matching edge, and returns it.
 */
static edge_t *
tree_sequence_builder_find_match(tree_sequence_builder_t *self, edge_t *query)
{
    edge_t *ret = NULL;
    edge_t search, *found;
    avl_node_t *avl_node;

    search.left = query->left;
    search.right = query->right;
    search.parent = query->parent;
    search.child = 0;

    avl_search_closest(&self->path_index, &search, &avl_node);
    if (avl_node != NULL) {
        found = (edge_t *) avl_node->item;
        if (found->left == query->left && found->right == query->right
                && found->parent == query->parent) {
            ret = found;
        } else {
            /* Check the adjacent nodes. */
            if (avl_node->prev != NULL) {
                found = (edge_t *) avl_node->prev->item;
                if (found->left == query->left && found->right == query->right
                        && found->parent == query->parent) {
                    ret = found;
                }
            }
            if (ret == NULL && avl_node->next != NULL) {
                found = (edge_t *) avl_node->next->item;
                if (found->left == query->left && found->right == query->right
                        && found->parent == query->parent) {
                    ret = found;
                }
            }
        }
    }
    return ret;
}

typedef struct {
    edge_t *source;
    edge_t *dest;
} edge_map_t;

typedef struct {
    node_id_t node;
    uint32_t count;
} node_counter_t;


/* Remap the edges in the set of matches to point to the already existing
 * synthethic node. */
static int
tree_sequence_builder_remap_synthetic(tree_sequence_builder_t *self,
        node_id_t mapped_child, int num_mapped, edge_map_t *mapped)
{
    int ret = 0;
    int j;

    for (j = 0; j < num_mapped; j++) {
        if (mapped[j].dest->child == mapped_child) {
            mapped[j].source->parent = mapped_child;
        }
    }
    return ret;
}

static void
tree_sequence_builder_squash_edges(tree_sequence_builder_t *self, node_id_t node)
{
    edge_t *x, *prev, *next;

    prev = self->path[node];
    assert(prev != NULL);
    x = prev->next;
    while (x != NULL) {
        next = x->next;
        assert(x->child == node);
        if (prev->right == x->left && prev->parent == x->parent) {
            prev->right = x->right;
            prev->next = next;
            tree_sequence_builder_free_edge(self, x);
        } else {
            prev = x;
        }
        x = next;
    }
}

/* Squash edges that can be squashed, but take into account that any modified
 * edges must be re-indexed. Some edges in the input chain may already be unindexed,
 * which are marked with a child value of NULL_NODE. */
static int WARN_UNUSED
tree_sequence_builder_squash_indexed_edges(tree_sequence_builder_t *self, node_id_t node)
{
    int ret = 0;
    edge_t *x, *prev, *next;

    prev = self->path[node];
    assert(prev != NULL);
    x = prev->next;
    while (x != NULL) {
        next = x->next;
        if (prev->right == x->left && prev->parent == x->parent) {
            /* We are pulling x out of the chain and extending prev to cover
             * the corresponding interval. Therefore, we must unindex prev and x. */
            if (prev->child != NULL_NODE) {
                ret = tree_sequence_builder_unindex_edge(self, prev);
                if (ret != 0) {
                    goto out;
                }
                prev->child = NULL_NODE;
            }
            if (x->child != NULL_NODE) {
                ret = tree_sequence_builder_unindex_edge(self, x);
                if (ret != 0) {
                    goto out;
                }
            }
            prev->right = x->right;
            prev->next = next;
            tree_sequence_builder_free_edge(self, x);
        } else {
            prev = x;
        }
        x = next;
    }

    /* Now index all the edges that have been unindexed */
    for (x = self->path[node]; x != NULL; x = x->next) {
        if (x->child == NULL_NODE) {
            x->child = node;
            ret = tree_sequence_builder_index_edge(self, x);
            if (ret != 0) {
                goto out;
            }
        }
    }
out:
    return ret;
}

/* Create a new synthetic ancestor which consists of the shared path
 * segments of existing ancestors. */
static int
tree_sequence_builder_make_synthetic_node(tree_sequence_builder_t *self,
        node_id_t mapped_child, int num_mapped, edge_map_t *mapped)
{
    int ret = 0;
    node_id_t synthetic_node;
    edge_t *edge;
    edge_t *head = NULL;
    edge_t *prev = NULL;
    double min_parent_time;
    int j;

    min_parent_time = self->time[0] + 1;
    for (j = 0; j < num_mapped; j++) {
        if (mapped[j].dest->child == mapped_child) {
            min_parent_time = TSI_MIN(
                min_parent_time, self->time[mapped[j].source->parent]);
        }
    }
    ret = tree_sequence_builder_add_node(self, min_parent_time - 0.125, false);
    if (ret < 0) {
        goto out;
    }
    synthetic_node = ret;

    for (j = 0; j < num_mapped; j++) {
        if (mapped[j].dest->child == mapped_child) {
            edge = tree_sequence_builder_alloc_edge(self,
                    mapped[j].source->left,
                    mapped[j].source->right,
                    mapped[j].source->parent,
                    synthetic_node, NULL);
            if (edge == NULL) {
                ret = TSI_ERR_NO_MEMORY;
                goto out;
            }
            if (head == NULL) {
                head = edge;
            } else {
                prev->next = edge;
            }
            prev = edge;
            mapped[j].source->parent = synthetic_node;
            /* We are modifying the existing edge, so we must remove it
             * from the indexes. Mark that it is unindexed by setting the
             * child value to NULL_NODE. */
            ret = tree_sequence_builder_unindex_edge(self, mapped[j].dest);
            if (ret != 0) {
                goto out;
            }
            mapped[j].dest->parent = synthetic_node;
            mapped[j].dest->child = NULL_NODE;
        }
    }
    self->path[synthetic_node] = head;
    tree_sequence_builder_squash_edges(self, synthetic_node);
    ret = tree_sequence_builder_squash_indexed_edges(self, mapped_child);
    if (ret != 0) {
        goto out;
    }
    ret = tree_sequence_builder_index_edges(self, synthetic_node);
    if (ret != 0) {
        goto out;
    }
out:
    return ret;
}

static int
tree_sequence_builder_compress_path(tree_sequence_builder_t *self, node_id_t child)
{
    int ret = 0;
    edge_t *c_edge, *match_edge;
    edge_map_t *mapped = NULL;
    node_counter_t *child_count = NULL;
    size_t path_length = 0;
    int num_mapped = 0;
    int num_mapped_children = 0;
    int j, k;
    node_id_t mapped_child;

    for (c_edge = self->path[child]; c_edge != NULL; c_edge = c_edge->next) {
        path_length++;
    }
    mapped = malloc(path_length * sizeof(*mapped));
    child_count = calloc(path_length, sizeof(*child_count));
    if (mapped == NULL || child_count == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }

    for (c_edge = self->path[child]; c_edge != NULL; c_edge = c_edge->next) {
        /* Can we find a match for this edge? */
        match_edge = tree_sequence_builder_find_match(self, c_edge);
        if (match_edge != NULL) {
            mapped[num_mapped].source = c_edge;
            mapped[num_mapped].dest = match_edge;
            num_mapped++;
        }
    }
    for (j = 0; j < num_mapped; j++) {
        mapped_child = mapped[j].dest->child;
        /* Increment the counter for this child. */
        for (k = 0; k < num_mapped_children; k++) {
            if (child_count[k].node == mapped_child) {
                break;
            }
        }
        if (k == num_mapped_children) {
            num_mapped_children++;
        }
        child_count[k].node = mapped_child;
        child_count[k].count++;
    }

    for (k = 0; k < num_mapped_children; k++) {
        if (child_count[k].count > 1) {
            mapped_child = child_count[k].node;
            if (self->node_flags[mapped_child] == 0) {
                ret = tree_sequence_builder_remap_synthetic(self, mapped_child,
                        num_mapped, mapped);
            } else {
                ret = tree_sequence_builder_make_synthetic_node(self, mapped_child,
                        num_mapped, mapped);
            }
            if (ret != 0) {
                goto out;
            }
        }
    }
    tree_sequence_builder_squash_edges(self, child);
out:
    tsi_safe_free(mapped);
    tsi_safe_free(child_count);
    return ret;
}

int
tree_sequence_builder_add_path(tree_sequence_builder_t *self,
        node_id_t child, size_t num_edges, site_id_t *left, site_id_t *right,
        node_id_t *parent, int flags)
{
    int ret = 0;
    edge_t *head = NULL;
    edge_t *prev = NULL;
    edge_t *edge;
    int j;

    /* Edges must be provided in reverese order */
    for (j = (int) num_edges - 1; j >= 0; j--) {
        edge = tree_sequence_builder_alloc_edge(self, left[j], right[j], parent[j],
                child, NULL);
        if (edge == NULL) {
            ret = TSI_ERR_NO_MEMORY;
            goto out;
        }
        if (head == NULL) {
            head = edge;
        } else {
            prev->next = edge;
            if (prev->right != edge->left) {
                ret = TSI_ERR_NONCONTIGUOUS_EDGES;
                goto out;
            }
        }
        prev = edge;
    }
    self->path[child] = head;
    if (flags & TSI_COMPRESS_PATH) {
        ret = tree_sequence_builder_compress_path(self, child);
        if (ret != 0) {
            goto out;
        }
    }
    ret = tree_sequence_builder_index_edges(self, child);
    if (flags & TSI_EXTENDED_CHECKS) {
        tree_sequence_builder_check_state(self);
    }
out:
    return ret;
}

int
tree_sequence_builder_add_mutations(tree_sequence_builder_t *self,
        node_id_t node, size_t num_mutations, site_id_t *site, allele_t *derived_state)
{
    int ret = 0;
    size_t j;

    for (j = 0; j < num_mutations; j++) {
        ret = tree_sequence_builder_add_mutation(self, site[j], node, derived_state[j]);
        if (ret != 0) {
            goto out;
        }
    }
out:
    return ret;
}

int
tree_sequence_builder_restore_nodes(tree_sequence_builder_t *self, size_t num_nodes,
        uint32_t *flags, double *time)
{
    int ret = -1;
    size_t j;

    for (j = 0; j < num_nodes; j++) {
        ret = tree_sequence_builder_add_node(self, time[j], flags[j] == 1);
        if (ret < 0) {
            goto out;
        }
    }
    ret = 0;
out:
    return ret;
}

int
tree_sequence_builder_restore_edges(tree_sequence_builder_t *self, size_t num_edges,
        site_id_t *left, site_id_t *right, node_id_t *parent, node_id_t *child)
{
    int ret = -1;
    size_t j;
    edge_t *edge, *prev;

    prev = NULL;
    for (j = 0; j < num_edges; j++) {
        if (j > 0 && child[j - 1] > child[j]) {
            ret = TSI_ERR_UNSORTED_EDGES;
            goto out;
        }
        edge = tree_sequence_builder_alloc_edge(self, left[j], right[j], parent[j],
                child[j], NULL);
        if (edge == NULL) {
            ret = TSI_ERR_NO_MEMORY;
            goto out;
        }
        if (self->path[child[j]] == NULL) {
            self->path[child[j]] = edge;
        } else {
            if (prev->right > edge->left) {
                ret = TSI_ERR_UNSORTED_EDGES;
                goto out;
            }
            prev->next = edge;
        }
        ret = tree_sequence_builder_index_edge(self, edge);
        if (ret != 0) {
            goto out;
        }
        prev = edge;
    }
    ret = 0;
out:
    return ret;
}

int
tree_sequence_builder_restore_mutations(tree_sequence_builder_t *self,
        size_t num_mutations, site_id_t *site, node_id_t *node, allele_t *derived_state)
{
    int ret = 0;
    size_t j = 0;

    for (j = 0; j < num_mutations; j++) {
        ret = tree_sequence_builder_add_mutation(self, site[j], node[j], derived_state[j]);
        if (ret != 0) {
            goto out;
        }
    }
out:
    return ret;
}

int
tree_sequence_builder_dump_nodes(tree_sequence_builder_t *self, uint32_t *flags,
        double *time)
{
    int ret = 0;
    size_t j;

    for (j = 0; j < self->num_nodes; j++) {
        flags[j] = self->node_flags[j];
        time[j] = self->time[j];
    }
    return ret;
}

int
tree_sequence_builder_dump_edges(tree_sequence_builder_t *self,
        node_id_t *left, node_id_t *right, ancestor_id_t *parent, ancestor_id_t *child)
{
    int ret = 0;
    size_t j, u;
    edge_t *e;

    j = 0;
    for (u = 0; u < self->num_nodes; u++) {
        e = self->path[u];
        while (e != NULL) {
            left[j] = e->left;
            right[j] = e->right;
            parent[j] = e->parent;
            child[j] = e->child;
            e = e->next;
            j++;
        }
    }
    return ret;
}

int
tree_sequence_builder_dump_mutations(tree_sequence_builder_t *self,
        site_id_t *site, ancestor_id_t *node, allele_t *derived_state,
        mutation_id_t *parent)
{
    int ret = 0;
    site_id_t l;
    mutation_list_node_t *u;
    mutation_id_t p;
    mutation_id_t j = 0;

    for (l = 0; l < (site_id_t) self->num_sites; l++) {
        p = j;
        for (u = self->sites.mutations[l]; u != NULL; u = u->next) {
            site[j] = l;
            node[j] = u->node;
            derived_state[j] = u->derived_state;
            parent[j] = -1;
            if (u->derived_state == 0) {
                parent[j] = p;
            }
            j++;
        }
    }
    return ret;
}

size_t
tree_sequence_builder_get_num_nodes(tree_sequence_builder_t *self)
{
    return self->num_nodes;
}

size_t
tree_sequence_builder_get_num_edges(tree_sequence_builder_t *self)
{
    return avl_count(&self->left_index);
}

size_t
tree_sequence_builder_get_num_mutations(tree_sequence_builder_t *self)
{
    return self->num_mutations;
}
