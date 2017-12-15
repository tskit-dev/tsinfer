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
        ret = (ca->right < cb->right) - (ca->right > cb->right);
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

#if 0
/* Sorts edges by (left, right, parent, child) order. */
static int
cmp_edge_lrpc(const void *a, const void *b) {
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

/* Sorts edges by (child, left, right, parent) order. */
static int
cmp_edge_clrp(const void *a, const void *b) {
    const edge_t *ca = (const edge_t *) a;
    const edge_t *cb = (const edge_t *) b;

    int ret = (ca->child > cb->child) - (ca->child < cb->child);
    if (ret == 0) {
        ret = (ca->left > cb->left) - (ca->left < cb->left);
        if (ret == 0) {
            ret = (ca->right > cb->right) - (ca->right < cb->right);
            if (ret == 0) {
                ret = (ca->parent > cb->parent) - (ca->parent < cb->parent);
            }
        }
    }
    return ret;
}
#endif


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
                /* TODO this can be violated for synethetic nodes */
                assert(edge->next->left == edge->right);
            }
        }
    }
    assert(avl_count(&self->left_index) == total_edges);
    assert(avl_count(&self->right_index) == total_edges);
    assert(avl_count(&self->path_index) == total_edges);
    assert(total_edges == object_heap_get_num_allocated(&self->edge_heap));
    assert(3 * total_edges == object_heap_get_num_allocated(&self->avl_node_heap));
}

int
tree_sequence_builder_print_state(tree_sequence_builder_t *self, FILE *out)
{
    size_t j;
    mutation_list_node_t *u;

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
        double sequence_length, size_t num_sites, double *position,
        double *recombination_rate, size_t nodes_chunk_size,
        size_t edges_chunk_size, int flags)
{
    int ret = 0;
    memset(self, 0, sizeof(tree_sequence_builder_t));

    assert(num_sites < INT32_MAX);

    self->sequence_length = sequence_length;
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
    self->sites.position = malloc(self->num_sites * sizeof(double));
    self->sites.recombination_rate = malloc(self->num_sites * sizeof(double));
    if (self->time == NULL || self->node_flags == NULL || self->path == NULL
            || self->sites.mutations == NULL || self->sites.position == NULL
            || self->sites.recombination_rate == NULL)  {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    memcpy(self->sites.position, position, self->num_sites * sizeof(double));
    memcpy(self->sites.recombination_rate, recombination_rate,
            self->num_sites * sizeof(double));
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
    tsi_safe_free(self->sites.position);
    tsi_safe_free(self->sites.recombination_rate);
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

/* static inline void */
/* tree_sequence_builder_free_avl_node(tree_sequence_builder_t *self, avl_node_t *node) */
/* { */
/*     object_heap_free_object(&self->avl_node_heap, node); */
/* } */

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

/* static inline void */
/* tree_sequence_builder_free_edge(tree_sequence_builder_t *self, edge_t *edge) */
/* { */
/*     object_heap_free_object(&self->edge_heap, edge); */
/* } */


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


#if 0

typedef struct {
    site_id_t start;
    site_id_t end;
} edge_group_t;

/* Returns true if the specified pair of paths through the specified set
 * of edges are considered equal. */
static bool
paths_equal(edge_t *edges, edge_group_t p1, edge_group_t p2)
{
    bool ret = false;
    site_id_t len = p1.end - p1.start;
    site_id_t j;
    edge_t edge1, edge2;

    if (len == (p2.end - p2.start)) {
        ret = true;
        for (j = 0; j < len; j++) {
            edge1 = edges[p1.start + j];
            edge2 = edges[p2.start + j];
            if (edge1.left != edge2.left || edge1.right != edge2.right
                    || edge1.parent != edge2.parent) {
                ret = false;
                break;
            }
        }
    }
    return ret;
}


static int
tree_sequence_builder_resolve_shared_recombs(tree_sequence_builder_t *self)
{
    int ret = 0;
    /* These are both probably way too much, but should be safe upper bounds */
    size_t max_edges = 2 * self->num_edges;
    size_t max_paths = self->num_edges;
    size_t j, k, num_output, num_active, num_filtered, num_paths, num_matches;
    size_t num_shared_recombinations = 0;
    site_id_t l;
    edge_t *active = NULL;
    edge_t *filtered = NULL;
    edge_t *output = NULL;
    edge_group_t *paths = NULL;
    edge_group_t path;
    bool *match_found = NULL;
    bool *marked = NULL;
    size_t *matches = NULL;
    size_t **shared_recombinations = NULL;
    edge_t *tmp;
    bool prev_cond, next_cond;
    double parent_time, children_time, new_time;
    site_id_t left, right;
    node_id_t new_node;

    active = malloc(max_edges * sizeof(edge_t));
    filtered = malloc(max_edges * sizeof(edge_t));
    output = malloc(max_edges * sizeof(edge_t));
    paths = malloc(max_paths * sizeof(edge_group_t));
    if (active == NULL || filtered == NULL || output == NULL || paths == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }

    /* Take a copy of all the extant edges and put them into the active buffer */
    for (j = 0; j < self->num_edges; j++) {
        active[j] = self->edges[j];
        assert(self->time[active[j].child] < self->time[active[j].parent]);
    }

    num_active = self->num_edges;
    num_filtered = 0;
    num_output = 0;
    num_paths = 0;
    /* First filter out all edges covering the full interval */
    for (j = 0; j < num_active; j++) {
        if (! (active[j].left == 0 && active[j].right == (site_id_t) self->num_sites)) {
            filtered[num_filtered] = active[j];
            num_filtered++;
        } else {
            output[num_output] = active[j];
            num_output++;
        }
    }
    tmp = active;
    active = filtered;
    filtered = tmp;
    num_active = num_filtered;
    num_filtered = 0;

    if (num_active > 0) {
        /* Sort by (l, r, p, c) to group together all identical (l, r, p) values. */
        qsort(active, num_active, sizeof(edge_t), cmp_edge_lrpc);
        prev_cond = false;
        for (j = 0; j < num_active - 1; j++) {
            next_cond = (
                active[j].left == active[j + 1].left &&
                active[j].right == active[j + 1].right &&
                active[j].parent == active[j + 1].parent);
            if (prev_cond || next_cond) {
                filtered[num_filtered] = active[j];
                num_filtered++;
            } else {
                output[num_output] = active[j];
                num_output++;
            }
            prev_cond = next_cond;
        }
        j = num_active - 1;
        if (prev_cond) {
            filtered[num_filtered] = active[j];
            num_filtered++;
        } else {
            output[num_output] = active[j];
            num_output++;
        }

        tmp = active;
        active = filtered;
        filtered = tmp;
        num_active = num_filtered;
        num_filtered = 0;
    }

    if (num_active > 0) {
        /* sort by (child, left, right) to group together all contiguous */
        /* TODO comparing by right and parent is probably redundant given the
         * previous filtering step */
        qsort(active, num_active, sizeof(edge_t), cmp_edge_clrp);
        prev_cond = false;
        for (j = 0; j < num_active - 1; j++) {
            next_cond = (
                active[j].right == active[j + 1].left &&
                active[j].child == active[j + 1].child);
            if (prev_cond || next_cond) {
                filtered[num_filtered] = active[j];
                num_filtered++;
            } else {
                output[num_output] = active[j];
                num_output++;
            }
            prev_cond = next_cond;
        }
        j = num_active - 1;
        if (prev_cond) {
            filtered[num_filtered] = active[j];
            num_filtered++;
        } else {
            output[num_output] = active[j];
            num_output++;
        }

        tmp = active;
        active = filtered;
        filtered = tmp;
        num_active = num_filtered;
        num_filtered = 0;
    }

    if (num_active > 0) {
        /* TODO is this step really necessary?? */
        /* In any case, this block is identical to the one above so it should
         * be abstracted out .*/

        /* We sort by left, right, parent again to find identical edges.
         * Remove any that there is only one of. */
        qsort(active, num_active, sizeof(edge_t), cmp_edge_lrpc);
        prev_cond = false;
        for (j = 0; j < num_active - 1; j++) {
            next_cond = (
                active[j].left == active[j + 1].left &&
                active[j].right == active[j + 1].right &&
                active[j].parent == active[j + 1].parent);
            if (prev_cond || next_cond) {
                filtered[num_filtered] = active[j];
                num_filtered++;
            } else {
                output[num_output] = active[j];
                num_output++;
            }
            prev_cond = next_cond;
        }
        j = num_active - 1;
        if (prev_cond) {
            filtered[num_filtered] = active[j];
            num_filtered++;
        } else {
            output[num_output] = active[j];
            num_output++;
        }

        tmp = active;
        active = filtered;
        filtered = tmp;
        num_active = num_filtered;
        num_filtered = 0;
    }

    if (num_active > 0) {
        assert(num_active + num_output == self->num_edges);
        /* Sort by the child, left again so that we can find the contiguous paths */
        qsort(active, num_active, sizeof(edge_t), cmp_edge_clrp);
        paths[0].start = 0;
        for (j = 1; j < num_active; j++) {
            if (active[j - 1].right != active[j].left ||
                    active[j - 1].child != active[j].child) {
                if (j - paths[num_paths].start > 1) {
                    paths[num_paths].end = j;
                    num_paths++;
                    assert(num_paths < max_paths);
                }
                paths[num_paths].start = j;
            }
        }
        j = num_active;
        if (j - paths[num_paths].start > 1) {
            paths[num_paths].end = j;
            num_paths++;
            assert(num_paths < max_paths);
        }
    }

    if (num_paths > 0) {
        match_found = calloc(num_paths, sizeof(bool));
        matches = malloc(num_paths * sizeof(size_t));
        shared_recombinations = calloc(num_paths, sizeof(size_t *));
        if (match_found == NULL || matches == NULL || shared_recombinations == NULL) {
            ret = TSI_ERR_NO_MEMORY;
            goto out;
        }
        for (j = 0; j < num_paths; j++) {
            num_matches = 0;
            if (! match_found[j]) {
                for (k = j + 1; k < num_paths; k++) {
                    if ((!match_found[k])
                            && paths_equal(active, paths[j], paths[k])) {
                        match_found[k] = true;
                        matches[num_matches] = k;
                        num_matches++;
                    }
                }
            }
            if (num_matches > 0) {
                shared_recombinations[num_shared_recombinations] = malloc(
                        (num_matches + 2) * sizeof(size_t));
                if (shared_recombinations[num_shared_recombinations] == NULL) {
                    ret = TSI_ERR_NO_MEMORY;
                    goto out;
                }
                shared_recombinations[num_shared_recombinations][0] = j;
                for (k = 0; k < num_matches; k++) {
                    shared_recombinations[num_shared_recombinations][k + 1] = matches[k];
                }
                /* Insert the sentinel */
                shared_recombinations[num_shared_recombinations][num_matches + 1] = -1;
                num_shared_recombinations++;
            }
        }
    }

    if (num_shared_recombinations > 0) {

        /* printf("Active  = \n"); */
        /* for (j = 0; j < num_active; j++) { */
        /*     printf("\t%d\t%d\t%d\t%d\n", active[j].left, active[j].right, */
        /*             active[j].parent, active[j].child); */
        /* } */
        /* printf("Paths:\n"); */
        /* for (j = 0; j < num_paths; j++) { */
        /*     printf("Path %d (%d, %d)\n", (int) j, paths[j].start, paths[j].end); */
        /*     for (k = paths[j].start; k < paths[j].end; k++) { */
        /*         printf("\t\t%d\t%d\t%d\t%d\n", active[k].left, active[k].right, */
        /*                 active[k].parent, active[k].child); */
        /*         assert(self->time[active[k].child] < self->time[active[k].parent]); */
        /*     } */
        /* } */
        /* printf("%d shared recombinations \n", (int) num_shared_recombinations); */
        /* for (j = 0; j < num_shared_recombinations; j++) { */
        /*     for (k = 0; shared_recombinations[j][k] != (size_t) (-1); k++) { */
        /*         printf("%d, ", (int) shared_recombinations[j][k]); */
        /*     } */
        /*     printf("\n"); */
        /* } */

        marked = calloc(num_active, sizeof(bool));
        if (marked == NULL) {
            ret = TSI_ERR_NO_MEMORY;
            goto out;
        }
        for (j = 0; j < num_shared_recombinations; j++) {
            /* printf("SHARED RECOMBINATION\n"); */
            /* for (k = 0; shared_recombinations[j][k] != (size_t) (-1); k++) { */
            /*     path = paths[shared_recombinations[j][k]]; */
            /*     printf("%d: (%d, %d)\n", (int) shared_recombinations[j][k], */
            /*             path.start, path.end); */
            /*     for (l = path.start; l < path.end; l++) { */
            /*         printf("\t%d\t%d\t%d\t%d\t:%.14f %.14f:%d %d\n", active[l].left, active[l].right, */
            /*                 active[l].parent, active[l].child, */
            /*                 self->time[active[l].parent], self->time[active[l].child], */
            /*                 self->node_flags[active[l].parent], self->node_flags[active[l].child]); */
            /*         assert(self->time[active[l].child] < self->time[active[l].parent]); */
            /*     } */
            /* } */
            /* check if we have a synthetic child on any of the paths */
            node_id_t synthetic_child = -1;
            for (k = 0; shared_recombinations[j][k] != (size_t) (-1); k++) {
                path = paths[shared_recombinations[j][k]];
                if (self->node_flags[active[path.start].child] == 0) {
                    synthetic_child = active[path.start].child;
                }
            }
            if (synthetic_child != -1) {
                /* If we have a synthetic child, the we already have a path covering
                 * the region. Update all other paths to use this existing path.
                 */
                /* printf("synthetic child = %d\n", synthetic_child); */

                for (k = 0; shared_recombinations[j][k] != (size_t) (-1); k++) {
                    path = paths[shared_recombinations[j][k]];
                    left = active[path.start].left;
                    right = active[path.end - 1].right;
                    parent_time = self->time[synthetic_child];
                    node_id_t child = active[path.start].child;
                    /* We can have situations where multiple paths occur at the same
                     * time. Easiest to just skip these */
                    double child_time = self->time[child];
                    if (child != synthetic_child && parent_time > child_time) {
                        /* Mark these edges as unused */
                        for (l = path.start; l < path.end; l++) {
                            assert(!marked[l]);
                            marked[l] = true;
                        }
                        l = path.start;
                        assert(num_output < max_edges);
                        output[num_output].left = left;
                        output[num_output].right = right;
                        output[num_output].parent = synthetic_child;
                        output[num_output].child = child;
                        /* printf("\tADD y %d\t%d\t%d\t%d\n", output[num_output].left, */
                        /*         output[num_output].right, */
                        /*         output[num_output].parent, output[num_output].child); */
                        num_output++;
                    }
                }
            } else {

                /* Mark these edges as used */
                for (k = 0; shared_recombinations[j][k] != (size_t) (-1); k++) {
                    path = paths[shared_recombinations[j][k]];
                    for (l = path.start; l < path.end; l++) {
                        assert(!marked[l]);
                        marked[l] = true;
                    }
                }
                path = paths[shared_recombinations[j][0]];
                left = active[path.start].left;
                right = active[path.end - 1].right;
                /* The parents from the first path */
                parent_time = self->time[0] + 1;
                for (l = path.start; l < path.end; l++) {
                     parent_time = TSI_MIN(parent_time, self->time[active[l].parent]);
                }
                /* Get the chilren time from the first edge in each path. */
                children_time = -1;
                for (k = 0; shared_recombinations[j][k] != (size_t) (-1); k++) {
                    path = paths[shared_recombinations[j][k]];
                    children_time = TSI_MAX(children_time, self->time[active[path.start].child]);
                }
                new_time = children_time + (parent_time - children_time) / 2;
                new_node = tree_sequence_builder_add_node(self, new_time, false);
                if (new_node < 0) {
                    ret = new_node;
                    goto out;
                }
                /* printf("parent_time = %f, children_time = %f node_time=%.14f node=%d\n", */
                /*         parent_time, children_time, new_time, new_node); */
                /* For each edge in the path, add a new edge with the new node as the
                 * child. */
                path = paths[shared_recombinations[j][0]];
                for (l = path.start; l < path.end; l++) {
                    assert(num_output < max_edges);
                    output[num_output].left = active[l].left;
                    output[num_output].right = active[l].right;
                    output[num_output].parent = active[l].parent;
                    output[num_output].child = new_node;
                    /* printf("\tADD x %d\t%d\t%d\t%d\n", output[num_output].left, */
                    /*         output[num_output].right, */
                    /*         output[num_output].parent, output[num_output].child); */
                    num_output++;
                }
                /* For each child add a new edge covering the whole interval */
                for (k = 0; shared_recombinations[j][k] != (size_t) (-1); k++) {
                    path = paths[shared_recombinations[j][k]];
                    l = path.start;
                    assert(num_output < max_edges);
                    output[num_output].left = left;
                    output[num_output].right = right;
                    output[num_output].parent = new_node;
                    output[num_output].child = active[l].child;
                    /* printf("\tADD y %d\t%d\t%d\t%d\n", output[num_output].left, */
                    /*         output[num_output].right, */
                    /*         output[num_output].parent, output[num_output].child); */
                    num_output++;
                }
            }
        }

        /* Finally append any unmarked edges to the output and save */
        for (j = 0; j < num_active; j++) {
            if (! marked[j]) {
                assert(num_output < max_edges);
                output[num_output] = active[j];
                num_output++;
            }
        }
        /* printf("OUTPUT\n"); */
        /* for (j = 0; j < num_output; j++) { */
        /*     printf("%d\t%d\t%d\t%d\n", output[j].left, output[j].right, */
        /*             output[j].parent, output[j].child); */

        /* } */
        memcpy(self->edges, output, num_output * sizeof(edge_t));
        self->num_edges = num_output;
    }

out:
    tsi_safe_free(active);
    tsi_safe_free(filtered);
    tsi_safe_free(output);
    tsi_safe_free(paths);
    tsi_safe_free(match_found);
    tsi_safe_free(matches);
    tsi_safe_free(marked);
    for (j = 0; j < num_shared_recombinations; j++) {
        tsi_safe_free(shared_recombinations[j]);
    }
    tsi_safe_free(shared_recombinations);
    return ret;
}
#endif

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

    /* printf("adding %d edges for child = %d\n", (int) num_edges, child); */
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
    ret = tree_sequence_builder_index_edges(self, child);

    tree_sequence_builder_check_state(self);
    /* tree_sequence_builder_print_state(self, stdout); */
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


/* int */
/* tree_sequence_builder_update(tree_sequence_builder_t *self, */
/*         size_t num_nodes, double time, */
/*         size_t num_edges, site_id_t *left, site_id_t *right, node_id_t *parent, */
/*         node_id_t *child, size_t num_mutations, site_id_t *site, node_id_t *node, */
/*         allele_t *derived_state) */
/* { */
/*     int ret = 0; */
/*     size_t j; */

/*     for (j = 0; j < num_nodes; j++) { */
/*         ret = tree_sequence_builder_add_node(self, time, true); */
/*         if (ret < 0) { */
/*             goto out; */
/*         } */
/*     } */
/*     for (j = 0; j < num_edges; j++) { */
/*         /1* printf("Insert edge left=%d, right=%d, parent=%d child=%d\n", *1/ */
/*         /1*         left[j], right[j], parent[j], child[j]); *1/ */
/*         ret = tree_sequence_builder_add_edge(self, left[j], right[j], parent[j], child[j]); */
/*         if (ret != 0) { */
/*             goto out; */
/*         } */
/*     } */
/*     for (j = 0; j < num_mutations; j++) { */
/*         ret = tree_sequence_builder_add_mutation(self, site[j], node[j], derived_state[j]); */
/*         if (ret != 0) { */
/*             goto out; */
/*         } */
/*     } */
/*     /1* if (self->flags & TSI_RESOLVE_SHARED_RECOMBS) { *1/ */
/*     /1*     ret = tree_sequence_builder_resolve_shared_recombs(self); *1/ */
/*     /1*     if (ret != 0) { *1/ */
/*     /1*         goto out; *1/ */
/*     /1*     } *1/ */
/*     /1* } *1/ */
/*     /1* ret = tree_sequence_builder_index_edges(self); *1/ */
/*     if (ret != 0) { */
/*         goto out; */
/*     } */
/* out: */
/*     return ret; */
/* } */

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
