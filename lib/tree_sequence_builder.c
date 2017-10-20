#include "tsinfer.h"
#include "err.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

/* TODO remove */
#include <gsl/gsl_math.h>

#include "avl.h"


static int
cmp_index_sort(const void *a, const void *b) {
    const index_sort_t *ca = (const index_sort_t *) a;
    const index_sort_t *cb = (const index_sort_t *) b;
    int ret = (ca->position > cb->position) - (ca->position < cb->position);
    if (ret == 0) {
        ret = (ca->time > cb->time) - (ca->time < cb->time);
    }
    return ret;
}

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

/* Sorts edges by (parent, left, right, child) order. */
static int
cmp_edge_plrc(const void *a, const void *b) {
    const edge_t *ca = (const edge_t *) a;
    const edge_t *cb = (const edge_t *) b;

    int ret = (ca->parent > cb->parent) - (ca->parent < cb->parent);
    if (ret == 0) {
        ret = (ca->left > cb->left) - (ca->left < cb->left);
        if (ret == 0) {
            ret = (ca->right > cb->right) - (ca->right < cb->right);
            if (ret == 0) {
                ret = (ca->child > cb->child) - (ca->child < cb->child);
            }
        }
    }
    return ret;
}

static void
tree_sequence_builder_check_state(tree_sequence_builder_t *self)
{
}


int
tree_sequence_builder_print_state(tree_sequence_builder_t *self, FILE *out)
{
    size_t j;
    edge_t *edge;
    mutation_list_node_t *u;

    fprintf(out, "Tree sequence builder state\n");
    fprintf(out, "flags = %d\n", (int) self->flags);
    fprintf(out, "num_sites = %d\n", (int) self->num_sites);
    fprintf(out, "num_nodes = %d\n", (int) self->num_nodes);
    fprintf(out, "num_edges = %d\n", (int) self->num_edges);
    fprintf(out, "max_nodes = %d\n", (int) self->max_nodes);
    fprintf(out, "max_edges = %d\n", (int) self->max_edges);

    fprintf(out, "edges = \n");
    fprintf(out, "left\tright\tparent\tchild\n");
    for (j = 0; j < self->num_edges; j++) {
        edge = self->edges + j;
        fprintf(out, "%d\t%d\t%d\t%d\t\t%d\t%d\n",
                edge->left, edge->right, edge->parent, edge->child,
                self->insertion_order[j], self->removal_order[j]);
    }
    fprintf(out, "Insertion order\n");
    for (j = 0; j < self->num_edges; j++) {
        edge = self->edges + self->insertion_order[j];
        fprintf(out, "%d\t%d\t%d\t%d\n",
                edge->left, edge->right, edge->parent, edge->child);
    }
    fprintf(out, "Removal order\n");
    for (j = 0; j < self->num_edges; j++) {
        edge = self->edges + self->removal_order[j];
        fprintf(out, "%d\t%d\t%d\t%d\n",
                edge->left, edge->right, edge->parent, edge->child);
    }
    fprintf(out, "nodes = \n");
    fprintf(out, "id\ttime\n");
    for (j = 0; j < self->num_nodes; j++) {
        fprintf(out, "%d\t%f\n", (int) j, self->time[j]);
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

    tree_sequence_builder_check_state(self);
    return 0;
}

int
tree_sequence_builder_alloc(tree_sequence_builder_t *self,
        double sequence_length, size_t num_sites, double *position,
        size_t max_nodes, size_t max_edges, int flags)
{
    int ret = 0;
    /* TODO put in a check on the number of sites. We currently use an integer
     * in the tree transition algorithm, so the max value of this and the
     * max value of site_id_t are the practical limits. Probably simpler make
     * site_id_t a signed integer in the long run */
    memset(self, 0, sizeof(tree_sequence_builder_t));
    self->sequence_length = sequence_length;
    self->num_sites = num_sites;
    self->max_nodes = max_nodes;
    self->max_edges = max_edges;
    self->flags = flags;
    self->num_nodes = 0;
    self->num_edges = 0;

    self->edges = malloc(self->max_edges * sizeof(edge_t));
    self->sort_buffer = malloc(self->max_edges * sizeof(index_sort_t));
    self->insertion_order = malloc(self->max_edges * sizeof(node_id_t));
    self->removal_order = malloc(self->max_edges * sizeof(node_id_t));
    self->time = malloc(self->max_nodes * sizeof(double));
    self->node_flags = malloc(self->max_nodes * sizeof(uint32_t));
    self->sites.mutations = calloc(self->num_sites, sizeof(mutation_list_node_t));
    self->sites.position = malloc(self->num_sites * sizeof(double));
    if (self->edges == NULL || self->time == NULL
            || self->insertion_order == NULL || self->removal_order == NULL
            || self->sort_buffer == NULL || self->sites.mutations == NULL
            || self->sites.position == NULL)  {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    memcpy(self->sites.position, position, self->num_sites * sizeof(double));

    ret = block_allocator_alloc(&self->block_allocator,
            GSL_MIN(1024, num_sites * sizeof(mutation_list_node_t) / 4));
    if (ret != 0) {
        goto out;
    }
out:
    return ret;
}

int
tree_sequence_builder_free(tree_sequence_builder_t *self)
{
    tsi_safe_free(self->edges);
    tsi_safe_free(self->time);
    tsi_safe_free(self->node_flags);
    tsi_safe_free(self->insertion_order);
    tsi_safe_free(self->removal_order);
    tsi_safe_free(self->sort_buffer);
    tsi_safe_free(self->sites.mutations);
    tsi_safe_free(self->sites.position);
    block_allocator_free(&self->block_allocator);
    return 0;
}

static node_id_t WARN_UNUSED
tree_sequence_builder_add_node(tree_sequence_builder_t *self, double time,
        bool is_sample)
{
    int ret = 0;
    uint32_t flags = 0;

    if (self->num_nodes == self->max_nodes) {
        /* FIXME */
        ret = -6;
        goto out;
    }
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
tree_sequence_builder_add_edge(tree_sequence_builder_t *self,
        site_id_t left, site_id_t right, node_id_t parent, node_id_t child)
{
    int ret = 0;
    edge_t *e;

    if (self->num_edges == self->max_edges) {
        /* FIXME */
        ret = -7;
        goto out;
    }
    e = self->edges + self->num_edges;
    e->left = left;
    e->right = right;
    e->parent = parent;
    e->child = child;
    self->num_edges++;
out:
    return ret;
}

static int WARN_UNUSED
tree_sequence_builder_index_edges(tree_sequence_builder_t *self)
{
    int ret = 0;
    size_t j;
    node_id_t u;
    index_sort_t *sort_buff = self->sort_buffer;

    /* sort by left and increasing time to give us the order in which
     * records should be inserted */
    for (j = 0; j < self->num_edges; j++) {
        sort_buff[j].index = (node_id_t ) j;
        sort_buff[j].position = self->edges[j].left;
        u = self->edges[j].parent;
        assert(u < (node_id_t) self->num_nodes);
        sort_buff[j].time = self->time[u];
    }
    qsort(sort_buff, self->num_edges, sizeof(index_sort_t), cmp_index_sort);
    for (j = 0; j < self->num_edges; j++) {
        self->insertion_order[j] = sort_buff[j].index;
    }
    /* sort by right and decreasing time to give us the order in which
     * records should be removed. */
    for (j = 0; j < self->num_edges; j++) {
        sort_buff[j].index = (node_id_t ) j;
        sort_buff[j].position = self->edges[j].right;
        u = self->edges[j].parent;
        assert(u < (node_id_t) self->num_nodes);
        sort_buff[j].time = -1 * self->time[u];
    }
    qsort(sort_buff, self->num_edges, sizeof(index_sort_t), cmp_index_sort);
    for (j = 0; j < self->num_edges; j++) {
        self->removal_order[j] = sort_buff[j].index;
    }
    return ret;
}

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
    size_t j, k, l, num_output, num_active, num_filtered, num_paths, num_matches;
    size_t num_shared_recombinations = 0;
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
    }

    num_active = self->num_edges;
    num_filtered = 0;
    num_output = 0;
    num_paths = 0;
    /* First filter out all edges covering the full interval */
    for (j = 0; j < num_active; j++) {
        if (! (active[j].left == 0 && active[j].right == self->num_sites)) {
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
            /*         printf("\t%d\t%d\t%d\t%d\n", active[l].left, active[l].right, */
            /*                 active[l].parent, active[l].child); */
            /*     } */
            /* } */

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
                 parent_time = GSL_MIN(parent_time, self->time[active[l].parent]);
            }
            /* Get the chilren time from the first edge in each path. */
            children_time = -1;
            for (k = 0; shared_recombinations[j][k] != (size_t) (-1); k++) {
                path = paths[shared_recombinations[j][k]];
                children_time = GSL_MAX(children_time, self->time[active[path.start].child]);
            }
            new_time = children_time + (parent_time - children_time) / 2;
            new_node = tree_sequence_builder_add_node(self, new_time, false);
            if (new_node < 0) {
                ret = new_node;
                goto out;
            }
            /* printf("New node %d at time %f\n", new_node, new_time); */
            /* For each edge in the path, add a new edge with the new node as the
             * child. If there are overhangs on either side of this interval we
             * insert edges pointing the new node to 0. */
            if (left != 0) {
                assert(num_output < max_edges);
                output[num_output].left = 0;
                output[num_output].right = left;
                output[num_output].parent = 0;
                output[num_output].child = new_node;
                num_output++;
            }
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
            if (right != self->num_sites) {
                assert(num_output < max_edges);
                output[num_output].left = right;
                output[num_output].right = self->num_sites;
                output[num_output].parent = 0;
                output[num_output].child = new_node;
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

static int
tree_sequence_builder_resolve_polytomies(tree_sequence_builder_t *self)
{
    int ret = 0;
    edge_t *edges = self->edges;
    size_t max_groups = self->num_edges;
    size_t *parent_count = NULL;
    edge_group_t *groups = NULL;
    size_t j, g, num_groups, size;
    node_id_t parent, new_node;
    double parent_time, children_time, new_time;

    parent_count = calloc(self->num_nodes, sizeof(size_t));
    groups = malloc(max_groups * sizeof(edge_group_t));
    if (parent_count == NULL || groups == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    for (j = 0; j < self->num_edges; j++) {
        parent_count[edges[j].parent]++;
    }

    /* Sort by (parent, left, right) to group together all edges for a given parent.
     */
    qsort(edges, self->num_edges, sizeof(edge_t), cmp_edge_plrc);

    /* printf("RESOLVE\n"); */
    /* for (j = 0; j < self->num_edges; j++) { */
    /*     printf("\t%d\t%d\t%d\t%d\n", edges[j].left, edges[j].right, */
    /*             edges[j].parent, edges[j].child); */
    /* } */

    num_groups = 0;
    groups[0].start = 0;
    for (j = 1; j < self->num_edges; j++) {
        if (edges[j - 1].left != edges[j].left
                || edges[j - 1].right != edges[j].right
                || edges[j - 1].parent != edges[j].parent) {
            size = j - groups[num_groups].start;
            if (size > 1 && size != parent_count[edges[j - 1].parent]) {
                groups[num_groups].end = j;
                num_groups++;
                assert(num_groups < max_groups);
            }
            groups[num_groups].start = j;
        }
    }
    j = self->num_edges;
    size = j - groups[num_groups].start;
    if (size > 1 && size != parent_count[edges[j - 1].parent]) {
        groups[num_groups].end = j;
        num_groups++;
        assert(num_groups < max_groups);
    }

    for (g = 0; g < num_groups; g++) {
        /* printf("Group: %d %d\n", groups[g].start, groups[g].end); */
        /* for (j = groups[g].start; j < groups[g].end; j++) { */
        /*     printf("\t%d\t%d\t%d\t%d\n", edges[j].left, edges[j].right, */
        /*             edges[j].parent, edges[j].child); */
        /* } */
        parent = edges[groups[g].start].parent;
        parent_time = self->time[parent];
        children_time = -1;
        for (j = groups[g].start; j < groups[g].end; j++) {
            children_time = GSL_MAX(children_time, self->time[edges[j].child]);
        }
        new_time = children_time + (parent_time - children_time) / 2;
        new_node = tree_sequence_builder_add_node(self, new_time, false);
        if (new_node < 0) {
            ret = new_node;
            goto out;
        }
        /* Update the existing edges to point to this new node. */
        for (j = groups[g].start; j < groups[g].end; j++) {
            edges[j].parent = new_node;
            /* printf("U\t%d\t%d\t%d\t%d\n", edges[j].left, edges[j].right, */
            /*         edges[j].parent, edges[j].child); */
        }
        /* Insert a new edge */
        ret = tree_sequence_builder_add_edge(self, 0, self->num_sites, parent, new_node);
        if (ret != 0) {
            goto out;
        }
        /* printf("Inserted:"); */
        /* j = self->num_edges - 1; */
        /* printf("U\t%d\t%d\t%d\t%d\n", edges[j].left, edges[j].right, */
        /*         edges[j].parent, edges[j].child); */
    }
out:
    tsi_safe_free(parent_count);
    tsi_safe_free(groups);
    return ret;
}

int
tree_sequence_builder_update(tree_sequence_builder_t *self,
        size_t num_nodes, double time,
        size_t num_edges, site_id_t *left, site_id_t *right, node_id_t *parent,
        node_id_t *child, size_t num_mutations, site_id_t *site, node_id_t *node,
        allele_t *derived_state)
{
    int ret = 0;
    size_t j;
    edge_t *e;
    mutation_list_node_t *list_node, *tail;

    for (j = 0; j < num_nodes; j++) {
        ret = tree_sequence_builder_add_node(self, time, true);
        if (ret < 0) {
            goto out;
        }
    }
    /* We assume that the edges are given in reverse order, so we insert them this
     * way around to get closer to sortedness */
    for (j = 0; j < num_edges; j++) {
        assert(self->num_edges < self->max_edges);
        e = self->edges + self->num_edges;
        e->left = left[j];
        e->right = right[j];
        e->parent = parent[j];
        e->child = child[j];
        assert(e->left < e->right);
        assert(e->parent != NULL_NODE);
        assert(e->child != NULL_NODE);
        assert(e->child < (node_id_t) self->num_nodes);
        assert(e->parent < (node_id_t) self->num_nodes);
        self->num_edges++;
    }

    for (j = 0; j < num_mutations; j++) {
        assert(node[j] < (node_id_t) self->num_nodes);
        assert(node[j] >= 0);
        assert(site[j] < self->num_sites);
        assert(derived_state[j] == 0 || derived_state[j] == 1);
        list_node = block_allocator_get(&self->block_allocator,
                sizeof(mutation_list_node_t));
        if (list_node == NULL) {
            ret = TSI_ERR_NO_MEMORY;
            goto out;
        }
        list_node->node = node[j];
        list_node->derived_state = derived_state[j];
        list_node->next = NULL;
        if (self->sites.mutations[site[j]] == NULL) {
            self->sites.mutations[site[j]] = list_node;
            assert(list_node->derived_state == 1);
        } else {
            tail = self->sites.mutations[site[j]];
            while (tail->next != NULL) {
                tail = tail->next;
            }
            tail->next = list_node;
        }
    }
    self->num_mutations += num_mutations;

    if (self->flags & TSI_RESOLVE_SHARED_RECOMBS) {
        ret = tree_sequence_builder_resolve_shared_recombs(self);
        if (ret != 0) {
            goto out;
        }
    }
    if (self->flags & TSI_RESOLVE_POLYTOMIES) {
        if (self->num_edges > 1) {
            ret = tree_sequence_builder_resolve_polytomies(self);
            if (ret != 0) {
                goto out;
            }
        }
    }
    ret = tree_sequence_builder_index_edges(self);
    if (ret != 0) {
        goto out;
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

/* Translates the specified site-index edge coordinate into the appropriate value
 * in the external coordinate system */
static double
tree_sequence_builder_translate_coord(tree_sequence_builder_t *self, site_id_t coord)
{
    double ret = -1;

    if (coord == 0) {
        ret = 0;
    } else if (coord == self->num_sites) {
        ret = self->sequence_length;
    } else if (coord < self->num_sites) {
        ret = self->sites.position[coord];
    }
    assert(ret != -1);
    return ret;
}

int
tree_sequence_builder_dump_edges(tree_sequence_builder_t *self,
        double *left, double *right, ancestor_id_t *parent, ancestor_id_t *child)
{
    int ret = 0;
    size_t j;
    edge_t *e;

    for (j = 0; j < self->num_edges; j++) {
        e = self->edges + j;
        left[j] = tree_sequence_builder_translate_coord(self, (site_id_t) e->left);
        right[j] = tree_sequence_builder_translate_coord(self, (site_id_t) e->right);
        parent[j] = e->parent;
        child[j] = e->child;
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

    for (l = 0; l < self->num_sites; l++) {
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
