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


static void
tree_sequence_builder_check_state(tree_sequence_builder_t *self)
{
}


int
tree_sequence_builder_print_state(tree_sequence_builder_t *self, FILE *out)
{
    size_t j;
    edge_t *edge;

    fprintf(out, "Tree sequence builder state\n");
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
    fprintf(out, "nodes = \n");
    fprintf(out, "id\ttime\n");
    for (j = 0; j < self->num_nodes; j++) {
        fprintf(out, "%d\t%f\n", (int) j, self->time[j]);
    }
    fprintf(out, "mutations = \n");
    fprintf(out, "site\tnode\n");
    for (j = 0; j < self->num_sites; j++) {
        if (self->mutations[j] != NULL_NODE) {
            fprintf(out, "%d\t%d\n", (int) j, self->mutations[j]);
        }
    }
    tree_sequence_builder_check_state(self);
    return 0;
}

int
tree_sequence_builder_alloc(tree_sequence_builder_t *self, size_t num_sites,
        size_t max_nodes, size_t max_edges)
{
    int ret = 0;
    /* TODO put in a check on the number of sites. We currently use an integer
     * in the tree transition algorithm, so the max value of this and the
     * max value of site_id_t are the practical limits. Probably simpler make
     * site_id_t a signed integer in the long run */
    memset(self, 0, sizeof(tree_sequence_builder_t));
    self->num_sites = num_sites;
    self->max_nodes = max_nodes;
    self->max_edges = max_edges;
    self->num_nodes = 0;
    self->num_edges = 0;

    self->edges = malloc(self->max_edges * sizeof(edge_t));
    self->sort_buffer = malloc(self->max_edges * sizeof(index_sort_t));
    self->insertion_order = malloc(self->max_edges * sizeof(node_id_t));
    self->removal_order = malloc(self->max_edges * sizeof(node_id_t));
    self->time = malloc(self->max_nodes * sizeof(double));
    self->mutations = malloc(self->num_sites * sizeof(node_id_t));
    if (self->edges == NULL || self->time == NULL
            || self->insertion_order == NULL || self->removal_order == NULL
            || self->sort_buffer == NULL || self->mutations == NULL)  {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    memset(self->mutations, 0xff, self->num_sites * sizeof(node_id_t));
out:
    return ret;
}

int
tree_sequence_builder_free(tree_sequence_builder_t *self)
{
    tsi_safe_free(self->edges);
    tsi_safe_free(self->time);
    tsi_safe_free(self->insertion_order);
    tsi_safe_free(self->removal_order);
    tsi_safe_free(self->sort_buffer);
    tsi_safe_free(self->mutations);
    return 0;
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

int
tree_sequence_builder_update(tree_sequence_builder_t *self,
        size_t num_nodes, double time,
        size_t num_edges, site_id_t *left, site_id_t *right, node_id_t *parent,
        node_id_t *child, size_t num_mutations, site_id_t *site, node_id_t *node)
{
    int ret = 0;
    size_t j;
    edge_t *e;

    assert(self->num_nodes + num_nodes < self->max_nodes);
    for (j = 0; j < num_nodes; j++) {
        self->time[self->num_nodes + j] = time;
    }
    self->num_nodes += num_nodes;
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
    ret = tree_sequence_builder_index_edges(self);
    if (ret != 0) {
        goto out;
    }
    for (j = 0; j < num_mutations; j++) {
        assert(node[j] < (node_id_t) self->num_nodes);
        assert(node[j] >= 0);
        assert(site[j] < self->num_sites);
        assert(self->mutations[site[j]] == NULL_NODE);
        self->mutations[site[j]] = node[j];
    }
    self->num_mutations += num_mutations;
    tree_sequence_builder_print_state(self, stdout);
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
        flags[j] = 1;
        time[j] = self->time[j];
    }
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
        left[j] = e->left;
        right[j] = e->right;
        parent[j] = e->parent;
        child[j] = e->child;
    }
    return ret;
}

int
tree_sequence_builder_dump_mutations(tree_sequence_builder_t *self,
        site_id_t *site, ancestor_id_t *node, allele_t *derived_state)
{
    int ret = 0;
    site_id_t l;
    size_t j = 0;

    for (l = 0; l < self->num_sites; l++) {
        if (self->mutations[l] != NULL_NODE) {
            site[j] = l;
            node[j] = self->mutations[l];
            derived_state[j] = 1;
            j++;
        }
    }
    return ret;
}
