#include "tsinfer.h"
#include "err.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#include "avl.h"


static int
cmp_edge_insertion(const void *a, const void *b) {
    const edge_sort_t *ca = (const edge_sort_t *) a;
    const edge_sort_t *cb = (const edge_sort_t *) b;
    int ret = (ca->left > cb->left) - (ca->left < cb->left);
    if (ret == 0) {
        ret = (ca->time > cb->time) - (ca->time < cb->time);
        if (ret == 0) {
            ret = (ca->parent > cb->parent) - (ca->parent < cb->parent);
            if (ret == 0) {
                ret = (ca->child > cb->child) - (ca->child < cb->child);
            }
        }
    }
    return ret;
}

static int
cmp_edge_removal(const void *a, const void *b) {
    const edge_sort_t *ca = (const edge_sort_t *) a;
    const edge_sort_t *cb = (const edge_sort_t *) b;
    int ret = (ca->right > cb->right) - (ca->right < cb->right);
    if (ret == 0) {
        ret = (ca->time < cb->time) - (ca->time > cb->time);
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
        fprintf(out, "%d\t%d\t%d\t%d\t\t%d\n",
                edge->left, edge->right, edge->parent, edge->child,
                self->removal_order[j]);
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
        double *recombination_rate, size_t max_nodes, size_t max_edges, int flags)
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
    self->sort_buffer = malloc(self->max_edges * sizeof(edge_sort_t));
    self->removal_order = malloc(self->max_edges * sizeof(node_id_t));
    self->time = malloc(self->max_nodes * sizeof(double));
    self->node_flags = malloc(self->max_nodes * sizeof(uint32_t));
    self->sites.mutations = calloc(self->num_sites, sizeof(mutation_list_node_t));
    self->sites.position = malloc(self->num_sites * sizeof(double));
    self->sites.recombination_rate = malloc(self->num_sites * sizeof(double));
    if (self->edges == NULL || self->time == NULL || self->removal_order == NULL
            || self->sort_buffer == NULL || self->sites.mutations == NULL
            || self->sites.position == NULL || self->sites.recombination_rate == NULL)  {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    memcpy(self->sites.position, position, self->num_sites * sizeof(double));
    memcpy(self->sites.recombination_rate, recombination_rate,
            self->num_sites * sizeof(double));

    ret = block_allocator_alloc(&self->block_allocator,
            TSI_MAX(8192, num_sites * sizeof(mutation_list_node_t) / 4));
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
    tsi_safe_free(self->removal_order);
    tsi_safe_free(self->sort_buffer);
    tsi_safe_free(self->sites.mutations);
    tsi_safe_free(self->sites.position);
    tsi_safe_free(self->sites.recombination_rate);
    block_allocator_free(&self->block_allocator);
    return 0;
}

static node_id_t WARN_UNUSED
tree_sequence_builder_add_node(tree_sequence_builder_t *self, double time, bool is_sample)
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
tree_sequence_builder_index_edges(tree_sequence_builder_t *self)
{
    int ret = 0;
    size_t j;
    node_id_t u;
    edge_sort_t *sort_buff = self->sort_buffer;

    /* sort by left and increasing time to give us the order in which
     * records should be inserted */
    for (j = 0; j < self->num_edges; j++) {
        sort_buff[j].left = self->edges[j].left;
        sort_buff[j].right = self->edges[j].right;
        sort_buff[j].parent = self->edges[j].parent;
        sort_buff[j].child = self->edges[j].child;
        u = self->edges[j].parent;
        assert(u < (node_id_t) self->num_nodes);
        sort_buff[j].time = self->time[u];
    }
    qsort(sort_buff, self->num_edges, sizeof(edge_sort_t), cmp_edge_insertion);
    for (j = 0; j < self->num_edges; j++) {
        self->edges[j].left = sort_buff[j].left;
        self->edges[j].right = sort_buff[j].right;
        self->edges[j].parent = sort_buff[j].parent;
        self->edges[j].child = sort_buff[j].child;
    }
    /* sort by right and decreasing time to give us the order in which
     * records should be removed. */
    for (j = 0; j < self->num_edges; j++) {
        sort_buff[j].index = j;
        sort_buff[j].left = self->edges[j].left;
        sort_buff[j].right = self->edges[j].right;
        sort_buff[j].parent = self->edges[j].parent;
        sort_buff[j].child = self->edges[j].child;
        u = self->edges[j].parent;
        assert(u < (node_id_t) self->num_nodes);
        sort_buff[j].time = self->time[u];
    }

    qsort(sort_buff, self->num_edges, sizeof(edge_sort_t), cmp_edge_removal);
    for (j = 0; j < self->num_edges; j++) {
        self->removal_order[j] = sort_buff[j].index;
    }
    return ret;
}

static int WARN_UNUSED
tree_sequence_builder_expand_edges(tree_sequence_builder_t *self)
{
    int ret = 0;
    void *tmp;

    self->max_edges *= 2;
    tmp = realloc(self->edges, self->max_edges * sizeof(edge_t));
    if (tmp == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    self->edges = tmp;
    tmp = realloc(self->sort_buffer, self->max_edges * sizeof(edge_sort_t));
    if (tmp == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    self->sort_buffer = tmp;
    tmp = realloc(self->removal_order, self->max_edges * sizeof(node_id_t));
    if (tmp == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    self->removal_order = tmp;
out:
    return ret;
}

static int WARN_UNUSED
tree_sequence_builder_add_edge(tree_sequence_builder_t *self, site_id_t left,
        site_id_t right, node_id_t parent, node_id_t child)
{
    int ret = 0;
    edge_t *e;

    if (self->num_edges == self->max_edges) {
        ret = tree_sequence_builder_expand_edges(self);
        if (ret != 0) {
            goto out;
        }
    }
    assert(self->num_edges < self->max_edges);
    e = self->edges + self->num_edges;
    e->left = left;
    e->right = right;
    e->parent = parent;
    e->child = child;
    assert(e->left < e->right);
    assert(e->parent != NULL_NODE);
    assert(e->child != NULL_NODE);
    assert(e->child < (node_id_t) self->num_nodes);
    assert(e->parent < (node_id_t) self->num_nodes);
    self->num_edges++;
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

int
tree_sequence_builder_update(tree_sequence_builder_t *self,
        size_t num_nodes, double time,
        size_t num_edges, site_id_t *left, site_id_t *right, node_id_t *parent,
        node_id_t *child, size_t num_mutations, site_id_t *site, node_id_t *node,
        allele_t *derived_state)
{
    int ret = 0;
    size_t j;

    for (j = 0; j < num_nodes; j++) {
        ret = tree_sequence_builder_add_node(self, time, true);
        if (ret < 0) {
            goto out;
        }
    }
    for (j = 0; j < num_edges; j++) {
        ret = tree_sequence_builder_add_edge(self, left[j], right[j], parent[j], child[j]);
        if (ret != 0) {
            goto out;
        }
    }
    for (j = 0; j < num_mutations; j++) {
        ret = tree_sequence_builder_add_mutation(self, site[j], node[j], derived_state[j]);
        if (ret != 0) {
            goto out;
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
tree_sequence_builder_restore_nodes(tree_sequence_builder_t *self, size_t num_nodes,
        double *time)
{
    int ret = 0;
    size_t j;

    for (j = 0; j < num_nodes; j++) {
        ret = tree_sequence_builder_add_node(self, time[j], true);
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
    int ret = 0;
    size_t j;

    for (j = 0; j < num_edges; j++) {
        ret = tree_sequence_builder_add_edge(self, left[j], right[j], parent[j], child[j]);
        if (ret != 0) {
            goto out;
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
