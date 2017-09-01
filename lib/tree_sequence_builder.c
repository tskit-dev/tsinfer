#include "tsinfer.h"
#include "err.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

/* TODO remove */
#include <gsl/gsl_math.h>

#include "avl.h"

#define NULL_LIKELIHOOD (-1)
#define NULL_NODE (-1)

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

static int
cmp_node_id(const void *a, const void *b) {
    const node_id_t *ia = (const node_id_t *) a;
    const node_id_t *ib = (const node_id_t *) b;
    return (*ia > *ib) - (*ia < *ib);
}

/* Returns true if x is approximately equal to one. */
static bool
approximately_one(double x)
{
    double eps = 1e-8;
    return fabs(x - 1.0) < eps;
}

static void
tree_sequence_builder_check_state(tree_sequence_builder_t *self)
{
    size_t num_likelihoods;
    avl_node_t *a;
    size_t j;
    node_id_t u;
    double x;
    likelihood_list_t *z;

    /* Check the properties of the likelihood map */
    for (a = self->likelihood_nodes.head; a != NULL; a = a->next) {
        u = *((node_id_t *) a->item);
        assert(self->likelihood[u] != NULL_LIKELIHOOD);
        x = self->likelihood[u];
        u = self->parent[u];
        if (u != NULL_NODE) {
            /* Traverse up to the next L value, and ensure it's not equal to x */
            while (self->likelihood[u] == NULL_LIKELIHOOD) {
                u = self->parent[u];
                assert(u != NULL_NODE);
            }
            assert(self->likelihood[u] != x);
        }
    }
    /* Make sure that there are no other non null likelihoods in the array */
    num_likelihoods = 0;
    for (u = 0; u < (node_id_t) self->num_nodes; u++) {
        if (self->likelihood[u] != NULL_LIKELIHOOD) {
            num_likelihoods++;
        }
    }
    assert(num_likelihoods == avl_count(&self->likelihood_nodes));
    assert(avl_count(&self->likelihood_nodes) ==
            object_heap_get_num_allocated(&self->avl_node_heap));

    for (j = 0; j < self->num_sites; j++) {
        z = self->traceback[j];
        if (z != NULL) {
            /* There must be at least one node with likelihood == 1. */
            u = NULL_NODE;
            while (z != NULL) {
                if (approximately_one(z->likelihood)) {
                    u = z->node;
                }
                z = z->next;
            }
            assert(u != NULL_NODE);
        }
    }
}

int
tree_sequence_builder_print_state(tree_sequence_builder_t *self, FILE *out)
{
    size_t j;
    likelihood_list_t *l;
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
    fprintf(out, "tree = \n");
    fprintf(out, "id\tparent\tlikelihood\n");
    for (j = 0; j < self->num_nodes; j++) {
        fprintf(out, "%d\t%d\t%f\n", (int) j, self->parent[j], self->likelihood[j]);
    }
    fprintf(out, "traceback\n");
    for (j = 0; j < self->num_sites; j++) {
        if (self->traceback[j] != NULL) {
            fprintf(out, "\t%d\t", (int) j);
            for (l = self->traceback[j]; l != NULL; l = l->next) {
                fprintf(out, "(%d, %f)", l->node, l->likelihood);
            }
            fprintf(out, "\n");
        }
    }
    object_heap_print_state(&self->avl_node_heap, out);
    block_allocator_print_state(&self->likelihood_list_allocator, out);

    tree_sequence_builder_check_state(self);
    return 0;
}

int
tree_sequence_builder_alloc(tree_sequence_builder_t *self, size_t num_sites,
        size_t max_nodes, size_t max_edges)
{
    int ret = 0;
    size_t j;
    /* TODO fix this. */
    size_t avl_node_block_size = 8192;
    size_t likelihood_list_block_size = 8192;

    /* TODO put in a check on the number of sites. We currently use an integer
     * in the tree transition algorithm, so the max value of this and the
     * max value of site_id_t are the practical limits. Probably simpler make
     * site_id_t a signed integer in the long run */

    memset(self, 0, sizeof(tree_sequence_builder_t));
    self->recombination_rate = 1e-8;
    self->num_sites = num_sites;
    self->max_nodes = max_nodes;
    self->max_edges = max_edges;
    self->max_output_edges = num_sites; /* We can probably make this smaller */
    self->num_nodes = 0;
    self->num_edges = 0;

    self->edges = malloc(self->max_edges * sizeof(edge_t));
    self->sort_buffer = malloc(self->max_edges * sizeof(index_sort_t));
    self->insertion_order = malloc(self->max_edges * sizeof(node_id_t));
    self->removal_order = malloc(self->max_edges * sizeof(node_id_t));
    self->time = malloc(self->max_nodes * sizeof(double));
    self->parent = malloc(self->max_nodes * sizeof(node_id_t));
    self->likelihood = malloc(self->max_nodes * sizeof(double));
    self->traceback = calloc(self->num_sites, sizeof(likelihood_list_t *));
    self->mutations = malloc(self->num_sites * sizeof(node_id_t));
    self->output_edge_buffer = malloc(self->max_output_edges * sizeof(edge_t));

    if (self->edges == NULL || self->time == NULL || self->parent == NULL
            || self->insertion_order == NULL || self->removal_order == NULL
            || self->sort_buffer == NULL || self->likelihood == NULL
            || self->traceback == NULL || self->output_edge_buffer == NULL
            || self->mutations == NULL)  {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    /* The AVL node heap stores the avl node and the node_id_t payload in
     * adjacent memory. */
    ret = object_heap_init(&self->avl_node_heap,
            sizeof(avl_node_t) + sizeof(node_id_t), avl_node_block_size, NULL);
    if (ret != 0) {
        goto out;
    }
    avl_init_tree(&self->likelihood_nodes, cmp_node_id, NULL);
    ret = block_allocator_alloc(&self->likelihood_list_allocator,
            likelihood_list_block_size);
    if (ret != 0) {
        goto out;
    }
    /* Initialise the likelihoods and tree. */
    for (j = 0; j < self->max_nodes; j++) {
        self->likelihood[j] = NULL_LIKELIHOOD;
        self->parent[j] = NULL_NODE;
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
    tsi_safe_free(self->parent);
    tsi_safe_free(self->insertion_order);
    tsi_safe_free(self->removal_order);
    tsi_safe_free(self->sort_buffer);
    tsi_safe_free(self->likelihood);
    tsi_safe_free(self->traceback);
    tsi_safe_free(self->output_edge_buffer);
    tsi_safe_free(self->mutations);
    object_heap_free(&self->avl_node_heap);
    block_allocator_free(&self->likelihood_list_allocator);
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
tree_sequence_builder_update(tree_sequence_builder_t *self, size_t num_nodes,
        double time, size_t num_edges, edge_t *edges, size_t num_site_mutations,
        site_mutation_t *site_mutations)
{
    int ret = 0;
    size_t j;

    assert(self->num_nodes + num_nodes < self->max_nodes);
    for (j = 0; j < num_nodes; j++) {
        self->time[self->num_nodes + j] = time;
    }
    self->num_nodes += num_nodes;
    /* We assume that the edges are given in reverse order, so we insert them this
     * way around to get closer to sortedness */
    for (j = 0; j < num_edges; j++) {
        assert(self->num_edges < self->max_edges);
        self->edges[self->num_edges] = edges[num_edges - j - 1];
        self->num_edges++;
    }
    ret = tree_sequence_builder_index_edges(self);
    if (ret != 0) {
        goto out;
    }
    for (j = 0; j < num_site_mutations; j++) {
        assert(site_mutations[j].node < (node_id_t) self->num_nodes);
        assert(site_mutations[j].node >= 0);
        assert(site_mutations[j].site < self->num_sites);
        self->mutations[site_mutations[j].site] = site_mutations[j].node;
    }
    self->num_mutations += num_site_mutations;
    /* tree_sequence_builder_print_state(self, stdout); */
out:
    return ret;
}

static inline void
tree_sequence_builder_free_avl_node(tree_sequence_builder_t *self, avl_node_t *node)
{
    object_heap_free_object(&self->avl_node_heap, node);
}

static inline avl_node_t * WARN_UNUSED
tree_sequence_builder_alloc_avl_node(tree_sequence_builder_t *self, node_id_t node)
{
    avl_node_t *ret = NULL;
    node_id_t *payload;

    if (object_heap_empty(&self->avl_node_heap)) {
        if (object_heap_expand(&self->avl_node_heap) != 0) {
            goto out;
        }
    }
    ret = (avl_node_t *) object_heap_alloc_object(&self->avl_node_heap);
    if (ret == NULL) {
        goto out;
    }
    /* We store the node_id_t value after the avl_node */
    payload = (node_id_t *) (ret + 1);
    *payload = node;
    avl_init_node(ret, payload);
out:
    return ret;
}

static int
tree_sequence_builder_insert_likelihood(tree_sequence_builder_t *self, node_id_t node,
        double likelihood)
{
    int ret = 0;
    avl_node_t *avl_node;

    assert(likelihood >= 0);
    avl_node = tree_sequence_builder_alloc_avl_node(self, node);
    if (avl_node == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    avl_node = avl_insert_node(&self->likelihood_nodes, avl_node);
    assert(self->likelihood[node] == NULL_LIKELIHOOD);
    assert(avl_node != NULL);
    self->likelihood[node] = likelihood;
out:
    return ret;
}

static int
tree_sequence_builder_delete_likelihood(tree_sequence_builder_t *self, node_id_t node)
{
    avl_node_t *avl_node;

    avl_node = avl_search(&self->likelihood_nodes, &node);
    assert(self->likelihood[node] != NULL_LIKELIHOOD);
    assert(avl_node != NULL);
    avl_unlink_node(&self->likelihood_nodes, avl_node);
    tree_sequence_builder_free_avl_node(self, avl_node);
    self->likelihood[node] = NULL_LIKELIHOOD;
    return 0;
}

/* Store the current state of the likelihood tree in the traceback.
 */
static int WARN_UNUSED
tree_sequence_builder_store_traceback(tree_sequence_builder_t *self, site_id_t site_id)
{
    int ret = 0;
    avl_node_t *a;
    node_id_t u;
    likelihood_list_t *list_node;

    for (a = self->likelihood_nodes.head; a != NULL; a = a->next) {
        u = *((node_id_t *) a->item);
        list_node = block_allocator_get(&self->likelihood_list_allocator,
                sizeof(likelihood_list_t));
        if (list_node == NULL) {
            ret = TSI_ERR_NO_MEMORY;
            goto out;
        }
        list_node->node = u;
        list_node->likelihood = self->likelihood[u];
        list_node->next = self->traceback[site_id];
        self->traceback[site_id] = list_node;
    }
    self->total_traceback_size += avl_count(&self->likelihood_nodes);
out:
    return ret;
}

/* Returns true if the node u is a descendant of v; i.e. if v is present on the
 * path from u to root. Returns false in all other situations, including
 * error conditions. */
static inline bool
tree_sequence_builder_is_descendant(tree_sequence_builder_t *self, node_id_t u,
        node_id_t v)
{
    node_id_t *pi = self->parent;

    while (u != NULL_NODE && u != v) {
        u = pi[u];
    }
    return u == v;
}

static int WARN_UNUSED
tree_sequence_builder_update_site_likelihood_values(tree_sequence_builder_t *self,
        node_id_t mutation_node, char state)
{
    int ret = 0;
    double n = (double) self->num_nodes;
    double r = 1 - exp(-self->recombination_rate / n);
    double recomb_proba = r / n;
    double no_recomb_proba = 1 - r + r / n;
    double *L = self->likelihood;
    double x, y, max_L, emission;
    bool is_descendant;
    node_id_t u;
    avl_node_t *a;

    max_L = -1;
    for (a = self->likelihood_nodes.head; a != NULL; a = a->next) {
        u = *((node_id_t *) a->item);
        x = L[u] * no_recomb_proba;
        assert(x >= 0);
        if (x > recomb_proba) {
            y = x;
        } else {
            y = recomb_proba;
        }
        is_descendant = tree_sequence_builder_is_descendant(self, u, mutation_node);
        if (state == 1) {
            emission = (double) is_descendant;
        } else {
            emission = (double) (! is_descendant);
        }
        L[u] = y * emission;
        if (L[u] > max_L) {
            max_L = L[u];
        }
        /* printf("mutation_node = %d u = %d, x = %f, y = %f, emission = %f\n", */
        /*         mutation_node, u, x, y, emission); */
    }
    assert(max_L > 0);
    /* Normalise */
    for (a = self->likelihood_nodes.head; a != NULL; a = a->next) {
        u = *((node_id_t *) a->item);
        L[u] /= max_L;
    }
    return ret;
}

static int WARN_UNUSED
tree_sequence_builder_coalesce_likelihoods(tree_sequence_builder_t *self)
{
    int ret = 0;
    avl_node_t *a, *tmp;
    node_id_t u, v;

    a = self->likelihood_nodes.head;
    while (a != NULL) {
        tmp = a->next;
        u = *((node_id_t *) a->item);
        if (self->parent[u] != NULL_NODE) {
            /* If we can find an equal L value higher in the tree, delete
             * this one.
             */
            v = self->parent[u];
            while (self->likelihood[v] == NULL_LIKELIHOOD) {
                v = self->parent[v];
                assert(v != NULL_NODE);
            }
            if (self->likelihood[u] == self->likelihood[v]) {
                /* Delete this likelihood value */
                avl_unlink_node(&self->likelihood_nodes, a);
                tree_sequence_builder_free_avl_node(self, a);
                self->likelihood[u] = NULL_LIKELIHOOD;
            }
        }
        a = tmp;
    }
    return ret;
}

static int
tree_sequence_builder_update_site_state(tree_sequence_builder_t *self, site_id_t site,
        allele_t state)
{
    int ret = 0;
    node_id_t mutation_node = self->mutations[site];
    node_id_t *pi = self->parent;
    double *L = self->likelihood;
    node_id_t u;

    /* tree_sequence_builder_print_state(self, stdout); */
    /* tree_sequence_builder_check_state(self); */
    if (mutation_node == NULL_NODE) {
        /* TODO We should be able to just put a pointer in to the previous site
         * here to save some time and memory. */
        ret = tree_sequence_builder_store_traceback(self, site);
        if (ret != 0) {
            goto out;
        }
        assert(state == 0);
    } else {
        /* Insert a new L-value for the mutation node if needed */
        if (L[mutation_node] == NULL_LIKELIHOOD) {
            u = mutation_node;
            while (L[u] == NULL_LIKELIHOOD) {
                u = pi[u];
                assert(u != NULL_NODE);
            }
            ret = tree_sequence_builder_insert_likelihood(self, mutation_node, L[u]);
            if (ret != 0) {
                goto out;
            }
        }
        ret = tree_sequence_builder_store_traceback(self, site);
        if (ret != 0) {
            goto out;
        }
        ret = tree_sequence_builder_update_site_likelihood_values(self, mutation_node, state);
        if (ret != 0) {
            goto out;
        }
        ret = tree_sequence_builder_coalesce_likelihoods(self);
        if (ret != 0) {
            goto out;
        }
    }
out:
    return ret;
}

static int
tree_sequence_builder_reset(tree_sequence_builder_t *self)
{
    int ret = 0;
    size_t j;

    assert(avl_count(&self->likelihood_nodes) == 0);
    for (j = 0; j < self->num_nodes; j++) {
        ret = tree_sequence_builder_insert_likelihood(self, (node_id_t) j, 1.0);
        if (ret != 0) {
            goto out;
        }
    }
    memset(self->traceback, 0, self->num_sites * sizeof(likelihood_list_t *));
    ret = block_allocator_reset(&self->likelihood_list_allocator);
    if (ret != 0) {
        goto out;
    }
out:
    return ret;
}


static int WARN_UNUSED
tree_sequence_builder_run_traceback(tree_sequence_builder_t *self,
       node_id_t child, size_t *num_output_edges, edge_t **output_edges)
{
    int ret = 0;
    int M = self->num_edges;
    int j, k, l;
    size_t output_edge_index;
    node_id_t *pi = self->parent;
    double *L = self->likelihood;
    edge_t *edges = self->edges;
    edge_t *output_edge;
    node_id_t *I, *O, u, max_likelihood_node;
    site_id_t left, right;
    avl_node_t *a, *tmp;
    likelihood_list_t *z;

    /* Prepare for the traceback and get the memory ready for recording
     * the output edges. */
    output_edge_index = 0;
    output_edge = self->output_edge_buffer + output_edge_index;
    output_edge->right = self->num_sites;
    output_edge->parent = NULL_NODE;
    output_edge->child = child;

    /* Process the final likelihoods and find the maximum value. Reset
     * the likelihood values so that we can reused the buffer during
     * traceback. */
    a = self->likelihood_nodes.head;
    while (a != NULL) {
        tmp = a->next;
        u = *((node_id_t *) a->item);
        if (approximately_one(L[u])) {
            output_edge->parent = u;
        }
        L[u] = NULL_LIKELIHOOD;
        tree_sequence_builder_free_avl_node(self, a);
        a = tmp;
    }
    avl_clear_tree(&self->likelihood_nodes);
    assert(output_edge->parent != NULL_NODE);

    /* Now go through the trees in reverse and run the traceback */
    j = M - 1;
    k = M - 1;
    I = self->removal_order;
    O = self->insertion_order;
    while (j >= 0) {
        right = edges[I[j]].right;
        while (edges[O[k]].left == right) {
            pi[edges[O[k]].child] = -1;
            k--;
        }
        left = edges[O[k]].left;
        while (j >= 0 && edges[I[j]].right == right) {
            pi[edges[I[j]].child] = edges[I[j]].parent;
            j--;
        }
        /* # print("left = ", left, "right = ", right) */
        /* printf("left = %d right = %d\n", left, right); */
        for (l = right - 1; l >= (int) left; l--) {
            /* Reset the likelihood values for this locus from the traceback */
            max_likelihood_node = NULL_NODE;
            for (z = self->traceback[l]; z != NULL; z = z->next) {
                L[z->node] = z->likelihood;
                if (approximately_one(z->likelihood)) {
                    max_likelihood_node = z->node;
                }
            }
            if (max_likelihood_node == NULL_NODE) {
                tree_sequence_builder_print_state(self, stdout);
            }
            assert(max_likelihood_node != NULL_NODE);
            u = output_edge->parent;
            /* Get the likelihood for u */
            while (L[u] == NULL_LIKELIHOOD) {
                u = pi[u];
                assert(u != NULL_NODE);
            }
            if (L[u] != 1.0) {
                /* printf("RECOMB: %d\n", l); */
                /* Need to recombine */
                output_edge->left = l;
                output_edge_index++;
                assert(output_edge_index < self->max_output_edges);
                output_edge++;
                /* Start the next output edge */
                output_edge->right = l;
                output_edge->child = child;
                output_edge->parent = max_likelihood_node;
            }
            /* Reset the likelihoods for the next site */
            for (z = self->traceback[l]; z != NULL; z = z->next) {
                L[z->node] = NULL_LIKELIHOOD;
            }
            /* tree_sequence_builder_check_state(self); */
        }
    }
    output_edge->left = 0;

    *num_output_edges = output_edge_index + 1;
    *output_edges = self->output_edge_buffer;
/* out: */
    return ret;
}

int
tree_sequence_builder_find_path(tree_sequence_builder_t *self, allele_t *haplotype,
       node_id_t node, size_t *num_output_edges, edge_t **output_edges)
{
    int ret = 0;
    int M = self->num_edges;
    int j, k, l;
    node_id_t *pi = self->parent;
    double *L = self->likelihood;
    edge_t *edges = self->edges;
    node_id_t *I, *O, u, parent, child;
    site_id_t left, right;

    ret = tree_sequence_builder_reset(self);
    if (ret != 0) {
        goto out;
    }

    j = 0;
    k = 0;
    I = self->insertion_order;
    O = self->removal_order;
    while (j < M) {
        left = edges[I[j]].left;
        while (edges[O[k]].right == left) {
            parent = edges[O[k]].parent;
            child = edges[O[k]].child;
            k++;
            pi[child] = -1;
            if (L[child] == NULL_LIKELIHOOD) {
                /* Traverse upwards until we find and L value for the child. */
                u = parent;
                while (L[u] == NULL_LIKELIHOOD) {
                    u = pi[u];
                    assert(u != NULL_NODE);
                }
                ret = tree_sequence_builder_insert_likelihood(self, child, L[u]);
                if (ret != 0) {
                    goto out;
                }
            }
        }
        right = edges[O[k]].right;
        while (j < M && edges[I[j]].left == left) {
            parent = edges[I[j]].parent;
            child = edges[I[j]].child;
            pi[child] = parent;
            j++;
            /* Traverse upwards until we find the L value for the parent. */
            u = parent;
            while (L[u] == NULL_LIKELIHOOD) {
                u = pi[u];
                assert(u != NULL_NODE);
            }
            assert(L[child] != NULL_LIKELIHOOD);
            /* if the child's L value is the same as the parent we can delete it */
            if (L[child] == L[u]) {
                tree_sequence_builder_delete_likelihood(self, child);
            }
        }
        /* printf("NEW TREE: %d-%d\n", left, right); */
        for (l = left; l < (int) right; l++) {
            /* printf("update site %d\n", l); */
            ret = tree_sequence_builder_update_site_state(self, l, haplotype[l]);
            if (ret != 0) {
                goto out;
            }
        }
    }
    ret =  tree_sequence_builder_run_traceback(self, node, num_output_edges,
            output_edges);
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


