#include "tsinfer.h"
#include "err.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#include <gsl/gsl_math.h>

/* TODO move this into a general utilities file. */
void
__tsi_safe_free(void **ptr) {
    if (ptr != NULL) {
        if (*ptr != NULL) {
            free(*ptr);
            *ptr = NULL;
        }
    }
}

static int
cmp_node_id(const void *a, const void *b) {
    const node_id_t *ia = (const node_id_t *) a;
    const node_id_t *ib = (const node_id_t *) b;
    return (*ia > *ib) - (*ia < *ib);
}

static bool
approximately_equal(double a, double b)
{
    double epsilon = 1e-9;
    bool ret = fabs(a - b) <= ( (fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * epsilon);
    return ret;
}

/* Returns true if x is approximately equal to one. */
static bool
approximately_one(double x)
{
    return approximately_equal(x, 1.0);
}

static inline bool
ancestor_matcher_is_nonzero_root(ancestor_matcher_t *self, node_id_t u)
{
    return u != 0 && self->parent[u] == NULL_NODE && self->left_child[u] == NULL_NODE;
}

static void
ancestor_matcher_check_state(ancestor_matcher_t *self)
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
        x = self->likelihood[u];
        assert(x >= 0);
    }
    /* Make sure that there are no other non null likelihoods in the array */
    num_likelihoods = 0;
    for (u = 0; u < (node_id_t) self->num_nodes; u++) {
        if (self->likelihood[u] >= 0) {
            num_likelihoods++;
        }
        if (ancestor_matcher_is_nonzero_root(self, u)) {
            assert(self->likelihood[u] == NONZERO_ROOT_LIKELIHOOD);
        } else {
            assert(self->likelihood[u] != NONZERO_ROOT_LIKELIHOOD);
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
ancestor_matcher_print_state(ancestor_matcher_t *self, FILE *out)
{
    size_t j;
    likelihood_list_t *l;
    avl_node_t *a;
    node_id_t u;

    fprintf(out, "Ancestor matcher state\n");
    tree_sequence_builder_print_state(self->tree_sequence_builder, out);
    fprintf(out, "tree = \n");
    fprintf(out, "id\tparent\tlchild\trchild\tlsib\trsib\tlikelihood\n");
    for (j = 0; j < self->num_nodes; j++) {
        fprintf(out, "%d\t%d\t%d\t%d\t%d\t%d\t%g\n", (int) j, self->parent[j],
                self->left_child[j], self->right_child[j], self->left_sib[j],
                self->right_sib[j], self->likelihood[j]);
    }
    fprintf(out, "likelihood nodes\n");
    /* Check the properties of the likelihood map */
    for (a = self->likelihood_nodes.head; a != NULL; a = a->next) {
        u = *((node_id_t *) a->item);
        printf("%d -> %g\n", u, self->likelihood[u]);
    }

    fprintf(out, "traceback\n");
    for (j = 0; j < self->num_sites; j++) {
        if (self->traceback[j] != NULL) {
            fprintf(out, "\t%d\t", (int) j);
            for (l = self->traceback[j]; l != NULL; l = l->next) {
                fprintf(out, "(%d, %g)", l->node, l->likelihood);
            }
            fprintf(out, "\n");
        }
    }
    object_heap_print_state(&self->avl_node_heap, out);
    block_allocator_print_state(&self->likelihood_list_allocator, out);

    ancestor_matcher_check_state(self);
    return 0;
}

int
ancestor_matcher_alloc(ancestor_matcher_t *self,
        tree_sequence_builder_t *tree_sequence_builder, double observation_error)
{
    int ret = 0;
    size_t j;
    /* TODO make these input parameters. */
    size_t avl_node_block_size = 8192;
    size_t likelihood_list_block_size = 64 * 1024 * 1024;

    memset(self, 0, sizeof(ancestor_matcher_t));
    self->tree_sequence_builder = tree_sequence_builder;
    self->observation_error = observation_error;
    self->num_sites = tree_sequence_builder->num_sites;
    self->output.max_size = self->num_sites; /* We can probably make this smaller */
    self->max_num_mismatches = self->num_sites; /* Ditto here */
    self->max_nodes = tree_sequence_builder->max_nodes;
    self->parent = malloc(self->max_nodes * sizeof(node_id_t));
    self->left_child = malloc(self->max_nodes * sizeof(node_id_t));
    self->right_child = malloc(self->max_nodes * sizeof(node_id_t));
    self->left_sib = malloc(self->max_nodes * sizeof(node_id_t));
    self->right_sib = malloc(self->max_nodes * sizeof(node_id_t));
    self->likelihood = malloc(self->max_nodes * sizeof(double));
    self->traceback = calloc(self->num_sites, sizeof(likelihood_list_t *));
    self->output.left = malloc(self->output.max_size * sizeof(site_id_t));
    self->output.right = malloc(self->output.max_size * sizeof(site_id_t));
    self->output.parent = malloc(self->output.max_size * sizeof(node_id_t));
    self->mismatches = malloc(self->max_num_mismatches * sizeof(site_id_t));
    if (self->parent == NULL
            || self->left_child == NULL || self->right_child == NULL
            || self->left_sib == NULL || self->right_sib == NULL
            || self->likelihood == NULL || self->traceback == NULL
            || self->output.left == NULL || self->output.right == NULL
            || self->output.parent == NULL || self->mismatches == NULL) {
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
out:
    return ret;
}

int
ancestor_matcher_free(ancestor_matcher_t *self)
{
    tsi_safe_free(self->parent);
    tsi_safe_free(self->left_child);
    tsi_safe_free(self->right_child);
    tsi_safe_free(self->left_sib);
    tsi_safe_free(self->right_sib);
    tsi_safe_free(self->likelihood);
    tsi_safe_free(self->traceback);
    tsi_safe_free(self->output.left);
    tsi_safe_free(self->output.right);
    tsi_safe_free(self->output.parent);
    tsi_safe_free(self->mismatches);
    object_heap_free(&self->avl_node_heap);
    block_allocator_free(&self->likelihood_list_allocator);
    return 0;
}

static inline void
ancestor_matcher_free_avl_node(ancestor_matcher_t *self, avl_node_t *node)
{
    object_heap_free_object(&self->avl_node_heap, node);
}

static inline avl_node_t * WARN_UNUSED
ancestor_matcher_alloc_avl_node(ancestor_matcher_t *self, node_id_t node)
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
ancestor_matcher_insert_likelihood(ancestor_matcher_t *self, node_id_t node,
        double likelihood)
{
    int ret = 0;
    avl_node_t *avl_node;

    assert(likelihood >= 0);
    avl_node = ancestor_matcher_alloc_avl_node(self, node);
    if (avl_node == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    avl_node = avl_insert_node(&self->likelihood_nodes, avl_node);
    assert(self->likelihood[node] == NULL_LIKELIHOOD
            || self->likelihood[node] == NONZERO_ROOT_LIKELIHOOD);
    assert(avl_node != NULL);
    self->likelihood[node] = likelihood;
out:
    return ret;
}

static int
ancestor_matcher_delete_likelihood(ancestor_matcher_t *self, node_id_t node)
{
    avl_node_t *avl_node;

    avl_node = avl_search(&self->likelihood_nodes, &node);
    assert(self->likelihood[node] != NULL_LIKELIHOOD);
    assert(avl_node != NULL);
    avl_unlink_node(&self->likelihood_nodes, avl_node);
    ancestor_matcher_free_avl_node(self, avl_node);
    self->likelihood[node] = NULL_LIKELIHOOD;
    return 0;
}

/* Store the current state of the likelihood tree in the traceback.
 */
static int WARN_UNUSED
ancestor_matcher_store_traceback(ancestor_matcher_t *self, site_id_t site_id)
{
    int ret = 0;
    avl_node_t *restrict a;
    node_id_t u;
    likelihood_list_t *restrict list_node;
    double *restrict L = self->likelihood;
    likelihood_list_t **restrict T = self->traceback;
    bool match, loop_completed;

    /* Check to see if the previous site has the same likelihoods. If so,
     * we can reuse the same list. */
    match = false;
    if (site_id > 0) {
        loop_completed = true;
        list_node = T[site_id - 1];
        for (a = self->likelihood_nodes.tail; a != NULL; a = a->prev) {
            u = *((node_id_t *) a->item);
            if (list_node == NULL || list_node->node != u || list_node->likelihood != L[u]) {
                loop_completed = false;
                break;
            }
            list_node = list_node->next;
        }
        match = loop_completed && list_node == NULL;
    }
    if (match) {
        T[site_id] = T[site_id - 1];
    } else {
        /* Allocate the entire list at once to save some overhead */
        list_node = block_allocator_get(&self->likelihood_list_allocator,
                avl_count(&self->likelihood_nodes) * sizeof(likelihood_list_t));
        if (list_node == NULL) {
            ret = TSI_ERR_NO_MEMORY;
            goto out;
        }
        for (a = self->likelihood_nodes.head; a != NULL; a = a->next) {
            u = *((node_id_t *) a->item);
            list_node->node = u;
            list_node->likelihood = L[u];
            list_node->next = T[site_id];
            T[site_id] = list_node;
            list_node++;
        }
    }
    self->total_traceback_size += avl_count(&self->likelihood_nodes);
out:
    return ret;
}

/* Returns true if the node u is a descendant of v; i.e. if v is present on the
 * path from u to root. Returns false in all other situations, including
 * error conditions. */
static inline bool
ancestor_matcher_is_descendant(ancestor_matcher_t *self, node_id_t u,
        node_id_t v)
{
    node_id_t *restrict pi = self->parent;

    while (u != NULL_NODE && u != v) {
        u = pi[u];
    }
    return u == v;
}

static int WARN_UNUSED
ancestor_matcher_update_site_likelihood_values(ancestor_matcher_t *self, site_id_t site,
        node_id_t mutation_node, char state)
{
    int ret = 0;
    double n = (double) self->num_nodes;
    double rho = self->tree_sequence_builder->sites.recombination_rate[site];
    double r = 1 - exp(-rho / n);
    double err = self->observation_error;
    double recomb_proba = r / n;
    double no_recomb_proba = 1 - r + r / n;
    double *restrict L = self->likelihood;
    double x, y, max_L, emission;
    bool is_descendant;
    node_id_t u;
    avl_node_t *a;
    double distance = 1;

    if (site > 0) {
        distance = self->tree_sequence_builder->sites.position[site] -
                self->tree_sequence_builder->sites.position[site - 1];
    }
    recomb_proba *= distance;
    no_recomb_proba *= distance;

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
        is_descendant = false;
        if (mutation_node != NULL_NODE) {
            is_descendant = ancestor_matcher_is_descendant(self, u, mutation_node);
        }
        if (state == 1) {
            emission = (1 - err) * is_descendant + err * (! is_descendant);
        } else {
            emission = err * is_descendant + (1 - err) * (! is_descendant);
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
ancestor_matcher_coalesce_likelihoods(ancestor_matcher_t *self)
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
            if (approximately_equal(self->likelihood[u], self->likelihood[v])) {
                /* Delete this likelihood value */
                avl_unlink_node(&self->likelihood_nodes, a);
                ancestor_matcher_free_avl_node(self, a);
                self->likelihood[u] = NULL_LIKELIHOOD;
            }
        }
        a = tmp;
    }
    return ret;
}

static int
ancestor_matcher_update_site_state(ancestor_matcher_t *self, site_id_t site,
        allele_t state)
{
    int ret = 0;
    node_id_t mutation_node = NULL_NODE;
    node_id_t *pi = self->parent;
    double *L = self->likelihood;
    node_id_t u;

    if (self->tree_sequence_builder->sites.mutations[site] != NULL) {
        mutation_node = self->tree_sequence_builder->sites.mutations[site]->node;
    }
    /* ancestor_matcher_print_state(self, stdout); */
    /* ancestor_matcher_check_state(self); */
    /* FIXME: disabling this optimisation for now */
    if (false) {
    /* if (self->observation_error == 0.0 */
    /*             && mutation_node == NULL_NODE */
    /*             && site > 0 */
    /*             && self->tree_sequence_builder->sites.mutations[site - 1] == NULL) { */
        /* If there are no mutations at this or the last site, then
         * we are guaranteed that the likelihoods are equal. */
        self->traceback[site] = self->traceback[site - 1];
        self->total_traceback_size += avl_count(&self->likelihood_nodes);
    } else {
        if (mutation_node != NULL_NODE) {
            /* Insert a new L-value for the mutation node if needed */
            if (L[mutation_node] == NULL_LIKELIHOOD) {
                u = mutation_node;
                while (L[u] == NULL_LIKELIHOOD) {
                    u = pi[u];
                    assert(u != NULL_NODE);
                }
                ret = ancestor_matcher_insert_likelihood(self, mutation_node, L[u]);
                if (ret != 0) {
                    goto out;
                }
            }
        }
        ret = ancestor_matcher_store_traceback(self, site);
        if (ret != 0) {
            goto out;
        }
        if (self->observation_error > 0 || mutation_node != NULL_NODE) {
            ret = ancestor_matcher_update_site_likelihood_values(self, site,
                    mutation_node, state);
            if (ret != 0) {
                goto out;
            }
            ret = ancestor_matcher_coalesce_likelihoods(self);
            if (ret != 0) {
                goto out;
            }
        }
    }
out:
    return ret;
}

static void
ancestor_matcher_reset_tree(ancestor_matcher_t *self)
{
    memset(self->parent, 0xff, self->num_nodes * sizeof(node_id_t));
    memset(self->left_child, 0xff, self->num_nodes * sizeof(node_id_t));
    memset(self->right_child, 0xff, self->num_nodes * sizeof(node_id_t));
    memset(self->left_sib, 0xff, self->num_nodes * sizeof(node_id_t));
    memset(self->right_sib, 0xff, self->num_nodes * sizeof(node_id_t));
}

static int
ancestor_matcher_reset(ancestor_matcher_t *self)
{
    int ret = 0;

    /* TODO realloc when this grows */
    assert(self->max_nodes == self->tree_sequence_builder->max_nodes);
    self->num_nodes = self->tree_sequence_builder->num_nodes;

    assert(avl_count(&self->likelihood_nodes) == 0);
    memset(self->traceback, 0, self->num_sites * sizeof(likelihood_list_t *));
    ret = block_allocator_reset(&self->likelihood_list_allocator);
    if (ret != 0) {
        goto out;
    }
    self->total_traceback_size = 0;
    ancestor_matcher_reset_tree(self);
out:
    return ret;
}

static int WARN_UNUSED
ancestor_matcher_run_traceback(ancestor_matcher_t *self, site_id_t start,
        site_id_t end, allele_t *haplotype, allele_t *match)
{
    int ret = 0;
    int M = self->tree_sequence_builder->num_edges;
    int j, k, l;
    node_id_t *restrict pi = self->parent;
    double *restrict L = self->likelihood;
    edge_t *edges = self->tree_sequence_builder->edges;
    node_id_t *I, *O, u, max_likelihood_node;
    site_id_t left, right, pos;
    avl_node_t *a, *tmp;
    likelihood_list_t *z;
    mutation_list_node_t *mut_list;
    bool is_descendant;

    /* Prepare for the traceback and get the memory ready for recording
     * the output edges. */
    self->output.size = 0;
    self->output.right[self->output.size] = end;
    self->output.parent[self->output.size] = NULL_NODE;

    /* Process the final likelihoods and find the maximum value. Reset
     * the likelihood values so that we can reused the buffer during
     * traceback. */
    a = self->likelihood_nodes.head;
    while (a != NULL) {
        tmp = a->next;
        u = *((node_id_t *) a->item);
        if (approximately_one(L[u])) {
            self->output.parent[self->output.size] = u;
        }
        L[u] = NULL_LIKELIHOOD;
        ancestor_matcher_free_avl_node(self, a);
        a = tmp;
    }
    avl_clear_tree(&self->likelihood_nodes);
    assert(self->output.parent[self->output.size] != NULL_NODE);

    /* Now go through the trees in reverse and run the traceback */
    memset(match, 0xff, self->num_sites * sizeof(allele_t));
    memset(pi, 0xff, self->num_nodes * sizeof(node_id_t));
    j = M - 1;
    k = M - 1;
    I = self->tree_sequence_builder->removal_order;
    O = self->tree_sequence_builder->insertion_order;
    pos = self->num_sites;

    while (pos > start) {
        while (k >= 0 && edges[O[k]].left == pos) {
            pi[edges[O[k]].child] = NULL_NODE;
            k--;
        }
        while (j >= 0 && edges[I[j]].right == pos) {
            pi[edges[I[j]].child] = edges[I[j]].parent;
            j--;
        }
        right = pos;
        left = 0;
        if (k >= 0) {
            left = GSL_MAX(left, edges[O[k]].left);
        }
        if (j >= 0) {
            left = GSL_MAX(left, edges[I[j]].right);
        }
        pos = left;

        assert(left < right);

        /* # print("left = ", left, "right = ", right) */
        /* printf("left = %d right = %d\n", left, right); */
        for (l = GSL_MIN(right, end) - 1; l >= (int) GSL_MAX(left, start); l--) {
            match[l] = 0;
            /* Reset the likelihood values for this locus from the traceback */
            max_likelihood_node = NULL_NODE;
            for (z = self->traceback[l]; z != NULL; z = z->next) {
                L[z->node] = z->likelihood;
                if (approximately_one(z->likelihood)) {
                    max_likelihood_node = z->node;
                }
            }
            assert(max_likelihood_node != NULL_NODE);
            u = self->output.parent[self->output.size];
            /* Set the state of the matched haplotype */
            mut_list = self->tree_sequence_builder->sites.mutations[l];
            if (mut_list != NULL) {
                is_descendant = ancestor_matcher_is_descendant(self, u, mut_list->node);
                if (is_descendant) {
                    match[l] = 1;
                }
            }
            /* Get the likelihood for u */
            while (L[u] == NULL_LIKELIHOOD) {
                u = pi[u];
                assert(u != NULL_NODE);
            }
            if (!approximately_one(L[u])) {
                /* printf("RECOMB: %d\n", l); */
                /* Need to recombine */
                assert(max_likelihood_node != self->output.parent[self->output.size]);

                self->output.left[self->output.size] = l;
                self->output.size++;
                assert(self->output.size < self->output.max_size);
                /* Start the next output edge */
                self->output.right[self->output.size] = l;
                self->output.parent[self->output.size] = max_likelihood_node;
            }
            /* Reset the likelihoods for the next site */
            for (z = self->traceback[l]; z != NULL; z = z->next) {
                L[z->node] = NULL_LIKELIHOOD;
            }
            /* ancestor_matcher_check_state(self); */
        }
    }
    self->output.left[self->output.size] = start;
    self->output.size++;
    /* Occassionally we will output a spurious edge. Simplest to detect it afterwards
     * and remove it. */
    if (self->output.right[self->output.size - 1] == start) {
        self->output.size--;
    }
    return ret;
}

static inline void
ancestor_matcher_remove_edge(ancestor_matcher_t *self, edge_t edge)
{
    node_id_t p = edge.parent;
    node_id_t c = edge.child;
    node_id_t lsib = self->left_sib[c];
    node_id_t rsib = self->right_sib[c];

    /* printf("REMOVE EDGE %d -> %d\n", edge.child, edge.parent); */
    if (lsib == NULL_NODE) {
        self->left_child[p] = rsib;
    } else {
        self->right_sib[lsib] = rsib;
    }
    if (rsib == NULL_NODE) {
        self->right_child[p] = lsib;
    } else {
        self->left_sib[rsib] = lsib;
    }
    self->parent[c] = NULL_NODE;
    self->left_sib[c] = NULL_NODE;
    self->right_sib[c] = NULL_NODE;
}

static inline void
ancestor_matcher_insert_edge(ancestor_matcher_t *self, edge_t edge)
{
    node_id_t p = edge.parent;
    node_id_t c = edge.child;
    node_id_t u = self->right_child[p];
    /* printf("INSERT EDGE %d -> %d\n", edge.child, edge.parent); */

    self->parent[c] = p;
    if (u == NULL_NODE) {
        self->left_child[p] = c;
        self->left_sib[c] = NULL_NODE;
        self->right_sib[c] =NULL_NODE;
    } else {
        self->right_sib[u] = c;
        self->left_sib[c] = u;
        self->right_sib[c] = NULL_NODE;
    }
    self->right_child[p] = c;
}

/* After we have removed a 1.0 valued node, we must renormalise the likelihoods */
static void
ancestor_matcher_renormalise_likelihoods(ancestor_matcher_t *self)
{
    avl_node_t *a;
    double max_L = -1;
    double *restrict L = self->likelihood;
    node_id_t u;

    for (a = self->likelihood_nodes.head; a != NULL; a = a->next) {
        u = *((node_id_t *) a->item);
        max_L = GSL_MAX(max_L,  L[u]);
    }
    assert(max_L > 0);
    for (a = self->likelihood_nodes.head; a != NULL; a = a->next) {
        u = *((node_id_t *) a->item);
        L[u] /= max_L;
    }
}

int
ancestor_matcher_find_path(ancestor_matcher_t *self,
        site_id_t start, site_id_t end, allele_t *haplotype,
        allele_t *matched_haplotype, size_t *num_output_edges,
        site_id_t **left_output, site_id_t **right_output, node_id_t **parent_output)
{
    int ret = 0;
    int M = self->tree_sequence_builder->num_edges;
    int j, k, l, remove_start;
    site_id_t site;
    edge_t edge;
    node_id_t u;
    /* Can't use restrict here for L because we access it in the functions called
     * from here. */
    double *L = self->likelihood;
    edge_t *edges = self->tree_sequence_builder->edges;
    node_id_t *I, *O;
    site_id_t pos, left, right;
    bool renormalise_required;

    ret = ancestor_matcher_reset(self);
    if (ret != 0) {
        goto out;
    }
    /* Load the tree for start */
    j = 0;
    k = 0;
    left = 0;
    pos = 0;
    right = self->num_sites;
    I = self->tree_sequence_builder->insertion_order;
    O = self->tree_sequence_builder->removal_order;

    while (j < M && k < M && edges[I[j]].left <= start) {
        while (k < M && edges[O[k]].right == pos) {
            ancestor_matcher_remove_edge(self, edges[O[k]]);
            k++;
        }
        while (j < M && edges[I[j]].left == pos) {
            ancestor_matcher_insert_edge(self, edges[I[j]]);
            j++;
        }
        left = pos;
        right = self->num_sites;
        if (j < M) {
            right = GSL_MIN(right, edges[I[j]].left);
        }
        if (k < M) {
            right = GSL_MIN(right, edges[O[k]].right);
        }
        pos = right;
    }

    /* Insert the initial likelihoods. Zero is the root, and it has likelihood
     * one. All non-zero roots are marked with a special value so we can
     * identify them when the enter the tree */
    ret = ancestor_matcher_insert_likelihood(self, 0, 1.0);
    if (ret != 0) {
        goto out;
    }
    for (u = 1; u < (node_id_t) self->num_nodes; u++) {
        if (self->parent[u] != NULL_NODE) {
            L[u] = NULL_LIKELIHOOD;
        } else {
            L[u] = NONZERO_ROOT_LIKELIHOOD;
        }
    }

    /* printf("initial tree %d-%d\n", left, right); */
    /* printf("j = %d k = %d\n", j, k); */
    /* ancestor_matcher_print_state(self, stdout); */
    /* ancestor_matcher_check_state(self); */

    remove_start = k;
    while (left < end) {
        assert(left < right);
        /* printf("NEW TREE %d-%d\n", left, right); */

        /* Remove the likelihoods for any nonzero roots that have just left
         * the tree */
        renormalise_required = false;
        for (l = remove_start; l < k; l++) {
            edge = edges[O[l]];
            if (ancestor_matcher_is_nonzero_root(self, edge.child)) {
                if (approximately_one(L[edge.child])) {
                    renormalise_required = true;
                }
                if (L[edge.child] != NULL_LIKELIHOOD) {
                    ancestor_matcher_delete_likelihood(self, edge.child);
                }
                L[edge.child] = NONZERO_ROOT_LIKELIHOOD;
            }
        }
        if (renormalise_required) {
            ancestor_matcher_renormalise_likelihoods(self);
        }
        /* ancestor_matcher_print_state(self, stdout); */
        /* ancestor_matcher_check_state(self); */
        for (site = GSL_MAX(left, start); site < GSL_MIN(right, end); site++) {
            ret = ancestor_matcher_update_site_state(self, site, haplotype[site]);
            if (ret != 0) {
                goto out;
            }
        }

        /* Move on to the next tree */
        remove_start = k;
        while (k < M  && edges[O[k]].right == right) {
            edge = edges[O[k]];
            ancestor_matcher_remove_edge(self, edge);
            k++;
/*             if (L[edge.child] == NULL_LIKELIHOOD) { */
/*                 /1* Traverse upwards until we find and L value for the child. *1/ */
/*                 u = edge.parent; */
/*                 while (L[u] == NULL_LIKELIHOOD) { */
/*                     u = self->parent[u]; */
/*                     /1* assert(u != NULL_NODE); *1/ */
/*                 } */
/*                 ret = ancestor_matcher_insert_likelihood(self, edge.child, L[u]); */
/*                 if (ret != 0) { */
/*                     goto out; */
/*                 } */
/*             } */
        }
        left = right;
        /* printf("Inserting for j = %d and left = %d (%d)\n", (int) j, (int) left, */
        /*         edges[I[j]].left); */
        while (j < M && edges[I[j]].left == left) {
            edge = edges[I[j]];
            ancestor_matcher_insert_edge(self, edge);
            j++;
            /* Insert zero likelihoods for any nonzero roots that have entered
             * the tree. Note we don't bother trying to compress the tree here
             * because this will be done for the next site anyway. */
            if (L[edge.parent] == NONZERO_ROOT_LIKELIHOOD) {
                ret = ancestor_matcher_insert_likelihood(self, edge.parent, 0);
                if (ret != 0) {
                    goto out;
                }
            }
            if (L[edge.child] == NONZERO_ROOT_LIKELIHOOD) {
                ret = ancestor_matcher_insert_likelihood(self, edge.child, 0);
                if (ret != 0) {
                    goto out;
                }
            }
        }
        right = self->num_sites;
        if (j < M) {
            right = GSL_MIN(right, edges[I[j]].left);
        }
        if (k < M) {
            right = GSL_MIN(right, edges[O[k]].right);
        }
    }

    ret = ancestor_matcher_run_traceback(self, start, end, haplotype,
            matched_haplotype);
    if (ret != 0) {
        goto out;
    }
    *left_output = self->output.left;
    *right_output = self->output.right;
    *parent_output = self->output.parent;
    *num_output_edges = self->output.size;
out:
    return ret;
}

double
ancestor_matcher_get_mean_traceback_size(ancestor_matcher_t *self)
{
    return self->total_traceback_size / ((double) self->num_sites);
}

size_t
ancestor_matcher_get_total_memory(ancestor_matcher_t *self)
{
    size_t total = self->likelihood_list_allocator.total_size;
    /* TODO add contributions from other objects */

    return total;
}
