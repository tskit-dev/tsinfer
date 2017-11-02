#include "tsinfer.h"
#include "err.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

static bool
approximately_equal(const double a, const double b)
{
    const double epsilon = 1e-9;
    bool ret = fabs(a - b) <= ( (fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * epsilon);
    return ret;
}

/* Returns true if x is approximately equal to one. */
static bool
approximately_one(const double x)
{
    return approximately_equal(x, 1.0);
}

static inline bool
is_nonzero_root(const node_id_t u, const node_id_t *restrict parent,
        const node_id_t *restrict left_child)
{
    return u != 0 && parent[u] == NULL_NODE && left_child[u] == NULL_NODE;
}

static void
ancestor_matcher_check_state(ancestor_matcher_t *self)
{
    int num_likelihoods;
    int j;
    node_id_t u;
    double x;
    likelihood_list_t *z;

    /* Check the properties of the likelihood map */
    for (j = 0; j < self->num_likelihood_nodes; j++) {
        u = self->likelihood_nodes[j];
        x = self->likelihood[u];
        assert(x >= 0);
    }
    /* Make sure that there are no other non null likelihoods in the array */
    num_likelihoods = 0;
    for (u = 0; u < (node_id_t) self->num_nodes; u++) {
        if (self->likelihood[u] >= 0) {
            num_likelihoods++;
        }
        if (is_nonzero_root(u, self->parent, self->left_child)) {
            assert(self->likelihood[u] == NONZERO_ROOT_LIKELIHOOD);
        } else {
            assert(self->likelihood[u] != NONZERO_ROOT_LIKELIHOOD);
        }
    }
    assert(num_likelihoods == self->num_likelihood_nodes);

    for (j = 0; j < (int) self->num_sites; j++) {
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
    int j;
    likelihood_list_t *l;
    node_id_t u;

    fprintf(out, "Ancestor matcher state\n");
    tree_sequence_builder_print_state(self->tree_sequence_builder, out);
    fprintf(out, "tree = \n");
    fprintf(out, "id\tparent\tlchild\trchild\tlsib\trsib\tlikelihood\n");
    for (j = 0; j < (int) self->num_nodes; j++) {
        fprintf(out, "%d\t%d\t%d\t%d\t%d\t%d\t%g\n", (int) j, self->parent[j],
                self->left_child[j], self->right_child[j], self->left_sib[j],
                self->right_sib[j], self->likelihood[j]);
    }
    fprintf(out, "likelihood nodes\n");
    /* Check the properties of the likelihood map */
    for (j = 0; j < self->num_likelihood_nodes; j++) {
        u = self->likelihood_nodes[j];
        printf("%d -> %g\n", u, self->likelihood[u]);
    }

    fprintf(out, "traceback\n");
    for (j = 0; j < (int) self->num_sites; j++) {
        if (self->traceback[j] != NULL) {
            fprintf(out, "\t%d\t", (int) j);
            for (l = self->traceback[j]; l != NULL; l = l->next) {
                fprintf(out, "(%d, %g)", l->node, l->likelihood);
            }
            fprintf(out, "\n");
        }
    }
    block_allocator_print_state(&self->likelihood_list_allocator, out);

    ancestor_matcher_check_state(self);
    return 0;
}

int
ancestor_matcher_alloc(ancestor_matcher_t *self,
        tree_sequence_builder_t *tree_sequence_builder, double observation_error)
{
    int ret = 0;
    /* TODO make these input parameters. */
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
    self->likelihood_cache = malloc(self->max_nodes * sizeof(double));
    self->likelihood_nodes = malloc(self->max_nodes * sizeof(node_id_t));
    self->likelihood_nodes_tmp = malloc(self->max_nodes * sizeof(node_id_t));
    self->path_cache = malloc(self->max_nodes * sizeof(int8_t));
    self->traceback = calloc(self->num_sites, sizeof(likelihood_list_t *));
    self->output.left = malloc(self->output.max_size * sizeof(site_id_t));
    self->output.right = malloc(self->output.max_size * sizeof(site_id_t));
    self->output.parent = malloc(self->output.max_size * sizeof(node_id_t));
    self->mismatches = malloc(self->max_num_mismatches * sizeof(site_id_t));
    if (self->parent == NULL
            || self->left_child == NULL || self->right_child == NULL
            || self->left_sib == NULL || self->right_sib == NULL
            || self->likelihood == NULL || self->likelihood_cache == NULL
            || self->likelihood_nodes == NULL
            || self->likelihood_nodes_tmp == NULL || self->traceback == NULL
            || self->output.left == NULL || self->output.right == NULL
            || self->output.parent == NULL || self->mismatches == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    ret = block_allocator_alloc(&self->likelihood_list_allocator,
            likelihood_list_block_size);
    if (ret != 0) {
        goto out;
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
    tsi_safe_free(self->likelihood_cache);
    tsi_safe_free(self->likelihood_nodes);
    tsi_safe_free(self->likelihood_nodes_tmp);
    tsi_safe_free(self->path_cache);
    tsi_safe_free(self->traceback);
    tsi_safe_free(self->output.left);
    tsi_safe_free(self->output.right);
    tsi_safe_free(self->output.parent);
    tsi_safe_free(self->mismatches);
    block_allocator_free(&self->likelihood_list_allocator);
    return 0;
}

static int
ancestor_matcher_delete_likelihood(ancestor_matcher_t *self, const node_id_t node,
        double *restrict L)
{
    /* Remove the specified node from the list of nodes */
    int j, k;
    node_id_t *restrict L_nodes = self->likelihood_nodes;

    k = 0;
    for (j = 0; j < self->num_likelihood_nodes; j++) {
        L_nodes[k] = L_nodes[j];
        if (L_nodes[j] != node) {
            k++;
        }
    }
    assert(self->num_likelihood_nodes == k + 1);
    self->num_likelihood_nodes = k;
    L[node] = NULL_LIKELIHOOD;
    return 0;
}

/* Store the current state of the likelihood tree in the traceback.
 */
static int WARN_UNUSED
ancestor_matcher_store_traceback(ancestor_matcher_t *self, const site_id_t site_id,
        const double *restrict L)
{
    int ret = 0;
    node_id_t u;
    int j;
    likelihood_list_t *restrict list_node;
    likelihood_list_t **restrict T = self->traceback;
    bool match, loop_completed;

    /* Check to see if the previous site has the same likelihoods. If so,
     * we can reuse the same list. */
    match = false;
    if (site_id > 0) {
        loop_completed = true;
        list_node = T[site_id - 1];
        for (j = self->num_likelihood_nodes - 1; j >= 0; j--) {
            u = self->likelihood_nodes[j];
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
                self->num_likelihood_nodes * sizeof(likelihood_list_t));
        if (list_node == NULL) {
            ret = TSI_ERR_NO_MEMORY;
            goto out;
        }
        for (j = 0; j < self->num_likelihood_nodes; j++) {
            u = self->likelihood_nodes[j];
            list_node->node = u;
            list_node->likelihood = L[u];
            list_node->next = T[site_id];
            T[site_id] = list_node;
            list_node++;
        }
    }
    self->total_traceback_size += self->num_likelihood_nodes;
out:
    return ret;
}

/* Returns true if the node u is a descendant of v; i.e. if v is present on the
 * path from u to root. Returns false in all other situations, including
 * error conditions. */
static inline bool
is_descendant(const node_id_t u, const node_id_t v, const node_id_t *restrict parent)
{
    bool ret = false;
    node_id_t w = u;

    if (v != NULL_NODE) {
        /* Because we allocate node IDs in nondecreasing order forwards in time,
         * if the node ID is of u is less than v it cannot be a descendant of v */
        if (u >= v) {
            while (w != NULL_NODE && w != v) {
                w = parent[w];
            }
            ret = w == v;
        }
    }
    return ret;
}

static int WARN_UNUSED
ancestor_matcher_update_site_likelihood_values(ancestor_matcher_t *self,
        const site_id_t site, const node_id_t mutation_node, const char state,
        const node_id_t *restrict parent, double *restrict L)
{
    int ret = 0;
    const double n = (double) self->num_nodes;
    const double rho = self->tree_sequence_builder->sites.recombination_rate[site];
    const double r = 1 - exp(-rho / n);
    const double err = self->observation_error;
    const int num_likelihood_nodes = self->num_likelihood_nodes;
    const node_id_t *restrict L_nodes = self->likelihood_nodes;
    double recomb_proba = r / n;
    double no_recomb_proba = 1 - r + r / n;
    double x, y, max_L, max_L_inv, emission;
    int8_t *restrict path_cache = self->path_cache;
    int j;
    bool descendant;
    node_id_t u, v;
    double distance = 1;

    if (site > 0) {
        distance = self->tree_sequence_builder->sites.position[site] -
                self->tree_sequence_builder->sites.position[site - 1];
    }
    recomb_proba *= distance;
    no_recomb_proba *= distance;

    max_L = -1;
    /* printf("likelihoods for node=%d, n=%d\n", mutation_node, self->num_likelihood_nodes); */
    for (j = 0; j < num_likelihood_nodes; j++) {
        u = L_nodes[j];
        /* Determine if the node this likelihood is associated with is a descendant
         * of the mutation node. To avoid the cost of repeatedly traversing up the
         * tree, we keep a cache of the paths that we have already traversed. When
         * we meet one of these paths we can immediately finish.
         */
        descendant = false;
        if (mutation_node != NULL_NODE) {
            v = u;
            while (likely(v != NULL_NODE)
                    && likely(v != mutation_node)
                    && likely(path_cache[v] == CACHE_UNSET)) {
                v = parent[v];
            }
            if (likely(v != NULL_NODE) && likely(path_cache[v] != CACHE_UNSET)) {
                descendant = (bool) path_cache[v];
            } else {
                descendant = v == mutation_node;
            }
            /* Insert this path into the cache */
            v = u;
            while (likely(v != NULL_NODE)
                    && likely(v != mutation_node)
                    && likely(path_cache[v] == CACHE_UNSET)) {
                path_cache[v] = descendant;
                v = parent[v];
            }
        }

        /* assert(descendant == is_descendant(u, mutation_node, parent)); */

        x = L[u] * no_recomb_proba;
        assert(x >= 0);
        if (x > recomb_proba) {
            y = x;
        } else {
            y = recomb_proba;
        }
        if (state == 1) {
            emission = (1 - err) * descendant + err * (! descendant);
        } else {
            emission = err * descendant + (1 - err) * (! descendant);
        }
        L[u] = y * emission;
        if (L[u] > max_L) {
            max_L = L[u];
        }
        /* printf("mutation_node = %d u = %d, x = %f, y = %f, emission = %f\n", */
        /*         mutation_node, u, x, y, emission); */
    }
    assert(max_L > 0);
    max_L_inv = 1.0 / max_L;
    /* Normalise and reset the path cache. */
    for (j = 0; j < num_likelihood_nodes; j++) {
        u = L_nodes[j];
        L[u] *= max_L_inv;
        v = u;
        while (likely(v != NULL_NODE) && likely(path_cache[v] != CACHE_UNSET)) {
            path_cache[v] = CACHE_UNSET;
            v = parent[v];
        }
    }
    return ret;
}

static int WARN_UNUSED
ancestor_matcher_coalesce_likelihoods(ancestor_matcher_t *self,
        const node_id_t *restrict parent, double *restrict L, double *restrict L_cache)
{
    int ret = 0;
    double L_p;
    node_id_t u, v, p;
    node_id_t *restrict cached_paths = self->likelihood_nodes_tmp;
    const int old_num_likelihood_nodes = self->num_likelihood_nodes;
    node_id_t *restrict L_nodes = self->likelihood_nodes;
    int j, num_cached_paths, num_likelihood_nodes;

    num_cached_paths = 0;
    num_likelihood_nodes = 0;
    for (j = 0; j < old_num_likelihood_nodes; j++) {
        u = L_nodes[j];
        p = parent[u];
        if (p != NULL_NODE) {
            cached_paths[num_cached_paths] = p;
            num_cached_paths++;
            v = p;
            while (likely(L[v] == NULL_LIKELIHOOD) && likely(L_cache[v] == CACHE_UNSET)) {
                v = parent[v];
            }
            L_p = L_cache[v];
            if (unlikely(L_p == CACHE_UNSET)) {
                L_p = L[v];
            }
            /* Fill in the L cache */
            v = p;
            while (likely(L[v] == NULL_LIKELIHOOD) && likely(L_cache[v] == CACHE_UNSET)) {
                L_cache[v] = L_p;
                v = parent[v];
            }
            /* If the likelihood for the parent is equal to the child we can
             * delete the child likelihood */
            if (approximately_equal(L[u], L_p)) {
                L[u] = NULL_LIKELIHOOD;
            }
        }
        if (L[u] >= 0) {
            L_nodes[num_likelihood_nodes] = L_nodes[j];
            num_likelihood_nodes++;
        }
    }
    assert(num_likelihood_nodes > 0);

    self->num_likelihood_nodes = num_likelihood_nodes;
    /* Reset the L cache */
    for (j = 0; j < num_cached_paths; j++) {
        v = cached_paths[j];
        while (likely(v != NULL_NODE) && likely(L_cache[v] != CACHE_UNSET)) {
            L_cache[v] = CACHE_UNSET;
            v = parent[v];
        }
    }
    return ret;
}

static int
ancestor_matcher_update_site_state(ancestor_matcher_t *self, const site_id_t site,
        const allele_t state, node_id_t *restrict parent, double *restrict L,
        double *restrict L_cache)
{
    int ret = 0;
    node_id_t mutation_node = NULL_NODE;
    node_id_t u;

    if (self->tree_sequence_builder->sites.mutations[site] != NULL) {
        mutation_node = self->tree_sequence_builder->sites.mutations[site]->node;
    }
    /* ancestor_matcher_print_state(self, stdout); */
    /* ancestor_matcher_check_state(self); */
    if (self->observation_error == 0.0
                && mutation_node == NULL_NODE
                && site > 0
                && self->tree_sequence_builder->sites.mutations[site - 1] == NULL) {
        /* If there are no mutations at this or the last site, then
         * we are guaranteed that the likelihoods are equal. */
        self->traceback[site] = self->traceback[site - 1];
        self->total_traceback_size += self->num_likelihood_nodes;
    } else {
        if (mutation_node != NULL_NODE) {
            /* Insert a new L-value for the mutation node if needed */
            if (L[mutation_node] == NULL_LIKELIHOOD) {
                u = mutation_node;
                while (L[u] == NULL_LIKELIHOOD) {
                    u = parent[u];
                    assert(u != NULL_NODE);
                }
                L[mutation_node] = L[u];
                self->likelihood_nodes[self->num_likelihood_nodes] = mutation_node;
                self->num_likelihood_nodes++;
            }
        }
        ret = ancestor_matcher_store_traceback(self, site, L);
        if (ret != 0) {
            goto out;
        }
        if (self->observation_error > 0 || mutation_node != NULL_NODE) {
            ret = ancestor_matcher_update_site_likelihood_values(self, site,
                    mutation_node, state, parent, L);
            if (ret != 0) {
                goto out;
            }
            ret = ancestor_matcher_coalesce_likelihoods(self, parent, L, L_cache);
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

    memset(self->traceback, 0, self->num_sites * sizeof(likelihood_list_t *));
    memset(self->path_cache, 0xff, self->num_nodes * sizeof(int8_t));
    ret = block_allocator_reset(&self->likelihood_list_allocator);
    if (ret != 0) {
        goto out;
    }
    self->total_traceback_size = 0;
    self->num_likelihood_nodes = 0;
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
    node_id_t *restrict parent = self->parent;
    double *restrict L = self->likelihood;
    const edge_t *restrict edges = self->tree_sequence_builder->edges;
    const node_id_t *restrict I = self->tree_sequence_builder->removal_order;
    node_id_t u, max_likelihood_node;
    site_id_t left, right, pos;
    likelihood_list_t *restrict z;
    mutation_list_node_t *mut_list;

    /* Prepare for the traceback and get the memory ready for recording
     * the output edges. */
    self->output.size = 0;
    self->output.right[self->output.size] = end;
    self->output.parent[self->output.size] = NULL_NODE;

    /* Process the final likelihoods and find the maximum value. Reset
     * the likelihood values so that we can reused the buffer during
     * traceback. */
    for (j = 0; j < self->num_likelihood_nodes; j++) {
        u = self->likelihood_nodes[j];
        if (approximately_one(L[u])) {
            self->output.parent[self->output.size] = u;
        }
        L[u] = NULL_LIKELIHOOD;
    }
    self->num_likelihood_nodes = 0;
    assert(self->output.parent[self->output.size] != NULL_NODE);

    /* Now go through the trees in reverse and run the traceback */
    memset(match, 0xff, self->num_sites * sizeof(allele_t));
    memset(parent, 0xff, self->num_nodes * sizeof(node_id_t));
    j = M - 1;
    k = M - 1;
    pos = self->num_sites;

    while (pos > start) {
        while (k >= 0 && edges[k].left == pos) {
            parent[edges[k].child] = NULL_NODE;
            k--;
        }
        while (j >= 0 && edges[I[j]].right == pos) {
            parent[edges[I[j]].child] = edges[I[j]].parent;
            j--;
        }
        right = pos;
        left = 0;
        if (k >= 0) {
            left = TSI_MAX(left, edges[k].left);
        }
        if (j >= 0) {
            left = TSI_MAX(left, edges[I[j]].right);
        }
        pos = left;

        assert(left < right);

        /* # print("left = ", left, "right = ", right) */
        /* printf("left = %d right = %d\n", left, right); */
        for (l = TSI_MIN(right, end) - 1; l >= (int) TSI_MAX(left, start); l--) {
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
                if (is_descendant(u, mut_list->node, parent)) {
                    match[l] = 1;
                }
            }
            /* Get the likelihood for u */
            while (L[u] == NULL_LIKELIHOOD) {
                u = parent[u];
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
remove_edge(edge_t edge, node_id_t *restrict parent, node_id_t *restrict left_child,
        node_id_t *restrict right_child, node_id_t *restrict left_sib,
        node_id_t *restrict right_sib)
{
    node_id_t p = edge.parent;
    node_id_t c = edge.child;
    node_id_t lsib = left_sib[c];
    node_id_t rsib = right_sib[c];

    /* printf("REMOVE EDGE %d\t%d\t%d\t%d\n", edge.left, edge.right, edge.parent, edge.child); */
    if (lsib == NULL_NODE) {
        left_child[p] = rsib;
    } else {
        right_sib[lsib] = rsib;
    }
    if (rsib == NULL_NODE) {
        right_child[p] = lsib;
    } else {
        left_sib[rsib] = lsib;
    }
    parent[c] = NULL_NODE;
    left_sib[c] = NULL_NODE;
    right_sib[c] = NULL_NODE;
}

static inline void
insert_edge(edge_t edge, node_id_t *restrict parent, node_id_t *restrict left_child,
        node_id_t *restrict right_child, node_id_t *restrict left_sib,
        node_id_t *restrict right_sib)
{
    node_id_t p = edge.parent;
    node_id_t c = edge.child;
    node_id_t u = right_child[p];
    /* printf("INSERT EDGE %d\t%d\t%d\t%d\n", edge.left, edge.right, edge.parent, edge.child); */

    parent[c] = p;
    if (u == NULL_NODE) {
        left_child[p] = c;
        left_sib[c] = NULL_NODE;
        right_sib[c] =NULL_NODE;
    } else {
        right_sib[u] = c;
        left_sib[c] = u;
        right_sib[c] = NULL_NODE;
    }
    right_child[p] = c;
}

/* After we have removed a 1.0 valued node, we must renormalise the likelihoods */
static void
ancestor_matcher_renormalise_likelihoods(ancestor_matcher_t *self, double *restrict L)
{
    double max_L = -1;
    const int num_likelihood_nodes = self->num_likelihood_nodes;
    node_id_t *restrict L_nodes = self->likelihood_nodes;
    node_id_t u;
    int j;

    for (j = 0; j < num_likelihood_nodes; j++) {
        u = L_nodes[j];
        max_L = TSI_MAX(max_L,  L[u]);
    }
    assert(max_L > 0);
    for (j = 0; j < num_likelihood_nodes; j++) {
        u = L_nodes[j];
        L[u] /= max_L;
    }
}

static int
ancestor_matcher_run_forwards_match(ancestor_matcher_t *self, site_id_t start,
        site_id_t end, allele_t *haplotype)
{
    int ret = 0;
    int M = self->tree_sequence_builder->num_edges;
    int j, k, l, remove_start;
    site_id_t site;
    edge_t edge;
    node_id_t u;
    double L_child = 0;
    /* Use the restrict keyword here to try to improve performance by avoiding
     * unecessary loads. We must be very careful to to ensure that all references
     * to this memory for the duration of this function is through these variables.
     */
    double *restrict L = self->likelihood;
    double *restrict L_cache = self->likelihood_cache;
    node_id_t *restrict parent = self->parent;
    node_id_t *restrict left_child = self->left_child;
    node_id_t *restrict right_child = self->right_child;
    node_id_t *restrict left_sib = self->left_sib;
    node_id_t *restrict right_sib = self->right_sib;
    const edge_t *restrict edges = self->tree_sequence_builder->edges;
    const node_id_t *restrict O = self->tree_sequence_builder->removal_order;
    site_id_t pos, left, right;
    bool renormalise_required;

    /* Load the tree for start */
    j = 0;
    k = 0;
    left = 0;
    pos = 0;
    right = self->num_sites;

    /* printf("FILLING FIRST TREE\n"); */
    while (j < M && k < M && edges[j].left <= start) {
        while (k < M && edges[O[k]].right == pos) {
            remove_edge(edges[O[k]], parent, left_child, right_child, left_sib, right_sib);
            k++;
        }
        while (j < M && edges[j].left == pos) {
            insert_edge(edges[j], parent, left_child, right_child, left_sib, right_sib);
            j++;
        }
        left = pos;
        right = self->num_sites;
        if (j < M) {
            right = TSI_MIN(right, edges[j].left);
        }
        if (k < M) {
            right = TSI_MIN(right, edges[O[k]].right);
        }
        pos = right;
    }

    /* Insert the initial likelihoods. Zero is the root, and it has likelihood
     * one. All non-zero roots are marked with a special value so we can
     * identify them when the enter the tree */
    L[0] = 1.0;
    L_cache[0] = CACHE_UNSET;
    self->likelihood_nodes[0] = 0;
    self->num_likelihood_nodes = 1;
    for (u = 1; u < (node_id_t) self->num_nodes; u++) {
        L_cache[u] = CACHE_UNSET;
        if (parent[u] != NULL_NODE) {
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
            if (unlikely(is_nonzero_root(edge.child, parent, left_child))) {
                if (approximately_one(L[edge.child])) {
                    renormalise_required = true;
                }
                if (L[edge.child] != NULL_LIKELIHOOD) {
                    ancestor_matcher_delete_likelihood(self, edge.child, L);
                }
                L[edge.child] = NONZERO_ROOT_LIKELIHOOD;
            }
        }
        if (unlikely(renormalise_required)) {
            ancestor_matcher_renormalise_likelihoods(self, L);
        }

        /* ancestor_matcher_print_state(self, stdout); */
        /* ancestor_matcher_check_state(self); */
        for (site = TSI_MAX(left, start); site < TSI_MIN(right, end); site++) {
            ret = ancestor_matcher_update_site_state(self, site, haplotype[site],
                    parent, L, L_cache);
            if (ret != 0) {
                goto out;
            }
        }

        /* Move on to the next tree */
        remove_start = k;
        while (k < M  && edges[O[k]].right == right) {
            edge = edges[O[k]];
            remove_edge(edge, parent, left_child, right_child, left_sib, right_sib);
            k++;
            if (L[edge.child] == NULL_LIKELIHOOD) {
                u = edge.parent;
                while (likely(L[u] == NULL_LIKELIHOOD)
                        && likely(L_cache[u] == CACHE_UNSET)) {
                    u = parent[u];
                }
                L_child = L_cache[u];
                if (unlikely(L_child == CACHE_UNSET)) {
                    L_child = L[u];
                }
                assert(L_child >= 0);
                u = edge.parent;
                /* Fill in the cache by traversing back upwards */
                /* printf("Filling cache"); */
                while (likely(L[u] == NULL_LIKELIHOOD)
                        && likely(L_cache[u] == CACHE_UNSET)) {
                    /* printf("%d ", u); */
                    L_cache[u] = L_child;
                    u = parent[u];
                }
                /* printf("\n"); */
                L[edge.child] = L_child;
                self->likelihood_nodes[self->num_likelihood_nodes] = edge.child;
                self->num_likelihood_nodes++;
            }
        }
        /* reset the L cache */
        for (l = remove_start; l < k; l++) {
            edge = edges[O[l]];
            u = edge.parent;
            while (likely(L_cache[u] != CACHE_UNSET)) {
                L_cache[u] = CACHE_UNSET;
                u = parent[u];
            }
        }

        left = right;
        /* printf("Inserting for j = %d and left = %d (%d)\n", (int) j, (int) left, */
        /*         edges[I[j]].left); */
        while (j < M && edges[j].left == left) {
            edge = edges[j];
            insert_edge(edge, parent, left_child, right_child, left_sib, right_sib);
            j++;
            /* Insert zero likelihoods for any nonzero roots that have entered
             * the tree. Note we don't bother trying to compress the tree here
             * because this will be done for the next site anyway. */
            if (unlikely(L[edge.parent] == NONZERO_ROOT_LIKELIHOOD)) {
                L[edge.parent] = 0;
                self->likelihood_nodes[self->num_likelihood_nodes] = edge.parent;
                self->num_likelihood_nodes++;
            }
            if (unlikely(L[edge.child] == NONZERO_ROOT_LIKELIHOOD)) {
                L[edge.child] = 0;
                self->likelihood_nodes[self->num_likelihood_nodes] = edge.child;
                self->num_likelihood_nodes++;
            }
        }
        right = self->num_sites;
        if (j < M) {
            right = TSI_MIN(right, edges[j].left);
        }
        if (k < M) {
            right = TSI_MIN(right, edges[O[k]].right);
        }
    }
out:
    return ret;
}


int
ancestor_matcher_find_path(ancestor_matcher_t *self,
        site_id_t start, site_id_t end, allele_t *haplotype,
        allele_t *matched_haplotype, size_t *num_output_edges,
        site_id_t **left_output, site_id_t **right_output, node_id_t **parent_output)
{
    int ret = 0;

    ret = ancestor_matcher_reset(self);
    if (ret != 0) {
        goto out;
    }
    ret = ancestor_matcher_run_forwards_match(self, start, end, haplotype);
    if (ret != 0) {
        goto out;
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
