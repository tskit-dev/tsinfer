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

    /* Check the properties of the likelihood map */
    for (j = 0; j < self->num_likelihood_nodes; j++) {
        u = self->likelihood_nodes[j];
        assert(self->likelihood[u] >= 0 && self->likelihood[u] <= 2);
    }
    /* Make sure that there are no other non null likelihoods in the array */
    num_likelihoods = 0;
    for (u = 0; u < (node_id_t) self->num_nodes; u++) {
        if (self->likelihood[u] >= 0) {
            num_likelihoods++;
        }
        if (is_nonzero_root(u, self->parent, self->left_child)) {
            assert(self->likelihood[u] == NONZERO_ROOT_LIKELIHOOD);
        }
    }
    assert(num_likelihoods == self->num_likelihood_nodes);
}

int
ancestor_matcher_print_state(ancestor_matcher_t *self, FILE *out)
{
    int j, k;
    node_id_t u;

    fprintf(out, "Ancestor matcher state\n");
    fprintf(out, "tree = \n");
    fprintf(out, "id\tparent\tlchild\trchild\tlsib\trsib\tlikelihood\n");
    for (j = 0; j < (int) self->num_nodes; j++) {
        fprintf(out, "%d\t%d\t%d\t%d\t%d\t%d\t%d\n", (int) j, self->parent[j],
                self->left_child[j], self->right_child[j], self->left_sib[j],
                self->right_sib[j], self->likelihood[j]);
    }
    fprintf(out, "likelihood nodes\n");
    /* Check the properties of the likelihood map */
    for (j = 0; j < self->num_likelihood_nodes; j++) {
        u = self->likelihood_nodes[j];
        printf("%d -> %d\n", u, self->likelihood[u]);
    }
    fprintf(out, "traceback\n");
    for (j = 0; j < (int) self->num_sites; j++) {
        fprintf(out, "\t%d:%d (%d)\t", (int) j,
                self->max_likelihood_node[j], self->traceback[j].size);
        for (k = 0; k < self->traceback[j].size; k++) {
            fprintf(out, "(%d, %d)", self->traceback[j].node[k],
                    self->traceback[j].recombination_required[k]);
        }
        fprintf(out, "\n");
    }
    block_allocator_print_state(&self->traceback_allocator, out);

    /* ancestor_matcher_check_state(self); */
    return 0;
}

int
ancestor_matcher_alloc(ancestor_matcher_t *self,
        tree_sequence_builder_t *tree_sequence_builder, int flags)
{
    int ret = 0;
    /* TODO make these input parameters. */
    size_t traceback_block_size = 64 * 1024 * 1024;

    memset(self, 0, sizeof(ancestor_matcher_t));
    /* All allocs for arrays related to nodes are done in expand_nodes */
    self->flags = flags;
    self->max_nodes = 0;
    self->tree_sequence_builder = tree_sequence_builder;
    self->num_sites = tree_sequence_builder->num_sites;
    self->output.max_size = self->num_sites; /* We can probably make this smaller */
    self->traceback = calloc(self->num_sites, sizeof(node_state_list_t));
    self->max_likelihood_node = malloc(self->num_sites * sizeof(node_id_t));
    self->output.left = malloc(self->output.max_size * sizeof(site_id_t));
    self->output.right = malloc(self->output.max_size * sizeof(site_id_t));
    self->output.parent = malloc(self->output.max_size * sizeof(node_id_t));
    if (self->traceback == NULL || self->max_likelihood_node == NULL
            || self->output.left == NULL || self->output.right == NULL
            || self->output.parent == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    ret = block_allocator_alloc(&self->traceback_allocator, traceback_block_size);
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
    tsi_safe_free(self->recombination_required);
    tsi_safe_free(self->likelihood);
    tsi_safe_free(self->likelihood_cache);
    tsi_safe_free(self->likelihood_nodes);
    tsi_safe_free(self->likelihood_nodes_tmp);
    tsi_safe_free(self->path_cache);
    tsi_safe_free(self->max_likelihood_node);
    tsi_safe_free(self->traceback);
    tsi_safe_free(self->output.left);
    tsi_safe_free(self->output.right);
    tsi_safe_free(self->output.parent);
    block_allocator_free(&self->traceback_allocator);
    return 0;
}

static int
ancestor_matcher_delete_likelihood(ancestor_matcher_t *self, const node_id_t node,
        int8_t *restrict L)
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

/* Store the recombination_required state in the traceback */
static int WARN_UNUSED
ancestor_matcher_store_traceback(ancestor_matcher_t *self, const site_id_t site_id)
{
    int ret = 0;
    node_id_t u;
    int j;
    int8_t *restrict list_R;
    node_id_t *restrict list_node;
    node_state_list_t *restrict list;
    node_state_list_t *restrict T = self->traceback;
    const node_id_t *restrict nodes = self->likelihood_nodes;
    const int8_t *restrict R = self->recombination_required;
    const int num_likelihood_nodes = self->num_likelihood_nodes;
    bool match;

    /* Check to see if the previous site has the same recombination_required. If so,
     * we can reuse the same list. */
    match = false;
    if (site_id > 0) {
        list = &T[site_id - 1];
        if (list->size == num_likelihood_nodes) {
            list_node = list->node;
            list_R = list->recombination_required;
            match = true;
            for (j = 0; j < num_likelihood_nodes; j++) {
                if (list_node[j] != nodes[j] || list_R[j] != R[nodes[j]]) {
                    match = false;
                    break;
                }
            }
        }
    }

    if (match) {
        T[site_id].size = T[site_id - 1].size;
        T[site_id].node = T[site_id - 1].node;
        T[site_id].recombination_required = T[site_id - 1].recombination_required;
    } else {
        list_node = block_allocator_get(&self->traceback_allocator,
                num_likelihood_nodes * sizeof(node_id_t));
        list_R = block_allocator_get(&self->traceback_allocator,
                num_likelihood_nodes * sizeof(int8_t));
        if (list_node == NULL || list_R == NULL) {
            ret = TSI_ERR_NO_MEMORY;
            goto out;
        }
        T[site_id].node = list_node;
        T[site_id].recombination_required = list_R;
        T[site_id].size = num_likelihood_nodes;
        for (j = 0; j < num_likelihood_nodes; j++) {
            u = nodes[j];
            list_node[j] = u;
            list_R[j] = R[u];
        }
    }
    self->total_traceback_size += num_likelihood_nodes;
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
        const node_id_t *restrict parent, int8_t *restrict L)
{
    int ret = 0;
    const int num_likelihood_nodes = self->num_likelihood_nodes;
    const node_id_t *restrict L_nodes = self->likelihood_nodes;
    int8_t *restrict path_cache = self->path_cache;
    int8_t *restrict recombination_required = self->recombination_required;
    int j, descendant;
    node_id_t u, v, max_L_node;
    int8_t max_L;

    max_L = -1;
    max_L_node = NULL_NODE;
    assert(num_likelihood_nodes > 0);
    /* printf("likelihoods for node=%d, n=%d\n", mutation_node, self->num_likelihood_nodes); */
    for (j = 0; j < num_likelihood_nodes; j++) {
        u = L_nodes[j];
        /* Determine if the node this likelihood is associated with is a descendant
         * of the mutation node. To avoid the cost of repeatedly traversing up the
         * tree, we keep a cache of the paths that we have already traversed. When
         * we meet one of these paths we can immediately finish.
         */
        descendant = 0;
        if (mutation_node != NULL_NODE) {
            v = u;
            while (likely(v != NULL_NODE)
                    && likely(v != mutation_node)
                    && likely(path_cache[v] == CACHE_UNSET)) {
                v = parent[v];
            }
            if (likely(v != NULL_NODE) && likely(path_cache[v] != CACHE_UNSET)) {
                descendant = path_cache[v];
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
        recombination_required[u] = false;
        if (L[u] == MISMATCH_LIKELIHOOD) {
            recombination_required[u] = true;
        }
        if (mutation_node != NULL_NODE && descendant != state) {
            L[u] = MISMATCH_LIKELIHOOD;
        } else if (L[u] == MISMATCH_LIKELIHOOD) {
            L[u] = RECOMB_LIKELIHOOD;
        }
        if (L[u] > max_L) {
            max_L = L[u];
            max_L_node = u;
        }
    }
    assert(max_L >= 0);
    assert(max_L_node != NULL_NODE);
    self->max_likelihood_node[site] = max_L_node;

    /* Reset the path cache and renormalise the likelihoods. */
    for (j = 0; j < num_likelihood_nodes; j++) {
        u = L_nodes[j];
        if (L[u] == max_L) {
            L[u] = MATCH_LIKELIHOOD;
        }
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
        const node_id_t *restrict parent, int8_t *restrict L, int8_t *restrict L_cache)
{
    int ret = 0;
    int8_t L_p;
    node_id_t u, v, p;
    node_id_t *restrict cached_paths = self->likelihood_nodes_tmp;
    const int old_num_likelihood_nodes = self->num_likelihood_nodes;
    node_id_t *restrict L_nodes = self->likelihood_nodes;
    int j, num_cached_paths, num_likelihood_nodes;

    num_cached_paths = 0;
    num_likelihood_nodes = 0;
    assert(old_num_likelihood_nodes > 0);
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
            if (L[u] == L_p) {
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
        const allele_t state, node_id_t *restrict parent, int8_t *restrict L,
        int8_t *restrict L_cache)
{
    int ret = 0;
    node_id_t mutation_node = NULL_NODE;
    node_id_t u;

    assert(self->num_likelihood_nodes > 0);

    if (self->tree_sequence_builder->sites.mutations[site] != NULL) {
        mutation_node = self->tree_sequence_builder->sites.mutations[site]->node;
    }
    if (self->flags & TSI_EXTENDED_CHECKS) {
        ancestor_matcher_check_state(self);
    }
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
    ret = ancestor_matcher_update_site_likelihood_values(self, site,
            mutation_node, state, parent, L);
    if (ret != 0) {
        goto out;
    }
    ret = ancestor_matcher_store_traceback(self, site);
    if (ret != 0) {
        goto out;
    }
    ret = ancestor_matcher_coalesce_likelihoods(self, parent, L, L_cache);
    if (ret != 0) {
        goto out;
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
    memset(self->recombination_required, 0xff, self->num_nodes * sizeof(int8_t));
}

static int WARN_UNUSED
ancestor_matcher_expand_nodes(ancestor_matcher_t *self)
{
    int ret = TSI_ERR_NO_MEMORY;

    tsi_safe_free(self->parent);
    tsi_safe_free(self->left_child);
    tsi_safe_free(self->right_child);
    tsi_safe_free(self->left_sib);
    tsi_safe_free(self->right_sib);
    tsi_safe_free(self->recombination_required);
    tsi_safe_free(self->likelihood);
    tsi_safe_free(self->likelihood_cache);
    tsi_safe_free(self->likelihood_nodes);
    tsi_safe_free(self->likelihood_nodes_tmp);
    tsi_safe_free(self->path_cache);

    assert(self->max_nodes > 0);
    self->parent = malloc(self->max_nodes * sizeof(node_id_t));
    self->left_child = malloc(self->max_nodes * sizeof(node_id_t));
    self->right_child = malloc(self->max_nodes * sizeof(node_id_t));
    self->left_sib = malloc(self->max_nodes * sizeof(node_id_t));
    self->right_sib = malloc(self->max_nodes * sizeof(node_id_t));
    self->recombination_required = malloc(self->max_nodes * sizeof(int8_t));
    self->likelihood = malloc(self->max_nodes * sizeof(int8_t));
    self->likelihood_cache = malloc(self->max_nodes * sizeof(int8_t));
    self->likelihood_nodes = malloc(self->max_nodes * sizeof(node_id_t));
    self->likelihood_nodes_tmp = malloc(self->max_nodes * sizeof(node_id_t));
    self->path_cache = malloc(self->max_nodes * sizeof(int8_t));

    if (self->parent == NULL
            || self->left_child == NULL || self->right_child == NULL
            || self->left_sib == NULL || self->right_sib == NULL
            || self->recombination_required == NULL
            || self->likelihood == NULL || self->likelihood_cache == NULL
            || self->likelihood_nodes == NULL
            || self->likelihood_nodes_tmp == NULL || self->path_cache == NULL) {
        goto out;
    }
    ret = 0;
out:
    return ret;
}

static int
ancestor_matcher_reset(ancestor_matcher_t *self)
{
    int ret = 0;

    /* TODO realloc when this grows */
    if (self->max_nodes != self->tree_sequence_builder->max_nodes) {
        self->max_nodes = self->tree_sequence_builder->max_nodes;
        ret = ancestor_matcher_expand_nodes(self);
        if (ret != 0) {
            goto out;
        }
    }
    self->num_nodes = self->tree_sequence_builder->num_nodes;
    assert(self->num_nodes <= self->max_nodes);

    memset(self->path_cache, 0xff, self->num_nodes * sizeof(int8_t));
    ret = block_allocator_reset(&self->traceback_allocator);
    if (ret != 0) {
        goto out;
    }
    self->total_traceback_size = 0;
    self->num_likelihood_nodes = 0;
    ancestor_matcher_reset_tree(self);
out:
    return ret;
}

/* Resets the recombination_required array from the traceback at the specified site.
 */
static inline void
ancestor_matcher_set_recombination_required(ancestor_matcher_t *self, site_id_t site,
        int8_t *restrict recombination_required)
{
    int j;
    const int8_t *restrict R = self->traceback[site].recombination_required;
    const node_id_t *restrict node = self->traceback[site].node;
    const int size = self->traceback[site].size;

    /* We always set recombination_required for node 0 to false for the cases
     * where no recombination is needed at a particular site (which are
     * encoded by a traceback of size 0) */
    recombination_required[0] = 0;
    for (j = 0; j < size; j++) {
        recombination_required[node[j]] = R[j];
    }
}

/* Unsets the likelihood array from the traceback at the specified site.
 */
static inline void
ancestor_matcher_unset_recombination_required(ancestor_matcher_t *self, site_id_t site,
        int8_t *restrict recombination_required)
{
    int j;
    const node_id_t *restrict node = self->traceback[site].node;
    const int size = self->traceback[site].size;

    for (j = 0; j < size; j++) {
        recombination_required[node[j]] = -1;
    }
    recombination_required[0] = -1;
}

static int WARN_UNUSED
ancestor_matcher_run_traceback(ancestor_matcher_t *self, site_id_t start,
        site_id_t end, allele_t *haplotype, allele_t *match)
{
    int ret = 0;
    site_id_t l;
    edge_t edge;
    node_id_t u, max_likelihood_node;
    site_id_t left, right, pos;
    mutation_list_node_t *mut_list;
    node_id_t *restrict parent = self->parent;
    int8_t *restrict recombination_required = self->recombination_required;
    const edge_t *restrict in = self->tree_sequence_builder->right_index_edges;
    const edge_t *restrict out = self->tree_sequence_builder->left_index_edges;
    int_fast32_t in_index = (int_fast32_t) self->tree_sequence_builder->num_edges - 1;
    int_fast32_t out_index = (int_fast32_t) self->tree_sequence_builder->num_edges - 1;

    /* Prepare for the traceback and get the memory ready for recording
     * the output edges. */
    self->output.size = 0;
    self->output.right[self->output.size] = end;
    self->output.parent[self->output.size] = NULL_NODE;

    max_likelihood_node = self->max_likelihood_node[end - 1];
    assert(max_likelihood_node != NULL_NODE);
    self->output.parent[self->output.size] = max_likelihood_node;
    assert(self->output.parent[self->output.size] != NULL_NODE);

    /* Now go through the trees in reverse and run the traceback */
    memset(parent, 0xff, self->num_nodes * sizeof(*parent));
    memset(recombination_required, 0xff,
            self->num_nodes * sizeof(*recombination_required));
    pos = self->num_sites;

    while (pos > start) {
        while (out_index >= 0 && out[out_index].left == pos) {
            edge = out[out_index];
            out_index--;
            parent[edge.child] = NULL_NODE;
        }
        while (in_index >= 0 && in[in_index].right == pos) {
            edge = in[in_index];
            in_index--;
            parent[edge.child] = edge.parent;
        }
        right = pos;
        left = 0;
        if (out_index >= 0) {
            left = TSI_MAX(left, out[out_index].left);
        }
        if (in_index >= 0) {
            left = TSI_MAX(left, in[in_index].right);
        }
        pos = left;

        /* The tree is ready; perform the traceback at each site in this tree */
        assert(left < right);
        for (l = TSI_MIN(right, end) - 1; l >= (int) TSI_MAX(left, start); l--) {
            match[l] = 0;
            u = self->output.parent[self->output.size];
            /* Set the state of the matched haplotype */
            mut_list = self->tree_sequence_builder->sites.mutations[l];
            if (mut_list != NULL) {
                if (is_descendant(u, mut_list->node, parent)) {
                    match[l] = 1;
                }
            }
            /* Mark the traceback nodes on the tree */
            ancestor_matcher_set_recombination_required(self, l, recombination_required);

            /* Traverse up the tree from the current node. The first marked node that we
             * meed tells us whether we need to recombine */
            while (u != 0 && recombination_required[u] == -1) {
                u = parent[u];
                assert(u != NULL_NODE);
            }
            if (recombination_required[u] && l > start) {
                max_likelihood_node = self->max_likelihood_node[l - 1];
                assert(max_likelihood_node != self->output.parent[self->output.size]);
                assert(max_likelihood_node != NULL_NODE);
                self->output.left[self->output.size] = l;
                self->output.size++;
                assert(self->output.size < self->output.max_size);
                /* Start the next output edge */
                self->output.right[self->output.size] = l;
                self->output.parent[self->output.size] = max_likelihood_node;
            }
            /* Unset the values in the tree for the next site. */
            ancestor_matcher_unset_recombination_required(self, l, recombination_required);
        }
    }

    self->output.left[self->output.size] = start;
    self->output.size++;
    assert(self->output.right[self->output.size - 1] != start);
    return ret;
}

static inline void
remove_edge(edge_t edge, node_id_t *restrict parent, node_id_t *restrict left_child,
        node_id_t *restrict right_child, node_id_t *restrict left_sib,
        node_id_t *restrict right_sib)
{
    const node_id_t p = edge.parent;
    const node_id_t c = edge.child;
    const node_id_t lsib = left_sib[c];
    const node_id_t rsib = right_sib[c];

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
    const node_id_t p = edge.parent;
    const node_id_t c = edge.child;
    const node_id_t u = right_child[p];

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

static int
ancestor_matcher_run_forwards_match(ancestor_matcher_t *self, site_id_t start,
        site_id_t end, allele_t *haplotype)
{
    int ret = 0;
    site_id_t site;
    edge_t edge;
    node_id_t u, root, last_root;
    int8_t L_child = 0;
    /* Use the restrict keyword here to try to improve performance by avoiding
     * unecessary loads. We must be very careful to to ensure that all references
     * to this memory for the duration of this function is through these variables.
     */
    int8_t *restrict L = self->likelihood;
    int8_t *restrict L_cache = self->likelihood_cache;
    node_id_t *restrict parent = self->parent;
    node_id_t *restrict left_child = self->left_child;
    node_id_t *restrict right_child = self->right_child;
    node_id_t *restrict left_sib = self->left_sib;
    node_id_t *restrict right_sib = self->right_sib;
    site_id_t pos, left, right;
    const edge_t *restrict in = self->tree_sequence_builder->left_index_edges;
    const edge_t *restrict out = self->tree_sequence_builder->right_index_edges;
    const int_fast32_t M = (site_id_t) self->tree_sequence_builder->num_edges;
    int_fast32_t in_index, out_index, l, remove_start;

    /* Load the tree for start */
    left = 0;
    pos = 0;
    in_index = 0;
    out_index = 0;
    right = self->num_sites;
    if (in_index < M && start < in[in_index].left) {
        right = in[in_index].left;
    }

    /* TODO there's probably quite a big gain to made here by seeking
     * directly to the tree that we're interested in rather than just
     * building the trees sequentially */
    while (in_index < M && out_index < M && in[in_index].left <= start) {
        while (out_index < M && out[out_index].right == pos) {
            remove_edge(out[out_index], parent, left_child, right_child,
                    left_sib, right_sib);
            out_index++;
        }
        while (in_index < M && in[in_index].left == pos) {
            insert_edge(in[in_index], parent, left_child, right_child,
                    left_sib, right_sib);
            in_index++;
        }
        left = pos;
        right = self->num_sites;
        if (in_index < M) {
            right = TSI_MIN(right, in[in_index].left);
        }
        if (out_index < M) {
            right = TSI_MIN(right, out[out_index].right);
        }
        pos = right;
    }

    /* Insert the initial likelihoods. All non-zero roots are marked with a
     * special value so we can identify them when the enter the tree */
    L_cache[0] = CACHE_UNSET;
    for (u = 0; u < (node_id_t) self->num_nodes; u++) {
        L_cache[u] = CACHE_UNSET;
        if (parent[u] != NULL_NODE) {
            L[u] = NULL_LIKELIHOOD;
        } else {
            L[u] = NONZERO_ROOT_LIKELIHOOD;
        }
    }
    if (self->flags & TSI_EXTENDED_CHECKS) {
        ancestor_matcher_check_state(self);
    }
    last_root = 0;
    if (left_child[0] != NULL_NODE) {
        last_root = left_child[0];
        assert(right_sib[last_root] == NULL_NODE);
    }
    L[last_root] = 1.0;
    self->likelihood_nodes[0] = last_root;
    self->num_likelihood_nodes = 1;

    remove_start = out_index;
    while (left < end) {
        assert(left < right);

        /* Remove the likelihoods for any nonzero roots that have just left
         * the tree */
        for (l = remove_start; l < out_index; l++) {
            edge = out[l];
            if (unlikely(is_nonzero_root(edge.child, parent, left_child))) {
                if (L[edge.child] >= 0 ) {
                    ancestor_matcher_delete_likelihood(self, edge.child, L);
                }
                L[edge.child] = NONZERO_ROOT_LIKELIHOOD;
            }
            if (unlikely(is_nonzero_root(edge.parent, parent, left_child))) {
                if (L[edge.parent] >= 0) {
                    ancestor_matcher_delete_likelihood(self, edge.parent, L);
                }
                L[edge.parent] = NONZERO_ROOT_LIKELIHOOD;
            }
        }

        root = 0;
        if (left_child[0] != NULL_NODE) {
            root = left_child[0];
            assert(right_sib[root] == NULL_NODE);
        }
        if (root != last_root) {
            if (last_root == 0) {
                ancestor_matcher_delete_likelihood(self, last_root, L);
                L[last_root] = NONZERO_ROOT_LIKELIHOOD;
            }
            if (L[root] == NONZERO_ROOT_LIKELIHOOD) {
                L[root] = MISMATCH_LIKELIHOOD;
                self->likelihood_nodes[self->num_likelihood_nodes] = root;
                self->num_likelihood_nodes++;
            }
            last_root = root;
        }

        if (self->flags & TSI_EXTENDED_CHECKS) {
            ancestor_matcher_check_state(self);
        }
        for (site = TSI_MAX(left, start); site < TSI_MIN(right, end); site++) {
            ret = ancestor_matcher_update_site_state(self, site, haplotype[site],
                    parent, L, L_cache);
            if (ret != 0) {
                goto out;
            }
        }

        /* Move on to the next tree */
        remove_start = out_index;
        while (out_index < M && out[out_index].right == right) {
            edge = out[out_index];
            out_index++;
            remove_edge(edge, parent, left_child, right_child, left_sib, right_sib);
            assert(L[edge.child] != NONZERO_ROOT_LIKELIHOOD);
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
                while (likely(L[u] == NULL_LIKELIHOOD)
                        && likely(L_cache[u] == CACHE_UNSET)) {
                    L_cache[u] = L_child;
                    u = parent[u];
                }
                L[edge.child] = L_child;
                self->likelihood_nodes[self->num_likelihood_nodes] = edge.child;
                self->num_likelihood_nodes++;
            }
        }
        /* reset the L cache */
        for (l = remove_start; l < out_index; l++) {
            edge = out[l];
            u = edge.parent;
            while (likely(L_cache[u] != CACHE_UNSET)) {
                L_cache[u] = CACHE_UNSET;
                u = parent[u];
            }
        }

        left = right;
        while (in_index < M && in[in_index].left == left) {
            edge = in[in_index];
            in_index++;
            insert_edge(edge, parent, left_child, right_child, left_sib, right_sib);
            /* Insert zero likelihoods for any nonzero roots that have entered
             * the tree. Note we don't bother trying to compress the tree here
             * because this will be done for the next site anyway. */
            if (unlikely(edge.parent != 0 && L[edge.parent] == NONZERO_ROOT_LIKELIHOOD)) {
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
        if (in_index < M) {
            right = TSI_MIN(right, in[in_index].left);
        }
        if (out_index < M) {
            right = TSI_MIN(right, out[out_index].right);
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
    /* Reset some memory for the next call */
    memset(self->traceback + start, 0, (end - start) * sizeof(*self->traceback));
    memset(self->max_likelihood_node + start, 0xff,
            (end - start) * sizeof(*self->max_likelihood_node));

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
    size_t total = self->traceback_allocator.total_size;
    /* TODO add contributions from other objects */

    return total;
}
