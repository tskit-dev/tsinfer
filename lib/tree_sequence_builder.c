#include "tsinfer.h"
#include "err.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

static void
tree_sequence_builder_check_state(tree_sequence_builder_t *self)
{
    size_t l;
    list_segment_t *list_node, *last_list_node;
    child_list_node_t *u;
    size_t num_children = 0;
    size_t num_edgesets = 0;

    for (l = 0; l < self->num_ancestors; l++) {
        last_list_node = NULL;
        assert(self->children[l]->start == 0);
        for (list_node = self->children[l]; list_node != NULL;
                list_node = list_node->next) {
            if (list_node->head != NULL) {
                num_edgesets++;
            }
            for (u = list_node->head; u != NULL; u = u->next) {
                num_children++;
                if (u->next != NULL) {
                    assert(u->node < u->next->node);
                }
            }
            if (last_list_node != NULL) {
                assert(last_list_node->end == list_node->start);
            }
            last_list_node = list_node;
        }
        assert(last_list_node->end == self->num_sites);
    }
    assert(self->num_children == num_children);
    assert(self->num_edgesets == num_edgesets);

}

int
tree_sequence_builder_print_state(tree_sequence_builder_t *self, FILE *out)
{
    size_t l;
    list_segment_t *list_node;
    child_list_node_t *u;

    fprintf(out, "Tree sequence builder state\n");
    fprintf(out, "num_samples = %d\n", (int) self->num_samples);
    fprintf(out, "num_sites = %d\n", (int) self->num_sites);
    fprintf(out, "num_ancestors = %d\n", (int) self->num_ancestors);
    fprintf(out, "num_edgesets = %d\n", (int) self->num_edgesets);
    fprintf(out, "num_children = %d\n", (int) self->num_children);
    fprintf(out, "segment heap:\n");
    object_heap_print_state(&self->segment_heap, out);
    fprintf(out, "child_list_node heap:\n");
    object_heap_print_state(&self->child_list_node_heap, out);
    fprintf(out, "children:\n");
    for (l = 0; l < self->num_ancestors; l++) {
        fprintf(out, "%d\t:", (int) l);
        for (list_node = self->children[l]; list_node != NULL;
                list_node = list_node->next) {
            fprintf(out, "(%d,%d->[", list_node->start, list_node->end);
            for (u = list_node->head; u != NULL; u = u->next) {
                fprintf(out, "%d", u->node);
                if (u->next != NULL) {
                    fprintf(out, ",");
                }
            }
            fprintf(out, "])");
        }
        fprintf(out, "\n");
    }
    tree_sequence_builder_check_state(self);
    return 0;
}


static inline child_list_node_t * WARN_UNUSED
tree_sequence_builder_alloc_child_list_node(tree_sequence_builder_t *self,
        ancestor_id_t child)
{
    child_list_node_t *ret = NULL;

    if (object_heap_empty(&self->child_list_node_heap)) {
        if (object_heap_expand(&self->child_list_node_heap) != 0) {
            goto out;
        }
    }
    ret = (child_list_node_t *) object_heap_alloc_object(&self->child_list_node_heap);
    ret->node = child;
    ret->next = NULL;
    self->num_children++;
out:
    return ret;
}

static inline int
tree_sequence_builder_add_child(tree_sequence_builder_t *self, list_segment_t *seg,
        ancestor_id_t child)
{
    int ret = 0;
    child_list_node_t *u = tree_sequence_builder_alloc_child_list_node(self, child);

    if (u == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    if (seg->head == NULL) {
        assert(seg->tail == NULL);
        seg->head = u;
        seg->tail = u;
        self->num_edgesets++;
    } else {
        assert(seg->tail != NULL);
        seg->tail->next = u;
        seg->tail = u;
    }
out:
    return ret;
}

static inline list_segment_t * WARN_UNUSED
tree_sequence_builder_alloc_segment(tree_sequence_builder_t *self, site_id_t start,
        site_id_t end, child_list_node_t *head, list_segment_t *next)
{
    int err;
    list_segment_t *ret = NULL;
    child_list_node_t *u;

    if (object_heap_empty(&self->segment_heap)) {
        if (object_heap_expand(&self->segment_heap) != 0) {
            goto out;
        }
    }
    ret = (list_segment_t *) object_heap_alloc_object(&self->segment_heap);
    ret->start = start;
    ret->end = end;
    ret->head = NULL;
    ret->tail = NULL;
    /* Copy the input node list */
    u = head;
    while (u != NULL) {
        err = tree_sequence_builder_add_child(self, ret, u->node);
        if (err != 0) {
            goto out;
        }
        u = u->next;
    }
    ret->next = next;
out:
    return ret;
}

int
tree_sequence_builder_alloc(tree_sequence_builder_t *self,
        ancestor_store_t *store, size_t num_samples,
        size_t segment_block_size, size_t child_list_node_block_size)
{
    int ret = 0;
    size_t j;

    memset(self, 0, sizeof(tree_sequence_builder_t));
    self->store = store;
    self->num_sites = store->num_sites;
    self->num_ancestors = store->num_ancestors;
    self->num_samples = num_samples;
    self->segment_block_size = segment_block_size;
    self->child_list_node_block_size = child_list_node_block_size;
    ret = object_heap_init(&self->segment_heap, sizeof(list_segment_t),
           self->segment_block_size, NULL);
    if (ret != 0) {
        goto out;
    }
    ret = object_heap_init(&self->child_list_node_heap, sizeof(child_list_node_t),
           self->child_list_node_block_size, NULL);
    if (ret != 0) {
        goto out;
    }
    self->children = calloc(self->num_ancestors, sizeof(list_segment_t *));
    if (self->children == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    for (j = 0; j < self->num_ancestors; j++) {
        self->children[j] = tree_sequence_builder_alloc_segment(self, 0, self->num_sites,
                NULL, NULL);
        if (self->children[j] == NULL) {
            ret = TSI_ERR_NO_MEMORY;
            goto out;
        }
    }
out:
    return ret;
}

int
tree_sequence_builder_free(tree_sequence_builder_t *self)
{
    object_heap_free(&self->segment_heap);
    object_heap_free(&self->child_list_node_heap);
    tsi_safe_free(self->children);
    return 0;
}


static inline int
tree_sequence_builder_add_mapping(tree_sequence_builder_t *self, site_id_t start,
        site_id_t end, ancestor_id_t parent, ancestor_id_t child)
{
    int ret = 0;
    list_segment_t *u, *v;

    assert(start < end);
    assert(parent < (ancestor_id_t) self->num_ancestors);
    u = self->children[parent];
    /* Skip leading segments */
    while (u->end <= start) {
        u = u->next;
    }
    /* Trim off leading segment */
    if (u->start < start) {
        v = tree_sequence_builder_alloc_segment(self, start, u->end, u->head, u->next);
        if (v == NULL) {
            ret = TSI_ERR_NO_MEMORY;
            goto out;
        }
        u->end = start;
        u->next = v;
        u = v;
    }
    /* Update intersecting segments */
    while (u != NULL && u->end <= end) {
        ret = tree_sequence_builder_add_child(self, u, child);
        if (ret != 0) {
            goto out;
        }
        u = u->next;
    }
    /* Deal with trailing segment */
    if (u != NULL && end > u->start) {
        v = tree_sequence_builder_alloc_segment(self, end, u->end, u->head, u->next);
        if (v == NULL) {
            ret = TSI_ERR_NO_MEMORY;
            goto out;
        }
        u->end = end;
        u->next = v;
        ret = tree_sequence_builder_add_child(self, u, child);
        if (ret != 0) {
            goto out;
        }
    }
out:
    return ret;
}

static inline int
tree_sequence_builder_add_gap(tree_sequence_builder_t *self, site_id_t start,
        site_id_t end, ancestor_id_t child)
{
    return tree_sequence_builder_add_mapping(self, start, end, 0, child);
}

int
tree_sequence_builder_update(tree_sequence_builder_t *self, ancestor_id_t child,
        allele_t *haplotype, site_id_t start_site, site_id_t end_site,
        ancestor_id_t end_site_parent, traceback_t *traceback)
{
    int ret = 0;
    site_id_t l;
    site_id_t end = end_site;
    ancestor_id_t p;
    ancestor_id_t parent = end_site_parent;
    allele_t state;
    segment_t *u;

    for (l = end_site - 1; l > start_site; l--) {
        /* printf("Tracing back at site %d\n", l); */
        /* print_segment_chain(T_head[l], 1, stdout); */
        /* printf("\n"); */
        ret = ancestor_store_get_state(self->store, l, parent, &state);
        if (ret != 0) {
            goto out;
        }
        if (state != haplotype[l]) {
            /* mutation_sites[local_num_mutations] = l; */
            /* local_num_mutations++; */
        }
        p = -1;
        u = traceback->sites_head[l];
        while (u != NULL) {
            if (u->start <= parent && parent < u->end) {
                p = (ancestor_id_t) u->value;
                break;
            }
            if (u->start > parent) {
                break;
            }
            u = u->next;
        }
        if (p != (ancestor_id_t) -1) {
            assert(l < end);
            ret = tree_sequence_builder_add_mapping(self, l, end, parent, child);
            if (ret != 0) {
                goto out;
            }
            end = l;
            parent = p;
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
        /* mutation_sites[local_num_mutations] = l; */
        /* local_num_mutations++; */
    }
    if (start_site != 0) {
        ret = tree_sequence_builder_add_gap(self, 0, start_site, child);
        if (ret != 0) {
            goto out;
        }
    }
    if (end_site != self->num_sites) {
        ret = tree_sequence_builder_add_gap(self, end_site, self->num_sites, child);
        if (ret != 0) {
            goto out;
        }
    }
out:
    return ret;
}

int
tree_sequence_builder_dump_edgesets(tree_sequence_builder_t *self,
        double *left, double *right, ancestor_id_t *parent, ancestor_id_t *children,
        uint32_t *children_length)
{
    int ret = 0;
    ancestor_id_t j;
    size_t num_edgesets = 0;
    size_t num_children = 0;
    list_segment_t *list_node;
    child_list_node_t *u;

    for (j = self->num_ancestors - 1; j >= 0; j--) {
        for (list_node = self->children[j]; list_node != NULL;
                list_node = list_node->next) {
            if (list_node->head != NULL) {
                left[num_edgesets] = list_node->start;
                right[num_edgesets] = list_node->end;
                parent[num_edgesets] = j;
                children_length[num_edgesets] = 0;
                for (u = list_node->head; u != NULL; u = u->next) {
                    children[num_children] = u->node;
                    children_length[num_edgesets]++;
                    num_children++;
                }
                num_edgesets++;
                assert(num_edgesets <= self->num_edgesets);
            }
        }
    }
    assert(num_edgesets == self->num_edgesets);
    assert(num_children == self->num_children);
    return ret;
}
