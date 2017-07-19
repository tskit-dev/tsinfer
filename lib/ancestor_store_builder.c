#include "tsinfer.h"
#include "err.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

static void
print_segment_chain(node_segment_list_node_t *head, FILE *out)
{
    node_segment_list_node_t *u = head;

    while (u != NULL) {
        fprintf(out, "(%d-%d)", u->start, u->end);
        u = u->next;
        if (u != NULL) {
            fprintf(out, "=>");
        }
    }
}

static void
ancestor_store_builder_check_state(ancestor_store_builder_t *self)
{
    site_id_t l;
    node_segment_list_node_t *u, *last_u;
    size_t total_segments = 0;

    for (l = 0; l < self->num_sites; l++) {
        last_u = NULL;
        u = self->sites_head[l];
        while (u != NULL) {
            assert(u->start < u->end);
            if (u->next != NULL) {
                assert(u->next->start >= u->end);
            }
            last_u = u;
            u = u->next;
            total_segments++;
        }
        assert(self->sites_tail[l] == last_u);
    }
    assert(total_segments == object_heap_get_num_allocated(&self->segment_heap));
    assert(total_segments == self->total_segments);
}

int
ancestor_store_builder_print_state(ancestor_store_builder_t *self, FILE *out)
{
    site_id_t l;
    node_segment_list_node_t *seg;
    int num_segments;

    fprintf(out, "Ancestor store builder state\n");
    fprintf(out, "num_sites = %d\n", (int) self->num_sites);
    fprintf(out, "num_ancestors = %d\n", (int) self->num_ancestors);
    fprintf(out, "total_segments = %d\n", (int) self->total_segments);
    fprintf(out, "Segment heap:\n");
    object_heap_print_state(&self->segment_heap, out);
    fprintf(out, "Sites:\n");
    for (l = 0; l < self->num_sites; l++) {
        num_segments = 0;
        seg = self->sites_head[l];
        while (seg != NULL) {
            seg = seg->next;
            num_segments++;
        }
        fprintf(out, "%d\t(%d):", l, num_segments);
        print_segment_chain(self->sites_head[l], out);
        fprintf(out, "\n");
    }

    ancestor_store_builder_check_state(self);
    return 0;
}

static inline node_segment_list_node_t * WARN_UNUSED
ancestor_store_builder_alloc_segment(ancestor_store_builder_t *self, ancestor_id_t start,
        ancestor_id_t end)
{
    node_segment_list_node_t *ret = NULL;

    if (object_heap_empty(&self->segment_heap)) {
        if (object_heap_expand(&self->segment_heap) != 0) {
            goto out;
        }
    }
    ret = (node_segment_list_node_t *) object_heap_alloc_object(&self->segment_heap);
    if (ret == NULL) {
        goto out;
    }
    ret->start = start;
    ret->end = end;
    ret->next = NULL;
    self->total_segments++;
out:
    return ret;
}

int
ancestor_store_builder_alloc(ancestor_store_builder_t *self, size_t num_sites,
        size_t segment_block_size)
{
    int ret = 0;

    memset(self, 0, sizeof(ancestor_store_builder_t));
    self->num_sites = num_sites;
    self->num_ancestors = 1;
    self->sites_head = calloc(self->num_sites, sizeof(node_segment_list_node_t *));
    self->sites_tail = calloc(self->num_sites, sizeof(node_segment_list_node_t *));
    if (self->sites_head == NULL || self->sites_tail == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;

    }
    ret = object_heap_init(&self->segment_heap, sizeof(node_segment_list_node_t),
           segment_block_size, NULL);
    if (ret != 0) {
        goto out;
    }
    self->total_segments = 0;
out:
    return ret;
}

int
ancestor_store_builder_free(ancestor_store_builder_t *self)
{
    tsi_safe_free(self->sites_head);
    tsi_safe_free(self->sites_tail);
    object_heap_free(&self->segment_heap);

    return 0;
}

int
ancestor_store_builder_add(ancestor_store_builder_t *self, allele_t *ancestor)
{
    int ret = 0;
    site_id_t l;
    node_segment_list_node_t *u;
    ancestor_id_t n = (ancestor_id_t) self->num_ancestors;
    allele_t state;

    for (l = 0; l < self->num_sites; l++) {
        state = ancestor[l];
        if (self->sites_head[l] == NULL) {
            if (state == 1) {
                u = ancestor_store_builder_alloc_segment(self, n, n + 1);
                if (u == NULL) {
                    ret = TSI_ERR_NO_MEMORY;
                    goto out;
                }
                self->sites_head[l] = u;
                self->sites_tail[l] = u;
            }
        } else {
            if (state == 1) {
                if (self->sites_tail[l]->end == n) {
                    self->sites_tail[l]->end = n + 1;
                } else {
                    u = ancestor_store_builder_alloc_segment(self, n, n + 1);
                    if (u == NULL) {
                        ret = TSI_ERR_NO_MEMORY;
                        goto out;
                    }
                    self->sites_tail[l]->next = u;
                    self->sites_tail[l] = u;
                }
            }
        }
    }
    self->num_ancestors++;
out:
    return ret;
}

int
ancestor_store_builder_dump(ancestor_store_builder_t *self, site_id_t *site, ancestor_id_t *start,
        ancestor_id_t *end)
{
    int ret = 0;
    site_id_t l;
    node_segment_list_node_t *u;
    size_t k;

    k = 0;
    for (l = 0; l < self->num_sites; l++) {
        u = self->sites_head[l];
        while (u != NULL) {
            assert(k < self->total_segments);
            site[k] = l;
            start[k] = u->start;
            end[k] = u->end;
            u = u->next;
            k++;
        }
    }
    return ret;
}
