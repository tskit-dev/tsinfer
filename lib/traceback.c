#include "tsinfer.h"
#include "err.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>


int
traceback_print_state(traceback_t *self, FILE *out)
{
    size_t l;
    node_segment_list_node_t *v;

    fprintf(out, "Traceback\n");
    fprintf(out, "num_sites = %d\n", (int) self->num_sites);
    block_allocator_print_state(&self->allocator, out);
    for (l = 0; l < self->num_sites; l++) {
        v = self->sites_head[l];
        if (v != NULL) {
            fprintf(out, "%d\t%d\t", (int) l, self->best_match[l]);
            while (v != NULL) {
                fprintf(out, "(%d,%d)", (int) v->start, (int) v->end);
                v = v->next;
            }
            fprintf(out, "\n");
        }
    }
    return 0;
}

int
traceback_alloc(traceback_t *self, size_t num_sites, size_t segment_block_size)
{
    int ret = 0;

    memset(self, 0, sizeof(traceback_t));
    self->num_sites = num_sites;
    self->segment_block_size = segment_block_size;
    ret = block_allocator_alloc(&self->allocator, self->segment_block_size);
    if (ret != 0) {
        goto out;
    }
    self->best_match = malloc(self->num_sites * sizeof(ancestor_id_t));
    self->sites_head = calloc(self->num_sites, sizeof(node_segment_list_node_t *));
    self->sites_tail = calloc(self->num_sites, sizeof(node_segment_list_node_t *));
    if (self->best_match == NULL
            || self->sites_head == NULL || self->sites_tail == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
out:
    return ret;
}

int
traceback_free(traceback_t *self)
{
    block_allocator_free(&self->allocator);
    tsi_safe_free(self->best_match);
    tsi_safe_free(self->sites_head);
    tsi_safe_free(self->sites_tail);
    return 0;
}

static inline node_segment_list_node_t * WARN_UNUSED
traceback_alloc_segment(traceback_t *self, ancestor_id_t start, ancestor_id_t end)
{
    node_segment_list_node_t *ret = NULL;

    ret = block_allocator_get(&self->allocator, sizeof(* ret));
    if (ret == NULL) {
        goto out;
    }
    ret->start = start;
    ret->end = end;
    ret->next = NULL;
out:
    return ret;
}

int
traceback_reset(traceback_t *self)
{
    size_t l;

    for (l = 0; l < self->num_sites; l++) {
        self->sites_head[l] = NULL;
        self->sites_tail[l] = NULL;
    }
    block_allocator_reset(&self->allocator);
    return 0;
}

int
traceback_set_best_match(traceback_t *self, site_id_t site_id, ancestor_id_t best_match)
{
    self->best_match[site_id] = best_match;
    return 0;
}

int
traceback_add_recombination(traceback_t *self, site_id_t site_id, ancestor_id_t start,
        ancestor_id_t end)
{
    int ret = 0;
    node_segment_list_node_t **head = self->sites_head;
    node_segment_list_node_t **tail = self->sites_tail;
    node_segment_list_node_t *tmp;
    assert(head != NULL);
    assert(tail != NULL);
    assert(site_id < self->num_sites);

    if (head[site_id] == NULL) {
        head[site_id] = traceback_alloc_segment(self, start, end);
        if (head[site_id] == NULL) {
            ret = TSI_ERR_NO_MEMORY;
            goto out;
        }
        tail[site_id] = head[site_id];
    } else {
        assert(tail[site_id] != NULL);
        if (tail[site_id]->end == start) {
            tail[site_id]->end = end;
        } else {
            tmp = traceback_alloc_segment(self, start, end);
            if (tmp == NULL) {
                ret = TSI_ERR_NO_MEMORY;
                goto out;
            }
            tail[site_id]->next = tmp;
            tail[site_id] = tmp;
        }
    }
out:
    return ret;
}
