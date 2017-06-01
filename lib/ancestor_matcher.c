#include "tsinfer.h"
#include "err.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>


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


static void
print_segment_chain(segment_t *head, FILE *out)
{
    segment_t *u = head;

    while (u != NULL) {
        fprintf(out, "(%d-%d:%f)", u->start, u->end, u->value);
        u = u->next;
        if (u != NULL) {
            fprintf(out, "=>");
        }
    }
}

static inline segment_t * WARN_UNUSED
ancestor_matcher_alloc_segment(ancestor_matcher_t *self, site_id_t start,
        site_id_t end, double value)
{
    segment_t *ret = NULL;

    if (object_heap_empty(&self->segment_heap)) {
        if (object_heap_expand(&self->segment_heap) != 0) {
            goto out;
        }
    }
    ret = (segment_t *) object_heap_alloc_object(&self->segment_heap);
    ret->start = start;
    ret->end = end;
    ret->value = value;
    ret->next = NULL;
out:
    return ret;
}

/* static inline void */
/* ancestor_matcher_free_segment(ancestor_matcher_t *self, segment_t *seg) */
/* { */
/*     object_heap_free_object(&self->segment_heap, seg); */
/* } */

int
ancestor_matcher_alloc(ancestor_matcher_t *self, size_t num_sites,
        size_t segment_block_size)
{
    int ret = 0;
    site_id_t l;

    memset(self, 0, sizeof(ancestor_matcher_t));
    self->num_sites = num_sites;
    self->num_ancestors = 1;
    self->sites_head = malloc(self->num_sites * sizeof(segment_t *));
    self->sites_tail = malloc(self->num_sites * sizeof(segment_t *));
    if (self->sites_head == NULL || self->sites_tail == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;

    }
    ret = object_heap_init(&self->segment_heap, sizeof(segment_t),
           segment_block_size, NULL);
    if (ret != 0) {
        goto out;
    }
    for (l = 0; l < self->num_sites; l++) {
        self->sites_head[l] = ancestor_matcher_alloc_segment(self, 0, 1, 0);
        if (self->sites_head[l] == NULL) {
            ret = TSI_ERR_NO_MEMORY;
            goto out;
        }
        self->sites_tail[l] = self->sites_head[l];
    }

out:
    return ret;
}

int
ancestor_matcher_free(ancestor_matcher_t *self)
{
    tsi_safe_free(self->sites_head);
    tsi_safe_free(self->sites_tail);
    object_heap_free(&self->segment_heap);

    return 0;
}

int
ancestor_matcher_add(ancestor_matcher_t *self, allele_t *haplotype)
{
    int ret = 0;

    return ret;
}

int
ancestor_matcher_print_state(ancestor_matcher_t *self, FILE *out)
{

    site_id_t l;

    fprintf(out, "Ancestor matcher state\n");
    fprintf(out, "num_sites = %d\n", (int) self->num_sites);
    fprintf(out, "num_ancestors = %d\n", (int) self->num_ancestors);

    for (l = 0; l < self->num_sites; l++) {
        fprintf(out, "%d\t:", l);
        print_segment_chain(self->sites_head[l], out);
        fprintf(out, "\n");
    }
    return 0;
}
