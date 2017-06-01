#include "tsinfer.h"
#include "err.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

int
ancestor_matcher_alloc(ancestor_matcher_t *self, size_t num_sites)
{
    int ret = 0;

    memset(self, 0, sizeof(ancestor_matcher_t));
    self->num_sites = num_sites;
    self->num_ancestors = 1;
    self->sites_head = malloc(self->num_sites * sizeof(segment_t *));
    self->sites_tail = malloc(self->num_sites * sizeof(segment_t *));
    if (self->sites_head == NULL || self->sites_tail == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;

    }

out:
    return ret;
}

int
ancestor_matcher_free(ancestor_matcher_t *self)
{
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

    return 0;
}
