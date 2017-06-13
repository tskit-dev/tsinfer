#include "tsinfer.h"
#include "err.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>


int
ancestor_builder_alloc(ancestor_builder_t *self, size_t num_samples,
        size_t num_sites, allele_t *haplotypes)
{
    int ret = 0;

    // TODO error checking

    self->num_samples = num_samples;
    self->num_sites = num_sites;
    self->haplotypes = haplotypes;

    return ret;
}

int
ancestor_builder_free(ancestor_builder_t *self)
{
    return 0;
}
