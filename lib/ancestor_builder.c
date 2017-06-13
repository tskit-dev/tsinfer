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



int
ancestor_builder_print_state(ancestor_builder_t *self, FILE *out)
{
    size_t j, k;

    fprintf(out, "Ancestor builder\n");
    fprintf(out, "num_samples = %d\n", (int) self->num_samples);
    fprintf(out, "num_sites = %d\n", (int) self->num_sites);
    fprintf(out, "haplotypes = \n");

    for (j = 0; j < self->num_samples; j++) {
        fprintf(out, "\t");
        for (k = 0; k < self->num_sites; k++) {
            fprintf(out, "%d", self->haplotypes[j * self->num_sites + k]);
        }
        fprintf(out, "\n");
    }
    return 0;
}
