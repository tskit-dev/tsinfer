#include "tsinfer.h"
#include "err.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>


static int
cmp_site_by_frequency(const void *a, const void *b) {
    const site_t *ia = (const site_t *) a;
    const site_t *ib = (const site_t *) b;
    return (ia->frequency < ib->frequency) - (ia->frequency > ib->frequency);
}


int
ancestor_builder_alloc(ancestor_builder_t *self, size_t num_samples,
        size_t num_sites, double *positions, allele_t *haplotypes)
{
    int ret = 0;
    size_t j, k, l, frequency;
    // TODO error checking

    memset(self, 0, sizeof(ancestor_builder_t));
    self->num_samples = num_samples;
    self->num_sites = num_sites;
    self->haplotypes = haplotypes;
    self->sites = calloc(num_sites, sizeof(site_t));

    if (self->sites == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }

    for (l = 0; l < self->num_sites; l++) {
        self->sites[l].id = (site_id_t) l;
        self->sites[l].position = positions[l];
        self->sites[l].frequency = 0;
    }
    /* Compute the site frequency */
    for (j = 0; j < self->num_samples; j++) {
        for (l = 0; l < self->num_sites; l++) {
            self->sites[l].frequency += (size_t) (
                    self->haplotypes[j * self->num_sites + l] == 1);
        }
    }
    qsort(self->sites, self->num_sites, sizeof(site_t), cmp_site_by_frequency);

    /* compute the number of frequency classes */
    self->num_frequency_classes = 0;
    frequency = self->num_samples + 1;
    for (j = 0; j < self->num_sites && self->sites[j].frequency > 1; j++) {
        if (self->sites[j].frequency != frequency) {
            frequency = self->sites[j].frequency;
            self->num_frequency_classes++;
        }
    }
    self->frequency_classes = calloc(self->num_frequency_classes, sizeof(frequency_class_t));
    if (self->frequency_classes == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    frequency = self->num_samples + 1;
    k = SIZE_MAX;
    for (j = 0; j < self->num_sites && self->sites[j].frequency > 1; j++) {
        if (self->sites[j].frequency != frequency) {
            k++;
            frequency = self->sites[j].frequency;
            self->frequency_classes[k].num_sites = 1;
            self->frequency_classes[k].sites = self->sites + j;
            self->frequency_classes[k].frequency = frequency;
        } else {
            self->frequency_classes[k].num_sites++;
        }
    }
out:
    return ret;
}

int
ancestor_builder_free(ancestor_builder_t *self)
{
    tsi_safe_free(self->sites);
    tsi_safe_free(self->frequency_classes);
    return 0;
}

/* Build the ancestors for sites in the specified frequency class */
int
ancestor_builder_make_ancestors(ancestor_builder_t *self, size_t frequency_class,
        allele_t *ancestors)
{
    int ret = 0;

    assert(frequency_class < self->num_frequency_classes);

    return ret;
}

static void
ancestor_builder_check_state(ancestor_builder_t *self)
{
    size_t j, k, l;

    l = 0;
    for (j = 0; j < self->num_frequency_classes; j++) {
        assert(self->frequency_classes[j].sites != NULL);
        for (k = 0; k < self->frequency_classes[j].num_sites; k++) {
            /* Ensure that we have correctly partitioned the site pointers */
            assert(self->frequency_classes[j].sites + k == self->sites + l);
            l++;
            assert(self->frequency_classes[j].frequency ==
                    self->frequency_classes[j].sites[k].frequency);
        }
    }
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
    printf("Sites:\n");
    for (j = 0; j < self->num_sites; j++) {
        printf("\t%d\t%f\t%d\n", self->sites[j].id, self->sites[j].position,
                (int) self->sites[j].frequency);
    }
    printf("Frequency classes\n");
    for (j = 0; j < self->num_frequency_classes; j++) {
        printf("\t%d\t%d\t%d\n", (int) j, (int) self->frequency_classes[j].frequency,
                (int) self->frequency_classes[j].num_sites);
    }
    ancestor_builder_check_state(self);
    return 0;
}
