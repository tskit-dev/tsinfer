#include "tsinfer.h"
#include "err.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#include "uthash.h"

static int
cmp_site_by_frequency(const void *a, const void *b) {
    const site_t *ia = *(site_t * const *) a;
    const site_t *ib = *(site_t * const *) b;
    int ret = (ia->frequency < ib->frequency) - (ia->frequency > ib->frequency);
    if (ret == 0) {
        ret = (ia->position > ib->position) - (ia->position < ib->position);
    }
    return ret;
}

int
ancestor_builder_alloc(ancestor_builder_t *self, size_t num_samples,
        size_t num_sites, double *positions, allele_t *haplotypes)
{
    int ret = 0;
    size_t j, k, l, frequency;
    // TODO error checking
    //
    assert(num_samples > 1);
    assert(num_sites > 0);

    memset(self, 0, sizeof(ancestor_builder_t));
    self->num_samples = num_samples;
    self->num_sites = num_sites;
    self->haplotypes = haplotypes;
    self->sites = calloc(num_sites, sizeof(site_t));
    self->sorted_sites = malloc(num_sites * sizeof(site_t *));

    if (self->sites == NULL || self->sorted_sites == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }

    for (l = 0; l < self->num_sites; l++) {
        self->sites[l].id = (site_id_t) l;
        self->sites[l].position = positions[l];
        self->sites[l].frequency = 0;
        self->sorted_sites[l] = self->sites + l;
    }
    /* Compute the site frequency */
    for (j = 0; j < self->num_samples; j++) {
        for (l = 0; l < self->num_sites; l++) {
            self->sites[l].frequency += (size_t) (
                    self->haplotypes[j * self->num_sites + l] == 1);
        }
    }
    qsort(self->sorted_sites, self->num_sites, sizeof(site_t *), cmp_site_by_frequency);

    /* compute the number of frequency classes */
    self->num_frequency_classes = 0;
    frequency = self->num_samples + 1;
    for (j = 0; j < self->num_sites && self->sorted_sites[j]->frequency > 1; j++) {
        if (self->sorted_sites[j]->frequency != frequency) {
            frequency = self->sorted_sites[j]->frequency;
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
    for (j = 0; j < self->num_sites && self->sorted_sites[j]->frequency > 1; j++) {
        if (self->sorted_sites[j]->frequency != frequency) {
            k++;
            frequency = self->sorted_sites[j]->frequency;
            self->frequency_classes[k].num_sites = 1;
            self->frequency_classes[k].sites = self->sorted_sites + j;
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
    tsi_safe_free(self->sorted_sites);
    tsi_safe_free(self->frequency_classes);
    return 0;
}

typedef struct {
    ancestor_id_t value;
    UT_hash_handle hh;
} ancestor_id_hash_t;


static inline void
ancestor_builder_make_site(ancestor_builder_t *self, site_id_t focal_site_id,
        site_id_t site_id, ancestor_id_hash_t **consistent_samples,
        allele_t *ancestor)
{
    size_t num_sites = self->num_sites;
    size_t ones, zeros;
    site_t focal_site;
    focal_site = self->sites[focal_site_id];
    ancestor_id_hash_t *s, *tmp;

    ancestor[site_id] = 0;
    if (self->sites[site_id].frequency > focal_site.frequency) {
        ones = 0;
        HASH_ITER(hh, *consistent_samples, s, tmp) {
            ones += self->haplotypes[s->value * num_sites + site_id] == 1;
            /* printf("\t\tsample %d\n", s->value); */
        }
        zeros = HASH_COUNT(*consistent_samples) - ones;
        if (ones >= zeros)  {
            ancestor[site_id] = 1;
        }
        /* printf("\t\tExamining site %d: ones=%d, zeros=%d\n", (int) site_id, */
        /*         (int) ones, (int) zeros); */
        HASH_ITER(hh, *consistent_samples, s, tmp) {
            if (self->haplotypes[s->value * num_sites + site_id] != ancestor[site_id]) {
                HASH_DEL(*consistent_samples, s);
            }
        }
    }
}

static inline void
ancestor_builder_get_consistent_samples(ancestor_builder_t *self, site_id_t focal_site_id,
        ancestor_id_hash_t **consistent_samples, ancestor_id_hash_t *consistent_samples_mem,
        size_t *num_consistent_samples)
{
    size_t j, k;
    ancestor_id_hash_t *s;

    k = 0;
    for (j = 0; j < self->num_samples; j++) {
        if (self->haplotypes[j * self->num_sites + focal_site_id] == 1) {
            s = consistent_samples_mem + k;
            k++;
            assert(k <= self->num_samples);
            s->value = (ancestor_id_t) j;
            HASH_ADD(hh, *consistent_samples, value, sizeof(ancestor_id_t), s);
        }
    }
}

/* Build the ancestors for sites in the specified focal site */
int
ancestor_builder_make_ancestor(ancestor_builder_t *self, site_id_t focal_site_id,
        allele_t *ancestor)
{
    int ret = 0;
    int64_t l;
    size_t num_sites = self->num_sites;
    size_t num_consistent_samples;
    ancestor_id_hash_t *consistent_samples = NULL;
    ancestor_id_hash_t *consistent_samples_mem = malloc(
            self->num_samples * sizeof(ancestor_id_hash_t));

    if (consistent_samples_mem == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    // TODO proper error checking.
    assert(focal_site_id < self->num_sites);
    assert(self->sites[focal_site_id].frequency > 1);

    memset(ancestor, 0xff, num_sites * sizeof(allele_t));
    ancestor[focal_site_id] = 1;

    ancestor_builder_get_consistent_samples(self, focal_site_id, &consistent_samples,
            consistent_samples_mem, &num_consistent_samples);
    /* printf("focal site = %d\n", focal_site_id); */
    for (l = ((int64_t) focal_site_id) - 1; l >= 0
            && HASH_COUNT(consistent_samples) > 1; l--) {
        /* printf("LEFT: l = %d, count = %d\n", (int) l, HASH_COUNT(consistent_samples)); */
        ancestor_builder_make_site(self, focal_site_id, l, &consistent_samples, ancestor);
    }
    HASH_CLEAR(hh, consistent_samples);

    ancestor_builder_get_consistent_samples(self, focal_site_id, &consistent_samples,
            consistent_samples_mem, &num_consistent_samples);
    for (l = focal_site_id + 1; l < (int64_t) num_sites
            && HASH_COUNT(consistent_samples) > 1; l++) {
        /* printf("RIGHT: l = %d, count = %d\n", (int) l, HASH_COUNT(consistent_samples)); */
        ancestor_builder_make_site(self, focal_site_id, l, &consistent_samples, ancestor);
    }
    HASH_CLEAR(hh, consistent_samples);
out:
    tsi_safe_free(consistent_samples_mem);
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
            assert(self->frequency_classes[j].sites + k == self->sorted_sites + l);
            l++;
            assert(self->frequency_classes[j].frequency ==
                    self->frequency_classes[j].sites[k]->frequency);
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
