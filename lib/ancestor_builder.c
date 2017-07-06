#include "tsinfer.h"
#include "err.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#include "uthash.h"

typedef struct {
    allele_t *state;
    site_id_t id;
    size_t num_samples;
} site_equality_t;

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

static int
cmp_site_equality(const void *a, const void *b) {
    const site_equality_t *ia = (site_equality_t const *) a;
    const site_equality_t *ib = (site_equality_t const *) b;
    int ret = memcmp(ia->state, ib->state, ia->num_samples * sizeof(allele_t));
    if (ret == 0) {
        /* break ties by site_id */
        ret = (ia->id > ib->id) - (ia->id < ib->id);
    }
    return ret;
}


static int
ancestor_builder_compute_focal_sites(ancestor_builder_t *self,
        frequency_class_t *frequency_class)
{
    int ret = 0;
    size_t j, k, site_id;
    site_id_t first_site;
    allele_t *sites = NULL;
    allele_t *site;
    site_equality_t *ordered_sites = NULL;

    sites = malloc(frequency_class->num_sites * self->num_samples * sizeof(allele_t));
    ordered_sites = malloc(frequency_class->num_sites * sizeof(site_equality_t));
    /* Note that this is slightly inefficient use of memory here on average,
     * as we always allocate enough space to allow for each site to be a unique
     * ancestor. The total wastage is likely quite small though */
    frequency_class->ancestor_focal_sites = malloc(
            frequency_class->num_sites * sizeof(site_id_t *));
    frequency_class->num_ancestor_focal_sites = malloc(
            frequency_class->num_sites * sizeof(size_t));
    frequency_class->ancestor_focal_site_mem = malloc(
            frequency_class->num_sites * sizeof(site_id_t));
    if (sites == NULL || ordered_sites == NULL
            || frequency_class->ancestor_focal_sites == NULL
            || frequency_class->num_ancestor_focal_sites == NULL
            || frequency_class->ancestor_focal_site_mem == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }

    /* printf("FREQ CLASS %d\n", (int) frequency_class->frequency); */
    for (j = 0; j < frequency_class->num_sites; j++) {
        site_id = frequency_class->sites[j]->id;
        site = sites + j * self->num_samples;
        ordered_sites[j].id = site_id;
        ordered_sites[j].state = site;
        ordered_sites[j].num_samples = self->num_samples;
        for (k = 0; k < self->num_samples; k++) {
            site[k] = self->haplotypes[k * self->num_sites + site_id];
        }

        /* printf("\tsite=%d\t", (int) site_id); */
        /* for (k = 0; k < self->num_samples; k++) { */
        /*     printf("%d", site[k]); */
        /* } */
        /* printf("\n"); */
    }
    /* NOTE: we should really be doing this by using a hash table using the
     * site states as keys. Only using this sorting algorighm because I couldn't
     * get this to work quickly with uthash */
    qsort(ordered_sites, frequency_class->num_sites, sizeof(site_equality_t),
            cmp_site_equality);

    /* printf("DONE\n"); */
    frequency_class->num_ancestors = 0;
    first_site = 0;
    frequency_class->ancestor_focal_site_mem[0] = ordered_sites[0].id;
    frequency_class->ancestor_focal_sites[0] = frequency_class->ancestor_focal_site_mem;
    frequency_class->num_ancestor_focal_sites[0] = 0;
    for (j = 1; j < frequency_class->num_sites; j++) {
        frequency_class->ancestor_focal_site_mem[j] = ordered_sites[j].id;
        frequency_class->num_ancestor_focal_sites[frequency_class->num_ancestors]++;
        if (memcmp(ordered_sites[first_site].state, ordered_sites[j].state,
                    self->num_samples * sizeof(allele_t)) != 0) {
            first_site = j;
            frequency_class->num_ancestors++;
            frequency_class->num_ancestor_focal_sites[frequency_class->num_ancestors] = 0;
            frequency_class->ancestor_focal_sites[frequency_class->num_ancestors] =
                frequency_class->ancestor_focal_site_mem + j;
            /* printf("BREAK\n"); */
        }

/*         printf("\tsite=%d\t", (int) ordered_sites[j].id); */
/*         site = ordered_sites[j].state; */
/*         for (k = 0; k < self->num_samples; k++) { */
/*             printf("%d", site[k]); */
/*         } */
/*         printf("\n"); */

    }
    frequency_class->num_ancestor_focal_sites[frequency_class->num_ancestors]++;
    frequency_class->num_ancestors++;
out:
    tsi_safe_free(sites);
    tsi_safe_free(ordered_sites);
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
    for (j = 0; j < self->num_frequency_classes; j++) {
        ret = ancestor_builder_compute_focal_sites(self, self->frequency_classes + j);
        if (ret != 0) {
            goto out;
        }
    }
out:
    return ret;
}

int
ancestor_builder_free(ancestor_builder_t *self)
{
    size_t j;

    for (j = 0; j < self->num_frequency_classes; j++) {
        tsi_safe_free(self->frequency_classes[j].ancestor_focal_sites);
        tsi_safe_free(self->frequency_classes[j].num_ancestor_focal_sites);
        tsi_safe_free(self->frequency_classes[j].ancestor_focal_site_mem);
    }
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
        site_id_t site_id, bool remove_inconsistent,
        ancestor_id_hash_t **consistent_samples, allele_t *ancestor)
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
        if (remove_inconsistent) {
            HASH_ITER(hh, *consistent_samples, s, tmp) {
                if (self->haplotypes[s->value * num_sites + site_id] != ancestor[site_id]) {
                    HASH_DEL(*consistent_samples, s);
                }
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

/* Build the ancestors for sites in the specified focal sites */
int
ancestor_builder_make_ancestor(ancestor_builder_t *self, size_t num_focal_sites,
        site_id_t *focal_sites, allele_t *ancestor)
{
    int ret = 0;
    int64_t l;
    site_id_t focal_site;
    size_t j, k;
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
    assert(num_focal_sites > 0);
    /* printf("FOCAL SITES:"); */
    for (j = 0; j < num_focal_sites; j++) {
        /* printf("%d, ", focal_sites[j]); */
        assert(focal_sites[j] < self->num_sites);
        assert(self->sites[focal_sites[j]].frequency > 1);
        if (j > 0) {
            assert(focal_sites[j - 1] < focal_sites[j]);
        }
    }
    /* printf("\n"); */

    memset(ancestor, 0xff, num_sites * sizeof(allele_t));
    /* Fill in the sites within the bounds of the focal sites */
    ancestor_builder_get_consistent_samples(self, focal_sites[0], &consistent_samples,
            consistent_samples_mem, &num_consistent_samples);
    ancestor[focal_sites[0]] = 1;
    for (j = 1; j < num_focal_sites; j++) {
        for (k = focal_sites[j - 1] + 1; k < focal_sites[j]; k++) {
            ancestor_builder_make_site(self, focal_sites[j], k, false,
                    &consistent_samples, ancestor);
        }
        /* printf("Setting %d: %d\n", focal_sites[j], HASH_COUNT(consistent_samples)); */
        ancestor[focal_sites[j]] = 1;
    }
    /* printf("DONE INTER\n"); */
    /* fflush(stdout); */

    /* Work leftwards from the first focal site */
    focal_site = focal_sites[0];
    /* printf("focal site = %d\n", focal_site_id); */
    for (l = ((int64_t) focal_site) - 1; l >= 0
            && HASH_COUNT(consistent_samples) > 1; l--) {
        /* printf("LEFT: l = %d, count = %d\n", (int) l, HASH_COUNT(consistent_samples)); */
        ancestor_builder_make_site(self, focal_site, l, true, &consistent_samples, ancestor);
    }
    HASH_CLEAR(hh, consistent_samples);

    /* Work rightwards from the last focal site */
    focal_site = focal_sites[num_focal_sites - 1];
    ancestor_builder_get_consistent_samples(self, focal_site, &consistent_samples,
            consistent_samples_mem, &num_consistent_samples);
    for (l = focal_site + 1; l < (int64_t) num_sites
            && HASH_COUNT(consistent_samples) > 1; l++) {
        /* printf("RIGHT: l = %d, count = %d\n", (int) l, HASH_COUNT(consistent_samples)); */
        ancestor_builder_make_site(self, focal_site, l, true, &consistent_samples, ancestor);
    }
    HASH_CLEAR(hh, consistent_samples);
out:
    tsi_safe_free(consistent_samples_mem);
    return ret;
}

static void
ancestor_builder_check_state(ancestor_builder_t *self)
{
    size_t j, k, l, sum;
    frequency_class_t *fq;

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
    /* Check the ancestor_focal_sites */
    for (j = 0; j < self->num_frequency_classes; j++) {
        fq = self->frequency_classes + j;
        sum = 0;
        for (k = 0; k < fq->num_ancestors; k++) {
            sum += fq->num_ancestor_focal_sites[k];
        }
        if (sum != fq->num_sites) {
            printf("ERROR: j = %d, k = %d, sum= %d, num_sites=  %d\n",
                    (int) j, (int) k, (int) sum, (int) fq->num_sites);

        }
        assert(sum == fq->num_sites);
    }
}

int
ancestor_builder_print_state(ancestor_builder_t *self, FILE *out)
{
    size_t j, k, l;
    frequency_class_t *fq;

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
    fprintf(out, "Sites:\n");
    for (j = 0; j < self->num_sites; j++) {
        fprintf(out, "\t%d\t%f\t%d\n", self->sites[j].id, self->sites[j].position,
                (int) self->sites[j].frequency);
    }
    fprintf(out, "Frequency classes\n");
    for (j = 0; j < self->num_frequency_classes; j++) {
        fq = self->frequency_classes + j;
        fprintf(out, "\t%d\tfreq=%d\tnum_sites=%d\tnum_ancestors=%d\n", (int) j,
                (int) fq->frequency, (int) fq->num_sites, (int) fq->num_ancestors);
        for (k = 0; k < fq->num_ancestors; k++) {
            fprintf(out, "\t\t%d [%d]:\t(", (int) k, (int) fq->num_ancestor_focal_sites[k]);
            for (l = 0; l < fq->num_ancestor_focal_sites[k]; l++) {
                fprintf(out, "%d,", fq->ancestor_focal_sites[k][l]);
            }
            fprintf(out, ")\n");
        }
    }
    ancestor_builder_check_state(self);
    return 0;
}
