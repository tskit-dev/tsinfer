
#include "ls.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    size_t index;
    size_t frequency;
} site_t;

/* Order site by reverse frequency and index */
static int
cmp_site(const void *a, const void *b) {
    const site_t *ca = (const site_t *) a;
    const site_t *cb = (const site_t *) b;
    int ret = (ca->frequency < cb->frequency) - (ca->frequency > cb->frequency);
    if (ret == 0) {
        ret = (ca->index > cb->index) - (ca->index < cb->index);
    }
    return ret;
}

static int
reference_panel_infer_ancestors(reference_panel_t *self, size_t num_sites, site_t *sites)
{
    int ret = -1;
    size_t j, k, s, samples_with_mutation, total_samples;
    size_t N = self->num_haplotypes;
    size_t m = self->num_sites;
    size_t n = self->num_samples;
    site_t site_j, site_k;

    qsort(sites, num_sites, sizeof(site_t), cmp_site);
    /* printf("%d non singleton sites:\n", (int) num_sites); */
    /* for (j = 0; j < num_sites; j++) { */
    /*     printf("\t%d\t%d\n", (int) sites[j].index, (int) sites[j].frequency); */
    /* } */
    for (j = 0; j < num_sites; j++) {
        site_j = sites[j];
        self->haplotypes[(N - j - 2) * m + site_j.index] = 1;
        for (k = 0; k < j; k++) {
            site_k = sites[k];
            total_samples = 0;
            samples_with_mutation = 0;
            for (s = 0; s < n; s++) {
                if (self->haplotypes[s * m + site_j.index] == 1) {
                    samples_with_mutation += self->haplotypes[s * m + site_k.index];
                    total_samples++;
                }
            }
            if (samples_with_mutation >= total_samples / 2.0) {
                self->haplotypes[(N - j - 2) * m + site_k.index] = 1;
            }
        }
    }
    ret = 0;
    return ret;
}

int
reference_panel_alloc(reference_panel_t *self, size_t num_samples, size_t num_sites,
        uint8_t *sample_haplotypes)
{
    int ret = -1;
    size_t j, l, freq;
    size_t num_non_singleton_sites = 0;
    site_t *non_singleton_sites = NULL;

    memset(self, 0, sizeof(reference_panel_t));
    self->num_samples = num_samples;
    self->num_sites = num_sites;
    self->num_haplotypes = num_samples + 1;

    non_singleton_sites = malloc(num_sites * sizeof(site_t));
    if (non_singleton_sites == NULL) {
        ret = -1;
        goto out;
    }
    /* Count the number of set bits in columns that are not singletons */
    for (l = 0; l < num_sites; l++) {
        freq = 0;
        for (j = 0; j < num_samples; j++) {
            freq += sample_haplotypes[j * num_sites + l];
        }
        if (freq > 1) {
            non_singleton_sites[num_non_singleton_sites].index = l;
            non_singleton_sites[num_non_singleton_sites].frequency = freq;
            num_non_singleton_sites++;
        }
    }
    self->num_haplotypes += num_non_singleton_sites;
    self->haplotypes = calloc(self->num_haplotypes * num_sites, sizeof(uint8_t));
    if (self->haplotypes == NULL) {
        ret = -1;
        goto out;
    }
    /* Copy the sample haplotypes in */
    memcpy(self->haplotypes, sample_haplotypes,
            num_samples * num_sites * sizeof(uint8_t));
    ret = reference_panel_infer_ancestors(self, num_non_singleton_sites,
            non_singleton_sites);
out:
    if (non_singleton_sites != NULL) {
        free(non_singleton_sites);
    }
    return ret;
}

int
reference_panel_free(reference_panel_t *self)
{
    if (self->haplotypes != NULL) {
        free(self->haplotypes);
    }
    return 0;
}

int
reference_panel_print_state(reference_panel_t *self)
{
    uint32_t l, j;

    printf("Reference panel n = %d, m = %d\n", (int) self->num_haplotypes,
            (int) self->num_sites);
    for (j = 0; j < self->num_haplotypes; j++) {
        printf("%3d\t", j);
        for (l = 0; l < self->num_sites; l++) {
            printf("%d", self->haplotypes[j * self->num_sites + l]);
        }
        printf("\n");
    }
    return 0;
}

int
threader_alloc(threader_t *self, reference_panel_t *reference_panel)
{
    int ret = 0;
    uint32_t n = reference_panel->num_haplotypes;
    uint32_t m = reference_panel->num_sites;

    memset(self, 0, sizeof(threader_t));
    self->reference_panel = reference_panel;
    self->V = malloc(n * sizeof(double));
    self->V_next = malloc(n * sizeof(double));
    self->T = malloc(n * m * sizeof(uint32_t));
    if (self->V == NULL || self->V_next == NULL || self->T == NULL) {
        ret = -2;
        goto out;
    }
out:
    return ret;
}

int
threader_free(threader_t *self)
{
    if (self->V != NULL) {
        free(self->V);
    }
    if (self->V_next != NULL) {
        free(self->V_next);
    }
    if (self->T != NULL) {
        free(self->T);
    }
    return 0;
}

int
threader_run(threader_t *self,
        uint32_t haplotype_index, uint32_t panel_size, double recombination_rate,
        uint32_t *P)
{
    int ret = 0;
    const uint32_t n = panel_size;
    const uint32_t N = self->reference_panel->num_haplotypes;
    const uint32_t m = self->reference_panel->num_sites;
    const double log_1_n = log(1.0 / n);
    const double r = 1 - exp(-recombination_rate / n);
    const double recomb_proba = log(r / panel_size);
    const double no_recomb_proba = log(1 - r + r / n);
    /* The emission matrix for observed haplotypes and the matched
     * references. E[1][0] is the probability of seeing a zero in the
     * observed haplotype when there is a 1 in the panel. We have
     * a small but finite probability of changing from 0->1. This mutation
     * probablity must be less than many recombinations, or we may have
     * multiple mutations at the same locus.
     */
    const double mutation_proba = m * recomb_proba;
    const double E[2][2] = {
        {0, mutation_proba},
        {-DBL_MAX, 0}
    };
    /* Local working storage */
    const uint8_t *H = self->reference_panel->haplotypes;
    const uint8_t *h = &self->reference_panel->haplotypes[haplotype_index * m];
    double *V = self->V;
    double *V_next = self->V_next;
    uint32_t *T = self->T;
    uint32_t j, l, max_k, max_V_index, max_V_next_index;
    double max_p, x, y, *tmp;

    if (haplotype_index >= N) {
        ret = -3;
        goto out;
    }
    y = 0;
    max_V_index = N - n;

    for (j = N - n; j < N; j++) {
        V[j] = log_1_n + E[H[j * m]][h[0]];
        if (V[j] > V[max_V_index]) {
            max_V_index = j;
        }
    }
    for (l = 1; l < m; l++) {
        max_V_next_index = N - n;
        for (j = N - n; j < N; j++) {
            x = V[j] + no_recomb_proba;
            y = V[max_V_index] + recomb_proba;
            if (x > y) {
                max_p = x;
                max_k = j;
            } else {
                max_k = max_V_index;
                max_p = y;
            }
            T[j * m + l] = max_k;
            V_next[j] = max_p + E[H[j * m + l]][h[l]];
            if (V_next[j] > V_next[max_V_next_index]) {
                max_V_next_index = j;
            }
        }
        /* swap V and V_next */
        tmp = V;
        V = V_next;
        V_next = tmp;
        max_V_index = max_V_next_index;
    }
    P[m - 1] = max_V_index;
    for (l = m - 1; l > 0; l--) {
        P[l - 1] = T[P[l] * m + l];
    }
out:
    return ret;
}
