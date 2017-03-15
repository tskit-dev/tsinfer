
#include "ls.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

typedef struct {
    size_t index;
    double position;
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
        uint8_t *sample_haplotypes, double *positions)
{
    int ret = -1;
    size_t j, l, freq, offset, hap_index;
    size_t num_non_singleton_sites = 0;
    site_t *non_singleton_sites = NULL;
    size_t *singleton_sites = NULL;
    size_t *singleton_haplotypes = NULL;
    size_t num_singleton_sites = 0;

    memset(self, 0, sizeof(reference_panel_t));
    self->num_samples = num_samples;
    self->num_sites = num_sites;
    self->num_haplotypes = num_samples + 1;

    non_singleton_sites = malloc(num_sites * sizeof(site_t));
    singleton_sites = malloc(num_sites * sizeof(site_t));
    singleton_haplotypes = malloc(num_sites * sizeof(size_t));
    self->positions = malloc((num_sites) * sizeof(double));
    if (non_singleton_sites == NULL || positions == NULL) {
        ret = -1;
        goto out;
    }
    memcpy(self->positions, positions, num_sites * sizeof(double));
    /* Count the number of set bits in columns that are not singletons */
    for (l = 0; l < num_sites; l++) {
        freq = 0;
        for (j = 0; j < num_samples; j++) {
            if (sample_haplotypes[j * num_sites + l] == 1) {
                freq++;
                singleton_haplotypes[num_singleton_sites] = j;
            }
        }
        if (freq > 1) {
            non_singleton_sites[num_non_singleton_sites].index = l;
            non_singleton_sites[num_non_singleton_sites].frequency = freq;
            num_non_singleton_sites++;
        } else {
            singleton_sites[num_singleton_sites] = l;
            num_singleton_sites++;
        }
    }
    self->num_haplotypes += num_non_singleton_sites;
    self->haplotypes = calloc(self->num_haplotypes * num_sites, sizeof(uint8_t));
    self->new_mutations = malloc(self->num_haplotypes * sizeof(uint32_t *));
    self->num_new_mutations = calloc(self->num_haplotypes, sizeof(uint32_t));
    self->new_mutations_mem = malloc(self->num_sites * sizeof(uint32_t));
    if (self->haplotypes == NULL || self->new_mutations == NULL
            || self->num_new_mutations == NULL || self->new_mutations_mem == NULL) {
        ret = -1;
        goto out;
    }
    /* Copy the sample haplotypes in */
    memcpy(self->haplotypes, sample_haplotypes,
            num_samples * num_sites * sizeof(uint8_t));

    /* Set the new_mutations for the singletons */
    /* First pass, figure out how many singletons each sample has */
    for (j = 0; j < num_singleton_sites; j++) {
        self->num_new_mutations[singleton_haplotypes[j]]++;
    }
    /* Second pass, allocate memory for each of these */
    offset = 0;
    for (j = 0; j < num_singleton_sites; j++) {
        hap_index = singleton_haplotypes[j];
        if (self->num_new_mutations[hap_index] != 0) {
            self->new_mutations[hap_index] = self->new_mutations_mem + offset;
            offset += self->num_new_mutations[hap_index];
            self->num_new_mutations[hap_index] = 0;
        }
    }
    /* Third pass, insert the singleton mutations. */
    for (j = 0; j < num_singleton_sites; j++) {
        hap_index = singleton_haplotypes[j];
        self->new_mutations[hap_index][
            self->num_new_mutations[hap_index]] = singleton_sites[j];
        self->num_new_mutations[hap_index]++;
    }
    /* Now deal with the non-singleton sites for the ancestors */
    qsort(non_singleton_sites, num_non_singleton_sites, sizeof(site_t), cmp_site);
    for (j = 0; j < num_non_singleton_sites; j++) {
        hap_index = self->num_haplotypes - j - 2;
        self->num_new_mutations[hap_index] = 1;
        self->new_mutations[hap_index] = self->new_mutations_mem + offset;
        self->new_mutations_mem[offset] = non_singleton_sites[j].index;
        offset++;
    }
    ret = reference_panel_infer_ancestors(self, num_non_singleton_sites,
            non_singleton_sites);
out:
    if (non_singleton_sites != NULL) {
        free(non_singleton_sites);
    }
    if (singleton_sites != NULL) {
        free(singleton_sites);
    }
    if (singleton_haplotypes != NULL) {
        free(singleton_haplotypes);
    }
    return ret;
}

int
reference_panel_free(reference_panel_t *self)
{
    if (self->haplotypes != NULL) {
        free(self->haplotypes);
    }
    if (self->positions != NULL) {
        free(self->positions);
    }
    if (self->new_mutations != NULL) {
        free(self->new_mutations);
    }
    if (self->num_new_mutations != NULL) {
        free(self->num_new_mutations);
    }
    if (self->new_mutations_mem != NULL) {
        free(self->new_mutations_mem);
    }
    return 0;
}

int
reference_panel_print_state(reference_panel_t *self)
{
    uint32_t l, j;

    printf("Reference panel n = %d, m = %d\n", (int) self->num_haplotypes,
            (int) self->num_sites);
    printf("Positions = ");
    for (l = 0; l < self->num_sites; l++) {
        printf("%f, ", self->positions[l]);
    }
    printf("\n");
    for (j = 0; j < self->num_haplotypes; j++) {
        printf("%3d\t", j);
        for (l = 0; l < self->num_sites; l++) {
            printf("%d", self->haplotypes[j * self->num_sites + l]);
        }
        printf("\t%d:", (int) self->num_new_mutations[j]);
        for (l = 0; l < self->num_new_mutations[j]; l++) {
            printf("%d ", (int) self->new_mutations[j][l]);
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
        double error_probablilty, uint32_t *P, uint32_t *num_mutations, uint32_t *mutations)
{
    int ret = 0;
    const uint32_t n = panel_size;
    const uint32_t N = self->reference_panel->num_haplotypes;
    const uint32_t m = self->reference_panel->num_sites;
    const uint32_t num_new_mutations = self->reference_panel->num_new_mutations[haplotype_index];
    const uint32_t *new_mutations = self->reference_panel->new_mutations[haplotype_index];
    const double r = 1 - exp(-recombination_rate / n);
    const double recomb_proba = r / n;
    const double no_recomb_proba = 1 - r + r / n;
    /* Local working storage */
    const uint8_t *H = self->reference_panel->haplotypes;
    const uint8_t *h = &self->reference_panel->haplotypes[haplotype_index * m];
    double *V = self->V;
    double *V_next = self->V_next;
    double *X = self->reference_panel->positions;
    uint32_t *T = self->T;
    uint32_t j, k, l, max_j, max_V_index, max_V_next_index;
    double scale, d, max_p, x, y, emission_p, *tmp;
    bool new_mutation;

    /* zero out the traceback matrix. This isn't necessary, but helps with debugging. */
    memset(self->T, 0, N * m * sizeof(uint32_t));

    if (haplotype_index >= N) {
        ret = -3;
        goto out;
    }
    y = 0;
    max_V_index = N - n;
    scale = 1;
    d = 1;
    for (j = N - n; j < N; j++) {
        V[j] = 1;
    }
    k = 0;
    for (l = 0; l < m; l++) {
        new_mutation = false;
        if (k < num_new_mutations && new_mutations[k] == l) {
            new_mutation = true;
            k++;
        }
        for (j = N - n; j < N; j++) {
            V[j] /= scale;
        }
        max_V_next_index = N - n;
        for (j = N - n; j < N; j++) {
            x = V[j] * no_recomb_proba * d;
            y = V[max_V_index] * recomb_proba * d;
            if (x > y) {
                max_p = x;
                max_j = j;
            } else {
                max_p = y;
                max_j = max_V_index;
            }
            T[j * m + l] = max_j;
            if (new_mutation) {
                emission_p = 1;
            } else {
                emission_p = 1;
                if (H[j * m + l] != h[l]) {
                    emission_p = error_probablilty;
                }
            }
            V_next[j] = max_p * emission_p;
            if (V_next[j] >= V_next[max_V_next_index]) {
                /* If we have several haplotypes with equal liklihood, choose the oldest */
                max_V_next_index = j;
            }
        }
        /* swap V and V_next */
        tmp = V;
        V = V_next;
        V_next = tmp;
        max_V_index = max_V_next_index;
        scale = V[max_V_index];
        if (l < m - 1) {
            d = X[l + 1] - X[l];
        }
    }
    P[m - 1] = max_V_index;
    for (l = m - 1; l > 0; l--) {
        P[l - 1] = T[P[l] * m + l];
    }
    j = 0;
    for (l = 0; l < m; l++) {
        if (h[l] != H[P[l] * m + l]) {
            mutations[j] = l;
            j++;
        }
    }
    *num_mutations = j;
out:
    return ret;
}
