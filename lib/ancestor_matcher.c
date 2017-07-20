#include "tsinfer.h"
#include "err.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#include <gsl/gsl_math.h>

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
ancestor_matcher_check_state(ancestor_matcher_t *self)
{
}

int
ancestor_matcher_print_state(ancestor_matcher_t *self, FILE *out)
{

    fprintf(out, "Ancestor matcher state\n");
    ancestor_store_print_state(self->store, out);
    ancestor_matcher_check_state(self);
    return 0;
}

int
ancestor_matcher_alloc(ancestor_matcher_t *self, ancestor_store_t *store,
        double recombination_rate)
{
    int ret = 0;

    memset(self, 0, sizeof(ancestor_matcher_t));
    self->store = store;
    self->recombination_rate = recombination_rate;
    return ret;
}

int
ancestor_matcher_free(ancestor_matcher_t *self)
{
    return 0;
}

int
ancestor_matcher_best_path(ancestor_matcher_t *self, size_t num_ancestors,
        allele_t *haplotype, site_id_t start_site, site_id_t end_site,
        size_t num_focal_sites, site_id_t *focal_sites, double error_rate,
        traceback_t *traceback)
{
    int ret = 0;
    double rho, r, pr, qr, possible_recombinants;
    ancestor_id_t N = (ancestor_id_t) num_ancestors;
    site_id_t site_id;
    ancestor_id_t start, end;
    double likelihood, next_likelihood, max_likelihood, x, y, z, *double_tmp;
    ancestor_id_t *L_start = NULL;
    ancestor_id_t *L_end = NULL;
    double *L_likelihood = NULL;
    ancestor_id_t *L_next_start = NULL;
    ancestor_id_t *L_next_end = NULL;
    ancestor_id_t *ancestor_id_tmp;
    double *L_next_likelihood = NULL;
    ancestor_id_t *S_start, *S_end;
    allele_t state;
    size_t l, s, L_size, S_size, L_next_size, focal_site_index;
    /* TODO Is it really safe to have an upper bound here? */
    size_t max_segments = self->store->max_num_site_segments * 32;
    double last_position;

    /* Error rate and focal sites are mutually exclusive */
    if (error_rate == 0) {
        assert(num_focal_sites > 0);
    } else {
        assert(num_focal_sites == 0);
    }

    L_start = malloc(max_segments * sizeof(ancestor_id_t));
    L_end = malloc(max_segments * sizeof(ancestor_id_t));
    L_likelihood = malloc(max_segments * sizeof(double));
    L_next_start = malloc(max_segments * sizeof(ancestor_id_t));
    L_next_end = malloc(max_segments * sizeof(ancestor_id_t));
    L_next_likelihood = malloc(max_segments * sizeof(double));
    if (L_start == NULL || L_end == NULL || L_likelihood == NULL
            || L_next_start == NULL || L_next_end == NULL || L_next_likelihood == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;

    }
    /* Initialise L to carry one segment covering entire interval. */
    L_start[0] = 0;
    L_end[0] = N;
    L_likelihood[0] = 1;
    L_size = 1;
    /* ensure that that the initial recombination rate is 0 */
    last_position = self->store->sites[start_site].position;
    possible_recombinants = N;

    /* Skip any focal sites that are not within this segment. */
    focal_site_index = 0;
    while (focal_site_index < num_focal_sites
            && focal_sites[focal_site_index] < start_site) {
        focal_site_index++;
    }
    for (site_id = start_site; site_id < end_site; site_id++) {
        if (focal_site_index < num_focal_sites && site_id > focal_sites[focal_site_index]) {
            focal_site_index++;
        }
        /* printf("site = %d next_focal_site = %d, focal_site_index = %d\n", */
        /*         (int) site_id, (int) focal_sites[focal_site_index], (int) focal_site_index); */

        /* Compute the recombination rate back to the last site */
        rho = self->recombination_rate * (
                self->store->sites[site_id].position - last_position);
        r = 1 - exp(-rho / possible_recombinants);
        pr = r / possible_recombinants;
        qr = 1 - r + r / possible_recombinants;

        L_next_size = 0;
        /* Make a get_site function in the store here? */
        S_start = self->store->sites[site_id].start;
        S_end = self->store->sites[site_id].end;
        S_size = self->store->sites[site_id].num_segments;

        /* printf("site = %d\n", site_id); */
        /* printf("S = "); */
        /* for (s = 0; s < S_size; s++) { */
        /*     printf("(%d, %d: %d)", S_start[s], S_end[s], state); */
        /* } */
        /* printf("\n"); */
        /* printf("L = "); */
        /* for (l = 0; l < L_size; l++) { */
        /*     printf("(%d, %d: %.6g)", L_start[l], L_end[l], L_likelihood[l]); */
        /* } */
        /* printf("\n"); */

        l = 0;
        s = 0;
        start = 0;
        while (start != N) {
            end = N;
            if (l < L_size) {
                if (L_start[l] > start) {
                    end = GSL_MIN(end, L_start[l]);
                } else {
                    end = GSL_MIN(end, L_end[l]);
                }
            }
            state = 0;
            if (s < S_size) {
                if (S_start[s] > start) {
                    end = GSL_MIN(end, S_start[s]);
                } else {
                    end = GSL_MIN(end, S_end[s]);
                }
                if (S_start[s] <= start && end <= S_end[s]) {
                    state = 1;
                }
            }

            /* printf("\tINNER LOOP: start = %d, end = %d\n", start, end); */
            assert(start < end);

            /* If this interval does not intersect with L, the likelihood is 0 */
            likelihood = 0;
            if (l < L_size && !(L_start[l] >= end || L_end[l] <= start)) {
                likelihood = L_likelihood[l];
            }
            x = likelihood * qr;
            y = pr; /* value for maximum is 1 by normalisation */
            if (x >= y) {
                z = x;
            } else {
                z = y;
                /* printf("%d add_recombination %d %d\n", site_id, start, end); */
                ret = traceback_add_recombination(traceback, site_id, start, end);
                if (ret != 0) {
                    goto out;
                }
            }

            if (error_rate == 0) {
                /* Ancestor matching */
                next_likelihood = z * (state == haplotype[site_id]);

                if (focal_site_index < num_focal_sites) {
                    if (site_id == focal_sites[focal_site_index]) {
                        assert(haplotype[site_id] == 1);
                        assert(state == 0);
                        next_likelihood = z;
                    } else {
                        assert(site_id < focal_sites[focal_site_index]);
                    }
                }
            } else {
                /* Sample matching */
                if (state == haplotype[site_id]) {
                    next_likelihood = z * (1 - error_rate);
                } else {
                    next_likelihood = z * error_rate;
                }
            }
            /* printf("next_likelihood = %f\n", next_likelihood); */

            /* Update L_next */
            if (L_next_size == 0) {
                L_next_size = 1;
                L_next_start[0] = start;
                L_next_end[0] = end;
                L_next_likelihood[0] = next_likelihood;
            } else {
                /* printf("updating L_next: %d\n", (int) L_next_size); */
                if (L_next_end[L_next_size - 1] == start
                        && L_next_likelihood[L_next_size - 1] == next_likelihood) {
                    L_next_end[L_next_size - 1] = end;
                } else  {
                    assert(L_next_size < max_segments);
                    L_next_start[L_next_size] = start;
                    L_next_end[L_next_size] = end;
                    L_next_likelihood[L_next_size] = next_likelihood;
                    L_next_size++;
                }
            }

            start = end;
            if (l < L_size && L_end[l] <= start) {
                l++;
            }
            if (s < S_size && S_end[s] <= start) {
                s++;
            }
        }

        /* Swap L and L_next */
        L_size = L_next_size;
        ancestor_id_tmp = L_start;
        L_start = L_next_start;
        L_next_start = ancestor_id_tmp;
        ancestor_id_tmp = L_end;
        L_end = L_next_end;
        L_next_end = ancestor_id_tmp;
        double_tmp = L_likelihood;
        L_likelihood = L_next_likelihood;
        L_next_likelihood = double_tmp;
        /* Normalise L and get set the best matching ancestor for this site. */
        max_likelihood = -1;
        for (l = 0; l < L_size; l++) {
            if (L_likelihood[l] > max_likelihood) {
                max_likelihood = L_likelihood[l];
            }
        }
        assert(max_likelihood > 0);
        for (l = 0; l < L_size; l++) {
            if (L_likelihood[l] == max_likelihood) {
                /* Set the best match to the oldest ancestor with the maximum
                 * likelihood */
                assert(L_start[l] < N);
                ret = traceback_set_best_match(traceback, site_id, L_start[l]);
                if (ret != 0) {
                    goto out;
                }
            }
            L_likelihood[l] /= max_likelihood;
            /* printf("\t%d,%d -> %f\n", L_start[l], L_end[l], L_likelihood[l]); */
        }
    }
    /* assert(focal_site_index == num_focal_sites); */

out:
    tsi_safe_free(L_start);
    tsi_safe_free(L_end);
    tsi_safe_free(L_likelihood);
    tsi_safe_free(L_next_start);
    tsi_safe_free(L_next_end);
    tsi_safe_free(L_next_likelihood);
    return ret;
}
