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
        traceback_t *traceback, ancestor_id_t *end_site_value)
{
    int ret = 0;
    double rho, r, pr, qr, possible_recombinants;
    ancestor_id_t N = (ancestor_id_t) num_ancestors;
    site_id_t site_id;
    ancestor_id_t start, end;
    double likelihood, next_likelihood, max_likelihood, x, y, z, *double_tmp;
    ancestor_id_t best_match;
    ancestor_id_t *L_start = NULL;
    ancestor_id_t *L_end = NULL;
    double *L_likelihood = NULL;
    ancestor_id_t *L_next_start = NULL;
    ancestor_id_t *L_next_end = NULL;
    ancestor_id_t *ancestor_id_tmp;
    double *L_next_likelihood = NULL;
    ancestor_id_t *S_start, *S_end;
    allele_t *S_state;
    size_t l, s, L_size, S_size, L_next_size;
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
    possible_recombinants = 1;
    best_match = 0;
    /* focal_site_index = 0; */

    for (site_id = start_site; site_id < end_site; site_id++) {

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
        S_state = self->store->sites[site_id].state;
        S_size = self->store->sites[site_id].num_segments;

        /* printf("site = %d\n", site_id); */
        /* printf("S = "); */
        /* for (s = 0; s < S_size; s++) { */
        /*     printf("(%d, %d: %d)", S_start[s], S_end[s], S_state[s]); */
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
            if (s < S_size) {
                if (S_start[s] > start) {
                    end = GSL_MIN(end, S_start[s]);
                } else {
                    end = GSL_MIN(end, S_end[s]);
                }
            }
            /* printf("\tINNER LOOP: start = %d, end = %d\n", start, end); */
            assert(start < end);
            /* The likelihood of this interval is always 0 if it does not intersect
             * with S. */
            if (s < S_size && !(S_start[s] >= end || S_end[s] <= start)) {
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
                    ret = traceback_add_recombination(traceback, site_id, start, end, best_match);
                    if (ret != 0) {
                        goto out;
                    }
                }
                if (error_rate == 0) {
                    /* Ancestor matching */
                    next_likelihood = z * (S_state[s] == haplotype[site_id]);


                    /* /1* Ancestor matching *1/ */
                    /* next_likelihood = z * (S_state[s] == haplotype[site_id]); */
                    /* if (site_id== focal_site) { */
                    /*     assert(haplotype[site_id] == 1); */
                    /*     assert(S_state[s] == 0); */
                    /*     next_likelihood = z; */
                    /* } */

                    if (site_id == focal_sites[0]) {
                        assert(haplotype[site_id] == 1);
                        assert(S_state[s] == 0);
                        next_likelihood = z;
                    }
                    /* TODO this code _should_ work, but leads to really weird effects.
                     * Need to rethink the ancestor matching process! */

                    /* if (focal_site_index < num_focal_sites) { */
                    /*     if (site_id == focal_sites[focal_site_index]) { */
                    /*         assert(haplotype[site_id] == 1); */
                    /*         assert(S_state[s] == 0); */
                    /*         next_likelihood = z; */
                    /*         focal_site_index++; */
                    /*     } else { */
                    /*         assert(site_id < focal_sites[focal_site_index]); */
                    /*     } */
                    /* } */
                } else {
                    /* Sample matching */
                    if (S_state[s] == haplotype[site_id]) {
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
        /* Normalise L and get the best haplotype */
        max_likelihood = -1;
        best_match = -1;
        for (l = 0; l < L_size; l++) {
            if (L_likelihood[l] > max_likelihood) {
                max_likelihood = L_likelihood[l];
                best_match = L_end[l] - 1;
            }
        }
        for (l = 0; l < L_size; l++) {
            L_likelihood[l] /= max_likelihood;
        }
    }
    /* assert(focal_site_index == num_focal_sites); */
    *end_site_value = best_match;

    /* ret = ancestor_matcher_run_traceback(self, haplotype, T_head, start_site, end_site, */
    /*         best_match, path, num_mutations, mutation_sites); */
out:
    tsi_safe_free(L_start);
    tsi_safe_free(L_end);
    tsi_safe_free(L_likelihood);
    tsi_safe_free(L_next_start);
    tsi_safe_free(L_next_end);
    tsi_safe_free(L_next_likelihood);
    return ret;
}
