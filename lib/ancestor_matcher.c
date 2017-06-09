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

/* static void */
/* print_segment_chain(segment_t *head, bool as_int, FILE *out) */
/* { */
/*     segment_t *u = head; */

/*     while (u != NULL) { */
/*         if (as_int) { */
/*             fprintf(out, "(%d-%d:%d)", u->start, u->end, (int) u->value); */
/*         } else { */
/*             fprintf(out, "(%d-%d:%f)", u->start, u->end, u->value); */
/*         } */
/*         u = u->next; */
/*         if (u != NULL) { */
/*             fprintf(out, "=>"); */
/*         } */
/*     } */
/* } */

static inline segment_t * WARN_UNUSED
ancestor_matcher_alloc_segment(ancestor_matcher_t *self, ancestor_id_t start,
        ancestor_id_t end, double value)
{
    segment_t *ret = NULL;

    if (object_heap_empty(&self->segment_heap)) {
        if (object_heap_expand(&self->segment_heap) != 0) {
            goto out;
        }
    }
    ret = (segment_t *) object_heap_alloc_object(&self->segment_heap);
    ret->start = start;
    ret->end = end;
    ret->value = value;
    ret->next = NULL;
out:
    return ret;
}

static inline void
ancestor_matcher_free_segment(ancestor_matcher_t *self, segment_t *seg)
{
    object_heap_free_object(&self->segment_heap, seg);
}

int
ancestor_matcher_alloc(ancestor_matcher_t *self, size_t num_sites,
        size_t segment_block_size)
{
    int ret = 0;
    site_id_t l;

    memset(self, 0, sizeof(ancestor_matcher_t));
    self->num_sites = num_sites;
    self->num_ancestors = 1;
    self->segment_block_size = segment_block_size;
    self->sites = calloc(self->num_sites, sizeof(site_t));
    if (self->sites == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    for (l = 0; l < self->num_sites; l++) {
        self->sites[l].start = malloc(self->segment_block_size * sizeof(ancestor_id_t));
        self->sites[l].end = malloc(self->segment_block_size * sizeof(ancestor_id_t));
        self->sites[l].state = malloc(self->segment_block_size * sizeof(allele_t));
        if (self->sites[l].start == NULL || self->sites[l].end == NULL
                || self->sites[l].state == NULL) {
            ret = TSI_ERR_NO_MEMORY;
            goto out;
        }
        self->sites[l].max_num_segments = segment_block_size;
        /* Create the oldest ancestor that is 0 everywhere */
        self->sites[l].num_segments = 1;
        self->sites[l].start[0] = 0;
        self->sites[l].end[0] = 1;
        self->sites[l].state[0] = 0;
    }
    ret = object_heap_init(&self->segment_heap, sizeof(segment_t),
           segment_block_size, NULL);
    if (ret != 0) {
        goto out;
    }
out:
    return ret;
}

int
ancestor_matcher_free(ancestor_matcher_t *self)
{
    site_id_t l;

    for (l = 0; l < self->num_sites; l++) {
        tsi_safe_free(self->sites[l].start);
        tsi_safe_free(self->sites[l].end);
        tsi_safe_free(self->sites[l].state);
    }
    tsi_safe_free(self->sites);
    object_heap_free(&self->segment_heap);
    return 0;
}

static int
ancestor_matcher_expand_site_segments(ancestor_matcher_t *self, site_id_t site_id)
{
    int ret = 0;
    /* TODO implement this */
    ret = TSI_ERR_NO_MEMORY;
    return ret;
}


int
ancestor_matcher_add(ancestor_matcher_t *self, allele_t *haplotype)
{
    int ret = 0;
    site_id_t l;
    site_t *site;
    size_t k;
    ancestor_id_t n = (ancestor_id_t) self->num_ancestors;

    for (l = 0; l < self->num_sites; l++) {
        if (haplotype[l] != -1) {
            site = &self->sites[l];
            k = site->num_segments;
            if (site->end[k - 1] == n && site->state[k - 1] == haplotype[l]) {
                site->end[k - 1] = n + 1;
            } else {
                if (k == site->max_num_segments) {
                    ret = ancestor_matcher_expand_site_segments(self, l);
                    if (ret != 0) {
                        goto out;
                    }
                }
                site->start[k] = n;
                site->end[k] = n + 1;
                site->state[k] = haplotype[l];
                site->num_segments = k + 1;
            }
        }
    }
    self->num_ancestors++;
out:
    return ret;
}

/*
Returns the state of the specified ancestor at the specified site.
*/
static int
ancestor_matcher_get_state(ancestor_matcher_t *self, site_id_t site_id,
        ancestor_id_t ancestor, allele_t *state)
{
    int ret = 0;
    site_t *site = &self->sites[site_id];
    size_t j = 0;

    while (j < site->num_segments && site->end[j] <= ancestor) {
        j++;
    }
    assert(j < site->num_segments);
    assert(site->start[j] <= ancestor && ancestor < site->end[j]);
    *state = site->state[j];
    return ret;
}

static int
ancestor_matcher_run_traceback(ancestor_matcher_t *self, allele_t *haplotype,
        segment_t **T_head, site_id_t start_site, site_id_t end_site,
        ancestor_id_t best_match, ancestor_id_t *path, size_t *num_mutations,
        site_id_t *mutation_sites)
{
    int ret = 0;
    site_id_t l;
    ancestor_id_t p;
    segment_t *u;
    allele_t state;
    size_t local_num_mutations = 0;

    /* printf("traceback for %d-%d, best=%d\n", start_site, end_site, best_match); */
    /* Set everything to -1 */
    memset(path, 0xff, self->num_sites * sizeof(ancestor_id_t));
    path[end_site] = best_match;
    for (l = end_site; l > start_site; l--) {
        /* printf("Tracing back at site %d\n", l); */
        /* print_segment_chain(T_head[l], 1, stdout); */
        /* printf("\n"); */
        ret = ancestor_matcher_get_state(self, l, path[l], &state);
        if (ret != 0) {
            goto out;
        }
        if (state != haplotype[l]) {
            mutation_sites[local_num_mutations] = l;
            local_num_mutations++;
        }

        p = (ancestor_id_t) -1;
        u = T_head[l];
        while (u != NULL) {
            if (u->start <= path[l] && path[l] < u->end) {
                p = (ancestor_id_t) u->value;
                break;
            }
            if (u->start > path[l]) {
                break;
            }
            u = u->next;
        }
        if (p == (ancestor_id_t) -1) {
            p = path[l];
        }
        path[l - 1] = p;
    }
    l = start_site;
    ret = ancestor_matcher_get_state(self, l, path[l], &state);
    if (ret != 0) {
        goto out;
    }
    if (state != haplotype[l]) {
        mutation_sites[local_num_mutations] = l;
        local_num_mutations++;
    }
    *num_mutations = local_num_mutations;
out:
    return ret;
}

int
ancestor_matcher_best_path(ancestor_matcher_t *self, allele_t *haplotype,
        double recombination_rate, double mutation_rate, ancestor_id_t *path,
        size_t *num_mutations, site_id_t *mutation_sites)
{
    int ret = 0;
    size_t max_segments = 8192;
    double rho = recombination_rate;
    double theta = mutation_rate;
    double n = (double) self->num_ancestors;
    double r = 1 - exp(-rho / n);
    double pr = r / n;
    double qr = 1 - r + r / n;
    // pm = mutation; qm no mutation
    double pm = 0.5 * theta / (n + theta);
    double qm = n / (n + theta) + 0.5 * theta / (n + theta);
    ancestor_id_t N = (ancestor_id_t) self->num_ancestors;
    site_id_t start_site, end_site, site_id;
    ancestor_id_t start, end;
    segment_t *tmp, *v;
    double likelihood, next_likelihood, max_likelihood, x, y, z, *double_tmp;
    ancestor_id_t best_match;
    segment_t **T_head = NULL;
    segment_t **T_tail = NULL;
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
    T_head = calloc(self->num_sites, sizeof(segment_t *));
    T_tail = calloc(self->num_sites, sizeof(segment_t *));
    if (T_head == NULL || T_tail == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    /* Initialise L to carry one segment covering entire interval. */
    L_start[0] = 0;
    L_end[0] = N;
    L_likelihood[0] = 1;
    L_size = 1;

    /* skip any leading unset values in the input haplotype */
    start_site = 0;
    end_site = (site_id_t) self->num_sites - 1;
    while (haplotype[start_site] == -1) {
        start_site++;
        // TODO error check this properly
        assert(start_site < self->num_sites);
    }

    best_match = 0;
    site_id = start_site;
    while (site_id < self->num_sites && haplotype[site_id] != -1) {
        L_next_size = 0;
        S_start = self->sites[site_id].start;
        S_end = self->sites[site_id].end;
        S_state = self->sites[site_id].state;
        S_size = self->sites[site_id].num_segments;

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
                    if (T_head[site_id] == NULL) {
                        T_head[site_id] = ancestor_matcher_alloc_segment(self, start, end, best_match);
                        T_tail[site_id] = T_head[site_id];
                    } else {
                        if (T_tail[site_id]->end == start
                                && T_tail[site_id]->value == (double) best_match) {
                            T_tail[site_id]->end = end;
                        } else {
                            tmp = ancestor_matcher_alloc_segment(self, start, end, best_match);
                            T_tail[site_id]->next = tmp;
                            T_tail[site_id] = tmp;
                        }
                    }
                }
                if (S_state[s] == haplotype[site_id]) {
                    next_likelihood = z * qm;
                } else {
                    next_likelihood = z * pm;
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

        end_site = site_id;
        site_id++;
    }

    ret = ancestor_matcher_run_traceback(self, haplotype, T_head, start_site, end_site,
            best_match, path, num_mutations, mutation_sites);
    /* free the segments in T */
    for (l = 0; l < self->num_sites; l++) {
        v = T_head[l];
        while (v != NULL) {
            tmp = v;
            v = v->next;
            ancestor_matcher_free_segment(self, tmp);
        }
    }
out:
    tsi_safe_free(L_start);
    tsi_safe_free(L_end);
    tsi_safe_free(L_likelihood);
    tsi_safe_free(L_next_start);
    tsi_safe_free(L_next_end);
    tsi_safe_free(L_next_likelihood);
    tsi_safe_free(T_head);
    tsi_safe_free(T_tail);
    return ret;
}


static void
ancestor_matcher_check_state(ancestor_matcher_t *self)
{
    /* site_id_t l; */
    /* segment_t *u; */
    /* size_t total_segments = 0; */

    /* for (l = 0; l < self->num_sites; l++) { */
    /*     u = self->sites_head[l]; */
    /*     assert(u != NULL); */
    /*     assert(u->start == 0); */
    /*     while (u->next != NULL) { */
    /*         assert(u->next->start == u->end); */
    /*         u = u->next; */
    /*         total_segments++; */
    /*     } */
    /*     total_segments++; */
    /*     assert(self->sites_tail[l] == u); */
    /*     assert(u->end == (ancestor_id_t) self->num_ancestors); */
    /* } */
    /* assert(total_segments == object_heap_get_num_allocated(&self->segment_heap)); */
}

int
ancestor_matcher_print_state(ancestor_matcher_t *self, FILE *out)
{
    site_id_t l;
    site_t *site;
    size_t j;

    fprintf(out, "Ancestor matcher state\n");
    fprintf(out, "num_sites = %d\n", (int) self->num_sites);
    fprintf(out, "num_ancestors = %d\n", (int) self->num_ancestors);
    fprintf(out, "segment_block_size = %d\n", (int) self->segment_block_size);
    for (l = 0; l < self->num_sites; l++) {
        site = &self->sites[l];
        printf("%d\t[%d]:: ", (int) l, (int) site->num_segments);
        for (j = 0; j < site->num_segments; j++) {
            printf("(%d, %d: %d)", site->start[j], site->end[j], site->state[j]);
        }
        printf("\n");
    }
    fprintf(out, "Segment heap:\n");
    object_heap_print_state(&self->segment_heap, out);


    ancestor_matcher_check_state(self);
    return 0;
}
