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
ancestor_matcher_alloc(ancestor_matcher_t *self, ancestor_store_t *store,
        double recombination_rate, double mutation_rate)
{
    int ret = 0;

    memset(self, 0, sizeof(ancestor_matcher_t));
    self->store = store;
    self->recombination_rate = recombination_rate;
    self->mutation_rate = mutation_rate;

    self->segment_block_size = 1024;
    ret = object_heap_init(&self->segment_heap, sizeof(segment_t),
           self->segment_block_size, NULL);
    if (ret != 0) {
        goto out;
    }
out:
    return ret;
}

int
ancestor_matcher_free(ancestor_matcher_t *self)
{
    object_heap_free(&self->segment_heap);
    return 0;
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
    memset(path, 0xff, self->store->num_sites * sizeof(ancestor_id_t));
    path[end_site] = best_match;
    for (l = end_site; l > start_site; l--) {
        /* printf("Tracing back at site %d\n", l); */
        /* print_segment_chain(T_head[l], 1, stdout); */
        /* printf("\n"); */
        ret = ancestor_store_get_state(self->store, l, path[l], &state);
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
    ret = ancestor_store_get_state(self->store, l, path[l], &state);
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
ancestor_matcher_best_path(ancestor_matcher_t *self, size_t num_ancestors,
        allele_t *haplotype, ancestor_id_t *path, size_t *num_mutations,
        site_id_t *mutation_sites)
{
    int ret = 0;
    double rho = self->recombination_rate;
    double theta = self->mutation_rate;
    double n = (double) self->store->num_ancestors;
    double r = 1 - exp(-rho / n);
    double pr = r / n;
    double qr = 1 - r + r / n;
    // pm = mutation; qm no mutation
    double pm = 0.5 * theta / (n + theta);
    double qm = n / (n + theta) + 0.5 * theta / (n + theta);
    ancestor_id_t N = (ancestor_id_t) num_ancestors;
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
    /* TODO Is it really safe to have an upper bound here? */
    size_t max_segments = self->store->max_num_site_segments * 8;

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
    T_head = calloc(self->store->num_sites, sizeof(segment_t *));
    T_tail = calloc(self->store->num_sites, sizeof(segment_t *));
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
    end_site = (site_id_t) self->store->num_sites - 1;
    while (haplotype[start_site] == -1) {
        start_site++;
        // TODO error check this properly
        assert(start_site < self->store->num_sites);
    }

    best_match = 0;
    site_id = start_site;
    while (site_id < self->store->num_sites && haplotype[site_id] != -1) {
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
    for (l = 0; l < self->store->num_sites; l++) {
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

    fprintf(out, "Ancestor matcher state\n");
    ancestor_store_print_state(self->store, out);
    fprintf(out, "segment_block_size = %d\n", (int) self->segment_block_size);
    fprintf(out, "Segment heap:\n");
    object_heap_print_state(&self->segment_heap, out);

    ancestor_matcher_check_state(self);
    return 0;
}
