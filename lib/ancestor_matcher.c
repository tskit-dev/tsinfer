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
print_segment_chain(segment_t *head, bool as_int, FILE *out)
{
    segment_t *u = head;

    while (u != NULL) {
        if (as_int) {
            fprintf(out, "(%d-%d:%d)", u->start, u->end, (int) u->value);
        } else {
            fprintf(out, "(%d-%d:%f)", u->start, u->end, u->value);
        }
        u = u->next;
        if (u != NULL) {
            fprintf(out, "=>");
        }
    }
}

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
    self->sites_head = malloc(self->num_sites * sizeof(segment_t *));
    self->sites_tail = malloc(self->num_sites * sizeof(segment_t *));
    if (self->sites_head == NULL || self->sites_tail == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;

    }
    ret = object_heap_init(&self->segment_heap, sizeof(segment_t),
           segment_block_size, NULL);
    if (ret != 0) {
        goto out;
    }
    for (l = 0; l < self->num_sites; l++) {
        self->sites_head[l] = ancestor_matcher_alloc_segment(self, 0, 1, 0);
        if (self->sites_head[l] == NULL) {
            ret = TSI_ERR_NO_MEMORY;
            goto out;
        }
        self->sites_tail[l] = self->sites_head[l];
    }

out:
    return ret;
}

int
ancestor_matcher_free(ancestor_matcher_t *self)
{
    tsi_safe_free(self->sites_head);
    tsi_safe_free(self->sites_tail);
    object_heap_free(&self->segment_heap);

    return 0;
}

int
ancestor_matcher_add(ancestor_matcher_t *self, allele_t *haplotype)
{
    int ret = 0;
    site_id_t l;
    segment_t *tail;
    ancestor_id_t n = (ancestor_id_t) self->num_ancestors;

    for (l = 0; l < self->num_sites; l++) {
        tail = self->sites_tail[l];
        if (tail->value == (double) haplotype[l]) {
            tail->end++;
        } else {
            tail = ancestor_matcher_alloc_segment(self, n, n + 1, haplotype[l]);
            self->sites_tail[l]->next = tail;
            self->sites_tail[l] = tail;
        }
    }
    self->num_ancestors++;
    return ret;
}

static int
ancestor_matcher_run_traceback(ancestor_matcher_t *self, segment_t **T_head,
        site_id_t start_site, site_id_t end_site, ancestor_id_t best_match,
        ancestor_id_t *path)
{
    int ret = 0;
    site_id_t l;
    ancestor_id_t p;
    segment_t *u;

    /* printf("traceback for %d-%d, best=%d\n", start_site, end_site, best_match); */
    /* Set everything to -1 */
    memset(path, 0xff, self->num_sites * sizeof(ancestor_id_t));
    path[end_site] = best_match;
    for (l = end_site; l > start_site; l--) {
        /* printf("Tracing back at site %d\n", l); */
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
    return ret;
}

int
ancestor_matcher_best_match(ancestor_matcher_t *self, allele_t *haplotype,
        ancestor_id_t *path)
{
    int ret = 0;
    double rho = 0.01;
    double theta = 1e-200;
    double n = (double) self->num_ancestors;
    double r = 1 - exp(-rho / n);
    double pr = r / n;
    double qr = 1 - r + r / n;
    // pm = mutation; qm no mutation
    double pm = 0.5 * theta / (n + theta);
    double qm = n / (n + theta) + 0.5 * theta / (n + theta);
    ancestor_id_t N = (ancestor_id_t) self->num_ancestors;
    site_id_t start_site, end_site, l;
    ancestor_id_t start, end;
    segment_t *V_head, *v, *w, *V_next_head, *V_next_tail, *tmp;
    double value, max_value, x, y, z;
    allele_t state;
    ancestor_id_t best_match;
    segment_t **T_head = NULL;
    segment_t **T_tail = NULL;

    T_head = calloc(self->num_sites, sizeof(segment_t *));
    T_tail = calloc(self->num_sites, sizeof(segment_t *));
    if (T_head == NULL || T_tail == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }


    /* skip any leading unset values in the input haplotype */
    start_site = 0;
    while (haplotype[start_site] == -1) {
        start_site++;
        // TODO error check this properly
        assert(start_site < self->num_sites);
    }
    V_head = ancestor_matcher_alloc_segment(self, 0, N, 1);
    /* TODO check for NULL on all of these calls */
    /* V_tail = V_head; */

    l = start_site;
    while (l < self->num_sites && haplotype[l] != -1) {
        v = V_head;
        max_value = -1;
        /* Find the match in the previous iteration */
        best_match = (ancestor_id_t) -1;
        assert(v->start == 0);
        while (v != NULL) {
            if (v->value >= max_value) {
                max_value = v->value;
                best_match = v->end - 1;
            }
            if (v->next != NULL) {
                assert(v->next->start == v->end);
            }
            v = v->next;
        }
        /* Renormalise V */
        v = V_head;
        while (v != NULL) {
            v->value /= max_value;
            v = v->next;
        }
        v = V_head;
        w = self->sites_head[l];

        /* printf("l = %d\n", l); */
        /* printf("v = "); */
        /* print_segment_chain(v, false, stdout); */
        /* printf("\n"); */
        /* printf("w = "); */
        /* print_segment_chain(w, true, stdout); */
        /* printf("\n"); */
        /* printf("h = %d\n", haplotype[l]); */
        /* printf("b = %d\n", best_match); */

        V_next_head = NULL;
        V_next_tail = NULL;
        while (v != NULL && w != NULL) {
            start = GSL_MAX(v->start, w->start);
            end = GSL_MIN(v->end, w->end);
            value = v->value;
            state = (allele_t) w->value;
            if (w->end == v->end) {
                w = w->next;
                tmp = v;
                v = v->next;
                ancestor_matcher_free_segment(self, tmp);
            } else if (w->end < v->end) {
                w = w->next;
            } else if (v->end < w->end) {
                tmp = v;
                v = v->next;
                ancestor_matcher_free_segment(self, tmp);
            } else {
                assert(false);
            }

            /* printf("\t%d\t%d\t%d\t%g\n", start, end, state, value); */
            x = value * qr;
            y = pr; /* value for maximum is 1 by normalisation */
            if (x >= y) {
                z = x;
            } else {
                z = y;
                if (T_head[l] == NULL) {
                    T_head[l] = ancestor_matcher_alloc_segment(self, start, end, best_match);
                    T_tail[l] = T_head[l];
                } else {
                    if (T_tail[l]->end == start
                            && T_tail[l]->value == (double) best_match) {
                        T_tail[l]->end = end;
                    } else {
                        tmp = ancestor_matcher_alloc_segment(self, start, end, best_match);
                        T_tail[l]->next = tmp;
                        T_tail[l] = tmp;
                    }


                }
            }
            if (state == -1) {
                value = 0;
            } else if (state == haplotype[l]) {
                value = z * qm;
            } else {
                value = z * pm;
            }
            if (V_next_head == NULL) {
                V_next_head = ancestor_matcher_alloc_segment(self, start, end, value);
                V_next_tail = V_next_head;
            } else {
                if (V_next_tail->end == start && V_next_tail->value == value) {
                    V_next_tail->end = end;
                } else {
                    tmp = ancestor_matcher_alloc_segment(self, start, end, value);
                    V_next_tail->next = tmp;
                    V_next_tail = tmp;
                }

            }

        }
        V_head = V_next_head;
        end_site = l;
        l++;
    }

    max_value = -1;
    best_match = (ancestor_id_t) -1;
    v = V_head;
    while (v != NULL) {
        if (v->value >= max_value) {
            max_value = v->value;
            best_match = v->end - 1;
        }
        tmp = v;
        v = v->next;
        ancestor_matcher_free_segment(self, tmp);
    }
    ret = ancestor_matcher_run_traceback(self, T_head, start_site, end_site,
            best_match, path);
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
    tsi_safe_free(T_head);
    tsi_safe_free(T_tail);
    return ret;
}


static void
ancestor_matcher_check_state(ancestor_matcher_t *self)
{
    site_id_t l;
    segment_t *u;
    size_t total_segments = 0;

    for (l = 0; l < self->num_sites; l++) {
        u = self->sites_head[l];
        assert(u != NULL);
        assert(u->start == 0);
        while (u->next != NULL) {
            assert(u->next->start == u->end);
            u = u->next;
            total_segments++;
        }
        total_segments++;
        assert(self->sites_tail[l] == u);
        assert(u->end == self->num_ancestors);
    }
    assert(total_segments == object_heap_get_num_allocated(&self->segment_heap));
}

int
ancestor_matcher_print_state(ancestor_matcher_t *self, FILE *out)
{

    site_id_t l;

    fprintf(out, "Ancestor matcher state\n");
    fprintf(out, "num_sites = %d\n", (int) self->num_sites);
    fprintf(out, "num_ancestors = %d\n", (int) self->num_ancestors);
    for (l = 0; l < self->num_sites; l++) {
        fprintf(out, "%d\t:", l);
        print_segment_chain(self->sites_head[l], true, out);
        fprintf(out, "\n");
    }
    fprintf(out, "Segment heap:\n");
    object_heap_print_state(&self->segment_heap, out);


    ancestor_matcher_check_state(self);
    return 0;
}
