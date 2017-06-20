#include "tsinfer.h"
#include "err.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

static int
cmp_permutation_sort(const void *a, const void *b) {
    const permutation_sort_t *ia = (const permutation_sort_t *) a;
    const permutation_sort_t *ib = (const permutation_sort_t *) b;
    int ret = (ia->state > ib->state) - (ia->state < ib->state);
    if (ret == 0) {
        ret = (ia->current_position > ib->current_position)
            - (ia->current_position < ib->current_position);
    }
    return ret;
}

static void
ancestor_sorter_check_state(ancestor_sorter_t *self)
{
    size_t l;
    int *q = calloc(self->num_sites, sizeof(int));
    site_id_t *p = self->permutation;
    assert(q != NULL);

    for (l = 0; l < self->num_sites; l++) {
        assert(p[l] < self->num_sites);
        q[p[l]]++;
    }
    for (l = 0; l < self->num_sites; l++) {
        assert(q[l] == 1);
    }
    free(q);
}

int
ancestor_sorter_print_state(ancestor_sorter_t *self, FILE *out)
{
    size_t j, l;
    allele_t *a;

    fprintf(out, "Ancestor sorter state:\n");
    fprintf(out, "num_ancestors = %d:\n", (int) self->num_ancestors);
    fprintf(out, "num_sites = %d:\n", (int) self->num_sites);
    fprintf(out, "permutation = ");
    for (l = 0; l < self->num_sites; l++) {
        fprintf(out, "%d, ", (int) self->permutation[l]);
    }
    fprintf(out, "\n");
    fprintf(out, "ancestors = \n");
    for (j = 0; j < self->num_ancestors; j++) {
        a = self->ancestors + self->permutation[j] * self->num_sites;
        for (l = 0; l < self->num_sites; l++) {
            if (a[l] == -1) {
                fprintf(out, "*");
            } else {
                fprintf(out, "%d", a[l]);
            }
        }
        fprintf(out, "\n");
    }
    ancestor_sorter_check_state(self);
    return 0;
}

int
ancestor_sorter_alloc(ancestor_sorter_t *self, size_t num_ancestors, size_t num_sites,
        allele_t *ancestors, site_id_t *permutation)
{
    int ret = 0;
    size_t l;

    memset(self, 0, sizeof(ancestor_sorter_t));
    self->num_ancestors = num_ancestors;
    self->num_sites = num_sites;
    self->ancestors = ancestors;
    self->permutation = permutation;
    self->sort_buffer = malloc(num_ancestors * sizeof(permutation_sort_t));
    if (self->sort_buffer == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }

    /* initialise the identity permutation */
    for (l = 0; l < num_ancestors; l++) {
        permutation[l] = l;
    }
out:
    return ret;
}

int
ancestor_sorter_free(ancestor_sorter_t *self)
{
    tsi_safe_free(self->sort_buffer);
    return 0;
}

static int
ancestor_sorter_sort_slice(ancestor_sorter_t *self, size_t start, size_t end)
{
    int ret = 0;
    size_t segment_breaks, max_segment_breaks, sort_site, l, j, k;
    const size_t m = self->num_sites;
    site_id_t *p = self->permutation;
    allele_t *A = self->ancestors;

    assert(end > start);
    if ((end - start) > 1) {
        max_segment_breaks = 0;
        sort_site = 0;
        for (l = 0; l < self->num_sites; l++) {
            segment_breaks = 0;
            for (j = start; j < end - 1; j++) {
                if (A[p[j] * m + l] > A[p[j + 1] * m + l]) {
                    segment_breaks++;
                }
            }
            if (segment_breaks > max_segment_breaks) {
                max_segment_breaks = segment_breaks;
                sort_site = l;
            }
        }
        if (max_segment_breaks > 1) {
            l = sort_site;
            k = 0;
            for (j = start; j < end; j++) {
                self->sort_buffer[k].index = p[j];
                self->sort_buffer[k].current_position = k;
                self->sort_buffer[k].state = A[p[j] * m + l];
                k++;
            }
            qsort(self->sort_buffer, k, sizeof(permutation_sort_t), cmp_permutation_sort);
            k = 0;
            for (j = start; j < end; j++) {
                p[j] = self->sort_buffer[k].index;
                k++;
            }
            /* Recurse on the subsegments. */
            k = start;
            for (j = start; j < end - 1; j++) {
                if (A[p[j] * m + l] != A[p[j + 1] * m + l]) {
                    ret = ancestor_sorter_sort_slice(self, k, j + 1);
                    if (ret != 0) {
                        goto out;
                    }
                    k = j + 1;
                }
            }
        }
    }
out:
    return ret;
}

int
ancestor_sorter_sort(ancestor_sorter_t *self)
{
    int ret =  ancestor_sorter_sort_slice(self, 0, self->num_ancestors);
    /* ancestor_sorter_print_state(self, stdout); */
    return ret;
}
