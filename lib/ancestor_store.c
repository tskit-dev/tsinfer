#include "tsinfer.h"
#include "err.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

static void
ancestor_store_check_state(ancestor_store_t *self)
{
    int ret;
    site_id_t l, start, end, focal;
    size_t j;
    size_t total_segments = 0;
    size_t max_site_segments = 0;
    allele_t *a = malloc(self->num_sites * sizeof(allele_t));
    assert(a != NULL);

    for (l = 0; l < self->num_sites; l++) {
        total_segments += self->sites[l].num_segments;
        if (self->sites[l].num_segments > max_site_segments) {
            max_site_segments = self->sites[l].num_segments;
        }
    }
    assert(total_segments == self->total_segments);
    assert(max_site_segments == self->max_num_site_segments);
    for (j = 0; j < self->num_ancestors; j++) {
        ret = ancestor_store_get_ancestor(self, j, a, &start, &focal, &end);
        assert(ret == 0);
        if (j > 0) {
            assert(a[focal] == 1);
            assert(start <= focal);
            assert(focal < end);
        }
        assert(end <= self->num_sites);
        assert(start < end);
        for (l = 0; l < self->num_sites; l++) {
            if (l < start || l >= end) {
                assert(a[l] == -1);
            } else {
                assert(a[l] != -1);
            }
        }
    }
    free(a);
}

int
ancestor_store_print_state(ancestor_store_t *self, FILE *out)
{
    site_id_t l;
    site_state_t *site;
    size_t j;

    fprintf(out, "Ancestor store\n");
    fprintf(out, "num_sites = %d\n", (int) self->num_sites);
    fprintf(out, "num_ancestors = %d\n", (int) self->num_ancestors);
    fprintf(out, "total_segments  = %d\n", (int) self->total_segments);
    fprintf(out, "max_num_site_segments = %d\n", (int) self->max_num_site_segments);
    fprintf(out, "total_memory = %d\n", (int) self->total_memory);
    for (l = 0; l < self->num_sites; l++) {
        site = &self->sites[l];
        printf("%d\t%.3f\t[%d]:: ", (int) l, site->position, (int) site->num_segments);
        for (j = 0; j < site->num_segments; j++) {
            printf("(%d, %d: %d)", site->start[j], site->end[j], site->state[j]);
        }
        printf("\n");
    }
    fprintf(out, "ancestors = \n");
    for (j = 0; j < self->num_ancestors; j++) {
        fprintf(out, "%d\t%d\n", (int) j, self->ancestors.focal_site[j]);
    }
    ancestor_store_check_state(self);
    return 0;
}

int
ancestor_store_alloc(ancestor_store_t *self, size_t num_sites, double *position,
        size_t num_ancestors, site_id_t *focal_site,
        size_t num_segments, site_id_t *site, ancestor_id_t *start, ancestor_id_t *end,
        allele_t *state)
{
    int ret = 0;
    site_id_t j, l, site_start, site_end;
    size_t k, num_site_segments;
    ancestor_id_t seg_num_ancestors;

    memset(self, 0, sizeof(ancestor_store_t));
    self->num_sites = num_sites;
    self->num_ancestors = num_ancestors;
    self->sites = calloc(num_sites, sizeof(site_state_t));
    self->ancestors.focal_site = malloc(num_ancestors * sizeof(site_id_t));
    if (self->sites == NULL || self->ancestors.focal_site == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    memcpy(self->ancestors.focal_site, focal_site, num_ancestors * sizeof(site_id_t));
    site_start = 0;
    site_end = 0;
    self->max_num_site_segments = 0;
    seg_num_ancestors = 0;
    for (l = 0; l < self->num_sites; l++) {
        if (l > 0) {
            // TODO raise an error here.
            assert(position[l] > position[l - 1]);
        }
        self->sites[l].position = position[l];
        assert(site[site_start] == l);
        assert(site[site_end] == l);
        while (site_end < num_segments && site[site_end] == l) {
            site_end++;
        }
        assert(site_end == num_segments || site[site_end] == l + 1);
        num_site_segments = site_end - site_start;
        assert(num_site_segments > 0);
        if (num_site_segments > self->max_num_site_segments) {
            self->max_num_site_segments = num_site_segments;
        }
        self->total_memory += num_site_segments * (2 * sizeof(ancestor_id_t) + sizeof(allele_t));
        self->sites[l].start = malloc(num_site_segments * sizeof(ancestor_id_t));
        self->sites[l].end = malloc(num_site_segments * sizeof(ancestor_id_t));
        self->sites[l].state = malloc(num_site_segments * sizeof(allele_t));
        if (self->sites[l].start == NULL || self->sites[l].end == NULL
                || self->sites[l].state == NULL) {
            ret = TSI_ERR_NO_MEMORY;
            goto out;
        }
        k = 0;
        for (j = site_start; j < site_end; j++) {
            assert(site[j] == l);
            self->sites[l].start[k] = start[j];
            self->sites[l].end[k] = end[j];
            self->sites[l].state[k] = state[j];
            self->sites[l].num_segments++;
            self->total_segments++;
            if (end[j] > seg_num_ancestors) {
                seg_num_ancestors = end[j];
            }
            k++;
        }
        site_start = site_end;
    }
    // TODO error checking.
    assert(self->total_segments == num_segments);
    assert(seg_num_ancestors == (ancestor_id_t) num_ancestors);
out:
    return ret;
}

int
ancestor_store_free(ancestor_store_t *self)
{
    site_id_t l;

    for (l = 0; l < self->num_sites; l++) {
        tsi_safe_free(self->sites[l].start);
        tsi_safe_free(self->sites[l].end);
        tsi_safe_free(self->sites[l].state);
    }
    tsi_safe_free(self->sites);
    tsi_safe_free(self->ancestors.focal_site);
    return 0;
}

/*
 Returns the state of the specified ancestor at the specified site.
*/
int
ancestor_store_get_state(ancestor_store_t *self, site_id_t site_id,
        ancestor_id_t ancestor_id, allele_t *state)
{
    int ret = 0;
    site_state_t *site = &self->sites[site_id];
    size_t j = 0;


    /* TODO use bsearch here to find the closest segment */
    while (j < site->num_segments && site->end[j] <= ancestor_id) {
        j++;
    }
    *state = -1;
    if (j < site->num_segments &&
        site->start[j] <= ancestor_id && ancestor_id < site->end[j]) {
        *state = site->state[j];
    }
    return ret;
}

int
ancestor_store_get_ancestor(ancestor_store_t *self, ancestor_id_t ancestor_id,
        allele_t *ancestor, site_id_t *start_site, site_id_t *focal_site,
        site_id_t *end_site)
{
    int ret = 0;
    site_id_t l, start;
    bool started = false;

    memset(ancestor, 0xff, self->num_sites * sizeof(allele_t));
    start = 0;
    for (l = 0; l < self->num_sites; l++) {
        ret = ancestor_store_get_state(self, l, ancestor_id, ancestor + l);
        if (ret != 0) {
            goto out;
        }
        if (ancestor[l] != -1 && ! started) {
            start = l;
            started = true;
        }
        if (ancestor[l] == -1 && started) {
            break;
        }
    }
    *start_site = start;
    *focal_site = self->ancestors.focal_site[ancestor_id];
    *end_site = l;
out:
    return ret;
}
