#include "tsinfer.h"
#include "err.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

static void
ancestor_store_check_state(ancestor_store_t *self)
{
    site_id_t l;
    size_t total_segments = 0;
    size_t max_site_segments = 0;

    for (l = 0; l < self->num_sites; l++) {
        total_segments += self->sites[l].num_segments;
        if (self->sites[l].num_segments > max_site_segments) {
            max_site_segments = self->sites[l].num_segments;
        }
    }
    assert(total_segments == self->total_segments);
    assert(max_site_segments == self->max_num_site_segments);
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
    fprintf(out, "segment_block_size = %d\n", (int) self->segment_block_size);
    fprintf(out, "num_site_segment_expands = %d\n", (int) self->num_site_segment_expands);
    fprintf(out, "total_memory = %d\n", (int) self->total_memory);
    for (l = 0; l < self->num_sites; l++) {
        site = &self->sites[l];
        printf("%d\t[%d]:: ", (int) l, (int) site->num_segments);
        for (j = 0; j < site->num_segments; j++) {
            printf("(%d, %d: %d)", site->start[j], site->end[j], site->state[j]);
        }
        printf("\n");
    }
    ancestor_store_check_state(self);
    return 0;
}

int
ancestor_store_alloc(ancestor_store_t *self, size_t num_sites)
{
    int ret = 0;

    memset(self, 0, sizeof(ancestor_store_t));
    self->num_sites = num_sites;

    self->num_ancestors = 1;
    self->segment_block_size = 0;
    self->total_memory = self->num_sites * sizeof(site_state_t);
    self->sites = calloc(self->num_sites, sizeof(site_state_t));
    if (self->sites == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
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
    return 0;
}

int
ancestor_store_init_build(ancestor_store_t *self, size_t segment_block_size)
{
    int ret = 0;
    size_t l;

    self->num_ancestors = 1;
    self->segment_block_size = segment_block_size;
    self->total_segments = self->num_sites;
    self->max_num_site_segments = 1;
    self->total_memory += self->num_sites * self->segment_block_size *
        (2 * sizeof(ancestor_id_t) + sizeof(allele_t));
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
out:
    return ret;
}

static int
ancestor_store_expand_site_segments(ancestor_store_t *self, site_id_t site_id)
{
    int ret = 0;
    ancestor_id_t *start;
    ancestor_id_t *end;
    allele_t *state;
    site_state_t *site = self->sites + site_id;

    assert(self->segment_block_size > 0);
    site->max_num_segments += self->segment_block_size;
    self->num_site_segment_expands++;
    self->total_memory += self->segment_block_size *
        (2 * sizeof(ancestor_id_t) + sizeof(allele_t));

    start = realloc(site->start, site->max_num_segments * sizeof(ancestor_id_t));
    if (start == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    site->start = start;
    end = realloc(site->end, site->max_num_segments * sizeof(ancestor_id_t));
    if (end == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    site->end = end;
    state = realloc(site->state, site->max_num_segments * sizeof(allele_t));
    if (state == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    site->state = state;
out:
    return ret;
}

int
ancestor_store_add(ancestor_store_t *self, allele_t *ancestor)
{
    int ret = 0;
    site_id_t l;
    site_state_t *site;
    size_t k;
    ancestor_id_t n = (ancestor_id_t) self->num_ancestors;

    for (l = 0; l < self->num_sites; l++) {
        if (ancestor[l] != -1) {
            site = &self->sites[l];
            k = site->num_segments;
            if (site->end[k - 1] == n && site->state[k - 1] == ancestor[l]) {
                site->end[k - 1] = n + 1;
            } else {
                if (k == site->max_num_segments) {
                    ret = ancestor_store_expand_site_segments(self, l);
                    if (ret != 0) {
                        goto out;
                    }
                }
                site->start[k] = n;
                site->end[k] = n + 1;
                site->state[k] = ancestor[l];
                site->num_segments = k + 1;
                self->total_segments++;
                if (site->num_segments > self->max_num_site_segments) {
                    self->max_num_site_segments = site->num_segments;
                }
            }
        }
    }
    self->num_ancestors++;
out:
    return ret;
}

int
ancestor_store_load(ancestor_store_t *self, size_t num_segments, site_id_t *site,
        ancestor_id_t *start, ancestor_id_t *end, allele_t *state)
{
    int ret = 0;
    site_id_t j, l, site_start, site_end;
    size_t k, num_site_segments;

    site_start = 0;
    site_end = 0;
    self->max_num_site_segments = 0;
    for (l = 0; l < self->num_sites; l++) {
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
        self->sites[l].max_num_segments = num_site_segments;
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
            if (end[j] > (ancestor_id_t) self->num_ancestors) {
                self->num_ancestors = end[j];
            }
            k++;
        }
        site_start = site_end;
    }
    assert(self->total_segments == num_segments);
out:
    return ret;
}

int
ancestor_store_dump(ancestor_store_t *self, site_id_t *site, ancestor_id_t *start,
        ancestor_id_t *end, allele_t *state)
{
    int ret = 0;
    site_id_t l;
    site_state_t s;
    size_t j, k;

    k = 0;
    for (l = 0; l < self->num_sites; l++) {
        s = self->sites[l];
        for (j = 0; j < s.num_segments; j++) {
            assert(k < self->total_segments);
            site[k] = l;
            start[k] = s.start[j];
            end[k] = s.end[j];
            state[k] = s.state[j];
            k++;
        }
    }
    return ret;
}

size_t
ancestor_store_get_num_segments(ancestor_store_t *self)
{
    return self->total_segments;
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
        allele_t *ancestor)
{
    int ret = 0;
    site_id_t l;

    for (l = 0; l < self->num_sites; l++) {
        ret = ancestor_store_get_state(self, l, ancestor_id, ancestor + l);
        if (ret != 0) {
            goto out;
        }
    }
out:
    return ret;
}
