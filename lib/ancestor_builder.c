/*
** Copyright (C) 2018 University of Oxford
**
** This file is part of tsinfer.
**
** tsinfer is free software: you can redistribute it and/or modify
** it under the terms of the GNU General Public License as published by
** the Free Software Foundation, either version 3 of the License, or
** (at your option) any later version.
**
** tsinfer is distributed in the hope that it will be useful,
** but WITHOUT ANY WARRANTY; without even the implied warranty of
** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
** GNU General Public License for more details.
**
** You should have received a copy of the GNU General Public License
** along with tsinfer.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "tsinfer.h"
#include "err.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#include "avl.h"

typedef struct {
    allele_t *state;
    site_id_t id;
    size_t num_samples;
} site_equality_t;


static int
cmp_pattern_map(const void *a, const void *b) {
    const pattern_map_t *ia = (pattern_map_t const *) a;
    const pattern_map_t *ib = (pattern_map_t const *) b;
    int ret = memcmp(ia->genotypes, ib->genotypes, ia->num_samples * sizeof(allele_t));
    return ret;
}

int
ancestor_builder_alloc(ancestor_builder_t *self, size_t num_samples, size_t num_sites,
        int flags)
{
    int ret = 0;
    size_t j;
    // TODO error checking
    //
    assert(num_samples > 1);
    /* TODO need to be able to handle zero sites */
    /* assert(num_sites > 0); */

    memset(self, 0, sizeof(ancestor_builder_t));
    self->num_samples = num_samples;
    self->num_sites = num_sites;
    self->flags = flags;
    self->sites = calloc(num_sites, sizeof(site_t));
    self->frequency_map = calloc(num_samples + 1, sizeof(avl_tree_t));
    self->descriptors = calloc(num_sites, sizeof(ancestor_descriptor_t));
    if (self->sites == NULL || self->frequency_map == NULL
            || self->descriptors == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    ret = block_allocator_alloc(&self->allocator, 1024 * 1024);
    if (ret != 0) {
        goto out;
    }
    for (j = 0; j < num_samples + 1; j++) {
        avl_init_tree(&self->frequency_map[j], cmp_pattern_map, NULL);
    }
out:
    return ret;
}

int
ancestor_builder_free(ancestor_builder_t *self)
{
    tsi_safe_free(self->sites);
    tsi_safe_free(self->frequency_map);
    tsi_safe_free(self->descriptors);
    block_allocator_free(&self->allocator);
    return 0;
}

static inline void
ancestor_builder_get_consistent_samples(ancestor_builder_t *self, site_id_t site,
        node_id_t *samples, size_t *num_samples)
{
    node_id_t j, k;
    allele_t *restrict genotypes = self->sites[site].genotypes;

    k = 0;
    for (j = 0; j < (node_id_t) self->num_samples; j++) {
        if (genotypes[j] == 1) {
            samples[k] = j;
            k++;
        }
    }
    *num_samples = (size_t) k;
}

static int
ancestor_builder_compute_ancestral_states(ancestor_builder_t *self,
        int direction, site_id_t focal_site, allele_t *ancestor,
        node_id_t *restrict sample_set, bool *restrict disagree,
        site_id_t *last_site_ret)
{
    int ret = 0;
    site_id_t last_site = focal_site;
    int64_t l;
    node_id_t u;
    size_t j, ones, zeros, tmp_size, sample_set_size;
    size_t focal_site_frequency = self->sites[focal_site].frequency;
    size_t min_sample_set_size = focal_site_frequency / 2;
    const site_t *restrict sites = self->sites;
    const size_t num_sites = self->num_sites;
    const allele_t *restrict genotypes;
    allele_t consensus;

    ancestor_builder_get_consistent_samples(self, focal_site,
            sample_set, &sample_set_size);
    assert(sample_set_size == focal_site_frequency);
    memset(disagree, 0, self->num_samples * sizeof(*disagree));

    /* printf("site=%d, direction=%d\n", (int) focal_site, direction); */
    for (l = focal_site + direction; l >= 0 && l < (int64_t) num_sites; l += direction) {
        /* printf("\tl = %d\n", l); */
        ancestor[l] = 0;
        last_site = (site_id_t) l;
        if (sites[l].frequency > focal_site_frequency) {

            /* printf("\t%d\t%d:", l, (int) sample_set_size); */
            /* for (j = 0; j < sample_set_size; j++) { */
            /*     printf("%d, ", sample_set[j]); */
            /* } */

            genotypes = self->sites[l].genotypes;
            ones = 0;
            for (j = 0; j < sample_set_size; j++) {
                ones += genotypes[sample_set[j]];
            }
            zeros = sample_set_size - ones;
            consensus = 0;
            if (ones >= zeros) {
                consensus = 1;
            }
            /* printf("\t:ones=%d, consensus=%d\n", (int) ones, consensus); */
            for (j = 0; j < sample_set_size; j++) {
                u = sample_set[j];
                if (disagree[u] && genotypes[u] != consensus) {
                    /* This sample has disagreed with consensus twice in a row,
                     * so remove it */
                    /* printf("\t\tremoving %d\n", sample_set[j]); */
                    sample_set[j] = -1;
                }
            }
            /* Repack the sample set */
            tmp_size = 0;
            for (j = 0; j < sample_set_size; j++) {
                if (sample_set[j] != -1) {
                    sample_set[tmp_size] = sample_set[j];
                    tmp_size++;
                }
            }
            sample_set_size = tmp_size;
            if (sample_set_size <= min_sample_set_size) {
                /* printf("BREAK\n"); */
                break;
            }
            ancestor[l] = consensus;
            /* For the remaining sample set, set the disagree flags based
             * on whether they agree with the consensus for this site. */
            for (j = 0; j < sample_set_size; j++) {
                u = sample_set[j];
                disagree[u] = genotypes[u] != consensus;
            }
        }
    }
    *last_site_ret = last_site;
    return ret;
}

static int
ancestor_builder_compute_between_focal_sites(ancestor_builder_t *self,
        size_t num_focal_sites, site_id_t *focal_sites,
        allele_t *ancestor, node_id_t *sample_set)
{
    int ret = 0;
    site_id_t l;
    size_t j, k, ones, zeros, sample_set_size, focal_site_frequency;
    const site_t *restrict sites = self->sites;
    const allele_t *restrict genotypes;

    assert(num_focal_sites > 0);
    ancestor_builder_get_consistent_samples(self, focal_sites[0],
            sample_set, &sample_set_size);
    focal_site_frequency = self->sites[focal_sites[0]].frequency;
    assert(sample_set_size == focal_site_frequency);

    ancestor[focal_sites[0]] = 1;
    for (j = 1; j < num_focal_sites; j++) {
        ancestor[focal_sites[j]] = 1;
        for (l = focal_sites[j - 1] + 1; l < focal_sites[j]; l++) {
            ancestor[l] = 0;
            if (sites[l].frequency > focal_site_frequency) {
                /* printf("\t%d\t%d:", l, (int) sample_set_size); */
                /* for (k = 0; k < sample_set_size; k++) { */
                /*     printf("%d, ", sample_set[k]); */
                /* } */
                genotypes = self->sites[l].genotypes;
                ones = 0;
                for (k = 0; k < sample_set_size; k++) {
                    ones += genotypes[sample_set[k]];
                }
                zeros = sample_set_size - ones;
                if (ones >= zeros) {
                    ancestor[l] = 1;
                }
            }
        }
    }
    return ret;
}

/* Build the ancestors for sites in the specified focal sites */
int
ancestor_builder_make_ancestor(ancestor_builder_t *self, size_t num_focal_sites,
        site_id_t *focal_sites, site_id_t *ret_start, site_id_t *ret_end,
        allele_t *ancestor)
{
    int ret = 0;
    site_id_t focal_site, last_site;
    node_id_t *sample_set = malloc(self->num_samples * sizeof(node_id_t));
    bool *restrict disagree = calloc(self->num_samples, sizeof(*disagree));

    if (sample_set == NULL || disagree == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    memset(ancestor, 0xff, self->num_sites * sizeof(*ancestor));

    ret = ancestor_builder_compute_between_focal_sites(self,
            num_focal_sites, focal_sites, ancestor, sample_set);
    if (ret != 0) {
        goto out;
    }

    focal_site = focal_sites[num_focal_sites - 1];
    ret = ancestor_builder_compute_ancestral_states(self,
            +1, focal_site, ancestor, sample_set, disagree, &last_site);
    if (ret != 0) {
        goto out;
    }
    assert(ancestor[last_site] != -1);
    *ret_end = last_site + 1;

    focal_site = focal_sites[0];
    ret = ancestor_builder_compute_ancestral_states(self,
            -1, focal_site, ancestor, sample_set, disagree, &last_site);
    if (ret != 0) {
        goto out;
    }
    assert(ancestor[last_site] != -1);
    *ret_start = last_site;
out:
    tsi_safe_free(sample_set);
    tsi_safe_free(disagree);
    return ret;
}


int WARN_UNUSED
ancestor_builder_add_site(ancestor_builder_t *self, site_id_t l, size_t frequency,
        allele_t *genotypes)
{
    int ret = 0;
    site_t *site;
    avl_node_t *avl_node;
    site_list_t *list_node;
    pattern_map_t search, *map_elem;
    avl_tree_t *pattern_map = &self->frequency_map[frequency];

    assert(frequency <= self->num_samples);
    assert(l < (site_id_t) self->num_sites);
    site = &self->sites[l];
    site->frequency = frequency;
    if (frequency > 1) {
        search.genotypes = genotypes;
        search.num_samples = self->num_samples;
        avl_node = avl_search(pattern_map, &search);
        if (avl_node == NULL) {
            avl_node = block_allocator_get(&self->allocator, sizeof(avl_node_t));
            map_elem = block_allocator_get(&self->allocator, sizeof(pattern_map_t));
            site->genotypes = block_allocator_get(&self->allocator,
                    self->num_samples * sizeof(allele_t));
            if (avl_node == NULL || map_elem == NULL || site->genotypes == NULL) {
                ret = TSI_ERR_NO_MEMORY;
                goto out;
            }
            memcpy(site->genotypes, genotypes, self->num_samples * sizeof(allele_t));
            avl_init_node(avl_node, map_elem);
            map_elem->genotypes = site->genotypes;
            map_elem->num_samples = self->num_samples;
            map_elem->sites = NULL;
            map_elem->num_sites = 0;
            avl_node = avl_insert_node(pattern_map, avl_node);
            assert(avl_node != NULL);
            if (site->genotypes == NULL) {
                ret = TSI_ERR_NO_MEMORY;
                goto out;
            }
        } else {
            map_elem = (pattern_map_t *) avl_node->item;
            site->genotypes = map_elem->genotypes;
        }
        map_elem->num_sites++;

        list_node = block_allocator_get(&self->allocator, sizeof(site_list_t));
        if (list_node == NULL) {
            ret = TSI_ERR_NO_MEMORY;
            goto out;
        }
        list_node->site = l;
        list_node->next = map_elem->sites;
        map_elem->sites = list_node;
    }
out:
    return ret;
}

static void
ancestor_builder_check_state(ancestor_builder_t *self)
{
    size_t f, k, count;
    avl_node_t *a;
    pattern_map_t *map_elem;
    site_list_t *s;

    for (f = 0; f < self->num_samples + 1; f++) {
        for (a = self->frequency_map[f].head; a != NULL; a = a->next) {
            map_elem = (pattern_map_t *) a->item;
            count = 0;
            for (k = 0; k < self->num_samples; k++) {
                count += map_elem->genotypes[k] == 1;
            }
            assert(count == f);
            count = 0;
            for (s = map_elem->sites; s != NULL; s = s->next) {
                assert(self->sites[s->site].frequency == f);
                assert(self->sites[s->site].genotypes == map_elem->genotypes);
                count++;
            }
            assert(map_elem->num_sites == count);
        }
    }
}

int
ancestor_builder_print_state(ancestor_builder_t *self, FILE *out)
{
    size_t j, k;
    avl_node_t *a;
    pattern_map_t *map_elem;
    site_list_t *s;

    fprintf(out, "Ancestor builder\n");
    fprintf(out, "num_samples = %d\n", (int) self->num_samples);
    fprintf(out, "num_sites = %d\n", (int) self->num_sites);
    fprintf(out, "num_ancestors = %d\n", (int) self->num_ancestors);

    fprintf(out, "Sites:\n");
    for (j = 0; j < self->num_sites; j++) {
        fprintf(out, "%d\t%d\t%p\n", (int) j, (int) self->sites[j].frequency,
                self->sites[j].genotypes);
    }
    fprintf(out, "Frequency map:\n");
    for (j = 0; j < self->num_samples + 1; j++) {
        printf("Frequency = %d: %d ancestors\n", (int) j,
                avl_count(&self->frequency_map[j]));
        for (a = self->frequency_map[j].head; a != NULL; a = a->next) {
            map_elem = (pattern_map_t *) a->item;
            printf("\t");
            for (k = 0; k < self->num_samples; k++) {
                printf("%d", map_elem->genotypes[k]);
            }
            printf("\t");
            for (s = map_elem->sites; s != NULL; s = s->next) {
                printf("%d ", s->site);
            }
            printf("\n");
        }
    }
    fprintf(out, "Descriptors:\n");
    for (j = 0; j < self->num_ancestors; j++) {
        fprintf(out, "%d\t%d: ",  (int) self->descriptors[j].frequency,
                (int) self->descriptors[j].num_focal_sites);
        for (k = 0; k < self->descriptors[j].num_focal_sites; k++) {
            fprintf(out, "%d, ", self->descriptors[j].focal_sites[k]);
        }
        fprintf(out, "\n");
    }
    block_allocator_print_state(&self->allocator, out);
    ancestor_builder_check_state(self);
    return 0;
}

/* Returns true if we should break the an ancestor that spans from focal
 * site a to focal site b */
static bool
ancestor_builder_break_ancestor(ancestor_builder_t *self, site_id_t a,
        site_id_t b, node_id_t *restrict samples, size_t num_samples)
{
    bool ret = false;
    site_id_t j, k;
    size_t ones;

    for (j = a + 1; j < b && !ret; j++) {
        if (self->sites[j].frequency > self->sites[a].frequency) {
            ones = 0;
            for (k = 0; k < (site_id_t) num_samples; k++) {
                ones += self->sites[j].genotypes[samples[k]];
            }
            if (ones != num_samples && ones != 0) {
                ret = true;
            }
        }
    }
    return ret;
}

int
ancestor_builder_finalise(ancestor_builder_t *self)
{
    int ret = 0;
    size_t j, k, num_consistent_samples;
    avl_node_t *a;
    pattern_map_t *map_elem;
    site_list_t *s;
    ancestor_descriptor_t *descriptor;
    site_id_t *focal_sites = NULL;
    site_id_t *p;
    site_id_t *consistent_samples = malloc(self->num_samples * sizeof(node_id_t));

    if (consistent_samples == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    num_consistent_samples = 0;  /* Keep the compiler happy */
    self->num_ancestors = 0;
    for (j = self->num_samples; j > 1; j--) {
        for (a = self->frequency_map[j].head; a != NULL; a = a->next) {
            descriptor = self->descriptors + self->num_ancestors;
            self->num_ancestors++;
            descriptor->frequency = j;
            map_elem = (pattern_map_t *) a->item;
            focal_sites = block_allocator_get(&self->allocator,
                    map_elem->num_sites * sizeof(site_id_t));
            if (focal_sites == NULL) {
                ret = TSI_ERR_NO_MEMORY;
                goto out;
            }
            descriptor->focal_sites = focal_sites;
            descriptor->num_focal_sites = map_elem->num_sites;
            k = map_elem->num_sites - 1;
            for (s = map_elem->sites; s != NULL; s = s->next) {
                focal_sites[k] = s->site;
                k--;
            }
            /* Now check to see if we need to split this ancestor up
             * further */
            if (map_elem->num_sites > 1) {
                ancestor_builder_get_consistent_samples(self, focal_sites[0],
                        consistent_samples, &num_consistent_samples);
                assert(num_consistent_samples == descriptor->frequency);
            }
            for (k = 0; k < map_elem->num_sites - 1; k++) {
                if (ancestor_builder_break_ancestor(
                        self, focal_sites[k], focal_sites[k + 1],
                        consistent_samples, num_consistent_samples)) {
                    p = focal_sites + k + 1;
                    descriptor->num_focal_sites = p - descriptor->focal_sites;
                    descriptor = self->descriptors + self->num_ancestors;
                    self->num_ancestors++;
                    descriptor->frequency = j;
                    descriptor->num_focal_sites = map_elem->num_sites - k - 1;
                    descriptor->focal_sites = p;
                }
            }
        }
    }
out:
    tsi_safe_free(consistent_samples);
    return ret;
}
