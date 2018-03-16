#include "tsinfer.h"
#include "err.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#include "uthash.h"
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
    /* Note that this means that we cannot use threads to build multiple
     * ancestors at the same time!! Will need to add a num_threads
     * argument here and a thread index or something. */
    self->consistent_samples_mem = malloc(self->num_samples
            * sizeof(ancestor_id_hash_t));
    if (self->sites == NULL || self->frequency_map == NULL
            || self->descriptors == NULL || self->consistent_samples_mem == NULL) {
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
    tsi_safe_free(self->consistent_samples_mem);
    block_allocator_free(&self->allocator);
    return 0;
}

static inline uint8_t
ancestor_builder_get_fgt_patterns(ancestor_builder_t *self, site_id_t focal_site_id,
        site_id_t site_id, ancestor_id_hash_t **consistent_samples)
{
    allele_t *restrict focal_genotypes = self->sites[focal_site_id].genotypes;
    allele_t *restrict site_genotypes = self->sites[site_id].genotypes;
    uint8_t fgt_patterns = 0;
    int index;
    allele_t a, b;
    ancestor_id_hash_t *s, *tmp;

    /* for (j = 0; j < (int) self->num_samples; j++) { */
    HASH_ITER(hh, *consistent_samples, s, tmp) {
        a = focal_genotypes[s->value];
        b = site_genotypes[s->value];
        /* Unrank the bit pattern 00, 01, 10, 11 to an index 0,...,3
         * and set the correponding bit in ret */
        index = a * 2 + b;
        fgt_patterns = fgt_patterns | (1 << index);
        /* printf("\t %d %d = %d -> %d\n", a, b, a * 2 + b, ret); */
    }
    return fgt_patterns;
}

static inline bool
ancestor_builder_make_site(ancestor_builder_t *self, site_id_t focal_site_id,
        site_id_t site_id, bool remove_inconsistent,
        ancestor_id_hash_t **consistent_samples, allele_t *ancestor)
{
    bool ret = true;
    size_t ones;
    const  site_t focal_site = self->sites[focal_site_id];;
    allele_t *restrict site_genotypes = self->sites[site_id].genotypes;
    ancestor_id_hash_t *s, *tmp;

    ancestor[site_id] = 0;
    /* printf("make site %d %d: count = %d\n", focal_site_id, site_id, HASH_COUNT(*consistent_samples)); */
    if (self->sites[site_id].frequency > focal_site.frequency) {
        ones = 0;
        HASH_ITER(hh, *consistent_samples, s, tmp) {
            ones += site_genotypes[s->value];
            /* printf("\t\tsample %d\n", s->value); */
        }
        if (ones == HASH_COUNT(*consistent_samples))  {
            ancestor[site_id] = 1;
        } else if (ones == 0) {
            ancestor[site_id] = 0;
        } else {
            ret = false;
        }
        /* printf("\t\tExamining site %d: ones=%d, zeros=%d\n", (int) site_id, */
        /*         (int) ones, (int) zeros); */
        /* if (remove_inconsistent) { */
        /*     HASH_ITER(hh, *consistent_samples, s, tmp) { */
        /*         if (site_genotypes[s->value] != ancestor[site_id]) { */
        /*             HASH_DEL(*consistent_samples, s); */
        /*         } */
        /*     } */
        /* } */
    }
    return ret;
}

static inline void
ancestor_builder_get_consistent_samples(ancestor_builder_t *self, site_id_t focal_site_id,
        ancestor_id_hash_t **consistent_samples)
{
    size_t j, k;
    ancestor_id_hash_t *s;
    allele_t *restrict genotypes = self->sites[focal_site_id].genotypes;

    k = 0;
    for (j = 0; j < self->num_samples; j++) {
        /* if (self->haplotypes[j * self->num_sites + focal_site_id] == 1) { */
        if (genotypes[j] == 1) {
            s = self->consistent_samples_mem + k;
            k++;
            assert(k <= self->num_samples);
            s->value = (ancestor_id_t) j;
            HASH_ADD(hh, *consistent_samples, value, sizeof(ancestor_id_t), s);
        }
    }
}

/* Build the ancestors for sites in the specified focal sites */
int
ancestor_builder_make_ancestor(ancestor_builder_t *self, size_t num_focal_sites,
        site_id_t *focal_sites, site_id_t *ret_start, site_id_t *ret_end,
        allele_t *ancestor)
{
    int ret = 0;
    int64_t l;
    site_id_t focal_site, start, end;
    site_id_t j;
    size_t num_sites = self->num_sites;
    uint8_t fgt_patterns;
    /* We map the four gametes to the first four bits of the fgt_patterns
     * variable. If all four are set, this is equal to 15. */
    const uint8_t fgt_all = 15;
    bool fgt_break = self->flags & TSI_FGT_BREAK;
    ancestor_id_hash_t *consistent_samples = NULL;
    bool consistent;

    // TODO proper error checking.
    assert(num_focal_sites > 0);
    /* printf("FOCAL SITES (%d)", (int) num_focal_sites); */
    for (j = 0; j < (site_id_t) num_focal_sites; j++) {
        /* printf("%d, ", focal_sites[j]); */
        assert(focal_sites[j] < (site_id_t) self->num_sites);
        assert(self->sites[focal_sites[j]].frequency > 1);
        if (j > 0) {
            assert(focal_sites[j - 1] < focal_sites[j]);
        }
    }
    /* printf("\n"); */

    /* Set any unknown values to -1 */
    memset(ancestor, 0xff, num_sites * sizeof(allele_t));

    /* Fill in the sites within the bounds of the focal sites */
    ancestor_builder_get_consistent_samples(self, focal_sites[0], &consistent_samples);
    ancestor[focal_sites[0]] = 1;
    assert(num_focal_sites == 1);

/*     for (j = 1; j < (site_id_t) num_focal_sites; j++) { */
/*         for (k = focal_sites[j - 1] + 1; k < focal_sites[j]; k++) { */
/*             ancestor_builder_make_site(self, focal_sites[j], k, false, */
/*                     &consistent_samples, ancestor); */
/*         } */
/*         /1* printf("Setting %d: %d\n", focal_sites[j], HASH_COUNT(consistent_samples)); *1/ */
/*         ancestor[focal_sites[j]] = 1; */
/*     } */
/*     /1* printf("DONE INTER\n"); *1/ */
/*     /1* fflush(stdout); *1/ */

    /* Work leftwards from the first focal site */
    fgt_patterns = 0;
    focal_site = focal_sites[0];
    /* printf("focal site = %d\n", focal_site); */
    consistent = true;
    for (l = ((int64_t) focal_site) - 1; l >= 0 && consistent; l--) {
            /* && HASH_COUNT(consistent_samples) > 1; l--) { */
        /* printf("LEFT: l = %d, count = %d fgt=%d\n", (int) l, */
        /*         HASH_COUNT(consistent_samples), fgt_patterns); */
        if (fgt_break) {
            fgt_patterns |= ancestor_builder_get_fgt_patterns(self, focal_site, l,
                    &consistent_samples);
            if (fgt_patterns == fgt_all) {
                break;
            }
        }
        consistent = ancestor_builder_make_site(self, focal_site, l, true,
                &consistent_samples, ancestor);
    }
    start = l + 1;
    HASH_CLEAR(hh, consistent_samples);

    /* Work rightwards from the last focal site */
    consistent = true;
    fgt_patterns = 0;
    focal_site = focal_sites[num_focal_sites - 1];
    ancestor_builder_get_consistent_samples(self, focal_site, &consistent_samples);
    for (l = focal_site + 1; l < (int64_t) num_sites && consistent; l++) {
            /* && HASH_COUNT(consistent_samples) > 1; l++) { */
        /* printf("RIGHT: l = %d, count = %d fgt=%d\n", (int) l, */
        /*         HASH_COUNT(consistent_samples), fgt_patterns); */
        if (fgt_break) {
            fgt_patterns |= ancestor_builder_get_fgt_patterns(self, focal_site, l,
                    &consistent_samples);
            if (fgt_patterns == fgt_all) {
                break;
            }
        }
        consistent = ancestor_builder_make_site(self, focal_site, l, true,
                &consistent_samples, ancestor);
    }
    end = l;
    HASH_CLEAR(hh, consistent_samples);
    *ret_start = start;
    *ret_end = end;
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
        site_id_t b)
{

    /* bool ret = false; */
    /* site_id_t j, k; */
    /* const uint8_t fgt_all = 15; */
    /* uint8_t fgt_patterns; */
    /* ancestor_id_hash_t *consistent_samples = NULL; */
    /* ancestor_builder_get_consistent_samples(self, a, &consistent_samples); */
    /* for (j = a; j <= b && !ret; j++) { */
     /*    if (self->sites[j].frequency > 1) { */
     /*        for (k = j + 1; k <= b; k++) { */
     /*            if (self->sites[k].frequency > 1) { */
     /*                fgt_patterns = ancestor_builder_get_fgt_patterns( */
     /*                        self, j, k, &consistent_samples); */
     /*                if (fgt_patterns == fgt_all) { */
     /*                    /1* printf("Consider %d - %d = %d\n", j, k, fgt_patterns); *1/ */
     /*                    ret = true; */
     /*                    break; */
     /*                } */
     /*            } */
     /*        } */
     /*    } */
    /* } */
    /* HASH_CLEAR(hh, consistent_samples); */
    /* /1* ret = false; *1/ */
    /* return ret; */
    return true;
}

int
ancestor_builder_finalise(ancestor_builder_t *self)
{
    int ret = 0;
    size_t j, k;
    avl_node_t *a;
    pattern_map_t *map_elem;
    site_list_t *s;
    ancestor_descriptor_t *descriptor;
    site_id_t *focal_sites = NULL;
    site_id_t *p;

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
            for (k = 0; k < map_elem->num_sites - 1; k++) {
                if (ancestor_builder_break_ancestor(
                        self, focal_sites[k], focal_sites[k + 1])) {
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
    return ret;
}
