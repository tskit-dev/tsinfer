/*
 * Li and Stephens haplotype matching to infer tree sequences.
 */
#include <stdint.h>
#include <stdlib.h>

typedef struct {
    uint8_t *haplotypes;
    double *positions;
    uint32_t **new_mutations;
    uint32_t *num_new_mutations;
    uint32_t *new_mutations_mem;
    size_t num_samples;
    size_t num_haplotypes;
    size_t num_sites;
} reference_panel_t;

typedef struct {
    reference_panel_t *reference_panel;
    double *V;
    double *V_next;
    uint32_t *T;
} threader_t;

int reference_panel_alloc(reference_panel_t *self,
        size_t num_samples, size_t num_sites, uint8_t *sample_haplotypes,
        double *positions);
int reference_panel_free(reference_panel_t *self);
int reference_panel_print_state(reference_panel_t *self);

int threader_alloc(threader_t *self, reference_panel_t *reference_panel);
int threader_free(threader_t *self);
int threader_run(threader_t *self,
        uint32_t haplotype_index, uint32_t panel_size, double recombination_rate,
        double error_probablilty, uint32_t *P, uint32_t *num_mutations,
        uint32_t *mutations);
