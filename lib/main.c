#define _GNU_SOURCE

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdint.h>
#include <errno.h>
#include <string.h>

#include "tsinfer.h"

static void
fatal_error(const char *msg, ...)
{
    va_list argp;
    fprintf(stderr, "infer:");
    va_start(argp, msg);
    vfprintf(stderr, msg, argp);
    va_end(argp);
    fprintf(stderr, "\n");
    exit(EXIT_FAILURE);
}

#if 0
static reference_panel_t *
build_reference_panel(const char *input_file)
{
    int ret;
    reference_panel_t *panel = NULL;
    char * line = NULL;
    size_t len = 0;
    size_t j, k;
    int64_t read;
    int64_t num_samples = 0;
    int64_t num_sites = -1;
    uint8_t *sample_haplotypes;
    double *positions;
    FILE *f = fopen(input_file, "r");

    if (f == NULL) {
        fatal_error("Cannot open %s: %s", input_file, strerror(errno));
    }
    panel = malloc(sizeof(reference_panel_t));
    if (panel == NULL) {
        fatal_error("No memory");
    }
    while ((read = getline(&line, &len, f)) != -1) {
        if (num_sites == -1) {
            num_sites = read - 1;
        } else if (num_sites != read - 1) {
            fatal_error("Bad input: line lengths not equal");
        }
        num_samples++;
    }
    if (fseek(f, 0, 0) != 0) {
        fatal_error("Cannot seek in file");
    }
    sample_haplotypes = malloc(num_samples * num_sites * sizeof(uint8_t));
    positions = malloc(num_sites * sizeof(double));
    if (sample_haplotypes == NULL || positions == NULL) {
        fatal_error("malloc failed");
    }

    j = 0;
    while ((read = getline(&line, &len, f)) != -1) {
        for (k = 0; k < read - 1; k++) {
            assert(j * num_samples + k < num_samples * num_sites);
            sample_haplotypes[j * num_sites + k] = (uint8_t) (line[k] - '0');
        }
        j++;
    }
    /* TODO read in variant positions properly. */
    for (j = 0; j < num_sites; j++) {
        positions[j] = j;
    }
    ret = reference_panel_alloc(panel, (uint32_t) num_samples, (uint32_t) num_sites,
            sample_haplotypes, positions);
    if (ret != 0) {
        fatal_error("Error allocing reference panel: %d", ret);
    }
    free(line);
    fclose(f);
    free(sample_haplotypes);
    free(positions);
    return panel;
}

static void
thread(reference_panel_t *panel, double recombination_rate, int verbose)
{
    uint32_t *p, j, l;
    uint32_t N = panel->num_haplotypes;
    uint32_t n = panel->num_samples;
    uint32_t *mutations, num_mutations;
    threader_t threader;
    int ret;

    p = malloc(panel->num_sites * sizeof(uint32_t));
    mutations = malloc(panel->num_sites * sizeof(uint32_t));
    if (p == NULL || mutations == NULL) {
        fatal_error("no memory");
    }
    ret = threader_alloc(&threader, panel);
    if (ret != 0) {
        fatal_error("Error allocing threader");
    }
    for (j = N - 2; j >= n; j--) {
        ret = threader_run(&threader, j, N - j - 1, recombination_rate, 0, p,
                &num_mutations, mutations);
        if (ret != 0) {
            fatal_error("cannot run thread");
        }
        if (verbose > 0) {
            printf("Threaded %d into %d haplotypes\n\t", j, N - j - 1);
            for (l = 0; l < panel->num_sites; l++) {
                printf("%d ", p[l]);
            }
            printf("\n\t");
            printf("%d mutations @", num_mutations);
            for (l = 0; l < num_mutations; l++) {
                printf("%d ", mutations[l]);
            }
            printf("\n");
        }
    }
    for (j = 0; j < n; j++) {
        ret = threader_run(&threader, j, N - n, recombination_rate, 0, p,
                &num_mutations, mutations);
        if (ret != 0) {
            fatal_error("cannot run thread");
        }
        if (verbose > 0) {
            printf("Threaded %d into %d haplotypes\n\t", j, N - n);
            for (l = 0; l < panel->num_sites; l++) {
                printf("%d ", p[l]);
            }
            printf("\n\t");
            printf("%d mutations @", num_mutations);
            for (l = 0; l < num_mutations; l++) {
                printf("%d ", mutations[l]);
            }
            printf("\n");
        }
    }
    threader_free(&threader);
    free(p);
    free(mutations);
}

#endif

int
main(int argc, char **argv)
{
    /* reference_panel_t *panel; */
    /* double rho; */
    /* int verbose = 1; */

    /* if (argc != 3) { */
    /*     fatal_error("usage: main <samples-file> <rho> "); */
    /* } */
    /* panel = build_reference_panel(argv[1]); */
    /* rho = atof(argv[2]); */
    /* printf("Read panel with %d samples, %d haplotypes and %d sites\n", */
    /*         (int) panel->num_samples, (int) panel->num_haplotypes, (int) panel->num_sites); */
    /* if (verbose > 0) { */
    /*     reference_panel_print_state(panel); */
    /* } */
    /* thread(panel, rho, verbose); */
    /* reference_panel_free(panel); */
    /* free(panel); */
    ancestor_matcher_t am;
    int ret = ancestor_matcher_alloc(&am, 10, 1);

    if (ret != 0) {
        fatal_error("alloc error");
    }
    ancestor_matcher_print_state(&am, stdout);
    ancestor_matcher_free(&am);

    return EXIT_SUCCESS;
}
