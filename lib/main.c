#define _GNU_SOURCE

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdint.h>
#include <errno.h>
#include <string.h>
#include <stdbool.h>

#include <regex.h>
#include "argtable3.h"

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

static void
print_ancestor(allele_t *a, size_t num_sites)
{
    size_t l;

    for (l = 0; l < num_sites; l++) {
        if (a[l] == -1) {
            printf("*");
        } else {
            printf("%d", a[l]);
        }
    }
    printf("\n");
}

static void
read_sites(const char *input_file, size_t *r_num_samples, size_t *r_num_sites,
        allele_t **r_haplotypes, double **r_positions)
{
    char *line = NULL;
    size_t len = 0;
    size_t j, k;
    size_t num_line_samples;
    size_t num_samples = (size_t) -1;
    size_t num_sites = 0;
    const char delimiters[] = " \t";
    char *token;
    allele_t *haplotypes = NULL;
    double *position = NULL;
    FILE *f = fopen(input_file, "r");

    if (f == NULL) {
        fatal_error("Cannot open %s: %s", input_file, strerror(errno));
    }
    while (getline(&line, &len, f) != -1) {
        /* read the number of tokens */
        token = strtok(line, delimiters);
        if (token == NULL) {
            fatal_error("File format error");
        }
        token = strtok(NULL, delimiters);
        if (token == NULL) {
            fatal_error("File format error");
        }
        num_line_samples = strlen(token) - 1;
        if (num_samples == (size_t) -1) {
            num_samples = num_line_samples;
        } else if (num_samples != num_line_samples) {
            fatal_error("Bad input: line lengths not equal");
        }
        num_sites++;
    }
    if (fseek(f, 0, 0) != 0) {
        fatal_error("Cannot seek in file");
    }

    haplotypes = malloc(num_samples * num_sites * sizeof(allele_t));
    position = malloc(num_sites * sizeof(double));
    if (haplotypes == NULL || position == NULL) {
        fatal_error("No memory");
    }
    k = 0;
    while (getline(&line, &len, f) != -1) {
        token = strtok(line, delimiters);
        if (token == NULL) {
            fatal_error("File format error");
        }
        position[k] = atof(token);
        token = strtok(NULL, delimiters);
        if (token == NULL) {
            fatal_error("File format error");
        }
        for (j = 0; j < num_samples; j++) {
            haplotypes[j * num_sites + k] = (allele_t) ((int) token[j] - '0');
        }
        k++;
    }
    free(line);
    fclose(f);

    *r_num_samples = num_samples;
    *r_num_sites = num_sites;
    *r_haplotypes = haplotypes;
    *r_positions = position;
}

static void
read_ancestors(ancestor_store_t *store, const char *infile)
{
    char *line = NULL;
    size_t len = 0;
    size_t j;
    size_t num_segments = 0;
    const char delimiters[] = " \t";
    char *token;
    site_id_t *seg_site = NULL;
    ancestor_id_t *seg_start = NULL;
    ancestor_id_t *seg_end = NULL;
    allele_t *seg_state = NULL;
    FILE *f = fopen(infile, "r");
    int ret;

    if (f == NULL) {
        fatal_error("Cannot open %s: %s", infile, strerror(errno));
    }
    while (getline(&line, &len, f) != -1) {
        num_segments++;
    }
    if (fseek(f, 0, 0) != 0) {
        fatal_error("Cannot seek in file");
    }
    seg_site = malloc(num_segments * sizeof(site_id_t));
    seg_start = malloc(num_segments * sizeof(ancestor_id_t));
    seg_end = malloc(num_segments * sizeof(ancestor_id_t));
    seg_state = malloc(num_segments * sizeof(allele_t));
    if (seg_site == NULL || seg_start == NULL || seg_end == NULL || seg_state == NULL) {
        fatal_error("Malloc error");
    }
    j = 0;
    while (getline(&line, &len, f) != -1) {
        token = strtok(line, delimiters);
        if (token == NULL) {
            fatal_error("File format error");
        }
        seg_site[j] = atoi(token);
        token = strtok(NULL, delimiters);
        if (token == NULL) {
            fatal_error("File format error");
        }
        seg_start[j] = atoi(token);
        token = strtok(NULL, delimiters);
        if (token == NULL) {
            fatal_error("File format error");
        }
        seg_end[j] = atoi(token);
        token = strtok(NULL, delimiters);
        if (token == NULL) {
            fatal_error("File format error");
        }
        seg_state[j] = atoi(token);
        j++;
    }

    ret = ancestor_store_alloc(store, seg_site[num_segments - 1] + 1);
    if (ret != 0) {
        fatal_error("store alloc error");
    }
    ret = ancestor_store_load(store, num_segments, seg_site, seg_start, seg_end, seg_state);
    if (ret != 0) {
        fatal_error("store load error");
    }

    free(line);
    free(seg_site);
    free(seg_start);
    free(seg_end);
    free(seg_state);
    fclose(f);
}

static void
write_ancestors(ancestor_store_t *store, const char *outfile)
{
    int ret;
    size_t j, num_segments;
    site_id_t *seg_site = NULL;
    ancestor_id_t *seg_start = NULL;
    ancestor_id_t *seg_end = NULL;
    allele_t *seg_state = NULL;
    FILE *out = fopen(outfile, "w");

    num_segments = ancestor_store_get_num_segments(store);
    seg_site = malloc(num_segments * sizeof(site_id_t));
    seg_start = malloc(num_segments * sizeof(ancestor_id_t));
    seg_end = malloc(num_segments * sizeof(ancestor_id_t));
    seg_state = malloc(num_segments * sizeof(allele_t));
    if (seg_site == NULL || seg_start == NULL || seg_end == NULL || seg_state == NULL) {
        fatal_error("Malloc error");
    }
    ret = ancestor_store_dump(store, seg_site, seg_start, seg_end, seg_state);
    if (ret != 0) {
        fatal_error("Dump error");
    }
    if (out == NULL) {
        fatal_error("Error opening file");
    }
    for (j = 0; j < num_segments; j++) {
        fprintf(out, "%d\t%d\t%d\t%d\n", seg_site[j], seg_start[j], seg_end[j], seg_state[j]);
    }
    if (fclose(out) != 0) {
        fatal_error("Close error");

    }

    tsi_safe_free(seg_site);
    tsi_safe_free(seg_start);
    tsi_safe_free(seg_end);
    tsi_safe_free(seg_state);
}

static void
run_generate(const char *infile, const char *outfile, int sort, int verbose)
{
    size_t num_samples, num_sites, j, k, num_ancestors;
    allele_t *haplotypes = NULL;
    allele_t *ancestors = NULL;
    site_id_t *permutation = NULL;
    double *positions = NULL;
    site_t *focal_site;
    ancestor_builder_t builder;
    ancestor_sorter_t sorter;
    ancestor_store_t store;
    allele_t *a;
    int ret;

    read_sites(infile, &num_samples, &num_sites, &haplotypes, &positions);
    ret = ancestor_builder_alloc(&builder, num_samples, num_sites, positions, haplotypes);
    if (ret != 0) {
        fatal_error("Builder alloc error.");
    }
    ret = ancestor_store_alloc(&store, num_sites);
    if (ret != 0) {
        fatal_error("store alloc error.");
    }
    ret = ancestor_store_init_build(&store, 16);
    if (ret != 0) {
        fatal_error("store init error.");
    }
    if (verbose > 0) {
        ancestor_builder_print_state(&builder, stdout);
    }
    permutation = malloc(num_sites * sizeof(site_id_t));
    if (permutation == NULL) {
        fatal_error("permutation alloc error");
    }

    for (j = 0; j < builder.num_frequency_classes; j++) {
        num_ancestors = builder.frequency_classes[j].num_sites;
        if (verbose > 0) {
            printf("Generating for frequency class %d: num_ancestors = %d\n",
                    (int) j, (int) num_ancestors);
        }
        ancestors = malloc(num_ancestors * num_sites * sizeof(allele_t));
        if (ancestors == NULL) {
            fatal_error("Alloc ancestors");
        }
        ret = ancestor_sorter_alloc(&sorter, num_ancestors, num_sites, ancestors,
                permutation);
        if (ret != 0) {
            fatal_error("Ancestor sorter alloc error");
        }
        for (k = 0; k < num_ancestors; k++) {
            focal_site = builder.frequency_classes[j].sites[k];
            a = ancestors + k * num_sites;
            /* printf("\tfocal site = %d\n", focal_site->id); */
            ret = ancestor_builder_make_ancestor(&builder, focal_site->id, a);
            if (ret != 0) {
                fatal_error("Error in make ancestor");
            }
        }
        if (sort > 0) {
            ret = ancestor_sorter_sort(&sorter);
            if (ret != 0) {
                fatal_error("Error sorting");
            }
        }
        for (k = 0; k < num_ancestors; k++) {
            a = ancestors + permutation[k] * num_sites;
            ret = ancestor_store_add(&store, a);
            if (ret != 0) {
                fatal_error("Error in add ancestor");
            }
            if (verbose > 0) {
                print_ancestor(a, num_sites);
            }
        }
        tsi_safe_free(ancestors);
        ancestor_sorter_free(&sorter);
    }
    if (verbose > 0) {
        ancestor_store_print_state(&store, stdout);
    }
    write_ancestors(&store, outfile);

    ancestor_builder_free(&builder);
    ancestor_store_free(&store);
    tsi_safe_free(haplotypes);
    tsi_safe_free(positions);
    tsi_safe_free(ancestors);
    tsi_safe_free(permutation);
}

static void
run_match(const char *infile, int verbose)
{
    int ret;
    ancestor_store_t store;
    ancestor_matcher_t matcher;
    ancestor_id_t *path = NULL;
    site_id_t *mutation_sites = NULL;
    allele_t *ancestor = NULL;
    size_t j, l, num_mutations;

    read_ancestors(&store, infile);
    ret = ancestor_matcher_alloc(&matcher, &store, 0.01, 1e-200);
    if (ret != 0) {
        fatal_error("alloc error");
    }
    if (verbose > 0) {
        ancestor_matcher_print_state(&matcher, stdout);
    }

    path = calloc(store.num_sites, sizeof(ancestor_id_t));
    mutation_sites = malloc(store.num_sites * sizeof(site_id_t));
    ancestor = malloc(store.num_sites * sizeof(allele_t));
    if (path == NULL || mutation_sites == NULL || ancestor == NULL) {
        fatal_error("alloc error");
    }
    for (j = 1; j < store.num_ancestors; j++) {
        ret = ancestor_store_get_ancestor(&store, j, ancestor);
        if (ret != 0) {
            fatal_error("get_ancestor error");
        }
        if (verbose > 0) {
            printf("ancestor %d = \t", (int) j);
            for (l = 0; l < store.num_sites; l++) {
                if (ancestor[l] == -1) {
                    printf("*");
                } else {
                    printf("%d", ancestor[l]);
                }
            }
            printf("\n");
        }
        ret = ancestor_matcher_best_path(&matcher, j, ancestor, path,
                &num_mutations, mutation_sites);
        if (ret != 0) {
            fatal_error("match error");
        }
        /* printf("Best match = \n"); */
        if (verbose > 0) {
            printf("%d:\t", (int) num_mutations);
            for (l = 0; l < store.num_sites; l++) {
                printf("%d", path[l]);
                if (l < store.num_sites - 1) {
                    printf("\t");
                }
            }
            printf("\n");
        }
    }

    ancestor_matcher_free(&matcher);
    ancestor_store_free(&store);
    free(path);
    free(mutation_sites);
    free(ancestor);
}

int
main(int argc, char** argv)
{
    /* SYNTAX 1: generate [-v] <input-file> <output-file> */
    struct arg_rex *cmd1 = arg_rex1(NULL, NULL, "generate", NULL, REG_ICASE, NULL);
    struct arg_lit *verbose1 = arg_lit0("v", "verbose", NULL);
    struct arg_lit *sort1 = arg_lit0("s", "sort", NULL);
    struct arg_file *infiles1 = arg_file1(NULL, NULL, NULL, NULL);
    struct arg_file *outfiles1 = arg_file1(NULL, NULL, NULL, NULL);
    struct arg_end *end1 = arg_end(20);
    void* argtable1[] = {cmd1, verbose1, sort1, infiles1, outfiles1, end1};
    int nerrors1;

    /* SYNTAX 2: match [-v] <input-file> */
    struct arg_rex *cmd2 = arg_rex1(NULL, NULL, "match", NULL, REG_ICASE, NULL);
    struct arg_lit *verbose2 = arg_lit0("v", "verbose", NULL);
    struct arg_file *infiles2 = arg_file1(NULL, NULL, NULL, NULL);
    struct arg_end *end2 = arg_end(20);
    void* argtable2[] = {cmd2, verbose2, infiles2, end2};
    int nerrors2;

    int exitcode = EXIT_SUCCESS;
    const char *progname = "main";

    nerrors1 = arg_parse(argc, argv, argtable1);
    nerrors2 = arg_parse(argc, argv, argtable2);

    if (nerrors1 == 0) {
        run_generate(infiles1->filename[0], outfiles1->filename[0], sort1->count,
                verbose1->count);
    } else if (nerrors2 == 0) {
        run_match(infiles2->filename[0], verbose2->count);
    } else {
        /* We get here if the command line matched none of the possible syntaxes */
        if (cmd1->count > 0) {
            arg_print_errors(stdout, end1, progname);
            printf("usage: %s ", progname);
            arg_print_syntax(stdout, argtable1, "\n");
        } else if (cmd2->count > 0) {
            arg_print_errors(stdout, end2, progname);
            printf("usage: %s ", progname);
            arg_print_syntax(stdout, argtable2, "\n");
        } else {
            /* no correct cmd literals were given, so we cant presume which syntax was intended */
            printf("%s: missing command.\n",progname);
            printf("usage 1: %s ", progname);  arg_print_syntax(stdout, argtable1, "\n");
            printf("usage 2: %s ", progname);  arg_print_syntax(stdout, argtable2, "\n");
        }
    }

    arg_freetable(argtable1, sizeof(argtable1) / sizeof(argtable1[0]));
    arg_freetable(argtable2, sizeof(argtable2) / sizeof(argtable2[0]));

    return exitcode;
}
