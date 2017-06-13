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
read_sites(const char *input_file, size_t *r_num_samples, size_t *r_num_sites,
        allele_t **r_haplotypes, double **r_positions)
{
    char * line = NULL;
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

#if 0
int
old_main(int argc, char **argv)
{
    int8_t *ancestors = NULL;
    int8_t *h;
    ancestor_id_t *path = NULL;
    site_id_t *mutation_sites = NULL;
    size_t num_ancestors, num_sites, num_mutations;
    size_t j, l;
    ancestor_matcher_t am;
    int ret;
    bool show_matches = false;

    if (argc != 2) {
        fatal_error("usage: main <ancestors-file>");
    }
    read_ancestors(argv[1], &num_ancestors, &num_sites, &ancestors);

    ret = ancestor_matcher_alloc(&am, num_sites, 1024 * 1024);
    if (ret != 0) {
        fatal_error("alloc error");
    }
    path = calloc(num_sites, sizeof(ancestor_id_t));
    mutation_sites = malloc(num_sites * sizeof(site_id_t));
    if (path == NULL || mutation_sites == NULL) {
        fatal_error("alloc error");
    }
    printf("total ancestors = %d\n", (int) num_ancestors);
    for (j = 0; j < num_ancestors; j++) {
        h = ancestors + j * num_sites;
        ret = ancestor_matcher_best_path(&am, h, 0.01, 1e-200, path,
                &num_mutations, mutation_sites);
        if (ret != 0) {
            fatal_error("match error");
        }
        /* printf("Best match = \n"); */
        if (show_matches) {
            printf("%d:\t", (int) num_mutations);
            for (l = 0; l < num_sites; l++) {
                printf("%d", path[l]);
                if (l < num_sites - 1) {
                    printf("\t");
                }
            }
            printf("\n");
        }
        ret = ancestor_matcher_add(&am, h);
        if (ret != 0) {
            fatal_error("add error");
        }
        /* printf("\n"); */
        /* ancestor_matcher_print_state(&am, stdout); */
    }
    /* ancestor_matcher_print_state(&am, stdout); */

    ancestor_matcher_free(&am);
    free(ancestors);
    free(mutation_sites);
    free(path);
    return EXIT_SUCCESS;
}
#endif

static void
run_generate(const char *infile, const char *outfile, int verbose)
{
    size_t num_samples, num_sites;
    allele_t *haplotypes = NULL;
    double *positions = NULL;
    ancestor_builder_t builder;
    int ret;

    printf("Generate: %s %s\n", infile, outfile);

    read_sites(infile, &num_samples, &num_sites, &haplotypes, &positions);
    ret = ancestor_builder_alloc(&builder, num_samples, num_sites, haplotypes);
    if (ret != 0) {
        fatal_error("Builder alloc error.");
    }
    ancestor_builder_print_state(&builder, stdout);


    ancestor_builder_free(&builder);

    if (haplotypes != NULL) {
        free(haplotypes);
    }
    if (positions != NULL) {
        free(positions);
    }
}

static void
run_match(const char *infile, int verbose)
{
    printf("Match\n");

}

int
main(int argc, char** argv)
{
    /* SYNTAX 1: generate [-v] <input-file> <output-file> */
    struct arg_rex *cmd1 = arg_rex1(NULL, NULL, "generate", NULL, REG_ICASE, NULL);
    struct arg_lit *verbose1 = arg_lit0("v", "verbose", NULL);
    struct arg_file *infiles1 = arg_file1(NULL, NULL, NULL, NULL);
    struct arg_file *outfiles1 = arg_file1(NULL, NULL, NULL, NULL);
    struct arg_end *end1 = arg_end(20);
    void* argtable1[] = {cmd1, verbose1, infiles1, outfiles1, end1};
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
        run_generate(infiles1->filename[0], outfiles1->filename[0], verbose1->count);
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
