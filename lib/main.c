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
read_samples(const char *input_file, size_t *r_num_samples, size_t *r_num_sites,
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
read_ancestors(const char *infile, size_t *r_num_ancestors, site_id_t **r_focal_sites)
{
    char *line = NULL;
    size_t len = 0;
    size_t j, num_ancestors;
    const char delimiters[] = " \t";
    char *token;
    site_id_t *focal_sites = NULL;
    FILE *f = fopen(infile, "r");

    if (f == NULL) {
        fatal_error("Cannot open %s: %s", infile, strerror(errno));
    }
    num_ancestors = 0;
    while (getline(&line, &len, f) != -1) {
        num_ancestors++;
    }
    if (fseek(f, 0, 0) != 0) {
        fatal_error("Cannot seek in file");
    }
    focal_sites = malloc(num_ancestors * sizeof(site_id_t));
    if (focal_sites == NULL) {
        fatal_error("Malloc error");
    }
    j = 0;
    while (getline(&line, &len, f) != -1) {
        token = strtok(line, delimiters);
        if (token == NULL) {
            fatal_error("File format error");
        }
        focal_sites[j] = atoi(token);
        j++;
    }
    free(line);
    fclose(f);
    *r_num_ancestors = num_ancestors;
    *r_focal_sites = focal_sites;
}

static void
read_sites(const char *infile, ancestor_store_t *store, size_t num_sites, double *position,
        size_t num_ancestors, site_id_t *focal_site)
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
    assert(num_sites == seg_site[num_segments - 1] + 1);
    ret = ancestor_store_alloc(store, num_sites, position,
            num_ancestors, focal_site,
            num_segments, seg_site, seg_start, seg_end, seg_state);
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
write_sites(ancestor_store_builder_t *store_builder, const char *outfile)
{
    int ret;
    size_t j, num_segments;
    site_id_t *seg_site = NULL;
    ancestor_id_t *seg_start = NULL;
    ancestor_id_t *seg_end = NULL;
    allele_t *seg_state = NULL;
    FILE *out = fopen(outfile, "w");

    num_segments = store_builder->total_segments;
    seg_site = malloc(num_segments * sizeof(site_id_t));
    seg_start = malloc(num_segments * sizeof(ancestor_id_t));
    seg_end = malloc(num_segments * sizeof(ancestor_id_t));
    seg_state = malloc(num_segments * sizeof(allele_t));
    if (seg_site == NULL || seg_start == NULL || seg_end == NULL || seg_state == NULL) {
        fatal_error("Malloc error");
    }
    ret = ancestor_store_builder_dump(store_builder, seg_site, seg_start, seg_end, seg_state);
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
run_generate(const char *sample_file, const char *ancestor_file,
        const char *site_file, int sort, int verbose)
{
    size_t num_samples, num_sites, j, k, num_ancestors;
    allele_t *haplotypes = NULL;
    allele_t *ancestors = NULL;
    site_id_t *permutation = NULL;
    double *positions = NULL;
    site_t *focal_site;
    ancestor_builder_t builder;
    ancestor_sorter_t sorter;
    ancestor_store_builder_t store_builder;
    allele_t *a;
    int ret;
    FILE *ancestor_out = fopen(ancestor_file, "w");

    if (ancestor_out == NULL) {
        fatal_error("cannot open file");
    }

    /* TODO Make sort the default here */
    read_samples(sample_file, &num_samples, &num_sites, &haplotypes, &positions);
    ret = ancestor_builder_alloc(&builder, num_samples, num_sites, positions, haplotypes);
    if (ret != 0) {
        fatal_error("Builder alloc error.");
    }
    ret = ancestor_store_builder_alloc(&store_builder, num_sites, 1024);
    if (ret != 0) {
        fatal_error("store alloc error.");
    }
    permutation = malloc(num_sites * sizeof(site_id_t));
    if (permutation == NULL) {
        fatal_error("permutation alloc error");
    }
    fprintf(ancestor_out, "-1\t-1\n");
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
            focal_site = builder.frequency_classes[j].sites[permutation[k]];
            ret = ancestor_store_builder_add(&store_builder, a);
            if (ret != 0) {
                fatal_error("Error in add ancestor");
            }
            if (verbose > 0) {
                print_ancestor(a, num_sites);
            }
            fprintf(ancestor_out, "%d\t%d\n", focal_site->id,
                    (int) focal_site->frequency);
        }
        tsi_safe_free(ancestors);
        ancestor_sorter_free(&sorter);
    }
    if (verbose > 0) {
        ancestor_store_builder_print_state(&store_builder, stdout);
    }
    write_sites(&store_builder, site_file);


    if (fclose(ancestor_out) != 0) {
        fatal_error("Close error");
    }
    ancestor_builder_free(&builder);
    ancestor_store_builder_free(&store_builder);
    tsi_safe_free(haplotypes);
    tsi_safe_free(positions);
    tsi_safe_free(ancestors);
    tsi_safe_free(permutation);

}

static void
output_ts(tree_sequence_builder_t *ts_builder)
{
    int ret = 0;
    size_t j, k, offset;
    size_t num_edgesets = ts_builder->num_edgesets;
    size_t num_children = ts_builder->num_children;
    size_t num_mutations = ts_builder->num_mutations;
    double *left = malloc(num_edgesets * sizeof(double));
    double *right = malloc(num_edgesets * sizeof(double));
    ancestor_id_t *parent = malloc(num_edgesets * sizeof(ancestor_id_t));
    ancestor_id_t *children = malloc(num_children * sizeof(ancestor_id_t));
    uint32_t *children_length = malloc(num_edgesets * sizeof(uint32_t));
    site_id_t *site = malloc(num_mutations * sizeof(site_id_t));
    ancestor_id_t *node = malloc(num_mutations * sizeof(ancestor_id_t));
    allele_t *derived_state = malloc(num_mutations * sizeof(allele_t));

    if (left == NULL || right == NULL || parent == NULL || children == NULL
            || children_length == NULL || site == NULL || node == NULL
            || derived_state == NULL) {
        fatal_error("malloc error\n");
    }
    ret = tree_sequence_builder_dump_edgesets(ts_builder,
            left, right, parent, children, children_length);
    if (ret != 0) {
        fatal_error("dump error");
    }
    printf("EDGESETS\n");
    offset = 0;
    for (j = 0; j < num_edgesets; j++) {
        printf("%.3f\t%.3f\t%d\t", left[j], right[j], parent[j]);
        for (k = 0; k < children_length[j]; k++) {
            printf("%d", children[offset]);
            if (k < children_length[j] - 1) {
                printf(",");
            }
            offset++;
        }
        printf("\n");
    }
    ret = tree_sequence_builder_dump_mutations(ts_builder, site, node, derived_state);
    printf("MUTATIONS\n");
    for (j = 0; j < num_mutations; j++) {
        printf("%d\t%d\t%d\n", site[j], node[j], derived_state[j]);
    }
    free(left);
    free(right);
    free(parent);
    free(children);
    free(children_length);
    free(site);
    free(node);
    free(derived_state);
}

static void
run_match(const char *sample_file, const char *ancestor_file, const char *site_file,
        int verbose)
{
    int ret;
    ancestor_store_t store;
    ancestor_matcher_t matcher;
    traceback_t traceback;
    tree_sequence_builder_t ts_builder;
    ancestor_id_t end_site_value, sample_id;
    site_id_t *focal_sites = NULL;
    site_id_t start, focal, end;
    allele_t *ancestor = NULL;
    size_t j, l, num_ancestors;
    double mutation_rate = 1e-200;
    allele_t *samples = NULL;
    allele_t *sample;
    double *positions;
    size_t num_samples, num_sites;

    read_samples(sample_file, &num_samples, &num_sites, &samples, &positions);
    read_ancestors(ancestor_file, &num_ancestors, &focal_sites);
    read_sites(site_file, &store, num_sites, positions, num_ancestors, focal_sites);
    ret = ancestor_matcher_alloc(&matcher, &store, 0.01);
    if (ret != 0) {
        fatal_error("alloc error");
    }
    ret = traceback_alloc(&traceback, num_sites, 8192);
    if (ret != 0) {
        fatal_error("alloc error");
    }
    ret = tree_sequence_builder_alloc(&ts_builder, &store, num_samples,
            8192, 8192, num_sites / 4);
    if (ret != 0) {
        fatal_error("alloc error");
    }
    ancestor = malloc(store.num_sites * sizeof(allele_t));
    if (ancestor == NULL) {
        fatal_error("alloc error");
    }
    if (verbose > 0) {
        ancestor_store_print_state(&store, stdout);
    }
    /* Copy ancestors */
    for (j = store.num_ancestors - 1; j > 0; j--) {
        ret = ancestor_store_get_ancestor(&store, j, ancestor, &start, &focal, &end);
        if (ret != 0) {
            fatal_error("get_ancestor error");
        }
        if (verbose > 0) {
            printf("ancestor %d: %d= \t", (int) j, focal);
            for (l = 0; l < store.num_sites; l++) {
                if (ancestor[l] == -1) {
                    printf("*");
                } else {
                    printf("%d", ancestor[l]);
                }
            }
            printf("\n");
        }
        ret = ancestor_matcher_best_path(&matcher, j, ancestor, start,
                end, focal, mutation_rate, &traceback, &end_site_value);
        if (ret != 0) {
            fatal_error("match error");
        }
        ret = tree_sequence_builder_update(&ts_builder, j, ancestor, start, end,
                end_site_value, &traceback);
        if (ret != 0) {
            fatal_error("update error");
        }
        ret = traceback_reset(&traceback);
        if (ret != 0) {
            fatal_error("traceback reset error");
        }
    }

    /* Copy samples */
    for (j = 0; j < num_samples; j++) {
        sample = samples + j * num_sites;
        sample_id = num_ancestors + j;
        ret = ancestor_matcher_best_path(&matcher, sample_id, sample,
                0, num_sites, -1, mutation_rate, &traceback, &end_site_value);
        if (ret != 0) {
            fatal_error("match error");
        }
        ret = tree_sequence_builder_update(&ts_builder, sample_id, sample,
                0, num_sites, end_site_value, &traceback);
        if (ret != 0) {
            fatal_error("update error");
        }
        ret = traceback_reset(&traceback);
        if (ret != 0) {
            fatal_error("traceback reset error");
        }
    }

    if (verbose > 0) {
        tree_sequence_builder_print_state(&ts_builder, stdout);
    }

    output_ts(&ts_builder);

    tree_sequence_builder_free(&ts_builder);
    ancestor_matcher_free(&matcher);
    ancestor_store_free(&store);
    traceback_free(&traceback);
    free(focal_sites);
    free(ancestor);
    free(positions);
    free(samples);
}

int
main(int argc, char** argv)
{
    /* SYNTAX 1: generate [-v] <input-file> <output-file> */
    struct arg_rex *cmd1 = arg_rex1(NULL, NULL, "generate", NULL, REG_ICASE, NULL);
    struct arg_lit *verbose1 = arg_lit0("v", "verbose", NULL);
    struct arg_lit *sort1 = arg_lit0("s", "sort", NULL);
    struct arg_file *sample_file1 = arg_file1(NULL, NULL, NULL, NULL);
    struct arg_file *ancestor_file1 = arg_file1(NULL, NULL, NULL, NULL);
    struct arg_file *site_file1 = arg_file1(NULL, NULL, NULL, NULL);
    struct arg_end *end1 = arg_end(20);
    void* argtable1[] = {cmd1, verbose1, sort1, sample_file1, ancestor_file1, site_file1, end1};
    int nerrors1;

    /* SYNTAX 2: match [-v] */
    struct arg_rex *cmd2 = arg_rex1(NULL, NULL, "match", NULL, REG_ICASE, NULL);
    struct arg_lit *verbose2 = arg_lit0("v", "verbose", NULL);
    struct arg_file *sample_file2 = arg_file1(NULL, NULL, NULL, NULL);
    struct arg_file *ancestor_file2 = arg_file1(NULL, NULL, NULL, NULL);
    struct arg_file *site_file2 = arg_file1(NULL, NULL, NULL, NULL);
    struct arg_end *end2 = arg_end(20);
    void* argtable2[] = {cmd2, verbose2, sample_file2, ancestor_file2, site_file2, end2};
    int nerrors2;

    int exitcode = EXIT_SUCCESS;
    const char *progname = "main";

    nerrors1 = arg_parse(argc, argv, argtable1);
    nerrors2 = arg_parse(argc, argv, argtable2);

    if (nerrors1 == 0) {
        run_generate(sample_file1->filename[0], ancestor_file1->filename[0],
                site_file1->filename[0], sort1->count, verbose1->count);
    } else if (nerrors2 == 0) {
        run_match(sample_file2->filename[0], ancestor_file2->filename[0],
                site_file2->filename[0], verbose2->count);
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
