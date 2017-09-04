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
output_ts(tree_sequence_builder_t *ts_builder)
{
    int ret = 0;
    size_t j;
    size_t num_nodes = ts_builder->num_nodes;
    size_t num_edges = ts_builder->num_edges;
    size_t num_mutations = ts_builder->num_mutations;
    double *time = malloc(num_nodes * sizeof(double));
    uint32_t *flags = malloc(num_nodes * sizeof(uint32_t));
    double *left = malloc(num_edges * sizeof(double));
    double *right = malloc(num_edges * sizeof(double));
    ancestor_id_t *parent = malloc(num_edges * sizeof(ancestor_id_t));
    ancestor_id_t *children = malloc(num_edges * sizeof(ancestor_id_t));
    site_id_t *site = malloc(num_mutations * sizeof(site_id_t));
    ancestor_id_t *node = malloc(num_mutations * sizeof(ancestor_id_t));
    allele_t *derived_state = malloc(num_mutations * sizeof(allele_t));

    if (time == NULL || flags == NULL
            || left == NULL || right == NULL || parent == NULL || children == NULL
            || site == NULL || node == NULL || derived_state == NULL) {
        fatal_error("malloc error\n");
    }
    ret = tree_sequence_builder_dump_nodes(ts_builder, flags, time);
    if (ret != 0) {
        fatal_error("dump error");
    }
    printf("NODES\n");
    for (j = 0; j < num_nodes; j++) {
        printf("%d\t%d\t%f\n", (int) j, flags[j], time[j]);
    }
    ret = tree_sequence_builder_dump_edges(ts_builder, left, right, parent, children);
    if (ret != 0) {
        fatal_error("dump error");
    }
    printf("EDGES\n");
    for (j = 0; j < num_edges; j++) {
        printf("%.3f\t%.3f\t%d\t%d\n", left[j], right[j], parent[j], children[j]);
    }
    ret = tree_sequence_builder_dump_mutations(ts_builder, site, node, derived_state);
    printf("MUTATIONS\n");
    for (j = 0; j < num_mutations; j++) {
        printf("%d\t%d\t%d\n", site[j], node[j], derived_state[j]);
    }
    free(time);
    free(flags);
    free(left);
    free(right);
    free(parent);
    free(children);
    free(site);
    free(node);
    free(derived_state);
}

static void
run_generate(const char *sample_file, const char *ancestor_file,
        const char *site_file, int verbose)
{
    size_t num_samples, num_sites, j, k, l, num_ancestors;
    allele_t *haplotypes = NULL;
    double *positions = NULL;
    site_id_t *focal_sites;
    size_t num_focal_sites;
    ancestor_builder_t ancestor_builder;
    tree_sequence_builder_t ts_builder;
    ancestor_matcher_t matcher;
    allele_t *a;
    size_t age;
    int ret;
    size_t total_edges;
    edge_t *edges_buffer = NULL;
    edge_t *output_edges;
    size_t num_output_edges;
    size_t max_edges = 1024;
    site_mutation_t *site_mutation_buffer = NULL;
    size_t max_site_mutations = 1024;
    size_t num_site_mutations;
    node_id_t child;

    read_samples(sample_file, &num_samples, &num_sites, &haplotypes, &positions);
    ret = ancestor_builder_alloc(&ancestor_builder, num_samples, num_sites, positions, haplotypes);
    if (ret != 0) {
        fatal_error("Builder alloc error.");
    }
    num_ancestors = ancestor_builder.num_ancestors;
    ret = tree_sequence_builder_alloc(&ts_builder, num_sites, num_ancestors + 1, 65536);
    if (ret != 0) {
        fatal_error("alloc error");
    }
    ret = ancestor_matcher_alloc(&matcher, &ts_builder, 1e-8);
    if (ret != 0) {
        fatal_error("alloc error");
    }

    if (verbose > 0) {
        ancestor_builder_print_state(&ancestor_builder, stdout);
        ancestor_matcher_print_state(&matcher, stdout);
    }

    age = ancestor_builder.num_frequency_classes + 1;
    ret = tree_sequence_builder_update(&ts_builder, 1, age, 0, NULL, 0, NULL);
    if (ret != 0) {
        fatal_error("initial update");
    }
    a = malloc(num_sites * sizeof(allele_t));
    edges_buffer = malloc(max_edges * sizeof(edge_t));
    site_mutation_buffer = malloc(max_site_mutations * sizeof(site_mutation_t));
    if (a == NULL || edges_buffer == NULL || site_mutation_buffer == NULL) {
        fatal_error("ancestor alloc");
    }
    child = 1;
    for (j = 0; j < ancestor_builder.num_frequency_classes; j++) {
        age--;
        num_ancestors = ancestor_builder.frequency_classes[j].num_ancestors;
        if (verbose > 0) {
            printf("Generating for frequency class %d: age = %d num_ancestors = %d\n",
                    (int) j, (int) age, (int) num_ancestors);
        }
        total_edges = 0;
        num_site_mutations = 0;
        for (k = 0; k < num_ancestors; k++) {
            focal_sites = ancestor_builder.frequency_classes[j].ancestor_focal_sites[k];
            num_focal_sites = ancestor_builder.frequency_classes[j].num_ancestor_focal_sites[k];
            ret = ancestor_builder_make_ancestor(&ancestor_builder, num_focal_sites, focal_sites, a);
            if (ret != 0) {
                fatal_error("Error in make ancestor");
            }
            for (l = 0; l < num_focal_sites; l++) {
                assert(num_site_mutations < max_site_mutations);
                site_mutation_buffer[num_site_mutations].node = child;
                site_mutation_buffer[num_site_mutations].site = focal_sites[l];
                num_site_mutations++;
                assert(a[focal_sites[l]] == 1);
                a[focal_sites[l]] = 0;
            }
            /* printf("Got ancestor (%d-%d), %d, %d\n", start, end, (int) num_older_ancestors, */
            /*         (int) num_focal_sites); */
            ret = ancestor_matcher_find_path(&matcher, a, child, &num_output_edges, &output_edges);
            if (ret != 0) {
                fatal_error("find_path error");
            }
            if (total_edges + num_output_edges > max_edges) {
                fatal_error("out of edge buffer space\n");
            }
            memcpy(edges_buffer + total_edges, output_edges, num_output_edges * sizeof(edge_t));
            total_edges += num_output_edges;
            if (verbose > 0) {
                printf("ancestor %d:\t", (int) child);
                for (l = 0; l < num_sites; l++) {
                    printf("%d", a[l]);
                }
                printf("\n");
                printf("\tnum_focal=%d", (int) num_focal_sites);
                printf("\tedges = (%d):: \t", (int) num_output_edges);
                for (l = 0; l < num_output_edges; l++) {
                    printf("(%d, %d, %d, %d), ", output_edges[l].left,
                            output_edges[l].right, output_edges[l].parent,
                            output_edges[l].child);
                }
                printf("\n");
            }
            child++;
        }
        ret = tree_sequence_builder_update(&ts_builder, num_ancestors, age,
                total_edges, edges_buffer, num_site_mutations, site_mutation_buffer);
        if (ret != 0) {
            fatal_error("builder update");
        }
        /* tree_sequence_builder_print_state(&ts_builder, stdout); */
    }

    /* /1* Copy samples *1/ */
    /* for (j = 0; j < num_samples; j++) { */
    /*     sample = samples + j * num_sites; */
    /*     sample_id = num_ancestors + j; */
    /*     ret = ancestor_matcher_best_path(&matcher, num_ancestors, sample, */
    /*             0, num_sites, 0, NULL, mutation_rate, &traceback); */
    /*     if (ret != 0) { */
    /*         fatal_error("match error"); */
    /*     } */
    /*     ret = tree_sequence_builder_update(&ts_builder, sample_id, sample, */
    /*             0, num_sites, &traceback); */
    /*     if (ret != 0) { */
    /*         fatal_error("update error"); */
    /*     } */
    /*     ret = traceback_reset(&traceback); */
    /*     if (ret != 0) { */
    /*         fatal_error("traceback reset error"); */
    /*     } */
    /*     /1* printf("COPIED SAMPLE %d->%d\n", (int) j, (int) sample_id); *1/ */
    /*     /1* tree_sequence_builder_print_state(&ts_builder, stdout); *1/ */
    /* } */

    output_ts(&ts_builder);

    ancestor_builder_free(&ancestor_builder);
    tree_sequence_builder_free(&ts_builder);
    ancestor_matcher_free(&matcher);
    tsi_safe_free(haplotypes);
    tsi_safe_free(positions);
    tsi_safe_free(a);
    tsi_safe_free(edges_buffer);
    tsi_safe_free(site_mutation_buffer);
}


int
main(int argc, char** argv)
{
    /* SYNTAX 1: generate [-v] <input-file> <output-file> */
    struct arg_rex *cmd1 = arg_rex1(NULL, NULL, "generate", NULL, REG_ICASE, NULL);
    struct arg_lit *verbose1 = arg_lit0("v", "verbose", NULL);
    struct arg_file *sample_file1 = arg_file1(NULL, NULL, NULL, NULL);
    struct arg_file *ancestor_file1 = arg_file1(NULL, NULL, NULL, NULL);
    struct arg_file *site_file1 = arg_file1(NULL, NULL, NULL, NULL);
    struct arg_end *end1 = arg_end(20);
    void* argtable1[] = {cmd1, verbose1, sample_file1, ancestor_file1, site_file1, end1};
    int nerrors1;

    int exitcode = EXIT_SUCCESS;
    const char *progname = "main";

    nerrors1 = arg_parse(argc, argv, argtable1);

    if (nerrors1 == 0) {
        run_generate(sample_file1->filename[0], ancestor_file1->filename[0],
                site_file1->filename[0], verbose1->count);
    } else {
        /* We get here if the command line matched none of the possible syntaxes */
        if (cmd1->count > 0) {
            arg_print_errors(stdout, end1, progname);
            printf("usage: %s ", progname);
            arg_print_syntax(stdout, argtable1, "\n");
        } else {
            /* no correct cmd literals were given, so we cant presume which syntax was intended */
            printf("%s: missing command.\n",progname);
            printf("usage 1: %s ", progname);  arg_print_syntax(stdout, argtable1, "\n");
        }
    }
    arg_freetable(argtable1, sizeof(argtable1) / sizeof(argtable1[0]));

    return exitcode;
}
