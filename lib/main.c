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
read_ancestors(const char *infile, size_t num_sites,
        size_t *r_num_ancestors, uint32_t **r_ancestor_age,
        size_t *r_num_focal_sites, ancestor_id_t **r_focal_site_ancestor,
        site_id_t **r_focal_site)
{
    char *line = NULL;
    size_t len = 0;
    size_t j, num_ancestors;
    const char delimiters[] = " \t";
    const char comma[] = ",";
    char *token, *subtoken;
    size_t num_focal_sites = 0;
    uint32_t *ancestor_age = NULL;
    site_id_t *focal_site = NULL;
    ancestor_id_t *focal_site_ancestor = NULL;
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
    focal_site = malloc(num_sites * sizeof(site_id_t));
    focal_site_ancestor = malloc(num_sites * sizeof(ancestor_id_t));
    ancestor_age = malloc(num_ancestors * sizeof(uint32_t));
    if (focal_site == NULL || focal_site_ancestor == NULL || ancestor_age == NULL) {
        fatal_error("Malloc error");
    }
    j = 0;
    while (getline(&line, &len, f) != -1) {
        token = strtok(line, delimiters);
        if (token == NULL) {
            fatal_error("File format error");
        }
        ancestor_age[j] = atoi(token);
        token = strtok(NULL, delimiters);
        if (token == NULL) {
            fatal_error("File format error");
        }
        if (ancestor_age[j] != UINT32_MAX) {
            subtoken = strtok(token, comma);
            while (subtoken != NULL) {
                assert(num_focal_sites < num_sites);
                focal_site_ancestor[num_focal_sites] = j;
                focal_site[num_focal_sites] = atoi(subtoken);
                num_focal_sites++;
                subtoken = strtok(NULL, comma);
            }
        }
        j++;
    }
    free(line);
    fclose(f);
    *r_num_ancestors = num_ancestors;
    *r_ancestor_age = ancestor_age;
    *r_num_focal_sites = num_focal_sites;
    *r_focal_site_ancestor = focal_site_ancestor;
    *r_focal_site = focal_site;
}

static void
read_sites(const char *infile, ancestor_store_t *store,
        size_t num_sites, double *position,
        size_t num_ancestors, uint32_t *ancestor_age,
        size_t num_focal_sites, ancestor_id_t *focal_site_ancestor, site_id_t *focal_site)
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
    if (seg_site == NULL || seg_start == NULL || seg_end == NULL) {
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
        j++;
    }
    assert(num_sites == seg_site[num_segments - 1] + 1);
    ret = ancestor_store_alloc(store,
            num_sites, position,
            num_ancestors, ancestor_age,
            num_focal_sites, focal_site_ancestor, focal_site,
            num_segments, seg_site, seg_start, seg_end);
    if (ret != 0) {
        fatal_error("store load error");
    }

    free(line);
    free(seg_site);
    free(seg_start);
    free(seg_end);
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
    FILE *out = fopen(outfile, "w");

    num_segments = store_builder->total_segments;
    seg_site = malloc(num_segments * sizeof(site_id_t));
    seg_start = malloc(num_segments * sizeof(ancestor_id_t));
    seg_end = malloc(num_segments * sizeof(ancestor_id_t));
    if (seg_site == NULL || seg_start == NULL || seg_end == NULL) {
        fatal_error("Malloc error");
    }
    ret = ancestor_store_builder_dump(store_builder, seg_site, seg_start, seg_end);
    if (ret != 0) {
        fatal_error("Dump error");
    }
    if (out == NULL) {
        fatal_error("Error opening file");
    }
    for (j = 0; j < num_segments; j++) {
        fprintf(out, "%d\t%d\t%d\n", seg_site[j], seg_start[j], seg_end[j]);
    }
    if (fclose(out) != 0) {
        fatal_error("Close error");
    }
    tsi_safe_free(seg_site);
    tsi_safe_free(seg_start);
    tsi_safe_free(seg_end);
}


static void
run_generate(const char *sample_file, const char *ancestor_file,
        const char *site_file, int sort, int verbose)
{
    size_t num_samples, num_sites, j, k, l, num_ancestors;
    allele_t *haplotypes = NULL;
    allele_t *ancestors = NULL;
    site_id_t *permutation = NULL;
    double *positions = NULL;
    site_id_t *focal_sites;
    size_t num_focal_sites;
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
    ret = ancestor_store_builder_alloc(&store_builder, num_sites, 8192);
    if (ret != 0) {
        fatal_error("store alloc error.");
    }
    permutation = malloc(num_sites * sizeof(site_id_t));
    if (permutation == NULL) {
        fatal_error("permutation alloc error");
    }
    fprintf(ancestor_out, "-1\t-1\n");
    for (j = 0; j < builder.num_frequency_classes; j++) {
        num_ancestors = builder.frequency_classes[j].num_ancestors;
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
            focal_sites = builder.frequency_classes[j].ancestor_focal_sites[k];
            num_focal_sites = builder.frequency_classes[j].num_ancestor_focal_sites[k];
            a = ancestors + k * num_sites;
            ret = ancestor_builder_make_ancestor(&builder, num_focal_sites, focal_sites, a);
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
            ret = ancestor_store_builder_add(&store_builder, a);
            if (ret != 0) {
                fatal_error("Error in add ancestor");
            }
            if (verbose > 0) {
                print_ancestor(a, num_sites);
            }
            focal_sites = builder.frequency_classes[j].ancestor_focal_sites[k];
            num_focal_sites = builder.frequency_classes[j].num_ancestor_focal_sites[k];
            fprintf(ancestor_out, "%d\t", (int) builder.frequency_classes[j].frequency);
            for (l = 0; l < num_focal_sites; l++) {
                fprintf(ancestor_out, "%d", focal_sites[l]);
                if (l < num_focal_sites - 1) {
                    fprintf(ancestor_out, ",");
                }
            }
            fprintf(ancestor_out, "\n");
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
run_match(const char *sample_file, const char *ancestor_file, const char *site_file,
        int verbose)
{
    int ret;
    ancestor_store_t store;
    ancestor_matcher_t matcher;
    /* segment_list_node_t *seg; */
    tree_sequence_builder_t ts_builder;
    /* ancestor_id_t sample_id; */
    site_id_t *focal_site = NULL;
    ancestor_id_t *focal_site_ancestor = NULL;
    ancestor_id_t *epoch_ancestors = NULL;
    uint32_t *ancestor_age = NULL;
    site_id_t start, *focal, end;
    allele_t *ancestor = NULL;
    size_t j, k, l, num_focal_sites, num_ancestors, num_older_ancestors;
    /* double mutation_rate = 1e-200; */
    allele_t *samples = NULL;
    /* allele_t *sample; */
    double *positions;
    size_t num_samples, num_sites, num_epoch_ancestors;
    size_t total_edges;
    edge_t *edges_buffer = NULL;
    edge_t *output_edges;
    size_t num_output_edges;
    size_t max_edges = 1024;
    site_mutation_t *site_mutation_buffer = NULL;
    size_t max_site_mutations = 1024;
    size_t num_site_mutations;
    node_id_t child;

    read_samples(sample_file, &num_samples, &num_sites, &samples, &positions);
    read_ancestors(ancestor_file, num_sites, &num_ancestors, &ancestor_age,
            &num_focal_sites, &focal_site_ancestor, &focal_site);
    read_sites(site_file, &store, num_sites, positions,
            num_ancestors, ancestor_age,
            num_focal_sites, focal_site_ancestor, focal_site);
    ret = tree_sequence_builder_alloc(&ts_builder, num_sites, store.num_ancestors + 1,
            65536);
    if (ret != 0) {
        fatal_error("alloc error");
    }

    ret = ancestor_matcher_alloc(&matcher, &ts_builder, 0.01);
    if (ret != 0) {
        fatal_error("alloc error");
    }
    ancestor = malloc(store.num_sites * sizeof(allele_t));
    epoch_ancestors = malloc(store.num_ancestors * sizeof(ancestor_id_t *));
    edges_buffer = malloc(max_edges * sizeof(edge_t));
    site_mutation_buffer = malloc(max_site_mutations * sizeof(site_mutation_t));
    if (ancestor == NULL || epoch_ancestors == NULL || edges_buffer == NULL
            || site_mutation_buffer == NULL) {
        fatal_error("alloc error");
    }
    ret = tree_sequence_builder_update(&ts_builder, 1, store.num_epochs, 0, NULL,
            0, NULL);
    if (ret != 0) {
        fatal_error("initial update");
    }
    child = 1;
    for (j = store.num_epochs - 2; j > 0; j--) {
        /* printf("STARTING EPOCH %d\n", (int) j); */
        ret = ancestor_store_get_epoch_ancestors(&store, j, epoch_ancestors,
                &num_epoch_ancestors);
        if (ret != 0) {
            fatal_error("error getting epoch ancestors");
        }

        total_edges = 0;
        num_site_mutations = 0;
        for (k = 0; k < num_epoch_ancestors; k++) {
            ret = ancestor_store_get_ancestor(
                    &store, epoch_ancestors[k], ancestor, &start, &end,
                    &num_older_ancestors, &num_focal_sites, &focal);
            if (ret != 0) {
                fatal_error("get_ancestor error");
            }
            for (l = 0; l < num_focal_sites; l++) {
                assert(num_site_mutations < max_site_mutations);
                site_mutation_buffer[num_site_mutations].node = child;
                site_mutation_buffer[num_site_mutations].site = focal[l];
                num_site_mutations++;
                assert(ancestor[focal[l]] == 1);
                ancestor[focal[l]] = 0;
            }
            /* printf("Got ancestor (%d-%d), %d, %d\n", start, end, (int) num_older_ancestors, */
            /*         (int) num_focal_sites); */
            ret = ancestor_matcher_find_path(&matcher, ancestor, child,
                    &num_output_edges, &output_edges);
            if (ret != 0) {
                fatal_error("find_path error");
            }
            if (total_edges + num_output_edges > max_edges) {
                fatal_error("out of edge buffer space\n");
            }
            memcpy(edges_buffer + total_edges, output_edges,
                    num_output_edges * sizeof(edge_t));
            total_edges += num_output_edges;
            if (verbose > 0) {
                printf("ancestor %d:\t", (int) epoch_ancestors[k]);
                for (l = 0; l < store.num_sites; l++) {
                    printf("%d", ancestor[l]);
                }
                printf("\n");
                printf("\tnum_focal=%d num_older=%d\n",
                        (int) num_focal_sites, (int) num_older_ancestors);
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
        ret = tree_sequence_builder_update(&ts_builder, num_epoch_ancestors, j + 1,
                total_edges, edges_buffer, num_site_mutations, site_mutation_buffer);
    }
    /* epoch_ancestors[0] = 0; */
    /* ret = tree_sequence_builder_resolve(&ts_builder, store.num_epochs - 1, epoch_ancestors, 1); */
    /* if (ret != 0) { */
    /*     fatal_error("error resolve"); */
    /* } */
    /* if (verbose > 0) { */
    /*     tree_sequence_builder_print_state(&ts_builder, stdout); */
    /* } */

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

    tree_sequence_builder_free(&ts_builder);
    ancestor_matcher_free(&matcher);
    ancestor_store_free(&store);
    free(focal_site);
    free(focal_site_ancestor);
    free(ancestor_age);
    free(ancestor);
    free(epoch_ancestors);
    free(positions);
    free(samples);
    free(edges_buffer);
    free(site_mutation_buffer);
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
