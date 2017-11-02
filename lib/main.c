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

#include <hdf5.h>

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
fatal_hdf5_error(const char *msg)
{
    fprintf(stderr, "infer: %s", msg);
    H5Eprint1(stderr);
    exit(EXIT_FAILURE);
}

static void
read_hdf5_dimensions(hid_t file_id, size_t *num_samples, size_t *num_sites)
{
    hid_t dataset_id, dataspace_id;
    herr_t status;
    htri_t exists;
    int rank;
    hsize_t dims[2];
    const char *name = "/samples/haplotypes";

    exists = H5Lexists(file_id, name, H5P_DEFAULT);
    if (exists < 0) {
        fatal_hdf5_error("error reading samples/haplotypes");
    }
    if (!exists) {
        fatal_error("cannot find samples/haplotypes");
    }

    dataset_id = H5Dopen(file_id, name, H5P_DEFAULT);
    if (dataset_id < 0) {
        fatal_hdf5_error("Reading samples/haplotypes");
    }
    dataspace_id = H5Dget_space(dataset_id);
    if (dataspace_id < 0) {
        fatal_hdf5_error("Reading samples/haplotypes");
    }
    rank = H5Sget_simple_extent_ndims(dataspace_id);
    if (rank != 2) {
        fatal_error("samples/haplotypes not 2D");
    }
    status = H5Sget_simple_extent_dims(dataspace_id, dims, NULL);
    if (status < 0) {
        fatal_hdf5_error("Reading samples/haplotypes");
    }
    status = H5Sclose(dataspace_id);
    if (status < 0) {
        fatal_hdf5_error("Reading samples/haplotypes");
    }
    status = H5Dclose(dataset_id);
    if (status < 0) {
        fatal_hdf5_error("Reading samples/haplotypes");
    }
    *num_samples = dims[0];
    *num_sites = dims[1];
}

static void
check_hdf5_dimensions(hid_t file_id, size_t num_samples, size_t num_sites)
{
    hid_t dataset_id, dataspace_id;
    herr_t status;
    int rank;
    hsize_t dims[2];
    htri_t exists;
    struct _dimension_check {
        const char *name;
        size_t size;
    };
    struct _dimension_check fields[] = {
        {"/sites/position", num_sites},
        {"/sites/recombination_rate", num_sites},
    };
    size_t num_fields = sizeof(fields) / sizeof(struct _dimension_check);
    size_t j;

    for (j = 0; j < num_fields; j++) {
        exists = H5Lexists(file_id, fields[j].name, H5P_DEFAULT);
        if (exists < 0) {
            fatal_hdf5_error("read_dimensions");
        }
        if (! exists) {
            fatal_error("Cannot find field '%s'", fields[j].name);
        }
        dataset_id = H5Dopen(file_id, fields[j].name, H5P_DEFAULT);
        if (dataset_id < 0) {
            fatal_hdf5_error("read_dimensions");
        }
        dataspace_id = H5Dget_space(dataset_id);
        if (dataspace_id < 0) {
            fatal_hdf5_error("read_dimensions");
        }
        rank = H5Sget_simple_extent_ndims(dataspace_id);
        if (rank != 1) {
            fatal_error("dimension != 1");
        }
        status = H5Sget_simple_extent_dims(dataspace_id, dims, NULL);
        if (status < 0) {
            fatal_hdf5_error("read_dimensions");
        }
        status = H5Sclose(dataspace_id);
        if (status < 0) {
            fatal_hdf5_error("read_dimensions");
        }
        status = H5Dclose(dataset_id);
        if (status < 0) {
            fatal_hdf5_error("read_dimensions");
        }
        if (dims[0] != fields[j].size) {
            fatal_error("size mismatch for '%s'", fields[j].name);
        }
    }
}

static void
read_hdf5_data(hid_t file_id, allele_t *haplotypes, double *position,
        double *recombination_rate)
{
    herr_t status;
    hid_t dataset_id;
    htri_t exists;
    struct _hdf5_field_read {
        const char *name;
        hid_t type;
        void *dest;
    };
    struct _hdf5_field_read fields[] = {
        {"/sites/recombination_rate", H5T_NATIVE_DOUBLE, recombination_rate},
        {"/sites/position", H5T_NATIVE_DOUBLE, position},
        {"/samples/haplotypes", H5T_NATIVE_CHAR, haplotypes},
    };
    size_t num_fields = sizeof(fields) / sizeof(struct _hdf5_field_read);
    size_t j;

    for (j = 0; j < num_fields; j++) {
        exists = H5Lexists(file_id, fields[j].name, H5P_DEFAULT);
        if (exists < 0) {
            fatal_hdf5_error("reading site data");
        }
        if (!exists) {
            fatal_error("field missing");
        }
        dataset_id = H5Dopen(file_id, fields[j].name, H5P_DEFAULT);
        if (dataset_id < 0) {
            fatal_hdf5_error("reading site data");
        }
        status = H5Dread(dataset_id, fields[j].type, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                fields[j].dest);
        if (status < 0) {
            fatal_hdf5_error("reading site data");
        }
        status = H5Dclose(dataset_id);
        if (status < 0) {
            fatal_hdf5_error("reading site data");
        }
    }
}

static void
read_input(const char *filename, size_t *r_num_samples, size_t *r_num_sites,
        allele_t **r_haplotypes, double **r_position, double **r_recombination_rate)
{

    hid_t file_id = -1;
    herr_t status;
    size_t num_sites, num_samples;
    allele_t *haplotypes;
    double *position, *recombination_rate;

    file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        fatal_hdf5_error("Opening HDF5 file");
    }
    /* TODO read metadata attributes and sequence length */
    read_hdf5_dimensions(file_id, &num_samples, &num_sites);
    check_hdf5_dimensions(file_id, num_samples, num_sites);
    position = malloc(num_sites * sizeof(double));
    recombination_rate = malloc(num_sites * sizeof(double));
    haplotypes = malloc(num_samples * num_sites * sizeof(allele_t));
    if (position == NULL || recombination_rate == NULL || haplotypes == NULL) {
        fatal_error("malloc failure");
    }
    read_hdf5_data(file_id, haplotypes, position, recombination_rate);
    status = H5Fclose(file_id);
    if (status < 0) {
        fatal_hdf5_error("Closing HDF5 file");
    }

    *r_num_samples = num_samples;
    *r_num_sites = num_sites;
    *r_haplotypes = haplotypes;
    *r_position = position;
    *r_recombination_rate = recombination_rate;
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
    mutation_id_t *mutation_parent = malloc(num_mutations * sizeof(mutation_id_t));

    if (time == NULL || flags == NULL
            || left == NULL || right == NULL || parent == NULL || children == NULL
            || site == NULL || node == NULL || derived_state == NULL
            || mutation_parent == NULL) {
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
    ret = tree_sequence_builder_dump_mutations(ts_builder, site, node,
            derived_state, mutation_parent);
    printf("MUTATIONS\n");
    for (j = 0; j < num_mutations; j++) {
        printf("%d\t%d\t%d\t%d\n", site[j], node[j], derived_state[j], mutation_parent[j]);
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
    free(mutation_parent);
}

static void
run_generate(const char *input_file, int verbose)
{
    size_t num_samples, num_sites, j, k, l, num_ancestors;
    allele_t *haplotypes = NULL;
    double *positions = NULL;
    double *recombination_rate = NULL;
    site_id_t *focal_sites, start, end;
    size_t num_focal_sites;
    ancestor_builder_t ancestor_builder;
    tree_sequence_builder_t ts_builder;
    ancestor_matcher_t matcher;
    allele_t *a, *sample, *match;
    size_t age;
    int ret;
    /* Buffers for edge output */
    size_t total_edges;
    size_t num_edges;
    size_t max_edges = 1024 * 1024;
    site_id_t *left_buffer, *left_output;
    site_id_t *right_buffer, *right_output;
    node_id_t *parent_buffer, *parent_output;
    node_id_t *child_buffer;
    /* Buffers for mutation output */
    size_t max_mutations = 8192;
    size_t total_mutations;
    site_id_t *site_buffer;
    node_id_t *node_buffer;
    allele_t *derived_state_buffer;
    node_id_t child;
    /* int flags = TSI_RESOLVE_SHARED_RECOMBS|TSI_RESOLVE_POLYTOMIES; */
    int flags = 0;

    read_input(input_file, &num_samples, &num_sites, &haplotypes, &positions,
            &recombination_rate);
    ret = ancestor_builder_alloc(&ancestor_builder, num_samples, num_sites,
            positions, haplotypes);
    if (ret != 0) {
        fatal_error("Builder alloc error.");
    }
    num_ancestors = ancestor_builder.num_ancestors;
    ret = tree_sequence_builder_alloc(&ts_builder, positions[num_sites - 1] + 1,
            num_sites, positions, recombination_rate,
            100 * (num_samples + num_ancestors), 10, flags);
    if (ret != 0) {
        fatal_error("alloc error");
    }
    ret = ancestor_matcher_alloc(&matcher, &ts_builder, 0.0);
    if (ret != 0) {
        fatal_error("alloc error");
    }

    if (verbose > 0) {
        ancestor_builder_print_state(&ancestor_builder, stdout);
        ancestor_matcher_print_state(&matcher, stdout);
    }

    age = ancestor_builder.num_frequency_classes + 1;
    ret = tree_sequence_builder_update(&ts_builder, 1, age,
            0, NULL, NULL, NULL, NULL, 0, NULL, NULL, NULL);
    if (ret != 0) {
        fatal_error("initial update");
    }
    a = malloc(num_sites * sizeof(allele_t));
    match = malloc(num_sites * sizeof(allele_t));
    left_buffer = malloc(max_edges * sizeof(site_id_t));
    right_buffer = malloc(max_edges * sizeof(site_id_t));
    parent_buffer = malloc(max_edges * sizeof(node_id_t));
    child_buffer = malloc(max_edges * sizeof(node_id_t));
    node_buffer = malloc(max_mutations * sizeof(node_id_t));
    derived_state_buffer = malloc(max_mutations * sizeof(allele_t));
    site_buffer = malloc(max_mutations * sizeof(site_id_t));
    if (a == NULL || match == NULL || left_buffer == NULL || right_buffer == NULL
            || parent_buffer == NULL || child_buffer == NULL || node_buffer == NULL
            || site_buffer == NULL || derived_state_buffer == NULL) {
        fatal_error("alloc");
    }

    for (j = 0; j < ancestor_builder.num_frequency_classes; j++) {
        age--;
        num_ancestors = ancestor_builder.frequency_classes[j].num_ancestors;
        if (verbose > 0) {
            printf("Generating for frequency class %d: age = %d num_ancestors = %d\n",
                    (int) j, (int) age, (int) num_ancestors);
        }
        /* printf("AGE = %d\n", (int) age); */
        total_edges = 0;
        total_mutations = 0;
        child = ts_builder.num_nodes;
        for (k = 0; k < num_ancestors; k++) {
            focal_sites = ancestor_builder.frequency_classes[j].ancestor_focal_sites[k];
            num_focal_sites = ancestor_builder.frequency_classes[j].num_ancestor_focal_sites[k];
            ret = ancestor_builder_make_ancestor(&ancestor_builder, num_focal_sites,
                    focal_sites, &start, &end, a);
            if (ret != 0) {
                fatal_error("Error in make ancestor");
            }
            for (l = 0; l < num_focal_sites; l++) {
                assert(total_mutations < max_mutations);
                node_buffer[total_mutations] = child;
                site_buffer[total_mutations] = focal_sites[l];
                derived_state_buffer[total_mutations] = 1;
                total_mutations++;
                assert(a[focal_sites[l]] == 1);
                a[focal_sites[l]] = 0;
            }
            ret = ancestor_matcher_find_path(&matcher, start, end, a, match,
                    &num_edges, &left_output, &right_output, &parent_output);
            if (ret != 0) {
                fatal_error("find_path error");
            }
            for (l = 0; l < num_sites; l++) {
                if (a[l] != match[l]) {
                    printf("Mismatch at %d : %d %d \n", (int) l, a[l], match[l]);
                }
                assert(a[l] == match[l]);
            }
            if (total_edges + num_edges > max_edges) {
                fatal_error("out of edge buffer space\n");
            }
            memcpy(left_buffer + total_edges, left_output, num_edges * sizeof(site_id_t));
            memcpy(right_buffer + total_edges, right_output, num_edges * sizeof(site_id_t));
            memcpy(parent_buffer + total_edges, parent_output, num_edges * sizeof(site_id_t));
            /* Update the child buffer */
            for (l = 0; l < num_edges; l++) {
                child_buffer[total_edges + l] = child;
            }
            total_edges += num_edges;

            if (verbose > 0) {
                printf("ancestor %d:\t", (int) child);
                for (l = 0; l < num_sites; l++) {
                    if (a[l] == -1) {
                        assert(l < start || l >= end);
                        printf("*");
                    } else {
                        assert(l >= start && l < end);
                        printf("%d", a[l]);
                    }
                }
                printf("\n");
                printf("\tnum_focal=%d, start=%d, end=%d", (int) num_focal_sites, start, end);
                printf("\tedges = (%d):: \n", (int) num_edges);
                for (l = 0; l < num_edges; l++) {
                    printf("\t(%d, %d, %d, %d)\n", left_output[l], right_output[l],
                            parent_output[l], child);
                }
            }
            child++;
        }
        ret = tree_sequence_builder_update(&ts_builder, num_ancestors, age,
                total_edges, left_buffer, right_buffer, parent_buffer, child_buffer,
                total_mutations, site_buffer, node_buffer, derived_state_buffer);
        if (ret != 0) {
            fatal_error("builder update");
        }
        /* tree_sequence_builder_print_state(&ts_builder, stdout); */
    }

    total_edges = 0;
    total_mutations = 0;
    /* Copy samples */
    matcher.observation_error = 0.001;
    for (j = 0; j < num_samples; j++) {
        sample = haplotypes + j * num_sites;
        child = num_ancestors + j;
        ret = ancestor_matcher_find_path(&matcher, 0, num_sites, sample, match,
                &num_edges, &left_output, &right_output, &parent_output);
        if (ret != 0) {
            fatal_error("find_path error");
        }
        if (verbose > 0) {
            printf("sample %d:\t", (int) child);
            for (l = 0; l < num_sites; l++) {
                printf("%d", sample[l]);
            }
            printf("\nmatch = \t");
            for (l = 0; l < num_sites; l++) {
                printf("%d", match[l]);
            }
            printf("\n");
        }
        for (l = 0; l < num_sites; l++) {
            if (sample[l] != match[l]) {
                assert(total_mutations < max_mutations);
                node_buffer[total_mutations] = child;
                site_buffer[total_mutations] = l;
                derived_state_buffer[total_mutations] = sample[l];
                total_mutations++;
            }
        }

        if (total_edges + num_edges > max_edges) {
            fatal_error("out of edge buffer space\n");
        }
        memcpy(left_buffer + total_edges, left_output, num_edges * sizeof(site_id_t));
        memcpy(right_buffer + total_edges, right_output, num_edges * sizeof(site_id_t));
        memcpy(parent_buffer + total_edges, parent_output, num_edges * sizeof(site_id_t));
        /* Update the child buffer */
        for (l = 0; l < num_edges; l++) {
            child_buffer[total_edges + l] = child;
        }
        total_edges += num_edges;

        /* printf("COPIED SAMPLE %d->%d\n", (int) j, (int) sample_id); */
        /* tree_sequence_builder_print_state(&ts_builder, stdout); */
    }

    ret = tree_sequence_builder_update(&ts_builder, num_samples, 0,
            total_edges, left_buffer, right_buffer, parent_buffer, child_buffer,
            total_mutations, site_buffer, node_buffer, derived_state_buffer);
    if (ret != 0) {
        fatal_error("builder update");
    }

    if (1) {
        output_ts(&ts_builder);
    }


    ancestor_builder_free(&ancestor_builder);
    tree_sequence_builder_free(&ts_builder);
    ancestor_matcher_free(&matcher);
    tsi_safe_free(haplotypes);
    tsi_safe_free(positions);
    tsi_safe_free(recombination_rate);
    tsi_safe_free(a);
    tsi_safe_free(match);
    tsi_safe_free(left_buffer);
    tsi_safe_free(right_buffer);
    tsi_safe_free(parent_buffer);
    tsi_safe_free(child_buffer);
    tsi_safe_free(node_buffer);
    tsi_safe_free(site_buffer);
    tsi_safe_free(derived_state_buffer);
}


int
main(int argc, char** argv)
{
    /* SYNTAX 1: generate [-v] <input-file> <output-file> */
    struct arg_rex *cmd1 = arg_rex1(NULL, NULL, "generate", NULL, REG_ICASE, NULL);
    struct arg_lit *verbose1 = arg_lit0("v", "verbose", NULL);
    struct arg_file *sample_file1 = arg_file1(NULL, NULL, NULL, NULL);
    struct arg_end *end1 = arg_end(20);
    void* argtable1[] = {cmd1, verbose1, sample_file1, end1};
    int nerrors1;

    int exitcode = EXIT_SUCCESS;
    const char *progname = "main";

    nerrors1 = arg_parse(argc, argv, argtable1);

    if (nerrors1 == 0) {
        run_generate(sample_file1->filename[0], verbose1->count);
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
