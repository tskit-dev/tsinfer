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
    const char *name = "/haplotypes";

    exists = H5Lexists(file_id, name, H5P_DEFAULT);
    if (exists < 0) {
        fatal_hdf5_error("error reading /haplotypes");
    }
    if (!exists) {
        fatal_error("cannot find /haplotypes");
    }

    dataset_id = H5Dopen(file_id, name, H5P_DEFAULT);
    if (dataset_id < 0) {
        fatal_hdf5_error("Reading /haplotypes");
    }
    dataspace_id = H5Dget_space(dataset_id);
    if (dataspace_id < 0) {
        fatal_hdf5_error("Reading /haplotypes");
    }
    rank = H5Sget_simple_extent_ndims(dataspace_id);
    if (rank != 2) {
        fatal_error("/haplotypes not 2D");
    }
    status = H5Sget_simple_extent_dims(dataspace_id, dims, NULL);
    if (status < 0) {
        fatal_hdf5_error("Reading /haplotypes");
    }
    status = H5Sclose(dataspace_id);
    if (status < 0) {
        fatal_hdf5_error("Reading /haplotypes");
    }
    status = H5Dclose(dataset_id);
    if (status < 0) {
        fatal_hdf5_error("Reading /haplotypes");
    }
    *num_samples = dims[0];
    *num_sites = dims[1];
}

static void
read_hdf5_data(hid_t file_id, allele_t *haplotypes)
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
        {"/haplotypes", H5T_NATIVE_CHAR, haplotypes},
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
        allele_t **r_haplotypes)
{

    hid_t file_id = -1;
    herr_t status;
    size_t num_sites, num_samples;
    allele_t *haplotypes;

    file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        fatal_hdf5_error("Opening HDF5 file");
    }
    read_hdf5_dimensions(file_id, &num_samples, &num_sites);
    haplotypes = malloc(num_samples * num_sites * sizeof(allele_t));
    if (haplotypes == NULL) {
        fatal_error("malloc failure");
    }
    read_hdf5_data(file_id, haplotypes);
    status = H5Fclose(file_id);
    if (status < 0) {
        fatal_hdf5_error("Closing HDF5 file");
    }

    *r_num_samples = num_samples;
    *r_num_sites = num_sites;
    *r_haplotypes = haplotypes;
}

static void
output_ts(tree_sequence_builder_t *ts_builder)
{
    int ret = 0;
    size_t j;
    size_t num_nodes = tree_sequence_builder_get_num_nodes(ts_builder);
    size_t num_edges = tree_sequence_builder_get_num_edges(ts_builder);
    size_t num_mutations = tree_sequence_builder_get_num_mutations(ts_builder);
    double *time = malloc(num_nodes * sizeof(double));
    uint32_t *flags = malloc(num_nodes * sizeof(uint32_t));
    site_id_t *left = malloc(num_edges * sizeof(site_id_t));
    site_id_t *right = malloc(num_edges * sizeof(site_id_t));
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
        printf("%d\t%d\t%d\t%d\n", left[j], right[j], parent[j], children[j]);
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
run_generate(const char *input_file, int verbose, int path_compression)
{
    size_t num_samples, num_sites, j, k, num_ancestors;
    allele_t *haplotypes = NULL;
    allele_t *genotypes = NULL;
    site_id_t *focal_sites = NULL;
    site_id_t l, start, end;
    size_t num_focal_sites;
    ancestor_builder_t ancestor_builder;
    tree_sequence_builder_t ts_builder;
    ancestor_matcher_t matcher;
    allele_t *a, *sample, *match;
    size_t frequency, edge_offset, mutation_offset;
    int ret;
    /* Buffers for edge output */
    size_t total_edges;
    size_t num_edges, *num_edges_buffer, *num_mutations_buffer;
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
    int flags = 0;
    avl_node_t *avl_node;
    pattern_map_t *map_elem;
    site_list_t *s;
    int add_path_flags = 0;

    if (path_compression) {
        add_path_flags = TSI_COMPRESS_PATH;
    }
    flags = 0;

    read_input(input_file, &num_samples, &num_sites, &haplotypes);
    ret = ancestor_builder_alloc(&ancestor_builder, num_samples, num_sites, 0);
    if (ret != 0) {
        fatal_error("Builder alloc error.");
    }
    genotypes = malloc(num_sites * sizeof(allele_t));
    focal_sites = malloc(num_sites * sizeof(site_id_t));
    if (genotypes == NULL || focal_sites == NULL) {
        fatal_error("Error allocing genotypes");
    }

    for (l = 0; l < (site_id_t) num_sites; l++) {
        /* Copy in the genotypes for this sites */
        frequency = 0;
        for (j = 0; j < num_samples; j++) {
            genotypes[j] = haplotypes[j * num_samples + l];
            frequency += genotypes[j];
        }
        ret = ancestor_builder_add_site(&ancestor_builder, l, frequency, genotypes);
        if (ret != 0) {
            fatal_error("Add site error");
        }
    }
    ret = ancestor_builder_finalise(&ancestor_builder);
    if (ret != 0) {
        fatal_error("builder finalise");
    }

    num_ancestors = ancestor_builder.num_ancestors;
    ret = tree_sequence_builder_alloc(&ts_builder, num_sites, 10, 10, flags);
    if (ret != 0) {
        fatal_error("alloc error");
    }
    ret = ancestor_matcher_alloc(&matcher, &ts_builder, 0);
    if (ret != 0) {
        fatal_error("alloc error");
    }

    if (verbose > 0) {
        ancestor_builder_print_state(&ancestor_builder, stdout);
        /* ancestor_matcher_print_state(&matcher, stdout); */
    }
    /* TODO This is all out of date now. Need to rewrite this inferface to
     * follow the high-level approach of generating ancestors and matching
     * samples and ancestors. Probably we'll need to map the input file
     * formats to HDF5 so that we can read them in C
     * */
    a = malloc(num_sites * sizeof(allele_t));
    match = malloc(num_sites * sizeof(allele_t));
    left_buffer = malloc(max_edges * sizeof(site_id_t));
    right_buffer = malloc(max_edges * sizeof(site_id_t));
    parent_buffer = malloc(max_edges * sizeof(node_id_t));
    child_buffer = malloc(max_edges * sizeof(node_id_t));
    num_edges_buffer = malloc(max_edges * sizeof(size_t));
    num_mutations_buffer = malloc(max_edges * sizeof(size_t));
    node_buffer = malloc(max_mutations * sizeof(node_id_t));
    derived_state_buffer = malloc(max_mutations * sizeof(allele_t));
    site_buffer = malloc(max_mutations * sizeof(site_id_t));
    if (a == NULL || match == NULL || left_buffer == NULL || right_buffer == NULL
            || parent_buffer == NULL || child_buffer == NULL || node_buffer == NULL
            || num_edges_buffer == NULL || site_buffer == NULL
            || derived_state_buffer == NULL) {
        fatal_error("alloc");
    }

    /* Add the ultimate ancestor */
    ret = tree_sequence_builder_add_node(&ts_builder, num_samples + 1, true);
    if (ret < 0) {
        fatal_error("add node");
    }
    /* Add the root ancestor */
    ret = tree_sequence_builder_add_node(&ts_builder, num_samples, true);
    if (ret < 0) {
        fatal_error("add node");
    }
    /* Add the ancestor nodes */
    for (frequency = num_samples - 1; frequency > 0; frequency--) {
        num_ancestors =  avl_count(&ancestor_builder.frequency_map[frequency]);
        for (j = 0; j < num_ancestors; j++) {
            ret = tree_sequence_builder_add_node(&ts_builder, frequency, true);
            if (ret < 0) {
                fatal_error("add node");
            }
        }
    }

    /* Add the path for the root ancestor */
    left_buffer[0] = 0;
    right_buffer[0] = num_sites;
    parent_buffer[0] = 0;
    ret = tree_sequence_builder_add_path(&ts_builder, 1, 1,
            left_buffer, right_buffer, parent_buffer, 0);
    if (ret != 0) {
        fatal_error("add_root_path");
    }

    child = 2;
    for (frequency = num_samples - 1; frequency > 0; frequency--) {
        ret = tree_sequence_builder_freeze_indexes(&ts_builder);
        if (ret != 0) {
            fatal_error("freeze");
        }
        num_ancestors =  avl_count(&ancestor_builder.frequency_map[frequency]);
        if (verbose > 0) {
            printf("Generating for frequency class frequency = %d num_ancestors = %d\n",
                    (int) frequency, (int) num_ancestors);
        }
        /* printf("AGE = %d\n", (int) frequency); */
        total_edges = 0;
        total_mutations = 0;
        j = 0;
        for (avl_node = ancestor_builder.frequency_map[frequency].head; avl_node != NULL;
                avl_node = avl_node->next) {
            map_elem = (pattern_map_t *) avl_node->item;
            num_focal_sites = map_elem->num_sites;
            /* The linked list is in reverse order, so insert backwards here */
            k = num_focal_sites - 1;
            for (s = map_elem->sites; s != NULL; s = s->next) {
                focal_sites[k] = s->site;
                k--;
            }
            ret = ancestor_builder_make_ancestor(&ancestor_builder, num_focal_sites,
                    focal_sites, &start, &end, a);
            if (ret != 0) {
                fatal_error("Error in make ancestor");
            }
            for (l = 0; l < (site_id_t) num_focal_sites; l++) {
                assert(total_mutations < max_mutations);
                node_buffer[total_mutations] = child;
                site_buffer[total_mutations] = focal_sites[l];
                derived_state_buffer[total_mutations] = 1;
                total_mutations++;
                assert(a[focal_sites[l]] == 1);
            }
            ret = ancestor_matcher_find_path(&matcher, start, end, a, match,
                    &num_edges, &left_output, &right_output, &parent_output);
            if (ret != 0) {
                fatal_error("find_path error");
            }
            for (l = 0; l < (site_id_t) num_focal_sites; l++) {
                a[focal_sites[l]] = 0;
            }
            for (l = start; l < end; l++) {
                assert(a[l] == match[l]);
            }
            if (total_edges + num_edges > max_edges) {
                fatal_error("out of edge buffer space\n");
            }
            memcpy(left_buffer + total_edges, left_output, num_edges * sizeof(site_id_t));
            memcpy(right_buffer + total_edges, right_output, num_edges * sizeof(site_id_t));
            memcpy(parent_buffer + total_edges, parent_output, num_edges * sizeof(node_id_t));
            child_buffer[j] = child;
            num_edges_buffer[j] = num_edges;
            num_mutations_buffer[j] = num_focal_sites;
            total_edges += num_edges;

            if (verbose > 0) {
                printf("ancestor %d:\t", (int) child);
                for (l = 0; l < (site_id_t) num_sites; l++) {
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
                for (l = 0; l < (int) num_edges; l++) {
                    printf("\t(%d, %d, %d, %d)\n", left_output[l], right_output[l],
                            parent_output[l], child);
                }
            }
            j++;
            child++;
        }

        edge_offset = 0;
        mutation_offset = 0;
        for (j = 0; j < num_ancestors; j++) {
            /* tree_sequence_builder_print_state(&ts_builder, stdout); */
            ret = tree_sequence_builder_add_path(&ts_builder, child_buffer[j],
                    num_edges_buffer[j], left_buffer + edge_offset,
                    right_buffer + edge_offset, parent_buffer + edge_offset,
                    add_path_flags);
            if (ret != 0) {
                fatal_error("add_path");
            }
            edge_offset += num_edges_buffer[j];
            ret = tree_sequence_builder_add_mutations(&ts_builder, child_buffer[j],
                    num_mutations_buffer[j], site_buffer + mutation_offset,
                    derived_state_buffer + mutation_offset);
            if (ret != 0) {
                fatal_error("add_path");
            }
            mutation_offset += num_mutations_buffer[j];
        }
    }

    total_edges = 0;
    total_mutations = 0;
    /* Copy samples */
    for (j = 0; j < num_samples; j++) {
        sample = haplotypes + j * num_sites;
        child = ts_builder.num_nodes + j;
        ret = ancestor_matcher_find_path(&matcher, 0, num_sites, sample, match,
                &num_edges, &left_output, &right_output, &parent_output);
        if (ret != 0) {
            fatal_error("find_path error");
        }
        if (verbose > 0) {
            printf("sample %d:\t", (int) child);
            for (l = 0; l < (site_id_t) num_sites; l++) {
                printf("%d", sample[l]);
            }
            printf("\nmatch = \t");
            for (l = 0; l < (site_id_t) num_sites; l++) {
                printf("%d", match[l]);
            }
            printf("\n");
        }
        num_mutations_buffer[j] = 0;
        for (l = 0; l < (site_id_t) num_sites; l++) {
            if (sample[l] != match[l]) {
                assert(total_mutations < max_mutations);
                node_buffer[total_mutations] = child;
                site_buffer[total_mutations] = l;
                derived_state_buffer[total_mutations] = sample[l];
                total_mutations++;
                num_mutations_buffer[j]++;
            }
        }

        if (total_edges + num_edges > max_edges) {
            fatal_error("out of edge buffer space\n");
        }
        memcpy(left_buffer + total_edges, left_output, num_edges * sizeof(site_id_t));
        memcpy(right_buffer + total_edges, right_output, num_edges * sizeof(site_id_t));
        memcpy(parent_buffer + total_edges, parent_output, num_edges * sizeof(site_id_t));
        child_buffer[j] = child;
        num_edges_buffer[j] = num_edges;
        total_edges += num_edges;

        /* printf("COPIED SAMPLE %d->%d\n", (int) j, (int) sample_id); */
        /* tree_sequence_builder_print_state(&ts_builder, stdout); */
    }


    for (j = 0; j < num_samples; j++) {
        ret = tree_sequence_builder_add_node(&ts_builder, 0, true);
        if (ret < 0) {
            fatal_error("add node");
        }
    }

    edge_offset = 0;
    mutation_offset = 0;
    for (j = 0; j < num_samples; j++) {
        ret = tree_sequence_builder_add_path(&ts_builder, child_buffer[j],
                num_edges_buffer[j], left_buffer + edge_offset,
                right_buffer + edge_offset, parent_buffer + edge_offset,
                add_path_flags);
        if (ret != 0) {
            fatal_error("add_path");
        }
        edge_offset += num_edges_buffer[j];
        ret = tree_sequence_builder_add_mutations(&ts_builder, child_buffer[j],
                num_mutations_buffer[j], site_buffer + mutation_offset,
                derived_state_buffer + mutation_offset);
        if (ret != 0) {
            fatal_error("add_path");
        }
        mutation_offset += num_mutations_buffer[j];
    }

    if (0) {
        output_ts(&ts_builder);
    }

    ancestor_builder_free(&ancestor_builder);
    tree_sequence_builder_free(&ts_builder);
    ancestor_matcher_free(&matcher);
    tsi_safe_free(genotypes);
    tsi_safe_free(focal_sites);
    tsi_safe_free(haplotypes);
    tsi_safe_free(a);
    tsi_safe_free(match);
    tsi_safe_free(left_buffer);
    tsi_safe_free(right_buffer);
    tsi_safe_free(parent_buffer);
    tsi_safe_free(child_buffer);
    tsi_safe_free(num_edges_buffer);
    tsi_safe_free(num_mutations_buffer);
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
    struct arg_lit *path_compression1 = arg_lit0("p", "path-compression", NULL);
    struct arg_file *sample_file1 = arg_file1(NULL, NULL, NULL, NULL);
    struct arg_end *end1 = arg_end(20);
    void* argtable1[] = {cmd1, verbose1, path_compression1, sample_file1, end1};
    int nerrors1;

    int exitcode = EXIT_SUCCESS;
    const char *progname = "main";

    nerrors1 = arg_parse(argc, argv, argtable1);

    if (nerrors1 == 0) {
        run_generate(sample_file1->filename[0], verbose1->count, path_compression1->count);
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
