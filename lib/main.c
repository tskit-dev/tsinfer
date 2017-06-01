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

static void
read_ancestors(const char *input_file, size_t *r_num_ancestors, size_t *r_num_sites,
        int8_t **r_ancestors)
{
    /* int ret; */
    char * line = NULL;
    size_t len = 0;
    size_t j, k;
    size_t num_line_tokens;
    size_t num_ancestors  = 0;
    size_t num_sites = (size_t) -1;
    const char delimiters[] = " \t";
    char *token;
    int8_t *ancestors = NULL;
    FILE *f = fopen(input_file, "r");

    if (f == NULL) {
        fatal_error("Cannot open %s: %s", input_file, strerror(errno));
    }
    while (getline(&line, &len, f) != -1) {
        /* read the number of tokens */
        token = strtok(line, delimiters);
        num_line_tokens = 0;
        while (token != NULL) {
            num_line_tokens++;
            token = strtok(NULL, delimiters);
        }

        if (num_sites == (size_t) -1) {
            num_sites = num_line_tokens;
        } else if (num_sites != num_line_tokens) {
            fatal_error("Bad input: line lengths not equal");
        }
        num_ancestors++;
    }
    if (fseek(f, 0, 0) != 0) {
        fatal_error("Cannot seek in file");
    }

    ancestors = malloc(num_ancestors * num_sites * sizeof(int8_t));
    if (ancestors == NULL) {
        fatal_error("No memory");
    }
    j = 0;
    while (getline(&line, &len, f) != -1) {
        k = 0;
        token = strtok(line, delimiters);
        while (token != NULL) {
            ancestors[j * num_sites + k] = (int8_t) atoi(token);
            token = strtok(NULL, delimiters);
            k++;
        }
        j++;
    }
    free(line);
    fclose(f);

    *r_num_ancestors = num_ancestors;
    *r_num_sites = num_sites;
    *r_ancestors = ancestors;
}

int
main(int argc, char **argv)
{
    int8_t *ancestors = NULL;
    size_t num_ancestors;
    size_t num_sites;
    size_t j;
    ancestor_matcher_t am;
    int ret;

    if (argc != 2) {
        fatal_error("usage: main <ancestors-file>");
    }
    read_ancestors(argv[1], &num_ancestors, &num_sites, &ancestors);

    ret = ancestor_matcher_alloc(&am, 10, 1);
    if (ret != 0) {
        fatal_error("alloc error");
    }
    for (j = 0; j < num_ancestors; j++) {
        ret = ancestor_matcher_add(&am, ancestors + j * num_sites);
        if (ret != 0) {
            fatal_error("alloc error");
        }
        ancestor_matcher_print_state(&am, stdout);
    }

    ancestor_matcher_free(&am);
    free(ancestors);
    return EXIT_SUCCESS;
}
