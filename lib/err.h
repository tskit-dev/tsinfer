#ifndef __ERR_H__
#define __ERR_H__

// clang-format off
#define TSI_ERR_GENERIC                                             -1
#define TSI_ERR_NO_MEMORY                                           -2
#define TSI_ERR_NONCONTIGUOUS_EDGES                                 -3
#define TSI_ERR_UNSORTED_EDGES                                      -4
#define TSI_ERR_ASSERTION_FAILURE                                   -5
#define TSI_ERR_BAD_PATH_CHILD                                      -6
#define TSI_ERR_BAD_PATH_PARENT                                     -7
#define TSI_ERR_BAD_PATH_TIME                                       -8
#define TSI_ERR_BAD_PATH_INTERVAL                                   -9
#define TSI_ERR_BAD_PATH_LEFT_LESS_ZERO                             -10
#define TSI_ERR_BAD_PATH_RIGHT_GREATER_NUM_SITES                    -11
#define TSI_ERR_MATCH_IMPOSSIBLE                                    -12
#define TSI_ERR_BAD_HAPLOTYPE_ALLELE                                -13
#define TSI_ERR_BAD_NUM_ALLELES                                     -14
#define TSI_ERR_BAD_MUTATION_NODE                                   -15
#define TSI_ERR_BAD_MUTATION_SITE                                   -16
#define TSI_ERR_BAD_MUTATION_DERIVED_STATE                          -17
#define TSI_ERR_BAD_MUTATION_DUPLICATE_NODE                         -18
#define TSI_ERR_BAD_NUM_SAMPLES                                     -19
#define TSI_ERR_TOO_MANY_SITES                                      -20
// clang-format on

#ifdef __GNUC__
#define WARN_UNUSED __attribute__((warn_unused_result))
#define unlikely(expr) __builtin_expect(!!(expr), 0)
#define likely(expr) __builtin_expect(!!(expr), 1)
#else
/* On windows we don't do any perf related stuff */
#define WARN_UNUSED
#define restrict
#define unlikely(expr) (expr)
#define likely(expr) (expr)
#endif

#endif /*__ERR_H__*/
