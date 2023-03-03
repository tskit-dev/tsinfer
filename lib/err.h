#ifndef __ERR_H__
#define __ERR_H__

// clang-format off
#define TSI_ERR_GENERIC                                             -1
#define TSI_ERR_NO_MEMORY                                           -2
#define TSI_ERR_NONCONTIGUOUS_EDGES                                 -3
#define TSI_ERR_UNSORTED_EDGES                                      -4
#define TSI_ERR_PC_ANCESTOR_TIME                                    -5
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
#define TSI_ERR_BAD_FOCAL_SITE                                      -21
#define TSI_ERR_MATCH_IMPOSSIBLE_EXTREME_MUTATION_PROBA             -22
#define TSI_ERR_MATCH_IMPOSSIBLE_ZERO_RECOMB_PRECISION              -23
#define TSI_ERR_ONE_BIT_NON_BINARY                                  -24
#define TSI_ERR_IO                                                  -25
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

const char *tsi_strerror(int err);

#endif /*__ERR_H__*/
