#ifndef __ERR_H__
#define __ERR_H__

// clang-format off
#define TSI_ERR_GENERIC                                             -1
#define TSI_ERR_NO_MEMORY                                           -2
#define TSI_ERR_BAD_NUM_SAMPLES                                     -19
#define TSI_ERR_TOO_MANY_SITES                                      -20
#define TSI_ERR_BAD_FOCAL_SITE                                      -21
#define TSI_ERR_ONE_BIT_NON_BINARY                                  -24
#define TSI_ERR_IO                                                  -25
// clang-format on

#ifdef __GNUC__
#define WARN_UNUSED    __attribute__((warn_unused_result))
#define unlikely(expr) __builtin_expect(!!(expr), 0)
#define likely(expr)   __builtin_expect(!!(expr), 1)
#else
/* On windows we don't do any perf related stuff */
#define WARN_UNUSED
#define restrict
#define unlikely(expr) (expr)
#define likely(expr)   (expr)
#endif

const char *tsi_strerror(int err);

#endif /*__ERR_H__*/
