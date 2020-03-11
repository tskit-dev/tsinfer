#ifndef __ERR_H__
#define __ERR_H__

#define TSI_ERR_GENERIC                                             -1
#define TSI_ERR_NO_MEMORY                                           -2
#define TSI_ERR_NONCONTIGUOUS_EDGES                                 -3
#define TSI_ERR_UNSORTED_EDGES                                      -4
#define TSI_ERR_ASSERTION_FAILURE                                   -5
#define TSI_ERR_BAD_PATH_PARENT                                     -6
#define TSI_ERR_BAD_PATH_TIME                                       -7
#define TSI_ERR_MATCH_IMPOSSIBLE                                    -8

#ifdef __GNUC__
    #define WARN_UNUSED __attribute__ ((warn_unused_result))
    #define unlikely(expr) __builtin_expect (!!(expr), 0)
    #define likely(expr) __builtin_expect (!!(expr), 1)
#else
    /* On windows we don't do any perf related stuff */
    #define WARN_UNUSED
    #define restrict
    #define unlikely(expr) (expr)
    #define likely(expr)   (expr)
#endif


#endif /*__ERR_H__*/
