#ifndef __ERR_H__
#define __ERR_H__

/*
 * raise a compiler warning if a potentially error raising function's return
 * value is not used.
 */
#ifdef __GNUC__
    #define WARN_UNUSED __attribute__ ((warn_unused_result))
#else
    #define WARN_UNUSED
#endif

#define TSI_ERR_GENERIC                                             -1
#define TSI_ERR_NO_MEMORY                                           -2
#define TSI_ERR_NONCONTIGUOUS_EDGES                                 -3
#define TSI_ERR_UNSORTED_EDGES                                      -4

#endif /*__ERR_H__*/
