/*
** Copyright (C) 2020 University of Oxford
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

#include "err.h"
#include <tskit.h>

const char *
tsi_strerror(int err)
{
    const char *ret = "Unknown error";

    switch (err) {
        case 0:
            ret = "Normal exit condition. This is not an error!";
            break;

        case TSI_ERR_GENERIC:
            ret = "Generic tsinfer error - please file a bug report.";
            break;
        case TSI_ERR_NO_MEMORY:
            ret = "Out of memory";
            break;
        case TSI_ERR_BAD_NUM_SAMPLES:
            ret = "Must have at least 2 samples.";
            break;
        case TSI_ERR_TOO_MANY_SITES:
            ret = "Cannot add more sites than the specified maximum.";
            break;
        case TSI_ERR_BAD_FOCAL_SITE:
            ret = "Bad focal site.";
            break;
        case TSI_ERR_ONE_BIT_NON_BINARY:
            ret = "One-bit genotype encoding only supports binary 0/1 data";
            break;
        case TSI_ERR_IO:
            ret = tsk_strerror(TSK_ERR_IO);
            break;
    }
    return ret;
}

/* shim allocator moved to lib/tsi_blkalloc.c */
