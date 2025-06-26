/*
 * Simplified CVXOPT MISC header for pure C use (no Python API)
 * Copyright stripped for simplification.
 */

#ifndef __MISC__
#define __MISC__

#include <stdio.h>
#include <stdlib.h>

#include <string.h>

#include "cvxopt.h"

#ifndef NO_ANSI99_COMPLEX
typedef union {
    double d;
    int_t i;
    double complex z;
} number;
#endif

#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))

#define ERR(s)        (fprintf(stderr, "%s\n", s), NULL)
#define ERR_INT(s)    (fprintf(stderr, "%s\n", s), -1)
#define ERR_TYPE(s)   (fprintf(stderr, "TypeError: %s\n", s), NULL)

/* C-style cyclic wrap-around for indices */
#define CWRAP(i,m) ((i) >= 0 ? (i) : ((m)+(i)))
#define OUT_RNG(i, dim) ((i) < -(dim) || (i) >= (dim))

#define VALID_TC_MAT(t) ((t)=='i' || (t)=='d' || (t)=='z')
#define VALID_TC_SP(t)  ((t)=='d' || (t)=='z')
#define TC2ID(c) ((c)=='i' ? 0 : ((c)=='d' ? 1 : 2))

#define X_ID(O)    (Matrix_Check(O) ? MAT_ID(O)    : SP_ID(O))
#define X_NROWS(O) (Matrix_Check(O) ? MAT_NROWS(O) : SP_NROWS(O))
#define X_NCOLS(O) (Matrix_Check(O) ? MAT_NCOLS(O) : SP_NCOLS(O))
#define X_Matrix_Check(O) (Matrix_Check(O) || SpMatrix_Check(O))

/* Error handling macros */
#define ERR_ABORT(msg) do { fprintf(stderr, "[ERROR] %s\n", msg); abort(); } while (0)

#define len(x) (Matrix_Check(x) ? MAT_LGT(x) : SP_LGT(x))

#define SCAL_SUCCESS 0

#define err_mtrx(s) ERR_ABORT(s " must be a matrix")
#define err_bool(s) ERR_ABORT(s " must be true or false")
#define err_conflicting_ids() ERR_ABORT("conflicting types for matrix arguments")
#define err_invalid_id() ERR_ABORT("matrix arguments must have type 'd' or 'z'")
#define err_nz_int(s) ERR_ABORT(s " must be a nonzero integer")
#define err_nn_int(s) ERR_ABORT(s " must be a nonnegative integer")
#define err_buf_len(s) ERR_ABORT("length of " s " is too small")
#define err_type(s) ERR_ABORT("incompatible type for " s)
#define err_p_int(s) ERR_ABORT(s " must be a positive integer")
#define err_char(s1,s2) ERR_ABORT("possible values of " s1 " are: " s2)
#define err_ld(s) ERR_ABORT("illegal value of " s)
#define err_int_mtrx(s) ERR_ABORT(s " must be a matrix with typecode 'i'")
#define err_dbl_mtrx(s) ERR_ABORT(s " must be a matrix with typecode 'd'")
#define err_msk_noparam ERR_ABORT("missing options dictionary")

#endif
