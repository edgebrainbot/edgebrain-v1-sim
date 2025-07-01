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

typedef struct {
    matrix *dnl;
    matrix *dnli;
    matrix *d;
    matrix *di;
    matrix **v;      // array of matrix pointers
    double *beta;    // array of doubles
    matrix **r;      // array of matrix pointers
    matrix **rti;    // array of matrix pointers
    int v_count;     // number of v matrices
    int r_count;     // number of r matrices
    int has_dnl;     // flag indicating if dnl/dnli entries exist
} scaling;

/*
- dims['l'] = ml, the dimension of the nonnegative orthant C_0.
    (ml >= 0.)
- dims['q'] = mq = [ mq[0], mq[1], ..., mq[N-1] ], a list of N
    integers with the dimensions of the second order cones C_1, ...,
    C_N.  (N >= 0 and mq[k] >= 1.)
- dims['s'] = ms = [ ms[0], ms[1], ..., ms[M-1] ], a list of M
    integers with the orders of the semidefinite cones C_{N+1}, ...,
    C_{N+M}.  (M >= 0 and ms[k] >= 0.)
The default value of dims is {'l': G.size[0], 'q': [], 's': []}.
*/

typedef struct {
    int l;              // 'l' dimension
    int *q;             // 'q' dimension
    int q_size;         // size of q array
    int *s;             // 's' dimension
    int s_size;         // size of s array
} DIMs;

#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))

/* Error handling macros */
#define ERR(s)        (fprintf(stderr, "%s\n", s), abort())
#define ERR_INT(s)    (fprintf(stderr, "%s\n", s), -1)
#define ERR_TYPE(s)   (fprintf(stderr, "TypeError: %s\n", s), abort())

#define err_mtrx(s)             ERR_TYPE(s " must be a matrix")
#define err_bool(s)             ERR_TYPE(s " must be true or false")
#define err_conflicting_ids     ERR_TYPE("conflicting types for matrix arguments")
#define err_invalid_id          ERR_TYPE("matrix arguments must have type 'd' or 'z'")
#define err_nz_int(s)           ERR_TYPE(s " must be a nonzero integer")
#define err_nn_int(s)           ERR_TYPE(s " must be a nonnegative integer")
#define err_buf_len(s)          ERR_TYPE("length of " s " is too small")
#define err_type(s)             ERR_TYPE("incompatible type for " s)
#define err_p_int(s)            ERR(s " must be a positive integer")
#define err_char(s1,s2)         ERR("possible values of " s1 " are: " s2)
#define err_ld(s)               ERR("illegal value of " s)
#define err_int_mtrx(s)         ERR_TYPE(s " must be a matrix with typecode 'i'")
#define err_dbl_mtrx(s)         ERR_TYPE(s " must be a matrix with typecode 'd'")
#define err_msk_noparam         "missing options dictionary"
#define err_no_memory           ERR("no memory use for matrix calculation")

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

#define len(x) (Matrix_Check(x) ? MAT_LGT(x) : SP_LGT(x))

// /* FUNC SUCCESS FLAG */
// #define SCAL_SUCCESS 0
// #define GEMM_SUCCESS 0

#endif
