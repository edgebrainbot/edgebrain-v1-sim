#define BLSA_MODULE

#include "cvxopt.h"
#include "misc.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#ifndef _MSC_VER
typedef complex double complex_t;
#else
typedef _Dcomplex complex_t;
#endif

#define USE_CBLAS_ZDOT 0


/* BLAS 1 prototypes */
extern void dswap_(int *n, double *x, int *incx, double *y, int *incy);
extern void zswap_(int *n, complex_t *x, int *incx, complex_t *y,
    int *incy);
extern void dscal_(int *n, double *alpha, double *x, int *incx);
extern void zscal_(int *n, complex_t *alpha, complex_t *x, int *incx);
extern void zdscal_(int *n, double *alpha, complex_t *x, int *incx);
extern void dcopy_(int *n, double *x, int *incx, double *y, int *incy);
extern void zcopy_(int *n, complex_t *x, int *incx, complex_t *y,
    int *incy);
extern void daxpy_(int *n, double *alpha, double *x, int *incx,
    double *y, int *incy);
extern void zaxpy_(int *n, complex_t *alpha, complex_t *x, int *incx,
    complex_t *y, int *incy);
extern double ddot_(int *n, double *x, int *incx, double *y, int *incy);
#if USE_CBLAS_ZDOT
extern void cblas_zdotc_sub(int n, void *x, int incx, void *y,
    int incy, void *result);
extern void cblas_zdotu_sub(int n, void *x, int incx, void *y, int incy,
    void *result);
#endif
extern double dnrm2_(int *n, double *x, int *incx);
extern double dznrm2_(int *n, complex_t *x, int *incx);
extern double dasum_(int *n, double *x, int *incx);
extern double dzasum_(int *n, complex_t *x, int *incx);
extern int idamax_(int *n, double *x, int *incx);
extern int izamax_(int *n, complex_t *x, int *incx);


/* BLAS 2 prototypes */
extern void dgemv_(char* trans, int *m, int *n, double *alpha,
    double *A, int *lda, double *x, int *incx, double *beta, double *y,
    int *incy);
extern void zgemv_(char* trans, int *m, int *n, complex_t *alpha,
    complex_t *A, int *lda, complex_t *x, int *incx, complex_t *beta,
    complex_t *y, int *incy);
extern void dgbmv_(char* trans, int *m, int *n, int *kl, int *ku,
    double *alpha, double *A, int *lda, double *x, int *incx,
    double *beta, double *y,  int *incy);
extern void zgbmv_(char* trans, int *m, int *n, int *kl, int *ku,
    complex_t *alpha, complex_t *A, int *lda, complex_t *x, int *incx,
    complex_t *beta, complex_t *y,  int *incy);
extern void dsymv_(char *uplo, int *n, double *alpha, double *A,
    int *lda, double *x, int *incx, double *beta, double *y, int *incy);
extern void zhemv_(char *uplo, int *n, complex_t *alpha, complex_t *A,
    int *lda, complex_t *x, int *incx, complex_t *beta, complex_t *y,
    int *incy);
extern void dsbmv_(char *uplo, int *n, int *k, double *alpha, double *A,
    int *lda, double *x, int *incx, double *beta, double *y, int *incy);
extern void zhbmv_(char *uplo, int *n, int *k, complex_t *alpha,
    complex_t *A, int *lda, complex_t *x, int *incx, complex_t *beta,
    complex_t *y, int *incy);
extern void dtrmv_(char *uplo, char *trans, char *diag, int *n,
    double *A, int *lda, double *x, int *incx);
extern void ztrmv_(char *uplo, char *trans, char *diag, int *n,
    complex_t *A, int *lda, complex_t *x, int *incx);
extern void dtbmv_(char *uplo, char *trans, char *diag, int *n, int *k,
    double *A, int *lda, double *x, int *incx);
extern void ztbmv_(char *uplo, char *trans, char *diag, int *n, int *k,
    complex_t *A, int *lda, complex_t *x, int *incx);
extern void dtrsv_(char *uplo, char *trans, char *diag, int *n,
    double *A, int *lda, double *x, int *incx);
extern void ztrsv_(char *uplo, char *trans, char *diag, int *n,
    complex_t *A, int *lda, complex_t *x, int *incx);
extern void dtbsv_(char *uplo, char *trans, char *diag, int *n, int *k,
    double *A, int *lda, double *x, int *incx);
extern void ztbsv_(char *uplo, char *trans, char *diag, int *n, int *k,
    complex_t *A, int *lda, complex_t *x, int *incx);
extern void dger_(int *m, int *n, double *alpha, double *x, int *incx,
    double *y, int *incy, double *A, int *lda);
extern void zgerc_(int *m, int *n, complex_t *alpha, complex_t *x,
    int *incx, complex_t *y, int *incy, complex_t *A, int *lda);
extern void zgeru_(int *m, int *n, complex_t *alpha, complex_t *x,
    int *incx, complex_t *y, int *incy, complex_t *A, int *lda);
extern void dsyr_(char *uplo, int *n, double *alpha, double *x,
    int *incx, double *A, int *lda);
extern void zher_(char *uplo, int *n, double *alpha, complex_t *x,
    int *incx, complex_t *A, int *lda);
extern void dsyr2_(char *uplo, int *n, double *alpha, double *x,
    int *incx, double *y, int *incy, double *A, int *lda);
extern void zher2_(char *uplo, int *n, complex_t *alpha, complex_t *x,
    int *incx, complex_t *y, int *incy, complex_t *A, int *lda);


/* BLAS 3 prototypes */
extern void dgemm_(char *transa, char *transb, int *m, int *n, int *k,
    double *alpha, double *A, int *lda, double *B, int *ldb,
    double *beta, double *C, int *ldc);
extern void zgemm_(char *transa, char *transb, int *m, int *n, int *k,
    complex_t *alpha, complex_t *A, int *lda, complex_t *B, int *ldb,
    complex_t *beta, complex_t *C, int *ldc);
extern void dsymm_(char *side, char *uplo, int *m, int *n,
    double *alpha, double *A, int *lda, double *B, int *ldb,
    double *beta, double *C, int *ldc);
extern void zsymm_(char *side, char *uplo, int *m, int *n,
    complex_t *alpha, complex_t *A, int *lda, complex_t *B, int *ldb,
    complex_t *beta, complex_t *C, int *ldc);
extern void zhemm_(char *side, char *uplo, int *m, int *n,
    complex_t *alpha, complex_t *A, int *lda, complex_t *B, int *ldb,
    complex_t *beta, complex_t *C, int *ldc);
extern void dsyrk_(char *uplo, char *trans, int *n, int *k,
    double *alpha, double *A, int *lda, double *beta, double *B,
    int *ldb);
extern void zsyrk_(char *uplo, char *trans, int *n, int *k,
    complex_t *alpha, complex_t *A, int *lda, complex_t *beta, complex_t *B,
    int *ldb);
extern void zherk_(char *uplo, char *trans, int *n, int *k,
    double *alpha, complex_t *A, int *lda, double *beta, complex_t *B,
    int *ldb);
extern void dsyr2k_(char *uplo, char *trans, int *n, int *k,
    double *alpha, double *A, int *lda, double *B, int *ldb,
    double *beta, double *C, int *ldc);
extern void zsyr2k_(char *uplo, char *trans, int *n, int *k,
    complex_t *alpha, complex_t *A, int *lda, complex_t *B, int *ldb,
    complex_t *beta, complex_t *C, int *ldc);
extern void zher2k_(char *uplo, char *trans, int *n, int *k,
    complex_t *alpha, complex_t *A, int *lda, complex_t *B, int *ldb,
    double *beta, complex_t *C, int *ldc);
extern void dtrmm_(char *side, char *uplo, char *transa, char *diag,
    int *m, int *n, double *alpha, double *A, int *lda, double *B,
    int *ldb);
extern void ztrmm_(char *side, char *uplo, char *transa, char *diag,
    int *m, int *n, complex_t *alpha, complex_t *A, int *lda, complex_t *B,
    int *ldb);
extern void dtrsm_(char *side, char *uplo, char *transa, char *diag,
    int *m, int *n, double *alpha, double *A, int *lda, double *B,
    int *ldb);
extern void ztrsm_(char *side, char *uplo, char *transa, char *diag,
    int *m, int *n, complex_t *alpha, complex_t *A, int *lda, complex_t *B,
    int *ldb);


// BLAS function prototypes
extern void dgemv_(char* trans, int* m, int* n, double* alpha, 
                   double* A, int* lda, double* x, int* incx, 
                   double* beta, double* y, int* incy);
extern void zgemv_(char* trans, int* m, int* n, double complex* alpha, 
                   double complex* A, int* lda, double complex* x, int* incx, 
                   double complex* beta, double complex* y, int* incy);
extern void dscal_(int* n, double* alpha, double* x, int* incx);
extern void zscal_(int* n, double complex* alpha, double complex* x, int* incx);


// Microsoft Visual C++ complex number support
#ifdef _MSC_VER
#define _Cbuild(r,i) ((double complex)((r) + (i)*I))
#endif

// General matrix-vector product function
// If trans is 'N', computes y := alpha*A*x + beta*y.
// If trans is 'T', computes y := alpha*A^T*x + beta*y.
// If trans is 'C', computes y := alpha*A^H*x + beta*y.
// The matrix A is m by n.
// Returns immediately if n=0 and trans is 'T' or 'C', or if m=0 
// and trans is 'N'.
// Computes y := beta*y if n=0, m>0 and trans is 'N', or if m=0, 
// n>0 and trans is 'T' or 'C'.
//
// ARGUMENTS
// A         'd' or 'z' matrix
// x         'd' or 'z' matrix.  Must have the same type as A.
// y         'd' or 'z' matrix.  Must have the same type as A.
// trans     'N', 'T' or 'C'
// alpha     number (int, float or complex).  Complex alpha is only
//           allowed if A is complex.
// beta      number (int, float or complex).  Complex beta is only
//           allowed if A is complex.
// m         integer.  If negative, the default value is used.
// n         integer.  If negative, the default value is used.
// ldA       nonnegative integer.  ldA >= max(1,m).
//           If zero, the default value is used.
// incx      nonzero integer
// incy      nonzero integer
// offsetA   nonnegative integer
// offsetx   nonnegative integer
// offsety   nonnegative integer
// alpha_ptr pointer to alpha value (can be NULL for default)
// beta_ptr  pointer to beta value (can be NULL for default)
// int gemv(matrix *A, matrix *x, matrix *y, char trans, 
//          number *alpha_ptr, number *beta_ptr, int m, int n, 
//          int ldA, int incx, int incy, int offsetA, int offsetx, int offsety)
// {
//     number a, b;
//     int ix = incx, iy = incy, oA = offsetA, ox = offsetx, oy = offsety;

//     // Validate input matrices
//     if (A == NULL) {
//         err_mtrx("A");
//         return -1;
//     }
//     if (x == NULL) {
//         err_mtrx("x");
//         return -1;
//     }
//     if (y == NULL) {
//         err_mtrx("y");
//         return -1;
//     }
    
//     // Check matrix type consistency
//     if (MAT_ID(A) != MAT_ID(x) || MAT_ID(A) != MAT_ID(y) ||
//         MAT_ID(x) != MAT_ID(y)) {
//         err_conflicting_ids();
//         return -1;
//     }

//     // Validate trans parameter
//     if (trans != 'N' && trans != 'T' && trans != 'C') {
//         err_char("trans", "'N','T','C'");
//         return -1;
//     }

//     // Validate increment parameters
//     if (ix == 0) {
//         err_nz_int("incx");
//         return -1;
//     }
//     if (iy == 0) {
//         err_nz_int("incy");
//         return -1;
//     }

//     // Set default dimensions if negative
//     if (m < 0) m = A->nrows;
//     if (n < 0) n = A->ncols;
    
//     // Early return for degenerate cases
//     if ((!m && trans == 'N') || (!n && (trans == 'T' || trans == 'C')))
//         return 0;

//     // Set default leading dimension
//     if (ldA == 0) ldA = MAX(1, A->nrows);
//     if (ldA < MAX(1, m)) {
//         err_ld("ldA");
//         return -1;
//     }

//     // Validate offsets and buffer bounds
//     if (oA < 0) {
//         err_nn_int("offsetA");
//         return -1;
//     }
//     if (n > 0 && m > 0 && oA + (n-1)*ldA + m > len(A)) {
//         err_buf_len("A");
//         return -1;
//     }

//     if (ox < 0) {
//         err_nn_int("offsetx");
//         return -1;
//     }
//     if ((trans == 'N' && n > 0 && ox + (n-1)*abs(ix) + 1 > len(x)) ||
//         ((trans == 'T' || trans == 'C') && m > 0 &&
//          ox + (m-1)*abs(ix) + 1 > len(x))) {
//         err_buf_len("x");
//         return -1;
//     }

//     if (oy < 0) {
//         err_nn_int("offsety");
//         return -1;
//     }
//     if ((trans == 'N' && oy + (m-1)*abs(iy) + 1 > len(y)) ||
//         ((trans == 'T' || trans == 'C') &&
//          oy + (n-1)*abs(iy) + 1 > len(y))) {
//         err_buf_len("y");
//         return -1;
//     }

//     // Handle alpha and beta parameters
//     if (alpha_ptr != NULL) {
//         a = *alpha_ptr;
//     }
//     if (beta_ptr != NULL) {
//         b = *beta_ptr;
//     }

//     // Perform the operation based on matrix type
//     switch (MAT_ID(x)) {
//         case DOUBLE:
//             // Set default values if not provided
//             if (alpha_ptr == NULL) a.d = 1.0;
//             if (beta_ptr == NULL) b.d = 0.0;
            
//             if (trans == 'N' && n == 0) {
//                 dscal_(&m, &b.d, MAT_BUFD(y)+oy, &iy);
//             }
//             else if ((trans == 'T' || trans == 'C') && m == 0) {
//                 dscal_(&n, &b.d, MAT_BUFD(y)+oy, &iy);
//             }
//             else {
//                 dgemv_(&trans, &m, &n, &a.d, MAT_BUFD(A)+oA, &ldA,
//                     MAT_BUFD(x)+ox, &ix, &b.d, MAT_BUFD(y)+oy, &iy);
//             }
//             break;

//         case COMPLEX:
//             // Set default values if not provided
// #ifndef _MSC_VER
//             if (alpha_ptr == NULL) a.z = 1.0;
//             if (beta_ptr == NULL) b.z = 0.0;
// #else
//             if (alpha_ptr == NULL) a.z = _Cbuild(1.0, 0.0);
//             if (beta_ptr == NULL) b.z = _Cbuild(0.0, 0.0);
// #endif
            
//             if (trans == 'N' && n == 0) {
//                 zscal_(&m, &b.z, MAT_BUFZ(y)+oy, &iy);
//             }
//             else if ((trans == 'T' || trans == 'C') && m == 0) {
//                 zscal_(&n, &b.z, MAT_BUFZ(y)+oy, &iy);
//             }
//             else {
//                 zgemv_(&trans, &m, &n, &a.z, MAT_BUFZ(A)+oA, &ldA,
//                     MAT_BUFZ(x)+ox, &ix, &b.z, MAT_BUFZ(y)+oy, &iy);
//             }
//             break;

//         default:
//             err_invalid_id();
//             return -1;
//     }

//     return 0;
// }


// Documentation string (converted to comment)
/*
 * Scales a vector by a constant (x := alpha*x).
 *
 * int scal(alpha, x, n=-1, inc=1, offset=0);
 *
 * ARGUMENTS
 * alpha     number (int, float or complex).  Complex alpha is only
 *           allowed if x is complex.
 *
 * x         'd' or 'z' matrix
 *
 * n         integer.  If n<0, the default value of n is used.
 *           The default value is equal to
 *           (len(x)>=offset+1) ? 1+(len-offset-1)/inc : 0.
 *
 * inc       positive integer
 *
 * offset    nonnegative integer
 */

// int scal(void* alpha, matrix* x, int n, int inc, int offset) {

//     number a;

//     if (!Matrix_Check(x)) err_mtrx("x");
//     if (inc <= 0) err_p_int("inc");
//     if (offset < 0) err_nn_int("offset");
//     if (n < 0) n = (len(x) >= offset+1) ? 1+(len(x)-offset-1)/inc : 0;
//     if (n == 0) return SCAL_SUCCESS;
//     if (len(x) < offset+1+(n-1)*inc) err_buf_len("x");

//     switch (MAT_ID(x)){
//         case DOUBLE:
//             if (number_from_raw(alpha, &a, MAT_ID(x)))
//                 err_type("alpha");
//             dscal_(&n, &a.d, MAT_BUFD(x)+offset, &inc);
// 	    break;

//         case COMPLEX:
//             if (!number_from_raw(alpha, &a, DOUBLE))
//                 zdscal_(&n, &a.d, MAT_BUFZ(x)+offset, &inc);
//             else if (!number_from_raw(alpha, &a, COMPLEX))
//                 zscal_(&n, &a.z, MAT_BUFZ(x)+offset, &inc);
//             else
//                 err_type("alpha");
// 	    break;

//         default:
//             err_invalid_id();
//     }

//     return SCAL_SUCCESS; // success
// }

/**
 * Interprets a void* input as either double or complex and stores the value in a `number` struct.
 *
 * Parameters:
 *   src - Pointer to a double or double complex value
 *   a   - Output pointer to the `number` struct
 *   id  - Type identifier (DOUBLE or COMPLEX)
 *
 * Returns:
 *   0 on success, -1 on invalid input or unsupported type
 */
int number_from_raw(const void *src, number *a, int id) {
    if (!src || !a) return -1;

    switch (id) {
        case DOUBLE:
            a->d = *(const double *)src;
            return 0;

        case COMPLEX:
            a->z = *(const double complex *)src;
            return 0;

        default:
            return -1;
    }
}

