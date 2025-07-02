/*
 * Copyright 2012-2023 M. Andersen and L. Vandenberghe.
 * Copyright 2010-2011 L. Vandenberghe.
 * Copyright 2004-2009 J. Dahl and L. Vandenberghe.
 *
 * This file is part of CVXOPT.
 *
 * CVXOPT is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * CVXOPT is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "cvxopt.h"
#include "misc.h"
#include "math.h"
#include "float.h"

/* misc_solver prototype */
extern void dcopy_(int *n, double *x, int *incx, double *y, int *incy);
extern double dnrm2_(int *n, double *x, int *incx);
extern double ddot_(int *n, double *x, int *incx, double *y, int *incy);
extern void dscal_(int *n, double *alpha, double *x, int *incx);
extern void daxpy_(int *n, double *alpha, double *x, int *incx, double *y,
    int *incy);
extern void dtbmv_(char *uplo, char *trans, char *diag, int *n, int *k,
    double *A, int *lda, double *x, int *incx);
extern void dtbsv_(char *uplo, char *trans, char *diag, int *n, int *k,
    double *A, int *lda, double *x, int *incx);
extern void dgemv_(char* trans, int *m, int *n, double *alpha, double *A,
    int *lda, double *x, int *incx, double *beta, double *y, int *incy);
extern void dger_(int *m, int *n, double *alpha, double *x, int *incx,
    double *y, int *incy, double *A, int *lda);
extern void dtrmm_(char *side, char *uplo, char *transa, char *diag,
    int *m, int *n, double *alpha, double *A, int *lda, double *B,
    int *ldb);
extern void dsyr2k_(char *uplo, char *trans, int *n, int *k, double *alpha,
    double *A, int *lda, double *B, int *ldb, double *beta, double *C,
    int *ldc);
extern void dlacpy_(char *uplo, int *m, int *n, double *A, int *lda,
    double *B, int *ldb);
extern void dsyevr_(char *jobz, char *range, char *uplo, int *n, double *A,
    int *ldA, double *vl, double *vu, int *il, int *iu, double *abstol,
    int *m, double *W, double *Z, int *ldZ, int *isuppz, double *work,
    int *lwork, int *iwork, int *liwork, int *info);
extern void dsyevd_(char *jobz, char *uplo, int *n, double *A, int *ldA,
    double *W, double *work, int *lwork, int *iwork, int *liwork,
    int *info);

/* blas library */
extern void blas_tbsv(matrix *A, matrix *x, char uplo, char trans, char diag, 
    int n, int k, int ldA, int incx, int offsetA, int offsetx);
extern number blas_dot(matrix *x, matrix *y, int n, int incx, int incy, 
    int offsetx, int offsety);
extern double blas_nrm2(matrix *x, int n, int inc, int offset);
extern void blas_scal(void* alpha, matrix* x, int n, int inc, int offset);
extern void blas_copy(matrix *x, matrix *y, int n, int ix, int iy, int ox, int oy);
extern void blas_axpy(matrix *x, matrix *y, number *alpha, int n, int incx, 
    int incy, int offsetx, int offsety);
extern void blas_tbmv(matrix *A, matrix *x, char uplo, char trans, char diag, 
    int n, int k, int ldA, int incx, int offsetA, int offsetx);

/**
 * @brief Applies Nesterov-Todd scaling or its inverse to a vector
 * 
 * misc_scale(x, W, trans = 'N', inverse = 'N')
 * 
 * @details
 * Computes one of four scaling operations based on parameters:
 * - x := W*x       (trans='N', inverse='N')
 * - x := Wᵀ*x      (trans='T', inverse='N')
 * - x := W⁻¹*x     (trans='N', inverse='I')
 * - x := W⁻ᵀ*x     (trans='T', inverse='I')
 *
 * The scaling matrix W is defined by a dictionary containing:
 * - dnl, dnli:  Positive vector and its inverse (optional, nonlinear solver only)
 * - d, di:      Positive vector and its inverse
 * - v:          List of 2nd-order cone vectors with unit hyperbolic norms
 * - beta:       List of positive scaling factors
 * - r:          List of square matrices
 * - rti:        List of inverse-transpose matrices of r[k]
 *
 * @param[in,out] x      Dense 'd' matrix to scale (input/output)
 * @param[in] W          Scaling dictionary containing NT parameters
 * @param[in] trans      Transpose flag ('N' or 'T')
 * @param[in] inverse    Inverse flag ('N' or 'I')
 *
 * @note
 * - Implements Nesterov-Todd scaling for conic optimization
 * - Handles both regular and inverse transformations
 * - Supports transposed operations
 * - dnl/dnli entries are optional and nonlinear-solver specific
 *
 * @warning
 * - x must be a dense 'd' matrix
 * - W must contain all required fields (except optional dnl/dnli)
 * - trans and inverse must be valid flags ('N', 'T', or 'I')
 * - All vectors/matrices must have compatible dimensions
 */
void misc_scale(matrix *x, scaling *W, char trans, char inverse) {
    matrix *d, *vk, *rk;

    // Set default character values
    if (trans == 0) trans = 'N';
    if (inverse == 0) inverse = 'N';

    int m, n, xr, xc, ind = 0, int0 = 0, int1 = 1, i, k, inc, len, ld, 
        maxn, N;
    double b, dbl0 = 0.0, dbl1 = 1.0, dblm1 = -1.0, dbl2 = 2.0, dbl5 = 0.5,
        *wrk;

    xr = x->nrows;
    xc = x->ncols;

    /*
     * Scaling for nonlinear component xk is xk := dnl .* xk; inverse is
     * xk ./ dnl = dnli .* xk, where dnl = W['dnl'], dnli = W['dnli'].
     */

    if ((d = (inverse == 'N') ? (matrix *) W->dnl : (matrix *) W->dnli)){
        m = len(d);
        for (i = 0; i < xc; ++i)
            dtbmv_("L", "N", "N", &m, &int0, MAT_BUFD(d), &int1,
                MAT_BUFD(x) + i*xr, &int1);
        ind += m;
    }

    /*
     * Scaling for 'l' component xk is xk := d .* xk; inverse scaling is
     * xk ./ d = di .* xk, where d = W['d'], di = W['di'].
     */

    if (!(d = (inverse == 'N') ? (matrix *) W->d : (matrix *) W->di)){
        ERR("missing item W['d'] or W['di']");
        return;
    }
    m = len(d);
    for (i = 0; i < xc; i++)
        dtbmv_("L", "N", "N", &m, &int0, MAT_BUFD(d), &int1, MAT_BUFD(x)
            + i*xr + ind, &int1);
    ind += m;

    /*
     * Scaling for 'q' component is
     *
     *     xk := beta * (2*v*v' - J) * xk
     *         = beta * (2*v*(xk'*v)' - J*xk)
     *
     * where beta = W['beta'][k], v = W['v'][k], J = [1, 0; 0, -I].
     *
     * Inverse scaling is
     *
     *     xk := 1/beta * (2*J*v*v'*J - J) * xk
     *         = 1/beta * (-J) * (2*v*((-J*xk)'*v)' + xk).
     */

    N = W->v_count;
    if (!(wrk = (double *) calloc(xc, sizeof(double)))) err_no_memory;
    for (k = 0; k < N; ++k){
        vk = (matrix *) W->v[k];
        m = vk->nrows;
        if (inverse == 'I')
            dscal_(&xc, &dblm1, MAT_BUFD(x) + ind, &xr);
        ld = MAX(xr, 1);
        dgemv_("T", &m, &xc, &dbl1, MAT_BUFD(x) + ind, &ld, MAT_BUFD(vk), 
            &int1, &dbl0, wrk, &int1);
        dscal_(&xc, &dblm1, MAT_BUFD(x) + ind, &xr);
        dger_(&m, &xc, &dbl2, MAT_BUFD(vk), &int1, wrk, &int1,
            MAT_BUFD(x) + ind, &ld);
        if (inverse == 'I')
            dscal_(&xc, &dblm1, MAT_BUFD(x) + ind, &xr);

        b = W->beta[k];
        if (inverse == 'I') b = 1.0 / b;
        for (i = 0; i < xc; i++)
            dscal_(&m, &b, MAT_BUFD(x) + ind + i*xr, &int1);
        ind += m;
    }
    free(wrk);


    /*
     * Scaling for 's' component xk is
     *
     *     xk := vec( r' * mat(xk) * r )  if trans = 'N'
     *     xk := vec( r * mat(xk) * r' )  if trans = 'T'.
     *
     * r is kth element of W['r'].
     *
     * Inverse scaling is
     *
     *     xk := vec( rti * mat(xk) * rti' )  if trans = 'N'
     *     xk := vec( rti' * mat(xk) * rti )  if trans = 'T'.
     *
     * rti is kth element of W['rti'].
     */

    N = W->r_count;
    for (k = 0, maxn = 0; k < N; ++k){
        rk = (inverse == 'N') ? W->r[k] : W->rti[k];
        maxn = MAX(maxn, rk->nrows);
    }
    if (!(wrk = (double *) calloc(maxn*maxn, sizeof(double)))) err_no_memory;
    for (k = 0; k < N; k++){
        rk = (inverse == 'N') ? W->r[k] : W->rti[k];
        n = rk->nrows;
        for (i = 0; i < xc; i++){

            /* scale diagonal of rk by 0.5 */
            inc = n + 1;
            dscal_(&n, &dbl5, MAT_BUFD(x) + ind + i*xr, &inc);

            /* wrk = r*tril(x) if inverse is 'N' and trans is 'T' or
             *                 inverse is 'I' and trans is 'N'
             * wrk = tril(x)*r otherwise. */
            len = n*n;
            dcopy_(&len, MAT_BUFD(rk), &int1, wrk, &int1);
            ld = MAX(1, n);
            dtrmm_( (( inverse == 'N' && trans == 'T') || ( inverse == 'I'
                && trans == 'N')) ? "R" : "L", "L", "N", "N", &n, &n,
                &dbl1, MAT_BUFD(x) + ind + i*xr, &ld, wrk, &ld);

            /* x := (r*wrk' + wrk*r') if inverse is 'N' and trans is 'T'
             *                        or inverse is 'I' and trans is 'N'
             * x := (r'*wrk + wrk'*r) otherwise. */
            dsyr2k_("L", ((inverse == 'N' && trans == 'T') ||
                (inverse == 'I' && trans == 'N')) ? "N" : "T", &n, &n,
                &dbl1, MAT_BUFD(rk), &ld, wrk, &ld, &dbl0, MAT_BUFD(x) +
                ind + i*xr, &ld);
        }
        ind += n*n;
    }
    free(wrk);
}

// Simplified BLAS implementations (basic versions for demonstration)
void dtbmv(const char *uplo, const char *trans, const char *diag, 
           int n, int k, const double *a, int lda, double *x, int incx) {
    // For simplicity, implement element-wise multiplication for k=0 case
    if (k == 0) {
        for (int i = 0; i < n; i++) {
            x[i * incx] *= a[i];
        }
    }
}

void dscal(int n, double alpha, double *x, int incx) {
    for (int i = 0; i < n; i++) {
        x[i * incx] *= alpha;
    }
}

void dgemv(const char *trans, int m, int n, double alpha, const double *a, 
           int lda, const double *x, int incx, double beta, double *y, int incy) {
    // Basic implementation for trans='T' case
    if (*trans == 'T') {
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int j = 0; j < m; j++) {
                sum += a[j * lda + i] * x[j * incx];
            }
            y[i * incy] = alpha * sum + beta * y[i * incy];
        }
    }
}

void dger(int m, int n, double alpha, const double *x, int incx, 
          const double *y, int incy, double *a, int lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            a[i * lda + j] += alpha * x[i * incx] * y[j * incy];
        }
    }
}

void dcopy(int n, const double *x, int incx, double *y, int incy) {
    for (int i = 0; i < n; i++) {
        y[i * incy] = x[i * incx];
    }
}

void dtrmm(const char *side, const char *uplo, const char *transa, 
           const char *diag, int m, int n, double alpha, const double *a, 
           int lda, double *b, int ldb) {
    // Simplified implementation - would need full BLAS implementation
    // This is a placeholder for the triangular matrix multiplication
}

void dsyr2k(const char *uplo, const char *trans, int n, int k, double alpha, 
            const double *a, int lda, const double *b, int ldb, double beta, 
            double *c, int ldc) {
    // Simplified implementation - would need full BLAS implementation
    // This is a placeholder for the symmetric rank-2k update
}


/**
 * @brief Multiplies with square root of the Hessian of the logarithmic barrier
 * 
 * misc_scale2(lmbda, x, dims, mnl = 0, inverse = 'N')
 * 
 * @details
 * Computes the product of x with the square root of the Hessian matrix H:
 * - x := H(λ^{1/2}) * x   (inverse = 'N')
 * - x := H(λ^{-1/2}) * x  (inverse = 'I')
 * 
 * where H is the Hessian of the logarithmic barrier function and λ are the
 * scaling parameters.
 *
 * @param[in] lmbda    Scaling parameters vector
 * @param[in,out] x    Input/output vector to be transformed
 * @param[in] dims     Dictionary containing cone dimensions:
 *                     - "l": linear dimension
 *                     - "q": list of quadratic cone dimensions
 *                     - "s": list of semidefinite cone dimensions
 * @param[in] mnl      Number of nonlinear variables (default = 0)
 * @param[in] inverse  Transformation mode ('N' for direct, 'I' for inverse)
 *                     (default = 'N')
 * 
 * @return None (result stored in x)
 *
 * @note
 * - Implements Nesterov-Todd scaling for conic optimization
 * - Handles both direct and inverse square root transformations
 * - Respects the cone structure defined in dims
 * - Uses efficient BLAS operations for the transformations
 *
 * @warning
 * - lmbda and x must have matching cone structure
 * - dims must contain valid dimension entries
 * - inverse must be either 'N' or 'I'
 * - All vectors must be properly allocated
 * - Undefined behavior if dimensions mismatch
 *
 * @see The Theory of Self-Dual Cones and Cone Programming by Nesterov and Todd
 */
void misc_scale2(matrix *lmbda, matrix *x, DIMs *dims, int mnl, char inverse)
{
    // Set default character values
    if (inverse == 0) inverse = 'N';

    double a, lx, x0, b, *c = NULL, *sql = NULL;
    int m = 0, mk, i, j, len, int0 = 0, int1 = 1, maxn = 0, ind2;

    /*
     * For nonlinear and 'l' blocks:
     *
     *     xk := xk ./ l  (invers is 'N')
     *     xk := xk .* l  (invers is 'I')
     *
     * where l is the first mnl + dims['l'] components of lmbda.
     */

    m += dims->l;
    if (inverse == 'N')
        dtbsv_("L", "N", "N", &m, &int0, MAT_BUFD(lmbda), &int1,
             MAT_BUFD(x), &int1);
    else
        dtbmv_("L", "N", "N", &m, &int0, MAT_BUFD(lmbda), &int1,
             MAT_BUFD(x), &int1);

    /*
     * For 'q' blocks, if inverse is 'N',
     *
     *     xk := 1/a * [ l'*J*xk;
     *         xk[1:] - (xk[0] + l'*J*xk) / (l[0] + 1) * l[1:] ].
     *
     *  If inverse is 'I',
     *
     *     xk := a * [ l'*xk;
     *         xk[1:] + (xk[0] + l'*xk) / (l[0] + 1) * l[1:] ].
     *
     * a = sqrt(lambda_k' * J * lambda_k), l = lambda_k / a.
     */

    for (i = 0; i < dims->q_size; i++){
        mk = dims->q[i];
        len = mk - 1;
        a = dnrm2_(&len, MAT_BUFD(lmbda) + m + 1, &int1);
        a = sqrt(MAT_BUFD(lmbda)[m] + a) * sqrt(MAT_BUFD(lmbda)[m] - a);
        if (inverse == 'N')
            lx = ( MAT_BUFD(lmbda)[m] * MAT_BUFD(x)[m] -
                ddot_(&len, MAT_BUFD(lmbda) + m + 1, &int1, MAT_BUFD(x) + m
                    + 1, &int1) ) / a;
        else
            lx = ddot_(&mk, MAT_BUFD(lmbda) + m, &int1, MAT_BUFD(x) + m,
                &int1) / a;
        x0 = MAT_BUFD(x)[m];
        MAT_BUFD(x)[m] = lx;
        b = (x0 + lx) / (MAT_BUFD(lmbda)[m]/a + 1.0) / a;
        if (inverse == 'N')  b *= -1.0;
        daxpy_(&len, &b, MAT_BUFD(lmbda) + m + 1, &int1,
            MAT_BUFD(x) + m + 1, &int1);
        if (inverse == 'N')  a = 1.0 / a;
        dscal_(&mk, &a, MAT_BUFD(x) + m, &int1);
        m += mk;
    }


    /*
     *  For the 's' blocks, if inverse is 'N',
     *
     *      xk := vec( diag(l)^{-1/2} * mat(xk) * diag(k)^{-1/2}).
     *
     *  If inverse is 'I',
     *
     *     xk := vec( diag(l)^{1/2} * mat(xk) * diag(k)^{1/2}).
     *
     * where l is kth block of lambda.
     *
     * We scale upper and lower triangular part of mat(xk) because the
     * inverse operation will be applied to nonsymmetric matrices.
     */

    for (i = 0; i < dims->s_size; i++){
        maxn = MAX(maxn, dims->s[i]);
    }
    if (!(c = (double *) calloc(maxn, sizeof(double))) ||
        !(sql = (double *) calloc(maxn, sizeof(double)))){
        free(c); free(sql);
        err_no_memory;
    }
    ind2 = m;
    for (i = 0; i < dims->s_size; i++){
        mk = dims->s[i];
        for (j = 0; j < mk; j++)
            sql[j] = sqrt(MAT_BUFD(lmbda)[ind2 + j]);
        for (j = 0; j < mk; j++){
            dcopy_(&mk, sql, &int1, c, &int1);
            b = sqrt(MAT_BUFD(lmbda)[ind2 + j]);
            dscal_(&mk, &b, c, &int1);
            if (inverse == 'N')
                dtbsv_("L", "N", "N", &mk, &int0, c, &int1, MAT_BUFD(x) +
                    m + j*mk, &int1);
            else
                dtbmv_("L", "N", "N", &mk, &int0, c, &int1, MAT_BUFD(x) +
                    m + j*mk, &int1);
        }
        m += mk*mk;
        ind2 += mk;
    }
    free(c); free(sql);
}


// static char doc_pack[] =
//     "Copy x to y using packed storage.\n\n"
//     "pack(x, y, dims, mnl = 0, offsetx = 0, offsety = 0)\n\n"
//     "The vector x is an element of S, with the 's' components stored in\n"
//     "unpacked storage.  On return, x is copied to y with the 's' \n"
//     "components stored in packed storage and the off-diagonal entries \n"
//     "scaled by sqrt(2).";

// static PyObject* pack(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     matrix *x, *y;
//     PyObject *O, *Ok, *dims;
//     double a;
//     int i, k, nlq = 0, ox = 0, oy = 0, np, iu, ip, int1 = 1, len, n;
//     char *kwlist[] = {"x", "y", "dims", "mnl", "offsetx", "offsety", NULL};

//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|iii", kwlist, &x,
//         &y, &dims, &nlq, &ox, &oy)) return NULL;

//     O = PyDict_GetItemString(dims, "l");
// #if PY_MAJOR_VERSION >= 3
//     nlq += (int) PyLong_AsLong(O);
// #else
//     nlq += (int) PyInt_AsLong(O);
// #endif

//     O = PyDict_GetItemString(dims, "q");
//     for (i = 0; i < (int) PyList_Size(O); i++){
//         Ok = PyList_GetItem(O, (Py_ssize_t) i);
// #if PY_MAJOR_VERSION >= 3
//         nlq += (int) PyLong_AsLong(Ok);
// #else
//         nlq += (int) PyInt_AsLong(Ok);
// #endif
//     }
//     dcopy_(&nlq, MAT_BUFD(x) + ox, &int1, MAT_BUFD(y) + oy, &int1);

//     O = PyDict_GetItemString(dims, "s");
//     for (i = 0, np = 0, iu = ox + nlq, ip = oy + nlq; i < (int)
//         PyList_Size(O); i++){
//         Ok = PyList_GetItem(O, (Py_ssize_t) i);
// #if PY_MAJOR_VERSION >= 3
//         n = (int) PyLong_AsLong(Ok);
// #else
//         n = (int) PyInt_AsLong(Ok);
// #endif
//         for (k = 0; k < n; k++){
//             len = n-k;
//             dcopy_(&len, MAT_BUFD(x) + iu + k*(n+1), &int1,  MAT_BUFD(y) +
//                 ip, &int1);
//             MAT_BUFD(y)[ip] /= sqrt(2.0);
//             ip += len;
//         }
//         np += n*(n+1)/2;
//         iu += n*n;
//     }

//     a = sqrt(2.0);
//     dscal_(&np, &a, MAT_BUFD(y) + oy + nlq, &int1);

//     return Py_BuildValue("");
// }


// static char doc_pack2[] =
//     "In-place version of pack().\n\n"
//     "pack2(x, dims, mnl = 0)\n\n"
//     "In-place version of pack(), which also accepts matrix arguments x.\n"
//     "The columns of x are elements of S, with the 's' components stored\n"
//     "in unpacked storage.  On return, the 's' components are stored in\n"
//     "packed storage and the off-diagonal entries are scaled by sqrt(2).";

// static PyObject* pack2(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     matrix *x;
//     PyObject *O, *Ok, *dims;
//     double a = sqrt(2.0), *wrk;
//     int i, j, k, nlq = 0, iu, ip, len, n, maxn, xr, xc;
//     char *kwlist[] = {"x", "dims", "mnl", NULL};

//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|i", kwlist, &x,
//         &dims, &nlq)) return NULL;

//     xr = x->nrows;
//     xc = x->ncols;

//     O = PyDict_GetItemString(dims, "l");
// #if PY_MAJOR_VERSION >= 3
//     nlq += (int) PyLong_AsLong(O);
// #else
//     nlq += (int) PyInt_AsLong(O);
// #endif

//     O = PyDict_GetItemString(dims, "q");
//     for (i = 0; i < (int) PyList_Size(O); i++){
//         Ok = PyList_GetItem(O, (Py_ssize_t) i);
// #if PY_MAJOR_VERSION >= 3
//         nlq += (int) PyLong_AsLong(Ok);
// #else
//         nlq += (int) PyInt_AsLong(Ok);
// #endif
//     }

//     O = PyDict_GetItemString(dims, "s");
//     for (i = 0, maxn = 0; i < (int) PyList_Size(O); i++){
//         Ok = PyList_GetItem(O, (Py_ssize_t) i);
// #if PY_MAJOR_VERSION >= 3
//         maxn = MAX(maxn, (int) PyLong_AsLong(Ok));
// #else
//         maxn = MAX(maxn, (int) PyInt_AsLong(Ok));
// #endif
//     }
//     if (!maxn) return Py_BuildValue("");
//     if (!(wrk = (double *) calloc(maxn * xc, sizeof(double))))
//         return PyErr_NoMemory();

//     for (i = 0, iu = nlq, ip = nlq; i < (int) PyList_Size(O); i++){
//         Ok = PyList_GetItem(O, (Py_ssize_t) i);
// #if PY_MAJOR_VERSION >= 3
//         n = (int) PyLong_AsLong(Ok);
// #else
//         n = (int) PyInt_AsLong(Ok);
// #endif
//         for (k = 0; k < n; k++){
//             len = n-k;
//             dlacpy_(" ", &len, &xc, MAT_BUFD(x) + iu + k*(n+1), &xr, wrk, 
//                 &maxn);
//             for (j = 1; j < len; j++)
//                 dscal_(&xc, &a, wrk + j, &maxn);
//             dlacpy_(" ", &len, &xc, wrk, &maxn, MAT_BUFD(x) + ip, &xr);
//             ip += len;
//         }
//         iu += n*n;
//     }

//     free(wrk);
//     return Py_BuildValue("");
// }


// static char doc_unpack[] =
//     "Unpacks x into y.\n\n"
//     "unpack(x, y, dims, mnl = 0, offsetx = 0, offsety = 0)\n\n"
//     "The vector x is an element of S, with the 's' components stored in\n"
//     "unpacked storage and off-diagonal entries scaled by sqrt(2).\n"
//     "On return, x is copied to y with the 's' components stored in\n"
//     "unpacked storage.";

// static PyObject* unpack(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     matrix *x, *y;
//     PyObject *O, *Ok, *dims;
//     double a = 1.0 / sqrt(2.0);
//     int m = 0, ox = 0, oy = 0, int1 = 1, iu, ip, len, i, k, n;
//     char *kwlist[] = {"x", "y", "dims", "mnl", "offsetx", "offsety", NULL};

//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|iii", kwlist, &x,
//         &y, &dims, &m, &ox, &oy)) return NULL;

//     O = PyDict_GetItemString(dims, "l");
// #if PY_MAJOR_VERSION >= 3
//     m += (int) PyLong_AsLong(O);
// #else
//     m += (int) PyInt_AsLong(O);
// #endif

//     O = PyDict_GetItemString(dims, "q");
//     for (i = 0; i < (int) PyList_Size(O); i++){
//         Ok = PyList_GetItem(O, (Py_ssize_t) i);
// #if PY_MAJOR_VERSION >= 3
//         m += (int) PyLong_AsLong(Ok);
// #else
//         m += (int) PyInt_AsLong(Ok);
// #endif
//     }
//     dcopy_(&m, MAT_BUFD(x) + ox, &int1, MAT_BUFD(y) + oy, &int1);

//     O = PyDict_GetItemString(dims, "s");
//     for (i = 0, ip = ox + m, iu = oy + m; i < (int) PyList_Size(O); i++){
//         Ok = PyList_GetItem(O, (Py_ssize_t) i);
// #if PY_MAJOR_VERSION >= 3
//         n = (int) PyLong_AsLong(Ok);
// #else
//         n = (int) PyInt_AsLong(Ok);
// #endif
//         for (k = 0; k < n; k++){
//             len = n-k;
//             dcopy_(&len, MAT_BUFD(x) + ip, &int1, MAT_BUFD(y) + iu +
//                 k*(n+1), &int1);
//             ip += len;
//             len -= 1;
//             dscal_(&len, &a, MAT_BUFD(y) + iu + k*(n+1) + 1, &int1);
//         }
//         iu += n*n;
//     }

//     return Py_BuildValue("");
// }


// static char doc_symm[] =
//     "Converts lower triangular matrix to symmetric.\n\n"
//     "symm(x, n, offset = 0)\n\n"
//     "Fills in the upper triangular part of the symmetric matrix stored\n"
//     "in x[offset : offset+n*n] using 'L' storage.";

// static PyObject* symm(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     matrix *x;
//     int n, ox = 0, k, len, int1 = 1;
//     char *kwlist[] = {"x", "n", "offset", NULL};

//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "Oi|i", kwlist, &x, &n,
//         &ox)) return NULL;

//     if (n > 1) for (k = 0; k < n; k++){
//         len = n-k-1;
//         dcopy_(&len, MAT_BUFD(x) + ox + k*(n+1) + 1, &int1, MAT_BUFD(x) +
//             ox + (k+1)*(n+1)-1, &n);
//     }
//     return Py_BuildValue("");
// }


/**
 * @brief Computes the symmetric product x := (y o x) in cone space
 * 
 * misc_sprod(x, y, dims, mnl = 0, diag = 'N')
 * 
 * @details
 * Computes the symmetric product operation between vectors x and y that respects
 * the cone structure defined by dims. The operation handles three cases:
 * 
 * 1. Nonlinear and linear blocks:
 *    yk o xk = yk .* xk (element-wise product)
 * 
 * 2. Quadratic cone blocks:
 *              [ l0   l1'  ]
 *    yk o xk = [           ] * xk
 *              [ l1   l0*I ]
 *    where yk = (l0, l1)
 * 
 * 3. Semidefinite cone blocks:
 *    yk o xk = 0.5 * (Yk*mat(xk) + mat(xk)*Yk)
 *    where Yk = mat(yk) if diag='N' or Yk = diag(yk) if diag='D'
 *
 * @param[in,out] x    Input/output vector in cone space
 * @param[in] y        Input vector in cone space
 * @param[in] dims     Dictionary containing cone dimensions:
 *                     - "l": linear dimension
 *                     - "q": list of quadratic cone dimensions
 *                     - "s": list of semidefinite cone dimensions
 * @param[in] mnl      Number of nonlinear variables (default = 0)
 * @param[in] diag     Diagonal flag ('N' for full, 'D' for diagonal storage) 
 *                     (default = 'N')
 *
 * @return None (result stored in x)
 *
 * @note
 * - Uses BLAS/LAPACK routines (dtbmv, ddot, dscal, daxpy, dsyr2k) for efficiency
 * - For 's' blocks with diag='D', only diagonal elements are stored/processed
 * - Handles Python 2/3 compatibility internally
 * - Allocates temporary workspace for semidefinite blocks
 *
 * @warning
 * - x and y must have matching cone structure
 * - dims must contain valid dimension entries
 * - diag must be either 'N' or 'D'
 * - Caller must ensure x has sufficient storage for result
 * - Memory allocation failures will raise Python memory error
 */

void misc_sprod(matrix *x, matrix *y, DIMs *dims, int mnl, char diag)
{
    int i, j, k, mk, length, maxn, ind = 0, ind2, int0 = 0, int1 = 1, ld;
    double a, *A = NULL, dbl2 = 0.5, dbl0 = 0.0;

    ind = mnl;
    if(diag == 0) diag = 'N';

    /*
     * For nonlinear and 'l' blocks:
     *
     *     yk o xk = yk .* xk
     */

    ind += dims->l;
    dtbmv_("L", "N", "N", &ind, &int0, MAT_BUFD(y), &int1, MAT_BUFD(x),
        &int1);

    /*
     * For 'q' blocks:
     *
     *                [ l0   l1'  ]
     *     yk o xk =  [           ] * xk
     *                [ l1   l0*I ]
     *
     * where yk = (l0, l1).
     */

    for (i = 0; i < dims->q_size; ++i){
        mk = dims->q[i];
        a = ddot_(&mk, MAT_BUFD(y) + ind, &int1, MAT_BUFD(x) + ind, &int1);
        length = mk - 1;
        dscal_(&length, MAT_BUFD(y) + ind, MAT_BUFD(x) + ind + 1, &int1);
        daxpy_(&length, MAT_BUFD(x) + ind, MAT_BUFD(y) + ind + 1, &int1,
            MAT_BUFD(x) + ind + 1, &int1);
        MAT_BUFD(x)[ind] = a;
        ind += mk;
    }


    /*
     * For the 's' blocks:
     *
     *    yk o sk = .5 * ( Yk * mat(xk) + mat(xk) * Yk )
     *
     * where Yk = mat(yk) if diag is 'N' and Yk = diag(yk) if diag is 'D'.
     */

    for (i = 0, maxn = 0; i < dims->s_size; ++i)
        maxn = MAX(maxn, dims->s[i]);
    
    if (diag == 'N'){
        if (!(A = (double *) calloc(maxn * maxn, sizeof(double)))) err_no_memory;
        for (i = 0; i < dims->s_size; ind += mk*mk, ++i){
            mk = dims->s_size;
            length = mk * mk;
            dcopy_(&length, MAT_BUFD(x) + ind, &int1, A, &int1);

            if (mk > 1) for (k = 0; k < mk; k++){
                length = mk - k - 1;
                dcopy_(&length, A + k*(mk+1) + 1, &int1, A + (k+1)*(mk+1)-1,
                    &mk);
                dcopy_(&length, MAT_BUFD(y) + ind + k*(mk+1) + 1, &int1,
                    MAT_BUFD(y) + ind + (k+1)*(mk+1)-1, &mk);
            }

            ld = MAX(1, mk);
            dsyr2k_("L", "N", &mk, &mk, &dbl2, A, &ld, MAT_BUFD(y) + ind,
                &ld, &dbl0, MAT_BUFD(x) + ind, &ld);
        }
    }
    else {
        if (!(A = (double *) calloc(maxn, sizeof(double)))) err_no_memory;
        for (i = 0, ind2 = ind; i < dims->s_size; ind += mk*mk,
            ind2 += mk, i++){
            mk = dims->s[i];
            for (k = 0; k < mk; k++){
                length = mk - k;
                dcopy_(&length, MAT_BUFD(y) + ind2 + k, &int1, A, &int1);
                for (j = 0; j < length; j++) A[j] += MAT_BUFD(y)[ind2 + k];
                dscal_(&length, &dbl2, A, &int1);
                dtbmv_("L", "N", "N", &length, &int0, A, &int1, MAT_BUFD(x) +
                    ind + k * (mk+1), &int1);
            }
        }
    }

    free(A);
}


// static char doc_sinv[] =
//     "The inverse of the product x := (y o x) when the 's' components of \n"
//     "y are diagonal.\n\n"
//     "sinv(x, y, dims, mnl = 0)";

// static PyObject* sinv(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     matrix *x, *y;
//     PyObject *dims, *O, *Ok;
//     int i, j, k, mk, len, maxn, ind = 0, ind2, int0 = 0, int1 = 1;
//     double a, c, d, alpha, *A = NULL, dbl2 = 0.5;
//     char *kwlist[] = {"x", "y", "dims", "mnl", NULL};

//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|i", kwlist, &x, &y,
//         &dims, &ind)) return NULL;


//     /*
//      * For nonlinear and 'l' blocks:
//      *
//      *     yk o\ xk = yk .\ xk
//      */

//     O = PyDict_GetItemString(dims, "l");
// #if PY_MAJOR_VERSION >= 3
//     ind += (int) PyLong_AsLong(O);
// #else
//     ind += (int) PyInt_AsLong(O);
// #endif
//     dtbsv_("L", "N", "N", &ind, &int0, MAT_BUFD(y), &int1, MAT_BUFD(x),
//         &int1);


//     /*
//      * For 'q' blocks:
//      *
//      *                        [  l0   -l1'               ]
//      *     yk o\ xk = 1/a^2 * [                          ] * xk
//      *                        [ -l1    (a*I + l1*l1')/l0 ]
//      *
//      * where yk = (l0, l1) and a = l0^2 - l1'*l1.
//      */

//     O = PyDict_GetItemString(dims, "q");
//     for (i = 0; i < (int) PyList_Size(O); i++){
//         Ok = PyList_GetItem(O, (Py_ssize_t) i);
// #if PY_MAJOR_VERSION >= 3
//         mk = (int) PyLong_AsLong(Ok);
// #else
//         mk = (int) PyInt_AsLong(Ok);
// #endif
//         len = mk - 1;
//         a = dnrm2_(&len, MAT_BUFD(y) + ind + 1, &int1);
//         a = (MAT_BUFD(y)[ind] + a) * (MAT_BUFD(y)[ind] - a);
//         c = MAT_BUFD(x)[ind];
//         d = ddot_(&len, MAT_BUFD(x) + ind + 1, &int1,
//             MAT_BUFD(y) + ind + 1, &int1);
//         MAT_BUFD(x)[ind] = c * MAT_BUFD(y)[ind] - d;
//         alpha = a / MAT_BUFD(y)[ind];
//         dscal_(&len, &alpha, MAT_BUFD(x) + ind + 1, &int1);
//         alpha = d / MAT_BUFD(y)[ind] - c;
//         daxpy_(&len, &alpha, MAT_BUFD(y) + ind + 1, &int1, MAT_BUFD(x) +
//             ind + 1, &int1);
//         alpha = 1.0 / a;
//         dscal_(&mk, &alpha, MAT_BUFD(x) + ind, &int1);
//         ind += mk;
//     }


//     /*
//      * For the 's' blocks:
//      *
//      *    yk o\ sk = xk ./ gamma
//      *
//      * where  gammaij = .5 * (yk_i + yk_j).
//      */

//     O = PyDict_GetItemString(dims, "s");
//     for (i = 0, maxn = 0; i < (int) PyList_Size(O); i++){
//         Ok = PyList_GetItem(O, (Py_ssize_t) i);
// #if PY_MAJOR_VERSION >= 3
//         maxn = MAX(maxn, (int) PyLong_AsLong(Ok));
// #else
//         maxn = MAX(maxn, (int) PyInt_AsLong(Ok));
// #endif
//     }
//     if (!(A = (double *) calloc(maxn, sizeof(double))))
//         return PyErr_NoMemory();
//     for (i = 0, ind2 = ind; i < (int) PyList_Size(O); ind += mk*mk,
//         ind2 += mk, i++){
//         Ok = PyList_GetItem(O, (Py_ssize_t) i);
// #if PY_MAJOR_VERSION >= 3
//         mk = (int) PyLong_AsLong(Ok);
// #else
//         mk = (int) PyInt_AsLong(Ok);
// #endif
//         for (k = 0; k < mk; k++){
//             len = mk - k;
//             dcopy_(&len, MAT_BUFD(y) + ind2 + k, &int1, A, &int1);
//             for (j = 0; j < len; j++) A[j] += MAT_BUFD(y)[ind2 + k];
//             dscal_(&len, &dbl2, A, &int1);
//             dtbsv_("L", "N", "N", &len, &int0, A, &int1, MAT_BUFD(x) + ind
//                 + k * (mk+1), &int1);
//         }
//     }

//     free(A);
//     return Py_BuildValue("");
// }



// static char doc_trisc[] =
//     "Sets the upper triangular part of the 's' components of x equal to\n"
//     "zero and scales the strictly lower triangular part\n\n"
//     "trisc(x, dims, offset = 0)";

// static PyObject* trisc(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     matrix *x;
//     double dbl0 = 0.0, dbl2 = 2.0;
//     int ox = 0, i, k, nk, len, int1 = 1;
//     PyObject *dims, *O, *Ok;
//     char *kwlist[] = {"x", "dims", "offset", NULL};

//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|i", kwlist, &x,
//         &dims, &ox)) return NULL;

//     O = PyDict_GetItemString(dims, "l");
// #if PY_MAJOR_VERSION >= 3
//     ox += (int) PyLong_AsLong(O);
// #else
//     ox += (int) PyInt_AsLong(O);
// #endif

//     O = PyDict_GetItemString(dims, "q");
//     for (i = 0; i < (int) PyList_Size(O); i++){
//         Ok = PyList_GetItem(O, (Py_ssize_t) i);
// #if PY_MAJOR_VERSION >= 3
//         ox += (int) PyLong_AsLong(Ok);
// #else
//         ox += (int) PyInt_AsLong(Ok);
// #endif
//     }

//     O = PyDict_GetItemString(dims, "s");
//     for (k = 0; k < (int) PyList_Size(O); k++){
//         Ok = PyList_GetItem(O, (Py_ssize_t) k);
// #if PY_MAJOR_VERSION >= 3
//         nk = (int) PyLong_AsLong(Ok);
// #else
//         nk = (int) PyInt_AsLong(Ok);
// #endif
//         for (i = 1; i < nk; i++){
//             len = nk - i;
//             dscal_(&len, &dbl0, MAT_BUFD(x) + ox + i*(nk+1) - 1, &nk);
//             dscal_(&len, &dbl2, MAT_BUFD(x) + ox + nk*(i-1) + i, &int1);
//         }
//         ox += nk*nk;
//     }

//     return Py_BuildValue("");
// }


// static char doc_triusc[] =
//     "Scales the strictly lower triangular part of the 's' components of\n"
//     "x by 0.5.\n\n"
//     "triusc(x, dims, offset = 0)";

// static PyObject* triusc(PyObject *self, PyObject *args, PyObject *kwrds)
// {
//     matrix *x;
//     double dbl5 = 0.5;
//     int ox = 0, i, k, nk, len, int1 = 1;
//     PyObject *dims, *O, *Ok;
//     char *kwlist[] = {"x", "dims", "offset", NULL};

//     if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|i", kwlist, &x,
//         &dims, &ox)) return NULL;

//     O = PyDict_GetItemString(dims, "l");
// #if PY_MAJOR_VERSION >= 3
//     ox += (int) PyLong_AsLong(O);
// #else
//     ox += (int) PyInt_AsLong(O);
// #endif

//     O = PyDict_GetItemString(dims, "q");
//     for (i = 0; i < (int) PyList_Size(O); i++){
//         Ok = PyList_GetItem(O, (Py_ssize_t) i);
// #if PY_MAJOR_VERSION >= 3
//         ox += (int) PyLong_AsLong(Ok);
// #else
//         ox += (int) PyInt_AsLong(Ok);
// #endif
//     }

//     O = PyDict_GetItemString(dims, "s");
//     for (k = 0; k < (int) PyList_Size(O); k++){
//         Ok = PyList_GetItem(O, (Py_ssize_t) k);
// #if PY_MAJOR_VERSION >= 3
//         nk = (int) PyLong_AsLong(Ok);
// #else
//         nk = (int) PyInt_AsLong(Ok);
// #endif
//         for (i = 1; i < nk; i++){
//             len = nk - i;
//             dscal_(&len, &dbl5, MAT_BUFD(x) + ox + nk*(i-1) + i, &int1);
//         }
//         ox += nk*nk;
//     }

//     return Py_BuildValue("");
// }

/**
 * @brief Computes the inner product of two vectors in the composite cone space S
 * 
 * @details
 * Calculates the specialized inner product between vectors x and y that respects
 * the cone structure defined by dims. The computation handles:
 * - Linear components (including nonlinear variables mnl)
 * - Quadratic cone components
 * - Semidefinite cone components (with proper scaling of off-diagonal elements)
 *
 * For semidefinite blocks, the inner product is computed as:
 * ⟨X,Y⟩ = Σ_{i,j} X_{i,j}Y_{i,j} with off-diagonal terms multiplied by 2
 *
 * @param[in] x     First input vector in cone space
 * @param[in] y     Second input vector in cone space (must match x structure)
 * @param[in] dims  Cone dimensions structure:
 *                  - l: linear dimension
 *                  - q: array of quadratic cone dimensions
 *                  - q_size: number of quadratic cones
 *                  - s: array of semidefinite cone dimensions
 *                  - s_size: number of semidefinite cones
 * @param[in] mnl   Number of nonlinear variables (default = 0)
 *
 * @return The computed inner product ⟨x,y⟩_S
 *
 * @note
 * - Uses BLAS ddot for efficient inner product computations
 * - For semidefinite blocks, off-diagonal terms are doubled
 * - Handles all three cone types (linear, quadratic, semidefinite)
 * - Vectors must have matching structure
 *
 * @warning
 * - x and y must be properly formatted for the cone dimensions
 * - dims must contain valid dimension information
 * - Undefined behavior if vectors have mismatched structure
 * - All cone components must be stored contiguously
 *
 * @see BLAS ddot documentation for inner product implementation
 */
double misc_sdot(matrix *x, matrix *y, DIMs *dims, int mnl)
{
    /*  Inner product of two vectors in S
        sdot(x, y, dims, mnl=0);
    */
    int m = mnl, int1 = 1, i, k, nk, inc, len;
    double a;

    // Add dims->l to m
    m += dims->l;

    // Add sum of dims->q to m
    for (i = 0; i < dims->q_size; i++) {
        m += dims->q[i];
    }

    // Compute initial dot product
    a = ddot_(&m, MAT_BUFD(x), &int1, MAT_BUFD(y), &int1);

    // Process dims->s array
    for (k = 0; k < dims->s_size; k++) {
        nk = dims->s[k];
        inc = nk + 1;
        
        // Add dot product for diagonal elements
        a += ddot_(&nk, MAT_BUFD(x) + m, &inc, MAT_BUFD(y) + m, &inc);
        
        // Add dot products for off-diagonal elements (multiplied by 2)
        for (i = 1; i < nk; i++) {
            len = nk - i;
            a += 2.0 * ddot_(&len, MAT_BUFD(x) + m + i, &inc,
                MAT_BUFD(y) + m + i, &inc);
        }
        
        // Update offset for next matrix block
        m += nk * nk;
    }

    return a;
}



/**
 * @brief Computes the maximum feasible step size for x to remain in the cone
 * 
 * @details
 * Calculates the minimum value t such that x + t*e remains in the composite cone,
 * where e is defined differently for each cone type:
 * - Nonlinear and linear ('l') blocks: vector of ones
 * - Quadratic ('q') cones: first unit vector
 * - Semidefinite ('s') cones: identity matrix
 *
 * Optionally computes eigenvalues/eigenvectors for semidefinite blocks when
 * sigma is provided.
 *
 * @param[in] x      Input matrix in cone space
 * @param[in] dims   Cone dimensions structure:
 *                   - l: linear dimension
 *                   - q: array of quadratic cone dimensions
 *                   - q_size: number of quadratic cones
 *                   - s: array of semidefinite cone dimensions
 *                   - s_size: number of semidefinite cones
 * @param[in] mnl    Number of nonlinear variables (default = 0)
 * @param[in] sigma  Optional matrix for eigenvalues (for 's' blocks)
 *
 * @return Maximum feasible step size t
 *
 * @note
 * - Uses LAPACK routines (dsyevd/dsyevr) for eigenvalue computations
 * - Handles three cone types separately with appropriate definitions of e
 * - For 'q' cones, uses second-order cone structure
 * - For 's' cones, computes minimal eigenvalue when sigma=NULL
 *
 * @warning
 * - x must be properly formatted for the cone dimensions
 * - dims must contain valid dimension information
 * - When sigma is provided, it must have sufficient storage for eigenvalues
 * - Memory allocation failures will trigger error handling
 *
 * @see LAPACK dsyevd/dsyevr documentation for eigenvalue computation details
 */
static double misc_max_step(matrix* x, DIMs* dims, int mnl, matrix* sigma)
{
    int i, mk, len, maxn, ind = 0, ind2, int1 = 1, ld, Ns = 0, info, lwork,
        *iwork = NULL, liwork, iwl, m;
    
    double t = -FLT_MAX, dbl0 = 0.0, *work = NULL, wl, *Q = NULL, *w = NULL;

    // 1. Handle nonlinear and 'l' part: vector of ones
    ind += mnl + dims->l;
    for (i = 0; i < ind; i++) t = MAX(t, -MAT_BUFD(x)[i]);

    // 2. Handle 'q' second-order cone part
    for (i = 0; i < dims->q_size; ++i) {
        mk = dims->q[i];
        len = mk - 1;
        t = MAX(t, dnrm2_(&len, MAT_BUFD(x) + ind + 1, &int1) -
            MAT_BUFD(x)[ind]);
        ind += mk;
    }

    // 3. Handle 's' semidefinite cone part
    // Find maximum dimension among 's' blocks for memory allocation
    for (i = 0, maxn = 0; i < dims->s_size; ++i) {
        maxn = MAX(maxn, dims->s[i]);
    }
    
    // If no 's' blocks or all have dimension 0, return current t
    if (!maxn) return (ind) ? t : 0.0;
    
    // Determine workspace size for LAPACK routines
    lwork = -1;
    liwork = -1;
    ld = MAX(1, maxn);
    if (sigma){
        dsyevd_("V", "L", &maxn, NULL, &ld, NULL, &wl, &lwork, &iwl,
            &liwork, &info);
    }
    else {
        if (!(Q = (double *) calloc(maxn * maxn, sizeof(double))) ||
            !(w = (double *) calloc(maxn, sizeof(double)))){
            free(Q); free(w);
            err_no_memory;
        }
        dsyevr_("N", "I", "L", &maxn, NULL, &ld, &dbl0, &dbl0, &int1,
            &int1, &dbl0, &maxn, NULL, NULL, &int1, NULL, &wl, &lwork,
            &iwl, &liwork, &info);
    }

    // Allocate workspace
    lwork = (int) wl;
    liwork = iwl;
    if (!(work = (double *) calloc(lwork, sizeof(double))) ||
        (!(iwork = (int *) calloc(liwork, sizeof(int))))){
        free(Q);  free(w);  free(work); free(iwork);
        err_no_memory;
    }

    // Process each 's' block
    for (i = 0, ind2 = 0; i < dims->s_size; ++i){
        mk = dims->s[i];
        if (mk){
            if (sigma){
                dsyevd_("V", "L", &mk, MAT_BUFD(x) + ind, &mk,
                    MAT_BUFD(sigma) + ind2, work, &lwork, iwork, &liwork,
                    &info);
                t = MAX(t, -MAT_BUFD(sigma)[ind2]);
            }
            else {
                len = mk*mk;
                dcopy_(&len, MAT_BUFD(x) + ind, &int1, Q, &int1);
                ld = MAX(1, mk);
                dsyevr_("N", "I", "L", &mk, Q, &mk, &dbl0, &dbl0, &int1,
                    &int1, &dbl0, &m, w, NULL, &int1, NULL, work, &lwork,
                    iwork, &liwork, &info);
                t = MAX(t, -w[0]);
            }
        }
        ind += mk*mk;
        ind2 += mk;
    }

    // Free allocated memory
    free(work);  free(iwork);  free(Q);  free(w);

    return (ind) ? t : 0.0;
}

/**
 * @brief Computes the S-norm (specialized norm) of a structured matrix
 * 
 * @details
 * Calculates the S-norm of matrix x, defined as the square root of the
 * S-dot product of x with itself. The S-norm is a specialized norm that
 * respects the matrix structure defined by dims:
 * - Handles linear, quadratic cone, and semidefinite cone components
 * - Properly accounts for non-negative variables (mnl)
 *
 * @param[in] x     Input matrix (structured according to dims)
 * @param[in] dims  Structure containing matrix dimensions:
 *                  - l: linear dimension
 *                  - q: array of quadratic cone dimensions
 *                  - q_size: number of quadratic cones
 *                  - s: array of semidefinite cone dimensions
 *                  - s_size: number of semidefinite cones
 * @param[in] mnl   Number of non-negative variables (default = 0)
 *
 * @return The S-norm of the matrix (sqrt of S-dot product with itself)
 */
double misc_snrm2(matrix *x, DIMs *dims, int mnl)
{
    // Compute the S-dot product of x with itself
    double inner_product = misc_sdot(x, x, dims, mnl);

    // Return the square root of the dot product (i.e., the norm)
    return sqrt(inner_product);
}


/**
 * @brief Computes the element-wise square product x := y ∘ y for a structured matrix
 * 
 * @details
 * Computes the Hadamard (element-wise) product of y with itself, storing the result in x.
 * Handles special structure where:
 * - The 's' components of y are diagonal
 * - Only the diagonals of x and y are stored
 * - Processes three types of matrix blocks:
 *   1. Linear blocks (mnl + dims->l)
 *   2. Quadratic cones (dims->q)
 *   3. Semidefinite cones (dims->s)
 *
 * @param[out] x    Output matrix (stores result y ∘ y)
 * @param[in] y     Input matrix (source values)
 * @param[in] dims  Structure containing matrix dimensions:
 *                  - l: linear dimension
 *                  - q: array of quadratic cone dimensions
 *                  - q_size: number of quadratic cones
 *                  - s: array of semidefinite cone dimensions
 *                  - s_size: number of semidefinite cones
 * @param[in] mnl   Number of non-negative variables (default = 0)
 *
 * @note
 * - Uses BLAS operations for efficient computation:
 *   1. blas_copy for initial copy operation
 *   2. blas_tbmv for triangular band matrix operations
 *   3. blas_nrm2 for Euclidean norm calculations
 *   4. blas_scal for scaling operations
 * - For quadratic cones, computes squared norms and scales appropriately
 * - For semidefinite cones, processes only diagonal elements
 * - All operations preserve the structured format of the matrices
 *
 * @warning
 * - Matrices x and y must be properly allocated and compatible in size
 * - dims structure must contain valid dimension information
 * - All diagonal elements must be stored contiguously
 * - For quadratic cones, assumes standard second-order cone structure
 */
void misc_ssqr(matrix *x, matrix *y, DIMs *dims, int mnl) {
    
    int ind = mnl + dims->l;

    // blas.copy(y, x)
    blas_copy(y, x, len(y), 1, 1, 0, 0);
    
    // blas.tbmv(y, x, n = mnl + dims['l'], k = 0, ldA = 1)
    blas_tbmv(y, x, 'L', 'N', 'N', ind, 0, 1, 1, 0, 0);
    
    // for m in dims['q']:
    for (int i = 0; i < dims->q_size; ++i) {
        int m = dims->q[i];
        
        // x[ind] = blas.nrm2(y, offset = ind, n = m)**2
        double nrm = blas_nrm2(y, m, 1, ind);
        ((double *)(x->buffer))[ind] = nrm * nrm;
        
        // blas.scal(2.0*y[ind], x, n = m-1, offset = ind+1)
        // blas_scal(m-1, 2.0 * y[ind], &x[ind+1], 1);
        double y_val = ((double *)(y->buffer))[ind] * 2.0;
        blas_scal(&y_val, x, m - 1, 1, ind + 1);
        ind += m;
    }
    
    // blas.tbmv(y, x, n = sum(dims['s']), k = 0, ldA = 1, offsetA = ind, offsetx = ind)
    int s_sum = sum_array(dims->s, dims->s_size);
    blas_tbmv(y, x, 0, 0, 0, s_sum, 0, 1, 1, ind, ind);
}


/**
 * Converts lower triangular matrix to symmetric.
 * 
 * Fills in the upper triangular part of the symmetric matrix stored
 * in x[offset : offset+n*n] using 'L' storage.
 * 
 * @param x        Pointer to the matrix data (double array)
 * @param n        Size of the matrix (n x n)
 * @param offset   Starting offset in the array (default 0)
 */

void misc_symm(matrix *x, int n, int offset)
{
    int ox = offset;
    if (ox < 0) ox = 0;

    int k, nn, int1 = 1;
    if (n > 1) for (k = 0; k < n; k++){
        nn = n-k-1;
        dcopy_(&nn, MAT_BUFD(x) + ox + k*(n+1) + 1, &int1, MAT_BUFD(x) +
            ox + (k+1)*(n+1)-1, &n);
    }
}


