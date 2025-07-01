#define BASE_MODULE

#include "cvxopt.h"
#include "misc.h"
#include "math.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


/***********************************************************************/
/****                                                               ****/
/****                      BLAS 1 prototypes                        ****/
/****                                                               ****/
/***********************************************************************/

/*************   sp_gemm   *************/
extern int (*sp_axpy[])(number, void *, void *, int, int, int, void **) ;

extern int (*sp_gemm[])(char, char, number, void *, void *, number, void *,
    int, int, int, int, void **, int, int, int);

extern int (*sp_gemv[])(char, int, int, number, void *, int, void *, int,
    number, void *, int) ;

/*************   axpy   *************/
extern void daxpy_(int *, void *, void *, int *, void *, int *) ;
extern void zaxpy_(int *, void *, void *, int *, void *, int *) ;
static void iaxpy_(int *n, void *a, void *x, int *incx, void *y, int *incy) {
  int i;
  for (i=0; i < *n; i++) {
    ((int_t *)y)[i*(*incy)] += *((int_t *)a)*((int_t *)x)[i*(*incx)];
  }
}
void (*axpy_[])(int *, void *, void *, int *, void *, int *) = { iaxpy_, daxpy_, zaxpy_ };

/*************   scal   *************/
extern void dscal_(int *, void *, void *, int *) ;
extern void zscal_(int *, void *, void *, int *) ;
static void iscal_(int *n, void *a, void *x, int *incx) {
  int i;
  for (i=0; i < *n; i++) {
    ((int_t *)x)[i*(*incx)] *= *((int_t *)a);
  }
}
void (*scal_[])(int *, void *, void *, int *) = { iscal_, dscal_, zscal_ };

/***********************************************************************/
/****                                                               ****/
/****                      BLAS 2 prototypes                        ****/
/****                                                               ****/
/***********************************************************************/

/*************   gemv   *************/
extern void dgemv_(char *, int *, int *, void *, void *, int *, void *,
    int *, void *, void *, int *);
extern void zgemv_(char *, int *, int *, void *, void *, int *, void *,
    int *, void *, void *, int *);
static void (*gemv_[])(char *, int *, int *, void *, void *, int *, void *,
    int *, void *, void *, int *) = { NULL, dgemv_, zgemv_ };


/***********************************************************************/
/****                                                               ****/
/****                      BLAS 3 prototypes                        ****/
/****                                                               ****/
/***********************************************************************/

/*************   gemm   *************/
extern void dgemm_(char *, char *, int *, int *, int *, void *, void *,
    int *, void *, int *, void *, void *, int *) ;
extern void zgemm_(char *, char *, int *, int *, int *, void *, void *,
    int *, void *, int *, void *, void *, int *) ;
static void igemm_(char *transA, char *transB, int *m, int *n, int *k,
    void *alpha, void *A, int *ldA, void *B, int *ldB, void *beta,
    void *C, int *ldC) {
  int i, j, l;
  for (j=0; j<*n; j++) {
    for (i=0; i<*m; i++) {
      ((int_t *)C)[i+j*(*m)] = 0;
      for (l=0; l<*k; l++)
        ((int_t *)C)[i+j*(*m)]+=((int_t *)A)[i+l*(*m)]*((int_t *)B)[j*(*k)+l];
    }
  }
}
void (*gemm_[])(char *, char *, int *, int *, int *, void *, void *, int *,
    void *, int *, void *, void *, int *) = { igemm_, dgemm_, zgemm_ };

/*************   symv   *************/
extern void dsymv_(char *, int *, void *, void *, int *, void *, int *,
    void *, void *, int *);
extern void zsymv_(char *, int *, void *, void *, int *, void *, int *,
    void *, void *, int *);
void (*symv[])(char *, int *, void *, void *, int *, void *, int *,
    void *, void *, int *) = { NULL, dsymv_, zsymv_ };

/*************   syrk   *************/
extern void dsyrk_(char *, char *, int *, int *, void *, void *,
    int *, void *, void *, int *);
extern void zsyrk_(char *, char *, int *, int *, void *, void *,
    int *, void *, void *, int *);
void (*syrk_[])(char *, char *, int *, int *, void *, void *,
    int *, void *, void *, int *) = { NULL, dsyrk_, zsyrk_ };

// extern int (*sp_symv[])(char, int, number, ccs *, int, void *, int,
    // number, void *, int) ;

// extern int (*sp_syrk[])(char, char, number, void *, number,
    // void *, int, int, int, int, void **) ;

// Element size array for different types
const int E_SIZE[] = {sizeof(int), sizeof(double), sizeof(double complex)};

// Constants for different types
const number One[] = {{.i=1}, {.d=1.0}, {.z=1.0+0.0*I}};
const number Zero[] = {{.i=0}, {.d=0.0}, {.z=0.0+0.0*I}};

int intOne = 1;

/* val_id: 0 = matrix, else = number
 */
static int
convert_inum(void *dest, void *val, int val_id, int_t offset)
{
  if (val_id==0) { /* 1x1 matrix */
    switch (MAT_ID(val)) {
    case INT:
      *(int_t *)dest = MAT_BUFI(val)[offset]; return 0;
    case DOUBLE:
      *(int_t *)dest = (int_t)round(MAT_BUFD(val)[offset]); return 0;
    default: 
      fprintf(stderr, "Error: cannot cast argument as integer\n");
      return -1;
    }
  } else { 
      *(int_t *)dest = ((int_t *)val)[offset]; return 0;
      return 0;
  }
}

static int
convert_dnum(void *dest, void *val, int val_id, int_t offset)
{
  if (val_id==0) { /* matrix */
    switch (MAT_ID(val)) {
    case INT:    *(double *)dest = MAT_BUFI(val)[offset]; return 0;
    case DOUBLE: *(double *)dest = MAT_BUFD(val)[offset]; return 0;
    default: 
      fprintf(stderr, "Error: cannot cast argument as double\n");
      return -1;
    }
  } else { /* normal number */
    *(double *)dest = ((double *)val)[offset]; return 0;
  }
}

static int
convert_znum(void *dest, void *val, int val_id, int_t offset)
{
  if (val_id==0) { /* 1x1 matrix */
    switch (MAT_ID(val)) {
    case INT:
#ifndef _MSC_VER
      *(double complex *)dest = MAT_BUFI(val)[offset]; return 0;
#else
      *(_Dcomplex *)dest = _Cbuild((double)MAT_BUFI(val)[offset],0.0); return 0;
#endif
    case DOUBLE:
#ifndef _MSC_VER
      *(double complex *)dest = MAT_BUFD(val)[offset]; return 0;
#else
      *(_Dcomplex *)dest = _Cbuild(MAT_BUFD(val)[offset],0.0); return 0;
#endif
    case COMPLEX:
#ifndef _MSC_VER
      *(double complex *)dest = MAT_BUFZ(val)[offset]; return 0;
#else
      *(_Dcomplex *)dest = MAT_BUFZ(val)[offset]; return 0;
#endif
    default: 
      fprintf(stderr, "Error: unsupported matrix type for complex conversion\n");
      return -1;
    }
  } else { 
    *(double complex *)dest = ((double complex *)val)[offset]; return 0;
  }
}

int (*convert_num[])(void *, void *, int, int_t) = {
    convert_inum, convert_dnum, convert_znum };

static void mtx_iabs(void *src, void *dest, int n) {
  int i;
  for (i=0; i<n; i++)
    ((int_t *)dest)[i] = labs(((int_t *)src)[i]);
}

static void mtx_dabs(void *src, void *dest, int n) {
  int i;
  for (i=0; i<n; i++)
    ((double *)dest)[i] = fabs(((double *)src)[i]);
}

static void mtx_zabs(void *src, void *dest, int n) {
  int i;
  for (i=0; i<n; i++)
#ifndef _MSC_VER
    ((double *)dest)[i] = cabs(((double complex *)src)[i]);
#else
    ((double *)dest)[i] = cabs(((_Dcomplex *)src)[i]);
#endif
}

void (*mtx_abs[])(void *, void *, int) = { mtx_iabs, mtx_dabs, mtx_zabs };


void base_gemv(void *A, matrix *x, matrix *y, char trans, 
               void *alpha_ptr, void *beta_ptr, int m, int n,
               int incx, int incy, int offsetA, int offsetx, int offsety)
{
    /* General matrix-vector product for sparse and dense matrices.
        *
        * base_gemv(A, x, y, trans, alpha, beta, m, n, incx, incy, offsetA, offsetx, offsety)
        *
        * PURPOSE
        * If trans is 'N', computes y := alpha*A*x + beta*y.
        * If trans is 'T', computes y := alpha*A^T*x + beta*y.
        * If trans is 'C', computes y := alpha*A^H*x + beta*y.
        * The matrix A is m by n.
        * Returns immediately if n=0 and trans is 'T' or 'C', or if m=0 
        * and trans is 'N'.
        * Computes y := beta*y if n=0, m>0 and trans is 'N', or if m=0, 
        * n>0 and trans is 'T' or 'C'.
        *
        * ARGUMENTS
        * A         'd' or 'z' (sp)matrix
        * x         'd' or 'z' matrix.  Must have the same type as A.
        * y         'd' or 'z' matrix.  Must have the same type as A.
        * trans     'N', 'T' or 'C'
        * alpha     number (int, float or complex).  Complex alpha is only
        *           allowed if A is complex.
        * beta      number (int, float or complex).  Complex beta is only
        *           allowed if A is complex.
        * m         integer.  If negative, the default value is used.
        * n         integer.  If negative, the default value is used.
        * incx      nonzero integer
        * incy      nonzero integer
        * offsetA   nonnegative integer
        * offsetx   nonnegative integer
        * offsety   nonnegative integer
        *
        * This sparse version of GEMV requires that
        *   m <= A.size[0] - (offsetA % A.size[0])
    */
    number a, b;
    
    int id;
    // Input validation
    if (!is_matrix(A) && !is_spmatrix(A)) 
        ERR("A must be a dense or sparse matrix");
    
    if (!is_matrix(x)) err_mtrx("x");
    if (!is_matrix(y)) err_mtrx("y");
    
    if (MAT_ID(x) == INT || MAT_ID(y) == INT || X_ID(A) == INT) {
        ERR_TYPE("invalid matrix types");
    }
    
    if (X_ID(A) != MAT_ID(x) || X_ID(A) != MAT_ID(y)) {
        err_conflicting_ids;
    }
    
    if (trans != 'N' && trans != 'T' && trans != 'C') {
        err_char("trans", "'N', 'T', 'C'");
    }
    
    if (incx == 0) err_nz_int("incx");
    if (incy == 0) err_nz_int("incy");
    
    id = MAT_ID(x);
    if (m < 0) m = X_NROWS(A);
    if (n < 0) n = X_NCOLS(A);
    if ((!m && trans == 'N') || (!n && (trans == 'T' || trans == 'C'))) 
        return;
    
    if (offsetA < 0) err_nn_int("offsetA");
    if (n > 0 && m > 0 && offsetA + (n-1)*MAX(1,X_NROWS(A)) + m >
        X_NROWS(A)*X_NCOLS(A)) 
        err_buf_len("A");
    
    if (offsetx < 0) err_nn_int("offsetx");
    if ((trans == 'N' && n > 0 && offsetx + (n-1)*abs(incx) + 1 > MAT_LGT(x)) ||
        ((trans == 'T' || trans == 'C') && m > 0 &&
            offsetx + (m-1)*abs(incx) + 1 > MAT_LGT(x))) err_buf_len("x");
    
    if (offsety < 0) err_nn_int("offsety");
    if ((trans == 'N' && offsety + (m-1)*abs(incy) + 1 > MAT_LGT(y)) ||
        ((trans == 'T' || trans == 'C') &&
            offsety + (n-1)*abs(incy) + 1 > MAT_LGT(y))) err_buf_len("y");
    
    if (alpha_ptr && convert_num[MAT_ID(x)](&a, alpha_ptr, 1, 0)) err_type("alpha");
    if (beta_ptr && convert_num[MAT_ID(x)](&b, beta_ptr, 1, 0)) err_type("beta");
    
    if (is_matrix(A)) {
        int ldA = MAX(1,X_NROWS(A));
        if (trans == 'N' && n == 0) {
            scal_[id](&m, (beta_ptr ? &b : &Zero[id]), 
                     (unsigned char*)MAT_BUF(y)+offsety*E_SIZE[id], &incy);
        }
        else if ((trans == 'T' || trans == 'C') && m == 0) {
            scal_[id](&n, (beta_ptr ? &b : &Zero[id]), 
                     (unsigned char*)MAT_BUF(y)+offsety*E_SIZE[id], &incy);
        }
        else {
            gemv_[id](&trans, &m, &n, (alpha_ptr ? &a : &One[id]),
                     (unsigned char*)MAT_BUF(A) + offsetA*E_SIZE[id], &ldA,
                     (unsigned char*)MAT_BUF(x) + offsetx*E_SIZE[id], &incx, 
                     (beta_ptr ? &b : &Zero[id]),
                     (unsigned char*)MAT_BUF(y) + offsety*E_SIZE[id], &incy);
        }
    } else {
        if (sp_gemv[id](trans, m, n, (alpha_ptr ? a : One[id]), ((spmatrix *)A)->obj,
                       offsetA, (unsigned char*)MAT_BUF(x) + offsetx*E_SIZE[id], incx, 
                       (beta_ptr ? b : Zero[id]),
                       (unsigned char*)MAT_BUF(y) + offsety*E_SIZE[id], incy)) {
            err_no_memory;
        }
    }
}



void base_gemm(void *A, void *B, void *C, char transA, char transB, 
                       number *alpha, number *beta, bool partial)
{
    /*
        * General matrix-matrix product.
        *
        * gemm(A, B, C, transA='N', transB='N', alpha=1.0, beta=0.0, 
        *      partial=False) 
        *
        * PURPOSE
        * Computes 
        * C := alpha*A*B + beta*C     if transA = 'N' and transB = 'N'.
        * C := alpha*A^T*B + beta*C   if transA = 'T' and transB = 'N'.
        * C := alpha*A^H*B + beta*C   if transA = 'C' and transB = 'N'.
        * C := alpha*A*B^T + beta*C   if transA = 'N' and transB = 'T'.
        * C := alpha*A^T*B^T + beta*C if transA = 'T' and transB = 'T'.
        * C := alpha*A^H*B^T + beta*C if transA = 'C' and transB = 'T'.
        * C := alpha*A*B^H + beta*C   if transA = 'N' and transB = 'C'.
        * C := alpha*A^T*B^H + beta*C if transA = 'T' and transB = 'C'.
        * C := alpha*A^H*B^H + beta*C if transA = 'C' and transB = 'C'.
        * If k=0, this reduces to C := beta*C.
        *
        * ARGUMENTS
        *
        * A         'd' or 'z' matrix
        *
        * B         'd' or 'z' matrix.  Must have the same type as A.
        *
        * C         'd' or 'z' matrix.  Must have the same type as A.
        *
        * transA    'N', 'T' or 'C'
        *
        * transB    'N', 'T' or 'C'
        *
        * alpha     number (int, float or complex).  Complex alpha is only
        *           allowed if A is complex.
        *
        * beta      number (int, float or complex).  Complex beta is only
        *           allowed if A is complex.
        *
        * partial   boolean. If C is sparse and partial is True, then only the
        *           nonzero elements of C are updated irrespective of the
        *           sparsity patterns of A and B.
    */
    number a, b;
    int m, n, k;
    char transA_, transB_;
    
    // Validate input matrices
    if (!(Matrix_Check(A) || SpMatrix_Check(A))) {
        ERR_TYPE("A must be a matrix or sparse matrix");
    }
    if (!(Matrix_Check(B) || SpMatrix_Check(B))) {
        ERR_TYPE("B must be a matrix or sparse matrix");
    }
    if (!(Matrix_Check(C) || SpMatrix_Check(C))) {
        ERR_TYPE("C must be a matrix or sparse matrix");
    }
    
    // Check type consistency
    if (X_ID(A) != X_ID(B) || X_ID(A) != X_ID(C) || X_ID(B) != X_ID(C)) {
        err_conflicting_ids;
    }
    
    // Validate transpose parameters
    if (transA != 'N' && transA != 'T' && transA != 'C') {
        err_char("transA", "'N', 'T', 'C'");
    }
    if (transB != 'N' && transB != 'T' && transB != 'C') {
        err_char("transB", "'N', 'T', 'C'");
    }
    
    // Calculate dimensions
    m = (transA == 'N') ? X_NROWS(A) : X_NCOLS(A);
    n = (transB == 'N') ? X_NCOLS(B) : X_NROWS(B);
    k = (transA == 'N') ? X_NCOLS(A) : X_NROWS(A);
    if (k != ((transB == 'N') ? X_NROWS(B) : X_NCOLS(B))) {
        ERR_TYPE("dimensions of A and B do not match");
    }
    
    // Early return for empty matrices
    if (m == 0 || n == 0) return;
    
    // Convert alpha and beta parameters
    if (alpha && convert_num[X_ID(A)](&a, alpha, 1, 0)) err_type("alpha");
    if (beta && convert_num[X_ID(A)](&b, beta, 1, 0)) err_type("beta");
    
    // Set character versions for compatibility
    transA_ = transA;
    transB_ = transB;
    
    int id = X_ID(A);
    
    // Handle dense matrix case
    if (Matrix_Check(A) && Matrix_Check(B) && Matrix_Check(C)) {
        int ldA = MAX(1, MAT_NROWS(A));
        int ldB = MAX(1, MAT_NROWS(B));
        int ldC = MAX(1, MAT_NROWS(C));
        
        if (id == INT) err_invalid_id;
        
        // Call appropriate BLAS GEMM function
        gemm_[id](&transA_, &transB_, &m, &n, &k, 
                 (alpha ? &a : &One[id]),
                 MAT_BUF(A), &ldA, 
                 MAT_BUF(B), &ldB, 
                 (beta ? &b : &Zero[id]),
                 MAT_BUF(C), &ldC);
    } 
    else {
        // Handle sparse matrix case
        void *z = NULL;
        
        if (sp_gemm[id](transA_, transB_, (alpha ? a : One[id]),
                        Matrix_Check(A) ? MAT_BUF(A) : ((spmatrix *)A)->obj,
                        Matrix_Check(B) ? MAT_BUF(B) : ((spmatrix *)B)->obj,
                        (beta ? b : Zero[id]),
                        Matrix_Check(C) ? MAT_BUF(C) : ((spmatrix *)C)->obj,
                        SpMatrix_Check(A), SpMatrix_Check(B), SpMatrix_Check(C),
                        partial, &z, m, n, k)) {
            err_no_memory;
        }
        
        // Update sparse matrix structure if needed
        if (z) {
            free_ccs(((spmatrix *)C)->obj);
            ((spmatrix *)C)->obj = z;
        }
    }

}



