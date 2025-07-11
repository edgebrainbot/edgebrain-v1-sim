#include "cvxopt.h"
#include "misc.h"

#ifndef __CONEPROG__
#define __CONEPROG__

// Python: options = {}
typedef struct {
    bool debug;
    double kktreg;
    int maxiters;
    double abstol;
    double reltol;
    double feastol;
    bool show_progress;
    int refinement;
    bool use_correction;
} Options;

static Options options = {
    .debug = false,
    .kktreg = -1.0, // -1 indicates None/unset
    .maxiters = 100,
    .abstol = 1e-7,
    .reltol = 1e-6,
    .feastol = 1e-7,
    .show_progress = true,
    .refinement = -1, // -1 indicates None/unset
    .use_correction = true
};


// Primal start structure
typedef struct {
    matrix* x;
    matrix* s;
} PrimalStart;

// Dual start structure  
typedef struct {
    matrix* y;
    matrix* z;
} DualStart;

// Result structure
typedef struct {
    char* status;
    matrix* x;
    matrix* s;
    matrix* y;
    matrix* z;
    double primal_objective;
    double dual_objective;
    double gap;
    double relative_gap;
    double primal_infeasibility;
    double dual_infeasibility;
    double primal_slack;
    double dual_slack;
    double residual_as_primal_infeasibility_certificate;
    double residual_as_dual_infeasibility_certificate;
    int iterations;
} ConeLPResult;


typedef struct {
    void *env;
    void (*call)(void *env, matrix *x, matrix *y, matrix *z);
} SolveClosure;

typedef struct {
    void *env;
    SolveClosure (*call)(void *env, scaling *W, matrix *H, matrix *Df);
} FactorClosure;


// Function declarations
void Gf_gemv(matrix *x, matrix *y, void *G, DIMs *dims, char trans, void* alpha, void* beta);
void Af_gemv(matrix *x, matrix *y, void *A, char trans, void* alpha, void* beta);
void xy_copy(matrix *x, matrix *y);

/************************* Declarations *************************/
extern KKTCholContext* kkt_chol(matrix *G, DIMs *dims, matrix *A, int mnl);
extern void factor_function(scaling *W, matrix *H, matrix *Df, KKTCholContext *ctx);
extern void solve_function(matrix *x, matrix *y, matrix *z, KKTCholContext *ctx);

/* misc library */
extern void misc_sgemv(void *A, matrix *x, matrix *y, DIMs *dims, char trans, double alpha, 
                        double beta, int n, int offsetA, int offsetx, int offsety);
extern void misc_scale(matrix *x, scaling *W, char trans, char inverse);
extern void misc_scale2(matrix *lmbda, matrix *x, DIMs *dims, int mnl, char inverse);
extern void misc_sprod(matrix *x, matrix *y, DIMs *dims, int mnl, char diag);
extern double misc_snrm2(matrix *x, DIMs *dims, int mnl);
extern double misc_max_step(matrix* x, DIMs* dims, int mnl, matrix* sigma);
extern double misc_sdot(matrix *x, matrix *y, DIMs *dims, int mnl);
extern void misc_symm(matrix *x, int n, int offset);
extern scaling* misc_compute_scaling(matrix *s, matrix *z, matrix *lmbda, DIMs *dims, int mnl);
extern void misc_update_scaling(scaling *W, matrix *lmbda, matrix *s, matrix *z);
extern void misc_ssqr(matrix *x, matrix *y, DIMs *dims, int mnl);
extern void misc_sinv(matrix *x, matrix *y, DIMs *dims, int mnl);

/* base library */
extern void base_gemv(void *A, matrix *x, matrix *y, char trans, void *alpha, void *beta, 
              int m, int n, int incx, int incy, int offsetA, int offsetx, int offsety);

/* blas library */
extern void blas_copy(matrix *x, matrix *y, int n, int ix, int iy, int ox, int oy);
extern void blas_axpy(matrix *x, matrix *y, number *alpha, int n, int incx, int incy, int offsetx, int offsety);
extern void blas_scal(void* alpha, matrix* x, int n, int inc, int offset);
extern number blas_dot(matrix *x, matrix *y, int n, int incx, int incy, int offsetx, int offsety);
extern double blas_nrm2(matrix *x, int n, int inc, int offset);
extern void blas_tbsv(matrix *A, matrix *x, char uplo, char trans, char diag, 
          int n, int k, int ldA, int incx, int offsetA, int offsetx);

/* dense library */
extern matrix * dense(spmatrix *sp_mat);

/* sparse library */
extern ccs * transpose(ccs *A, int conjugate);

// Function pointer types for custom operations
typedef matrix* (*XNewCopyFunc)(matrix* u);
typedef double (*XDotFunc)(matrix* u, matrix* v);
typedef void (*XScalFunc)(double alpha, matrix* u);
typedef void (*XAxpyFunc)(matrix* u, matrix* v, double alpha);
typedef matrix* (*YNewCopyFunc)(matrix* u);
typedef double (*YDotFunc)(matrix* u, matrix* v);
typedef void (*YScalFunc)(double alpha, matrix* u);
typedef void (*YAxpyFunc)(matrix* u, matrix* v, double alpha);

// Function pointer for matrix operations
typedef void (*GFunc)(matrix *, matrix *, void *, DIMs *, char, void *, void *);
typedef void (*AFunc)(matrix *, matrix *, void *, char, void *, void *);

// KKT solver function pointer
typedef void* (*KKTSolverFunc)(void* W);


static inline void ConeLPResult_Init(ConeLPResult *result) {
    if (!result) err_no_memory;

    result->status = NULL;
    result->x = NULL;
    result->s = NULL;
    result->y = NULL;
    result->z = NULL;
    result->primal_objective = 0.0;
    result->dual_objective = 0.0;
    result->gap = 0.0;
    result->relative_gap = 0.0;
    result->primal_infeasibility = 0.0;
    result->dual_infeasibility = 0.0;
    result->primal_slack = 0.0;
    result->dual_slack = 0.0;
    result->residual_as_primal_infeasibility_certificate = 0.0;
    result->residual_as_dual_infeasibility_certificate = 0.0;
    result->iterations = 0;
}

static inline void ConeLPResult_Free(ConeLPResult *result) {
    if (result) {
        if (result->status) free(result->status);
        if (result->x) Matrix_Free(result->x);
        if (result->s) Matrix_Free(result->s);
        if (result->y) Matrix_Free(result->y);
        if (result->z) Matrix_Free(result->z);
        free(result);
    }
}

#endif