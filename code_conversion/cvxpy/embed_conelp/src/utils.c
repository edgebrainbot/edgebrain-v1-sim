#include "coneprog.h"
#include "cvxopt.h"
#include "misc.h"

void validate_cvxopt_settings(DIMs *dims, CVXOPT_Settings *stgs);
void validate_problem_data(matrix *c, void *G, matrix *h, void *A, matrix *b, int cdim);
void validate_cvxopt_settings(DIMs *dims, CVXOPT_Settings *stgs);
void validate_cone_dimensions(DIMs* dims);
ConeLPContext *ConeLPContext_Init(PrimalStart *primalstart, DualStart *dualstart, DIMs *dims);

/**
 * Initialize the ConeLPResult structure.
 * This function sets all fields of the result structure to their initial values.
 *
 * @param result Pointer to the ConeLPResult structure to initialize.
 */
ConeLPContext* cvxopt_init(matrix *c, void *G, matrix *h, void *A, matrix *b, DIMs *dims, 
                            CVXOPT_Settings* settings) 
{
    ConeLPContext *conelp_ctx = ConeLPContext_Init(NULL, NULL, dims);
    validate_cone_dimensions(dims); // Validate the cone dimensions
    validate_cvxopt_settings(dims, settings);    // Validate the settings
    validate_problem_data(c, G, h, A, b, conelp_ctx->cdim); // Validate the problem data

    return conelp_ctx;
}

/**
 * Validate the settings for CVXOPT.
 * This function checks if the settings are valid and raises errors if they are not.
 *
 * @param stgs Pointer to the CVXOPT_Settings structure to validate.
 */
void validate_cvxopt_settings(DIMs *dims, CVXOPT_Settings *stgs) 
{
    if (stgs->kktreg != -1.0) { // -1.0 represents None/unset
        if (stgs->kktreg < 0.0) {
            ERR("options['kktreg'] must be a nonnegative scalar");
        }
    }

    if (stgs->maxiters < 1) {    // Times of iterations
        ERR("options['maxiters'] must be a positive integer");
    }

    if (stgs->reltol <= 0.0 && stgs->abstol <= 0.0) {   // Relative and absolute tolerance
        ERR("at least one of options['reltol'] and options['abstol'] must be positive");
    }

    // Set default refinement
    if (stgs->refinement == -1) { // -1 represents None/unset
        if (dims->q_size > 0 || dims->s_size > 0) {
            stgs->refinement = 1;
        } else {
            stgs->refinement = 0;
        }
    } else if (stgs->refinement < 0) {
        ERR_TYPE( "options['refinement'] must be a nonnegative integer");
    }

    // Validate kktsolver
    validate_kktsolver(dims, stgs->kktsolver);
}

/* Validate the KKT solver settings */
void validate_kktsolver(DIMs* dims, const char* kktsolver) {

    // Default solver selection
    char* default_kktsolver = NULL;
    if (kktsolver == NULL) {
        if (dims && (dims->q_size > 0 || dims->s_size > 0)) {
            default_kktsolver = "qr";
        } else {
            default_kktsolver = "chol2";
        }
        kktsolver = default_kktsolver;
    }
    
    // Check if kktsolver is one of the default string solvers
    bool is_string_solver = false;
    if (kktsolver && ((char*)kktsolver)[0] != '\0') { // Assume string if not function pointer
        for (int i = 0; i < 1; ++i) {
            if (strcmp((char*)kktsolver, defaultsolvers[i]) == 0) {
                is_string_solver = true;
                break;
            }
        }
        if (!is_string_solver) {
            snprintf(msg, sizeof(msg), "'%s' is not supported for kktsolver", (char*)kktsolver);
            ERR(msg);
        }
    }
    if (kktsolver == NULL || strlen(kktsolver) == 0) {
        ERR("kktsolver must be a non-empty string");
    }
}

/**
 * Validate the problem data for ConeLP.
 * This function checks if the provided matrices and dimensions are valid.
 *
 * @param c Coefficient matrix for the linear term.
 * @param G Coefficient matrix for the constraints.
 * @param h Right-hand side vector.
 * @param A Coefficient matrix for the constraints.
 * @param b Right-hand side vector for the constraints.
 * @param cdim Cone dimension.
 */
void validate_problem_data(matrix *c, void *G, matrix *h, void *A, matrix *b, int cdim) 
{
    // Validate c
    if (!Matrix_Check(c) || c->id != DOUBLE || c->ncols != 1) {
        ERR_TYPE("'c' must be a 'd' matrix with one column");
    }

    // Validate h 
    if (!Matrix_Check(h) || h->id != DOUBLE || h->ncols != 1) {
        ERR_TYPE("'h' must be a 'd' matrix with one column");
    }
    if (h->nrows != cdim) {
        snprintf(msg, sizeof(msg), "Error: 'h' must be a 'd' matrix of size (%d,1)", cdim);
        ERR_TYPE(msg);
    }

    // Validate G
    bool matrixG = Matrix_Check(G) || SpMatrix_Check(G);
    bool matrixA = Matrix_Check(A) || SpMatrix_Check(A);

    if ((!matrixG || (!matrixA && A != NULL))) {
        ERR("use of function valued G, A requires a user-provided kktsolver");
    }

    if (matrixG) {
        if (Matrix_Check(G) && !SpMatrix_Check(G)) {
            // G is a dense matrix
            matrix *G_mat = (matrix *)G;
            if (G_mat->id != DOUBLE || G_mat->nrows != cdim || G_mat->ncols != c->nrows) {
                snprintf(msg, sizeof(msg), "Error: 'G' must be a 'd' matrix of size (%d, %d)", cdim, c->nrows);
                ERR_TYPE(msg);
            }
        } else if (SpMatrix_Check(G) && !Matrix_Check(G)) {
            // G is a sparse matrix
            spmatrix *G_sp = (spmatrix *)G;
            if (G_sp->obj->id != DOUBLE || G_sp->obj->nrows != cdim || G_sp->obj->ncols != c->nrows) {
                snprintf(msg, sizeof(msg), "Error: 'G' must be a 'd' matrix of size (%d, %d)", cdim, c->nrows);
                ERR_TYPE(msg);
            }
        }
    } 

    // Validate A
    if (A == NULL) {
        // Create empty sparse matrix
        A = SpMatrix_New(0, c->nrows, 0, DOUBLE);
        matrixA = true;
    }
    
    if (matrixA) {
        if (Matrix_Check(A) && !SpMatrix_Check(A)) {
            // A is a dense matrix
            matrix *A_mat = (matrix *)A;
            if (A_mat->id != DOUBLE || A_mat->ncols != c->nrows) {
                snprintf(msg, sizeof(msg), "Error: 'A' must be a 'd' matrix with %d columns", c->nrows);
                ERR_TYPE(msg);
            }
        } else if (SpMatrix_Check(A) && !Matrix_Check(A)) {
            // A is a sparse matrix
            spmatrix *A_sp = (spmatrix *)A;
            if (A_sp->obj->id != DOUBLE  || A_sp->obj->ncols != c->nrows) {
                snprintf(msg, sizeof(msg), "Error: 'A' must be a 'd' matrix with %d columns", c->nrows);
                ERR_TYPE(msg);
            }
        }
    } 

    // Validate b
    if (b == NULL) {
        b = Matrix_New(0, 1, DOUBLE); // Create an empty vector
    }
    if (!Matrix_Check(b) || b->id != DOUBLE || b->ncols != 1) {
        snprintf(msg, sizeof(msg), "Error: 'b' must be a 'd' matrix with one column");
        ERR_TYPE(msg);
    }

    int A_nrows = 0;
    if (Matrix_Check(A) && !SpMatrix_Check(A)) {
        // A is a dense matrix
        matrix *A_mat = (matrix *)A;
        A_nrows = A_mat->nrows;
    } else if (SpMatrix_Check(A) && !Matrix_Check(A)) {
        // A is a sparse matrix
        spmatrix *A_sp = (spmatrix *)A;
        A_nrows = A_sp->obj->nrows;
    }

    if (matrixA && b->nrows != A_nrows) {
        snprintf(msg, sizeof(msg), "Error: 'b' must have length %d", A_nrows);
        ERR_TYPE(msg);
    }
    
}

/**
 * Validate the dimensions of the cone programming problem.
 * This function checks if the dimensions are valid and raises errors if they are not.
 *
 * @param dims Pointer to the DIMs structure containing the dimensions.
 */
void validate_cone_dimensions(DIMs* dims) 
{
// Validate dims structure
    if (dims->l < 0) {
        ERR_TYPE("'dims['l']' must be a nonnegative integer");
    }
    
    // Check q dimensions
    for (int k = 0; k < dims->q_size; ++k) {
        if (dims->q[k] < 1) {
            ERR_TYPE("'dims['q']' must be a list of positive integers");
        }
    }
    
    // Check s dimensions
    for (int k = 0; k < dims->s_size; ++k) {
        if (dims->s[k] < 0) {
            ERR_TYPE("'dims['s']' must be a list of nonnegative integers");
        }
    }
}

/**
 * Initialize the ConeLPRuntimeContext structure.
 * This function sets up the context with the provided parameters and calculates
 * the necessary dimensions and indices for the cone programming problem.
 *
 * @param ctx Pointer to the ConeLPRuntimeContext to initialize.
 * @param G Coefficient matrix for the linear term.
 * @param h Right-hand side vector.
 * @param dims Dimensions of the problem.
 * @param A Coefficient matrix for the constraints.
 * @param b Right-hand side vector for the constraints.
 * @param primalstart Primal start values.
 * @param dualstart Dual start values.
 */
ConeLPContext *ConeLPContext_Init(PrimalStart *primalstart, DualStart *dualstart, DIMs *dims) 
{
    ConeLPContext *ctx = (ConeLPContext*)malloc(sizeof(ConeLPContext));
    if (!ctx) err_no_memory;

    ConeLPResult *result = ConeLPResult_Init(); 
    if (!result) err_no_memory;

    ctx->result = result;

    ctx->primalstart = primalstart;
    ctx->dualstart = dualstart;
    
    ctx->sum_dims_q = sum_array(dims->q, dims->q_size);
    ctx->sum_dims_s = sum_array(dims->s, dims->s_size);

    // Calculate cone dimensions
    ctx->cdim = dims->l;
    ctx->cdim_pckd = dims->l;
    ctx->cdim_diag = dims->l;

    for (int i = 0; i < dims->q_size; ++i) {
        ctx->cdim += dims->q[i];
        ctx->cdim_pckd += dims->q[i];
        ctx->cdim_diag += dims->q[i];
    }

    for (int i = 0; i < dims->s_size; ++i) {
        ctx->cdim += dims->s[i] * dims->s[i];
        ctx->cdim_pckd += dims->s[i] * (dims->s[i] + 1) / 2;
        ctx->cdim_diag += dims->s[i];
    }

    // Data for kth 'q' constraint are found in rows indq[k]:indq[k+1] of G
    ctx->indq = (int*)malloc((dims->q_size + 1) * sizeof(int));
    if (ctx->indq == NULL) err_no_memory;
    ctx->indq[0] = dims->l;
    for (int k = 0; k < dims->q_size; ++k) {
        ctx->indq[k + 1] = ctx->indq[k] + dims->q[k];
    }
    
    // Data for kth 's' constraint are found in rows inds[k]:inds[k+1] of G
    ctx->inds = (int*)malloc((dims->s_size + 1) * sizeof(int));
    if (ctx->inds == NULL) err_no_memory;
    ctx->inds[0] = ctx->indq[dims->q_size];
    for (int k = 0; k < dims->s_size; ++k) {
        ctx->inds[k + 1] = ctx->inds[k] + dims->s[k] * dims->s[k];
    }

    // Initialize scaling structure
    if (primalstart == NULL || dualstart == NULL) {
        ctx->dummy_W = malloc(sizeof(scaling));
        Scaling_Init(ctx->dummy_W);

        number number_one;
        number_one.d = 1.0; // Initialize number for value 1.0
        ctx->dummy_W->d = Matrix_New_Val(dims->l, 1, DOUBLE, number_one);
        ctx->dummy_W->di = Matrix_New_Val(dims->l, 1, DOUBLE, number_one);
        ctx->dummy_W->d_count = dims->l;

        ctx->dummy_W->v = malloc(dims->q_size * sizeof(matrix*));
        ctx->dummy_W->beta = malloc(dims->q_size * sizeof(double));
        ctx->dummy_W->v_count = dims->q_size;
        ctx->dummy_W->b_count = dims->q_size;
        for (int i = 0; i < dims->q_size; ++i) {
            ctx->dummy_W->v[i] = Matrix_New(dims->q[i], 1, DOUBLE);
            ctx->dummy_W->beta[i] = 1.0;
            MAT_BUFD(ctx->dummy_W->v[i])[0] = 1.0;
        }

        ctx->dummy_W->r = malloc(dims->s_size * sizeof(matrix*));
        ctx->dummy_W->rti = malloc(dims->s_size * sizeof(matrix*));
        ctx->dummy_W->r_count = dims->s_size;
        for (int i = 0; i < dims->s_size; ++i) {
            int m = dims->s[i];
            ctx->dummy_W->r[i] = Matrix_New(m, m, DOUBLE);
            ctx->dummy_W->rti[i] = Matrix_New(m, m, DOUBLE);
            for (int j = 0; j < m; ++j) {
                MAT_BUFD(ctx->dummy_W->r[i])[j * m + j] = 1.0;
                MAT_BUFD(ctx->dummy_W->rti[i])[j * m + j] = 1.0;
            }
        }
    } else {
        ctx->dummy_W = NULL; // No dummy scaling structure needed
    }

    return ctx;
}

/**
 * Debugging function to print a matrix by projecting it onto a vector of ones.
 * This is useful for debugging purposes to visualize the matrix.
 *
 * @param m Pointer to the matrix to debug.
 */
void debug_matrix_by_project2vector(matrix *m) 
{
    number temp_n;
    double alpha = 1.0;
    double beta = 0.0;
    temp_n.d = 1.0;
    matrix *vector_one = Matrix_New_Val(m->ncols ,1, DOUBLE, temp_n);
    matrix *res = Matrix_New_Val(m->nrows, 1, DOUBLE, temp_n);
    blas_gemv(m, vector_one, res, 'N', &alpha, &beta, -1, -1, 0, 1, 1, 0, 0, 0);

    print_matrix_(res);

    // Free the allocated matrices
    Matrix_Free(vector_one);
    Matrix_Free(res);
}