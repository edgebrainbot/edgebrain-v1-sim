#include "solver.h"
#include "cvxopt.h"
#include "misc.h"

void validate_ecvxcone_settings(DIMs *dims, ECVXConeSettings *stgs);
void validate_problem_data(matrix *c, void *G, matrix *h, void *A, matrix *b, int cdim);
void validate_cone_dimensions(DIMs* dims);
ECVXConeContext *ECVXConeCtx_Init(PrimalStart *primalstart, DualStart *dualstart, DIMs *dims);

/**
 * Initialize the ConeLPResult structure.
 * This function sets all fields of the result structure to their initial values.
 *
 * @param result Pointer to the ConeLPResult structure to initialize.
 */
ECVXConeContext* ecvxcone_init(matrix *c, void *G, matrix *h, void *A, matrix *b, DIMs *dims, 
                            ECVXConeSettings* settings) 
{
    ECVXConeContext *ecvxcone_ctx = ECVXConeCtx_Init(NULL, NULL, dims);
    validate_cone_dimensions(dims); // Validate the cone dimensions
    validate_ecvxcone_settings(dims, settings);    // Validate the settings
    validate_problem_data(c, G, h, A, b, ecvxcone_ctx->cdim); // Validate the problem data

    return ecvxcone_ctx;
}

void ecvxcone_free(ECVXConeContext *ecvxcone_ctx) 
{
    if (ecvxcone_ctx) {
        if (ecvxcone_ctx->primalstart) {
            Matrix_Free(ecvxcone_ctx->primalstart->x);
            Matrix_Free(ecvxcone_ctx->primalstart->s);
            free(ecvxcone_ctx->primalstart);
        }
        if (ecvxcone_ctx->dualstart) {
            Matrix_Free(ecvxcone_ctx->dualstart->y);
            Matrix_Free(ecvxcone_ctx->dualstart->z);
            free(ecvxcone_ctx->dualstart);
        }
        if (ecvxcone_ctx->result) {
            ECVXConeResult_Free(ecvxcone_ctx->result);
        }
        Scaling_Free(ecvxcone_ctx->W_init);
        Scaling_Free(ecvxcone_ctx->W_nt);
        free(ecvxcone_ctx->indq);
        free(ecvxcone_ctx->inds);
        free(ecvxcone_ctx);
    }
}

/**
 * Validate the settings for ECVXCONE.
 * This function checks if the settings are valid and raises errors if they are not.
 *
 * @param stgs Pointer to the ECVXConeSettings structure to validate.
 */
void validate_ecvxcone_settings(DIMs *dims, ECVXConeSettings *stgs) 
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
 * Initialize the scaling structure with identity matrices.
 * This function sets up the scaling structure for initialization and iterations.
 *
 * @param dims Pointer to the DIMs structure containing the dimensions.
 * @return Pointer to the initialized scaling structure.
 */
scaling *init_identity_scaling(DIMs *dims) 
{
    scaling *W_init = malloc(sizeof(scaling));
    Scaling_Init(W_init);

    number number_one;
    number_one.d = 1.0; // Initialize number for value 1.0
    W_init->d = Matrix_New_Val(dims->l, 1, DOUBLE, number_one);
    W_init->di = Matrix_New_Val(dims->l, 1, DOUBLE, number_one);
    W_init->d_count = dims->l;

    W_init->v = malloc(dims->q_size * sizeof(matrix*));
    W_init->beta = malloc(dims->q_size * sizeof(double));
    W_init->v_count = dims->q_size;
    W_init->b_count = dims->q_size;
    for (int i = 0; i < dims->q_size; ++i) {
        W_init->v[i] = Matrix_New(dims->q[i], 1, DOUBLE);
        W_init->beta[i] = 1.0;
        MAT_BUFD(W_init->v[i])[0] = 1.0;
    }

    W_init->r = malloc(dims->s_size * sizeof(matrix*));
    W_init->rti = malloc(dims->s_size * sizeof(matrix*));
    W_init->r_count = dims->s_size;
    for (int i = 0; i < dims->s_size; ++i) {
        int m = dims->s[i];
        W_init->r[i] = Matrix_New(m, m, DOUBLE);
        W_init->rti[i] = Matrix_New(m, m, DOUBLE);
        for (int j = 0; j < m; ++j) {
            MAT_BUFD(W_init->r[i])[j * m + j] = 1.0;
            MAT_BUFD(W_init->rti[i])[j * m + j] = 1.0;
        }
    }
    return W_init;
}

/**
 * Initialize the scaling structure for iterations.
 * This function sets up the scaling structure for iterations based on the dimensions.
 *
 * @param dims Pointer to the DIMs structure containing the dimensions.
 * @return Pointer to the initialized scaling structure for iterations.
 */
scaling *init_nt_scaling(DIMs *dims) 
{
    scaling *W_nt = malloc(sizeof(scaling));
    Scaling_Init(W_nt);

    W_nt->v_count = dims->q_size;
    W_nt->v = (matrix**)malloc(dims->q_size * sizeof(matrix*));
    for (int k = 0; k < dims->q_size; ++k) {
        W_nt->v[k] = Matrix_New(dims->q[k], 1, DOUBLE);
    }

    W_nt->b_count = dims->q_size;
    W_nt->beta = (double*)calloc(dims->q_size, sizeof(double));

    W_nt->r_count = dims->s_size;
    W_nt->r = (matrix**)malloc(dims->s_size * sizeof(matrix*));
    W_nt->rti = (matrix**)malloc(dims->s_size * sizeof(matrix*));

    for (int k = 0; k < dims->s_size; ++k) {
        W_nt->r[k] = Matrix_New(dims->s[k], dims->s[k], DOUBLE);
        W_nt->rti[k] = Matrix_New(dims->s[k], dims->s[k], DOUBLE);
    }

    return W_nt;
}

/**
 * Initialize the ECVXCONE context.
 * This function sets up the context for the ECVXCONE solver with the provided dimensions and start values.
 *
 * @param primalstart Pointer to the PrimalStart structure containing primal start values.
 * @param dualstart Pointer to the DualStart structure containing dual start values.
 * @param dims Pointer to the DIMs structure containing the dimensions of the problem.
 * @return Pointer to the initialized ECVXCONECtx structure.
 */
ECVXConeContext *ECVXConeCtx_Init(PrimalStart *primalstart, DualStart *dualstart, DIMs *dims) 
{
    ECVXConeContext *ctx = (ECVXConeContext*)malloc(sizeof(ECVXConeContext));
    if (!ctx) err_no_memory;

    ECVXConeResult *result = ECVXConeResult_Init(); 
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
        ctx->W_init = init_identity_scaling(dims);
    } else {
        ctx->W_init = NULL; // No dummy scaling structure needed
    }

    // Initialize scaling structure for iterations
    ctx->W_nt = init_nt_scaling(dims);

    return ctx;
}