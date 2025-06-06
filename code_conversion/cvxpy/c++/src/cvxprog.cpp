
#include "cvxprog.hpp"

// Forward declaration for cp function
std::map<std::string, std::variant<std::string, Matrix, double>>
cp(std::function<std::variant<std::tuple<int, Matrix>, std::tuple<Matrix, Matrix>, std::tuple<Matrix, Matrix, Matrix>>(const Matrix*, const Matrix*)> F,
   const std::variant<Matrix, SparseMatrix>& G, const Matrix& h,
   const std::map<std::string, std::variant<int, std::vector<int>>>& dims,
   const std::variant<Matrix, SparseMatrix>& A, const Matrix& b,
   std::function<void()> kktsolver = nullptr,
   const std::map<std::string, std::variant<bool, int, double>>& options = {});

// Global options (assuming this exists)
extern std::map<std::string, std::variant<bool, int, double>> options;

std::map<std::string, std::variant<std::string, Matrix, double>>
gp(const std::vector<int>& K, const std::variant<Matrix, SparseMatrix>& F, const Matrix& g,
   const std::variant<Matrix, SparseMatrix>* G = nullptr, const Matrix* h = nullptr,
   const std::variant<Matrix, SparseMatrix>* A = nullptr, const Matrix* b = nullptr,
   std::function<void()> kktsolver = nullptr,
   const std::map<std::string, std::variant<bool, int, double>>& kwargs = {}) {

    /*
    Solves a geometric program

        minimize    log sum exp (F0*x+g0)
        subject to  log sum exp (Fi*x+gi) <= 0,  i=1,...,m
                    G*x <= h
                    A*x = b

    Input arguments.

        K is a list of positive integers [K0, K1, K2, ..., Km].

        F is a sum(K)xn dense or sparse 'd' matrix with block rows F0,
        F1, ..., Fm.  Each Fi is Kixn.

        g is a sum(K)x1 dense or sparse 'd' matrix with blocks g0, g1,
        g2, ..., gm.  Each gi is Kix1.

        G is an mxn dense or sparse 'd' matrix.

        h is an mx1 dense 'd' matrix.

        A is a pxn dense or sparse 'd' matrix.

        b is a px1 dense 'd' matrix.

        The default values for G, h, A and b are empty matrices with
        zero rows.


    Output arguments.

        Returns a dictionary with keys 'status', 'x', 'snl', 'sl',
        'znl', 'zl', 'y', 'primal objective', 'dual objective', 'gap',
        'relative gap', 'primal infeasibility', 'dual infeasibility',
        'primal slack', 'dual slack'.

        The 'status' field has values 'optimal' or 'unknown'.
        If status is 'optimal', x, snl, sl, y, znl, zl  are approximate
        solutions of the primal and dual optimality conditions

            f(x)[1:] + snl = 0,  G*x + sl = h,  A*x = b
            Df(x)'*[1; znl] + G'*zl + A'*y + c = 0
            snl >= 0,  znl >= 0,  sl >= 0,  zl >= 0
            snl'*znl + sl'* zl = 0,

        where fk(x) = log sum exp (Fk*x + gk).

        If status is 'unknown', x, snl, sl, y, znl, zl are the last
        iterates before termination.  They satisfy snl > 0, znl > 0,
        sl > 0, zl > 0, but are not necessarily feasible.

        The values of the other fields are the values returned by cpl()
        applied to the epigraph form problem

            minimize   t
            subjec to  f0(x) <= t
                       fk(x) <= 0, k = 1, ..., mnl
                       G*x <= h
                       A*x = b.

        Termination with status 'unknown' indicates that the algorithm
        failed to find a solution that satisfies the specified tolerances.
        In some cases, the returned solution may be fairly accurate.  If
        the primal and dual infeasibilities, the gap, and the relative gap
        are small, then x, y, snl, sl, znl, zl are close to optimal.


    Control parameters.

       The following control parameters can be modified by adding an
       entry to the dictionary options.

       options['show_progress'] True/False (default: True)
       options['maxiters'] positive integer (default: 100)
       options['refinement'] nonnegative integer (default: 1)
       options['abstol'] scalar (default: 1e-7)
       options['reltol'] scalar (default: 1e-6)
       options['feastol'] scalar (default: 1e-7).
    */

    std::map<std::string, std::variant<bool, int, double>> local_options;
    auto it = kwargs.find("options");
    if (it != kwargs.end()) {
        local_options = std::get<std::map<std::string, std::variant<bool, int, double>>>(it->second);
    } else {
        local_options = options;
    }

    // Validate K
    if (K.empty() || std::any_of(K.begin(), K.end(), [](int k) { return k <= 0; })) {
        throw std::invalid_argument("'K' must be a list of positive integers");
    }
    int mnl = K.size() - 1;
    int l = std::accumulate(K.begin(), K.end(), 0);

    // Validate F
    int n;
    if (std::holds_alternative<Matrix>(F)) {
        const Matrix& F_mat = std::get<Matrix>(F);
        if (F_mat.typecode != 'd' || F_mat.size().first != l) {
            throw std::invalid_argument("'F' must be a dense or sparse 'd' matrix with " + std::to_string(l) + " rows");
        }
        n = F_mat.size().second;
    } else {
        const SparseMatrix& F_mat = std::get<SparseMatrix>(F);
        if (F_mat.typecode != 'd' || F_mat.size().first != l) {
            throw std::invalid_argument("'F' must be a dense or sparse 'd' matrix with " + std::to_string(l) + " rows");
        }
        n = F_mat.size().second;
    }

    // Validate g
    if (g.typecode != 'd' || g.size() != std::make_pair(l, 1)) {
        throw std::invalid_argument("'g' must be a dense 'd' matrix of size (" + std::to_string(l) + ",1)");
    }

    // Handle G and h
    std::variant<Matrix, SparseMatrix> G_val;
    Matrix h_val(0.0, 0, 1);
    if (G == nullptr) {
        G_val = SparseMatrix({}, {}, {}, {0, n});
    } else {
        G_val = *G;
    }
    if (h == nullptr) {
        h_val = Matrix(0.0, 0, 1);
    } else {
        h_val = *h;
    }

    // Validate G
    int ml;
    if (std::holds_alternative<Matrix>(G_val)) {
        const Matrix& G_mat = std::get<Matrix>(G_val);
        if (G_mat.typecode != 'd' || G_mat.size().second != n) {
            throw std::invalid_argument("'G' must be a dense or sparse 'd' matrix with " + std::to_string(n) + " columns");
        }
        ml = G_mat.size().first;
    } else {
        const SparseMatrix& G_mat = std::get<SparseMatrix>(G_val);
        if (G_mat.typecode != 'd' || G_mat.size().second != n) {
            throw std::invalid_argument("'G' must be a dense or sparse 'd' matrix with " + std::to_string(n) + " columns");
        }
        ml = G_mat.size().first;
    }

    // Validate h
    if (h_val.typecode != 'd' || h_val.size() != std::make_pair(ml, 1)) {
        throw std::invalid_argument("'h' must be a dense 'd' matrix of size (" + std::to_string(ml) + ",1)");
    }

    std::map<std::string, std::variant<int, std::vector<int>>> dims = {
        {"l", ml}, {"s", std::vector<int>{}}, {"q", std::vector<int>{}}
    };

    // Handle A and b
    std::variant<Matrix, SparseMatrix> A_val;
    Matrix b_val(0.0, 0, 1);
    if (A == nullptr) {
        A_val = SparseMatrix({}, {}, {}, {0, n});
    } else {
        A_val = *A;
    }
    if (b == nullptr) {
        b_val = Matrix(0.0, 0, 1);
    } else {
        b_val = *b;
    }

    // Validate A
    int p;
    if (std::holds_alternative<Matrix>(A_val)) {
        const Matrix& A_mat = std::get<Matrix>(A_val);
        if (A_mat.typecode != 'd' || A_mat.size().second != n) {
            throw std::invalid_argument("'A' must be a dense or sparse 'd' matrix with " + std::to_string(n) + " columns");
        }
        p = A_mat.size().first;
    } else {
        const SparseMatrix& A_mat = std::get<SparseMatrix>(A_val);
        if (A_mat.typecode != 'd' || A_mat.size().second != n) {
            throw std::invalid_argument("'A' must be a dense or sparse 'd' matrix with " + std::to_string(n) + " columns");
        }
        p = A_mat.size().first;
    }

    // Validate b
    if (b_val.typecode != 'd' || b_val.size() != std::make_pair(p, 1)) {
        throw std::invalid_argument("'b' must be a dense 'd' matrix of size (" + std::to_string(p) + ",1)");
    }

    Matrix y(0.0, l, 1);
    int max_K = *std::max_element(K.begin(), K.end());
    Matrix u(0.0, max_K, 1);
    Matrix Fsc(0.0, max_K, n);

    std::vector<int> cs1, cs2;
    std::vector<std::tuple<int, int, int>> ind;

    cs1.push_back(0);
    for (int i = 1; i <= mnl; ++i) {
        cs1.push_back(std::accumulate(K.begin(), K.begin() + i, 0));
    }

    for (int i = 0; i <= mnl; ++i) {
        cs2.push_back(cs1[i] + K[i]);
        ind.push_back(std::make_tuple(i, cs1[i], cs2[i]));
    }

    // Define Fgp function
    auto Fgp = [&](const Matrix* x = nullptr, const Matrix* z = nullptr)
        -> std::variant<std::tuple<int, Matrix>, std::tuple<Matrix, Matrix>, std::tuple<Matrix, Matrix, Matrix>> {

        if (x == nullptr) {
            return std::make_tuple(mnl, Matrix(0.0, n, 1));
        }

        Matrix f(0.0, mnl + 1, 1);
        Matrix Df(0.0, mnl + 1, n);

        // y = F*x+g
        blas::copy(g, y);
        if (std::holds_alternative<Matrix>(F)) {
            base::gemv(std::get<Matrix>(F), *x, y, 'N', -1, 1.0, 1.0);
        } else {
            base::gemv(std::get<SparseMatrix>(F), *x, y, 'N', -1, 1.0, 1.0);
        }

        Matrix H;
        if (z != nullptr) {
            H = Matrix(0.0, n, n);
        }

        for (const auto& [i, start, stop] : ind) {
            // yi := exp(yi) = exp(Fi*x+gi)
            double ymax = *std::max_element(y.data.begin() + start, y.data.begin() + stop);
            for (int k = start; k < stop; ++k) {
                y[k] = std::exp(y[k] - ymax);
            }

            // fi = log sum yi = log sum exp(Fi*x+gi)
            double ysum = blas::asum(y, stop - start, start);
            f[i] = ymax + std::log(ysum);

            // yi := yi / sum(yi) = exp(Fi*x+gi) / sum(exp(Fi*x+gi))
            blas::scal(1.0 / ysum, y, stop - start, start);

            // gradfi := Fi' * yi
            //        = Fi' * exp(Fi*x+gi) / sum(exp(Fi*x+gi))
            if (std::holds_alternative<Matrix>(F)) {
                base::gemv(std::get<Matrix>(F), y, Df, 'T', stop - start, 1.0, 0.0, mnl + 1, start, start, i);
            } else {
                base::gemv(std::get<SparseMatrix>(F), y, Df, 'T', stop - start, 1.0, 0.0, mnl + 1, start, start, i);
            }

            if (z != nullptr) {
                // Hi = Fi' * (diag(yi) - yi*yi') * Fi
                //    = Fisc' * Fisc
                // where
                // Fisc = diag(yi)^1/2 * (I - 1*yi') * Fi
                //      = diag(yi)^1/2 * (Fi - 1*gradfi')

                // Copy F[start:stop, :] to Fsc[:K[i], :]
                if (std::holds_alternative<Matrix>(F)) {
                    const Matrix& F_mat = std::get<Matrix>(F);
                    for (int r = 0; r < K[i]; ++r) {
                        for (int c = 0; c < n; ++c) {
                            Fsc(r, c) = F_mat(start + r, c);
                        }
                    }
                }

                for (int k = start; k < stop; ++k) {
                    blas::axpy(Df, Fsc, n, -1.0, mnl + 1, Fsc.size().first, i, k - start);
                    blas::scal(std::sqrt(y[k]), Fsc, -1, k - start, Fsc.size().first);
                }

                // H += z[i]*Hi = z[i] * Fisc' * Fisc
                blas::syrk(Fsc, H, 'T', stop - start, (*z)[i], 1.0);
            }
        }

        if (z == nullptr) {
            return std::make_tuple(f, Df);
        } else {
            return std::make_tuple(f, Df, H);
        }
    };

    return cp(Fgp, G_val, h_val, dims, A_val, b_val, kktsolver, local_options);
}




#include <vector>
#include <map>
#include <string>
#include <functional>
#include <memory>
#include <stdexcept>
#include <cmath>

// Required dependencies for matrix operations
#include "matrix.h"  // Assuming matrix class with operations
#include "spmatrix.h" // Assuming sparse matrix class
#include "blas.h"    // BLAS operations
#include "base.h"    // Base matrix operations
#include "misc.h"    // Miscellaneous operations
#include "cpl.h"     // CPL solver

// Function type definitions
using MatrixPtr = std::shared_ptr<Matrix>;
using SparseMatrixPtr = std::shared_ptr<SparseMatrix>;
using VectorType = std::vector<double>;

// Function object types
using FunctionF = std::function<std::tuple<VectorType, std::function<void(const std::vector<double>&, std::vector<double>&, double, double, char)>, std::function<void(const std::vector<double>&, std::vector<double>&, double, double)>>(const std::vector<double>&, const std::vector<double>&)>;
using LinearOperator = std::function<void(const std::vector<double>&, std::vector<double>&, double, double, char)>;
using KKTSolver = std::function<std::function<void(std::vector<double>&, std::vector<double>&, std::vector<double>&)>(const std::vector<double>&, const std::vector<double>&, const std::map<std::string, std::vector<double>>&)>;

// Global variables that would be accessible in the context
extern int mnl;
extern std::vector<double> x0;
extern MatrixPtr matrix;
extern SparseMatrixPtr spmatrix;
extern std::map<std::string, std::vector<int>> dims;
extern std::string kktsolver;
extern double KKTREG;
extern int cdim;
extern FunctionF F;
extern LinearOperator G;
extern LinearOperator A;
extern std::function<std::vector<double>(const std::vector<double>&)> xnewcopy;
extern std::function<double(const std::vector<double>&, const std::vector<double>&)> xdot;
extern std::function<void(const std::vector<double>&, std::vector<double>&, double)> xaxpy;
extern std::function<void(double, std::vector<double>&)> xscal;
extern std::function<void(const std::vector<double>&, std::vector<double>&)> xcopy;
extern std::function<std::map<std::string, std::vector<double>>(const std::vector<double>&, const std::vector<double>&, const std::map<std::string, std::vector<double>>&)> factor;

// Function F_e definition
std::function<std::tuple<int, std::vector<std::variant<std::vector<double>, double>>>(std::optional<std::vector<std::variant<std::vector<double>, double>>>, std::optional<std::vector<double>>)> F_e =
    [](std::optional<std::vector<std::variant<std::vector<double>, double>>> x, std::optional<std::vector<double>> z) -> std::tuple<int, std::vector<std::variant<std::vector<double>, double>>> {

    if (!x.has_value()) {
        std::vector<std::variant<std::vector<double>, double>> result;
        result.push_back(x0);
        result.push_back(0.0);
        return std::make_tuple(mnl + 1, result);
    }
    else {
        std::vector<double> x_0 = std::get<std::vector<double>>(x.value()[0]);
        double x_1 = std::get<double>(x.value()[1]);

        if (!z.has_value()) {
            auto v = F(x_0, std::nullopt);
            if (v == std::nullopt || std::get<0>(v.value()).empty()) {
                return std::make_tuple(0, std::vector<std::variant<std::vector<double>, double>>());
            }

            VectorType val = matrix(std::get<0>(v.value()), 'd');
            val[0] -= x_1;
            auto Df = std::get<1>(v.value());

            // Define Df_e based on type of Df
            std::function<void(const std::vector<std::variant<std::vector<double>, double>>&, std::vector<double>&, double, double, char)> Df_e;

            if (typeid(Df) == typeid(MatrixPtr) || typeid(Df) == typeid(SparseMatrixPtr)) {
                Df_e = [Df](const std::vector<std::variant<std::vector<double>, double>>& u, std::vector<double>& v, double alpha = 1.0, double beta = 0.0, char trans = 'N') {
                    std::vector<double> u_0 = std::get<std::vector<double>>(u[0]);
                    double u_1 = std::get<double>(u[1]);

                    if (trans == 'N') {
                        base::gemv(*Df, u_0, v, alpha, beta, 'N');
                        v[0] -= alpha * u_1;
                    }
                    else {
                        std::vector<double> v_0(u_0.size());
                        std::copy(v.begin(), v.begin() + u_0.size(), v_0.begin());
                        base::gemv(*Df, u, v_0, alpha, beta, 'T');
                        std::copy(v_0.begin(), v_0.end(), v.begin());
                        v[1] = -alpha * u[0] + beta * v[1];
                    }
                };
            }
            else {
                Df_e = [Df](const std::vector<std::variant<std::vector<double>, double>>& u, std::vector<double>& v, double alpha = 1.0, double beta = 0.0, char trans = 'N') {
                    std::vector<double> u_0 = std::get<std::vector<double>>(u[0]);
                    double u_1 = std::get<double>(u[1]);

                    if (trans == 'N') {
                        Df(u_0, v, alpha, beta, 'N');
                        v[0] -= alpha * u_1;
                    }
                    else {
                        std::vector<double> v_0(u_0.size());
                        std::copy(v.begin(), v.begin() + u_0.size(), v_0.begin());
                        Df(u, v_0, alpha, beta, 'T');
                        std::copy(v_0.begin(), v_0.end(), v.begin());
                        v[1] = -alpha * u[0] + beta * v[1];
                    }
                };
            }

            return std::make_tuple(val.size(), std::vector<std::variant<std::vector<double>, double>>{val, Df_e});
        }
        else {
            auto [val, Df, H] = F(x_0, z.value());
            VectorType val_matrix = matrix(val, 'd');
            val_matrix[0] -= x_1;

            // Define Df_e based on type of Df
            std::function<void(const std::vector<std::variant<std::vector<double>, double>>&, std::vector<double>&, double, double, char)> Df_e;

            if (typeid(Df) == typeid(MatrixPtr) || typeid(Df) == typeid(SparseMatrixPtr)) {
                Df_e = [Df](const std::vector<std::variant<std::vector<double>, double>>& u, std::vector<double>& v, double alpha = 1.0, double beta = 0.0, char trans = 'N') {
                    std::vector<double> u_0 = std::get<std::vector<double>>(u[0]);
                    double u_1 = std::get<double>(u[1]);

                    if (trans == 'N') {
                        base::gemv(*Df, u_0, v, alpha, beta, 'N');
                        v[0] -= alpha * u_1;
                    }
                    else {
                        std::vector<double> v_0(u_0.size());
                        std::copy(v.begin(), v.begin() + u_0.size(), v_0.begin());
                        base::gemv(*Df, u, v_0, alpha, beta, 'T');
                        std::copy(v_0.begin(), v_0.end(), v.begin());
                        v[1] = -alpha * u[0] + beta * v[1];
                    }
                };
            }
            else {
                Df_e = [Df](const std::vector<std::variant<std::vector<double>, double>>& u, std::vector<double>& v, double alpha = 1.0, double beta = 0.0, char trans = 'N') {
                    std::vector<double> u_0 = std::get<std::vector<double>>(u[0]);
                    double u_1 = std::get<double>(u[1]);

                    if (trans == 'N') {
                        Df(u_0, v, alpha, beta, 'N');
                        v[0] -= alpha * u_1;
                    }
                    else {
                        std::vector<double> v_0(u_0.size());
                        std::copy(v.begin(), v.begin() + u_0.size(), v_0.begin());
                        Df(u, v_0, alpha, beta, 'T');
                        std::copy(v_0.begin(), v_0.end(), v.begin());
                        v[1] = -alpha * u[0] + beta * v[1];
                    }
                };
            }

            // Define H_e based on type of H
            std::function<void(const std::vector<std::variant<std::vector<double>, double>>&, std::vector<std::variant<std::vector<double>, double>>&, double, double)> H_e;

            if (typeid(H) == typeid(MatrixPtr) || typeid(H) == typeid(SparseMatrixPtr)) {
                H_e = [H](const std::vector<std::variant<std::vector<double>, double>>& u, std::vector<std::variant<std::vector<double>, double>>& v, double alpha = 1.0, double beta = 1.0) {
                    std::vector<double> u_0 = std::get<std::vector<double>>(u[0]);
                    std::vector<double>& v_0 = std::get<std::vector<double>>(v[0]);
                    double& v_1 = std::get<double>(v[1]);

                    base::symv(*H, u_0, v_0, alpha, beta);
                    v_1 += beta * v_1;
                };
            }
            else {
                H_e = [H](const std::vector<std::variant<std::vector<double>, double>>& u, std::vector<std::variant<std::vector<double>, double>>& v, double alpha = 1.0, double beta = 1.0) {
                    std::vector<double> u_0 = std::get<std::vector<double>>(u[0]);
                    std::vector<double>& v_0 = std::get<std::vector<double>>(v[0]);
                    double& v_1 = std::get<double>(v[1]);

                    H(u_0, v_0, alpha, beta);
                    v_1 += beta * v_1;
                };
            }

            return std::make_tuple(val_matrix.size(), std::vector<std::variant<std::vector<double>, double>>{val_matrix, Df_e, H_e});
        }
    }
};

// Linear inequality constraints.
//
//     G_e  = [ G, 0 ]
//

std::function<void(const std::vector<std::variant<std::vector<double>, double>>&, std::vector<double>&, double, double, char)> G_e;

if (typeid(G) == typeid(MatrixPtr) || typeid(G) == typeid(SparseMatrixPtr)) {
    G_e = [](const std::vector<std::variant<std::vector<double>, double>>& u, std::vector<double>& v, double alpha = 1.0, double beta = 0.0, char trans = 'N') {
        if (trans == 'N') {
            std::vector<double> u_0 = std::get<std::vector<double>>(u[0]);
            misc::sgemv(G, u_0, v, dims, alpha, beta);
        }
        else {
            std::vector<double>& v_0 = std::get<std::vector<double>>(u[0]);
            double& v_1 = std::get<double>(u[1]);
            misc::sgemv(G, u, v_0, dims, alpha, beta, 'T');
            v_1 *= beta;
        }
    };
}
else {
    G_e = [](const std::vector<std::variant<std::vector<double>, double>>& u, std::vector<double>& v, double alpha = 1.0, double beta = 0.0, char trans = 'N') {
        if (trans == 'N') {
            std::vector<double> u_0 = std::get<std::vector<double>>(u[0]);
            G(u_0, v, alpha, beta);
        }
        else {
            std::vector<double>& v_0 = std::get<std::vector<double>>(u[0]);
            double& v_1 = std::get<double>(u[1]);
            G(u, v_0, alpha, beta, 'T');
            v_1 *= beta;
        }
    };
}

// Linear equality constraints.
//
//     A_e = [ A, 0 ]
//

std::function<void(const std::vector<std::variant<std::vector<double>, double>>&, std::vector<double>&, double, double, char)> A_e;

if (typeid(A) == typeid(MatrixPtr) || typeid(A) == typeid(SparseMatrixPtr)) {
    A_e = [](const std::vector<std::variant<std::vector<double>, double>>& u, std::vector<double>& v, double alpha = 1.0, double beta = 0.0, char trans = 'N') {
        if (trans == 'N') {
            std::vector<double> u_0 = std::get<std::vector<double>>(u[0]);
            base::gemv(A, u_0, v, alpha, beta);
        }
        else {
            std::vector<double>& v_0 = std::get<std::vector<double>>(u[0]);
            double& v_1 = std::get<double>(u[1]);
            base::gemv(A, u, v_0, alpha, beta, 'T');
            v_1 *= beta;
        }
    };
}
else {
    A_e = [](const std::vector<std::variant<std::vector<double>, double>>& u, std::vector<double>& v, double alpha = 1.0, double beta = 0.0, char trans = 'N') {
        if (trans == 'N') {
            std::vector<double> u_0 = std::get<std::vector<double>>(u[0]);
            A(u_0, v, alpha, beta);
        }
        else {
            std::vector<double>& v_0 = std::get<std::vector<double>>(u[0]);
            double& v_1 = std::get<double>(u[1]);
            A(u, v_0, alpha, beta, 'T');
            v_1 *= beta;
        }
    };
}

// kktsolver(x, z, W) returns a routine for solving equations with
// coefficient matrix
//
//         [ H             A'   [Df[1:]; G]' ]
//     K = [ A             0    0            ].
//         [ [Df[1:]; G]   0    -W'*W        ]

if (kktsolver.empty()) {
    if (!dims.empty() && (!dims["q"].empty() || !dims["s"].empty())) {
        kktsolver = "chol";
    }
    else {
        kktsolver = "chol2";
    }
}

std::function<std::function<void(std::vector<double>&, std::vector<double>&, std::vector<double>&)>(const std::vector<double>&, const std::vector<double>&, const std::map<std::string, std::vector<double>>&)> kktsolver_func;

if (kktsolver == "ldl" || kktsolver == "chol" || kktsolver == "chol2" || kktsolver == "qr") {
    if (kktsolver == "ldl") {
        factor = misc::kkt_ldl(G, dims, A, mnl, KKTREG);
    }
    else if (kktsolver == "qr") {
        factor = misc::kkt_qr(G, dims, A, mnl);
    }
    else if (kktsolver == "chol") {
        factor = misc::kkt_chol(G, dims, A, mnl);
    }
    else {
        factor = misc::kkt_chol2(G, dims, A, mnl);
    }

    kktsolver_func = [](const std::vector<double>& x, const std::vector<double>& z, const std::map<std::string, std::vector<double>>& W) {
        auto [f, Df, H] = F(x, z);
        std::vector<double> Df_slice(Df.begin() + 1, Df.end());
        return factor(W, H, Df_slice);
    };
}

std::vector<double> ux = xnewcopy(x0);
std::vector<double> uz = matrix(0.0, mnl + cdim, 1);

std::function<std::function<void(std::vector<std::variant<std::vector<double>, double>>&, std::vector<double>&, std::vector<double>&)>(const std::vector<std::variant<std::vector<double>, double>>&, const std::vector<double>&, const std::map<std::string, std::vector<double>>&)> kktsolver_e =
    [&](const std::vector<std::variant<std::vector<double>, double>>& x, const std::vector<double>& znl, const std::map<std::string, std::vector<double>>& W) {

    std::map<std::string, std::vector<double>> We = W;
    std::vector<double> W_dnl_slice(W.at("dnl").begin() + 1, W.at("dnl").end());
    std::vector<double> W_dnli_slice(W.at("dnli").begin() + 1, W.at("dnli").end());
    We["dnl"] = W_dnl_slice;
    We["dnli"] = W_dnli_slice;

    std::vector<double> x_0 = std::get<std::vector<double>>(x[0]);
    auto g = kktsolver_func(x_0, znl, We);

    auto [f, Df] = F(x_0);
    std::vector<double> gradf0;

    if (typeid(Df) == typeid(MatrixPtr)) {
        // gradf0 = Df[0,:].T
        MatrixPtr Df_mat = std::static_pointer_cast<Matrix>(Df);
        gradf0 = Df_mat->getRow(0).transpose();
    }
    else if (typeid(Df) == typeid(SparseMatrixPtr)) {
        // gradf0 = matrix(Df[0,:].T)
        SparseMatrixPtr Df_sp = std::static_pointer_cast<SparseMatrix>(Df);
        gradf0 = matrix(Df_sp->getRow(0).transpose());
    }
    else {
        gradf0 = xnewcopy(x_0);
        std::vector<double> e0(mnl + 1, 0.0);
        e0[0] = 1.0;
        Df(e0, gradf0, 1.0, 0.0, 'T');
    }

    auto solve = [&](std::vector<std::variant<std::vector<double>, double>>& x, std::vector<double>& y, std::vector<double>& z) {

        // Solves
        //
        //    [ [ H   0  ]   [ A' ]  [ Df'  G'] ] [ ux ]    [ bx ]
        //    [ [ 0   0  ]   [ 0  ]  [ -e0' 0 ] ] [    ]    [    ]
        //    [                                 ] [    ]    [    ]
        //    [ [ A   0  ]   0       0          ] [ uy ] =  [ by ].
        //    [                                 ] [    ]    [    ]
        //    [ [ Df -e0 ]   0       -W'*W      ] [ uz ]    [ bz ]
        //    [ [ G   0  ]                      ] [    ]    [    ]
        //
        // The solution is:
        //
        //     uz[0] = -bx[1]
        //
        //     [ ux[0]  ]          [ bx[0] + bx[1] * gradf0 ]
        //     [ uy     ] = K^-1 * [ by                     ].
        //     [ uz[1:] ]          [ bz[1:]                 ]
        //
        //     ux[1] = gradf0' * ux[0] - W['dnl'][0]**2 * uz[0] - bz[0]
        //           = gradf0' * ux[0] + W['dnl'][0]**2 * bx[1] - bz[0].
        //
        // Instead of uz we return the scaled solution W*uz.

        double a = z[0];
        std::vector<double>& x_0 = std::get<std::vector<double>>(x[0]);
        double& x_1 = std::get<double>(x[1]);

        xcopy(x_0, ux);
        xaxpy(gradf0, ux, x_1);
        std::vector<double> z_slice(z.begin() + 1, z.end());
        blas::copy(z_slice, uz, 1);
        g(ux, y, uz);
        z[0] = -x_1 * W.at("dnl")[0];
        std::vector<double> uz_slice(uz.begin() + 1, uz.end());
        blas::copy(uz_slice, z_slice, 1);
        std::copy(z_slice.begin(), z_slice.end(), z.begin() + 1);
        xcopy(ux, x_0);
        x_1 = xdot(gradf0, x_0) + std::pow(W.at("dnl")[0], 2) * x_1 - a;
    };

    return solve;
};

auto xnewcopy_e = [](const std::vector<std::variant<std::vector<double>, double>>& x) -> std::vector<std::variant<std::vector<double>, double>> {
    std::vector<std::variant<std::vector<double>, double>> result;
    result.push_back(xnewcopy(std::get<std::vector<double>>(x[0])));
    result.push_back(std::get<double>(x[1]));
    return result;
};

auto xdot_e = [](const std::vector<std::variant<std::vector<double>, double>>& x, const std::vector<std::variant<std::vector<double>, double>>& y) -> double {
    return xdot(std::get<std::vector<double>>(x[0]), std::get<std::vector<double>>(y[0])) +
           std::get<double>(x[1]) * std::get<double>(y[1]);
};

auto xaxpy_e = [](const std::vector<std::variant<std::vector<double>, double>>& x, std::vector<std::variant<std::vector<double>, double>>& y, double alpha = 1.0) {
    xaxpy(std::get<std::vector<double>>(x[0]), std::get<std::vector<double>>(y[0]), alpha);
    std::get<double>(y[1]) += alpha * std::get<double>(x[1]);
};

auto xscal_e = [](double alpha, std::vector<std::variant<std::vector<double>, double>>& x) {
    xscal(alpha, std::get<std::vector<double>>(x[0]));
    std::get<double>(x[1]) *= alpha;
};

auto sol = cpl(c, F_e, G_e, h, dims, A_e, b, kktsolver_e, xnewcopy_e,
         xdot_e, xaxpy_e, xscal_e, options);

sol["x"] = std::get<std::vector<double>>(sol["x"][0]);
std::vector<double> sol_znl_slice(sol["znl"].begin() + 1, sol["znl"].end());
std::vector<double> sol_snl_slice(sol["snl"].begin() + 1, sol["snl"].end());
sol["znl"] = sol_znl_slice;
sol["snl"] = sol_snl_slice;

return sol;

