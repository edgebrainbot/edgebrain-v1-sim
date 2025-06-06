
#include <cmath>
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <iomanip>
#include <memory>

// Assuming these header files exist for the required functions
#include "blas.h"
#include "misc.h"
#include "matrix.h"
#include "conelp.hpp"

// Forward declarations for matrix and vector types
class Matrix;
class SparseMatrix;

// BLAS-like operations interface
namespace blas {
    double dot(const Matrix& x, const Matrix& y);
    void axpy(const Matrix& x, Matrix& y, double alpha = 1.0);
    void scal(double alpha, Matrix& x);
    void copy(const Matrix& x, Matrix& y);
}

// Miscellaneous mathematical operations interface
namespace misc {
    double snrm2(const Matrix& x, const std::map<std::string, std::vector<int>>& dims);
    double sdot(const Matrix& x, const Matrix& y, const std::map<std::string, std::vector<int>>& dims);
    double max_step(const Matrix& x, const std::map<std::string, std::vector<int>>& dims);
    void symm(Matrix& x, int m, int offset);
    void sgemv(const Matrix& A, const Matrix& x, Matrix& y,
               const std::map<std::string, std::vector<int>>& dims,
               char trans = 'N', double alpha = 1.0, double beta = 0.0);
    std::map<std::string, std::vector<Matrix>> compute_scaling(
        const Matrix& s, const Matrix& z, Matrix& lmbda,
        const std::map<std::string, std::vector<int>>& dims, int mnl = 0);
    void scale(Matrix& x, const std::map<std::string, std::vector<Matrix>>& W,
               char inverse = 'N', char trans = 'N');
    void ssqr(Matrix& lmbdasq, const Matrix& lmbda,
              const std::map<std::string, std::vector<int>>& dims);
    void sprod(Matrix& x, const Matrix& lmbda,
               const std::map<std::string, std::vector<int>>& dims,
               char diag = 'N');

    // KKT solver factories
    std::function<std::function<void(Matrix&, Matrix&, Matrix&)>(
        const std::map<std::string, std::vector<Matrix>>&)>
    kkt_ldl(const Matrix& G, const std::map<std::string, std::vector<int>>& dims,
            const Matrix& A, double kktreg = 0.0);

    std::function<std::function<void(Matrix&, Matrix&, Matrix&)>(
        const std::map<std::string, std::vector<Matrix>>&)>
    kkt_ldl2(const Matrix& G, const std::map<std::string, std::vector<int>>& dims,
             const Matrix& A);

    std::function<std::function<void(Matrix&, Matrix&, Matrix&)>(
        const std::map<std::string, std::vector<Matrix>>&)>
    kkt_qr(const Matrix& G, const std::map<std::string, std::vector<int>>& dims,
           const Matrix& A);

    std::function<std::function<void(Matrix&, Matrix&, Matrix&)>(
        const std::map<std::string, std::vector<Matrix>>&)>
    kkt_chol(const Matrix& G, const std::map<std::string, std::vector<int>>& dims,
             const Matrix& A);

    std::function<std::function<void(Matrix&, Matrix&, Matrix&)>(
        const std::map<std::string, std::vector<Matrix>>&)>
    kkt_chol2(const Matrix& G, const std::map<std::string, std::vector<int>>& dims,
              const Matrix& A);
}

// Base operations interface
namespace base {
    void gemv(const Matrix& A, const Matrix& x, Matrix& y,
              char trans = 'N', double alpha = 1.0, double beta = 0.0);
}

// Matrix class implementation (simplified interface)
class Matrix {
public:
    std::vector<double> data;
    int rows, cols;
    char typecode;

    Matrix(double val, int r, int c) : rows(r), cols(c), typecode('d') {
        data.resize(r * c, val);
    }

    Matrix(int r, int c) : rows(r), cols(c), typecode('d') {
        data.resize(r * c, 0.0);
    }

    std::pair<int, int> size() const { return {rows, cols}; }

    double& operator[](int idx) { return data[idx]; }
    const double& operator[](int idx) const { return data[idx]; }

    // Slice assignment operator
    void slice_assign(int start, int end, int step, double val) {
        for (int i = start; i < end; i += step) {
            data[i] += val;
        }
    }

    void slice_assign_range(int start, int end, double val) {
        for (int i = start; i < end; i++) {
            data[i] += val;
        }
    }
};

// SparseMatrix class (simplified)
class SparseMatrix {
public:
    std::vector<int> rows, cols;
    std::vector<double> values;
    int nrows, ncols;
    char typecode;

    SparseMatrix(const std::vector<int>& r, const std::vector<int>& c,
                 const std::vector<double>& v, int nr, int nc)
        : rows(r), cols(c), values(v), nrows(nr), ncols(nc), typecode('d') {}

    std::pair<int, int> size() const { return {nrows, ncols}; }
};

// Function type definitions
typedef std::function<Matrix(const Matrix&)> MatrixNewCopyFunc;
typedef std::function<double(const Matrix&, const Matrix&)> MatrixDotFunc;
typedef std::function<void(const Matrix&, Matrix&, double)> MatrixAxpyFunc;
typedef std::function<void(double, Matrix&)> MatrixScalFunc;
typedef std::function<void(const Matrix&, Matrix&, char, double, double)> MatrixFunc;

// Global constants
const int EXPON = 3;
const double STEP = 0.99;

// Main cone programming function
std::map<std::string, std::variant<Matrix*, double, std::string, int>> coneprog_solver(
    const Matrix& c, const Matrix& h, const Matrix& G, const Matrix& A, const Matrix& b,
    const std::map<std::string, std::vector<int>>& dims,
    const std::map<std::string, std::variant<double, int, bool, std::string>>& options,
    const std::map<std::string, Matrix>* primalstart = nullptr,
    const std::map<std::string, Matrix>* dualstart = nullptr,
    std::variant<std::string, std::function<std::function<void(Matrix&, Matrix&, Matrix&)>(
        const std::map<std::string, std::vector<Matrix>>&)>> kktsolver = std::string("chol2"),
    MatrixNewCopyFunc xnewcopy = nullptr,
    MatrixDotFunc xdot = nullptr,
    MatrixAxpyFunc xaxpy = nullptr,
    MatrixScalFunc xscal = nullptr,
    MatrixNewCopyFunc ynewcopy = nullptr,
    MatrixDotFunc ydot = nullptr,
    MatrixAxpyFunc yaxpy = nullptr,
    MatrixScalFunc yscal = nullptr
) {

    // Parse options
    bool DEBUG = false;
    if (options.find("debug") != options.end()) {
        DEBUG = std::get<bool>(options.at("debug"));
    }

    double KKTREG = 0.0;
    bool KKTREG_set = false;
    if (options.find("kktreg") != options.end()) {
        KKTREG = std::get<double>(options.at("kktreg"));
        KKTREG_set = true;
        if (KKTREG < 0.0) {
            throw std::invalid_argument("options['kktreg'] must be a nonnegative scalar");
        }
    }

    int MAXITERS = 100;
    if (options.find("maxiters") != options.end()) {
        MAXITERS = std::get<int>(options.at("maxiters"));
        if (MAXITERS < 1) {
            throw std::invalid_argument("options['maxiters'] must be a positive integer");
        }
    }

    double ABSTOL = 1e-7;
    if (options.find("abstol") != options.end()) {
        ABSTOL = std::get<double>(options.at("abstol"));
    }

    double RELTOL = 1e-6;
    if (options.find("reltol") != options.end()) {
        RELTOL = std::get<double>(options.at("reltol"));
    }

    if (RELTOL <= 0.0 && ABSTOL <= 0.0) {
        throw std::invalid_argument("at least one of options['reltol'] and "
                                   "options['abstol'] must be positive");
    }

    double FEASTOL = 1e-7;
    if (options.find("feastol") != options.end()) {
        FEASTOL = std::get<double>(options.at("feastol"));
        if (FEASTOL <= 0.0) {
            throw std::invalid_argument("options['feastol'] must be a positive scalar");
        }
    }

    bool show_progress = true;
    if (options.find("show_progress") != options.end()) {
        show_progress = std::get<bool>(options.at("show_progress"));
    }

    // Set default kktsolver
    std::string kktsolver_str;
    std::function<std::function<void(Matrix&, Matrix&, Matrix&)>(
        const std::map<std::string, std::vector<Matrix>>&)> kktsolver_func;
    bool kktsolver_is_string = false;

    if (std::holds_alternative<std::string>(kktsolver)) {
        kktsolver_str = std::get<std::string>(kktsolver);
        kktsolver_is_string = true;
        if (kktsolver_str.empty()) {
            if (!dims.at("q").empty() || !dims.at("s").empty()) {
                kktsolver_str = "qr";
            } else {
                kktsolver_str = "chol2";
            }
        }
    } else {
        kktsolver_func = std::get<std::function<std::function<void(Matrix&, Matrix&, Matrix&)>(
            const std::map<std::string, std::vector<Matrix>>&)>>(kktsolver);
    }

    std::vector<std::string> defaultsolvers = {"ldl", "ldl2", "qr", "chol", "chol2"};
    if (kktsolver_is_string &&
        std::find(defaultsolvers.begin(), defaultsolvers.end(), kktsolver_str) == defaultsolvers.end()) {
        throw std::invalid_argument("'" + kktsolver_str + "' is not a valid value for kktsolver");
    }

    // Argument error checking depends on level of customization
    bool customkkt = !kktsolver_is_string;
    bool matrixG = true; // Assuming G is always a matrix in this implementation
    bool matrixA = true; // Assuming A is always a matrix in this implementation

    bool customx = (xnewcopy != nullptr || xdot != nullptr || xaxpy != nullptr || xscal != nullptr);
    bool customy = (ynewcopy != nullptr || ydot != nullptr || yaxpy != nullptr || yscal != nullptr);

    if (c.typecode != 'd' || c.size().second != 1) {
        throw std::invalid_argument("'c' must be a 'd' matrix with one column");
    }

    if (h.typecode != 'd' || h.size().second != 1) {
        throw std::invalid_argument("'h' must be a 'd' matrix with 1 column");
    }

    // Validate dims
    int dims_l = dims.at("l")[0];
    if (dims_l < 0) {
        throw std::invalid_argument("'dims['l']' must be a nonnegative integer");
    }

    for (int k : dims.at("q")) {
        if (k < 1) {
            throw std::invalid_argument("'dims['q']' must be a list of positive integers");
        }
    }

    for (int k : dims.at("s")) {
        if (k < 0) {
            throw std::invalid_argument("'dims['s']' must be a list of nonnegative integers");
        }
    }

    // Set refinement
    int refinement = 0;
    if (options.find("refinement") != options.end()) {
        refinement = std::get<int>(options.at("refinement"));
        if (refinement < 0) {
            throw std::invalid_argument("options['refinement'] must be a nonnegative integer");
        }
    } else {
        if (!dims.at("q").empty() || !dims.at("s").empty()) {
            refinement = 1;
        } else {
            refinement = 0;
        }
    }

    // Calculate dimensions
    int cdim = dims_l;
    for (int q : dims.at("q")) cdim += q;
    for (int s : dims.at("s")) cdim += s * s;

    int cdim_pckd = dims_l;
    for (int q : dims.at("q")) cdim_pckd += q;
    for (int s : dims.at("s")) cdim_pckd += s * (s + 1) / 2;

    int cdim_diag = dims_l;
    for (int q : dims.at("q")) cdim_diag += q;
    for (int s : dims.at("s")) cdim_diag += s;

    if (h.size().first != cdim) {
        throw std::invalid_argument("'h' must be a 'd' matrix of size (" +
                                   std::to_string(cdim) + ",1)");
    }

    // Data for kth 'q' constraint are found in rows indq[k]:indq[k+1] of G
    std::vector<int> indq = {dims_l};
    for (int k : dims.at("q")) {
        indq.push_back(indq.back() + k);
    }

    // Data for kth 's' constraint are found in rows inds[k]:inds[k+1] of G
    std::vector<int> inds = {indq.back()};
    for (int k : dims.at("s")) {
        inds.push_back(inds.back() + k * k);
    }

    if (G.typecode != 'd' || G.size().first != cdim || G.size().second != c.size().first) {
        throw std::invalid_argument("'G' must be a 'd' matrix of size (" +
                                   std::to_string(cdim) + ", " +
                                   std::to_string(c.size().first) + ")");
    }

    // Define Gf function
    auto Gf = [&G, &dims](const Matrix& x, Matrix& y, char trans = 'N',
                          double alpha = 1.0, double beta = 0.0) {
        misc::sgemv(G, x, y, dims, trans, alpha, beta);
    };

    // Handle A matrix
    auto Af = [&A](const Matrix& x, Matrix& y, char trans = 'N',
                   double alpha = 1.0, double beta = 0.0) {
        base::gemv(A, x, y, trans, alpha, beta);
    };

    if (A.typecode != 'd' || A.size().second != c.size().first) {
        throw std::invalid_argument("'A' must be a 'd' matrix with " +
                                   std::to_string(c.size().first) + " columns");
    }

    if (b.typecode != 'd' || b.size().second != 1) {
        throw std::invalid_argument("'b' must be a 'd' matrix with one column");
    }

    if (b.size().first != A.size().first) {
        throw std::invalid_argument("'b' must have length " + std::to_string(A.size().first));
    }

    // kktsolver(W) returns a routine for solving 3x3 block KKT system
    std::function<std::function<void(Matrix&, Matrix&, Matrix&)>(
        const std::map<std::string, std::vector<Matrix>>&)> factor;

    if (kktsolver_is_string) {
        if (!KKTREG_set && (b.size().first > c.size().first ||
                           b.size().first + cdim_pckd < c.size().first)) {
            throw std::invalid_argument("Rank(A) < p or Rank([G; A]) < n");
        }

        if (kktsolver_str == "ldl") {
            factor = misc::kkt_ldl(G, dims, A, KKTREG);
        } else if (kktsolver_str == "ldl2") {
            factor = misc::kkt_ldl2(G, dims, A);
        } else if (kktsolver_str == "qr") {
            factor = misc::kkt_qr(G, dims, A);
        } else if (kktsolver_str == "chol") {
            factor = misc::kkt_chol(G, dims, A);
        } else {
            factor = misc::kkt_chol2(G, dims, A);
        }
    } else {
        factor = kktsolver_func;
    }

    auto kktsolver_final = [factor](const std::map<std::string, std::vector<Matrix>>& W) {
        return factor(W);
    };

    // res() evaluates residual in 5x5 block KKT system
    Matrix ws3(0.0, cdim, 1), wz3(0.0, cdim, 1);

    auto res = [&](const Matrix& ux, const Matrix& uy, const Matrix& uz, const Matrix& utau,
                   const Matrix& us, const Matrix& ukappa, Matrix& vx, Matrix& vy, Matrix& vz,
                   Matrix& vtau, Matrix& vs, Matrix& vkappa,
                   const std::map<std::string, std::vector<Matrix>>& W,
                   double dg, const Matrix& lmbda) {

        // vx := vx - A'*uy - G'*W^{-1}*uz - c*utau/dg
        Af(uy, vx, 'T', -1.0, 1.0);
        blas::copy(uz, wz3);
        misc::scale(wz3, W, 'I');
        Gf(wz3, vx, 'T', -1.0, 1.0);
        if (xaxpy) {
            xaxpy(c, vx, -utau[0]/dg);
        } else {
            blas::axpy(c, vx, -utau[0]/dg);
        }

        // vy := vy + A*ux - b*utau/dg
        Af(ux, vy, 'N', 1.0, 1.0);
        if (yaxpy) {
            yaxpy(b, vy, -utau[0]/dg);
        } else {
            blas::axpy(b, vy, -utau[0]/dg);
        }

        // vz := vz + G*ux - h*utau/dg + W'*us
        Gf(ux, vz, 'N', 1.0, 1.0);
        blas::axpy(h, vz, -utau[0]/dg);
        blas::copy(us, ws3);
        misc::scale(ws3, W, 'N', 'T');
        blas::axpy(ws3, vz);

        // vtau := vtau + c'*ux + b'*uy + h'*W^{-1}*uz + dg*ukappa
        double cx = (xdot) ? xdot(c, ux) : blas::dot(c, ux);
        double by = (ydot) ? ydot(b, uy) : blas::dot(b, uy);
        double hz = misc::sdot(h, wz3, dims);
        vtau[0] += dg*ukappa[0] + cx + by + hz;

        // vs := vs + lmbda o (uz + us)
        blas::copy(us, ws3);
        blas::axpy(uz, ws3);
        misc::sprod(ws3, lmbda, dims, 'D');
        blas::axpy(ws3, vs);

        // vkappa += vkappa + lmbdag * (utau + ukappa)
        vkappa[0] += lmbda[lmbda.size().first - 1] * (utau[0] + ukappa[0]);
    };

    // Set default functions if not provided
    auto xcopy = [&](const Matrix& x, Matrix& y) {
        if (xscal) xscal(0.0, y);
        else blas::scal(0.0, y);
        if (xaxpy) xaxpy(x, y);
        else blas::axpy(x, y);
    };

    auto ycopy = [&](const Matrix& x, Matrix& y) {
        if (yscal) yscal(0.0, y);
        else blas::scal(0.0, y);
        if (yaxpy) yaxpy(x, y);
        else blas::axpy(x, y);
    };

    double resx0 = std::max(1.0, std::sqrt((xdot) ? xdot(c, c) : blas::dot(c, c)));
    double resy0 = std::max(1.0, std::sqrt((ydot) ? ydot(b, b) : blas::dot(b, b)));
    double resz0 = std::max(1.0, misc::snrm2(h, dims));

    // Select initial points
    Matrix x = (xnewcopy) ? xnewcopy(c) : Matrix(c);
    if (xscal) xscal(0.0, x);
    else blas::scal(0.0, x);

    Matrix y = (ynewcopy) ? ynewcopy(b) : Matrix(b);
    if (yscal) yscal(0.0, y);
    else blas::scal(0.0, y);

    Matrix s(0.0, cdim, 1), z(0.0, cdim, 1);
    Matrix dx = (xnewcopy) ? xnewcopy(c) : Matrix(c);
    Matrix dy = (ynewcopy) ? ynewcopy(b) : Matrix(b);
    Matrix ds(0.0, cdim, 1), dz(0.0, cdim, 1);
    Matrix dkappa(0.0, 1, 1), dtau(0.0, 1, 1);

    std::function<void(Matrix&, Matrix&, Matrix&)> f;

    if (primalstart == nullptr || dualstart == nullptr) {
        // Factor
        // [ 0   A'  G' ]
        // [ A   0   0  ]
        // [ G   0  -I  ]

        std::map<std::string, std::vector<Matrix>> W;
        W["d"] = {Matrix(1.0, dims_l, 1)};
        W["di"] = {Matrix(1.0, dims_l, 1)};

        std::vector<Matrix> v_list;
        for (int m : dims.at("q")) {
            Matrix v(0.0, m, 1);
            v[0] = 1.0;
            v_list.push_back(v);
        }
        W["v"] = v_list;

        std::vector<double> beta_list(dims.at("q").size(), 1.0);
        // Note: beta is stored separately as it's not a matrix

        std::vector<Matrix> r_list, rti_list;
        for (int m : dims.at("s")) {
            Matrix r(0.0, m, m);
            Matrix rti(0.0, m, m);
            for (int i = 0; i < m; i++) {
                r[i * m + i] = 1.0;
                rti[i * m + i] = 1.0;
            }
            r_list.push_back(r);
            rti_list.push_back(rti);
        }
        W["r"] = r_list;
        W["rti"] = rti_list;

        try {
            f = kktsolver_final(W);
        } catch (const std::exception&) {
            throw std::invalid_argument("Rank(A) < p or Rank([G; A]) < n");
        }
    }

    if (primalstart == nullptr) {
        // minimize    || G * x - h ||^2
        // subject to  A * x = b
        //
        // by solving
        //
        //     [ 0   A'  G' ]   [ x  ]   [ 0 ]
        //     [ A   0   0  ] * [ dy ] = [ b ]
        //     [ G   0  -I  ]   [ -s ]   [ h ]

        if (xscal) xscal(0.0, x);
        else blas::scal(0.0, x);
        ycopy(b, dy);
        blas::copy(h, s);

        try {
            f(x, dy, s);
        } catch (const std::exception&) {
            throw std::invalid_argument("Rank(A) < p or Rank([G; A]) < n");
        }
        blas::scal(-1.0, s);
    } else {
        xcopy(primalstart->at("x"), x);
        blas::copy(primalstart->at("s"), s);
    }

    // ts = min{ t | s + t*e >= 0 }
    double ts = misc::max_step(s, dims);
    if (ts >= 0 && primalstart) {
        throw std::invalid_argument("initial s is not positive");
    }

    if (dualstart == nullptr) {
        // minimize   || z ||^2
        // subject to G'*z + A'*y + c = 0
        //
        // by solving
        //
        //     [ 0   A'  G' ] [ dx ]   [ -c ]
        //     [ A   0   0  ] [ y  ] = [  0 ]
        //     [ G   0  -I  ] [ z  ]   [  0 ]

        xcopy(c, dx);
        if (xscal) xscal(-1.0, dx);
        else blas::scal(-1.0, dx);
        if (yscal) yscal(0.0, y);
        else blas::scal(0.0, y);
        blas::scal(0.0, z);

        try {
            f(dx, y, z);
        } catch (const std::exception&) {
            throw std::invalid_argument("Rank(A) < p or Rank([G; A]) < n");
        }
    } else {
        if (dualstart->find("y") != dualstart->end()) {
            ycopy(dualstart->at("y"), y);
        }
        blas::copy(dualstart->at("z"), z);
    }

    // tz = min{ t | z + t*e >= 0 }
    double tz = misc::max_step(z, dims);
    if (tz >= 0 && dualstart) {
        throw std::invalid_argument("initial z is not positive");
    }

    double nrms = misc::snrm2(s, dims);
    double nrmz = misc::snrm2(z, dims);

    if (primalstart == nullptr && dualstart == nullptr) {
        double gap = misc::sdot(s, z, dims);
        double pcost = (xdot) ? xdot(c, x) : blas::dot(c, x);
        double dcost = -((ydot) ? ydot(b, y) : blas::dot(b, y)) - misc::sdot(h, z, dims);

        double relgap;
        bool relgap_valid = false;
        if (pcost < 0.0) {
            relgap = gap / -pcost;
            relgap_valid = true;
        } else if (dcost > 0.0) {
            relgap = gap / dcost;
            relgap_valid = true;
        }

        if ((ts <= 0 && tz <= 0 && (gap <= ABSTOL ||
            (relgap_valid && relgap <= RELTOL))) && !KKTREG_set) {

            // The initial points we constructed happen to be feasible and optimal
            int ind = dims_l;
            for (int q : dims.at("q")) ind += q;
            for (int m : dims.at("s")) {
                misc::symm(s, m, ind);
                misc::symm(z, m, ind);
                ind += m * m;
            }

            // rx = A'*y + G'*z + c
            Matrix rx = (xnewcopy) ? xnewcopy(c) : Matrix(c);
            Af(y, rx, 'T', 1.0, 1.0);
            Gf(z, rx, 'T', 1.0, 1.0);
            double resx = std::sqrt((xdot) ? xdot(rx, rx) : blas::dot(rx, rx));

            // ry = b - A*x
            Matrix ry = (ynewcopy) ? ynewcopy(b) : Matrix(b);
            Af(x, ry, 'N', -1.0, 1.0);
            double resy = std::sqrt((ydot) ? ydot(ry, ry) : blas::dot(ry, ry));

            // rz = s + G*x - h
            Matrix rz(0.0, cdim, 1);
            Gf(x, rz);
            blas::axpy(s, rz);
            blas::axpy(h, rz, -1.0);
            double resz = misc::snrm2(rz, dims);

            double pres = std::max(resy/resy0, resz/resz0);
            double dres = resx/resx0;
            double cx = (xdot) ? xdot(c, x) : blas::dot(c, x);
            double by = (ydot) ? ydot(b, y) : blas::dot(b, y);
            double hz = misc::sdot(h, z, dims);

            if (show_progress) {
                std::cout << "Optimal solution found." << std::endl;
            }

            std::map<std::string, std::variant<Matrix*, double, std::string, int>> result;
            result["x"] = new Matrix(x);
            result["y"] = new Matrix(y);
            result["s"] = new Matrix(s);
            result["z"] = new Matrix(z);
            result["status"] = std::string("optimal");
            result["gap"] = gap;
            result["relative gap"] = relgap_valid ? relgap : 0.0;
            result["primal objective"] = cx;
            result["dual objective"] = -(by + hz);
            result["primal infeasibility"] = pres;
            result["primal slack"] = -ts;
            result["dual slack"] = -tz;
            result["dual infeasibility"] = dres;
            result["iterations"] = 0;
            return result;
        }

        if (ts >= -1e-8 * std::max(nrms, 1.0)) {
            double a = 1.0 + ts;
            s.slice_assign_range(0, dims_l, a);
            for (int i = 0; i < indq.size() - 1; i++) {
                s[indq[i]] += a;
            }
            ind = dims_l;
            for (int q : dims.at("q")) ind += q;
            for (int m : dims.at("s")) {
                s.slice_assign(ind, ind + m*m, m+1, a);
                ind += m * m;
            }
        }

        if (tz >= -1e-8 * std::max(nrmz, 1.0)) {
            double a = 1.0 + tz;
            z.slice_assign_range(0, dims_l, a);
            for (int i = 0; i < indq.size() - 1; i++) {
                z[indq[i]] += a;
            }
            ind = dims_l;
            for (int q : dims.at("q")) ind += q;
            for (int m : dims.at("s")) {
                z.slice_assign(ind, ind + m*m, m+1, a);
                ind += m * m;
            }
        }
    } else if (primalstart == nullptr && dualstart != nullptr) {
        if (ts >= -1e-8 * std::max(nrms, 1.0)) {
            double a = 1.0 + ts;
            s.slice_assign_range(0, dims_l, a);
            for (int i = 0; i < indq.size() - 1; i++) {
                s[indq[i]] += a;
            }
            int ind = dims_l;
            for (int q : dims.at("q")) ind += q;
            for (int m : dims.at("s")) {
                s.slice_assign(ind, ind + m*m, m+1, a);
                ind += m * m;
            }
        }
    } else if (primalstart != nullptr && dualstart == nullptr) {
        if (tz >= -1e-8 * std::max(nrmz, 1.0)) {
            double a = 1.0 + tz;
            z.slice_assign_range(0, dims_l, a);
            for (int i = 0; i < indq.size() - 1; i++) {
                z[indq[i]] += a;
            }
            int ind = dims_l;
            for (int q : dims.at("q")) ind += q;
            for (int m : dims.at("s")) {
                z.slice_assign(ind, ind + m*m, m+1, a);
                ind += m * m;
            }
        }
    }

    double tau = 1.0, kappa = 1.0;

    Matrix rx = (xnewcopy) ? xnewcopy(c) : Matrix(c);
    Matrix hrx = (xnewcopy) ? xnewcopy(c) : Matrix(c);
    Matrix ry = (ynewcopy) ? ynewcopy(b) : Matrix(b);
    Matrix hry = (ynewcopy) ? ynewcopy(b) : Matrix(b);
    Matrix rz(0.0, cdim, 1), hrz(0.0, cdim, 1);
    Matrix sigs(0.0, std::accumulate(dims.at("s").begin(), dims.at("s").end(), 0), 1);
    Matrix sigz(0.0, std::accumulate(dims.at("s").begin(), dims.at("s").end(), 0), 1);
    Matrix lmbda(0.0, cdim_diag + 1, 1);
    Matrix lmbdasq(0.0, cdim_diag + 1, 1);

    double gap = misc::sdot(s, z, dims);

    for (int iters = 0; iters <= MAXITERS; iters++) {
        // hrx = -A'*y - G'*z
        Af(y, hrx, 'T', -1.0, 0.0);
        Gf(z, hrx, 'T', -1.0, 1.0);
        double hresx = std::sqrt((xdot) ? xdot(hrx, hrx) : blas::dot(hrx, hrx));

        // rx = hrx - c*tau = -A'*y - G'*z - c*tau
        xcopy(hrx, rx);
        if (xaxpy) xaxpy(c, rx, -tau);
        else blas::axpy(c, rx, -tau);
        double resx = std::sqrt((xdot) ? xdot(rx, rx) : blas::dot(rx, rx)) / tau;

        // hry = A*x
        Af(x, hry);
        double hresy = std::sqrt((ydot) ? ydot(hry, hry) : blas::dot(hry, hry));

        // ry = hry - b*tau = A*x - b*tau
        ycopy(hry, ry);
        if (yaxpy) yaxpy(b, ry, -tau);
        else blas::axpy(b, ry, -tau);
        double resy = std::sqrt((ydot) ? ydot(ry, ry) : blas::dot(ry, ry)) / tau;

        // hrz = s + G*x
        Gf(x, hrz);
        blas::axpy(s, hrz);
        double hresz = misc::snrm2(hrz, dims);

        // rz = hrz - h*tau = s + G*x - h*tau
        blas::scal(0.0, rz);
        blas::axpy(hrz, rz);
        blas::axpy(h, rz, -tau);
        double resz = misc::snrm2(rz, dims) / tau;

        // rt = kappa + c'*x + b'*y + h'*z
        double cx = (xdot) ? xdot(c, x) : blas::dot(c, x);
        double by = (ydot) ? ydot(b, y) : blas::dot(b, y);
        double hz = misc::sdot(h, z, dims);
        double rt = kappa + cx + by + hz;

        // Statistics for stopping criteria
        double pcost = cx / tau;
        double dcost = -(by + hz) / tau;

        double relgap;
        bool relgap_valid = false;
        if (pcost < 0.0) {
            relgap = gap / -pcost;
            relgap_valid = true;
        } else if (dcost > 0.0) {
            relgap = gap / dcost;
            relgap_valid = true;
        }

        double pres = std::max(resy/resy0, resz/resz0);
        double dres = resx/resx0;

        double pinfres;
        bool pinfres_valid = false;
        if (hz + by < 0.0) {
            pinfres = hresx / resx0 / (-hz - by);
            pinfres_valid = true;
        }

        double dinfres;
        bool dinfres_valid = false;
        if (cx < 0.0) {
            dinfres = std::max(hresy / resy0, hresz/resz0) / (-cx);
            dinfres_valid = true;
        }

        if (show_progress) {
            if (iters == 0) {
                std::cout << std::setw(10) << "pcost" << std::setw(12) << "dcost"
                         << std::setw(10) << "gap" << std::setw(8) << "pres"
                         << std::setw(7) << "dres" << std::setw(6) << "k/t" << std::endl;
            }
            std::cout << std::setw(2) << iters << ": "
                     << std::scientific << std::setprecision(4)
                     << std::setw(8) << pcost << " " << std::setw(8) << dcost << " "
                     << std::setw(4) << std::setprecision(0) << gap << " "
                     << std::setw(7) << std::setprecision(0) << pres << " "
                     << std::setw(7) << std::setprecision(0) << dres << " "
                     << std::setw(7) << std::setprecision(0) << kappa/tau << std::endl;
        }

        if ((pres <= FEASTOL && dres <= FEASTOL && (gap <= ABSTOL ||
            (relgap_valid && relgap <= RELTOL))) || iters == MAXITERS) {

            if (xscal) xscal(1.0/tau, x);
            else blas::scal(1.0/tau, x);
            if (yscal) yscal(1.0/tau, y);
            else blas::scal(1.0/tau, y);
            blas::scal(1.0/tau, s);
            blas::scal(1.0/tau, z);

            int ind = dims_l;
            for (int q : dims.at("q")) ind += q;
            for (int m : dims.at("s")) {
                misc::symm(s, m, ind);
                misc::symm(z, m, ind);
                ind += m * m;
            }
            ts = misc::max_step(s, dims);
            tz = misc::max_step(z, dims);

            std::map<std::string, std::variant<Matrix*, double, std::string, int>> result;
            result["x"] = new Matrix(x);
            result["y"] = new Matrix(y);
            result["s"] = new Matrix(s);
            result["z"] = new Matrix(z);

            if (iters == MAXITERS) {
                if (show_progress) {
                    std::cout << "Terminated (maximum number of iterations reached)." << std::endl;
                }
                result["status"] = std::string("unknown");
            } else {
                if (show_progress) {
                    std::cout << "Optimal solution found." << std::endl;
                }
                result["status"] = std::string("optimal");
            }

            result["gap"] = gap;
            result["relative gap"] = relgap_valid ? relgap : 0.0;
            result["primal objective"] = pcost;
            result["dual objective"] = dcost;
            result["primal infeasibility"] = pres;
            result["dual infeasibility"] = dres;
            result["primal slack"] = -ts;
            result["dual slack"] = -tz;
            result["iterations"] = iters;
            return result;
        }

        else if (pinfres_valid && pinfres <= FEASTOL) {
            if (yscal) yscal(1.0/(-hz - by), y);
            else blas::scal(1.0/(-hz - by), y);
            blas::scal(1.0/(-hz - by), z);

            int ind = dims_l;
            for (int q : dims.at("q")) ind += q;
            for (int m : dims.at("s")) {
                misc::symm(z, m, ind);
                ind += m * m;
            }
            tz = misc::max_step(z, dims);

            if (show_progress) {
                std::cout << "Certificate of primal infeasibility found." << std::endl;
            }

            std::map<std::string, std::variant<Matrix*, double, std::string, int>> result;
            result["x"] = static_cast<Matrix*>(nullptr);
            result["y"] = new Matrix(y);
            result["s"] = static_cast<Matrix*>(nullptr);
            result["z"] = new Matrix(z);
            result["status"] = std::string("primal infeasible");
            result["dual objective"] = 1.0;
            result["dual slack"] = -tz;
            result["iterations"] = iters;
            return result;
        }

        else if (dinfres_valid && dinfres <= FEASTOL) {
            if (xscal) xscal(1.0/(-cx), x);
            else blas::scal(1.0/(-cx), x);
            blas::scal(1.0/(-cx), s);

            int


// Structure to hold return values (equivalent to Python dictionary)
struct OptimizationResult {
    std::shared_ptr<Matrix> x;
    std::shared_ptr<Matrix> y;
    std::shared_ptr<Matrix> s;
    std::shared_ptr<Matrix> z;
    std::string status;
    double gap;
    std::shared_ptr<double> relative_gap; // Using pointer to represent None
    double primal_objective;
    double dual_objective;
    double primal_infeasibility;
    double dual_infeasibility;
    double primal_slack;
    double dual_slack;
    std::shared_ptr<double> primal_infeasibility_certificate;
    std::shared_ptr<double> dual_infeasibility_certificate;
    int iterations;
};

// Main optimization loop function
OptimizationResult optimization_loop(
    int MAXITERS, double FEASTOL, double ABSTOL, double RELTOL, double STEP, double EXPON,
    bool show_progress, bool DEBUG, bool refinement, bool primalstart, bool dualstart,
    std::function<void(const Matrix&, Matrix&, double, double, const char*)> Af,
    std::function<void(const Matrix&, Matrix&, double, double, const char*)> Gf,
    std::function<double(const Matrix&, const Matrix&)> xdot,
    std::function<double(const Matrix&, const Matrix&)> ydot,
    std::function<void(const Matrix&, Matrix&)> xcopy,
    std::function<void(const Matrix&, Matrix&)> ycopy,
    std::function<void(const Matrix&, Matrix&, double)> xaxpy,
    std::function<void(const Matrix&, Matrix&, double)> yaxpy,
    std::function<void(double, Matrix&)> xscal,
    std::function<void(double, Matrix&)> yscal,
    std::function<Matrix()> xnewcopy,
    std::function<Matrix()> ynewcopy,
    std::function<std::function<void(Matrix&, Matrix&, Matrix&)>(const Matrix&)> kktsolver,
    std::function<void(const Matrix&, const Matrix&, const Matrix&, const Matrix&, const Matrix&, const Matrix&, Matrix&, Matrix&, Matrix&, Matrix&, Matrix&, Matrix&, const Matrix&, double, const Matrix&)> res,
    Matrix& hrx, Matrix& rx, Matrix& hry, Matrix& ry, Matrix& hrz, Matrix& rz,
    Matrix& c, Matrix& b, Matrix& h, Matrix& x, Matrix& y, Matrix& s, Matrix& z,
    double& tau, double& kappa, double& gap,
    double resx0, double resy0, double resz0,
    const std::map<std::string, int>& dims, int cdim, int cdim_diag,
    Matrix& lmbda, Matrix& lmbdasq, Matrix& indq,
    Matrix& dx, Matrix& dy, Matrix& dz, Matrix& ds, Matrix& dtau, Matrix& dkappa,
    Matrix& sigs, Matrix& sigz, Matrix& ws3, double& wkappa3) {

    for (int iters = 0; iters <= MAXITERS; ++iters) {

        // hrx = -A'*y - G'*z
        Af(y, hrx, -1.0, 0.0, "T");
        Gf(z, hrx, -1.0, 1.0, "T");
        double hresx = std::sqrt(xdot(hrx, hrx));

        // rx = hrx - c*tau
        //    = -A'*y - G'*z - c*tau
        xcopy(hrx, rx);
        xaxpy(c, rx, -tau);
        double resx = std::sqrt(xdot(rx, rx)) / tau;

        // hry = A*x
        Af(x, hry, 1.0, 0.0, "");
        double hresy = std::sqrt(ydot(hry, hry));

        // ry = hry - b*tau
        //    = A*x - b*tau
        ycopy(hry, ry);
        yaxpy(b, ry, -tau);
        double resy = std::sqrt(ydot(ry, ry)) / tau;

        // hrz = s + G*x
        Gf(x, hrz, 1.0, 0.0, "");
        blas::axpy(s, hrz);
        double hresz = misc::snrm2(hrz, dims);

        // rz = hrz - h*tau
        //    = s + G*x - h*tau
        blas::scal(0.0, rz);
        blas::axpy(hrz, rz);
        blas::axpy(h, rz, -tau);
        double resz = misc::snrm2(rz, dims) / tau;

        // rt = kappa + c'*x + b'*y + h'*z
        double cx = xdot(c, x);
        double by = ydot(b, y);
        double hz = misc::sdot(h, z, dims);
        double rt = kappa + cx + by + hz;

        // Statistics for stopping criteria.
        double pcost = cx / tau;
        double dcost = -(by + hz) / tau;
        std::shared_ptr<double> relgap = nullptr;
        if (pcost < 0.0) {
            relgap = std::make_shared<double>(gap / (-pcost));
        } else if (dcost > 0.0) {
            relgap = std::make_shared<double>(gap / dcost);
        }

        double pres = std::max(resy/resy0, resz/resz0);
        double dres = resx/resx0;

        std::shared_ptr<double> pinfres = nullptr;
        if (hz + by < 0.0) {
            pinfres = std::make_shared<double>(hresx / resx0 / (-hz - by));
        }

        std::shared_ptr<double> dinfres = nullptr;
        if (cx < 0.0) {
            dinfres = std::make_shared<double>(std::max(hresy / resy0, hresz/resz0) / (-cx));
        }

        if (show_progress) {
            if (iters == 0) {
                std::cout << std::setw(10) << "pcost" << std::setw(12) << "dcost"
                         << std::setw(10) << "gap" << std::setw(8) << "pres"
                         << std::setw(7) << "dres" << std::setw(6) << "k/t" << std::endl;
            }
            std::cout << std::setw(2) << iters << ": "
                     << std::scientific << std::setprecision(4) << std::setw(8) << pcost << " "
                     << std::scientific << std::setprecision(4) << std::setw(8) << dcost << " "
                     << std::scientific << std::setprecision(0) << std::setw(4) << gap
                     << std::scientific << std::setprecision(0) << std::setw(7) << pres
                     << std::scientific << std::setprecision(0) << std::setw(7) << dres
                     << std::scientific << std::setprecision(0) << std::setw(7) << kappa/tau << std::endl;
        }

        if ((pres <= FEASTOL && dres <= FEASTOL && (gap <= ABSTOL ||
            (relgap != nullptr && *relgap <= RELTOL))) || iters == MAXITERS) {

            xscal(1.0/tau, x);
            yscal(1.0/tau, y);
            blas::scal(1.0/tau, s);
            blas::scal(1.0/tau, z);

            int ind = dims.at("l") + misc::sum_vector(dims.at("q"));
            for (int m : dims.at("s")) {
                misc::symm(s, m, ind);
                misc::symm(z, m, ind);
                ind += m * m;
            }

            double ts = misc::max_step(s, dims);
            double tz = misc::max_step(z, dims);

            OptimizationResult result;
            result.x = std::make_shared<Matrix>(x);
            result.y = std::make_shared<Matrix>(y);
            result.s = std::make_shared<Matrix>(s);
            result.z = std::make_shared<Matrix>(z);
            result.gap = gap;
            result.relative_gap = relgap;
            result.primal_objective = pcost;
            result.dual_objective = dcost;
            result.primal_infeasibility = pres;
            result.dual_infeasibility = dres;
            result.primal_slack = -ts;
            result.dual_slack = -tz;
            result.iterations = iters;

            if (iters == MAXITERS) {
                if (show_progress) {
                    std::cout << "Terminated (maximum number of iterations reached)." << std::endl;
                }
                result.status = "unknown";
                result.primal_infeasibility_certificate = pinfres;
                result.dual_infeasibility_certificate = dinfres;
            } else {
                if (show_progress) {
                    std::cout << "Optimal solution found." << std::endl;
                }
                result.status = "optimal";
                result.primal_infeasibility_certificate = nullptr;
                result.dual_infeasibility_certificate = nullptr;
            }

            return result;
        }
        else if (pinfres != nullptr && *pinfres <= FEASTOL) {
            yscal(1.0/(-hz - by), y);
            blas::scal(1.0/(-hz - by), z);

            int ind = dims.at("l") + misc::sum_vector(dims.at("q"));
            for (int m : dims.at("s")) {
                misc::symm(z, m, ind);
                ind += m * m;
            }

            double tz = misc::max_step(z, dims);

            if (show_progress) {
                std::cout << "Certificate of primal infeasibility found." << std::endl;
            }

            OptimizationResult result;
            result.x = nullptr;
            result.y = std::make_shared<Matrix>(y);
            result.s = nullptr;
            result.z = std::make_shared<Matrix>(z);
            result.status = "primal infeasible";
            result.gap = 0.0; // Using 0.0 instead of None
            result.relative_gap = nullptr;
            result.primal_objective = 0.0; // Using 0.0 instead of None
            result.dual_objective = 1.0;
            result.primal_infeasibility = 0.0; // Using 0.0 instead of None
            result.dual_infeasibility = 0.0; // Using 0.0 instead of None
            result.primal_slack = 0.0; // Using 0.0 instead of None
            result.dual_slack = -tz;
            result.primal_infeasibility_certificate = pinfres;
            result.dual_infeasibility_certificate = nullptr;
            result.iterations = iters;

            return result;
        }
        else if (dinfres != nullptr && *dinfres <= FEASTOL) {
            xscal(1.0/(-cx), x);
            blas::scal(1.0/(-cx), s);

            int ind = dims.at("l") + misc::sum_vector(dims.at("q"));
            for (int m : dims.at("s")) {
                misc::symm(s, m, ind);
                ind += m * m;
            }

            double ts = misc::max_step(s, dims);

            if (show_progress) {
                std::cout << "Certificate of dual infeasibility found." << std::endl;
            }

            OptimizationResult result;
            result.x = std::make_shared<Matrix>(x);
            result.y = nullptr;
            result.s = std::make_shared<Matrix>(s);
            result.z = nullptr;
            result.status = "dual infeasible";
            result.gap = 0.0; // Using 0.0 instead of None
            result.relative_gap = nullptr;
            result.primal_objective = -1.0;
            result.dual_objective = 0.0; // Using 0.0 instead of None
            result.primal_infeasibility = 0.0; // Using 0.0 instead of None
            result.dual_infeasibility = 0.0; // Using 0.0 instead of None
            result.primal_slack = -ts;
            result.dual_slack = 0.0; // Using 0.0 instead of None
            result.primal_infeasibility_certificate = nullptr;
            result.dual_infeasibility_certificate = dinfres;
            result.iterations = iters;

            return result;
        }

        // Compute initial scaling W:
        //
        //     W * z = W^{-T} * s = lambda
        //     dg * tau = 1/dg * kappa = lambdag.

        Matrix W;
        double dg, dgi;
        static Matrix x1, y1, z1, th;

        if (iters == 0) {
            W = misc::compute_scaling(s, z, lmbda, dims, 0);

            //     dg = sqrt( kappa / tau )
            //     dgi = sqrt( tau / kappa )
            //     lambda_g = sqrt( tau * kappa )
            //
            // lambda_g is stored in the last position of lmbda.

            dg = std::sqrt(kappa / tau);
            dgi = std::sqrt(tau / kappa);
            lmbda[lmbda.size() - 1] = std::sqrt(tau * kappa);
        }

        // lmbdasq := lmbda o lmbda
        misc::ssqr(lmbdasq, lmbda, dims);
        lmbdasq[lmbdasq.size() - 1] = std::pow(lmbda[lmbda.size() - 1], 2);

        // f3(x, y, z) solves
        //
        //     [ 0  A'  G'   ] [ ux        ]   [ bx ]
        //     [ A  0   0    ] [ uy        ] = [ by ].
        //     [ G  0  -W'*W ] [ W^{-1}*uz ]   [ bz ]
        //
        // On entry, x, y, z contain bx, by, bz.
        // On exit, they contain ux, uy, uz.
        //
        // Also solve
        //
        //     [ 0   A'  G'    ] [ x1        ]          [ c ]
        //     [-A   0   0     ]*[ y1        ] = -dgi * [ b ].
        //     [-G   0   W'*W  ] [ W^{-1}*z1 ]          [ h ]

        auto f3 = kktsolver(W);

        try {
            if (iters == 0) {
                x1 = xnewcopy();
                y1 = ynewcopy();
                z1 = Matrix(cdim, 1, 0.0);
            }
            xcopy(c, x1);
            xscal(-1.0, x1);
            ycopy(b, y1);
            blas::copy(h, z1);
            f3(x1, y1, z1);
            xscal(dgi, x1);
            yscal(dgi, y1);
            blas::scal(dgi, z1);
        }
        catch (const std::runtime_error& e) {
            if (iters == 0 && primalstart && dualstart) {
                throw std::runtime_error("Rank(A) < p or Rank([G; A]) < n");
            } else {
                xscal(1.0/tau, x);
                yscal(1.0/tau, y);
                blas::scal(1.0/tau, s);
                blas::scal(1.0/tau, z);

                int ind = dims.at("l") + misc::sum_vector(dims.at("q"));
                for (int m : dims.at("s")) {
                    misc::symm(s, m, ind);
                    misc::symm(z, m, ind);
                    ind += m * m;
                }

                double ts = misc::max_step(s, dims);
                double tz = misc::max_step(z, dims);

                if (show_progress) {
                    std::cout << "Terminated (singular KKT matrix)." << std::endl;
                }

                OptimizationResult result;
                result.x = std::make_shared<Matrix>(x);
                result.y = std::make_shared<Matrix>(y);
                result.s = std::make_shared<Matrix>(s);
                result.z = std::make_shared<Matrix>(z);
                result.status = "unknown";
                result.gap = gap;
                result.relative_gap = relgap;
                result.primal_objective = pcost;
                result.dual_objective = dcost;
                result.primal_infeasibility = pres;
                result.dual_infeasibility = dres;
                result.primal_slack = -ts;
                result.dual_slack = -tz;
                result.primal_infeasibility_certificate = pinfres;
                result.dual_infeasibility_certificate = dinfres;
                result.iterations = iters;

                return result;
            }
        }

        // f6_no_ir(x, y, z, tau, s, kappa) solves
        //
        //     [ 0         ]   [  0   A'  G'  c ] [ ux        ]    [ bx   ]
        //     [ 0         ]   [ -A   0   0   b ] [ uy        ]    [ by   ]
        //     [ W'*us     ] - [ -G   0   0   h ] [ W^{-1}*uz ] = -[ bz   ]
        //     [ dg*ukappa ]   [ -c' -b' -h'  0 ] [ utau/dg   ]    [ btau ]
        //
        //     lmbda o (uz + us) = -bs
        //     lmbdag * (utau + ukappa) = -bkappa.
        //
        // On entry, x, y, z, tau, s, kappa contain bx, by, bz, btau,
        // bkappa.  On exit, they contain ux, uy, uz, utau, ukappa.

        // th = W^{-T} * h
        if (iters == 0) {
            th = Matrix(cdim, 1, 0.0);
        }
        blas::copy(h, th);
        misc::scale(th, W, "T", "I");

        auto f6_no_ir = [&](Matrix& x, Matrix& y, Matrix& z, Matrix& tau, Matrix& s, Matrix& kappa) {
            // Solve
            //
            //     [  0   A'  G'    0   ] [ ux        ]
            //     [ -A   0   0     b   ] [ uy        ]
            //     [ -G   0   W'*W  h   ] [ W^{-1}*uz ]
            //     [ -c' -b' -h'    k/t ] [ utau/dg   ]
            //
            //           [ bx                    ]
            //           [ by                    ]
            //         = [ bz - W'*(lmbda o\ bs) ]
            //           [ btau - bkappa/tau     ]
            //
            //     us = -lmbda o\ bs - uz
            //     ukappa = -bkappa/lmbdag - utau.

            // First solve
            //
            //     [ 0  A' G'   ] [ ux        ]   [  bx                    ]
            //     [ A  0  0    ] [ uy        ] = [ -by                    ]
            //     [ G  0 -W'*W ] [ W^{-1}*uz ]   [ -bz + W'*(lmbda o\ bs) ]

            // y := -y = -by
            yscal(-1.0, y);

            // s := -lmbda o\ s = -lmbda o\ bs
            misc::sinv(s, lmbda, dims);
            blas::scal(-1.0, s);

            // z := -(z + W'*s) = -bz + W'*(lambda o\ bs)
            blas::copy(s, ws3);
            misc::scale(ws3, W, "T", "");
            blas::axpy(ws3, z);
            blas::scal(-1.0, z);

            // Solve system.
            f3(x, y, z);

            // Combine with solution of
            //
            //     [ 0   A'  G'    ] [ x1         ]          [ c ]
            //     [-A   0   0     ] [ y1         ] = -dgi * [ b ]
            //     [-G   0   W'*W  ] [ W^{-1}*dzl ]          [ h ]
            //
            // to satisfy
            //
            //     -c'*x - b'*y - h'*W^{-1}*z + dg*tau = btau - bkappa/tau.

            // kappa[0] := -kappa[0] / lmbd[-1] = -bkappa / lmbdag
            kappa[0] = -kappa[0] / lmbda[lmbda.size() - 1];

            // tau[0] = tau[0] + kappa[0] / dgi = btau[0] - bkappa / tau
            tau[0] += kappa[0] / dgi;

            tau[0] = dgi * (tau[0] + xdot(c, x) + ydot(b, y) +
                misc::sdot(th, z, dims)) / (1.0 + misc::sdot(z1, z1, dims));
            xaxpy(x1, x, tau[0]);
            yaxpy(y1, y, tau[0]);
            blas::axpy(z1, z, tau[0]);

            // s := s - z = - lambda o\ bs - z
            blas::axpy(z, s, -1.0);

            kappa[0] -= tau[0];
        };

        // f6(x, y, z, tau, s, kappa) solves the same system as f6_no_ir,
        // but applies iterative refinement.

        static Matrix wx, wy, wz, ws, wtau, wkappa;
        static Matrix wx2, wy2, wz2, ws2, wtau2, wkappa2;

        if (iters == 0) {
            if (refinement || DEBUG) {
                wx = xnewcopy();
                wy = ynewcopy();
                wz = Matrix(cdim, 1, 0.0);
                ws = Matrix(cdim, 1, 0.0);
                wtau = Matrix(1, 1, 0.0);
                wkappa = Matrix(1, 1, 0.0);
            }
            if (refinement) {
                wx2 = xnewcopy();
                wy2 = ynewcopy();
                wz2 = Matrix(cdim, 1, 0.0);
                ws2 = Matrix(cdim, 1, 0.0);
                wtau2 = Matrix(1, 1, 0.0);
                wkappa2 = Matrix(1, 1, 0.0);
            }
        }

        auto f6 = [&](Matrix& x, Matrix& y, Matrix& z, Matrix& tau, Matrix& s, Matrix& kappa) {
            if (refinement || DEBUG) {
                xcopy(x, wx);
                ycopy(y, wy);
                blas::copy(z, wz);
                wtau[0] = tau[0];
                blas::copy(s, ws);
                wkappa[0] = kappa[0];
            }
            f6_no_ir(x, y, z, tau, s, kappa);
            for (int i = 0; i < refinement; ++i) {
                xcopy(wx, wx2);
                ycopy(wy, wy2);
                blas::copy(wz, wz2);
                wtau2[0] = wtau[0];
                blas::copy(ws, ws2);
                wkappa2[0] = wkappa[0];
                res(x, y, z, tau, s, kappa, wx2, wy2, wz2, wtau2, ws2,
                    wkappa2, W, dg, lmbda);
                f6_no_ir(wx2, wy2, wz2, wtau2, ws2, wkappa2);
                xaxpy(wx2, x);
                yaxpy(wy2, y);
                blas::axpy(wz2, z);
                tau[0] += wtau2[0];
                blas::axpy(ws2, s);
                kappa[0] += wkappa2[0];
            }
            if (DEBUG) {
                res(x, y, z, tau, s, kappa, wx, wy, wz, wtau, ws, wkappa,
                    W, dg, lmbda);
                std::cout << "KKT residuals" << std::endl;
                std::cout << "    'x': " << std::scientific << std::sqrt(xdot(wx, wx)) << std::endl;
                std::cout << "    'y': " << std::scientific << std::sqrt(ydot(wy, wy)) << std::endl;
                std::cout << "    'z': " << std::scientific << misc::snrm2(wz, dims) << std::endl;
                std::cout << "    'tau': " << std::scientific << std::abs(wtau[0]) << std::endl;
                std::cout << "    's': " << std::scientific << misc::snrm2(ws, dims) << std::endl;
                std::cout << "    'kappa': " << std::scientific << std::abs(wkappa[0]) << std::endl;
            }
        };

        double mu = std::pow(blas::nrm2(lmbda), 2) / (1 + cdim_diag);
        double sigma = 0.0;

        for (int i = 0; i < 2; ++i) {
            // Solve
            //
            //     [ 0         ]   [  0   A'  G'  c ] [ dx        ]
            //     [ 0         ]   [ -A   0   0   b ] [ dy        ]
            //     [ W'*ds     ] - [ -G   0   0   h ] [ W^{-1}*dz ]
            //     [ dg*dkappa ]   [ -c' -b' -h'  0 ] [ dtau/dg   ]
            //
            //                       [ rx   ]
            //                       [ ry   ]
            //         = - (1-sigma) [ rz   ]
            //                       [ rtau ]
            //
            //     lmbda o (dz + ds) = -lmbda o lmbda + sigma*mu*e
            //     lmbdag * (dtau + dkappa) = - kappa * tau + sigma*mu

            // ds = -lmbdasq if i is 0
            //    = -lmbdasq - dsa o dza + sigma*mu*e if i is 1
            // dkappa = -lambdasq[-1] if i is 0
            //        = -lambdasq[-1] - dkappaa*dtaua + sigma*mu if i is 1.

            blas::copy(lmbdasq, ds, dims.at("l") + misc::sum_vector(dims.at("q")));
            int ind = dims.at("l") + misc::sum_vector(dims.at("q"));
            int ind2 = ind;
            blas::scal(0.0, ds, ind);
            for (int m : dims.at("s")) {
                blas::copy(lmbdasq, ds, m, ind2, ind, m+1);
                ind += m * m;
                ind2 += m;
            }
            dkappa[0] = lmbdasq[lmbdasq.size() - 1];

            if (i == 1) {
                blas::axpy(ws3, ds);
                for (int j = 0; j < dims.at("l"); ++j) {
                    ds[j] -= sigma * mu;
                }
                for (int j = 0; j < indq.size() - 1; ++j) {
                    ds[indq[j]] -= sigma * mu;
                }
                ind = dims.at("l") + misc::sum_vector(dims.at("q"));
                ind2 = ind;
                for (int m : dims.at("s")) {
                    for (int j = ind; j < ind + m*m; j += m+1) {
                        ds[j] -= sigma * mu;
                    }
                    ind += m * m;
                }
                dkappa[0] += wkappa3 - sigma * mu;
            }

            // (dx, dy, dz, dtau) = (1-sigma)*(rx, ry, rz, rt)
            xcopy(rx, dx);
            xscal(1.0 - sigma, dx);
            ycopy(ry, dy);
            yscal(1.0 - sigma, dy);
            blas::copy(rz, dz);
            blas::scal(1.0 - sigma, dz);
            dtau[0] = (1.0 - sigma) * rt;

            f6(dx, dy, dz, dtau, ds, dkappa);

            // Save ds o dz and dkappa * dtau for Mehrotra correction
            if (i == 0) {
                blas::copy(ds, ws3);
                misc::sprod(ws3, dz, dims);
                wkappa3 = dtau[0] * dkappa[0];
            }

            // Maximum step to boundary.
            //
            // If i is 1, also compute eigenvalue decomposition of the 's'
            // blocks in ds, dz.  The eigenvectors Qs, Qz are stored in
            // dsk, dzk.  The eigenvalues are stored in sigs, sigz.

            misc::scale2(lmbda, ds, dims);
            misc::scale2(lmbda, dz, dims);

            double ts, tz;
            if (i == 0) {
                ts = misc::max_step(ds, dims);
                tz = misc::max_step(dz, dims);
            } else {
                ts = misc::max_step(ds, dims, sigs);
                tz = misc::max_step(dz, dims, sigz);
            }

            double tt = -dtau[0] / lmbda[lmbda.size() - 1];
            double tk = -dkappa[0] / lmbda[lmbda.size() - 1];
            double t = std::max({0.0, ts, tz, tt, tk});

            double step;
            if (t == 0.0) {
                step = 1.0;
            } else {
                if (i == 0) {
                    step = std::min(1.0, 1.0 / t);
                } else {
                    step = std::min(1.0, STEP / t);
                }
            }

            if (i == 0) {
                sigma = std::pow(1.0 - step, EXPON);
            }
        }

        // Update x, y.
        xaxpy(dx, x, step);
        yaxpy(dy, y, step);

        // Replace 'l' and 'q' blocks of ds and dz with the updated
        // variables in the current scaling.
        // Replace 's' blocks of ds and dz with the factors Ls, Lz in a
        // factorization Ls*Ls', Lz*Lz' of the updated variables in the
        // current scaling.

        // ds := e + step*ds for 'l' and 'q' blocks.
        // dz := e + step*dz for 'l' and 'q' blocks.
        blas::scal(step, ds, dims.at("l") + misc::sum_vector(dims.at("q")));
        blas::scal(step, dz, dims.at("l") + misc::sum_vector(dims.at("q")));

        for (int j = 0; j < dims.at("l"); ++j) {
            ds[j] += 1.0;
            dz[j] += 1.0;
        }

        for (int j = 0; j < indq.size() - 1; ++j) {
            ds[indq[j]] += 1.0;
            dz[indq[j]] += 1.0;
        }

        // ds := H(lambda)^{-1/2} * ds and dz := H(lambda)^{-1/2} * dz.
        //
        // This replaces the 'l' and 'q' components of ds and dz with the
        // updated variables in the current scaling.
        // The 's' components of ds and dz are replaced with
        //
        //     diag(lmbda_k)^{1/2} * Qs * diag(lmbda_k)^{1/2}
        //     diag(lmbda_k)^{1/2} * Qz * diag(lmbda_k)^{1/2}
        //
        misc::scale2(lmbda, ds, dims, "I");
        misc::scale2(lmbda, dz, dims, "I");

        // sigs := ( e + step*sigs ) ./ lambda for 's' blocks.
        // sigz := ( e + step*sigz ) ./ lambda for 's' blocks.
        blas::scal(step, sigs);
        blas::scal(step, sigz);
        for (int j = 0; j < sigs.size(); ++j) {
            sigs[j] += 1.0;
            sigz[j] += 1.0;
        }

        int sum_s = misc::sum_vector(dims.at("s"));
        blas::tbsv(lmbda, sigs, sum_s, 0, 1, dims.at("l") + misc::sum_vector(dims.at("q")));
        blas::tbsv(lmbda, sigz, sum_s, 0, 1, dims.at("l") + misc::sum_vector(dims.at("q")));

        // dsk := Ls = dsk * sqrt(sigs).
        // dzk := Lz = dzk * sqrt(sigz).
        int ind2 = dims.at("l") + misc::sum_vector(dims.at("q"));
        int ind3 = 0;

        for (int k = 0; k < dims.at("s").size(); ++k) {
            int m = dims.at("s")[k];
            for (int i = 0; i < m; ++i) {
                blas::scal(std::sqrt(sigs[ind3 + i]), ds, ind2 + m * i, m);
                blas::scal(std::sqrt(sigz[ind3 + i]), dz, ind2 + m * i, m);
            }
            ind2 += m * m;
            ind3 += m;
        }

        // Update lambda and scaling.
        misc::update_scaling(W, lmbda, ds, dz);

        // For kappa, tau block:
        //
        //     dg := sqrt( (kappa + step*dkappa) / (tau + step*dtau) )
        //         = dg * sqrt( (1 - step*tk) / (1 - step*tt) )
        //
        //     lmbda[-1] := sqrt((tau + step*dtau) * (kappa + step*dkappa))
        //                = lmbda[-1] * sqrt(( 1 - step*tt) * (1 - step*tk))

        double tt = -dtau[0] / lmbda[lmbda.size() - 1];
        double tk = -dkappa[0] / lmbda[lmbda.size() - 1];

        dg *= std::sqrt(1.0 - step * tk) / std::sqrt(1.0 - step * tt);
        dgi = 1.0 / dg;
        lmbda[lmbda.size() - 1] *= std::sqrt(1.0 - step * tt) * std::sqrt(1.0 - step * tk);

        // Unscale s, z, tau, kappa (unscaled variables are used only to
        // compute feasibility residuals).

        blas::copy(lmbda, s, dims.at("l") + misc::sum_vector(dims.at("q")));
        int ind = dims.at("l") + misc::sum_vector(dims.at("q"));
        int ind2 = ind;

        for (int m : dims.at("s")) {
            blas::scal(0.0, s, ind2);
            blas::copy(lmbda, s, ind, ind2, m, m+1);
            ind += m;
            ind2 += m * m;
        }
        misc::scale(s, W, "T", "");

        blas::copy(lmbda, z, dims.at("l") + misc::sum_vector(dims.at("q")));
        ind = dims.at("l") + misc::sum_vector(dims.at("q"));
        ind2 = ind;

        for (int m : dims.at("s")) {
            blas::scal(0.0, z, ind2);
            blas::copy(lmbda, z, ind, ind2, m, m+1);
            ind += m;
            ind2 += m * m;
        }
        misc::scale(z, W, "", "I");

        kappa = lmbda[lmbda.size() - 1] / dgi;
        tau = lmbda[lmbda.size() - 1] * dgi;
        gap = std::pow(blas::nrm2(lmbda, lmbda.size() - 1) / tau, 2);
    }

    // This should never be reached due to the loop condition, but adding for completeness
    OptimizationResult result;
    result.status = "unknown";
    result.iterations = MAXITERS;
    return result;
}

