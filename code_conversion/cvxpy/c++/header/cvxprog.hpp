#include <vector>
#include <map>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <tuple>
#include <numeric>

// Matrix class declarations (assuming a CVXOPT-like C++ matrix library)
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

    double& operator()(int r, int c) { return data[r * cols + c]; }
    const double& operator()(int r, int c) const { return data[r * cols + c]; }

    Matrix slice(int start, int stop) const {
        Matrix result(stop - start, 1);
        for (int i = 0; i < stop - start; ++i) {
            result[i] = data[start + i];
        }
        return result;
    }
};

class SparseMatrix {
public:
    std::vector<int> I, J;
    std::vector<double> V;
    int rows, cols;
    char typecode;

    SparseMatrix(const std::vector<int>& i, const std::vector<int>& j,
                 const std::vector<double>& v, std::pair<int,int> size, char tc = 'd')
        : I(i), J(j), V(v), rows(size.first), cols(size.second), typecode(tc) {}

    std::pair<int, int> size() const { return {rows, cols}; }
};

// BLAS-like operations
namespace blas {
    void copy(const Matrix& src, Matrix& dst) {
        dst.data = src.data;
    }

    double asum(const Matrix& x, int n = -1, int offset = 0) {
        if (n == -1) n = x.data.size() - offset;
        double sum = 0.0;
        for (int i = 0; i < n; ++i) {
            sum += std::abs(x.data[offset + i]);
        }
        return sum;
    }

    void scal(double alpha, Matrix& x, int n = -1, int offset = 0, int inc = 1) {
        if (n == -1) n = x.data.size() - offset;
        for (int i = 0; i < n; ++i) {
            x.data[offset + i * inc] *= alpha;
        }
    }

    void axpy(const Matrix& x, Matrix& y, int n = -1, double alpha = 1.0,
              int incx = 1, int incy = 1, int offsetx = 0, int offsety = 0) {
        if (n == -1) n = std::min(x.data.size() - offsetx, y.data.size() - offsety);
        for (int i = 0; i < n; ++i) {
            y.data[offsety + i * incy] += alpha * x.data[offsetx + i * incx];
        }
    }

    void syrk(const Matrix& A, Matrix& C, char trans = 'N', int k = -1,
              double alpha = 1.0, double beta = 0.0) {
        // Simplified symmetric rank-k update: C := alpha*A*A^T + beta*C
        int n = C.rows;
        if (k == -1) k = (trans == 'T') ? A.rows : A.cols;

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j <= i; ++j) {
                double sum = 0.0;
                if (trans == 'T') {
                    for (int l = 0; l < k; ++l) {
                        sum += A(l, i) * A(l, j);
                    }
                } else {
                    for (int l = 0; l < k; ++l) {
                        sum += A(i, l) * A(j, l);
                    }
                }
                C(i, j) = alpha * sum + beta * C(i, j);
                if (i != j) C(j, i) = C(i, j);
            }
        }
    }
}

// Base operations
namespace base {
    void gemv(const Matrix& A, const Matrix& x, Matrix& y,
              char trans = 'N', int m = -1, double alpha = 1.0, double beta = 0.0,
              int incy = 1, int offsetA = 0, int offsetx = 0, int offsety = 0) {
        if (m == -1) m = (trans == 'N') ? A.rows : A.cols;
        int n = (trans == 'N') ? A.cols : A.rows;

        if (trans == 'N') {
            for (int i = 0; i < m; ++i) {
                double sum = 0.0;
                for (int j = 0; j < n; ++j) {
                    sum += A(offsetA + i, j) * x[offsetx + j];
                }
                y[offsety + i * incy] = alpha * sum + beta * y[offsety + i * incy];
            }
        } else {
            for (int j = 0; j < m; ++j) {
                double sum = 0.0;
                for (int i = 0; i < n; ++i) {
                    sum += A(offsetA + i, j) * x[offsetx + i];
                }
                y[offsety + j * incy] = alpha * sum + beta * y[offsety + j * incy];
            }
        }
    }

    void gemv(const SparseMatrix& A, const Matrix& x, Matrix& y,
              char trans = 'N', int m = -1, double alpha = 1.0, double beta = 0.0,
              int incy = 1, int offsetA = 0, int offsetx = 0, int offsety = 0) {
        // Simplified sparse matrix-vector multiplication
    }

    Matrix exp(const Matrix& x, double offset = 0.0) {
        Matrix result(x.rows, x.cols);
        for (size_t i = 0; i < x.data.size(); ++i) {
            result.data[i] = std::exp(x.data[i] + offset);
        }
        return result;
    }
}