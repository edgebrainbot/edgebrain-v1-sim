#ifndef LMI_OPTIMIZER_HPP
#define LMI_OPTIMIZER_HPP

#include <Eigen/Dense>
#include <cmath>
#include <ctime>
#include <cstdio>

#ifdef __cplusplus
extern "C" {
  #include "ecvxcone/base.h"
  #include "ecvxcone/blas.h"
  #include "ecvxcone/misc.h"
  #include "ecvxcone/solver.h"
  #include "ecvxcone/lapack.h"
  #include "ecvxcone/cvxopt.h"
  #include "ecvxcone/cpg_solve.h"
  #include "ecvxcone/cpg_workspace.h"
}
#endif

ECVXConeWorkspace* ecvxcone_setup(int n_var, int n_ineq, int n_eq, int nnz_G, int nnz_A, DIMs *dims, ECVXConeSettings *settings);
// extern ECVXConeWorkspace* ecvxcone_init(matrix *c, void *G, matrix *h, void *A, matrix *b, DIMs *dims, ECVXConeSettings *settings);


// Sampling period
constexpr double Ts = 1.0 / 20.0;

namespace lmi {

using Mat3x3 = Eigen::Matrix<double, 3, 3>;
using Mat6x6 = Eigen::Matrix<double, 6, 6>;
using Vec6d  = Eigen::Matrix<double, 6, 1>;

class LMIOptimizer {
public:
  LMIOptimizer();  // Constructor
  ~LMIOptimizer(); // Destructor

  Mat6x6 F_kp;  // Contains the solved Kp gain matrices
  Mat6x6 F_kd;  // Contains the solved Kd gain matrices

  Mat6x6 default_F_kp;  // Default Kp gain matrices
  Mat6x6 default_F_kd;  // Default Kd gain matrices

  double tracking_err[10];
  double tracking_err_square[10];

  void update();             // Update matrices and call solver
  void test_and_print();     // Test inputs and print results
  void benchmark(int iterations = 1000, bool verbose = false, double threshold = 0.1);
  
  // Rotation matrices
  void update_Rx(double roll);
  void update_Ry(double pitch);
  void update_Rz(double yaw);
  void update_Rzyx(double roll, double pitch, double yaw);

  void set_default_kp(Mat6x6 kp) { default_F_kp = kp; print_matrix6x6(default_F_kp, "Default Kp set to:");
  }
  void set_default_kd(Mat6x6 kd) { default_F_kd = kd; print_matrix6x6(default_F_kd, "Default Kd set to:");}

  // Utility
  static void print_matrix6x6(const Mat6x6& mat, const char* name);
  static void matrix_multiply(const Mat3x3& A, const Mat3x3& B, Mat3x3& result) {
    result = A * B;
  }

private:
  // matrix objects
  matrix* P_mat_;
  matrix* Q_mat_;
  matrix* R_mat_;
  matrix* aB_;
  matrix* aF_;
  matrix* ipiv_;

  Mat3x3 Rx_;
  Mat3x3 Ry_;
  Mat3x3 Rz_;
  Mat3x3 Rzyx_;

  // Initialization helpers
  void lmi_init();                 // Initialize LMI matrices
  void test_tracking_err();       // Set up tracking error for testing
  double time_diff(timespec start, timespec end, int count);

  // Matrix operations
  void update_Matrix_A();
  void update_Matrix_B();
  void update_TrackingErrorSquare();
  void post_processing();

};

} // namespace lmi


#endif  // LMI_OPTIMIZER_HPP
