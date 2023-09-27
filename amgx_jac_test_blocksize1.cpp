#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <amgx_c.h>

int main(){
    // Initialize AMGX
    AMGX_SAFE_CALL(AMGX_initialize());
    AMGX_SAFE_CALL(AMGX_initialize());

    // Create an AMGX configuration
    AMGX_config_handle cfg;
    AMGX_SAFE_CALL(AMGX_config_create(&cfg, "solver=MULTICOLOR_DILU, max_iters=1, relaxation_factor=0.9"));

    // Create AMGX resources
    AMGX_resources_handle rsrc;
    AMGX_SAFE_CALL(AMGX_resources_create_simple(&rsrc, cfg));

    // Define the linear system size
    int num_rows = 4;  
    int blocksize = 1;
    int nnz = 8; // number of nonzero elements

    // Create and populate the vector
    std::vector<double> input_vector_x({1, 2, 1, 1});
    std::vector<double> input_vector_b({2, 1, 3, 4});

    // Create AMGX vector
    AMGX_vector_handle x;
    AMGX_vector_handle b;
    AMGX_SAFE_CALL(AMGX_vector_create(&x, rsrc, AMGX_mode_dDDI));
    AMGX_SAFE_CALL(AMGX_vector_create(&b, rsrc, AMGX_mode_dDDI));

    // Upload the vector data to AMGX
    AMGX_SAFE_CALL(AMGX_vector_upload(x, num_rows, blocksize, input_vector_x.data()));
    AMGX_SAFE_CALL(AMGX_vector_upload(b, num_rows, blocksize, input_vector_b.data()));

    // Create the solver
    AMGX_solver_handle solver;
    AMGX_SAFE_CALL(AMGX_solver_create(&solver, rsrc, AMGX_mode_dDDI, cfg));

    // Setup the linear system (A is not used in this example)
    AMGX_matrix_handle A;
    AMGX_SAFE_CALL(AMGX_matrix_create(&A, rsrc, AMGX_mode_dDDI));

    // Set A as an identity matrix
    std::vector<int> row_ptrs({0,3,6,7,8});
    std::vector<int> col_indices({0,1,2,0,1,3,2,3});
    std::vector<double> values({3,1,1,2,1,1,-1,-1});

    // Upload the matrix data to AMGX using AMGX_matrix_upload_all()
    AMGX_SAFE_CALL(AMGX_matrix_upload_all(A, num_rows, nnz, blocksize, blocksize, row_ptrs.data(), col_indices.data(), values.data(), NULL));

    // Setup the solver
    AMGX_SAFE_CALL(AMGX_solver_setup(solver, A));

    // Perform Jacobi preconditioning
    AMGX_SAFE_CALL(AMGX_solver_solve(solver, b, x));

    // Download the result vector from AMGX
    AMGX_SAFE_CALL(AMGX_vector_download(x, &input_vector_x[0]));

    // Clean up resources
    AMGX_SAFE_CALL(AMGX_solver_destroy(solver));
    AMGX_SAFE_CALL(AMGX_matrix_destroy(A));
    AMGX_SAFE_CALL(AMGX_vector_destroy(x));
    AMGX_SAFE_CALL(AMGX_vector_destroy(b));
    AMGX_SAFE_CALL(AMGX_resources_destroy(rsrc));
    AMGX_SAFE_CALL(AMGX_config_destroy(cfg));
    AMGX_SAFE_CALL(AMGX_finalize());

    // Print the result
    for (int i = 0; i < num_rows; ++i) {
        std::cout << "x[" << i << "] = " << input_vector_x[i] << std::endl;
    }

    return 0;
}