#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <amgx_c.h>

int main(){
    // Initialize AMGX
    AMGX_SAFE_CALL(AMGX_initialize());

    // Create an AMGX configuration
    AMGX_config_handle cfg;
    AMGX_SAFE_CALL(AMGX_config_create(&cfg, "config_version=2, solver(my_solver)=AMG, my_solver:smoother(my_smoother)=BLOCK_JACOBI, my_smoother:max_iters=1, my_smoother:relaxation_factor=0.5"));

    // Create AMGX resources
    AMGX_resources_handle rsrc;
    AMGX_SAFE_CALL(AMGX_resources_create_simple(&rsrc, cfg));

    // Define the linear system size
    int num_rows = 2; 
    int blocksize = 3;

    int nnz = 3; //nonzero blockelements
    // Create and populate the vector
    double input_vector_x[num_rows*blocksize] = {1, 2, 1, 1, 2, 2};
    double input_vector_b[num_rows*blocksize] = {2, 1, 3, 4, 2, 1};
    int row_ptrs[num_rows] = {0,2,3};
    int col_indices[nnz] = {0,1,1};
    double values[nnz*blocksize*blocksize]  = {3,1,1, 2,1,1, 1,2,3,  1,0,0, 0,1,0, 0,0,1,  -1,0,0, 0,-1,0, 0,0,-1};

    // Create AMGX vector
    AMGX_vector_handle x;
    AMGX_vector_handle b;
    AMGX_SAFE_CALL(AMGX_vector_create(&x, rsrc, AMGX_mode_dDDI));
    AMGX_SAFE_CALL(AMGX_vector_create(&b, rsrc, AMGX_mode_dDDI));

    // Upload the vector data to AMGX
    AMGX_SAFE_CALL(AMGX_vector_upload(x, num_rows, blocksize, input_vector_x));
    AMGX_SAFE_CALL(AMGX_vector_upload(b, num_rows, blocksize, input_vector_b));

    // Create the solver
    AMGX_solver_handle solver;
    AMGX_SAFE_CALL(AMGX_solver_create(&solver, rsrc, AMGX_mode_dDDI, cfg));

    // Setup the linear system (A is not used in this example)
    AMGX_matrix_handle A;
    AMGX_SAFE_CALL(AMGX_matrix_create(&A, rsrc, AMGX_mode_dDDI));

    // Upload the matrix data to AMGX using AMGX_matrix_upload_all()
    AMGX_SAFE_CALL(AMGX_matrix_upload_all(A, num_rows, nnz, blocksize, blocksize, row_ptrs, col_indices, values, NULL));
    
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
    for (int i = 0; i < num_rows*blocksize; ++i) {
        std::cout << "x[" << i << "] = " << input_vector_x[i] << std::endl;
    }

    /*

    DILU OUTPUT:
    x[0] = 0.972318
    x[1] = 1.13841
    x[2] = -0.0553633
    x[3] = -0.323529

    */

    return 0;
}