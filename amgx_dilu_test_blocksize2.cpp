#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <amgx_c.h>

int main(){
    // Initialize AMGX
    AMGX_SAFE_CALL(AMGX_initialize());
    AMGX_config_handle cfg;

    // Two problematic solvers in AMGX:
    // From reading the reference manual it should not be a problem to use a solver that is just a preconditioner, although this might be outside of intended use?
    // It works fine for some preconditioners but not others
    // In some cases the reference manual says a preconditioner supports blocks when it does not in the code (multicolor ILU)
    // Others give unhelpful internal errors (block_jacobi)
    // AMGX_SAFE_CALL(AMGX_config_create(&cfg, "config_version=2, \
    //                                         solver(jac_solv)=BLOCK_JACOBI, \
    //                                         jac_solv:relaxation_factor=1.0, \
    //                                         jac_solv:max_iters=1")); // gives internal error
    // AMGX_SAFE_CALL(AMGX_config_create(&cfg, "config_version=2, \
    //                                         solver(gs_solv)=MULTICOLOR_GS, \
    //                                         gs_solv:coloring_level=1, \
    //                                         gs_solv:matrix_coloring_scheme=MIN_MAX, \
    //                                         gs_solv:max_uncolored_percentage=0.0, \
    //                                         gs_solv:relaxation_factor=1.0, \
    //                                         gs_solv:max_iters=1,\
    //                                         gs_solv:reorder_cols_by_color=1, \
    //                                         gs_solv:insert_diag_while_reordering=1")); // gives unsupported block size!

    // We have our own implementation on DILU that works great on SPE1, this version does not seem to precondition the systems effectively
    AMGX_SAFE_CALL(AMGX_config_create(&cfg, "config_version=2, \
                                            solver(dilu_solv)=MULTICOLOR_DILU, \
                                            dilu_solv:coloring_level=1, \
                                            dilu_solv:matrix_coloring_scheme=MIN_MAX, \
                                            dilu_solv:max_uncolored_percentage=0.0, \
                                            dilu_solv:relaxation_factor=1.0, \
                                            dilu_solv:max_iters=1,\
                                            dilu_solv:reorder_cols_by_color=1, \
                                            dilu_solv:insert_diag_while_reordering=1")); // is valid, but works terribly in practice when solving SPE1



    // Create AMGX resources
    AMGX_resources_handle rsrc;
    AMGX_SAFE_CALL(AMGX_resources_create_simple(&rsrc, cfg));

    // Define the linear system size
    int num_rows = 2;
    int blocksize = 2;
    int nnz = 3; //nonzero blockelementss
    // Create and populate the vector
    double input_vector_x[4] = {1, 2, 1, 1};
    double input_vector_b[4] = {2, 1, 3, 4};
    int row_ptrs[num_rows] = {0,2,3};
    int col_indices[nnz] = {0,1,1};
    double values[nnz*blocksize*blocksize]  = {3,1,1,2,1,0,0,1,-1,0,0,-1};

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

    // Setup the linear system
    AMGX_matrix_handle A;
    AMGX_SAFE_CALL(AMGX_matrix_create(&A, rsrc, AMGX_mode_dDDI));

    // Upload the matrix data to AMGX using AMGX_matrix_upload_all()
    AMGX_SAFE_CALL(AMGX_matrix_upload_all(A, num_rows, nnz, blocksize, blocksize, row_ptrs, col_indices, values, NULL));

    // Setup the solver
    AMGX_SAFE_CALL(AMGX_solver_setup(solver, A));

    /*
        Test data to validate preconditioner, expected result is x_1
            | |3 1|  | 1  0| |       | |1| |     | |2| |
            | |1 2|  | 0  1| |       | |2| |     | |1| |
        A = |                | x_0 = |     | b = |     |
            | |0 0|  |-1  0| |       | |1| |     | |3| |
            | |0 0|  | 0 -1| |       | |1| |     | |4| |
    */

    // Apply preconditioner
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

    return 0;
}