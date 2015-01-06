/**
 * Finalization step - write results and other computational vectors to files
 *
 * @date 22-Oct-2012
 * @author V. Petkov
 */
#include "finalization.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "util_write_files.h"
#include "initialization.h"

#include "mpi.h"

void finalization(char* file_in, char* out_prefix, int total_iters, double residual_ratio,
                  int nintci, int nintcf, int points_count, int** points, int* elems, double* var,
                  double* cgup, double* su, int* local_global_index, char* part_type,
                  int* local_global_index_internalSet) {
    int my_rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);  // Get current process id
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    char file_out[100];
    sprintf(file_out, "%s_summary.out", out_prefix);

    int status = store_simulation_stats(file_in, file_out, nintci, nintcf, var, total_iters,
                                        residual_ratio);

    // collecting data for finalization. var, cgup. su,
    int num_elems_internal_local = nintcf - nintci + 1;
    int *num_elems_internal_local_set = NULL, *displs = NULL;
    int num_elems_internal_total = 0;
//     double *var_total;
//     double *su_total;
    double *cgup_total;
//     double *var_reordered;
//     double *su_reordered;
    double *cgup_reordered;


    if (my_rank == 0) {
        num_elems_internal_local_set = (int*) calloc(sizeof(int), num_procs);
        displs = (int *) calloc(sizeof(int), num_procs);
    }

    MPI_Gather(&num_elems_internal_local, 1, MPI_INT,
                num_elems_internal_local_set, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        for (int i = 1; i < num_procs; i++) {
            displs[i] = displs[i-1] + num_elems_internal_local_set[i -1];
        }

        for (int i = 0; i < num_procs; i++) {
            num_elems_internal_total += num_elems_internal_local_set[i];
        }

//         var_total = (double *) calloc(sizeof(double), num_elems_internal_total);
        cgup_total = (double *) calloc(sizeof(double), num_elems_internal_total);
//         su_total = (double *) calloc(sizeof(double), num_elems_internal_total);

        sprintf(file_out, "%s_data.vtk", out_prefix);
        vtk_write_unstr_grid_header(file_in, file_out, 0, num_elems_internal_total - 1,
                                    points_count, points, elems);
    }


//     MPI_Gatherv(var, num_elems_internal_local, MPI_DOUBLE,
//                 var_total, num_elems_internal_local_set, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(cgup, num_elems_internal_local, MPI_DOUBLE,
                cgup_total, num_elems_internal_local_set, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//     MPI_Gatherv(su, num_elems_internal_local, MPI_DOUBLE,
//                 su_total, num_elems_internal_local_set, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);


    if (~strcmp(part_type, "dual")) {
        if (my_rank == 0) {
            int nc_global;
            double *var_reordered, *cgup_reordered, *su_reordered;
//             var_reordered = (double *) calloc(sizeof(double), num_elems_internal_total);
            cgup_reordered = (double *) calloc(sizeof(double), num_elems_internal_total);
//             su_reordered = (double *) calloc(sizeof(double), num_elems_internal_total);

            for (int nc = 0; nc < num_elems_internal_total; nc++) {
                nc_global = local_global_index_internalSet[nc];
//                 var_reordered[nc_global]=var_total[nc];
//                 su_reordered[nc_global]=su_total[nc];
                cgup_reordered[nc_global]=cgup_total[nc];
            }

            vtk_append_double(file_out, "CGUP", 0, num_elems_internal_total - 1, cgup_reordered);
//             vtk_append_double(file_out, "VAR", 0, num_elems_internal_total - 1, var_reordered);
//             vtk_append_double(file_out, "SU", 0, num_elems_internal_total - 1, su_reordered);

            free(num_elems_internal_local_set);
            free(displs);
//             free (var_total);
            free(cgup_total);
//             free (su_total);
//             free (var_reordered);
            free(cgup_reordered);
//             free (su_reordered);
        }
    } else {
        if (my_rank == 0) {
            vtk_append_double(file_out, "CGUP", 0, num_elems_internal_total - 1 , cgup_total);
//             vtk_append_double(file_out, "VAR", 0, num_elems_internal_total - 1, var_total);
//             vtk_append_double(file_out, "SU", 0, num_elems_internal_total - 1, su_total);

            free(num_elems_internal_local_set);
            free(displs);
//             free (var_total);
            free(cgup_total);
//             free (su_total);
        }
    }

    if ( status != 0 ) fprintf(stderr, "Error when trying to write to file %s\n", file_out);
    Program_Message("Return from finalization");
}

