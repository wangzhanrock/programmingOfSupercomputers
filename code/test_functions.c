/**
 * Functions to test the data distribution and communication lists creation algorithms
 *
 * @date 22-Oct-2012
 * @author V. Petkov
 */
#include "test_functions.h"
#include <stdio.h>
#include <stdlib.h>
#include "util_write_files.h"
#include "util_read_files.h"
#include "initialization.h"
#include "mpi.h"


int test_distribution(char *file_in, char *file_vtk_out,
                      int *local_global_index, int num_elems_local, double *cgup) {
    int i, my_rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);  /// Get current process id
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int nintci, nintcf, nextci, nextcf;
    int **lcc;
    double *bs, *be, *bn, *bw, *bl, *bh, *bp, *su;
    int points_count;  /// total number of points that define the geometry
    int** points;  /// coordinates of the points that define the cells - size [points_cnt][3]
    int* elems;
    int num_elems_internal;
    double *distr;

    int f_status = read_binary_geo(file_in, &nintci, &nintcf, &nextci, &nextcf,
                                   &lcc, &bs, &be, &bn, &bw, &bl, &bh, &bp, &su, &points_count,
                                   &points, &elems);
    if (f_status != 0) {
        return f_status;
    }

    num_elems_internal = nintcf - nintci + 1;
    distr = (double *) calloc(num_elems_internal, sizeof(double));
    for (i = 0; i < num_elems_local; i++) {
        distr[local_global_index[i]] = cgup[i];
    }
    vtk_write_unstr_grid_header(file_in, file_vtk_out, nintci, nintcf,
                                points_count, points, elems);
    vtk_append_double(file_vtk_out, "cgup", nintci, nintcf, distr);

    free(distr);

    return 0;
}

int test_communication(char *file_in, char *file_vtk_out,
                       int *local_global_index, int num_elems_local, int neighbors_count,
                       int *send_count, int** send_list, int* recv_count, int** recv_list) {
    int i, j, my_rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);  /// Get current process id
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int nintci, nintcf, nextci, nextcf;
    int **lcc;
    double *bs, *be, *bn, *bw, *bl, *bh, *bp, *su;
    int points_count;  /// total number of points that define the geometry
    int** points;  /// coordinates of the points that define the cells - size [points_cnt][3]
    int* elems;
    int num_elems_internal;
    double *commlist;

    int f_status = read_binary_geo(file_in, &nintci, &nintcf, &nextci, &nextcf,
                                   &lcc, &bs, &be, &bn, &bw, &bl, &bh, &bp, &su, &points_count,
                                   &points, &elems);
    if (f_status != 0)
        return f_status;

    num_elems_internal = nintcf - nintci + 1;
    commlist = (double *) calloc(num_elems_internal, sizeof(double));
    // prepare the value of commlist for writing the vtk file
    for (i = 0; i < num_elems_local; i++) {
        commlist[local_global_index[i]] = 15;
    }
    for (i = 0; i < neighbors_count; i++) {
        for (j = 0; j < send_count[i]; j++) {
            commlist[send_list[i][j]] = 10;
        }
    }
    for (i = 0; i < neighbors_count; i++) {
        for (j = 0; j < recv_count[i]; j++) {
            commlist[recv_list[i][j]] = 5;
        }
    }
    vtk_write_unstr_grid_header(file_in, file_vtk_out, nintci, nintcf,
                                points_count, points, elems);
    vtk_append_double(file_vtk_out, "commlist", nintci, nintcf, commlist);

    free(commlist);

    return 0;
}

