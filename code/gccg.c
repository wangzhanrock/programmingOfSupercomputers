/**
 * Main GCCG program
 *
 * @author E. Xue, V. Petkov
 * @date 22-May-2009, 22-Oct-2012
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mpi.h"

#include "initialization.h"
#include "compute_solution.h"
#include "finalization.h"
#include "test_functions.h"

// #include "scorep/SCOREP_User.h"

int main(int argc, char *argv[]) {
    int my_rank, num_procs, i;

    const int max_iters = 10000;   /// maximum number of iteration to perform

    /** Simulation parameters parsed from the input datasets */
    int nintci, nintcf;   /// internal cells start and end index
      /// external cells start and end index. The external cells are only ghost cells.
      /// They are accessed only through internal cells
    int nextci, nextcf;
    int **lcc;   /// link cell-to-cell array - stores neighboring information
      /// Boundary coefficients for each volume cell (South, East, North, West, High, Low)
    double *bs, *be, *bn, *bw, *bl, *bh;
    double *bp;   /// Pole coefficient
    double *su;   /// Source values

    double residual_ratio;   /// the ratio between the reference and the current residual
    double *var;   /// the variation vector -> keeps the result in the end

    /** Additional vectors required for the computation */
    double *cgup, *oc, *cnorm;

    /** Geometry data */
    int points_count;   /// total number of points that define the geometry
    int** points;   /// coordinates of the points that define the cells - size [points_cnt][3]
    int* elems;   /// definition of the cells using their nodes (points) - each cell has 8 points

    /** Mapping between local and remote cell indices */
    int* local_global_index;   /// local to global index mapping
    int* global_local_index;   /// global to local index mapping
    int* local_global_index_internalSet;

    /** Lists of cells requires for the communication */
    int neighbors_count = 0;   /// total number of neighbors to communicate with
    int* send_count;   /// number of elements to send to each neighbor (size: neighbors_count)
    /// send lists for the other neighbors(cell ids which should be sent)(size:[#neighbors][#cells]
    int** send_list;
    int* recv_count;   /// how many elements are in the recv lists for each neighbor
    int** recv_list;   /// send lists for the other neighbor (see send_list)

    /** Metis Results */
    int* epart;   /// partition vector for the elements of the mesh
    int* npart;   /// partition vector for the points (nodes) of the mesh
    int objval;
    /// resulting edgecut of total communication volume (classical distrib->zeros)

    MPI_Init(&argc, &argv);   /// Start MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);   /// Get current process id
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);   /// get number of processes
    if (argc < 3) {
        fprintf(stderr,
                "Usage: ./gccg <input_file> <output_prefix> <partition_type>\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    char *file_in = argv[1];
    char *out_prefix = argv[2];
    char *part_type = (argc == 3 ? "classical" : argv[3]);

    /********** START INITIALIZATION **********/
      // read-in the input file
//     SCOREP_USER_REGION_DEFINE(INIT_Phase);
//     SCOREP_USER_OA_PHASE_BEGIN(INIT_Phase, "INIT_Phase", SCOREP_USER_REGION_TYPE_COMMON);

    int init_status = initialization(file_in, part_type, &nintci, &nintcf,
                                     &nextci, &nextcf, &lcc, &bs, &be, &bn, &bw, &bl, &bh, &bp, &su,
                                     &points_count, &points, &elems, &var, &cgup, &oc, &cnorm,
                                     &local_global_index, &global_local_index, &neighbors_count,
                                     &send_count, &send_list, &recv_count, &recv_list, &epart,
                                     &npart, &objval, &local_global_index_internalSet);

//     SCOREP_USER_OA_PHASE_END(INIT_Phase);

    if (init_status != 0) {
        fprintf(stderr, "Failed to initialize data!\n");
        MPI_Abort(MPI_COMM_WORLD, my_rank);
    }

    /********** END INITIALIZATION **********/

    /********** START COMPUTATIONAL LOOP **********/
    int total_iters = compute_solution(max_iters, nintci, nintcf, nextcf, lcc, bp, bs, bw, bl, bn,
                                        be, bh, cnorm, var, su, cgup, &residual_ratio,
                                        local_global_index, global_local_index, neighbors_count,
                                        send_count, send_list, recv_count, recv_list);
     /********** END COMPUTATIONAL LOOP **********/

//     Program_Message("After computation\n");

     /********** START FINALIZATION **********/
    finalization(file_in, out_prefix, total_iters, residual_ratio, nintci, nintcf, points_count,
                 points, elems, var, cgup, su, local_global_index, part_type,
                 local_global_index_internalSet);

    /********** END FINALIZATION **********/

    if (my_rank ==0) {
        free(local_global_index_internalSet);
        free(elems);
        for (i = 0; i < points_count; i++) {
            free(points[i]);
        }
        free(points);
    }
    if ( ~strcmp(part_type, "dual") ) {
        free(epart);
        if (my_rank == 0) free(npart);
    }

    free(local_global_index);
    free(recv_count);
    free(send_count);
    free(cnorm);
    free(oc);
    free(var);
    free(cgup);
    free(su);
    free(bp);
    free(bh);
    free(bl);
    free(bw);
    free(bn);
    free(be);
    free(bs);

    for (i = 0; i < nintcf + 1; i++) {
        free(lcc[i]);
    }
    free(lcc);

    for (i = 0; i < num_procs; i++) {
        free(send_list[i]);
        free(recv_list[i]);
    }
    free(send_list);
    free(recv_list);

    Program_Message("Simulation finished!");

    MPI_Finalize();   /// Cleanup MPI

    return 0;
}

