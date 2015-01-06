/**
 * Computational loop
 *
 * @file compute_solution.c
 * @date 22-Oct-2012
 * @author V. Petkov
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"
#include "initialization.h"

int compute_solution(const int max_iters, int nintci, int nintcf, int nextcf, int** lcc, double* bp,
                     double* bs, double* bw, double* bl, double* bn, double* be, double* bh,
                     double* cnorm, double* var, double *su, double* cgup, double* residual_ratio,
                     int* local_global_index, int* global_local_index, int neighbors_count,
                     int* send_count, int** send_list, int* recv_count, int** recv_list) {
    int iter = 1;
    int if1 = 0;
    int if2 = 0;
    int nor = 1;
    int nor1 = nor - 1;
    int nc = 0;
    int my_rank, num_procs;
    int i = 0, j = 0;
    MPI_Status status;
    int nc_global;

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);  // Get current process id
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // allocate arrays used in gccg
    int nomax = 3;

    /** the reference residual*/
    double resref = 0.0;
    double resref_global = 0.0;

    /** array storing residuals */
    double *resvec = (double *) calloc(sizeof(double), (nintcf + 1));

    // initialize the reference residual
    for ( nc = nintci; nc <= nintcf; nc++ ) {
        resvec[nc] = su[nc];
        resref = resref + resvec[nc] * resvec[nc];
    }
    // Reduce resref to root processor
    MPI_Allreduce(&resref, &resref_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    resref_global = sqrt(resref_global);

    if (my_rank == 0) {
        if ( resref_global < 1.0e-15 ) {
            fprintf(stderr, "Residue sum less than 1.e-15 - %lf\n", resref_global);
            return 0;
        }
    }

    /** the computation vectors */
    double *direc1 = (double *) calloc(sizeof(double), (nextcf + 1));
    double *direc2 = (double *) calloc(sizeof(double), (nintcf + 1));   // change nex to nin
    double *adxor1 = (double *) calloc(sizeof(double), (nintcf + 1));
    double *adxor2 = (double *) calloc(sizeof(double), (nintcf + 1));
    double *dxor1 = (double *) calloc(sizeof(double), (nintcf + 1));
    double *dxor2 = (double *) calloc(sizeof(double), (nintcf + 1));
    double cnorm_global[4];

    // prepare the cnorm for the first iteration.
    for ( i = 0; i < 4; i++ ) {
        cnorm_global[i] = 1.0;
    }

    // debug info
//     printf("ComputePhase: my_rank=%d,nintci=%d,nintcf=%d,nextcf=%d\n",
//                           my_rank, nintci, nintcf, nextcf);

    // creat new send data type here.
    MPI_Datatype send_type[num_procs], recv_type[num_procs];
    int *block_length;
    for (i =0; i < num_procs; i++) {
        if (send_count[i] != 0) {
            block_length = (int *) malloc(sizeof(int) * send_count[i]);
            for (j = 0; j < send_count[i]; j++) {
                block_length[j] = 1;
            }
            MPI_Type_indexed(send_count[i], block_length, send_list[i], MPI_DOUBLE, &send_type[i]);
            MPI_Type_commit(&send_type[i]);
            free(block_length);
        }

        if (recv_count[i] != 0) {
            block_length = (int *) malloc(sizeof(int) * recv_count[i]);
            for (j = 0; j < recv_count[i]; j++) {
                block_length[j] = 1;
            }
            MPI_Type_indexed(recv_count[i], block_length, recv_list[i], MPI_DOUBLE, &recv_type[i]);
            MPI_Type_commit(&recv_type[i]);
            free(block_length);
        }
    }

//     Program_Sync("After define the new MPI send and recv data type.");

    while ( iter < max_iters ) {
        /**********  START COMP PHASE 1 **********/
        // update the old values of direc
        for ( nc = nintci; nc <= nintcf; nc++ ) {
            nc_global = local_global_index[nc];
            direc1[nc_global] = direc1[nc_global] + resvec[nc] * cgup[nc];
        }

        // direc1_communication
        for (i = 0; i < num_procs; i++) {
            if (send_count[i] != 0) {
                MPI_Sendrecv(direc1, 1, send_type[i], i, 1000,
                             direc1, 1, recv_type[i], i, 1000, MPI_COMM_WORLD, &status);
            }
        }

        // compute new guess (approximation) for direc
        for ( nc = nintci; nc <= nintcf; nc++ ) {
            direc2[nc] = bp[nc] * direc1[local_global_index[nc]] - bs[nc] * direc1[lcc[nc][0]]
                         - bw[nc] * direc1[lcc[nc][3]] - bl[nc] * direc1[lcc[nc][4]]
                         - bn[nc] * direc1[lcc[nc][2]] - be[nc] * direc1[lcc[nc][1]]
                         - bh[nc] * direc1[lcc[nc][5]];
        }

        /********** END COMP PHASE 1 **********/
        /********** START COMP PHASE 2 **********/
        // execute normalization steps
        double oc1, oc2, occ, occ_global;
        if ( nor1 == 1 ) {
            oc1 = 0;
            occ = 0;

            for ( nc = nintci; nc <= nintcf; nc++ ) {
                occ = occ + adxor1[nc] * direc2[nc];
            }
            // all_reduce occ
            // oc1 = occ / cnorm[1];
            MPI_Allreduce(&occ, &occ_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            oc1 = occ_global / cnorm_global[1];

            for ( nc = nintci; nc <= nintcf; nc++ ) {
                nc_global = local_global_index[nc];
                direc2[nc] = direc2[nc] - oc1 * adxor1[nc];
                direc1[nc_global] = direc1[nc_global] - oc1 * dxor1[nc];
            }

            if1++;
        } else {
            if ( nor1 == 2 ) {
                oc1 = 0;
                occ = 0;

                for ( nc = nintci; nc <= nintcf; nc++ ) {
                    occ = occ + adxor1[nc] * direc2[nc];
                }
                // oc1 = occ / cnorm[1];
                MPI_Allreduce(&occ, &occ_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                oc1 = occ_global / cnorm_global[1];

                oc2 = 0;
                occ = 0;
                for ( nc = nintci; nc <= nintcf; nc++ ) {
                    occ = occ + adxor2[nc] * direc2[nc];
                }
                // oc2 = occ / cnorm[2];
                MPI_Allreduce(&occ, &occ_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                oc2 = occ_global / cnorm_global[2];

                for ( nc = nintci; nc <= nintcf; nc++ ) {
                    nc_global = local_global_index[nc];
                    direc2[nc] = direc2[nc] - oc1 * adxor1[nc] - oc2 * adxor2[nc];
                    direc1[nc_global] = direc1[nc_global] - oc1 * dxor1[nc] - oc2 * dxor2[nc];
                }

                if2++;
            }
        }

        // compute the new residual
        cnorm[nor] = 0;
        double omega = 0;
        double omega_global;

        for ( nc = nintci; nc <= nintcf; nc++ ) {
            cnorm[nor] = cnorm[nor] + direc2[nc] * direc2[nc];
            omega = omega + resvec[nc] * direc2[nc];
        }
        // omega = omega / cnorm[nor];
        // reduce cnorm[nor] omega
        MPI_Allreduce(&cnorm[nor], &cnorm_global[nor], 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&omega, &omega_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        omega = omega_global / cnorm_global[nor];

        double res_updated = 0.0, res_updated_global = 0.0;
        for ( nc = nintci; nc <= nintcf; nc++ ) {
            var[nc] = var[nc] + omega * direc1[local_global_index[nc]];
            resvec[nc] = resvec[nc] - omega * direc2[nc];
            res_updated = res_updated + resvec[nc] * resvec[nc];
        }

//         res_updated = sqrt(res_updated);
//         *residual_ratio = res_updated / resref;
//         // exit on no improvements of residual
//         if ( *residual_ratio <= 1.0e-10 ) break;
        MPI_Allreduce(&res_updated, &res_updated_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        res_updated_global = sqrt(res_updated_global);
        *residual_ratio = res_updated_global / resref_global;
        if ( *residual_ratio <= 1.0e-10 ) break;

        iter++;

        // prepare additional arrays for the next iteration step
        if ( nor == nomax ) {
            nor = 1;
        } else {
            if ( nor == 1 ) {
                for ( nc = nintci; nc <= nintcf; nc++ ) {
                    dxor1[nc] = direc1[local_global_index[nc]];
                    adxor1[nc] = direc2[nc];
                }
            } else {
                if ( nor == 2 ) {
                    for ( nc = nintci; nc <= nintcf; nc++ ) {
                        dxor2[nc] = direc1[local_global_index[nc]];
                        adxor2[nc] = direc2[nc];
                    }
                }
            }
            nor++;
        }
        nor1 = nor - 1;
        /********** END COMP PHASE 2 **********/
//         Program_Message("END COMP PHASE2");
    }

    free(resvec);
    free(direc1);
    free(direc2);
    free(adxor1);
    free(adxor2);
    free(dxor1);
    free(dxor2);

    return iter;
}


