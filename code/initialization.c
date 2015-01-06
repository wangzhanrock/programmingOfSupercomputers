/**
 * Initialization step - parse the input file, compute data distribution, initialize LOCAL computational arrays
 *
 * @date 22-Oct-2012
 * @author V. Petkov
 */

#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "metis.h"
#include "mpi.h"
#include "util_read_files.h"
#include "initialization.h"

// #include "scorep/SCOREP_User.h"

int initialization(char* file_in, char* part_type, int* nintci, int* nintcf,
                   int* nextci, int* nextcf, int*** lcc, double** bs, double** be,
                   double** bn, double** bw, double** bl, double** bh, double** bp,
                   double** su, int* points_count, int*** points, int** elems,
                   double** var, double** cgup, double** oc, double** cnorm,
                   int** local_global_index, int** global_local_index,
                   int* neighbors_count, int** send_count, int*** send_list,
                   int** recv_count, int*** recv_list, int** epart, int** npart,
                   int* objval, int** local_global_index_internalSet) {
    /********** START INITIALIZATION **********/
    int i = 0, j = 0, k = 0;
    int my_rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);  // Get current process id
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Status status, status2;
    MPI_Request request[num_procs - 1], request2, request3[num_procs - 1], request4;
    int num_elems_internal_local;
    int *num_elems_internal_local_set = NULL;
    int *num_LCC_internal_local_set = NULL;  // only allocated memory in root.
    int *displs_LCC = NULL;  // only allocated memory in root.
    double *bs_local, *be_local, *bn_local, *bw_local, *bl_local, *bh_local,
           *bp_local, *su_local;
    int *lcc_local;
    double *bs_ordered = NULL, *be_ordered = NULL, *bn_ordered = NULL,
            *bw_ordered = NULL, *bl_ordered = NULL, *bh_ordered = NULL,
             *bp_ordered = NULL, *su_ordered = NULL;
    int *lcc_ordered = NULL;
    int num_points_allocated, *point_allocated_stamp;
    int *displs = NULL;
    int num_elems_internal, num_nodes_internal;

    // read-in the input file
    if (~strcmp(part_type, "dual")) {
        if (my_rank == 0) {
//             SCOREP_USER_REGION_DEFINE(RI_Phase);
//             SCOREP_USER_OA_PHASE_BEGIN(RI_Phase, "RI_Phase", SCOREP_USER_REGION_TYPE_COMMON);
            int f_status = read_binary_geo(file_in, &*nintci, &*nintcf,
                                           &*nextci, &*nextcf, &*lcc, &*bs, &*be, &*bn, &*bw, &*bl,
                                           &*bh, &*bp, &*su, &*points_count, &*points, &*elems);
//             SCOREP_USER_OA_PHASE_END(RI_Phase);
            if (f_status != 0) {
                return f_status;
            }

            /**********Devide the mesh by calling Metis***************/
//             printf("nintci=%d,nintcf=%d,nextci=%d,nextcf=%d\n", (*nintci), (*nintcf),
//                                                                 (*nextci), (*nextcf));

            num_elems_internal = (*nintcf - *nintci) + 1;
//             printf("points_count=%d\n", *points_count);
            num_nodes_internal = num_elems_internal * 8;
        }

        MPI_Bcast(&num_elems_internal, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&num_nodes_internal, 1, MPI_INT, 0, MPI_COMM_WORLD);
        // allcoate memory for epart adn npart in other processors
        if (my_rank != 0) {
            (*epart) = (int*) calloc(sizeof(int), num_elems_internal);
        }

        if (my_rank == 0) {
//             SCOREP_USER_REGION_DEFINE(Metis_Phase);
//          SCOREP_USER_OA_PHASE_BEGIN(Metis_Phase, "Metis_Phase", SCOREP_USER_REGION_TYPE_COMMON);
            idx_t ne = (idx_t) num_elems_internal;
            idx_t nn = (idx_t) *points_count;
            idx_t ncommon = 4;
            idx_t nparts = num_procs;
            idx_t objval_METIS;
            int METIS_status;
            idx_t options[METIS_NOPTIONS];

            METIS_SetDefaultOptions(options);
            options[METIS_OPTION_PTYPE] = METIS_PTYPE_KWAY;

            idx_t *eptr = (idx_t*) calloc(sizeof(idx_t), num_elems_internal + 1);
            for (i = 0; i < num_elems_internal + 1; i++) {
                eptr[i] = (idx_t) i * 8;
            }

            // transform the type of node arrays *(elems) from int to idx_t
            idx_t *eind = (idx_t*) calloc(sizeof(idx_t), num_nodes_internal);
            for (i = 0; i < num_nodes_internal; i++) {
                eind[i] = (idx_t) (*elems)[i];
            }
            (*epart) = (int*) calloc(sizeof(int), num_elems_internal);
            (*npart) = (int*) calloc(sizeof(int), num_nodes_internal);
            idx_t *epart_METIS = (idx_t*) calloc(sizeof(idx_t), num_elems_internal);
            idx_t *npart_METIS = (idx_t*) calloc(sizeof(idx_t), num_nodes_internal);

            METIS_status = METIS_PartMeshDual(&ne, &nn, eptr, eind, NULL, NULL,
                                              &ncommon, &nparts, NULL, options, &objval_METIS,
                                              epart_METIS, npart_METIS);

            // transform the type of epart_METIS and npart_METIS from idx_t to int.
            for (i = 0; i < num_elems_internal; i++) {
                (*epart)[i] = (int) epart_METIS[i];
            }
//             for (i = 0; i < num_nodes_internal; i++) {
//                 (*npart)[i] = (int) npart_METIS[i];
//             }

            for (i = 1; i < num_procs; i++) {
                MPI_Isend((*epart), num_elems_internal, MPI_INT, i, 40, MPI_COMM_WORLD,
                          &request[i - 1]);
            }

            free(eptr);
            free(eind);
            free(epart_METIS);
            free(npart_METIS);
            // debug information
            printf("METIS_status=%d", METIS_status);
            Program_Message("after metis");
//             SCOREP_USER_OA_PHASE_END(Metis_Phase);
        }

        if (my_rank != 0) {
            MPI_Irecv((*epart), num_elems_internal, MPI_INT, 0, 40, MPI_COMM_WORLD, &request2);
        }

            /***send the bs, be, bn, bw, bl, bh, bp, su, local_global_index to other processors***/
        if (my_rank == 0) {
            num_points_allocated = 0;
            point_allocated_stamp = (int*) calloc(sizeof(int), num_procs + 1);
            (*local_global_index_internalSet) = (int*) calloc(sizeof(int), num_elems_internal);

            // build the local_global_index_internalSet for whole internal elements.
            for (j  = 0; j < num_procs; j++) {
                point_allocated_stamp[j] = num_points_allocated;
//                 int local_index = 0;
                for (i = 0; i < num_elems_internal; i++) {
                    if ((*epart)[i] == j) {
                        (*local_global_index_internalSet)[num_points_allocated] = i;
//                         global_local_index_internalSet[i] = local_index;
//                         local_index++;
                        num_points_allocated++;
                    }
                }
            }
            point_allocated_stamp[num_procs] = num_points_allocated;

            // calculate the num of internal elements each processor get AND send them
            num_elems_internal_local_set = (int*) calloc(sizeof(int), num_procs);
            for (i = 0; i < num_procs; i++) {
                num_elems_internal_local_set[i] = point_allocated_stamp[i + 1]
                                                  - point_allocated_stamp[i];
            }

            num_elems_internal_local = num_elems_internal_local_set[0];
            for (i = 1; i < num_procs; i++) {
                MPI_Isend(&num_elems_internal_local_set[i], 1, MPI_INT, i, 70, MPI_COMM_WORLD,
                          &request3[i - 1]);
            }
        }  // rank=0;
        if (my_rank != 0) {
            MPI_Irecv(&num_elems_internal_local, 1, MPI_INT, i, 70, MPI_COMM_WORLD, &request4);
        }

        if (my_rank == 0) {
            // reorder arrays: only internal elements is send, external elements is still there.
            // the size of reordered array is num_elems_internal !!!!!
            bs_ordered = (double *) calloc(sizeof(double), num_elems_internal);
            be_ordered = (double *) calloc(sizeof(double), num_elems_internal);
            bn_ordered = (double *) calloc(sizeof(double), num_elems_internal);
            bw_ordered = (double *) calloc(sizeof(double), num_elems_internal);
            bl_ordered = (double *) calloc(sizeof(double), num_elems_internal);
            bh_ordered = (double *) calloc(sizeof(double), num_elems_internal);
            bp_ordered = (double *) calloc(sizeof(double), num_elems_internal);
            su_ordered = (double *) calloc(sizeof(double), num_elems_internal);
            lcc_ordered = (int *) calloc(sizeof(int), num_elems_internal * 6);
            // reorder the array according wicich processor the elements is going to be send to.
            for (i = 0; i < num_elems_internal; i++) {
                bs_ordered[i] = (*bs)[(*local_global_index_internalSet)[i]];
                be_ordered[i] = (*be)[(*local_global_index_internalSet)[i]];
                bn_ordered[i] = (*bn)[(*local_global_index_internalSet)[i]];
                bw_ordered[i] = (*bw)[(*local_global_index_internalSet)[i]];
                bl_ordered[i] = (*bl)[(*local_global_index_internalSet)[i]];
                bh_ordered[i] = (*bh)[(*local_global_index_internalSet)[i]];
                bp_ordered[i] = (*bp)[(*local_global_index_internalSet)[i]];
                su_ordered[i] = (*su)[(*local_global_index_internalSet)[i]];
            }
            // build one dimension ordered lcc array
            for (i = 0; i < num_elems_internal; i++) {
                for (j = 0; j < 6; j++) {
                    lcc_ordered[6 * i + j] =
                        (*lcc)[(*local_global_index_internalSet)[i]][j];
                }
            }
            // build displs according point_allocated_stamp
//             displs = (int *) calloc(sizeof(int), num_procs);
            displs = point_allocated_stamp;
            // build num_LCC_internal_local_set
            num_LCC_internal_local_set = (int*) calloc(sizeof(int), num_procs);
            displs_LCC = (int*) calloc(sizeof(int), num_procs);
            for (i = 0; i < num_procs; i++) {
                num_LCC_internal_local_set[i] = 6 * num_elems_internal_local_set[i];
                displs_LCC[i] = 6 * displs[i];
            }
        }  // end of rank 0
        /*************************************DISTRIBUTE DATA*************************************/

        if (my_rank != 0) MPI_Wait(&request4, &status2);

        bs_local = (double *) calloc(sizeof(double), num_elems_internal_local);
        be_local = (double *) calloc(sizeof(double), num_elems_internal_local);
        bn_local = (double *) calloc(sizeof(double), num_elems_internal_local);
        bw_local = (double *) calloc(sizeof(double), num_elems_internal_local);
        bl_local = (double *) calloc(sizeof(double), num_elems_internal_local);
        bh_local = (double *) calloc(sizeof(double), num_elems_internal_local);
        bp_local = (double *) calloc(sizeof(double), num_elems_internal_local);
        su_local = (double *) calloc(sizeof(double), num_elems_internal_local);
        *local_global_index = (int*) calloc(sizeof(int), num_elems_internal_local);
        lcc_local = (int *) calloc(sizeof(int), num_elems_internal_local * 6);

        // send and receive b*, su arrays.
        MPI_Scatterv(bs_ordered, num_elems_internal_local_set, displs,
                     MPI_DOUBLE, bs_local, num_elems_internal_local, MPI_DOUBLE, 0,
                     MPI_COMM_WORLD);
        MPI_Scatterv(be_ordered, num_elems_internal_local_set, displs,
                     MPI_DOUBLE, be_local, num_elems_internal_local, MPI_DOUBLE, 0,
                     MPI_COMM_WORLD);
        MPI_Scatterv(bn_ordered, num_elems_internal_local_set, displs,
                     MPI_DOUBLE, bn_local, num_elems_internal_local, MPI_DOUBLE, 0,
                     MPI_COMM_WORLD);
        MPI_Scatterv(bw_ordered, num_elems_internal_local_set, displs,
                     MPI_DOUBLE, bw_local, num_elems_internal_local, MPI_DOUBLE, 0,
                     MPI_COMM_WORLD);
        MPI_Scatterv(bl_ordered, num_elems_internal_local_set, displs,
                     MPI_DOUBLE, bl_local, num_elems_internal_local, MPI_DOUBLE, 0,
                     MPI_COMM_WORLD);
        MPI_Scatterv(bh_ordered, num_elems_internal_local_set, displs,
                     MPI_DOUBLE, bh_local, num_elems_internal_local, MPI_DOUBLE, 0,
                     MPI_COMM_WORLD);
        MPI_Scatterv(bp_ordered, num_elems_internal_local_set, displs,
                     MPI_DOUBLE, bp_local, num_elems_internal_local, MPI_DOUBLE, 0,
                     MPI_COMM_WORLD);
        MPI_Scatterv(su_ordered, num_elems_internal_local_set, displs,
                     MPI_DOUBLE, su_local, num_elems_internal_local, MPI_DOUBLE, 0,
                     MPI_COMM_WORLD);
        MPI_Scatterv((*local_global_index_internalSet),
                     num_elems_internal_local_set, displs, MPI_INT,
                     *local_global_index, num_elems_internal_local, MPI_INT, 0,
                     MPI_COMM_WORLD);
        MPI_Scatterv(lcc_ordered, num_LCC_internal_local_set, displs_LCC,
                     MPI_INT, lcc_local, 6 * num_elems_internal_local, MPI_INT, 0,
                     MPI_COMM_WORLD);

        /************************BUILDING SEND_LIST AND RECEIVE_LIST******************************/
        *recv_count = (int *) calloc(sizeof(int), num_procs);
        *send_count = (int *) calloc(sizeof(int), num_procs);
        bool *send_mark = (bool *) calloc(sizeof(bool), num_procs);
        int node_rank, node_id;

//         MPI_Wait(&request, &status);
        if (my_rank != 0) MPI_Wait(&request2, &status);

        for (i = 0; i < num_elems_internal_local; i++) {
            for (j = 0; j < 6; j++) {
                node_id = lcc_local[6 * i + j];  // stored global index.
                if (node_id < num_elems_internal) {  // exclude the external cell.
                    node_rank = (*epart)[node_id];
                    if (node_rank != my_rank) {
                        if (!send_mark[node_rank]) {
                            (*send_count)[node_rank]++;
                            send_mark[node_rank] = 1;
                        }
                    }
                }
            }
            for (k = 0; k < num_procs; k ++) {
                send_mark[k] = 0;
            }
        }
        // generate recv_count by MPI_Send send_count
        for (i = 0; i < num_procs; i++) {
            if (my_rank == i) {
                for (j = 0; j < num_procs; j++) {
                    if (j != my_rank) MPI_Send(&(*send_count)[j], 1, MPI_INT, j, 10,
                                               MPI_COMM_WORLD);
                }
            } else {
                MPI_Recv(&(*recv_count)[i], 1, MPI_INT, i, 10, MPI_COMM_WORLD, &status);
            }
        }

//         for (i = 0; i < num_procs; i++) {
//             printf("after calculate count: my_rank=%d, send_count[%d]=%d, recv_count[%d]=%d\n",
//                                             my_rank, i, (*send_count)[i], i, (*recv_count)[i]);
//         }

        // allcoate memory for recv_list and send_list according the count.
        (*recv_list) = (int**) calloc(sizeof(int*), num_procs);
        (*send_list) = (int**) calloc(sizeof(int*), num_procs);
        for (i = 0; i < num_procs; i++) {
            (*recv_list)[i] = (int*) calloc(sizeof(int), (*recv_count)[i]);
            (*send_list)[i] = (int*) calloc(sizeof(int), (*send_count)[i]);
        }

        for (i = 0; i < num_procs; i++) {
            (*send_count)[i] = 0;
        }

        for (i = 0; i < num_elems_internal_local; i++) {
            for (j = 0; j < 6; j++) {
                node_id = lcc_local[6 * i + j];  // stored global index.
                if (node_id < num_elems_internal) {  // exclude the external cell.
                    node_rank = (*epart)[node_id];
                    if (node_rank != my_rank) {
                        if (!send_mark[node_rank]) {
                            // store global index now!!!
                            (*send_list)[node_rank][(*send_count)[node_rank]] =
                                                                          (*local_global_index)[i];
                            (*send_count)[node_rank]++;
                            send_mark[node_rank] = 1;
                        }
                    }
                }
            }
            for (k = 0; k < num_procs; k ++) {
                send_mark[k] = 0;
            }
        }

        // generate recv_list by MPI_Send send_count
        for (i = 0; i < num_procs; i++) {
            if (my_rank == i) {
                for (j = 0; j < num_procs; j++) {
                    if (j != my_rank) {
                        MPI_Send((*send_list)[j], (*send_count)[j], MPI_INT, j, 100,
                                 MPI_COMM_WORLD);
                    }
                }
            } else {
                MPI_Recv((*recv_list)[i], (*recv_count)[i], MPI_INT, i, 100,
                         MPI_COMM_WORLD, &status);
            }
        }


        /************************END OF BUILDING SEND_LIST, RECEIVE_LIST************************/

        /*** ***************************FREE MEMORY*********************************************/
        free(send_mark);
        if (my_rank == 0) {
//             free(global_local_index_internalSet);
            free(num_elems_internal_local_set);
            free(num_LCC_internal_local_set);
            free(point_allocated_stamp);
            free(displs_LCC);
            free(bs_ordered);
            free(be_ordered);
            free(bn_ordered);
            free(bw_ordered);
            free(bl_ordered);
            free(bh_ordered);
            free(bp_ordered);
            free(su_ordered);
            free(lcc_ordered);
        }  // rank==0 end area

    } else {  //  end of dual
        /*****************************************************************************************/
        /*start of classical type partition                                                      */
         /****************************************************************************************/
        if (my_rank == 0) {
            int f_status = read_binary_geo(file_in, &*nintci, &*nintcf,
                                           &*nextci, &*nextcf, &*lcc, &*bs, &*be, &*bn, &*bw, &*bl,
                                           &*bh, &*bp, &*su, &*points_count, &*points, &*elems);

            if (f_status != 0) {
                return f_status;
            }
            printf("nintci=%d,nintcf=%d,nextci=%d,nextcf=%d\n", (*nintci),
                   (*nintcf), (*nextci), (*nextcf));
            printf("points_count=%d\n", *points_count);

            num_elems_internal = (*nintcf - *nintci) + 1;
            num_nodes_internal = num_elems_internal * 8;
            printf("After read: rank=%i, num_elems_internal=%d\n", my_rank, num_elems_internal);
            printf("After read: rank=%i, num_nodes_internal=%d\n", my_rank, num_nodes_internal);

            // devide the elements in to different parts.
            int size = floor(num_elems_internal / num_procs);
            displs = (int *) calloc(sizeof(int), num_procs);
            num_elems_internal_local_set = (int*) calloc(sizeof(int), num_procs);
            for (i = 0; i < num_procs; i++) {
                displs[i] = i * size;
            }
            for (i = 0; i < num_procs - 1; i++) {
                num_elems_internal_local_set[i] = size;
            }
            num_elems_internal_local_set[num_procs - 1] = num_elems_internal -
                                                          size * (num_procs - 1);

            // initialize the local_global_index_internalSet for sending.
            (*local_global_index_internalSet) = (int*) calloc(sizeof(int), num_elems_internal);
            for (i = 0; i < num_elems_internal; i++) {
                (*local_global_index_internalSet)[i] = i;
            }

            lcc_ordered = (int *) calloc(sizeof(int), num_elems_internal * 6);
            for (i = 0; i < num_elems_internal; i++) {
                for (j = 0; j < 6; j++) {
                    lcc_ordered[6 * i + j] = (*lcc)[(*local_global_index_internalSet)[i]][j];
                }
            }
            num_LCC_internal_local_set = (int*) calloc(sizeof(int), num_procs);
            displs_LCC = (int*) calloc(sizeof(int), num_procs);
            for (i = 0; i < num_procs; i++) {
                num_LCC_internal_local_set[i] = 6 * num_elems_internal_local_set[i];
                displs_LCC[i] = 6 * displs[i];
            }
        }  // end of rank 0

//         Program_Sync("Before distribute data");
        MPI_Bcast(&num_elems_internal, 1, MPI_INT, 0, MPI_COMM_WORLD);
        num_nodes_internal = num_elems_internal * 8;

        MPI_Scatter(num_elems_internal_local_set, 1, MPI_INT,
                    &num_elems_internal_local, 1, MPI_INT, 0, MPI_COMM_WORLD);

        bs_local = (double *) calloc(sizeof(double), num_elems_internal_local);
        be_local = (double *) calloc(sizeof(double), num_elems_internal_local);
        bn_local = (double *) calloc(sizeof(double), num_elems_internal_local);
        bw_local = (double *) calloc(sizeof(double), num_elems_internal_local);
        bl_local = (double *) calloc(sizeof(double), num_elems_internal_local);
        bh_local = (double *) calloc(sizeof(double), num_elems_internal_local);
        bp_local = (double *) calloc(sizeof(double), num_elems_internal_local);
        su_local = (double *) calloc(sizeof(double), num_elems_internal_local);
        *local_global_index = (int*) calloc(sizeof(int), num_elems_internal_local);
        lcc_local = (int *) calloc(sizeof(int), num_elems_internal_local * 6);

        //  send and receive b*, su arrays.
        MPI_Scatterv(*bs, num_elems_internal_local_set, displs, MPI_DOUBLE, bs_local,
                     num_elems_internal_local, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(*be, num_elems_internal_local_set, displs, MPI_DOUBLE, be_local,
                     num_elems_internal_local, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(*bn, num_elems_internal_local_set, displs, MPI_DOUBLE,
                     bn_local, num_elems_internal_local, MPI_DOUBLE, 0,
                     MPI_COMM_WORLD);
        MPI_Scatterv(*bw, num_elems_internal_local_set, displs, MPI_DOUBLE,
                     bw_local, num_elems_internal_local, MPI_DOUBLE, 0,
                     MPI_COMM_WORLD);
        MPI_Scatterv(*bl, num_elems_internal_local_set, displs, MPI_DOUBLE,
                     bl_local, num_elems_internal_local, MPI_DOUBLE, 0,
                     MPI_COMM_WORLD);
        MPI_Scatterv(*bh, num_elems_internal_local_set, displs, MPI_DOUBLE,
                     bh_local, num_elems_internal_local, MPI_DOUBLE, 0,
                     MPI_COMM_WORLD);
        MPI_Scatterv(*bp, num_elems_internal_local_set, displs, MPI_DOUBLE,
                     bp_local, num_elems_internal_local, MPI_DOUBLE, 0,
                     MPI_COMM_WORLD);
        MPI_Scatterv(*su, num_elems_internal_local_set, displs, MPI_DOUBLE,
                     su_local, num_elems_internal_local, MPI_DOUBLE, 0,
                     MPI_COMM_WORLD);
        MPI_Scatterv((*local_global_index_internalSet),
                     num_elems_internal_local_set, displs, MPI_INT,
                     *local_global_index, num_elems_internal_local, MPI_INT, 0,
                     MPI_COMM_WORLD);
        MPI_Scatterv(lcc_ordered, num_LCC_internal_local_set, displs_LCC,
                     MPI_INT, lcc_local, 6 * num_elems_internal_local, MPI_INT, 0,
                     MPI_COMM_WORLD);

        /*********************BUILDING SEND_LIST AND RECEIVE_LIST********************************/
        *recv_count = (int *) calloc(sizeof(int), num_procs);
        *send_count = (int *) calloc(sizeof(int), num_procs);
        bool *send_mark = (bool *) calloc(sizeof(bool), num_procs);

        int size = floor(num_elems_internal / num_procs);
        int node_rank, node_id;

        for (i = 0; i < num_elems_internal_local; i++) {
            for (j = 0; j < 6; j++) {
                node_id = lcc_local[6 * i + j];  // stored global index.
                if (node_id < num_elems_internal) {  // exclude the external cell.
                    // calculate the rank of each cell
                    if (floor(node_id / size) != num_procs) {
                        node_rank = floor(node_id / size);
                    } else {
                        node_rank = num_procs - 1;
                    }

                    if (node_rank != my_rank) {
                        if (!send_mark[node_rank]) {
                            (*send_count)[node_rank]++;
                            send_mark[node_rank] = 1;
                        }
                    }
                }
            }
            for (k = 0; k < num_procs; k ++) {
                send_mark[k] = 0;
            }
        }
        // get the recv_count by recving send_count
        for (i = 0; i < num_procs; i++) {
            if (my_rank == i) {
                for (j = 0; j < num_procs; j++) {
                    if (j != my_rank) MPI_Send(&(*send_count)[j], 1, MPI_INT, j, 10,
                                               MPI_COMM_WORLD);
                }
            } else {
                MPI_Recv(&(*recv_count)[i], 1, MPI_INT, i, 10, MPI_COMM_WORLD, &status);
            }
        }

        // allcoate memory for recv_list and send_list according the count.
        (*recv_list) = (int**) calloc(sizeof(int*), num_procs);
        (*send_list) = (int**) calloc(sizeof(int*), num_procs);

        for (i = 0; i < num_procs; i++) {
            (*recv_list)[i] = (int*) calloc(sizeof(int), (*recv_count)[i]);
            (*send_list)[i] = (int*) calloc(sizeof(int), (*send_count)[i]);
        }

        for (i = 0; i < num_procs; i++) {
            (*send_count)[i] = 0;
        }

        for (i = 0; i < num_elems_internal_local; i++) {
            for (j = 0; j < 6; j++) {
                node_id = lcc_local[6 * i + j];  // stored global index.
                if (node_id < num_elems_internal) {  // exclude the external cell.
                    if (floor(node_id / size) != num_procs) {
                        node_rank = floor(node_id / size);
                    } else {
                        node_rank = num_procs - 1;
                    }
                    if (node_rank != my_rank) {
                        if (!send_mark[node_rank]) {
                            // store: local index ---> global index now!!!
                            (*send_list)[node_rank][(*send_count)[node_rank]] =
                                                                          (*local_global_index)[i];
                            (*send_count)[node_rank]++;
                            send_mark[node_rank] = 1;
                        }
                    }
                }
            }
            for (k = 0; k < num_procs; k ++) {
                send_mark[k] = 0;
            }
        }

        // generate recv_list by MPI_Send send_count
        for (i = 0; i < num_procs; i++) {
            if (my_rank == i) {
                for (j = 0; j < num_procs; j++) {
                    if (j != my_rank) {
                        MPI_Send((*send_list)[j], (*send_count)[j], MPI_INT, j, 100,
                                 MPI_COMM_WORLD);
                    }
                }
            } else {
                MPI_Recv((*recv_list)[i], (*recv_count)[i], MPI_INT, i, 100,
                         MPI_COMM_WORLD, &status);
            }
        }

        /*****************************END OF BUILDING SEND_LIST, RECEIVE_LIST********************/
        free(send_mark);
        if (my_rank == 0) {
            free(num_elems_internal_local_set);
            free(num_LCC_internal_local_set);
            free(displs);
            free(displs_LCC);
            free(lcc_ordered);
        }  // rank==0 end area
    }
    /**********************************************************************************************/
    /*  END of classical type partition                                                           */
     /*********************************************************************************************/
     //  return local value back
     (*bs) = bs_local;
     (*be) = be_local;
     (*bn) = bn_local;
     (*bw) = bw_local;
     (*bl) = bl_local;
     (*bh) = bh_local;
     (*bp) = bp_local;
     (*su) = su_local;
     (*nintci) = 0;
     (*nintcf) = num_elems_internal_local - 1;
     MPI_Bcast(nextcf, 1, MPI_INT, 0, MPI_COMM_WORLD);

     // change lcc from one dimentision back to 2 dimensions: global index stored in lcc
     if (my_rank != 0) {
        if ( (*lcc = (int**) malloc((*nintcf + 1) * sizeof(int*))) == NULL ) {
            fprintf(stderr, "malloc failed to allocate first dimension of LCC");
            return -1;
        }

        for ( i = 0; i < *nintcf + 1; i++ ) {
            if ( ((*lcc)[i] = (int *) malloc(6 * sizeof(int))) == NULL ) {
                fprintf(stderr, "malloc failed to allocate second dimension of LCC\n");
                return -1;
            }
        }
     }

     for (i = 0; i < num_elems_internal_local; i++) {
         for (j = 0; j < 6; j++) {
             (*lcc)[i][j] = lcc_local[6 * i + j];
         }
     }

     free(lcc_local);

     *var = (double*) calloc(sizeof(double), (*nintcf + 1));  // change  from nextcf to nintcf
     *cgup = (double*) calloc(sizeof(double), (*nintcf + 1));  // change from nextcf to nintcf
     *oc = (double*) calloc(sizeof(double), (10 + 1));  // not be used!!!!  nintcf --> 0
     *cnorm = (double*) calloc(sizeof(double), (10 + 1));  // change the size from nintcf to 10

     // initialize the arrays
     for ( i = 0; i <= 10; i++ ) {
         (*oc)[i] = 0.0;
         (*cnorm)[i] = 1.0;
     }

     for ( i = (*nintci); i <= (*nintcf); i++ ) {
         (*cgup)[i] = 0.0;
         (*var)[i] = 0.0;
     }

//      for ( i = (*nextci); i <= (*nextcf); i++ ) {
//          (*var)[i] = 0.0;
//          (*cgup)[i] = 0.0;
//          (*bs)[i] = 0.0;
//          (*be)[i] = 0.0;
//          (*bn)[i] = 0.0;
//          (*bw)[i] = 0.0;
//          (*bl)[i] = 0.0;
//          (*bh)[i] = 0.0;
//      }

     for ( i = (*nintci); i <= (*nintcf); i++ )
         (*cgup)[i] = 1.0 / ((*bp)[i]);

//     printf("before return in initialization: my_rank=%d,nintci=%d,nintcf=%d,nextcf=%d\n",
//                                             my_rank, *nintci, *nintcf, *nextcf);
    return 0;
}

void Program_Message(char *txt) {
/* produces a stderr text output  */
    int myrank;

    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    fprintf(stderr, "-MESSAGE- P:%2d : %s\n", myrank, txt);
    fflush(stdout);
    fflush(stderr);
}

void Program_Sync(char *txt) {
/* produces a stderr textoutput and synchronize all processes */
    int myrank;

    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Barrier(MPI_COMM_WORLD); /* synchronize output */
    fprintf(stderr, "-MESSAGE- P:%2d : %s\n", myrank, txt);
    fflush(stdout);
    fflush(stderr);
    MPI_Barrier(MPI_COMM_WORLD);
}

void Program_Stop(char *txt) {
/* all processes will produce a text output, be synchronized and finished */
    int myrank;

    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Barrier(MPI_COMM_WORLD); /* synchronize output */
    fprintf(stderr, "-STOP- P:%2d : %s\n", myrank, txt);
    fflush(stdout);
    fflush(stderr);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    exit(1);
}

