/**
 * Initialization step - parse the input file, compute data distribution, initialize LOCAL computational arrays
 *
 * @date 22-Oct-2012
 * @author V. Petkov
 */

#ifndef INITIALIZATION_H_
#define INITIALIZATION_H_

int initialization(char* file_in, char* part_type, int* nintci, int* nintcf,
                   int* nextci, int* nextcf, int*** lcc, double** bs, double** be,
                   double** bn, double** bw, double** bl, double** bh, double** bp,
                   double** su, int* points_count, int*** points, int** elems,
                   double** var, double** cgup, double** oc, double** cnorm,
                   int** local_global_index, int** global_local_index,
                   int* neighbors_count, int** send_count, int*** send_list,
                   int** recv_count, int*** recv_list, int** epart, int** npart,
                   int* objval, int** local_global_index_internalSet);

// double* reorder_array(double * array_oringnal, int num_elems_internal);
void Program_Message(char *txt);
/* produces a stderr text output  */

void Program_Sync(char *txt);
/* produces a stderr textoutput and synchronize all processes */

void Program_Stop(char *txt);
/* all processes will produce a text output, be synchronized and finished */

#endif /* INITIALIZATION_H_ */

