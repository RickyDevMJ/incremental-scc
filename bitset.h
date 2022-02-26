#include <stdio.h>
#include <vector>
#include <set>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <fstream>
#include <sstream>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>

#define debug 0
#define INIT_COLOR 1
// 2^20 = 1048576
#define PIVOT_HASH_CONST 1048575
#define BLOCK_SIZE 1024

#define CUDA_SAFE_CALL_NO_SYNC( call) {                                        \
	cudaError err = call;                                                      \
    if( cudaSuccess != err) {                                                  \
        fprintf(stderr, "error %d: Cuda error in file '%s' in line %i : %s.\n",\
                err, __FILE__, __LINE__, cudaGetErrorString( err) );           \
        exit(EXIT_FAILURE);                                                    \
    } }

#  define CUDA_SAFE_CALL( call)     CUDA_SAFE_CALL_NO_SYNC(call);

static unsigned CudaTest(const char *msg) {
	cudaError_t e;
	if (cudaSuccess != (e = cudaGetLastError())) {
		fprintf(stderr, "%s: %d\n", msg, e); 
		fprintf(stderr, "%s\n", cudaGetErrorString(e));
		exit(-1);
	}
	return 0;
}

void fwd_reach(long n, long *out_row_offsets, long *out_column_indices, unsigned *colors, unsigned char *status, long *scc_root);
void fwd_reach_lb(long n, long *out_row_offsets, long *out_column_indices, unsigned char *status, long *scc_root);
void bwd_reach(long n, long *in_row_offsets, long *in_column_indices, unsigned *colors, unsigned char *status);
void bwd_reach_lb(long n, long *in_row_offsets, long *in_column_indices, unsigned char *status);
void iterative_trim(long n, long *in_row_offsets, long *in_column_indices, long *out_row_offsets, long *out_column_indices, unsigned *colors, unsigned char *status, long *scc_root);
void first_trim(long n, long *in_row_offsets, long *in_column_indices, long *out_row_offsets, long *out_column_indices, unsigned char *status);
void trim2(long n, long *in_row_offsets, long *in_column_indices, long *out_row_offsets, long *out_column_indices, unsigned *colors, unsigned char *status, long *scc_root);
bool update(long n, unsigned *colors, unsigned char *status, unsigned *locks, long *scc_root);
void update_colors(long n, unsigned *colors, unsigned char *status);
void find_removed_vertices(long n, unsigned char *status, long *mark);
void print_statistics(long n, long *scc_root, unsigned char *status);
bool find_wcc(long n, long *d_row_offsets, long *d_column_indices, unsigned *d_colors, unsigned char *d_status, long *scc_root, unsigned min_color);

__host__ __device__ inline bool is_bwd_visited(unsigned char states) {
	return (states & 1);
}

__host__ __device__ inline bool is_fwd_visited(unsigned char states) {
	return (states & 2);
}

__host__ __device__ inline bool is_removed(unsigned char states) {
	return (states & 4);
}

__host__ __device__ inline bool is_trimmed(unsigned char states) {
	return (states & 8);
}

__host__ __device__ inline bool is_pivot(unsigned char states) {
	return (states & 16);
}

__host__ __device__ inline bool is_bwd_extended(unsigned char states) {
	return (states & 32);
}

__host__ __device__ inline bool is_fwd_extended(unsigned char states) {
	return (states & 64);
}

__host__ __device__ inline bool is_bwd_front(unsigned char states) {
	return ((states & 37) == 1);
}

__host__ __device__ inline bool is_fwd_front(unsigned char states) {
	return ((states & 70) == 2);
}

__host__ __device__ inline void set_bwd_visited(unsigned char *states) {
	*states |= 1;
}

__host__ __device__ inline void set_fwd_visited(unsigned char *states) {
	*states |= 2;
}

__host__ __device__ inline void set_removed(unsigned char *states) {
	*states |= 4;
}

__host__ __device__ inline void set_trimmed(unsigned char *states) {
	*states |= 8;
}

__host__ __device__ inline void set_pivot(unsigned char *states) {
	*states |= 16;
}

__host__ __device__ inline void set_bwd_expanded(unsigned char *states) {
	*states |= 32;
}

__host__ __device__ inline void set_fwd_expanded(unsigned char *states) {
	*states |= 64;
}
