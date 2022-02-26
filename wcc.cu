#include "bitset.h"
#define debug_wcc 0

static __global__ void wcc_min(long n, long *row_offsets, long *column_indices, unsigned *colors, unsigned char *status, unsigned *wcc, bool *changed) {
	long src = blockDim.x * blockIdx.x + threadIdx.x;
	if (src < n && !is_removed(status[src])) {
		long row_begin = row_offsets[src];
		long row_end = row_offsets[src + 1] - 1;
		unsigned wcc_src = wcc[src];
		for(long offset = row_begin; offset < row_end; offset ++) {
			long dst = column_indices[offset];
			if(!is_removed(status[dst]) && colors[src] == colors[dst]) {
				if (wcc[dst] < wcc_src) {
					wcc_src = wcc[dst];
					*changed = true;
				}
			}
		}

		long *updates = (long *) column_indices[row_end];
		while (updates != NULL) {
			long length = updates[0];
			for (long i = 1; i <= length; i++) {
				long dst = updates[i];
				if(!is_removed(status[dst]) && colors[src] == colors[dst]) {
					if (wcc[dst] < wcc_src) {
						wcc_src = wcc[dst];
						*changed = true;
					}
				}
			}

			updates = (long *) updates[length + 1];
		}
		
		wcc[src] = wcc_src;
	}
}

static __global__ void wcc_update(long n, unsigned char *status, unsigned *wcc, bool *changed) {
	long src = blockDim.x * blockIdx.x + threadIdx.x;
	if (src < n && !is_removed(status[src])) {
		unsigned wcc_src = wcc[src];
		unsigned wcc_k = wcc[wcc_src];
		if (wcc_src != src && wcc_src != wcc_k) {
			wcc[src] = wcc_k;
			*changed = true;
		}
	}
}

static __global__ void update_pivot_color(long n, unsigned *wcc, unsigned *colors, unsigned char *status, bool *has_pivot, long *scc_root, unsigned *min_color) {
	long src = blockDim.x * blockIdx.x + threadIdx.x;
	if (src < n && !is_removed(status[src])) {
		if (wcc[src] == src) {
			unsigned new_color = atomicAdd(min_color, 1); 
			//printf("wcc: select vertex %ld as pivot, old_color=%u, new_color=%u\n", src, colors[src], new_color);
			colors[src] = new_color;
			status[src] = 19; // set as a pivot
			scc_root[src] = src;
			*has_pivot = true;
		}
	}
}

static __global__ void update_colors(long n, unsigned *wcc, unsigned *colors, unsigned char *status) {
	long src = blockIdx.x * blockDim.x + threadIdx.x;
	if (src < n && !is_removed(status[src])) {
		unsigned wcc_src = wcc[src];
		if (wcc_src != src)
			colors[src] = colors[wcc_src];
	}
}

bool find_wcc(long n, long *d_row_offsets, long *d_column_indices, unsigned *d_colors, unsigned char *d_status, long *d_scc_root, unsigned min_color) {
	bool h_changed, *d_changed;
	long iter = 0;
	unsigned *d_wcc, *d_min_color;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_wcc, sizeof(unsigned) * n));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_changed, sizeof(bool)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_min_color, sizeof(unsigned)));
	CUDA_SAFE_CALL(cudaMemcpy(d_min_color, &min_color, sizeof(unsigned), cudaMemcpyHostToDevice));
	long nthreads = BLOCK_SIZE;
	long nblocks = (n - 1) / nthreads + 1;
	bool has_pivot = false;
	bool *d_has_pivot;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_has_pivot, sizeof(bool)));
	CUDA_SAFE_CALL(cudaMemcpy(d_has_pivot, &has_pivot, sizeof(bool), cudaMemcpyHostToDevice));
	thrust::sequence(thrust::device, d_wcc, d_wcc + n);
	do {
		++ iter;
		h_changed = false;
		CUDA_SAFE_CALL(cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice));
		wcc_min<<<nblocks, nthreads>>>(n, d_row_offsets, d_column_indices, d_colors, d_status, d_wcc, d_changed);
		CudaTest("solving kernel wcc_min failed");
		wcc_update<<<nblocks, nthreads>>>(n, d_status, d_wcc, d_changed);
		CudaTest("solving kernel wcc_update failed");
		CUDA_SAFE_CALL(cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost));
	} while (h_changed);
	//CUDA_SAFE_CALL(cudaDeviceSynchronize());
	if(debug_wcc) {
		unsigned *h_wcc = (unsigned *)malloc(n*sizeof(unsigned));
		CUDA_SAFE_CALL(cudaMemcpy(h_wcc, d_wcc, n*sizeof(unsigned), cudaMemcpyDeviceToHost));
		FILE *fp=fopen("wcc.txt", "w");
		for(long i = 0; i < n; i ++) fprintf(fp, "wcc[%ld]=%u\n", i, h_wcc[i]);
		fclose(fp);
	}
	update_pivot_color<<<nblocks, nthreads>>>(n, d_wcc, d_colors, d_status, d_has_pivot, d_scc_root, d_min_color);
	CudaTest("solving kernel update_pivot_color failed");
	update_colors<<<nblocks, nthreads>>>(n, d_wcc, d_colors, d_status);
	CudaTest("solving kernel update_colors failed");
	CUDA_SAFE_CALL(cudaMemcpy(&has_pivot, d_has_pivot, sizeof(bool), cudaMemcpyDeviceToHost));
	if(debug_wcc) {
		unsigned char *h_status = (unsigned char *)malloc(n*sizeof(unsigned char));
		unsigned *h_colors = (unsigned *)malloc(n*sizeof(unsigned));
		CUDA_SAFE_CALL(cudaMemcpy(h_status, d_status, n*sizeof(unsigned char), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(h_colors, d_colors, n*sizeof(unsigned), cudaMemcpyDeviceToHost));
		FILE *fp1=fopen("pivot.txt", "w");
		for(long i = 0; i < n; i ++)
			if(!is_removed(h_status[i]) && h_status[i]==19)
				fprintf(fp1, "%ld\n", i);
		fclose(fp1);
	}
	//printf("wcc_iteration=%ld\n", iter);
	CUDA_SAFE_CALL(cudaFree(d_changed));
	CUDA_SAFE_CALL(cudaFree(d_wcc));
	return has_pivot;
}

