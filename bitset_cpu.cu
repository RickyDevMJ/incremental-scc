#include "bitset.h"
#include "timer.h"
#include <omp.h>
//#include "cub/cub.cuh"

// find forward reachable vertices
void fwd_step(long src, long n, long *row_offsets, long *column_indices, unsigned *colors, unsigned char *status, long *scc_root, bool &changed) {
	// if src is in the forward frontier (valid, visited but not expanded)
	if(src < n && is_fwd_front(status[src])) {
		set_fwd_expanded(&status[src]);
		long row_begin = row_offsets[src];
		long row_end = row_offsets[src + 1] - 1; 
		for (long offset = row_begin; offset < row_end; ++ offset) {
			long dst = column_indices[offset];
			// if dst is valid, not visited and has the same color as src
			if (!is_removed(status[dst]) && !is_fwd_visited(status[dst]) && (colors[dst] == colors[src])) {
				changed = true;
				set_fwd_visited(&status[dst]);
				scc_root[dst] = scc_root[src];
			}
		}

		long *updates = (long *) column_indices[row_end];
		while (updates != NULL) {
			long length = updates[0];
			for (long i = 1; i <= length; i++) {
				long dst = updates[i];
				// if dst is valid, not visited and has the same color as src
				if (!is_removed(status[dst]) && !is_fwd_visited(status[dst]) && (colors[dst] == colors[src])) {
					changed = true;
					set_fwd_visited(&status[dst]);
					scc_root[dst] = scc_root[src];
				}
			}

			updates = (long *) updates[length + 1];
		}
	}
}
/*
__global__ void fwd_step_lb(long n, long *row_offsets, long *column_indices, unsigned char *status, long *scc_root, bool *changed) {
	long tid = blockIdx.x * blockDim.x + threadIdx.x;
	typedef cub::BlockScan<long, BLOCK_SIZE> BlockScan;
	const long SCRATCHSIZE = BLOCK_SIZE;
	__shared__ BlockScan::TempStorage temp_storage;
	__shared__ long gather_offsets[SCRATCHSIZE];
	__shared__ unsigned srcsrc[BLOCK_SIZE];
	gather_offsets[threadIdx.x] = 0;
	long neighbor_size = 0;
	long neighbor_offset = 0;
	long scratch_offset = 0;
	long total_edges = 0;
	bool fwd_front = false;
	if(tid < n && is_fwd_front(status[tid])) {
		fwd_front = true;
		set_fwd_expanded(&status[tid]);
		neighbor_offset = row_offsets[tid];
		neighbor_size = row_offsets[tid+1] - neighbor_offset;
	}
	BlockScan(temp_storage).ExclusiveSum(neighbor_size, scratch_offset, total_edges);
	long done = 0;
	long neighbors_done = 0;
	while(total_edges > 0) {
		__syncthreads();
		long i, index;
		for(i = 0; neighbors_done + i < neighbor_size && (index = scratch_offset + i - done) < SCRATCHSIZE; i++) {
			gather_offsets[index] = neighbor_offset + neighbors_done + i;
			srcsrc[index] = tid;
		}
		neighbors_done += i;
		scratch_offset += i;
		__syncthreads();
		long src, dst = 0;
		long edge = gather_offsets[threadIdx.x];
		if(threadIdx.x < total_edges) {
			dst = column_indices[edge];
			src = srcsrc[threadIdx.x];

			if (row_offsets[src + 1] - 1 != edge) {
				// if dst is valid, not visited
				if (!is_removed(status[dst]) && !is_fwd_visited(status[dst])) {
					*changed = true;
					set_fwd_visited(&status[dst]);
					scc_root[dst] = scc_root[src];
				}
			}
		}
		total_edges -= BLOCK_SIZE;
		done += BLOCK_SIZE;
	}
	if (tid < n && fwd_front) {

		long *updates = (long *) column_indices[row_offsets[tid + 1] - 1];
		while (updates != NULL) {
			long length = updates[0];
			for (long i = 1; i <= length; i++) {
				long src = tid;
				long dst = updates[i];
				// if dst is valid, not visited
				if (!is_removed(status[dst]) && !is_fwd_visited(status[dst])) {
					*changed = true;
					set_fwd_visited(&status[dst]);
					scc_root[dst] = scc_root[src];
				}
			}

			updates = (long *) updates[length + 1];
		}
	}
}
*/
// find backward reachable vertices
void bwd_step(long src, long n, long *row_offsets, long *column_indices, unsigned *colors, unsigned char *status, bool &changed) {
	// if src is in the forward frontier (valid, visited but not expanded)
	if(src < n && is_bwd_front(status[src])) {
		set_bwd_expanded(&status[src]);
		long row_begin = row_offsets[src];
		long row_end = row_offsets[src + 1] - 1; 
		for (long offset = row_begin; offset < row_end; ++ offset) {
			long dst = column_indices[offset];
			// if dst is valid, not visited and has the same color as src
			if (!is_removed(status[dst]) && !is_bwd_visited(status[dst]) && (colors[dst] == colors[src])) {
				changed = true;
				set_bwd_visited(&status[dst]);
			}
		}

		long *updates = (long *) column_indices[row_end];
		while (updates != NULL) {
			long length = updates[0];
			for (long i = 1; i <= length; i++) {
				long dst = updates[i];
				// if dst is valid, not visited and has the same color as src
				if (!is_removed(status[dst]) && !is_bwd_visited(status[dst]) && (colors[dst] == colors[src])) {
					changed = true;
					set_bwd_visited(&status[dst]);
				}
			}

			updates = (long *) updates[length + 1];
		}
	}
}
/*
__global__ void bwd_step_lb(long n, long *row_offsets, long *column_indices, unsigned char *status, bool *changed) {
	long tid = blockIdx.x * blockDim.x + threadIdx.x;
	typedef cub::BlockScan<long, BLOCK_SIZE> BlockScan;
	const long SCRATCHSIZE = BLOCK_SIZE;
	__shared__ BlockScan::TempStorage temp_storage;
	__shared__ long gather_offsets[SCRATCHSIZE];
	__shared__ unsigned srcsrc[BLOCK_SIZE];
	gather_offsets[threadIdx.x] = 0;
	long neighbor_size = 0;
	long neighbor_offset = 0;
	long scratch_offset = 0;
	long total_edges = 0;
	bool bwd_front = false;
	if(tid < n && is_bwd_front(status[tid])) {
		bwd_front = true;
		set_bwd_expanded(&status[tid]);
		neighbor_offset = row_offsets[tid];
		neighbor_size = row_offsets[tid+1] - neighbor_offset;
	}
	BlockScan(temp_storage).ExclusiveSum(neighbor_size, scratch_offset, total_edges);
	long done = 0;
	long neighbors_done = 0;
	while(total_edges > 0) {
		__syncthreads();
		long i, index;
		for(i = 0; neighbors_done + i < neighbor_size && (index = scratch_offset + i - done) < SCRATCHSIZE; i++) {
			gather_offsets[index] = neighbor_offset + neighbors_done + i;
			srcsrc[index] = tid;
		}
		neighbors_done += i;
		scratch_offset += i;
		__syncthreads();
		long src = 0, dst = 0;
		long edge = gather_offsets[threadIdx.x];
		if(threadIdx.x < total_edges) {
			src = srcsrc[threadIdx.x];
			dst = column_indices[edge];

			if (row_offsets[src + 1] - 1 != edge) {
				// if dst is valid, not visited
				if (!is_removed(status[dst]) && !is_bwd_visited(status[dst])) {
					*changed = true;
					set_bwd_visited(&status[dst]);
				}
			}
			//else {
			//	if (dst != NULL) {
			//		long *updates = (long *)dst;
			//		long length = updates[0];
			//		for (long i = 1; i <= length; i++) {
			//			dst = updates[i];
			//			// if dst is valid, not visited
			//			if (!is_removed(status[dst]) && !is_bwd_visited(status[dst])) {
			//				*changed = true;
			//				set_bwd_visited(&status[dst]);
			//			}
			//		}
			//	}
			//}
		}
		total_edges -= BLOCK_SIZE;
		done += BLOCK_SIZE;
	}

	if (tid < n && bwd_front) {

		long *updates = (long *) column_indices[row_offsets[tid + 1] - 1];
		while (updates != NULL) {
			long length = updates[0];
			for (long i = 1; i <= length; i++) {
				long dst = updates[i];
				// if dst is valid, not visited
				if (!is_removed(status[dst]) && !is_bwd_visited(status[dst])) {
					*changed = true;
					set_bwd_visited(&status[dst]);
				}
			}

			updates = (long *) updates[length + 1];
		}
	}
}
*/
// trimming trivial SCCs
// Making sure self loops are removed before calling this routine
void trim_func(long src, long n, long *in_row_offsets, long *in_column_indices, long *out_row_offsets, long *out_column_indices, unsigned *colors, unsigned char *status, long *scc_root, bool &changed) {
	if(src < n && !is_removed(status[src])) {
		long in_degree = 0;
		long out_degree = 0;
		// calculate the number of incoming neighbors
		long row_begin = in_row_offsets[src];
		long row_end = in_row_offsets[src + 1] - 1;
		for (long offset = row_begin; offset < row_end; ++ offset) {
			long dst = in_column_indices[offset];
			if(!is_removed(status[dst]) && colors[dst] == colors[src]) { in_degree ++; break; }
		}
		long *updates = (long *) in_column_indices[row_end];
		while (in_degree == 0 && updates != NULL) {
			long length = updates[0];
			for (long i = 1; i <= length; i++) {
				long dst = updates[i];
				if(!is_removed(status[dst]) && colors[dst] == colors[src]) { in_degree ++; break; }
			}

			updates = (long *) updates[length + 1];
		}

		if (in_degree != 0) {
			// calculate the number of outgoing neighbors
			row_begin = out_row_offsets[src];
			row_end = out_row_offsets[src + 1] - 1; 
			for (long offset = row_begin; offset < row_end; ++ offset) {
				long dst = out_column_indices[offset];
				if(!is_removed(status[dst]) && colors[dst] == colors[src]) { out_degree ++; break; }
			}
			long *updates = (long *) out_column_indices[row_end];
			while (out_degree == 0 && updates != NULL) {
				long length = updates[0];
				for (long i = 1; i <= length; i++) {
					long dst = updates[i];
					if(!is_removed(status[dst]) && colors[dst] == colors[src]) { out_degree ++; break; }
				}
	
				updates = (long *) updates[length + 1];
			}
		}

		// remove (disable) the trival SCC
		if (in_degree == 0 || out_degree == 0) {
			set_removed(&status[src]);
			set_trimmed(&status[src]);
			scc_root[src] = src;
			if(debug) printf("found vertex %ld trimmed\n", src);
			changed = true;
		}
	}
}

void trim2_func(long src, long n, long *in_row_offsets, long *in_column_indices, long *out_row_offsets, long *out_column_indices, unsigned *colors, unsigned char *status, long *scc_root) {
	if (src < n && !is_removed(status[src])) {
		unsigned nbr, num_neighbors = 0;
		bool isActive = false;
		// outgoing edges
		long row_begin = out_row_offsets[src];
		long row_end = out_row_offsets[src + 1] - 1;
		unsigned c = colors[src];
		for (long offset = row_begin; offset < row_end; offset ++) {
			long dst = out_column_indices[offset];
			if (src != dst && c == colors[dst] && !is_removed(status[dst])) {
				num_neighbors++;
				if (num_neighbors > 1) break;
				nbr = dst;
			}
		}
		long *updates = (long *) out_column_indices[row_end];
		while (num_neighbors <= 1 && updates != NULL) {
			long length = updates[0];
			for (long i = 1; i <= length; i++) {
				long dst = updates[i];	
				if (src != dst && c == colors[dst] && !is_removed(status[dst])) {
					num_neighbors++;
					if (num_neighbors > 1) break;
					nbr = dst;
				}
			}

			updates = (long *) updates[length + 1];
		}

		if (num_neighbors == 1) {
			num_neighbors = 0;
			row_begin = out_row_offsets[nbr];
			row_end = out_row_offsets[nbr + 1] - 1;
			for (long offset = row_begin; offset < row_end; offset ++) {
				long dst = out_column_indices[offset];
				if (nbr != dst && c == colors[dst] && !is_removed(status[dst])) {
					if (dst != src) {
						isActive = true;
						break;
					}
					num_neighbors++;
				}
			}
			long *updates = (long *) out_column_indices[row_end];
			while (!isActive && updates != NULL) {
				long length = updates[0];
				for (long i = 1; i <= length; i++) {
					long dst = updates[i];	
					if (nbr != dst && c == colors[dst] && !is_removed(status[dst])) {
						if (dst != src) {
							isActive = true;
							break;
						}
						num_neighbors++;
					}
				}

				updates = (long *) updates[length + 1];
			}

			if (!isActive && num_neighbors == 1) {
				if (src < nbr) {
					status[src] = 20;
					status[nbr] = 4;
					scc_root[src] = src;
					scc_root[nbr] = src;
				} else {
					status[src] = 4;
					status[nbr] = 20;
					scc_root[src] = nbr;
					scc_root[nbr] = nbr;
				}
				return;
			}	
		}
		num_neighbors = 0;
		isActive = false;
		// incoming edges
		row_begin = in_row_offsets[src];
		row_end = in_row_offsets[src + 1] - 1;
		for (long offset = row_begin; offset < row_end; offset ++) {
			long dst = in_column_indices[offset];
			if (src != dst && c == colors[dst] && !is_removed(status[dst])) {
				num_neighbors++;
				if (num_neighbors > 1) break;
				nbr = dst;
			}
		}
		updates = (long *) in_column_indices[row_end];
		while (num_neighbors <= 1 && updates != NULL) {
			long length = updates[0];
			for (long i = 1; i <= length; i++) {
				long dst = updates[i];	
				if (src != dst && c == colors[dst] && !is_removed(status[dst])) {
					num_neighbors++;
					if (num_neighbors > 1) break;
					nbr = dst;
				}
			}

			updates = (long *) updates[length + 1];
		}

		if (num_neighbors == 1) {
			num_neighbors = 0;
			row_begin = in_row_offsets[nbr];
			row_end = in_row_offsets[nbr + 1] - 1;
			for (long offset = row_begin; offset < row_end; offset ++) {
				long dst = in_column_indices[offset];
				if (nbr != dst && c == colors[dst] && !is_removed(status[dst])) {
					if (dst != src) {
						isActive = true;
						break;
					}
					num_neighbors++;
				}
			}
			long *updates = (long *) in_column_indices[row_end];
			while (!isActive && updates != NULL) {
				long length = updates[0];
				for (long i = 1; i <= length; i++) {
					long dst = updates[i];	
					if (nbr != dst && c == colors[dst] && !is_removed(status[dst])) {
						if (dst != src) {
							isActive = true;
							break;
						}
						num_neighbors++;
					}
				}

				updates = (long *) updates[length + 1];
			}
			
			if (!isActive && num_neighbors == 1) {
				if (src < nbr) {
					status[src] = 20;
					status[nbr] = 4;
					scc_root[src] = src;
					scc_root[nbr] = src;
				} else {
					status[src] = 4;
					status[nbr] = 20;
					scc_root[src] = nbr;
					scc_root[nbr] = nbr;
				}
				return;
			}
		}
	}
}

void first_trim_func(long src, long n, long *in_row_offsets, long *in_column_indices, long *out_row_offsets, long *out_column_indices, unsigned char *status, bool &changed) {
	if(src < n && !is_removed(status[src])) {
		long in_degree = 0;
		long out_degree = 0;
		// calculate the number of incoming neighbors
		long row_begin = in_row_offsets[src];
		long row_end = in_row_offsets[src + 1] - 1; 
		for (long offset = row_begin; offset < row_end; ++ offset) {
			long dst = in_column_indices[offset];
			if(!is_removed(status[dst])) { in_degree ++; break; }
		}
		long *updates = (long *) in_column_indices[row_end];
		while (in_degree == 0 && updates != NULL) {
			long length = updates[0];
			for (long i = 1; i <= length; i++) {
				long dst = updates[i];
				if(!is_removed(status[dst])) { in_degree ++; break; }
			}

			updates = (long *) updates[length + 1];
		}

		if (in_degree != 0) {
			// calculate the number of outgoing neighbors
			row_begin = out_row_offsets[src];
			row_end = out_row_offsets[src + 1] - 1; 
			for (long offset = row_begin; offset < row_end; ++ offset) {
				long dst = out_column_indices[offset];
				if(!is_removed(status[dst])) { out_degree ++; break; }
			}
			
			long *updates = (long *) out_column_indices[row_end];
			while (out_degree == 0 && updates != NULL) {
				long length = updates[0];
				for (long i = 1; i <= length; i++) {
					long dst = updates[i];
					if(!is_removed(status[dst])) { out_degree ++; break; }
				}

				updates = (long *) updates[length + 1];
			}
		}

		// remove (disable) the trival SCC
		if (in_degree == 0 || out_degree == 0) {
			set_removed(&status[src]);
			set_trimmed(&status[src]);
			if(debug) printf("found vertex %ld trimmed\n", src);
			changed = true;
		}
	}
}

void update_func(long src, long n, unsigned *colors, unsigned char *status, unsigned *locks, long *scc_root, bool &has_pivot) {
	if (src < n && !is_removed(status[src])) {
		unsigned new_subgraph = (is_fwd_visited(status[src])?1:0) + (is_bwd_visited(status[src])?2:0); // F intersec B == 3, F/B == 1 B/F == 2 (V/F)/B == 0
		if (new_subgraph == 3) {
			set_removed(&status[src]);
			//if(debug) printf("\tfind %ld (color %ld) in the SCC\n", src, colors[src]);
			return;
		}
		unsigned par_subgraph = colors[src];
		unsigned new_color = 3 * par_subgraph + new_subgraph;
		colors[src] = new_color;
		status[src] = 0;
		//pivots generation
		if (locks[new_color & PIVOT_HASH_CONST] == 0) {
			unsigned temp;
			#pragma omp atomic capture
			temp = locks[new_color & PIVOT_HASH_CONST]++; 
			if (temp == 0) {
				has_pivot = true;
				status[src] = 19; // set fwd_visited bwd_visited & is_pivot
				scc_root[src] = src;
				//if(debug) printf("\tselect %ld (color %ld) as a pivot\n", src, colors[src]);
			}
		}
	}
}

void update_colors_func(long src, long n, unsigned *colors, unsigned char *status) {
	if (src < n && !is_removed(status[src])) {   
		unsigned new_subgraph = (is_fwd_visited(status[src])?1:0) + (is_bwd_visited(status[src])?2:0); // F intersec B == 3, F/B == 1 B/F == 2 (V/F)/B == 0
		if (new_subgraph == 3) {
			set_removed(&status[src]);
			return;
		}
		unsigned par_subgraph = colors[src];
		unsigned new_color = 3 * par_subgraph + new_subgraph;
		colors[src] = new_color;
		status[src] = 0;
	}
}	

void find_removed_vertices_func(long src, long n, unsigned char *status, long *mark) {
	if (src < n && is_removed(status[src]))
		mark[src] = 1;
}

// find forward reachable set
void fwd_reach(long n, long *out_row_offsets, long *out_column_indices, unsigned *colors, unsigned char *status, long *scc_root) {
	bool changed;
	do {
		changed = false;
		#pragma omp parallel for
		for (long i = 0; i < n; i++) {
			fwd_step(i, n, out_row_offsets, out_column_indices, colors, status, scc_root, changed);
		}
	} while (changed);
}
/*
void fwd_reach_lb(long n, long *out_row_offsets, long *out_column_indices, unsigned char *status, long *scc_root) {
	bool changed;
	do {
		changed = false;
		fwd_step_lb<<<nblocks, nthreads>>>(n, out_row_offsets, out_column_indices, status, scc_root, d_changed);
	} while (changed);
}
*/
// find backward reachable set
void bwd_reach(long n, long *in_row_offsets, long *in_column_indices, unsigned *colors, unsigned char *status) {
	bool changed;
	do {
		changed = false;
		#pragma omp parallel for
		for (long i = 0; i < n; i++) {
			bwd_step(i, n, in_row_offsets, in_column_indices, colors, status, changed);
		}
	} while (changed);
}
/*
void bwd_reach_lb(long n, long *in_row_offsets, long *in_column_indices, unsigned char *status) {
	long nthreads = BLOCK_SIZE;
	long nblocks = (n - 1) / nthreads + 1;
	bool h_changed, *d_changed;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_changed, sizeof(bool)));
	do {
		h_changed = false;
		CUDA_SAFE_CALL(cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice));
		bwd_step_lb<<<nblocks, nthreads>>>(n, in_row_offsets, in_column_indices, status, d_changed);
		CudaTest("solving kernel fw_step failed");
		CUDA_SAFE_CALL(cudaMemcpy(&h_changed, d_changed, sizeof(h_changed), cudaMemcpyDeviceToHost));
	} while (h_changed);
	CUDA_SAFE_CALL(cudaFree(d_changed));
}
*/
void iterative_trim(long n, long *in_row_offsets, long *in_column_indices, long *out_row_offsets, long *out_column_indices, unsigned *colors, unsigned char *status, long *scc_root) {
	bool changed;
	long iter = 0;
	do {
		iter ++;
		changed = false;
		#pragma omp parallel for
		for (long i = 0; i < n; i++) {
			trim_func(i, n, in_row_offsets, in_column_indices, out_row_offsets, out_column_indices, colors, status, scc_root, changed);
		}
	} while (changed);
}

void first_trim(long n, long *in_row_offsets, long *in_column_indices, long *out_row_offsets, long *out_column_indices, unsigned char *status) {
	bool changed;
	//Timer t;
	long iter = 0;
	do {
		iter ++;
		changed = false;
		//t.Start();
		#pragma omp parallel for
		for (long i = 0; i < n; i++) {
			first_trim_func(i, n, in_row_offsets, in_column_indices, out_row_offsets, out_column_indices, status, changed);
		}
		//t.Stop();
		//printf("func time: %f ms \n", t.Millisecs());
	} while (changed && iter < 3);
	printf ("iterations: %ld\n", iter);
}

bool update(long n, unsigned *colors, unsigned char *status, unsigned *locks, long *scc_root) {
	bool has_pivot = false;
	#pragma omp parallel for
	for (long i = 0; i < n; i++) {
		update_func(i, n, colors, status, locks, scc_root, has_pivot);
	}
	return has_pivot;
}

void update_colors(long n, unsigned *colors, unsigned char *status) {
	#pragma omp parallel for
	for (long i = 0; i < n; i++) {
		update_colors_func(i, n, colors, status);
	}
}

void trim2(long n, long *in_row_offsets, long *in_column_indices, long *out_row_offsets, long *out_column_indices, unsigned *colors, unsigned char *status, long *scc_root) {
	#pragma omp parallel for
	for (long i = 0; i < n; i++) {
		trim2_func(i, n, in_row_offsets, in_column_indices, out_row_offsets, out_column_indices, colors, status, scc_root);
	}
}

void find_removed_vertices(long n, unsigned char *status, long *mark) {
	#pragma omp parallel for
	for (long i = 0; i < n; i++) {
		find_removed_vertices_func(i, n, status, mark);
	}
}

void print_statistics(long n, long *scc_root, unsigned char *status) {
	long total_num_trimmed = 0;
	long total_num_pivots = 0;
	long num_trivial_scc = 0;
	long num_nontrivial_scc = 0;
	long total_num_scc = 0;
	long biggest_scc_size = 0;
	for (long i = 0; i < n; i ++) {
		if (is_trimmed(status[i])) {
			total_num_trimmed ++;
		}
		else if (is_pivot(status[i])) total_num_pivots ++;
	}
	std::vector<std::set<long> > scc_sets(n);
	for (long i = 0; i < n; i ++) {
		scc_sets[scc_root[i]].insert(i);
		if(scc_root[i] == i) total_num_scc ++;
	}
	for (long i = 0; i < n; i ++) {
		if (scc_sets[i].size() == 1) num_trivial_scc ++;
		else if (scc_sets[i].size() > 1) num_nontrivial_scc ++;
		if (scc_sets[i].size() > biggest_scc_size) biggest_scc_size = scc_sets[i].size();
	}
	printf("\tnum_trimmed=%ld, num_pivots=%ld, total_num_scc=%ld\n", total_num_trimmed, total_num_pivots, total_num_trimmed+total_num_pivots);
	printf("\tnum_trivial=%ld, num_nontrivial=%ld, total_num_scc=%ld, biggest_scc_size=%ld\n", num_trivial_scc, num_nontrivial_scc, total_num_scc, biggest_scc_size);

}

