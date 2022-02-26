#include "timer.h"
#include "bitset.h"
#include <thrust/reduce.h>

void SCCSolver(long n, long m, long *in_row_offsets, long *in_column_indices, long *out_row_offsets, long *out_column_indices, long *h_scc_root) {
	Timer t;
	long iter = 1;
	long *d_in_row_offsets, *d_in_column_indices, *d_out_row_offsets, *d_out_column_indices;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_in_row_offsets, (n + 1) * sizeof(long)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_in_column_indices, (m + n) * sizeof(long)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_out_row_offsets, (n + 1) * sizeof(long)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_out_column_indices, (m + n) * sizeof(long)));
	//CUDA_SAFE_CALL(cudaMalloc((void **)&d_in_degree, n * sizeof(long)));
	//CUDA_SAFE_CALL(cudaMalloc((void **)&d_out_degree, n * sizeof(long)));
	CUDA_SAFE_CALL(cudaMemcpy(d_in_row_offsets, in_row_offsets, (n + 1) * sizeof(long), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_in_column_indices, in_column_indices, (m + n) * sizeof(long), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_out_row_offsets, out_row_offsets, (n + 1) * sizeof(long), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_out_column_indices, out_column_indices, (m + n) * sizeof(long), cudaMemcpyHostToDevice));
	//CUDA_SAFE_CALL(cudaMemcpy(d_in_degree, in_degree, n * sizeof(long), cudaMemcpyHostToDevice));
	//CUDA_SAFE_CALL(cudaMemcpy(d_out_degree, out_degree, n * sizeof(long), cudaMemcpyHostToDevice));
	unsigned *d_colors, *d_locks;
	long *d_scc_root;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_colors, n * sizeof(unsigned)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_locks, (PIVOT_HASH_CONST+1) * sizeof(unsigned)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_scc_root, n * sizeof(long)));
	thrust::fill(thrust::device, d_colors, d_colors + n, INIT_COLOR);
	thrust::sequence(thrust::device, d_scc_root, d_scc_root + n);

	unsigned char *h_status = (unsigned char*)malloc(n * sizeof(unsigned char));
	unsigned char *d_status;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_status, n * sizeof(unsigned char)));
	CUDA_SAFE_CALL(cudaMemset(d_status, 0, n * sizeof(unsigned char)));
	bool has_pivot;
	long *d_mark;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_mark, n * sizeof(long)));
	CUDA_SAFE_CALL(cudaMemset(d_mark, 0, n * sizeof(long)));
	printf("Start solving SCC detection...\n");
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	Timer t1, t2, t3;
	t1.Start();
	t.Start();
	first_trim(n, d_in_row_offsets, d_in_column_indices, d_out_row_offsets, d_out_column_indices, d_status);
	CUDA_SAFE_CALL(cudaMemcpy(h_status, d_status, n * sizeof(unsigned char), cudaMemcpyDeviceToHost));
	/*
	if(debug) {
		long num_trimmed = 0;
		for (long i = 0; i < n; i ++) {
			if (is_trimmed(h_status[i]))
				num_trimmed ++;
		}
		printf("%ld vertices trimmed in the first trimming\n", num_trimmed);
	}
	//*/
	long source;
	for (long i = 0; i < n; i++) { 
		if(!is_removed(h_status[i])) {
			printf("Vertex %ld not eliminated, set as the first pivot\n", i);
			source = i;
			break;
		}
	}
	CUDA_SAFE_CALL(cudaMemset(&d_status[source], 19, 1));
	// phase-1
	printf("Start phase-1...\t");
	has_pivot = false;
	//fwd_reach(n, d_out_row_offsets, d_out_column_indices, d_colors, d_status, d_scc_root);
	//bwd_reach(n, d_in_row_offsets, d_in_column_indices, d_colors, d_status);
	t1.Stop();
	t2.Start();
	fwd_reach_lb(n, d_out_row_offsets, d_out_column_indices, d_status, d_scc_root);
	bwd_reach_lb(n, d_in_row_offsets, d_in_column_indices, d_status);
	t2.Stop();
	t3.Start();
	iterative_trim(n, d_in_row_offsets, d_in_column_indices, d_out_row_offsets, d_out_column_indices, d_colors, d_status, d_scc_root);
	update_colors(n, d_colors, d_status);
	printf("Done\n");
	CUDA_SAFE_CALL(cudaMemcpy(h_status, d_status, sizeof(unsigned char) * n, cudaMemcpyDeviceToHost));
	find_removed_vertices(n, d_status, d_mark);
	long num_removed = thrust::reduce(thrust::device, d_mark, d_mark + n, 0, thrust::plus<long>());;
	//for (long i = 0; i < n; i++) if(is_removed(h_status[i])) num_removed ++;
	//printf("%ld vertices removed in phase-1\n", num_removed);
	/*
	if(debug) {
		long first_scc_size = 0;
		long num_trimmed = 0;
		for (long i = 0; i < n; i++) { 
			if(is_trimmed(h_status[i])) num_trimmed ++;
			else if(is_removed(h_status[i])) first_scc_size ++;
		}
		printf("size of the first scc: %ld\n", first_scc_size);
		printf("number of trimmed vertices: %ld\n", num_trimmed);
	}
	//*/
	
	if(num_removed != n) {
		printf("Start Trim2...\t\t");
		trim2(n, d_in_row_offsets, d_in_column_indices, d_out_row_offsets, d_out_column_indices, d_colors, d_status, d_scc_root);
		printf("Done\n");
		unsigned min_color = 6;//thrust::reduce(thrust::device, d_colors, d_colors + n, 0, thrust::maximum<unsigned>());
		printf("Start finding WCC...\t");
		has_pivot = find_wcc(n, d_out_row_offsets, d_out_column_indices, d_colors, d_status, d_scc_root, min_color);
		printf("Done\n");
		//printf("min_color=%ld\n", min_color);

		printf("Start phase-2...\t");
		// phase-2
		while (has_pivot) {
			++ iter;
			has_pivot = false;
			//if(debug) printf("iteration=%ld\n", iter);
			fwd_reach(n, d_out_row_offsets, d_out_column_indices, d_colors, d_status, d_scc_root);
			bwd_reach(n, d_in_row_offsets, d_in_column_indices, d_colors, d_status);
			iterative_trim(n, d_in_row_offsets, d_in_column_indices, d_out_row_offsets, d_out_column_indices, d_colors, d_status, d_scc_root);
			CUDA_SAFE_CALL(cudaMemset(d_locks, 0, (PIVOT_HASH_CONST+1) * sizeof(unsigned)));
			has_pivot = update(n, d_colors, d_status, d_locks, d_scc_root);
		}
		printf("Done\n");
	}
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t3.Stop();
	t.Stop();
	printf("\truntime= %f ms.\n", t1.Millisecs());
	printf("\truntime= %f ms.\n", t2.Millisecs());
	printf("\truntime= %f ms.\n", t3.Millisecs());
	CUDA_SAFE_CALL(cudaMemcpy(h_scc_root, d_scc_root, sizeof(long) * n, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(h_status, d_status, sizeof(unsigned char) * n, cudaMemcpyDeviceToHost));
	print_statistics(n, h_scc_root, h_status);
	printf("\titerations = %ld.\n", iter);
	printf("\truntime= %f ms.\n", t.Millisecs());
	CUDA_SAFE_CALL(cudaFree(d_in_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_in_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_out_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_out_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_scc_root));
	CUDA_SAFE_CALL(cudaFree(d_status));
	CUDA_SAFE_CALL(cudaFree(d_locks));
	free(h_status);
}

void SCCSolverMeta(long n, long m, long *in_row_offsets, long *in_column_indices, long *out_row_offsets, long *out_column_indices, long *h_scc_root) {
	Timer t;
	long iter = 1;
	long *d_in_row_offsets, *d_in_column_indices, *d_out_row_offsets, *d_out_column_indices;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_in_row_offsets, (n + 1) * sizeof(long)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_in_column_indices, (m + n) * sizeof(long)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_out_row_offsets, (n + 1) * sizeof(long)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_out_column_indices, (m + n) * sizeof(long)));
	//CUDA_SAFE_CALL(cudaMalloc((void **)&d_in_degree, n * sizeof(long)));
	//CUDA_SAFE_CALL(cudaMalloc((void **)&d_out_degree, n * sizeof(long)));
	CUDA_SAFE_CALL(cudaMemcpy(d_in_row_offsets, in_row_offsets, (n + 1) * sizeof(long), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_out_row_offsets, out_row_offsets, (n + 1) * sizeof(long), cudaMemcpyHostToDevice));

	for (long i = 0; i < n; i ++) {
		long row_start = in_row_offsets[i], row_end = in_row_offsets[i+1] - 1;
		CUDA_SAFE_CALL(cudaMemcpy(&d_in_column_indices[row_start], &in_column_indices[row_start], (row_end - row_start) * sizeof(long), cudaMemcpyHostToDevice));
		
		long *ptr = (long *) in_column_indices[row_end];
		long *d_ptr = &d_in_column_indices[row_end];
		while (ptr != NULL) {
			long length = ptr[0];
			long *d_temp;
			CUDA_SAFE_CALL(cudaMalloc((long **)&d_temp, (length + 2) * sizeof(long)));
			CUDA_SAFE_CALL(cudaMemcpy(d_ptr, &d_temp, sizeof(long), cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL(cudaMemcpy(d_temp, ptr, (length + 2) * sizeof(long), cudaMemcpyHostToDevice));

			ptr = (long *) ptr[length + 1];
			d_ptr = &d_temp[length + 1];
		}

		CUDA_SAFE_CALL(cudaMemset(d_ptr, 0, sizeof(long)));
	}

	for (long i = 0; i < n; i ++) {
		long row_start = out_row_offsets[i], row_end = out_row_offsets[i+1] - 1;
		CUDA_SAFE_CALL(cudaMemcpy(&d_out_column_indices[row_start], &out_column_indices[row_start], (row_end - row_start) * sizeof(long), cudaMemcpyHostToDevice));
		
		long *ptr = (long *) out_column_indices[row_end];
		long *d_ptr = &d_out_column_indices[row_end];
		while (ptr != NULL) {
			long length = ptr[0];
			long *d_temp;
			CUDA_SAFE_CALL(cudaMalloc((long **)&d_temp, (length + 2) * sizeof(long)));
			CUDA_SAFE_CALL(cudaMemcpy(d_ptr, &d_temp, sizeof(long), cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL(cudaMemcpy(d_temp, ptr, (length + 2) * sizeof(long), cudaMemcpyHostToDevice));

			ptr = (long *) ptr[length + 1];
			d_ptr = &d_temp[length + 1];
		}

		CUDA_SAFE_CALL(cudaMemset(d_ptr, 0, sizeof(long)));
	}
	
	//CUDA_SAFE_CALL(cudaMemcpy(d_in_column_indices, in_column_indices, (m + n) * sizeof(long), cudaMemcpyHostToDevice));
	//CUDA_SAFE_CALL(cudaMemcpy(d_out_column_indices, out_column_indices, (m + n) * sizeof(long), cudaMemcpyHostToDevice));
	//CUDA_SAFE_CALL(cudaMemcpy(d_in_degree, in_degree, n * sizeof(long), cudaMemcpyHostToDevice));
	//CUDA_SAFE_CALL(cudaMemcpy(d_out_degree, out_degree, n * sizeof(long), cudaMemcpyHostToDevice));
	unsigned *d_colors, *d_locks;
	long *d_scc_root;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_colors, n * sizeof(unsigned)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_locks, (PIVOT_HASH_CONST+1) * sizeof(unsigned)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_scc_root, n * sizeof(long)));
	thrust::fill(thrust::device, d_colors, d_colors + n, INIT_COLOR);
	thrust::sequence(thrust::device, d_scc_root, d_scc_root + n);

	unsigned char *h_status = (unsigned char*)malloc(n * sizeof(unsigned char));
	unsigned char *d_status;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_status, n * sizeof(unsigned char)));
	CUDA_SAFE_CALL(cudaMemset(d_status, 0, n * sizeof(unsigned char)));
	bool has_pivot;
	long *d_mark;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_mark, n * sizeof(long)));
	CUDA_SAFE_CALL(cudaMemset(d_mark, 0, n * sizeof(long)));
	printf("Start solving SCC detection...\n");
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	Timer t1, t2, t3;
	t.Start();
	t1.Start();
	first_trim(n, d_in_row_offsets, d_in_column_indices, d_out_row_offsets, d_out_column_indices, d_status);
	CUDA_SAFE_CALL(cudaMemcpy(h_status, d_status, n * sizeof(unsigned char), cudaMemcpyDeviceToHost));
	t1.Stop();
	printf("\truntime= %f ms.\n", t1.Millisecs());
	
	//if(debug) {
	//	long num_trimmed = 0;
	//	for (long i = 0; i < n; i ++) {
	//		if (is_trimmed(h_status[i]))
	//			num_trimmed ++;
	//	}
	//	printf("%ld vertices trimmed in the first trimming\n", num_trimmed);
	//}
	
	long source;
	for (long i = 0; i < n; i++) { 
		if(!is_removed(h_status[i])) {
			printf("Vertex %ld not eliminated, set as the first pivot\n", i);
			source = i;
			break;
		}
	}
	CUDA_SAFE_CALL(cudaMemset(&d_status[source], 19, 1));
	// phase-1
	printf("Start phase-1...\t");
	has_pivot = false;
	t2.Start();
	//fwd_reach(n, d_out_row_offsets, d_out_column_indices, d_colors, d_status, d_scc_root);
	fwd_reach_lb(n, d_out_row_offsets, d_out_column_indices, d_status, d_scc_root);
	//bwd_reach(n, d_in_row_offsets, d_in_column_indices, d_colors, d_status);
	bwd_reach_lb(n, d_in_row_offsets, d_in_column_indices, d_status);
	t2.Stop();
	t3.Start();
	//iterative_trim(n, d_in_row_offsets, d_in_column_indices, d_out_row_offsets, d_out_column_indices, d_colors, d_status, d_scc_root);
	update_colors(n, d_colors, d_status);
	printf("Done\n");
	CUDA_SAFE_CALL(cudaMemcpy(h_status, d_status, sizeof(unsigned char) * n, cudaMemcpyDeviceToHost));
	find_removed_vertices(n, d_status, d_mark);
	long num_removed = thrust::reduce(thrust::device, d_mark, d_mark + n, 0, thrust::plus<long>());;
	//for (long i = 0; i < n; i++) if(is_removed(h_status[i])) num_removed ++;
	//printf("%ld vertices removed in phase-1\n", num_removed);
	
	//if(debug) {
	//	long first_scc_size = 0;
	//	long num_trimmed = 0;
	//	for (long i = 0; i < n; i++) { 
	//		if(is_trimmed(h_status[i])) num_trimmed ++;
	//		else if(is_removed(h_status[i])) first_scc_size ++;
	//	}
	//	printf("size of the first scc: %ld\n", first_scc_size);
	//	printf("number of trimmed vertices: %ld\n", num_trimmed);
	//}
	
	
	if(num_removed != n) {
		printf("Start Trim2...\t\t");
		trim2(n, d_in_row_offsets, d_in_column_indices, d_out_row_offsets, d_out_column_indices, d_colors, d_status, d_scc_root);
		printf("Done\n");
		unsigned min_color = 6;//thrust::reduce(thrust::device, d_colors, d_colors + n, 0, thrust::maximum<unsigned>());
		printf("Start finding WCC...\t");
		has_pivot = find_wcc(n, d_out_row_offsets, d_out_column_indices, d_colors, d_status, d_scc_root, min_color);
		printf("Done\n");
		//printf("min_color=%ld\n", min_color);

		printf("Start phase-2...\t");
		// phase-2
		while (has_pivot) {
			++ iter;
			has_pivot = false;
			//if(debug) printf("iteration=%ld\n", iter);
			fwd_reach(n, d_out_row_offsets, d_out_column_indices, d_colors, d_status, d_scc_root);
			bwd_reach(n, d_in_row_offsets, d_in_column_indices, d_colors, d_status);
			//iterative_trim(n, d_in_row_offsets, d_in_column_indices, d_out_row_offsets, d_out_column_indices, d_colors, d_status, d_scc_root);
			CUDA_SAFE_CALL(cudaMemset(d_locks, 0, (PIVOT_HASH_CONST+1) * sizeof(unsigned)));
			has_pivot = update(n, d_colors, d_status, d_locks, d_scc_root);
		}
		printf("Done\n");
	}
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t3.Stop();
	t.Stop();

	printf("\truntime= %f ms.\n", t2.Millisecs());
	printf("\truntime= %f ms.\n", t3.Millisecs());
	CUDA_SAFE_CALL(cudaMemcpy(h_scc_root, d_scc_root, sizeof(long) * n, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(h_status, d_status, sizeof(unsigned char) * n, cudaMemcpyDeviceToHost));
	print_statistics(n, h_scc_root, h_status);
	printf("\titerations = %ld.\n", iter);
	printf("\truntime= %f ms.\n", t.Millisecs());
	// TODO recursively free memory
	CUDA_SAFE_CALL(cudaFree(d_in_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_in_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_out_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_out_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_scc_root));
	CUDA_SAFE_CALL(cudaFree(d_status));
	CUDA_SAFE_CALL(cudaFree(d_locks));
	free(h_status);
}

void SCCSolverTopo(long n, long m, long *in_row_offsets, long *in_column_indices, long *out_row_offsets, long *out_column_indices, long *h_scc_root) {
	Timer t;
	long iter = 0;
	long *d_in_row_offsets, *d_in_column_indices, *d_out_row_offsets, *d_out_column_indices;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_in_row_offsets, (n + 1) * sizeof(long)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_in_column_indices, (m + n) * sizeof(long)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_out_row_offsets, (n + 1) * sizeof(long)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_out_column_indices, (m + n) * sizeof(long)));
	CUDA_SAFE_CALL(cudaMemcpy(d_in_row_offsets, in_row_offsets, (n + 1) * sizeof(long), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_in_column_indices, in_column_indices, (m + n) * sizeof(long), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_out_row_offsets, out_row_offsets, (n + 1) * sizeof(long), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_out_column_indices, out_column_indices, (m + n) * sizeof(long), cudaMemcpyHostToDevice));
	unsigned *d_colors, *d_locks;
	long *d_scc_root;
	unsigned *h_colors = (unsigned *)malloc(n * sizeof(unsigned));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_colors, n * sizeof(unsigned)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_locks, (PIVOT_HASH_CONST+1) * sizeof(unsigned)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_scc_root, n * sizeof(long)));
	thrust::fill(thrust::device, d_colors, d_colors + n, INIT_COLOR);
	thrust::sequence(thrust::device, d_scc_root, d_scc_root + n);

	unsigned char *h_status = (unsigned char*)malloc(n * sizeof(unsigned char));
	unsigned char *d_status;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_status, sizeof(unsigned char) * n));
	CUDA_SAFE_CALL(cudaMemset(d_status, 0, n * sizeof(unsigned char)));
	bool has_pivot;
	long source;
	printf("Start solving SCC detection...");
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	t.Start();
	first_trim(n, d_in_row_offsets, d_in_column_indices, d_out_row_offsets, d_out_column_indices, d_status);
	CUDA_SAFE_CALL(cudaMemcpy(h_status, d_status, n * sizeof(bool), cudaMemcpyDeviceToHost));
	for (long i = 0; i < n; i++) { 
		if(!is_removed(h_status[i])) {
			printf("vertex %ld not eliminated, set as the first pivot\n", i);
			source = i;
			break;
		}
	}
	CUDA_SAFE_CALL(cudaMemset(&d_status[source], 19, 1));
	do {
		++ iter;
		has_pivot = false;
		if(debug) printf("iteration=%ld\n", iter);
		fwd_reach(n, d_out_row_offsets, d_out_column_indices, d_colors, d_status, d_scc_root);
		bwd_reach(n, d_in_row_offsets, d_in_column_indices, d_colors, d_status);
		iterative_trim(n, d_in_row_offsets, d_in_column_indices, d_out_row_offsets, d_out_column_indices, d_colors, d_status, d_scc_root);
		CUDA_SAFE_CALL(cudaMemset(d_locks, 0, (PIVOT_HASH_CONST+1) * sizeof(unsigned)));
		has_pivot = update(n, d_colors, d_status, d_locks, d_scc_root);
	} while (has_pivot);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	printf("Done\n");
	CUDA_SAFE_CALL(cudaMemcpy(h_scc_root, d_scc_root, sizeof(long) * n, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(h_status, d_status, sizeof(unsigned char) * n, cudaMemcpyDeviceToHost));
	print_statistics(n, h_scc_root, h_status);
	printf("\titerations = %ld.\n", iter);
	printf("\truntime = %f ms.\n", t.Millisecs());
	CUDA_SAFE_CALL(cudaFree(d_in_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_in_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_out_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_out_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_colors));
	CUDA_SAFE_CALL(cudaFree(d_locks));
	CUDA_SAFE_CALL(cudaFree(d_status));
	free(h_status);
}


/*
void SCCSolverMeta(long n, long m, long *d_in_row_offsets, long *d_in_column_indices, long *d_out_row_offsets, long *d_out_column_indices, long *h_scc_root) {
	Timer t;
	long iter = 1;
	unsigned *d_colors, *d_locks;
	long *d_scc_root;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_colors, n * sizeof(unsigned)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_locks, (PIVOT_HASH_CONST+1) * sizeof(unsigned)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_scc_root, n * sizeof(long)));
	thrust::fill(thrust::device, d_colors, d_colors + n, INIT_COLOR);
	thrust::sequence(thrust::device, d_scc_root, d_scc_root + n);

	unsigned char *h_status = (unsigned char*)malloc(n * sizeof(unsigned char));
	unsigned char *d_status;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_status, n * sizeof(unsigned char)));
	CUDA_SAFE_CALL(cudaMemset(d_status, 0, n * sizeof(unsigned char)));
	bool has_pivot;
	long *d_mark;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_mark, n * sizeof(long)));
	CUDA_SAFE_CALL(cudaMemset(d_mark, 0, n * sizeof(long)));
	printf("Start solving SCC detection...\n");
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	t.Start();
	first_trim(n, d_in_row_offsets, d_in_column_indices, d_out_row_offsets, d_out_column_indices, d_status);
	CUDA_SAFE_CALL(cudaMemcpy(h_status, d_status, n * sizeof(unsigned char), cudaMemcpyDeviceToHost));

	long source;
	for (long i = 0; i < n; i++) { 
		if(!is_removed(h_status[i])) {
			printf("Vertex %ld not eliminated, set as the first pivot\n", i);
			source = i;
			break;
		}
	}
	CUDA_SAFE_CALL(cudaMemset(&d_status[source], 19, 1));
	// phase-1
	printf("Start phase-1...\t");
	has_pivot = false;
	//fwd_reach(n, d_out_row_offsets, d_out_column_indices, d_colors, d_status, d_scc_root);
	//bwd_reach(n, d_in_row_offsets, d_in_column_indices, d_colors, d_status);
	fwd_reach_lb(n, d_out_row_offsets, d_out_column_indices, d_status, d_scc_root);
	bwd_reach_lb(n, d_in_row_offsets, d_in_column_indices, d_status);
	iterative_trim(n, d_in_row_offsets, d_in_column_indices, d_out_row_offsets, d_out_column_indices, d_colors, d_status, d_scc_root);
	update_colors(n, d_colors, d_status);
	printf("Done\n");
	CUDA_SAFE_CALL(cudaMemcpy(h_status, d_status, sizeof(unsigned char) * n, cudaMemcpyDeviceToHost));
	find_removed_vertices(n, d_status, d_mark);
	long num_removed = thrust::reduce(thrust::device, d_mark, d_mark + n, 0, thrust::plus<long>());;
	
	if(num_removed != n) {
		printf("Start Trim2...\t\t");
		trim2(n, d_in_row_offsets, d_in_column_indices, d_out_row_offsets, d_out_column_indices, d_colors, d_status, d_scc_root);
		printf("Done\n");
		unsigned min_color = 6;//thrust::reduce(thrust::device, d_colors, d_colors + n, 0, thrust::maximum<unsigned>());
		printf("Start finding WCC...\t");
		has_pivot = find_wcc(n, d_out_row_offsets, d_out_column_indices, d_colors, d_status, d_scc_root, min_color);
		printf("Done\n");
		//printf("min_color=%ld\n", min_color);

		printf("Start phase-2...\t");
		// phase-2
		while (has_pivot) {
			++ iter;
			has_pivot = false;
			//if(debug) printf("iteration=%ld\n", iter);
			fwd_reach(n, d_out_row_offsets, d_out_column_indices, d_colors, d_status, d_scc_root);
			bwd_reach(n, d_in_row_offsets, d_in_column_indices, d_colors, d_status);
			iterative_trim(n, d_in_row_offsets, d_in_column_indices, d_out_row_offsets, d_out_column_indices, d_colors, d_status, d_scc_root);
			CUDA_SAFE_CALL(cudaMemset(d_locks, 0, (PIVOT_HASH_CONST+1) * sizeof(unsigned)));
			has_pivot = update(n, d_colors, d_status, d_locks, d_scc_root);
		}
		printf("Done\n");
	}
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	CUDA_SAFE_CALL(cudaMemcpy(h_scc_root, d_scc_root, sizeof(unsigned) * n, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(h_status, d_status, sizeof(unsigned char) * n, cudaMemcpyDeviceToHost));
	print_statistics(n, h_scc_root, h_status);
	printf("\titerations = %ld.\n", iter);
	printf("\truntime= %f ms.\n", t.Millisecs());
	CUDA_SAFE_CALL(cudaFree(d_in_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_in_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_out_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_out_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_scc_root));
	CUDA_SAFE_CALL(cudaFree(d_status));
	CUDA_SAFE_CALL(cudaFree(d_locks));
	free(h_status);
}
*/