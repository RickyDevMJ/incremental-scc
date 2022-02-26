#include "timer.h"
#include "bitset.h"
#include <thrust/reduce.h>

void SCCSolverCpu(long n, long m, long *in_row_offsets, long *in_column_indices, long *out_row_offsets, long *out_column_indices, long *scc_root) {
	Timer t;
	long iter = 1;

	unsigned *colors, *locks;
	colors = (unsigned *) malloc(n * sizeof(unsigned));
	locks = (unsigned *) malloc((PIVOT_HASH_CONST+1) * sizeof(unsigned));
	#pragma omp parallel for
	for (long i = 0; i < n; i++) {
		colors[i] = INIT_COLOR;
		scc_root[i] = i;
	}

	// status initialised to zero
	unsigned char *status = (unsigned char*) calloc(n, sizeof(unsigned char));
	long *mark = (long *) calloc(n, sizeof(long));
	bool has_pivot;
	printf("Start solving SCC detection...\n");

	Timer t1, t2, t3;
	t.Start();
	t1.Start();
	first_trim(n, in_row_offsets, in_column_indices, out_row_offsets, out_column_indices, status);
	t1.Stop();
	printf("\truntime= %f ms.\n", t1.Millisecs());
	
	//if(debug) {
	//	long num_trimmed = 0;
	//	for (long i = 0; i < n; i ++) {
	//		if (is_trimmed(status[i]))
	//			num_trimmed ++;
	//	}
	//	printf("%ld vertices trimmed in the first trimming\n", num_trimmed);
	//}
	
	long source;
	for (long i = 0; i < n; i++) { 
		if(!is_removed(status[i])) {
			printf("Vertex %ld not eliminated, set as the first pivot\n", i);
			source = i;
			break;
		}
	}
	status[source] = 19;
	// phase-1
	printf("Start phase-1...\t");
	has_pivot = false;
	t2.Start();
	fwd_reach(n, out_row_offsets, out_column_indices, colors, status, scc_root);
	//fwd_reach_lb(n, out_row_offsets, out_column_indices, status, scc_root);
	bwd_reach(n, in_row_offsets, in_column_indices, colors, status);
	//bwd_reach_lb(n, in_row_offsets, in_column_indices, status);
	t2.Stop();
	t3.Start();
	//iterative_trim(n, in_row_offsets, in_column_indices, out_row_offsets, out_column_indices, colors, status, scc_root);
	update_colors(n, colors, status);
	printf("Done\n");
	find_removed_vertices(n, status, mark);
	long num_removed = 0;
	#pragma omp parallel for reduction(+:num_removed)
	for (long i = 0; i < n; i++) {
		num_removed += mark[i];
	}

	//for (long i = 0; i < n; i++) if(is_removed(status[i])) num_removed ++;
	//printf("%ld vertices removed in phase-1\n", num_removed);
	
	//if(debug) {
	//	long first_scc_size = 0;
	//	long num_trimmed = 0;
	//	for (long i = 0; i < n; i++) { 
	//		if(is_trimmed(status[i])) num_trimmed ++;
	//		else if(is_removed(status[i])) first_scc_size ++;
	//	}
	//	printf("size of the first scc: %ld\n", first_scc_size);
	//	printf("number of trimmed vertices: %ld\n", num_trimmed);
	//}
	
	
	if(num_removed != n) {
		printf("Start Trim2...\t\t");
		trim2(n, in_row_offsets, in_column_indices, out_row_offsets, out_column_indices, colors, status, scc_root);
		printf("Done\n");
		unsigned min_color = 6;//thrust::reduce(thrust::host, colors, colors + n, 0, thrust::maximum<unsigned>());
		printf("Start finding WCC...\t");
		has_pivot = find_wcc(n, out_row_offsets, out_column_indices, colors, status, scc_root, min_color);
		printf("Done\n");
		//printf("min_color=%ld\n", min_color);

		printf("Start phase-2...\t");
		// phase-2
		while (has_pivot) {
			++ iter;
			has_pivot = false;
			//if(debug) printf("iteration=%ld\n", iter);
			fwd_reach(n, out_row_offsets, out_column_indices, colors, status, scc_root);
			bwd_reach(n, in_row_offsets, in_column_indices, colors, status);
			//iterative_trim(n, in_row_offsets, in_column_indices, out_row_offsets, out_column_indices, colors, status, scc_root);
			#pragma omp parallel for 
			for (long i = 0; i < PIVOT_HASH_CONST+1; i++) {
				locks[i] = 0;
			}
			has_pivot = update(n, colors, status, locks, scc_root);
		}
		printf("Done\n");
	}
	t3.Stop();
	t.Stop();

	printf("\truntime= %f ms.\n", t2.Millisecs());
	printf("\truntime= %f ms.\n", t3.Millisecs());
	print_statistics(n, scc_root, status);
	printf("\titerations = %ld.\n", iter);
	printf("\truntime= %f ms.\n", t.Millisecs());

	// TODO recursively free memory
	free(status);
	free(locks);
}