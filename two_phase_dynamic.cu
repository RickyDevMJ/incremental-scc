#include "bitset.h"
#include "graph_io.h"
#include "timer.h"
#include "kernels.h"

#define GPU_ERR_CHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, long line, bool abort=true) {
   	if (code != cudaSuccess) 
   	{
	  	fprintf(stderr,"GPUassert: %s %s %ld\n", cudaGetErrorString(code), file, line);
	  	if (abort) exit(code);
   	}
}

struct meta {	
	std::unordered_map<long, long> scc_to_id, id_to_scc;

	long *in_scc_row_offsets, *out_scc_row_offsets;
	long *in_scc_column_indices, *out_scc_column_indices;
};

int main(int argc, char *argv[]) {
	
	bool is_directed = true;
	bool symmetrize = false;
	if (argc < 4) {
		printf("Usage: %s <base-graph> <is_directed(0/1)> <update-graph-1> ...\n", argv[0]);
		exit(1);
	} else {
		is_directed = atoi(argv[2]);
		if(is_directed) printf("This is a directed graph\n");
		else printf("This is an undirected graph\n");
	}
	if (!is_directed) symmetrize = true;

    long n, m;
	long *h_weight = NULL;
	long *in_row_offsets, *out_row_offsets, *in_column_indices, *out_column_indices, *in_degree, *out_degree;
	
	read_graph(argc, argv[1], n, n, m, out_row_offsets, out_column_indices, out_degree, h_weight, symmetrize, false, false);
	read_graph(argc, argv[1], n, n, m, in_row_offsets, in_column_indices, in_degree, h_weight, symmetrize, true, false);

	//long maxi = -1, mini = 2 * n;
	//for(long i = 0; i < n; i++) {
	//	maxi = std::max(maxi, out_degree[i] + in_degree[i]);
	//	mini = std::min(mini, out_degree[i] + in_degree[i]);
	//}
	//
	//printf("\n%ld %ld", maxi, mini);


	long *scc_root = (long *)malloc(n * sizeof(long));
	SCCSolver(n, m, in_row_offsets, in_column_indices, out_row_offsets, out_column_indices, scc_root);

	long n_scc, m_scc;
	struct meta meta_graph;

	initialise_meta_graph(n, n_scc, m_scc, meta_graph, out_row_offsets, out_column_indices, scc_root);
	long *meta_scc = (long *) malloc(n_scc * sizeof(long));
	printf("Number of vertices in meta graph: %ld\n", n_scc);

	long tot_scc_edges = m_scc;
	for (long i = 3; i < argc; i++) {
		// todo n_update != n
		std::vector<std::pair<long, long>> updates;
		Timer t_read;
		t_read.Start();
		read_updates(argv[i], updates);
		t_read.Stop();

		update_meta_graph(n_scc, meta_graph, scc_root, updates, symmetrize);
		SCCSolverMeta(n_scc, m_scc, meta_graph.in_scc_row_offsets, meta_graph.in_scc_column_indices, meta_graph.out_scc_row_offsets, meta_graph.out_scc_column_indices, meta_scc);
	
		tot_scc_edges += updates.size();
		printf("Number of edges in meta graph: %ld\n", tot_scc_edges);
		printf("Update read runtime = %f ms.\n", t_read.Millisecs());
	
		std::unordered_set<long> set;
		for (long j = 0; j < n_scc; j++) {
			set.insert(meta_scc[j]);
		}
		printf("Number of scc: %lu\n", set.size());
	}
}
