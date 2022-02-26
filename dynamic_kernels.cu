#include "bitset.h"
#include "timer.h"
#include <numeric>

long rename_scc (long n, long *scc) {
	long *temp = (long *) malloc(n * sizeof(long));
	memset(temp, 0, n * sizeof(long));
	for (long i = 0; i < n; i++) {
		temp[scc[i]] = 1;
	}

	thrust::device_vector<long> d_temp(temp, temp + n);
	thrust::device_vector<long> d_scc(d_temp.size());
	thrust::inclusive_scan(d_temp.begin(), d_temp.end(), d_scc.begin());
	thrust::host_vector<long> h_scc(d_scc.begin(), d_scc.end());

	long num_scc = h_scc[n-1];

	for (long i = 0; i < n; i++) {
		scc[i] = h_scc[scc[i]] - 1;
	}

	free(temp);
	return num_scc;
}

void read_updates(char *filename, std::vector<std::pair<long, long>> &updates) {

	if (strstr(filename, ".mtx") == NULL) {
		printf("Update file not in mtx format!\n");
		exit(1);
	}

	std::ifstream input_file(filename);
	
	long n, m;
	input_file >> n >> n >> m;

	for (long i = 0; i < m; i++) {
		long u, v;

		input_file >> u >> v;
		u--; v--;

		updates.push_back(std::make_pair(u, v));
	}

	input_file.close();
}

struct meta {	
	std::unordered_map<long, long> scc_to_id, id_to_scc;

	long *in_scc_row_offsets, *out_scc_row_offsets;
	long *in_scc_column_indices, *out_scc_column_indices;
};

// TODO create a structure 'graph'
std::vector<std::unordered_set<long>> out_scc_adj;
std::vector<std::vector<long>> out_scc_adj_vector;
std::vector<std::unordered_set<long>> in_scc_adj;
std::vector<std::vector<long>> in_scc_adj_vector;

void initialise_meta_graph(long n, long &n_scc, long &m_scc, struct meta &meta_graph, long *out_row_offsets, long *out_column_indices, long *scc_root) {
	printf("Initialising meta graph... \n");

	long counter = 0;
	for (long i = 0; i < n; i++) {
		
		long u = scc_root[i];

		if (meta_graph.scc_to_id.find(u) == meta_graph.scc_to_id.end()) {
			meta_graph.scc_to_id[u] = counter;
			meta_graph.id_to_scc[counter] = u;
			counter++;

			std::unordered_set<long> set1, set2;
			std::vector<long> vector1, vector2;
			out_scc_adj.push_back(set1);
			out_scc_adj_vector.push_back(vector1);
			in_scc_adj.push_back(set2);
			in_scc_adj_vector.push_back(vector2);
		}
	}

	for (long i = 0; i < n; i++) {
		long row_start = out_row_offsets[i], row_end = out_row_offsets[i+1] - 1;
		long u = scc_root[i];

		u = meta_graph.scc_to_id[u];

		for (long j = row_start; j < row_end; j++) {
			long v = scc_root[out_column_indices[j]];

			v = meta_graph.scc_to_id[v];

			if (u == v)
				continue;
			
			if (out_scc_adj[u].find(v) == out_scc_adj[u].end()) {
				out_scc_adj[u].insert(v);
				out_scc_adj_vector[u].push_back(v);
			}
			if (in_scc_adj[v].find(u) == in_scc_adj[v].end()) {
				in_scc_adj[v].insert(u);
				in_scc_adj_vector[v].push_back(u);
			}
		}
	}
	
	n_scc = counter;

	m_scc = 0;
	for(long i = 0; i < n_scc; i++) {
		m_scc += out_scc_adj[i].size();
	}

	meta_graph.in_scc_row_offsets = (long *) malloc((n_scc + 1) * sizeof(long));
	meta_graph.out_scc_row_offsets = (long *) malloc((n_scc + 1) * sizeof(long));

	meta_graph.out_scc_column_indices = (long *) malloc((m_scc + n_scc) * sizeof(long));
	meta_graph.in_scc_column_indices = (long *) malloc((m_scc + n_scc) * sizeof(long));

	long column_index = 0;
	for (long i = 0; i < n_scc; i++) {
		long length = out_scc_adj_vector[i].size();
		meta_graph.out_scc_row_offsets[i] = column_index;
		std::copy(out_scc_adj_vector[i].begin(), out_scc_adj_vector[i].end(), &meta_graph.out_scc_column_indices[column_index]);
		column_index += length;
		meta_graph.out_scc_column_indices[column_index] = 0;
		column_index++;
	}
	meta_graph.out_scc_row_offsets[n_scc] = column_index;

	column_index = 0;
	for (long i = 0; i < n_scc; i++) {
		long length = in_scc_adj_vector[i].size();
		meta_graph.in_scc_row_offsets[i] = column_index;
		std::copy(in_scc_adj_vector[i].begin(), in_scc_adj_vector[i].end(), &meta_graph.in_scc_column_indices[column_index]);
		column_index += length;
		meta_graph.in_scc_column_indices[column_index] = 0;
		column_index++;
	}
	meta_graph.in_scc_row_offsets[n_scc] = column_index;

	printf("Done\n");
}

void update_meta_graph(long n_scc, struct meta &meta_graph, long *scc_root, const std::vector<std::pair<long, long>> &updates, bool symmetrize)
{
	std::vector<std::unordered_set<long>> out_scc_adj_update(n_scc);
	std::vector<std::vector<long>> out_scc_adj_vector_update(n_scc);
	std::vector<std::unordered_set<long>> in_scc_adj_update(n_scc);
	std::vector<std::vector<long>> in_scc_adj_vector_update(n_scc);

	Timer t_update;
	t_update.Start();
	
	for (long i = 0; i < updates.size(); i++) {
		long u = scc_root[updates[i].first], v = scc_root[updates[i].second];

		u = meta_graph.scc_to_id[u];
		v = meta_graph.scc_to_id[v];
		
		if (u == v)
			continue;
		
		if (out_scc_adj[u].find(v) == out_scc_adj[u].end() && out_scc_adj_update[u].find(v) == out_scc_adj_update[u].end()) {
			out_scc_adj_update[u].insert(v);
			out_scc_adj_vector_update[u].push_back(v);
			//printf ("%ld %ld\n", u, v);
		}
		if (in_scc_adj[v].find(u) == in_scc_adj[v].end() && in_scc_adj_update[v].find(u) == in_scc_adj_update[v].end()) {
			in_scc_adj_update[v].insert(u);
			in_scc_adj_vector_update[v].push_back(u);
		}

		if (symmetrize) {
			if (out_scc_adj[v].find(u) == out_scc_adj[v].end() && out_scc_adj_update[v].find(u) == out_scc_adj_update[v].end()) {
				out_scc_adj_update[v].insert(u);
				out_scc_adj_vector_update[v].push_back(u);
				//printf ("%ld %ld\n", v, u);
			}
			if (in_scc_adj[u].find(v) == in_scc_adj[u].end() && in_scc_adj_update[u].find(v) == in_scc_adj_update[u].end()) {
				in_scc_adj_update[u].insert(v);
				in_scc_adj_vector_update[u].push_back(v);
			}
		}
	}
	
	//long cou = 0, maxi = 0;
	for (long i = 0; i < n_scc; i++) {
		long length = in_scc_adj_vector_update[i].size();
		if (length > 0) {
			//cou++;
			//maxi = max(length, maxi);
			long *ptr = (long *) malloc((length + 2) * sizeof(long));
			long *dest = &meta_graph.in_scc_column_indices[meta_graph.in_scc_row_offsets[i + 1] - 1];
			while (*dest != 0) {
				dest = (long *) (*dest);
				dest = &dest[dest[0] + 1];
			}
			*dest = (long) ptr;
			ptr[0] = length;
			std::sort(in_scc_adj_vector_update[i].begin(), in_scc_adj_vector_update[i].end());
			std::copy(in_scc_adj_vector_update[i].begin(), in_scc_adj_vector_update[i].end(), &ptr[1]);
			ptr[length + 1] = 0;
		}
	}
	//printf("max updates %ld %ld\n", cou, maxi);

	for (long i = 0; i < n_scc; i++) {
		long length = out_scc_adj_vector_update[i].size();
		if (length > 0) {
			long *ptr = (long *) malloc((length + 2) * sizeof(long));
			long *dest = &meta_graph.out_scc_column_indices[meta_graph.out_scc_row_offsets[i + 1] - 1];
			while (*dest != 0) {
				dest = (long *) (*dest);
				dest = &dest[dest[0] + 1];
			}
			*dest = (long) ptr;
			ptr[0] = length;
			std::sort(out_scc_adj_vector_update[i].begin(), out_scc_adj_vector_update[i].end());
			std::copy(out_scc_adj_vector_update[i].begin(), out_scc_adj_vector_update[i].end(), &ptr[1]);
			ptr[length + 1] = 0;
		}
	}

	t_update.Stop();

	printf("Meta updation runtime = %f ms.\n", t_update.Millisecs());
}