#include <set>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string.h>
#include <algorithm>
#include "timer.h"

struct WeightedEdge {
	long src;
	long dst;
	long wt;
	long eid;
	//WeightedEdge() : src(0), dst(0), wt(0), eid(0) {}
	//std::string to_string() const;
};

bool compare_id(WeightedEdge a, WeightedEdge b) { return (a.dst < b.dst); }

void fill_data(long m, long &nnz, long *&row_offsets, long *&column_indices, long *&weight, std::vector<std::vector<WeightedEdge> > vertices, bool symmetrize, bool sorted, bool remove_selfloops, bool remove_redundents) {
	Timer t;
	t.Start();
	//sort the neighbor list
	if(sorted) {
		printf("Sorting the neighbor lists...");
		for(long i = 0; i < m; i++) {
			std::sort(vertices[i].begin(), vertices[i].end(), compare_id);
		}
		printf(" Done\n");
	}

	//remove self loops
	long num_selfloops = 0;
	if(remove_selfloops) {
		printf("Removing self loops...");
		for(long i = 0; i < m; i++) {
			for(unsigned j = 0; j < vertices[i].size(); j ++) {
				if(i == vertices[i][j].dst) {
					vertices[i].erase(vertices[i].begin()+j);
					num_selfloops ++;
					j --;
				}
			}
		}
		printf(" %ld selfloops are removed\n", num_selfloops);
	}

	// remove redundent
	long num_redundents = 0;
	if(remove_redundents) {
		printf("Removing redundent edges...");
		for (long i = 0; i < m; i++) {
			for (unsigned j = 1; j < vertices[i].size(); j ++) {
				if (vertices[i][j].dst == vertices[i][j-1].dst) {
					vertices[i].erase(vertices[i].begin()+j);
					num_redundents ++;
					j --;
				}
			}
		}
		printf(" %ld redundent edges are removed\n", num_redundents);
	}

/*
	// print some neighbor lists
	for (long i = 0; i < 3; i++) {
		cout << "src " << i << ": ";
		for (long j = 0; j < vertices[i].size(); j ++)
			cout << vertices[i][j].dst << "  ";
		cout << endl;
	}
*/
#ifdef SIM
	row_offsets = (long *)aligned_alloc(PAGE_SIZE, (m + 1) * sizeof(long));
#else
	row_offsets = (long *)malloc((m + 1) * sizeof(long));
#endif
	long count = 0;
	for (long i = 0; i < m; i++) {
		row_offsets[i] = count;
		count += vertices[i].size() + 1;
	}
	row_offsets[m] = count;
	count -= m;
	if (symmetrize) {
		if(count != nnz) {
			nnz = count;
		}
	} else {
		if (count + num_selfloops + num_redundents != nnz)
			printf("Error reading graph, number of edges in edge list %ld != %ld\n", count, nnz);
		nnz = count;
	}
	printf("num_vertices %ld num_edges %ld\n", m, nnz);
	count += m;
	/*
	double avgdeg;
	double variance = 0.0;
	long maxdeg = 0;
	long mindeg = m;
	avgdeg = (double)nnz / m;
	for (long i = 0; i < m; i++) {
		long deg_i = row_offsets[i + 1] - row_offsets[i];
		if (deg_i > maxdeg)
			maxdeg = deg_i;
		if (deg_i < mindeg)
			mindeg = deg_i;
		variance += (deg_i - avgdeg) * (deg_i - avgdeg) / m;
	}
	printf("min_degree %ld max_degree %ld avg_degree %.2f variance %.2f\n", mindeg, maxdeg, avgdeg, variance);
	*/
#ifdef SIM
	column_indices = (long *)aligned_alloc(PAGE_SIZE, count * sizeof(long));
	weight = (long *)aligned_alloc(PAGE_SIZE, count * sizeof(long));
#else
	column_indices = (long *)malloc(count * sizeof(long));
	weight = (long *)malloc(count * sizeof(long));
#endif
	std::vector<WeightedEdge>::iterator neighbor_list;
	for (long i = 0, index = 0; i < m; i++) {
		neighbor_list = vertices[i].begin();
		while (neighbor_list != vertices[i].end()) {
			column_indices[index] = (*neighbor_list).dst;
			weight[index] = (*neighbor_list).wt;
			index ++;
			neighbor_list ++;
		}
		column_indices[index] = 0;
		weight[index] = 0;
		index ++;
	}
	t.Stop();
	printf("\truntime [%s] = %f ms.\n", "read_graph", t.Millisecs());
/*	
	// print some neighbor lists
	for (long i = 0; i < 6; i++) {
		long row_begin = row_offsets[i];
		long row_end = row_offsets[i + 1];
		std::cout << "src " << i << ": ";
		for (long j = row_begin; j < row_end; j ++)
			std::cout << column_indices[j] << "  ";
		std::cout << std::endl;
	}
	//
	//for (long i = 0; i < 10; i++) cout << weight[i] << ", ";
	//cout << endl;*/
}

// transfer gr graph to CSR format
void gr2csr(char *gr, long &m, long &nnz, long *&row_offsets, long *&column_indices, long *&weight, bool symmetrize, bool transpose, bool sorted, bool remove_selfloops, bool remove_redundents) {
	printf("Reading 9th DIMACS (.gr) input file %s\n", gr);
	std::ifstream cfile;
	cfile.open(gr);
	std::string str;
	getline(cfile, str);
	char c;
	sscanf(str.c_str(), "%c", &c);
	while (c == 'c') {
		getline(cfile, str);
		sscanf(str.c_str(), "%c", &c);
	}
	char sp[3];
	sscanf(str.c_str(), "%c %s %ld %ld", &c, sp, &m, &nnz);
	printf("Before cleaning, the original num_vertices %ld num_edges %ld\n", m, nnz);

	getline(cfile, str);
	sscanf(str.c_str(), "%c", &c);
	while (c == 'c') {
		getline(cfile, str);
		sscanf(str.c_str(), "%c", &c);
	}
	std::vector<std::vector<WeightedEdge> > vertices;
	std::vector<WeightedEdge> neighbors;
	for (long i = 0; i < m; i++)
		vertices.push_back(neighbors);
	long src, dst;
	for (long i = 0; i < nnz; i++) {
#ifdef LONG_TYPES
		sscanf(str.c_str(), "%c %ld %ld", &c, &src, &dst);
#else
		sscanf(str.c_str(), "%c %ld %ld", &c, &src, &dst);
#endif
		if (c != 'a')
			printf("line %d\n", __LINE__);
		src--;
		dst--;
		WeightedEdge e1, e2;
		if(symmetrize) {
			e2.dst = src; e2.wt = 1;
			vertices[dst].push_back(e2);
			transpose = false;
		}
		if(!transpose) {
			e1.dst = dst; e1.wt = 1;
			vertices[src].push_back(e1);
		} else {
			e1.dst = src; e1.wt = 1;
			vertices[dst].push_back(e1);
		}
		if(i != nnz-1) getline(cfile, str);
	}
	fill_data(m, nnz, row_offsets, column_indices, weight, vertices, symmetrize, sorted, remove_selfloops, remove_redundents);
}

// transfer edgelist graph to CSR format
void el2csr(char *el, long &m, long &nnz, long *&row_offsets, long *&column_indices, long *&weight, bool symmetrize, bool transpose, bool sorted, bool remove_selfloops, bool remove_redundents) {
	printf("Reading edgelist (.el) input file %s\n", el);
	std::ifstream cfile;
	cfile.open(el);
	std::string str;
	getline(cfile, str);
	sscanf(str.c_str(), "%ld %ld", &m, &nnz);
	printf("Before cleaning, the original num_vertices %ld num_edges %ld\n", m, nnz);
	std::vector<std::vector<WeightedEdge> > vertices;
	std::vector<WeightedEdge> neighbors;
	for (long i = 0; i < m; i++)
		vertices.push_back(neighbors);
	long dst, src;
	long wt = 1;
	for (long i = 0; i < nnz; i ++) {
	//while (!cfile.eof()) {
		getline(cfile, str);
#ifdef LONG_TYPES
		long num = sscanf(str.c_str(), "%ld %ld %ld", &src, &dst, &wt);
#else
		long num = sscanf(str.c_str(), "%ld %ld %ld", &src, &dst, &wt);
#endif
		if (num == 2) wt = 1;
		if (wt < 0) wt = -wt; // non-negtive weight
		src--;
		dst--;
		WeightedEdge e1, e2;
		if(symmetrize && src != dst) {
			e2.dst = src; e2.wt = wt;
			vertices[dst].push_back(e2);
			transpose = false;
		}
		if(!transpose) {
			e1.dst = dst; e1.wt = wt;
			vertices[src].push_back(e1);
		} else {
			e1.dst = src; e1.wt = wt;
			vertices[dst].push_back(e1);
		}
	}
	cfile.close();
	fill_data(m, nnz, row_offsets, column_indices, weight, vertices, symmetrize, sorted, remove_selfloops, remove_redundents);
}

// transfer *.graph file to CSR format
void graph2csr(char *graph, long &m, long &nnz, long *&row_offsets, long *&column_indices, long *&weight, bool symmetrize, bool transpose, bool sorted, bool remove_selfloops, bool remove_redundents) {
	printf("Reading .graph input file %s\n", graph);
	std::ifstream cfile;
	cfile.open(graph);
	std::string str;
	getline(cfile, str);
	sscanf(str.c_str(), "%ld %ld", &m, &nnz);
	printf("Before cleaning, the original num_vertices %ld num_edges %ld\n", m, nnz);
	std::vector<std::vector<WeightedEdge> > vertices;
	std::vector<WeightedEdge> neighbors;
	for (long i = 0; i < m; i++)
		vertices.push_back(neighbors);
	long dst;
	for (long src = 0; src < m; src ++) {
		getline(cfile, str);
		std::istringstream istr;
		istr.str(str);
		while(istr>>dst) {
			dst --;
			WeightedEdge e1;//, e2;
			if(symmetrize && src != dst) {
				// for .graph format, the input file already contains edges in both directions
				//e2.dst = src; e2.wt = 1;
				//vertices[dst].push_back(e2);
				transpose = false;
			}
			if(!transpose) {
				e1.dst = dst; e1.wt = 1;
				vertices[src].push_back(e1);
			} else {
				e1.dst = src; e1.wt = 1;
				vertices[dst].push_back(e1);
			}
		}
		istr.clear();
	}
    cfile.close();
	fill_data(m, nnz, row_offsets, column_indices, weight, vertices, symmetrize, sorted, remove_selfloops, remove_redundents);
}

// transfer mtx graph to CSR format
void mtx2csr(char *mtx, long &m, long &n, long &nnz, long *&row_offsets, long *&column_indices, long *&weight, bool symmetrize, bool transpose, bool sorted, bool remove_selfloops, bool remove_redundents) {
	printf("Reading (.mtx) input file %s\n", mtx);
	std::ifstream cfile;
	cfile.open(mtx);
	std::string str;
	getline(cfile, str);
	char c;
	sscanf(str.c_str(), "%c", &c);
	while (c == '%') {
		getline(cfile, str);
		sscanf(str.c_str(), "%c", &c);
	}
	sscanf(str.c_str(), "%ld %ld %ld", &m, &n, &nnz);
	if (m != n) {
		printf("Warning, m(%ld) != n(%ld)\n", m, n);
	}
	printf("Before cleaning, the original num_vertices %ld num_edges %ld\n", m, nnz);
	std::vector<std::vector<WeightedEdge> > vertices;
	std::vector<WeightedEdge> neighbors;
	for (long i = 0; i < m; i ++)
		vertices.push_back(neighbors);
	long dst, src;
	long wt = 1;
	for (long i = 0; i < nnz; i ++) {
		getline(cfile, str);
#ifdef LONG_TYPES
		long num = sscanf(str.c_str(), "%ld %ld %ld", &src, &dst, &wt);
#else
		long num = sscanf(str.c_str(), "%ld %ld %ld", &src, &dst, &wt);
#endif
		if (num == 2) wt = 1;
		if (wt < 0) wt = -wt; // non-negtive weight
		src--;
		dst--;
		WeightedEdge e1, e2;
		if(symmetrize && src != dst) {
			e2.dst = src; e2.wt = wt;
			vertices[dst].push_back(e2);
			transpose = false;
		}
		if(!transpose) {
			e1.dst = dst; e1.wt = wt;
			vertices[src].push_back(e1);
		} else {
			e1.dst = src; e1.wt = wt;
			vertices[dst].push_back(e1);
		}
	}
	cfile.close();
	fill_data(m, nnz, row_offsets, column_indices, weight, vertices, symmetrize, sorted, remove_selfloops, remove_redundents);
}
/*
void sort_neighbors(long m, long *row_offsets, long *&column_indices) {
	std::vector<long> neighbors;
	#pragma omp parallel for
	for(long i = 0; i < m; i++) {
		long row_begin = row_offsets[i];
		long row_end = row_offsets[i + 1];
		for (long offset = row_begin; offset < row_end; ++ offset) {
			neighbors.push_back(column_indices[offset]);
		}
		std::sort(neighbors.begin(), neighbors.end());
		long k = 0;
		for (long offset = row_begin; offset < row_end; ++ offset) {
			column_indices[offset] = neighbors[k++];
		}
	}	
}
*/
void read_graph(long argc, char *filename, long &m, long &n, long &nnz, long *&row_offsets, long *&column_indices, long *&degree, long *&weight, bool is_symmetrize=false, bool is_transpose=false, bool sorted=true, bool remove_selfloops=true, bool remove_redundents=true) {
	//if(is_symmetrize) printf("Requiring symmetric graphs for this algorithm\n");
	if (strstr(filename, ".mtx"))
		mtx2csr(filename, m, n, nnz, row_offsets, column_indices, weight, is_symmetrize, is_transpose, sorted, remove_selfloops, remove_redundents);
	else if (strstr(filename, ".graph"))
		graph2csr(filename, m, nnz, row_offsets, column_indices, weight, is_symmetrize, is_transpose, sorted, remove_selfloops, remove_redundents);
	else if (strstr(filename, ".gr"))
		gr2csr(filename, m, nnz, row_offsets, column_indices, weight, is_symmetrize, is_transpose, sorted, remove_selfloops, remove_redundents);
	else { printf("Unrecognizable input file format\n"); exit(0); }

	printf("Calculating degree...");
	degree = (long *)malloc(m * sizeof(long));
	for (long i = 0; i < m; i++) {
		degree[i] = row_offsets[i + 1] - row_offsets[i] - 1;
	}
	printf(" Done\n");
}

void print_degree(long m, long *in_degree, long *out_degree) {
	if(in_degree != NULL) {
		FILE *fp = fopen("in_degree.txt", "w");
		fprintf(fp,"%ld\n", m);
		for(long i = 0; i < m; i ++)
			fprintf(fp,"%ld ", in_degree[i]);
		fclose(fp);
	}
	if(out_degree != NULL) {
		FILE *fp = fopen("out_degree.txt", "w");
		fprintf(fp,"%ld\n", m);
		for(long i = 0; i < m; i ++)
			fprintf(fp,"%ld ", out_degree[i]);
		fclose(fp);
	}
}

