long rename_scc (long n, long *scc);
void read_updates(char *filename, std::vector<std::pair<long, long>> &updates);
//void construct_meta_graph(long n, long &n_scc, long &m_scc, long *scc_root, const std::vector<std::pair<long, long>> &updates,
//	struct meta &meta_graph, long *out_row_offsets, long *out_column_indices, bool symmetrize);
void update_meta_graph(long n_scc, struct meta &meta_graph, long *scc_root, const std::vector<std::pair<long, long>> &updates, bool symmetrize);
void initialise_meta_graph(long n, long &n_scc, long &m_scc, struct meta &meta_graph, long *out_row_offsets, long *out_column_indices, long *scc_root);
void SCCSolver(long n, long m, long *in_row_offsets, long *in_column_indices, long *out_row_offsets, long *out_column_indices, long *scc_root);
void SCCSolverMeta(long m, long nnz, long *d_in_row_offsets, long *d_in_column_indices, long *d_out_row_offsets, long *d_out_column_indices, long *h_scc_root);
void SCCSolverCpu(long n, long m, long *in_row_offsets, long *in_column_indices, long *out_row_offsets, long *out_column_indices, long *scc_root);
void SCCSolverTopo(long n, long m, long *in_row_offsets, long *in_column_indices, long *out_row_offsets, long *out_column_indices, long *h_scc_root);
