#include "bitset.h"
#define debug_wcc 0

static void wcc_min(long src, long n, long *row_offsets, long *column_indices, unsigned *colors, unsigned char *status, unsigned *wcc, bool &changed) {
	if (src < n && !is_removed(status[src])) {
		long row_begin = row_offsets[src];
		long row_end = row_offsets[src + 1] - 1;
		unsigned wcc_src = wcc[src];
		for(long offset = row_begin; offset < row_end; offset ++) {
			long dst = column_indices[offset];
			if(!is_removed(status[dst]) && colors[src] == colors[dst]) {
				if (wcc[dst] < wcc_src) {
					wcc_src = wcc[dst];
					changed = true;
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
						changed = true;
					}
				}
			}

			updates = (long *) updates[length + 1];
		}
		
		wcc[src] = wcc_src;
	}
}

static void wcc_update(long src, long n, unsigned char *status, unsigned *wcc, bool &changed) {
	if (src < n && !is_removed(status[src])) {
		unsigned wcc_src = wcc[src];
		unsigned wcc_k = wcc[wcc_src];
		if (wcc_src != src && wcc_src != wcc_k) {
			wcc[src] = wcc_k;
			changed = true;
		}
	}
}

static void update_pivot_color(long src, long n, unsigned *wcc, unsigned *colors, unsigned char *status, bool &has_pivot, long *scc_root, unsigned &min_color) {
	if (src < n && !is_removed(status[src])) {
		if (wcc[src] == src) {
			unsigned new_color;
			#pragma omp atomic capture
			new_color = min_color++; 
			//printf("wcc: select vertex %ld as pivot, old_color=%u, new_color=%u\n", src, colors[src], new_color);
			colors[src] = new_color;
			status[src] = 19; // set as a pivot
			scc_root[src] = src;
			has_pivot = true;
		}
	}
}

static void update_colors(long src, long n, unsigned *wcc, unsigned *colors, unsigned char *status) {
	if (src < n && !is_removed(status[src])) {
		unsigned wcc_src = wcc[src];
		if (wcc_src != src)
			colors[src] = colors[wcc_src];
	}
}

bool find_wcc(long n, long *row_offsets, long *column_indices, unsigned *colors, unsigned char *status, long *scc_root, unsigned min_color) {
	bool changed;
	long iter = 0;
	unsigned *wcc = (unsigned *) malloc (sizeof(unsigned) * n);
	bool has_pivot = false;
	#pragma omp parallel for
	for (long i = 0; i < n; i++) {
		wcc[i] = i;
	}
	do {
		++ iter;
		changed = false;
		#pragma omp parallel for
		for (long i = 0; i < n; i++) {
			wcc_min(i, n, row_offsets, column_indices, colors, status, wcc, changed);
		}
		#pragma omp parallel for
		for (long i = 0; i < n; i++) {
			wcc_update(i, n, status, wcc, changed);
		}
	} while (changed);

	#pragma omp parallel for
	for (long i = 0; i < n; i++) {
		update_pivot_color(i, n, wcc, colors, status, has_pivot, scc_root, min_color);
	}
	#pragma omp parallel for
	for (long i = 0; i < n; i++) {
		update_colors(i, n, wcc, colors, status);
	}

	//printf("wcc_iteration=%ld\n", iter);
	free(wcc);
	return has_pivot;
}

