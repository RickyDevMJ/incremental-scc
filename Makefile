two_phase_dynamic.out: two_phase_dynamic.o dynamic_kernels.o bitset.o wcc.o two_phase.o
	nvcc -gencode arch=compute_75,code=sm_75 dynamic_kernels.o bitset.o wcc.o two_phase.o two_phase_dynamic.o -o two_phase_dynamic.out -O3 -Xptxas -O3

two_phase_dynamic_cpu.out: two_phase_dynamic.o dynamic_kernels.o bitset_cpu.o wcc_cpu.o two_phase_cpu.o
	nvcc -gencode arch=compute_75,code=sm_75 dynamic_kernels.o bitset_cpu.o wcc_cpu.o two_phase_cpu.o two_phase_dynamic.o -o two_phase_dynamic_cpu.out -O3 -Xptxas -O3 -Xcompiler -fopenmp -lgomp

two_phase_dynamic.o: two_phase_dynamic.cu
	nvcc -gencode arch=compute_75,code=sm_75 -c two_phase_dynamic.cu -o two_phase_dynamic.o -O3 -Xptxas -O3

dynamic_kernels.o: dynamic_kernels.cu
	nvcc -gencode arch=compute_75,code=sm_75 -c dynamic_kernels.cu -o dynamic_kernels.o -O3 -Xptxas -O3

two_phase.o: two_phase.cu
	nvcc -gencode arch=compute_75,code=sm_75 -c two_phase.cu -o two_phase.o -O3 -Xptxas -O3

two_phase_cpu.o: two_phase_cpu.cu
	nvcc -gencode arch=compute_75,code=sm_75 -c two_phase_cpu.cu -o two_phase_cpu.o -O3 -Xptxas -O3

bitset.o: bitset.cu
	nvcc -gencode arch=compute_75,code=sm_75 -c bitset.cu -o bitset.o -O3 -Xptxas -O3

bitset_cpu.o: bitset_cpu.cu
	nvcc -gencode arch=compute_75,code=sm_75 -c bitset_cpu.cu -o bitset_cpu.o -O3 -Xptxas -O3 -Xcompiler -fopenmp -lgomp

wcc_cpu.o: wcc_cpu.cu
	nvcc -gencode arch=compute_75,code=sm_75 -c wcc_cpu.cu -o wcc_cpu.o -O3 -Xptxas -O3 -Xcompiler -fopenmp -lgomp

wcc.o: wcc.cu
	nvcc -gencode arch=compute_75,code=sm_75 -c wcc.cu -o wcc.o -O3 -Xptxas -O3

clean:
	rm *.o
	rm *.out
	rm *.txt
