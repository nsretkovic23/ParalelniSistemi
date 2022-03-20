
// ReSharper disable All
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#define N 10
#define SCALAR 2


//  Izracunati izraz A - B*x, A i B su vektori, x skalar, napisati i kod za testiranje rezultata

__host__ void printVector(int* vector, const char* vecName)
{
	using std::cout;

	cout << "Vector " << vecName << ": {";
	for(int i = 0; i < N; ++i)
	{
		cout << vector[i];
		if (i != N - 1)
			cout << ", ";
	}
	cout << "}\n";
}

__global__ void calculateOnDevice(int* result, int* a, int* b, int scalar)
{
	int id = threadIdx.x;

	if(id < N)
		result[id] = a[id] - b[id] * scalar;
}

__host__ void calculateOnHost(int* result, int* a, int* b, int scalar)
{
	using std::cout;

	for(int i = 0; i < N; ++i)
	{
		result[i] = a[i] - b[i] * scalar;
	}
}

__host__ int compareCalculationsOnHost(int* host_result, int* device_result)
{
	using std::cout;

	for(int i = 0; i < N; ++i)
	{
		if (host_result[i] != device_result[i])
			return 0;
	}

	return 1;
}


int main()
{
	using  std::cout;

	int* host_a, *host_b, *host_res;
	int* dev_a, *dev_b, *dev_res;

	int* host_calculated_res;

	// HACK: Hardcoded scalar
	int scalar = 5;

	host_a = (int*)malloc(N * sizeof(int));
	host_b = (int*)malloc(N * sizeof(int));
	host_res = (int*)malloc(N * sizeof(int));
	host_calculated_res = (int*)malloc(N * sizeof(int));

	for(int i = 0; i < N; ++i)
	{
		host_a[i] = (i + 1) * 2;
		host_b[i] = (i + 1) * 3;
	}

	printVector(host_a, "A");
	printVector(host_b, "B");

	cudaMalloc((void**)&dev_a, N * sizeof(int));
	cudaMalloc((void**)&dev_b, N * sizeof(int));
	cudaMalloc((void**)&dev_res, N * sizeof(int));

	cudaMemcpy(dev_a, host_a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, host_b, N * sizeof(int), cudaMemcpyHostToDevice);

	calculateOnDevice <<<1,N>>> (dev_res, dev_a, dev_b, SCALAR);

	cudaMemcpy(host_res, dev_res, N * sizeof(int), cudaMemcpyDeviceToHost);

	calculateOnHost(host_calculated_res, host_a, host_b, SCALAR);

	printVector(host_res, "RESULT");
	printVector(host_calculated_res, "RES_SEQUENTIALLY");

	if (compareCalculationsOnHost(host_calculated_res, host_res) > 0)
		cout << "\nCalculations are correct!!!";
	else
		cout << "\nIncorrect calculations";

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_res);
	free(host_a);
	free(host_b);
	free(host_res);

	return 0;
}

