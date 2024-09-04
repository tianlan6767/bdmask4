// #include <cuda_fp16.h>
// #include <cuda_runtime.h>
// #include <device_launch_parameters.h>
// #include <cstdlib> // Include cstdlib for rand
// #include <ctime> // Include ctime for time(0)

// // Define a kernel function to calculate the minimum and maximum values of the array
// __global__ void findMinMaxKernel(const float* data, float* minMax, int size) {
//   // Get the thread ID
//   int idx = blockIdx.x * blockDim.x + threadIdx.x;

//   // Check if the thread ID is within the array bounds
//   if (idx < size) {
//     // Initialize the minimum and maximum values
//     float minVal = data[idx];
//     float maxVal = data[idx];

//     // Use shared memory for local reduction
//     __shared__ float sharedMin[256];
//     __shared__ float sharedMax[256];
//     sharedMin[threadIdx.x] = minVal;
//     sharedMax[threadIdx.x] = maxVal;
//     __syncthreads();

//     // Perform local reduction
//     for (int s = blockDim.x / 2; s > 0; s >>= 1) {
//       if (threadIdx.x < s) {
//         sharedMin[threadIdx.x] = min(sharedMin[threadIdx.x], sharedMin[threadIdx.x + s]);
//         sharedMax[threadIdx.x] = max(sharedMax[threadIdx.x], sharedMax[threadIdx.x + s]);
//       }
//       __syncthreads();
//     }

//     // If it is the first thread in the block, write the result to global memory
//     if (threadIdx.x == 0) {
//       minMax[0] = sharedMin[0]; // Store min in minMax[0]
//       minMax[1] = sharedMax[0]; // Store max in minMax[1]
//     }
//   }
// }

// // Define a kernel function to normalize the data
// __global__ void normalizeKernel(float* data, float* minMax, int size) {
//   // Get the thread ID
//   int idx = blockIdx.x * blockDim.x + threadIdx.x;

//   // Check if the thread ID is within the array bounds
//   if (idx < size) {
//     // Normalize the data using the min and max values
//     data[idx] = (data[idx] - minMax[0]) / (minMax[1] - minMax[0]);
//   }
// }

// int main() {
//   // Initialize the array size
//   int width = 512;
//   int height = 512;
//   int size = width * height;

//   // Allocate host memory
//   float* data = new float[size];

//   // Initialize the data (using rand)
//   srand(time(0)); // Seed the random number generator
//   for (int i = 0; i < size; i++) {
//     data[i] = static_cast<float>(rand()) / RAND_MAX;
//   }

//   // Allocate device memory
//   float* d_data;
//   cudaMalloc(&d_data, size * sizeof(float));

//   // Copy the data to device memory
//   cudaMemcpy(d_data, data, size * sizeof(float), cudaMemcpyHostToDevice);

//   // Allocate memory for the minMax array
//   float* minMax;
//   cudaMalloc(&minMax, 2 * sizeof(float)); // Allocate space for two floats

//   // Set the thread block and grid dimensions
//   int threadsPerBlock = 256;
//   int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

//   // Launch the kernel function to find min and max
//   findMinMaxKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, minMax, size);

//   // Launch the kernel function to normalize the data
//   normalizeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, minMax, size);

//   // Read the normalized data from device memory
//   cudaMemcpy(data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost);

//   // Print the normalized data (for verification)
//   for (int i = 0; i < size; i++) {
//     printf("data[%d]: %f\n", i, data[i]);
//   }

//   // Free device memory
//   cudaFree(d_data);
//   cudaFree(minMax);

//   // Free host memory
//   delete[] data;

//   return 0;
// }