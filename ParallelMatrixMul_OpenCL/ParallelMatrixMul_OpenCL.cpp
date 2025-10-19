#include <CL/cl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdlib>

#define N 2048
#define BLOCK_SIZE 16

void generateMatrix(std::vector<float>& mat) {
    for (auto& val : mat)
        val = static_cast<float>(rand() % 10);
}

void cpuMatMulTiled(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C) {
    std::fill(C.begin(), C.end(), 0.0f);
    for (int ii = 0; ii < N; ii += BLOCK_SIZE)
        for (int jj = 0; jj < N; jj += BLOCK_SIZE)
            for (int kk = 0; kk < N; kk += BLOCK_SIZE)
                for (int i = ii; i < std::min(ii + BLOCK_SIZE, N); i++)
                    for (int k = kk; k < std::min(kk + BLOCK_SIZE, N); k++) {
                        float a_ik = A[i * N + k];
                        for (int j = jj; j < std::min(jj + BLOCK_SIZE, N); j++)
                            C[i * N + j] += a_ik * B[k * N + j];
                    }
}

int main() {
    std::vector<float> A(N * N), B(N * N), C_cpu(N * N), C_gpu(N * N);
    generateMatrix(A);
    generateMatrix(B);

    auto start_cpu = std::chrono::high_resolution_clock::now();
    cpuMatMulTiled(A, B, C_cpu);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();
    std::cout << "CPU time (tiled): " << cpu_time << " ms\n";

    try {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) throw std::runtime_error("No OpenCL platforms found.");

        cl::Platform platform = platforms[0];
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (devices.empty()) throw std::runtime_error("No GPU devices found.");

        cl::Device device = devices[0];
        std::cout << "Using GPU: " << device.getInfo<CL_DEVICE_NAME>() << "\n";

        cl::Context context(device);
        cl::CommandQueue queue(context, device);

        cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * A.size(), A.data());
        cl::Buffer bufferB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * B.size(), B.data());
        cl::Buffer bufferC(context, CL_MEM_WRITE_ONLY, sizeof(float) * C_gpu.size());

        std::string kernel_code = R"CLC(
        __kernel void matMulTiled(__global float* A,
                                  __global float* B,
                                  __global float* C,
                                  int N) {
            __local float Asub[BLOCK_SIZE][BLOCK_SIZE];
            __local float Bsub[BLOCK_SIZE][BLOCK_SIZE];

            int row = get_global_id(0);
            int col = get_global_id(1);
            int localRow = get_local_id(0);
            int localCol = get_local_id(1);

            float sum = 0.0f;

            for (int t = 0; t < N / BLOCK_SIZE; t++) {
                Asub[localRow][localCol] = A[row * N + t * BLOCK_SIZE + localCol];
                Bsub[localRow][localCol] = B[(t * BLOCK_SIZE + localRow) * N + col];
                barrier(CLK_LOCAL_MEM_FENCE);

                for (int k = 0; k < BLOCK_SIZE; k++)
                    sum += Asub[localRow][k] * Bsub[k][localCol];

                barrier(CLK_LOCAL_MEM_FENCE);
            }

            C[row * N + col] = sum;
        }
        )CLC";

        std::string define_block = "#define BLOCK_SIZE " + std::to_string(BLOCK_SIZE) + "\n";
        kernel_code = define_block + kernel_code;

        cl::Program program(context, kernel_code);
        program.build({ device });

        cl::Kernel kernel(program, "matMulTiled");
        kernel.setArg(0, bufferA);
        kernel.setArg(1, bufferB);
        kernel.setArg(2, bufferC);
        kernel.setArg(3, N);

        cl::NDRange global(N, N);
        cl::NDRange local(BLOCK_SIZE, BLOCK_SIZE);

        auto start_gpu = std::chrono::high_resolution_clock::now();
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
        queue.finish();
        auto end_gpu = std::chrono::high_resolution_clock::now();

        double gpu_time = std::chrono::duration<double, std::milli>(end_gpu - start_gpu).count();
        std::cout << "GPU time (tiled): " << gpu_time << " ms\n";

        queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, sizeof(float) * C_gpu.size(), C_gpu.data());
    }
    catch (const cl::Error& e) {
        std::cerr << "OpenCL Error: " << e.what() << " (" << e.err() << ")\n";
        return 1;
    }
    catch (const std::exception& e) {
        std::cerr << "Runtime Error: " << e.what() << "\n";
        return 1;
    }

    bool correct = true;
    for (int i = 0; i < 1000; i++) {
        int idx = rand() % (N * N);
        if (fabs(C_cpu[idx] - C_gpu[idx]) > 1e-2f) {
            correct = false;
            break;
        }
    }
    std::cout << "Results are " << (correct ? "correct" : "incorrect") << std::endl;

    return 0;
}