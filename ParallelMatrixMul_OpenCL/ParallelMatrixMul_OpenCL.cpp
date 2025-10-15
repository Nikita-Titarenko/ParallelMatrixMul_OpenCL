#include <CL/cl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>

#define N 512

void generateMatrix(std::vector<float>& mat) {
    for (auto& val : mat)
        val = static_cast<float>(rand() % 10);
}

void cpuMatMul(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            C[i * N + j] = 0;
            for (int k = 0; k < N; k++)
                C[i * N + j] += A[i * N + k] * B[k * N + j];
        }
}

std::string loadKernel(const std::string& filename) {
    std::ifstream file(filename);
    std::ostringstream oss;
    oss << file.rdbuf();
    return oss.str();
}

int main() {
    std::vector<float> A(N * N), B(N * N), C_cpu(N * N), C_gpu(N * N);

    generateMatrix(A);
    generateMatrix(B);

    auto start_cpu = std::chrono::high_resolution_clock::now();
    cpuMatMul(A, B, C_cpu);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();
    std::cout << "CPU time: " << cpu_time << " ms\n";

    try {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        cl::Platform platform = platforms[0];

        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        cl::Device device = devices[0];

        cl::Context context(device);
        cl::CommandQueue queue(context, device);

        cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * A.size(), A.data());
        cl::Buffer bufferB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * B.size(), B.data());
        cl::Buffer bufferC(context, CL_MEM_WRITE_ONLY, sizeof(float) * C_gpu.size());

        std::string kernel_code = R"CLC(
        __kernel void matMul(__global float* A, __global float* B, __global float* C, int N) {
            int row = get_global_id(0);
            int col = get_global_id(1);
            float sum = 0;
            for(int k = 0; k < N; k++)
                sum += A[row*N + k] * B[k*N + col];
            C[row*N + col] = sum;
        }
        )CLC";

        cl::Program program(context, kernel_code);
        program.build({ device });

        cl::Kernel kernel(program, "matMul");
        kernel.setArg(0, bufferA);
        kernel.setArg(1, bufferB);
        kernel.setArg(2, bufferC);
        kernel.setArg(3, N);

        auto start_gpu = std::chrono::high_resolution_clock::now();

        cl::NDRange global(N, N);
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, global);
        queue.finish();

        auto end_gpu = std::chrono::high_resolution_clock::now();
        double gpu_time = std::chrono::duration<double, std::milli>(end_gpu - start_gpu).count();
        std::cout << "GPU time: " << gpu_time << " ms\n";

        queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, sizeof(float) * C_gpu.size(), C_gpu.data());
    }
    catch (cl::Error& e) {
        std::cerr << "OpenCL Error: " << e.what() << " : " << e.err() << std::endl;
        return 1;
    }

    bool correct = true;
    for (int i = 0; i < N * N; i++) {
        if (abs(C_cpu[i] - C_gpu[i]) > 1e-5) {
            correct = false;
            break;
        }
    }
    std::cout << "Results are " << (correct ? "correct" : "incorrect") << std::endl;

    return 0;
}
