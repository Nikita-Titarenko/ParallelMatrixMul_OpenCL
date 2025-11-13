#include <CL/cl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <omp.h>

#define N 1024
#define BLOCK_SIZE 16

using namespace std;

void generateMatrix(vector<float>& mat) {
    for (auto& val : mat)
        val = static_cast<float>(rand() % 10);
}

void printResult(const string& name, double time_ms, double base) {
    cout << name << ":\t" << time_ms << " ms";
    if (base > 0)
        cout << "\t(прискорення ×" << base / time_ms << ")";
    cout << endl;
}

void multiplyMatrices1(const vector<float>& A, const vector<float>& B, vector<float>& C) {
    fill(C.begin(), C.end(), 0.0f);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C[i * N + j] += A[i * N + k] * B[k * N + j];
}

void multiplyMatrices2(const vector<float>& A, const vector<float>& B, vector<float>& C) {
    fill(C.begin(), C.end(), 0.0f);
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++) {
            float a_ik = A[i * N + k];
            for (int j = 0; j < N; j++)
                C[i * N + j] += a_ik * B[k * N + j];
        }
}

void multiplyMatrices3_omp(const vector<float>& A, const vector<float>& B, vector<float>& C) {
    fill(C.begin(), C.end(), 0.0f);
#pragma omp parallel for
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++) {
            float a_ik = A[i * N + k];
            for (int j = 0; j < N; j++)
                C[i * N + j] += a_ik * B[k * N + j];
        }
}

void cpuMatMulTiled(const vector<float>& A, const vector<float>& B, vector<float>& C) {
    fill(C.begin(), C.end(), 0.0f);
    for (int ii = 0; ii < N; ii += BLOCK_SIZE)
        for (int jj = 0; jj < N; jj += BLOCK_SIZE)
            for (int kk = 0; kk < N; kk += BLOCK_SIZE)
                for (int i = ii; i < min(ii + BLOCK_SIZE, N); i++)
                    for (int k = kk; k < min(kk + BLOCK_SIZE, N); k++) {
                        float a_ik = A[i * N + k];
                        for (int j = jj; j < min(jj + BLOCK_SIZE, N); j++)
                            C[i * N + j] += a_ik * B[k * N + j];
                    }
}

double gpuMatMulOpenCL(const vector<float>& A, const vector<float>& B, vector<float>& C) {
    try {
        vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) throw runtime_error("OpenCL платформи не знайдено");

        cl::Platform platform = platforms[0];
        vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (devices.empty()) throw runtime_error("GPU пристрої не знайдено");

        cl::Device device = devices[0];
        cout << "OpenCL пристрій: " << device.getInfo<CL_DEVICE_NAME>() << endl;

        cl::Context context(device);
        cl::CommandQueue queue(context, device);

        cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * A.size(), (void*)A.data());
        cl::Buffer bufferB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * B.size(), (void*)B.data());
        cl::Buffer bufferC(context, CL_MEM_WRITE_ONLY, sizeof(float) * C.size());

        string kernel_code = R"CLC(
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

        string define_block = "#define BLOCK_SIZE " + to_string(BLOCK_SIZE) + "\n";
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

        auto start_gpu = chrono::high_resolution_clock::now();
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
        queue.finish();
        auto end_gpu = chrono::high_resolution_clock::now();

        queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, sizeof(float) * C.size(), C.data());

        double gpu_time = chrono::duration<double, milli>(end_gpu - start_gpu).count();
        return gpu_time;
    }
    catch (const cl::Error& e) {
        cerr << "OpenCL Error: " << e.what() << " (" << e.err() << ")" << endl;
        return -1;
    }
    catch (const exception& e) {
        cerr << "Runtime Error: " << e.what() << endl;
        return -1;
    }
}

int main() {
    srand(0);

    vector<float> A(N * N), B(N * N), C(N * N), tmp(N * N);
    generateMatrix(A);
    generateMatrix(B);

    cout << "Порівняння методів множення матриць (" << N << "x" << N << ")\n";
    cout << "------------------------------------------------------------\n";

    double base_time = 0;

    auto start = chrono::high_resolution_clock::now();
    multiplyMatrices1(A, B, C);
    auto end = chrono::high_resolution_clock::now();
    base_time = chrono::duration<double, milli>(end - start).count();
    printResult("1. Базовий (3 цикли)", base_time, 0);

    start = chrono::high_resolution_clock::now();
    multiplyMatrices2(A, B, tmp);
    end = chrono::high_resolution_clock::now();
    printResult("2. Оптимізований порядок", chrono::duration<double, milli>(end - start).count(), base_time);

    start = chrono::high_resolution_clock::now();
    multiplyMatrices3_omp(A, B, tmp);
    end = chrono::high_resolution_clock::now();
    printResult("3. OpenMP (CPU паралельно)", chrono::duration<double, milli>(end - start).count(), base_time);

    start = chrono::high_resolution_clock::now();
    cpuMatMulTiled(A, B, tmp);
    end = chrono::high_resolution_clock::now();
    printResult("4. Tiled CPU (блочний)", chrono::duration<double, milli>(end - start).count(), base_time);

    double gpu_time = gpuMatMulOpenCL(A, B, tmp);
    if (gpu_time > 0)
        printResult("5. OpenCL (GPU)", gpu_time, base_time);

    cout << "------------------------------------------------------------\n";
    cout << "Перевірка коректності (випадкові елементи): ";
    bool correct = true;
    for (int i = 0; i < 1000; i++) {
        int idx = rand() % (N * N);
        if (fabs(C[idx] - tmp[idx]) > 1e-2f) {
            correct = false;
            break;
        }
    }
    cout << (correct ? "результати збігаються\n" : "відмінності знайдено\n");

    return 0;
}
