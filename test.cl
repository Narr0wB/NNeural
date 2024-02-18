__kernel void test_kernel(
    __global float* A,
    __global float* B,
    __global float* C,
    const int N
)
{
    int threads = get_global_size(0);
    int thread_id = get_global_id(0);

    for (int i = thread_id; i < N; i += threads) {
        C[i] = A[i] + B[i];
    }
}