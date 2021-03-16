// kernel.cu

__global__ void kernel()
{
    // some code here...
}

void kernel_function()
{

    dim3 threads(2, 1);
    dim3 blocks(1, 1);

    kernel<<<blocks, threads>>>();
}
