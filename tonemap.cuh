#ifndef tonemap_h__
#define tonemap_h__

/*
* Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

/*
Parallel reduction kernels
*/

__device__ __forceinline__
float sum_op(const float& a, const float& b)
{
    return a + b;
}

__device__ __forceinline__
float sum_zero() {
    return 0.0f;
}

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
struct SharedMemory
{
    __device__ inline operator float *()
    {
        extern __shared__ int __smem[];
        return (float *)__smem;
    }

    __device__ inline operator const float *() const
    {
        extern __shared__ int __smem[];
        return (float *)__smem;
    }
};

/*
    Parallel sum reduction using shared memory
    - takes log(n) steps for n input elements
    - uses n threads
    - only works for power-of-2 arrays

    This version adds multiple elements per thread sequentially.  This reduces the overall
    cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
    (Brent's Theorem optimization)

    Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
    In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
    If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
template <unsigned int blockSize, bool nIsPow2>
__global__ void
max_kernel(float *g_idata, float *g_odata, unsigned int n)
{
    float *sdata = SharedMemory();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize * 2 + threadIdx.x;
    unsigned int gridSize = blockSize * 2 * gridDim.x;

    float myResult = sum_zero();

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        myResult = sum_op(myResult, g_idata[i]);

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n)
            myResult = sum_op(myResult, g_idata[i + blockSize]);

        i += gridSize;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = myResult;
    __syncthreads();


    // do reduction in shared mem
    if ((blockSize >= 512) && (tid < 256))
    {
        sdata[tid] = myResult = sum_op(myResult, sdata[tid + 256]);
    }

    __syncthreads();

    if ((blockSize >= 256) && (tid < 128))
    {
        sdata[tid] = myResult = sum_op(myResult, sdata[tid + 128]);
    }

    __syncthreads();

    if ((blockSize >= 128) && (tid <  64))
    {
        sdata[tid] = myResult = sum_op(myResult, sdata[tid + 64]);
    }

    __syncthreads();

    // fully unroll reduction within a single warp
    if ((blockSize >= 64) && (tid < 32))
    {
        sdata[tid] = myResult = sum_op(myResult, sdata[tid + 32]);
    }

    __syncthreads();

    if ((blockSize >= 32) && (tid < 16))
    {
        sdata[tid] = myResult = sum_op(myResult, sdata[tid + 16]);
    }

    __syncthreads();

    if ((blockSize >= 16) && (tid <  8))
    {
        sdata[tid] = myResult = sum_op(myResult, sdata[tid + 8]);
    }

    __syncthreads();

    if ((blockSize >= 8) && (tid <  4))
    {
        sdata[tid] = myResult = sum_op(myResult, sdata[tid + 4]);
    }

    __syncthreads();

    if ((blockSize >= 4) && (tid <  2))
    {
        sdata[tid] = myResult = sum_op(myResult, sdata[tid + 2]);
    }

    __syncthreads();

    if ((blockSize >= 2) && (tid <  1))
    {
        sdata[tid] = myResult = sum_op(myResult, sdata[tid + 1]);
    }

    __syncthreads();

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = myResult;
}

bool isPow2(unsigned int x)
{
    return x > 1 && ((x&(x - 1)) == 0);
}

////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
void reduce(int size, int threads, int blocks, float *d_idata, float *d_odata)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

    // choose which of the optimized versions of reduction to launch
    if (isPow2(size))
    {
        switch (threads)
        {
        case 512:
            max_kernel<512, true> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
            break;

        case 256:
            max_kernel<256, true> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
            break;

        case 128:
            max_kernel<128, true> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
            break;

        case 64:
            max_kernel<64, true> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
            break;

        case 32:
            max_kernel<32, true> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
            break;

        case 16:
            max_kernel<16, true> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
            break;

        case  8:
            max_kernel<8, true> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
            break;

        case  4:
            max_kernel<4, true> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
            break;

        case  2:
            max_kernel<2, true> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
            break;

        case  1:
            max_kernel<1, true> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
            break;
        }
    }
    else
    {
        switch (threads)
        {
        case 512:
            max_kernel<512, false> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
            break;

        case 256:
            max_kernel<256, false> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
            break;

        case 128:
            max_kernel<128, false> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
            break;

        case 64:
            max_kernel<64, false> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
            break;

        case 32:
            max_kernel<32, false> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
            break;

        case 16:
            max_kernel<16, false> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
            break;

        case  8:
            max_kernel<8, false> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
            break;

        case  4:
            max_kernel<4, false> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
            break;

        case  2:
            max_kernel<2, false> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
            break;

        case  1:
            max_kernel<1, false> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);
            break;
        }
    }
}

#define MAX_BLOCK_DIM_SIZE 65535

unsigned int nextPow2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

////////////////////////////////////////////////////////////////////////////////
// Compute the number of threads and blocks to use for the given reduction kernel
// For the kernels >= 3, we set threads / block to the minimum of maxThreads and
// n/2. For kernels < 3, we set to the minimum of maxThreads and n.  For kernel
// 6, we observe the maximum specified number of blocks, because each thread in
// that kernel can process a variable number of elements.
////////////////////////////////////////////////////////////////////////////////
void getNumBlocksAndThreads(int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
{
    threads = (n < maxThreads * 2) ? nextPow2((n + 1) / 2) : maxThreads;
    blocks = (n + (threads * 2 - 1)) / (threads * 2);
    blocks = maxBlocks < blocks ? maxBlocks : blocks;
}

float cudaReduce(int  n,
    int  numThreads,
    int  numBlocks,
    int  maxThreads,
    int  maxBlocks,
    float *d_idata,
    float *d_odata)
{
    float gpu_result = 0;

    cudaDeviceSynchronize();

    // execute the kernel
    reduce(n, numThreads, numBlocks, d_idata, d_odata);

    // sum partial block sums on GPU
    int s = numBlocks;

    while (s > 1)
    {
        int threads = 0, blocks = 0;
        getNumBlocksAndThreads(s, maxBlocks, maxThreads, blocks, threads);

        reduce(s, threads, blocks, d_odata, d_odata);

        s = (s + (threads * 2 - 1)) / (threads * 2);
    }

    cudaDeviceSynchronize();

    // copy final sum from device to host
    cudaMemcpy(&gpu_result, d_odata, sizeof(float), cudaMemcpyDeviceToHost);
    return gpu_result;
}

float cudaSum(float *d_idata, int size)
{
    int maxThreads = 256;  // number of threads per block
    int maxBlocks = 64;
    int numBlocks = 0;
    int numThreads = 0;
    getNumBlocksAndThreads(size, maxBlocks, maxThreads, numBlocks, numThreads);

    float *d_odata = nullptr;
    cudaMalloc((void **)&d_odata, numBlocks*sizeof(float));
    cudaMemset(d_odata, 0, numBlocks*sizeof(float));

    float gpu_result = cudaReduce(size, numThreads, numBlocks, maxThreads, maxBlocks,
        d_idata, d_odata);

    cudaFree(d_odata);
    return gpu_result;
}

__global__ void colorKernel(const Vec3f *color, int framenumber, float *r, float *g, float *b) {
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    int i = (scrheight - y - 1) * scrwidth + x; // pixel index in buffer	

    Vec3f tempcol = color[i] / framenumber;

    r[i] = tempcol.x;
    g[i] = tempcol.y;
    b[i] = tempcol.z;
}

/// 

__device__ Vec3f directcolor(const Vec3f& color, float r_mean, float g_mean, float b_mean) {
    float r = clamp(color.x, 0.0f, 1.0f);
    float g = clamp(color.y, 0.0f, 1.0f);
    float b = clamp(color.z, 0.0f, 1.0f);
    return Vec3f(r, g, b);
}
__device__ Vec3f gammacorrect(const Vec3f& color, float r_mean, float g_mean, float b_mean) {
    float r = clamp(color.x, 0.0f, 1.0f);
    float g = clamp(color.y, 0.0f, 1.0f);
    float b = clamp(color.z, 0.0f, 1.0f);
    return Vec3f(
        powf(r, 1.0f / 2.2f),
        powf(g, 1.0f / 2.2f),
        powf(b, 1.0f / 2.2f));
}
__device__ Vec3f filmic(const Vec3f& color) {
    //C = 0.8; , gamma correction only
    float C = 0.39;// , gamma correction with filmic tonemapping
    Vec3f x = color;// 1 - expf(-X);
    Vec3f t1 = x * x * 6.2f;
    Vec3f t2 = x * C;
    return (t1 + t2) / (t1 + t2 * 4.1 + Vec3f(0.05, 0.05, 0.05)) + x * (0.634 * C - 0.247);
}
__device__ Vec3f reinhard(const Vec3f& color, float r_mean, float g_mean, float b_mean) {
    float lum = 0.299f * color.x + 0.587f * color.y + 0.114f * color.z;
    float l_mean = 0.299f * r_mean + 0.587f * g_mean + 0.114f * b_mean;

    float m = 0.77;// %Contrast [0.3f, 1.f]
    float c = 0.5;// %Chromatic Adaptation  [0.f, 1.f]
    float a = 0.0;// %Light Adaptation  [0.f, 1.f]
    float f = 0.0; //%Intensity  [-35.f, 10.f] (void*)func = intuitiveintensity //specify by log scale

    f = expf(-f);
    float r(color.x), g(color.y), b(color.z);

    float r_lc = c * r + (1.0 - c) * lum;    //%local adaptation
    float r_gc = c * r_mean + (1.0 - c) * l_mean;  //%global adaptation
    float r_ca = a * r_lc + (1.0 - a) * r_gc;    // %pixel adaptation

    float g_lc = c * g + (1.0 - c) * lum;   // %local adaptation
    float g_gc = c * g_mean + (1.0 - c) * l_mean;//  %global adaptation
    float g_ca = a * g_lc + (1.0 - a) * g_gc;    // %pixel adaptation

    float b_lc = c * b + (1.0 - c) * lum;   // %local adaptation
    float b_gc = c * b_mean + (1.0 - c) * l_mean; // %global adaptation
    float b_ca = a * b_lc + (1.0 - a) * b_gc;   //  %pixel adaptation

    r = r / (r + powf(f * r_ca, m));
    g = g / (g + powf(f * g_ca, m));
    b = b / (b + powf(f * b_ca, m));

    r = clamp(r, 0.0f, 1.0f);
    g = clamp(g, 0.0f, 1.0f);
    b = clamp(b, 0.0f, 1.0f);

    return Vec3f(r, g, b);
}
#define tonemap_kernel reinhard

__global__ void tonemappingKernel(Vec3f* output, Vec3f* accumbuffer, unsigned int framenumber, float r_mean, float g_mean, float b_mean)
{
    // assign a CUDA thread to every pixel by using the threadIndex
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    // global threadId, see richiesams blogspot
    int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    //int pixelx = threadId % scrwidth; // pixel x-coordinate on screen
    //int pixely = threadId / scrwidth; // pixel y-coordintate on screen

    int i = (scrheight - y - 1) * scrwidth + x; // pixel index in buffer	
    int pixelx = x; // pixel x-coordinate on screen
    int pixely = scrheight - y - 1; // pixel y-coordintate on screen

    // averaged colour: divide colour by the number of calculated frames so far
    Vec3f tempcol = accumbuffer[i] / framenumber;

    Colour fcolour;
//     Vec3f colour = Vec3f(clamp(tempcol.x, 0.0f, 1.0f), clamp(tempcol.y, 0.0f, 1.0f), clamp(tempcol.z, 0.0f, 1.0f));
    Vec3f colour = tempcol;

    // convert from 96-bit to 24-bit colour + perform gamma correction
    Vec3f colorMapped = tonemap_kernel(colour, r_mean, g_mean, b_mean);
    fcolour.components = make_uchar4(
        (unsigned char)((colorMapped.x) * 255),
        (unsigned char)((colorMapped.y) * 255),
        (unsigned char)((colorMapped.z) * 255), 1);

    // store pixel coordinates and pixelcolour in OpenGL readable outputbuffer
    output[i] = Vec3f(x, y, fcolour.c);
}

#endif // tonemap_h__
