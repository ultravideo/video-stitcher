#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cuda_types.hpp>
#include "opencv2/core/cuda/common.hpp"
#include <stdio.h>

__global__
void resize(int tx, int ty, int cols, int rows, cv::cuda::PtrStepf in, cv::cuda::PtrStepf out)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u < tx && v < ty) {
        int left = u * (cols-1) / tx;
        int top = v * (rows-1) / ty;

        float uu = ((float)u) * (cols-1) / tx - left;
        float vv = ((float)v) * (rows-1) / ty - top;

        //printf("%i, %i, %f\n", left, u, uu);
        out.ptr(v)[u] = (1 - uu)*(1 - vv)*in.ptr(top)[left] +
                        uu*(1 - vv)*in.ptr(top)[left+1] +
                        (1 - uu)*vv*in.ptr(top+1)[left] +
                        uu*vv*in.ptr(top+1)[left+1];
    }
}

using namespace cv::cuda::device;
extern void custom_resize(cv::cuda::GpuMat &in, cv::cuda::GpuMat &out, cv::Size t_size)
{
    int cols = in.cols;
    int rows = in.rows;
    int tx = t_size.width;
    int ty = t_size.height;

    out.create(ty, tx, CV_32FC1);

    dim3 threads(16, 16);
    dim3 grid(divUp(tx, threads.x), divUp(ty, threads.y));

    resize<<<grid,threads>>>(tx, ty, cols, rows, in, out);

    cudaSafeCall(cudaDeviceSynchronize());
}