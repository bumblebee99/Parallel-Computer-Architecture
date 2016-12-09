#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <cmath>

#define HEIGHT 333
#define WIDTH 1366
#define PI 3.14159
#define TRUE 1
#define FALSE 0
#define MAX_BRIGHTNESS 255

typedef int pixel_t;

__device__ int global_max = 0;
__device__ float alpha = 0;

__host__ void gaussian_filter(float *, const float);
__global__ void convolution(pixel_t *, pixel_t *, float *, const int);
__global__ void convolution1(pixel_t *cin, pixel_t *cout, pixel_t *mask, const int mSize);
__device__ void Travers(int, int, pixel_t *, pixel_t *, pixel_t *);
__global__ void pixel_hypot(pixel_t *, pixel_t *, pixel_t *, pixel_t *);
__global__ void findNonMax(pixel_t *, pixel_t *, pixel_t *, const int, const int);
__global__ void hysterisisThreshold(pixel_t *, pixel_t *, pixel_t *);

__global__ void kernel2(pixel_t*, pixel_t*, int *);
__global__ void kernel3(pixel_t*, pixel_t*);
__global__ void houghlines(pixel_t*, pixel_t*, pixel_t*, int *, pixel_t*);
__global__ void dilation(pixel_t*, pixel_t*);
__global__ void erosion(pixel_t*, pixel_t*);
/***************************************************************************/

__global__ void houghlines(pixel_t* d_img_cannyout, pixel_t* d_hough_out_img, pixel_t* d_img_houghout, int *d_max, pixel_t* d_out_img)
{
	float theta;
	int nrho = (int)sqrt((float)(HEIGHT*HEIGHT) + (float)(WIDTH*WIDTH)) + 1;
	int ntheta = 271;    // -90 ~ 180
	float rho = 0, rad = PI / 180;
	__shared__ int max;
	max = 0;
	*d_max = 0;

	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if (i<HEIGHT && j<WIDTH)
	{
		if (d_img_cannyout[i*WIDTH + j] == 1)
		{
			for (theta = -90; theta <= 180; theta++)

			{
				rho = i*cos(theta*rad) + j*sin(theta*rad);
				if (rho>0 && rho<nrho)

				{

					atomicAdd(&d_img_houghout[(int)rho * ntheta + (int)(theta + 90)], 1); //TODO: SHOULD INITIALIZE TO ZERO

					if (max < (int)d_img_houghout[(int)rho* ntheta + (int)(theta + 90)])
					{

						max = (int)d_img_houghout[(int)rho* ntheta + (int)(theta + 90)];
					}
				}

			}
		}
	}
	atomicMax(&global_max, (int)max);   // if time permits change this to global_max
	*d_max = global_max;
}

/****************************************************************/


__global__ void kernel2(pixel_t* d_img_houghout, pixel_t* d_hough_out_img, int *d_max)
{
	int nrho = (int)sqrt((float)(HEIGHT*HEIGHT) + (float)(WIDTH*WIDTH)) + 1;
	int ntheta = 271;    // -90 ~ 180
	int k;
	k = *d_max;
	alpha = (float)255 / k;
	//printf("The alpha is:%f and k is:%d\n",alpha,k);
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	if (i<nrho && j<ntheta)
	{
		d_hough_out_img[i*ntheta + j] = (pixel_t)(alpha*d_img_houghout[i*ntheta + j]);
	}
}
/********************************************************************/
__global__ void kernel3(pixel_t* d_hough_out_img, pixel_t* d_out_img)
{
	float theta;
	int nrho = (int)sqrt((float)(HEIGHT*HEIGHT) + (float)(WIDTH*WIDTH)) + 1;
	int ntheta = 271;    // -90 ~ 180
	float rho = 0, rad = PI / 180;
	int thresh = 70;

	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	if (i<HEIGHT && j<WIDTH)
	{
		for (theta = -90; theta<180; theta++)
		{
			rho = i*cos(theta*rad) + j*sin(theta*rad);
			if (rho>0 && rho<nrho && d_hough_out_img[(int)rho* ntheta + (int)(theta + 90)]>thresh)
			{
				d_out_img[i*WIDTH + j] = 255;
			}

		}
	}
}


/****************************************************************/
__global__ void hysterisisThreshold(pixel_t *d_edgepoints, pixel_t *d_edges, pixel_t *d_visitedmap)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (d_edgepoints[row* WIDTH + col] == 1)
	{
		d_edges[row* WIDTH + col] = 1;
		Travers(row, col, d_edgepoints, d_edges, d_visitedmap);
		d_visitedmap[row* WIDTH + col] = 1;
	}

}

/**********************************************************************/
__device__ void Travers(int row, int col, pixel_t *d_edgepoints, pixel_t * d_edges, pixel_t *d_visitedmap)
{


	if (d_visitedmap[row * WIDTH + col] == 1)
		return;

	//1
	if (d_edgepoints[(row + 1) * WIDTH + col] == 2)
	{
		d_edges[(row + 1) * WIDTH + col] = 1;
		d_visitedmap[(row + 1) * WIDTH + col] = 1;
		Travers(row + 1, col, d_edgepoints, d_edges, d_visitedmap);
		return;
	}
	//2
	if (d_edgepoints[(row + 1) * WIDTH + col - 1] == 2)
	{
		d_edges[(row + 1) * WIDTH + col - 1] = 1;
		d_visitedmap[(row + 1) * WIDTH + col - 1] = 1;
		Travers(row + 1, col - 1, d_edgepoints, d_edges, d_visitedmap);
		return;
	}
	//3
	if (d_edgepoints[(row)* WIDTH + col - 1] == 2)
	{
		d_edges[(row)* WIDTH + col - 1] = 1;
		d_visitedmap[(row)* WIDTH + col - 1] = 1;
		Travers(row, col - 1, d_edgepoints, d_edges, d_visitedmap);
		return;
	}

	//4
	if (d_edgepoints[(row - 1) * WIDTH + col - 1] == 2)
	{
		d_edges[(row - 1) * WIDTH + col - 1] = 1;
		d_visitedmap[(row - 1) * WIDTH + col - 1] = 1;
		Travers(row - 1, col - 1, d_edgepoints, d_edges, d_visitedmap);
		return;
	}

	//5
	if (d_edgepoints[(row - 1) * WIDTH + col] == 2)
	{
		d_edges[(row - 1) * WIDTH + col] = 1;
		d_visitedmap[(row - 1) * WIDTH + col] = 1;
		Travers(row - 1, col, d_edgepoints, d_edges, d_visitedmap);
		return;
	}

	//6
	if (d_edgepoints[(row - 1) * WIDTH + col + 1] == 2)
	{
		d_edges[(row - 1) * WIDTH + col + 1] = 1;
		d_visitedmap[(row - 1) * WIDTH + col + 1] = 1;
		Travers(row - 1, col + 1, d_edgepoints, d_edges, d_visitedmap);
		return;
	}

	//7
	if (d_edgepoints[(row)* WIDTH + col + 1] == 2)
	{
		d_edges[(row)* WIDTH + col + 1] = 1;
		d_visitedmap[(row)* WIDTH + col + 1] = 1;
		Travers(row, col + 1, d_edgepoints, d_edges, d_visitedmap);
		return;
	}

	//8
	if (d_edgepoints[(row + 1) * WIDTH + col + 1] == 2)
	{
		d_edges[(row + 1) * WIDTH + col + 1] = 1;
		d_visitedmap[(row + 1) * WIDTH + col + 1] = 1;
		Travers(row + 1, col + 1, d_edgepoints, d_edges, d_visitedmap);
		return;
	}
	return;
}


/*********************************************************/
__global__ void findNonMax(pixel_t *d_nonMax, pixel_t *d_thisAngle, pixel_t *d_edgepoints, const int tmin, const int tmax)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	pixel_t thisAngle = d_thisAngle[row* WIDTH + col];
	pixel_t nonMax = d_nonMax[row* WIDTH + col];

	if (row >0 && col > 0 && row<HEIGHT && col<WIDTH)
	{
		//Horizontal Edge
		if (((-22.5 < thisAngle) && (thisAngle <= 22.5)) || ((157.5 < thisAngle) && (thisAngle <= -157.5)))
		{
			if ((d_nonMax[row* WIDTH + col] < d_nonMax[row* WIDTH + col + 1]) || (d_nonMax[row* WIDTH + col] < d_nonMax[row* WIDTH + col - 1]))
				nonMax = 0;
		}

		//Vertical Edge
		if (((-112.5 < thisAngle) && (thisAngle <= -67.5)) || ((67.5 < thisAngle) && (thisAngle <= 112.5)))
		{
			if ((d_nonMax[row* WIDTH + col] < d_nonMax[(row + 1)* WIDTH + col]) || (d_nonMax[row* WIDTH + col] < d_nonMax[(row - 1)* WIDTH + col]))
				nonMax = 0;
		}

		//+45 Degree Edge
		if (((-67.5 < thisAngle) && (thisAngle <= -22.5)) || ((112.5 < thisAngle) && (thisAngle <= 157.5)))
		{
			if ((d_nonMax[row* WIDTH + col] < d_nonMax[(row + 1)* WIDTH + col - 1]) || (d_nonMax[row* WIDTH + col] < d_nonMax[(row - 1)* WIDTH + col + 1]))
				nonMax = 0;
		}

		//-45 Degree Edge
		if (((-157.5 < thisAngle) && (thisAngle <= -112.5)) || ((67.5 < thisAngle) && (thisAngle <= 22.5)))
		{
			if ((d_nonMax[row* WIDTH + col] < d_nonMax[(row + 1)* WIDTH + (col + 1)]) || (d_nonMax[row* WIDTH + col] < d_nonMax[(row - 1)* WIDTH + col - 1]))
				nonMax = 0;
		}

		d_nonMax[row* WIDTH + col] = nonMax;

		if (d_nonMax[row* WIDTH + col] > tmax)
			d_edgepoints[row* WIDTH + col] = 1;
		else if ((d_nonMax[row* WIDTH + col] < tmax) && (d_nonMax[row* WIDTH + col] > tmin))
			d_edgepoints[row* WIDTH + col] = 2;
		else if (d_nonMax[row* WIDTH + col] < tmin)
			d_edgepoints[row* WIDTH + col] = 0;
	}
}

/*******************************************************/
__global__ void pixel_hypot(pixel_t *d_mGx, pixel_t *d_mGy, pixel_t *d_gradient, pixel_t *d_thisAngle)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	long int  gradient = 0;
	if (row >0 && col > 0 && row<HEIGHT && col<WIDTH)
	{
		pixel_t pixel_gx, pixel_gy;
		pixel_gx = d_mGx[row* WIDTH + col];
		pixel_gy = d_mGy[row* WIDTH + col];
		gradient = (pixel_t)sqrt((float)((pixel_gx* pixel_gx) + (pixel_gy * pixel_gy)));
		if (gradient >= 255)
			d_gradient[row* WIDTH + col] = 255;
		else
			d_gradient[row* WIDTH + col] = (pixel_t)gradient;

		if (pixel_gx != 0)
			d_thisAngle[row* WIDTH + col] = (atan2f((float)pixel_gy, (float)pixel_gx) * 180) / PI;
		else if (pixel_gy != 0)
			d_thisAngle[row* WIDTH + col] = 90;
	}
}

/*******************************************************/
__host__ void gaussian_filter(float *gout, const float sigma)
{
	int n = 7; //2 * (int)(2 * sigma) + 3;		// size of gaussian mask matrix
	const float mean = (float)floor(n / 2.0);
	int i, j;

	/* The gMask size is calculated from the value of sigma(n= 2 * (int)(2 * sigma) + 3 )
	However due to errors in unallocatiion of host memoery in gaussian_filter kernel , we fixed the value of sigma, hence n*/

	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			gout[i*n + j] = exp(-0.5 * (pow((double)(i - mean) / sigma, 2.0) +
				pow((double)(j - mean) / sigma, 2.0)))
				/ (2 * PI * sigma * sigma);
		}
	}
}

/*******************************************************/
__global__ void convolution(pixel_t *cin, pixel_t *cout, float *mask, const int mSize)
{

	int  k = mSize / 2;
	int i, j;
	float mpix = 0;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if ((row >= k && row < HEIGHT) && (col >= k && col < WIDTH))
	{

		for (i = -k; i <= k; i++)
		{
			for (j = -k; j <= k; j++)
			{

				mpix += cin[(row - i) * WIDTH + col - j] * (float)mask[(i + k) * mSize + (j + k)];
			}
		}

		if (mpix < 0.0)
			mpix = 0.0;
		else if (mpix >255.0)
			mpix = 255.0;

		cout[row * WIDTH + col] = (pixel_t)mpix;
	}
}

/*****************************************************************/

__global__ void convolution1(pixel_t *cin, pixel_t *cout, pixel_t *mask, const int mSize)
{
	int  k = mSize / 2;
	int i, j;
	float mpix = 0;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if ((row >= k && row < HEIGHT) && (col >= k && col < WIDTH))
	{

		for (i = -k; i <= k; i++)
		{
			for (j = -k; j <= k; j++)
			{
				mpix += cin[(row - i) * WIDTH + col - j] * (float)mask[(i + k) * mSize + (j + k)];
			}
		}

		if (mpix < 0.0)
			mpix = 0.0;
		else if (mpix >255.0)
			mpix = 255.0;

		cout[row * WIDTH + col] = (pixel_t)mpix;
	}
}
//////////////////////////////////////////////////////////////////
__global__ void dilation(pixel_t *diin, pixel_t *diout)
{
	int mSize = 5;

	int  k = mSize / 2;
	int i, j;
	int count = 0;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
   
	
	
	if ((row >= k && row < HEIGHT-k) && (col >= k && col < WIDTH-k))
	{

		for (i = -k; i <= k; i++)
		{
			for (j = -k; j <= k; j++)
			{

				if (diin[(row - i) * WIDTH + col - j] == 1)
					count += 1;
			}
		}

		if (count >= 1)
			diout[row * WIDTH + col] = 1;
		else
			diout[row * WIDTH + col] = 0;

	}
}

//////////////////////////////////////////////////////////////////
__global__ void erosion(pixel_t *erin, pixel_t *erout)
{
	int mSize = 11;

	int  k = mSize / 2;
	int i, j;
	int count = 0;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
   
	
	
	if ((row >= k && row < HEIGHT-k) && (col >= k && col < WIDTH-k))
	{

		for (i = -k; i <= k; i++)
		{
			for (j = -k; j <= k; j++)
			{

				if (erin[(row - i) * WIDTH + col - j] == 1)
					count += 1;
			}
		}

		if (count == mSize*mSize)
			erout[row * WIDTH + col] = 1;
		else
			erout[row * WIDTH + col] = 0;

	}
}

/**************************************************************/
int main(int argc, char *argv[])
{

	pixel_t *h_img_ip, *h_img_op;    // input and output images in the host
	pixel_t *d_img_ip, *d_img_op;	// input and output images in the device
	pixel_t *d_gradient, *h_gradient, *d_thisAngle, *h_thisAngle;
	int i, j;
	int dev_count;
	const float sigma = 1.4;
	float *h_gMask, *d_gMask;
	pixel_t *d_gGx, *h_gGx, *d_gGy, *h_gGy;
	pixel_t *d_mGx, *d_mGy;
	pixel_t *h_nonMax, *d_nonMax;
	pixel_t *h_edgepoints, *d_edgepoints;
	pixel_t *h_edges, *d_edges;
	pixel_t *d_visitedmap;
	int n = 7; //2 * (int)(2 * sigma) + 3;		// size of gaussian mask matrix

	pixel_t h_mGx[3][3] = { { -1, 0, 1 },
	{ -2, 0, 2 },
	{ -1, 0, 1 }
	};
	pixel_t h_mGy[3][3] = { { 1, 2, 1 },
	{ 0, 0, 0 },
	{ -1, -2, -1 }
	};

	h_img_ip = (pixel_t *)malloc(HEIGHT * WIDTH *sizeof(pixel_t));
	h_img_op = (pixel_t *)malloc(HEIGHT * WIDTH *sizeof(pixel_t));
	h_gGx = (pixel_t *)malloc(HEIGHT * WIDTH *sizeof(pixel_t));
	h_gGy = (pixel_t *)malloc(HEIGHT * WIDTH *sizeof(pixel_t));
	h_thisAngle = (pixel_t *)malloc(HEIGHT * WIDTH *sizeof(pixel_t));
	h_gradient = (pixel_t *)malloc(HEIGHT * WIDTH *sizeof(pixel_t));
	h_nonMax = (pixel_t *)malloc(HEIGHT * WIDTH *sizeof(pixel_t));
	h_edgepoints = (pixel_t *)malloc(HEIGHT * WIDTH *sizeof(pixel_t));
	h_edges = (pixel_t *)malloc(HEIGHT * WIDTH *sizeof(pixel_t));
	h_gMask = (float *)malloc(n * n * sizeof(float));
	FILE *fp1, *fp2, *fp3, *fp4, *fp5, *fp6;

	cudaGetDeviceCount(&dev_count);
	printf("Dev count: %d\n", dev_count);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);

	printf("  Total amount of constant memory:           %lu bytes\n", deviceProp.totalConstMem);
	printf("  Total amount of shared memory per block:       %lu bytes\n", deviceProp.sharedMemPerBlock);
	printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
	printf("  Warp size:                                     %d\n", deviceProp.warpSize);
	printf("  Maximum number of threads per multiprocessor:  %d\n", deviceProp.maxThreadsPerMultiProcessor);
	printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);

	cudaMalloc(&d_img_ip, HEIGHT * WIDTH * sizeof(pixel_t));
	cudaMalloc(&d_img_op, HEIGHT * WIDTH * sizeof(pixel_t));
	cudaMalloc(&d_gGx, HEIGHT * WIDTH * sizeof(pixel_t));
	cudaMalloc(&d_gGy, HEIGHT * WIDTH * sizeof(pixel_t));
	cudaMalloc(&d_gradient, HEIGHT * WIDTH * sizeof(pixel_t));
	cudaMalloc(&d_thisAngle, HEIGHT * WIDTH * sizeof(pixel_t));
	cudaMalloc(&d_nonMax, HEIGHT * WIDTH * sizeof(pixel_t));
	cudaMalloc(&d_edgepoints, HEIGHT * WIDTH * sizeof(pixel_t));
	cudaMalloc(&d_edges, HEIGHT * WIDTH * sizeof(pixel_t));
	cudaMalloc(&d_visitedmap, HEIGHT * WIDTH * sizeof(pixel_t));
	cudaMalloc(&d_mGx, 3 * 3 * sizeof(pixel_t));
	cudaMalloc(&d_mGy, 3 * 3 * sizeof(pixel_t));
	cudaMalloc(&d_gMask, n * n * sizeof(float));


	fp1 = fopen("car1.txt", "r");
	fp2 = fopen("cannyoutput.txt", "w");

	for (i = 0; i < HEIGHT; i++)
	{
		for (j = 0; j < WIDTH; j++)
		{
			fscanf(fp1, "%d ", &h_img_ip[i*WIDTH + j]);
		}
	}

	printf("before gaussian filter\n");
	gaussian_filter(h_gMask, sigma);

	cudaMemcpy(d_gMask, h_gMask, n * n* sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_mGx, h_mGx, 3 * 3 * sizeof(pixel_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_mGy, h_mGy, 3 * 3 * sizeof(pixel_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_img_ip, h_img_ip, HEIGHT * WIDTH * sizeof(pixel_t), cudaMemcpyHostToDevice);

	cudaMemset(d_img_op, 0, HEIGHT * WIDTH * sizeof(pixel_t));

	const dim3 block(16, 16, 1);
	const dim3 grid((WIDTH + block.x - 1) / block.x, (HEIGHT + block.y - 1) / block.y);


	printf("before canny filter\n");

	convolution << <grid, block >> >(d_img_ip, d_img_op, d_gMask, n);
	cudaDeviceSynchronize();
	cudaMemcpy(h_img_op, d_img_op, HEIGHT * WIDTH * sizeof(pixel_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(d_img_op, h_img_op, HEIGHT * WIDTH * sizeof(pixel_t), cudaMemcpyHostToDevice);
	printf("after 1st convolution\n");

	convolution1 << <grid, block >> >(d_img_op, d_gGx, d_mGx, 3);
	cudaDeviceSynchronize();
	cudaMemcpy(h_gGx, d_gGx, HEIGHT * WIDTH * sizeof(pixel_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(d_gGx, h_gGx, HEIGHT * WIDTH * sizeof(pixel_t), cudaMemcpyHostToDevice);

	convolution1 << <grid, block >> >(d_img_op, d_gGy, d_mGy, 3);
	cudaDeviceSynchronize();
	cudaMemcpy(h_gGy, d_gGy, HEIGHT * WIDTH * sizeof(pixel_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(d_gGy, h_gGy, HEIGHT * WIDTH * sizeof(pixel_t), cudaMemcpyHostToDevice);
	cudaMemset(d_gradient, 0, HEIGHT * WIDTH * sizeof(pixel_t));
	pixel_hypot << <grid, block >> >(d_gGx, d_gGy, d_gradient, d_thisAngle);
	cudaDeviceSynchronize();


	cudaMemcpy(h_gradient, d_gradient, HEIGHT * WIDTH * sizeof(pixel_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(d_gradient, h_gradient, HEIGHT * WIDTH * sizeof(pixel_t), cudaMemcpyHostToDevice);
	cudaMemcpy(h_thisAngle, d_thisAngle, HEIGHT * WIDTH * sizeof(pixel_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(d_thisAngle, h_thisAngle, HEIGHT * WIDTH * sizeof(pixel_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_nonMax, h_gradient, HEIGHT * WIDTH * sizeof(pixel_t), cudaMemcpyHostToDevice);   // for non maximal supression

	cudaMemset(d_edgepoints, 0, HEIGHT * WIDTH * sizeof(pixel_t));
	findNonMax << <grid, block >> >(d_nonMax, d_thisAngle, d_edgepoints, 15, 20);
	cudaDeviceSynchronize();
	cudaMemcpy(h_nonMax, d_nonMax, HEIGHT * WIDTH * sizeof(pixel_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_edgepoints, d_edgepoints, HEIGHT * WIDTH * sizeof(pixel_t), cudaMemcpyDeviceToHost);

	cudaMemcpy(d_edgepoints, h_edgepoints, HEIGHT * WIDTH * sizeof(pixel_t), cudaMemcpyHostToDevice);
	cudaMemcpy(h_edgepoints, d_edgepoints, HEIGHT * WIDTH * sizeof(pixel_t), cudaMemcpyDeviceToHost);
	cudaMemset(d_edges, 0, HEIGHT * WIDTH * sizeof(pixel_t));
	cudaMemset(d_visitedmap, 0, HEIGHT * WIDTH * sizeof(pixel_t));
	hysterisisThreshold << <grid, block >> >(d_edgepoints, d_edges, d_visitedmap);
	cudaDeviceSynchronize();
	cudaMemcpy(h_edges, d_edges, HEIGHT * WIDTH * sizeof(pixel_t), cudaMemcpyDeviceToHost);

	printf(" kernels executed\n");

	for (i = 0; i < HEIGHT; i++)
	{
		for (j = 0; j < WIDTH; j++)
		{

			fprintf(fp2, "%d\t", h_edges[i*WIDTH + j]);
		}
		fprintf(fp2, "\n");
	}
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));
	cudaFree(d_img_ip);
	cudaFree(h_img_ip);
	cudaFree(d_img_op);
	cudaFree(h_img_op);
	cudaFree(d_gradient);
	cudaFree(h_gradient);
	cudaFree(h_thisAngle);
	cudaFree(d_thisAngle);
	cudaFree(d_mGx);
	cudaFree(d_mGy);
	cudaFree(h_nonMax);
	cudaFree(d_nonMax);
	cudaFree(h_edgepoints);
	cudaFree(d_edgepoints);
	cudaFree(d_visitedmap);
	fclose(fp1);
	fclose(fp2);


	/*******************************************************************/

	pixel_t *h_hough_out_img, *d_hough_out_img;
	pixel_t *h_img_houghout, *d_img_houghout; //just for h_hough_out_img output without alpha multiplication
	pixel_t *h_out_img, *d_out_img; //display image

	int *d_max, *h_max;
	int nrho, ntheta;

	nrho = (int)sqrt(HEIGHT*HEIGHT + WIDTH*WIDTH) + 1;
	ntheta = 271;    // -90 ~ 18

	h_out_img = (pixel_t *)malloc(HEIGHT * WIDTH *sizeof(pixel_t));
	h_img_houghout = (pixel_t *)malloc(nrho*ntheta*sizeof(pixel_t));
	h_hough_out_img = (pixel_t *)malloc(nrho*ntheta*sizeof(pixel_t));
	h_max = (int *)malloc(sizeof(int));
	fp3 = fopen("houghoutput.txt", "w");
	fp4 = fopen("output.txt", "w");

	memset(h_hough_out_img, 0, ntheta * nrho * sizeof(pixel_t));
	memset(h_out_img, 0, HEIGHT*WIDTH*sizeof(pixel_t));

	//cudaMalloc(&d_image_cannyout, HEIGHT * WIDTH * sizeof(int));
	cudaMalloc(&d_img_houghout, ntheta * nrho * sizeof(pixel_t));
	cudaMalloc(&d_hough_out_img, ntheta * nrho * sizeof(pixel_t));
	cudaMalloc(&d_max, sizeof(int));
	cudaMalloc(&d_out_img, HEIGHT * WIDTH * sizeof(pixel_t));
	cudaMemset(d_img_houghout, 0, ntheta * nrho * sizeof(pixel_t));
	cudaMemset(d_hough_out_img, 0, ntheta * nrho * sizeof(pixel_t));
	cudaMemset(d_out_img, 0, HEIGHT * WIDTH * sizeof(pixel_t));

	cudaMemcpy(d_edges, h_edges, HEIGHT * WIDTH * sizeof(pixel_t), cudaMemcpyHostToDevice);

	dim3 grid1((WIDTH + block.x - 1) / block.x, (HEIGHT + block.y - 1) / block.y);
	houghlines << <grid1, block >> >(d_edges, d_hough_out_img, d_img_houghout, d_max, d_out_img);    // check the number of threds and blocks
	cudaDeviceSynchronize();

	cudaMemcpy(h_img_houghout, d_img_houghout, nrho * ntheta* sizeof(pixel_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(d_img_houghout, h_img_houghout, nrho * ntheta* sizeof(pixel_t), cudaMemcpyHostToDevice);
	cudaMemcpy(h_max, d_max, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(d_max, h_max, sizeof(int), cudaMemcpyHostToDevice);

	dim3 grid2(sqrt((float)(HEIGHT*HEIGHT + WIDTH*WIDTH)) + 1, 271);
	printf("BEFORE KERNEL2\n");
	kernel2 << <grid2, block >> >(d_img_houghout, d_hough_out_img, d_max);
	printf("AFTER KERNEL2\n");
	cudaDeviceSynchronize();

	dim3 grid3((WIDTH + block.x - 1) / block.x, (HEIGHT + block.y - 1) / block.y);
	printf("BEFORE KERNEL3\n");
	kernel3 << <grid3, block >> >(d_hough_out_img, d_out_img);
	printf("AFTER KERNEL3\n");
	cudaDeviceSynchronize();

	cudaMemcpy(h_out_img, d_out_img, HEIGHT*WIDTH*sizeof(pixel_t), cudaMemcpyDeviceToHost);
	printf("THe max value is:%d\n", *h_max);
	cudaMemcpy(h_hough_out_img, d_hough_out_img, nrho * ntheta* sizeof(pixel_t), cudaMemcpyDeviceToHost);

	for (i = 0; i<nrho; i++)
	{

		for (j = 0; j<ntheta; j++)

		{

			fprintf(fp3, "%d\t", h_hough_out_img[i*ntheta + j]);

		}

		fprintf(fp3, "\n");

	}
	for (i = 0; i<HEIGHT; i++)
	{

		for (j = 0; j<WIDTH; j++)

		{

			fprintf(fp4, "%d\t", h_out_img[i*WIDTH + j]);

		}

		fprintf(fp4, "\n");

	}

	printf("$$$nrho value is:%d$$$\n", nrho);

	cudaFree(d_img_houghout);
	cudaFree(d_max);
	fclose(fp3);
	fclose(fp4);


	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	pixel_t *h_dilout, *d_dilin, *d_dilout;
	h_dilout = (pixel_t *)malloc(HEIGHT * WIDTH *sizeof(pixel_t));


	fp5 = fopen("dilout.txt", "w");

	memset(h_dilout, 0, HEIGHT*WIDTH*sizeof(pixel_t));
	cudaMalloc(&d_dilin, HEIGHT * WIDTH * sizeof(pixel_t));
	cudaMalloc(&d_dilout, HEIGHT * WIDTH * sizeof(pixel_t));
	cudaMemset(d_dilout, 0, HEIGHT * WIDTH * sizeof(pixel_t));
	cudaMemcpy(d_dilin, h_edges, sizeof(pixel_t), cudaMemcpyHostToDevice);
	dilation <<<grid, block >>>(d_edges, d_dilout);
	cudaDeviceSynchronize();
	printf("AFTER DILATION\n");
	cudaMemcpy(h_dilout, d_dilout, HEIGHT*WIDTH*sizeof(pixel_t), cudaMemcpyDeviceToHost);


	for (i = 0; i<HEIGHT; i++)
	{

		for (j = 0; j<WIDTH; j++)

		{

			fprintf(fp5, "%d\t", h_dilout[i*WIDTH + j]);

		}

		fprintf(fp5, "\n");

	}
	///////////////////////////////////////////////////////////////////////
	pixel_t *h_morout, *d_morout;
	h_morout = (pixel_t *)malloc(HEIGHT * WIDTH *sizeof(pixel_t));


	fp6 = fopen("morout.txt", "w");

	memset(h_morout, 0, HEIGHT*WIDTH*sizeof(pixel_t));
	
	cudaMalloc(&d_morout, HEIGHT * WIDTH * sizeof(pixel_t));
	cudaMemset(d_morout, 0, HEIGHT * WIDTH * sizeof(pixel_t));
	
	dilation <<<grid, block >>>(d_dilout, d_morout);
	cudaDeviceSynchronize();
	printf("AFTER DILATION\n");
	cudaMemcpy(h_morout, d_morout, HEIGHT*WIDTH*sizeof(pixel_t), cudaMemcpyDeviceToHost);


	for (i = 0; i<HEIGHT; i++)
	{

		for (j = 0; j<WIDTH; j++)

		{

			fprintf(fp6, "%d\t", h_morout[i*WIDTH + j]);

		}

		fprintf(fp6, "\n");

	}
	
	
	
	
	
	return 0;
}
