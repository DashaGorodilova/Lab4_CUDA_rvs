
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "wb.h"
#include <stdio.h>

#define TILE_WIDTH 16

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void rgba_to_grey(float* rgbImage, float* greyImage, int channels, int width, int height)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x < width && y < height)
	{
		// получаем координату для изображения в градациях серого
		int greyOffset = y * width + x;
		int rgbOffset = greyOffset * channels;
		float r = rgbImage[rgbOffset]; // red value for pixel
		float g = rgbImage[rgbOffset + 1]; // green value for pixel
		float b = rgbImage[rgbOffset + 2]; // blue value for pixel
		// сохраняем изменения масштаба
		// умножаем их на константы с плавающей точкой
		greyImage[greyOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
	}

}

/*__global__ void rgba_to_grey(uchar4* const rgbaImage, unsigned char* const greyImage, int numRows, int numCols)
{
	const int id = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

	if (id < numRows * numCols)
	{
		const unsigned char R = rgbaImage[id].x;
		const unsigned char G = rgbaImage[id].y;
		const unsigned char B = rgbaImage[id].z;
		greyImage[id] = .299f * R + .587f * G + .114 * B;
	}
}*/

static void write_data(char* file_name, unsigned char* data, unsigned int width, unsigned int height, unsigned int channels) 
{
	FILE* handle = fopen(file_name, "w");
	if (channels == 1) 
	{
		fprintf(handle, "P5\n");
	}
	else 
	{
		fprintf(handle, "P6\n");
	}
	fprintf(handle, "#Created by %s\n", __FILE__);
	fprintf(handle, "%d %d\n", width, height);
	fprintf(handle, "255\n");

	fwrite(data, width * channels * sizeof(unsigned char), height, handle);

	fflush(handle);
	fclose(handle);
}

static unsigned char* generate_data(const unsigned int y, const unsigned int x) 
{
	unsigned int i;
	const int maxVal = 255;
	unsigned char* data = (unsigned char*)malloc(y * x * 3);

	unsigned char* p = data;
	for (i = 0; i < y * x; ++i) {
		unsigned short r = rand() % maxVal;
		unsigned short g = rand() % maxVal;
		unsigned short b = rand() % maxVal;
		*p++ = r;
		*p++ = g;
		*p++ = b;
	}
	return data;
}

int main(int argc, char* argv[])
{
	wbArg_t args;
	int imageChannels;
	int imageWidth;
	int imageHeight;
	char* inputImageFile;
	wbImage_t inputImage;
	wbImage_t outputImage;
	float* hostInputImageData;
	float* hostOutputImageData;
	float* deviceInputImageData;
	float* deviceOutputImageData;

	args = wbArg_read(argc, argv); /* чтение входных аргументов */
	inputImageFile = wbArg_getInputFile(args, 0);
	inputImage = wbImport(inputImageFile);
	imageWidth = wbImage_getWidth(inputImage);
	imageHeight = wbImage_getHeight(inputImage);
	// В данной лабораторной значение равно 3
	imageChannels = wbImage_getChannels(inputImage);
	// Так как изображение монохромное, оно содержит только 1 канал
	outputImage = wbImage_new(imageWidth, imageHeight, 1);

	hostInputImageData = wbImage_getData(inputImage);
	hostOutputImageData = wbImage_getData(outputImage);
	wbTime_start(GPU, "Doing GPU Computation(memory + compute) ");
	wbTime_start(GPU, "Doing GPU memory allocation ");
	cudaMalloc((void**)&deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
	cudaMalloc((void**)&deviceOutputImageData, imageWidth * imageHeight * sizeof(float));
	wbTime_stop(GPU, "Doing GPU memory allocation ");
	wbTime_start(Copy, "Copying data to the GPU ");
	cudaMemcpy(deviceInputImageData, hostInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);
	wbTime_stop(Copy, "Copying data to the GPU ");
	wbTime_start(Compute, "Doing the computation on the GPU ");
	dim3 dimGrid(ceil((float)imageWidth / TILE_WIDTH), ceil((float)imageHeight / TILE_WIDTH));
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	rgba_to_grey << <dimGrid, dimBlock>> > (deviceOutputImageData, deviceInputImageData, imageChannels, imageWidth, imageHeight);
	wbTime_stop(Compute, "Doing the computation on the GPU ");
	
	wbTime_start(Copy, "Copying data from the GPU ");
	cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageWidth * imageHeight * sizeof(float), cudaMemcpyDeviceToHost);
	wbTime_stop(Copy, "Copying data from the GPU ");
	wbTime_stop(GPU, "Doing GPU Computation(memory + compute) ");
	wbSolution(args, outputImage);
	cudaFree(deviceInputImageData);
	cudaFree(deviceOutputImageData);
	wbImage_delete(outputImage);
	wbImage_delete(inputImage);
	return 0;
}