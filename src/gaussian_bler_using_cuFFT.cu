#include <Exceptions.h>
#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>
#include <cuda_runtime.h>
#include <fstream>
#include <helper_cuda.h>
#include <helper_string.h>
#include <iostream>
#include <npp.h>
#include <string.h>
#include <vector>
#include <filesystem>

#include <cufft.h>
#include <opencv2/opencv.hpp>
#include <fftw3.h>

inline int cudaDeviceInit(int argc, const char **argv)
{
    int deviceCount;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cerr << "CUDA error: no devices supporting CUDA." << std::endl;
        exit(EXIT_FAILURE);
    }

    int dev = findCudaDevice(argc, argv);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    std::cout << "cudaSetDevice GPU" << dev << " = " << deviceProp.name << std::endl;

    checkCudaErrors(cudaSetDevice(dev));

    return dev;
}


namespace fs = std::filesystem;
std::vector<std::string> getAllFilesInDirectory(const std::string& directoryPath) {
    std::vector<std::string> filePaths;
    try {
        if (fs::exists(directoryPath) && fs::is_directory(directoryPath)) {
            for (const auto& entry : fs::directory_iterator(directoryPath)) {
                if (fs::is_regular_file(entry.status())) {
                    filePaths.push_back(entry.path().string());
                }
            }
        } else {
            std::cerr << "Error: Directory does not exist or is not a directory: " << directoryPath << std::endl;
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
    }
    return filePaths;
}

std::string gen_outputgfile(std::string outputFilePath,std::string  sFilename)
{
    std::string sResultFilename = sFilename;

    std::string::size_type dot = sResultFilename.rfind('.');

    if (dot != std::string::npos) {
        sResultFilename = sResultFilename.substr(0, dot);
    }

    dot = sResultFilename.rfind('/');

    if (dot != std::string::npos) {
        sResultFilename = sResultFilename.substr(dot, sResultFilename.length() - dot - 1);
    }

    sResultFilename += "_GaussianBlur.pgm";
    sResultFilename = std::string(outputFilePath) + "/" + sResultFilename;
    return sResultFilename;
}

__global__ void kernelMultiply(cufftComplex* data, const cufftComplex* kernel, int width, int height) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// Check if the thread is within the image bounds
	if (x < width && y < height) {
		int index = y * width + x;

		// Perform element-wise complex multiplication
		cufftComplex dataValue = data[index];
		cufftComplex kernelValue = kernel[index];
		cufftComplex result;

		result.x = dataValue.x * kernelValue.x - dataValue.y * kernelValue.y;
		result.y = dataValue.x * kernelValue.y + dataValue.y * kernelValue.x;

		// Store the result back to the data array
		data[index] = result;
	}
}

// Function to create a 2D Gaussian kernel
cv::Mat createGaussianKernel(int size, double sigma) {
	cv::Mat kernel(size, size, CV_64F);
	double sum = 0.0;
	int center = size / 2;

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			int x = i - center;
			int y = j - center;
			kernel.at<double>(i, j) = exp(-(x * x + y * y) / (2.0 * sigma * sigma)) / (2.0 * M_PI * sigma * sigma);
			sum += kernel.at<double>(i, j);
		}
	}

	// Normalize the kernel
	kernel /= sum;

	return kernel;
}

// Function to shift quadrants of an image
void shiftQuadrants(cv::Mat& image) {
	int cx = image.cols / 2;
	int cy = image.rows / 2;

	cv::Mat q0(image, cv::Rect(0, 0, cx, cy)); // Top-Left
	cv::Mat q1(image, cv::Rect(cx, 0, cx, cy)); // Top-Right
	cv::Mat q2(image, cv::Rect(0, cy, cx, cy)); // Bottom-Left
	cv::Mat q3(image, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

	cv::Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
}


// Function to apply Gaussian blur using FFT
cv::Mat applyGaussianBlurFFT(const cv::Mat& inputImage, int kernelSize, double sigma) {
	cv::Mat kernel = createGaussianKernel(kernelSize, sigma);
	cv::Mat paddedKernel(inputImage.rows, inputImage.cols, CV_64FC1, cv::Scalar(0));

	// Copy the kernel to the center of the paddedKernel
	int dx = (paddedKernel.cols - kernel.cols) / 2;
	int dy = (paddedKernel.rows - kernel.rows) / 2;
	kernel.copyTo(paddedKernel(cv::Rect(dx, dy, kernel.cols, kernel.rows)));

	// Perform 2D FFT on the input image and padded kernel
	cv::Mat inputImageDouble;
	inputImage.convertTo(inputImageDouble, CV_64FC1);
	cv::Mat fftInput, fftKernel;
	cv::dft(inputImageDouble, fftInput, cv::DFT_COMPLEX_OUTPUT);
	cv::dft(paddedKernel, fftKernel, cv::DFT_COMPLEX_OUTPUT);

	
	
	// Multiply the FFTs element-wise
	cv::Mat complexResult;
	cv::mulSpectrums(fftInput, fftKernel, complexResult, 0);

	// Perform inverse FFT to get the blurred image
	cv::Mat blurredImage;
	cv::idft(complexResult, blurredImage, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);

	// Correct quadrant order by shifting
	shiftQuadrants(blurredImage);

	// Convert back to the original image data type
	blurredImage.convertTo(blurredImage, inputImage.type());

	return blurredImage;

}

int main(int argc, char *argv[])
{
    printf("%s Starting...\n\n", argv[0]);
    
    cudaDeviceInit(argc, (const char **)argv);

    std::string path = "data";
    std::vector<std::string> files = getAllFilesInDirectory(path);

    if (files.empty()) {   
        std::cout << "No files found in directory '" << path << "' or an error occurred." << std::endl;
        return EXIT_FAILURE;
    }
    
    for (const std::string& filePath : files) 
    {
        std::string sFilename;
        sFilename = filePath;
        cv::Mat inputImage = cv::imread(sFilename, cv::IMREAD_GRAYSCALE);
	    if (inputImage.empty()) {
		    std::cerr << "Could not open or find the image!" << std::endl;
		    return -1;
	    }

        printf("Input image size: %d x %d\n", inputImage.cols, inputImage.rows);
        

	    // not use cuFFT to perform Gaussian blur
        printf("Applying Gaussian Blur without cuFFT...\n");
        // Apply Gaussian blur with kernel size 9x9 and sigma 2
	    int kernelSize = 12;
	    double sigma = 2;
	    cv::Mat blurredImage = applyGaussianBlurFFT(inputImage, kernelSize, sigma);
		
        std::string sResultFilename = gen_outputgfile("no_cuFFT_out_data", sFilename);
	    cv::imwrite(sResultFilename, blurredImage);

        // use cuFFT to perform Gaussian blur
        printf("Applying Gaussian Blur with cuFFT...\n");
        // Convert the grayscale image to 64-bit floating-point
	    cv::Mat inputImageDouble;
	    inputImage.convertTo(inputImageDouble, CV_64FC1);

        // Create Gaussian kernel
        int width = inputImage.cols;
	    int height = inputImage.rows;
	    dim3 blockSize(32, 32);
	    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

	    cv::Mat kernel = createGaussianKernel(12, 2);
	    cv::Mat paddedKernel(height, width, CV_64FC1, cv::Scalar(0));

	    // Copy the kernel to the center of the paddedKernel
	    int dx = (paddedKernel.cols - kernel.cols) / 2;
	    int dy = (paddedKernel.rows - kernel.rows) / 2;
	    kernel.copyTo(paddedKernel(cv::Rect(dx, dy, kernel.cols, kernel.rows)));

	    cufftComplex* d_data, *d_kernel;
	    cudaMalloc((void**)&d_data, width * height * sizeof(cufftComplex));
	    cudaMalloc((void**)&d_kernel, paddedKernel.cols * paddedKernel.rows * sizeof(cufftComplex));
        
	    // Copy the image and kernel data to the GPU
	    cudaMemcpy(d_data, inputImageDouble.data, width * height * sizeof(cufftComplex), cudaMemcpyHostToDevice);
	    cudaMemcpy(d_kernel, paddedKernel.data, paddedKernel.cols * paddedKernel.rows * sizeof(cufftComplex), cudaMemcpyHostToDevice);
        

	    // Create a cuFFT plan for a 2D complex-to-complex forward FFT
	    cufftHandle gpuFftInput, gpuFftKernel;
        
	    cufftPlan2d(&gpuFftInput, height, width, CUFFT_C2C);
	    cufftPlan2d(&gpuFftKernel, paddedKernel.rows, paddedKernel.cols, CUFFT_C2C);
        
        
	    // Perform the forward FFT on the GPU
	    cufftExecC2C(gpuFftInput, d_data, d_data, CUFFT_FORWARD);
	    cufftExecC2C(gpuFftKernel, d_kernel, d_kernel, CUFFT_FORWARD);
        

	    // Perform complex element-wise multiplication with the Gaussian kernel
	    kernelMultiply << <gridSize, blockSize >> > (d_data, d_kernel, width, height);
	    cudaDeviceSynchronize();
        
	    // Perform the inverse FFT on the GPU
	    cufftExecC2C(gpuFftInput, d_data, d_data, CUFFT_INVERSE);
        
	    // Copy the result back to the CPU
	    cv::Mat gpuBlurredImage(height, width, CV_64FC1);
	    cudaMemcpy(gpuBlurredImage.data, d_data, width * height * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
        
	    shiftQuadrants(gpuBlurredImage);

	    blurredImage.convertTo(gpuBlurredImage, inputImage.type());

        sResultFilename = gen_outputgfile("with_cuFFT_out_data", sFilename);
	    cv::imwrite(sResultFilename, gpuBlurredImage);
        
        
	    // Clean up
	    cufftDestroy(gpuFftInput);
	    cufftDestroy(gpuFftKernel);
	    cudaFree(d_data);
	    cudaFree(d_kernel);

    }

    return 0;
}
