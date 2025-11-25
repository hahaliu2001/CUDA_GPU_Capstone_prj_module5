# Gaussian blur using FFT with and without cuFFT

This project implement Gaussian blur algorithms on images using FFT.
Two methods are used to implement it, one is not using cuFFT, another is using cuFFT

The code read image files from data folder, and write Gaussian blur processed images back into two folders: no_cuFFT_out_data folder and with_cuFFT_out_data

please check printout logs from output.txt

## Code Organization

```bin/```
This folder should hold all binary/executable code that is built automatically or manually. Executable code should have use the .exe extension or programming language-specific extension.

```data/```
This folder should hold images sample.

```src/```
The source code should be placed here.

```README.md```
This file should hold the description of the project so that anyone cloning or deciding if they want to clone this repository can understand its purpose to help with their decision.

```Makefile ```
There should be some rudimentary scripts for building your project's code in an automatic fashion.

```run.sh```
directly run `./run.sh` to clean->build and then run the test, the log write to output.txt file

## Key Concepts

Performance Strategies, Image Processing, Gaussian Blur

## Supported OSes

Linux

## Dependencies needed to build/run
openCV, cuFFT

## Build and Run
### Linux
The Linux samples are built using makefiles. To use the makefiles, change the current directory to the sample directory you wish to build, and run make:
```
$ cd <sample_dir>
$ make

## Running the Program
After building the project, you can run the program using the following command:

```bash
make run
```

