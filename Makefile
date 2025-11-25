
# Define the compiler and flags
NVCC = /usr/local/cuda/bin/nvcc
CXX = g++
CXXFLAGS = -std=c++17 -I/usr/local/cuda/include -Iinclude
LDFLAGS = -L/usr/local/cuda/lib64 -lcudart -lnppc -lnppial -lnppicc -lnppidei -lnppif -lnppig -lnppim -lnppist -lnppisu -lnppitc

LDLIBS = -lnppisu_static -lnppif_static -lnppc_static -lculibos -lfreeimage  -lculibos -lcudart -lnppif_static 
LDLIBS += -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lcufft -lfftw3

# Define directories
SRC_DIR = src
BIN_DIR = bin
DATA_DIR = data
LIB_DIR = lib

INCLUDES  := -I./Common
INCLUDES += -I./Common/UtilNPP
INCLUDES += -I/usr/include 
INCLUDES += -I/usr/local/include/opencv4

CXXFLAGS += $(INCLUDES)

# Define source files and target executable
SRC = $(SRC_DIR)/gaussian_bler_using_cuFFT.cu
TARGET = $(BIN_DIR)/gaussian_bler_using_cuFFT

# Define the default rule
all: $(TARGET)

# Rule for building the target executable
$(TARGET): $(SRC)
	mkdir -p $(BIN_DIR)
	$(NVCC) $(CXXFLAGS)  $(SRC) -o $(TARGET) $(LDFLAGS) $(LDLIBS) 

# Rule for running the application
run: $(TARGET)
	./$(TARGET) 

# Clean up
clean:
	rm -rf $(BIN_DIR)/*

# Installation rule (not much to install, but here for completeness)
install:
	@echo "No installation required."

# Help command
help:
	@echo "Available make commands:"
	@echo "  make        - Build the project."
	@echo "  make run    - Run the project."
	@echo "  make clean  - Clean up the build files."
	@echo "  make install- Install the project (if applicable)."
	@echo "  make help   - Display this help message."
