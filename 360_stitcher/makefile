CC = gcc
CXX = g++
CFLAGS = -O3 -g
LD_FLAGS = -lcudart \
	 -lcublas \
	 -lopencv_core \
     -lopencv_cudaarithm \
     -lopencv_cudafeatures2d \
     -lopencv_cudaimgproc \
     -lopencv_cudafilters \
     -lopencv_cudalegacy \
     -lopencv_cudawarping \
     -lopencv_cudev \
     -lopencv_features2d \
     -lopencv_highgui \
     -lopencv_imgcodecs \
     -lopencv_imgproc \
     -lopencv_photo \
     -lopencv_shape \
     -lopencv_stitching \
     -lopencv_videoio \
     -lopencv_video \
     -lopencv_videostab \
     -lopencv_xfeatures2d \
     -lopencv_calib3d \
     -lkvazaar

OBJ_DIR=obj
BUILD_DIR=build

DEPS = calibration.h networking.h defs.h debug.h
OBJ = obj/calibration.o obj/networking.o obj/resize.o obj/timed.o obj/netlib.o obj/debug.o

$(OBJ_DIR)/%.o: %.cpp $(DEPS) | obj
	$(CXX) -c -o $@ $< $(CFLAGS) -pthread

stitcher: $(OBJ)
	$(CXX) -o $(BUILD_DIR)/$@ $^ $(LD_FLAGS) -pthread

$(OBJ_DIR)/resize.o: resize.cu
	nvcc -std=c++11 -c -o $@ $< $(CFLAGS)

$(OBJ_DIR)/timed.o: timed.cpp
	$(CXX) -c -o $@ $< $(CFLAGS)

$(OBJ_DIR)/netlib.o: netlib.c
	$(CC) -c -o $@ $< $(CFLAGS)

obj:
	@mkdir $(OBJ_DIR) $(BUILD_DIR)

clean:
	rm -rf $(OBJ_DIR) $(BUILD_DIR)

.PHONY: clean
