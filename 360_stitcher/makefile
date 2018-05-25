INCLUDES=-I./include -I../opencv_build/include -I.
LIBS=-L./lib64 -L../opencv_build/lib \
     -lcudart \
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

CC=g++-7
CFLAGS= $(INCLUDES) $(LIBS)

ODIR=obj
BUILD_DIR=build


DEPS = calibration.h networking.h defs.h

_OBJ = calibration.o networking.o resize.o timed.o
OBJ = obj/calibration.o obj/networking.o obj/resize.o obj/timed.o
#OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

$(ODIR)/%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS) -pthread

stitcher: $(OBJ)
	$(CC) -o $(BUILD_DIR)/$@ $^ $(CFLAGS) -pthread

$(ODIR)/resize.o: resize.cu
	nvcc -std=c++11 -c -o $@ $< $(CFLAGS)

$(ODIR)/timed.o: timed.cpp
	$(CC) -c -o $@ $< $(CFLAGS)

clean:
	rm -f $(ODIR)/*.o *~ core $(INCDIR)/*~
.PHONY: clean

