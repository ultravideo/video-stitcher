# Live Stitching

## Paper

Please cite [this paper](https://ieeexplore.ieee.org/document/8965900) for the video stitcher:

```M. Atokari, M. Viitanen, A. Mercat, E. Kattainen, and J. Vanne, “Parallax-tolerant 360 live video stitcher,” in Proc. IEEE Int. Conf. Visual Comm. and Image Proc., Sydney, Australia, Dec. 2019.```

Or in BibTex:

```
@inproceedings{Atokari2019,
  author={M. {Atokari} and M. {Viitanen} and A. {Mercat} and E. {Kattainen} and J. {Vanne}},
  booktitle={2019 IEEE Visual Communications and Image Processing (VCIP)},
  title={Parallax-Tolerant 360 Live Video Stitcher},
  year={2019},
  location = {Sydney, Australia},
  doi={10.1109/VCIP47243.2019.8965900},
}
```

## Building
### Requirements
- CUDA
- Custom OpenCV (Included in the directory)
   - Custom OpenCV has some changes in stitching module
- Eigen
- Kvazaar

#### CUDA
- Install CUDA
   - Windows: https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html
      - copy the cuda lib and include folders to $(ProjectDir)
   - Linux: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

#### OpenCV
- Make visual studio project with cmake from $(ProjectDir)../sources to $(ProjectDir)../opencv_build
- set WITH\_CUDA flag true and CUDA\_HOST\_COMPILER to the path where your visual c compiler is. (_VSINSTALLFOLDER_/VC/Tools/MSVC/_VERSIONNUMBER_/bin/Hostx64/x64/cl.exe)
- In visual studio build the project "INSTALL"
- On linux run following commands starting in base folder
```
sudo apt install libgtk2.0-dev pkg-config ffmpeg libavcodec-dev libavformat-dev libavdevice-dev
mkdir opencv_build
cd opencv_build
cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_CUDA=ON -D ENABLE_FAST_MATH=1 -D CUDA_FAST_MATH=1 -D WITH_CUBLAS=1 -D WITH_PNG=OFF -DBUILD_opencv_cudacodec=OFF ../sources/
make -j8
sudo make install
```

#### Eigen
- Windows:
   - Download Eigen and copy the Eigen folder to $(ProjectDir)include
      - http://eigen.tuxfamily.org/index.php?title=Main_Page
- Linux:
   - run following commands in base folder
```
git clone https://github.com/eigenteam/eigen-git-mirror
cp -r eigen-git-mirror/Eigen 360_stitcher
```

#### Kvazaar
https://gitlab.tut.fi/TIE/ultravideo/kvazaar#compiling-kvazaar

## Building and running on Linux

After building custom OpenCV and before building 360_stitcher, make symbolic link from CUDA lib directory (most likely /usr/local/cuda/lib64) to /usr:

```
sudo ln -s /usr/local/cuda/lib64 /usr
```

If the stitcher complains about not finding shared libraries (f.ex. libopencv_core), run ldconfig

```
sudo ldconfig
```

After that the build should work:

```
cd 360_stitcher
make
cd build
./stitcher
```

### Flags
- string base: Name of the folder where to search for input videos
- int skip_frames: Amount frames to skip from start of all video streams
- int offsets: Amount frames to skip from start of each video stream individually. (Use calibrate.py to get required offsets)
- bool wrapAround: If true input streams wrap around 360 degrees
- bool recalibrate: If true recalibrate the CPW mesh periodically.
- bool enable_local: If true enable content preserving warping
- bool save_video: If true save the stitched video to a file
- bool show_out: If true use imshow
- bool use_stream: If true use network streams as input. If false use video streams
- bool clear_buffers: If true, clear all frame buffers between threads before pushing more data. Not really needed, only slows the program down.
- bool debug_stream: If true show the input from network streams directly
- bool use_surf: If true use SURF to find features. If false use ORB instead. ORB is faster but SURF gives way better features
- bool send_results: If true, send the result to the player on PLAYER_ADDRESS, using PLAYER_TCP_PORT (defaults are localhost, 55555)
- bool send_height_info: If true and send_results is true, the height of the result image is sent to the player first. If send_results is false, this is ignored.
- bool keep_aspect_ratio: If true, keep the aspect ratio of the image when resizing to the final size. Otherwise stretch the image to fit the output size.
- bool add_black_bars: If true and keep_aspect_ratio is true, black bars are added above and below the image to fit the output size. If keep_aspect_ratio is false, this is ignored.
- int NUM_IMAGES: Amount of video or network streams to stitch together
- unsigned char clientAddrStart: Smallest value of the capture board IP's last octet. The boards have consecutive addresses (by default 10.21.25.41, ... , 10.21.25.46, i.e. clientAddrStart = 41). This is used so that every board is in the same position every time, and the video streams are always in the correct order.
- int OUTPUT_WIDTH, OUTPUT_HEIGHT: Dimensions of the output image. If keep_aspect_ratio is true and add_black_bars is false, the final output height may be smaller than this. Only the width of the output is guaranteed to always have this value.
- unsigned int RESULTS_MAX_SIZE: Max size of the results queue, 0 means no limit. This is meant to avoid memory overflow if the frames are produced faster than consumed. However, checking the size every time slows the program down, so 0 is the fastest setting.
- int RECALIB_DEL: Amount of frames inbetween recalibration
- double WORK_MEGAPIX: Size of images for feature finding and mesh warping
- double SEAM_MEGAPIX: Size of used seam masks
- double COMPOSE_MEGAPIX: Size of images when stitching
- float MATCH_CONF: Minimum confidence when matching SURF features
- int HESS_THRESH, NOCTAVES, NOCTAVES_LAYERS: Parameters for SURF feature finding
- int MAX_FEATURES_PER_IMAGE: Limit of features used per pair of images when optimizing CPW mesh
- bool VISUALIZE_MATCHES: Visualize feature matches in image pairs. It is a bit broken for images wrapping around border.
- bool VISUALIZE_WARPED: Visualize the CPW mesh
- int N, M: Size of the used mesh by CPW
- float ALPHAS[3]: Weights for different cost functions in CPW
- int GLOBAL_DIST: Max distance of a feature in global cost function in CPW
