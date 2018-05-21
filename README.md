# Live Stitching
## Building
### Requirements
- Custom OpenCV (Included in the directory)
- Custom OpenCV has some changes in stitching module
- Eigen
- CUDA

#### OpenCV
- Make visual studio project with cmake to $(ProjectDir)../opencv_build
- set WITH\_CUDA flag true and CUDA\_HOST\_COMPILER to the path where your visual c compiler is. (_VSINSTALLFOLDER_/VC/Tools/MSVC/_VERSIONNUMBER_/bin/Hostx64/x64/cl.exe)
- In visual studio build the project "INSTALL"

#### CUDA
- Install CUDA https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html
- copy the cuda lib and include folders to $(ProjectDir)

#### Eigen
- Download Eigen to $(ProjectDir)include
- http://eigen.tuxfamily.org/index.php?title=Main_Page

### Flags
string base: Name of the folder where to search for input videos
int skip_frames: Amount frames to skip from start of all video streams
int offsets: Amount frames to skip from start of each video stream individually
bool wrapAround: If true input streams wrap around 360 degrees
bool recalibrate: If true recalibrate the stitch periodically (Currently not used)
bool enable_local: If true enable content preserving warping
bool save_video: If true save the stitched video to a file
bool use_stream: If true use network streams as input. If false use video streams
bool debug_stream: If true show the input from network streams directly
bool use_surf: If true use SURF to find features. If false use ORB instead. ORB is faster but SURF gives way better features
int NUM_IMAGES: Amount of video or network streams to stitch together
int RECALIB_DEL: Amount of frames inbetween recalibration
double WORK_MEGAPIX: Size of images for feature finding and mesh warping
double SEAM_MEGAPIX: Size of used seam masks
double COMPOSE_MEGAPIX: Size of images when stitching
float MATCH_CONF: Minimum confidence when matching SURF features
int HESS_THRESH, NOCTAVES, NOCTAVES_LAYERS: Parameters for SURF feature finding