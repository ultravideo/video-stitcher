# Live Stitching
## Building
### Requirements
- Custom OpenCV (Included in the directory)
- Custom OpenCV has some changes in stitching module
- Eigen
- CUDA

#### OpenCV
- Make visual studio project with cmake
- set WITH\_CUDA flag true and CUDA\_HOST\_COMPILER to the path where your visual c compiler is. (_VSINSTALLFOLDER_/VC/Tools/MSVC/_VERSIONNUMBER_/bin/Hostx64/x64/cl.exe)
- In visual studio build the project "INSTALL"

#### CUDA
- Install CUDA https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html
- copy the cuda lib and include folders to $(ProjectDir)

#### Eigen
- Download Eigen to $(ProjectDir)include
- http://eigen.tuxfamily.org/index.php?title=Main_Page
