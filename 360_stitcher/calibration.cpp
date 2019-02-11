#include "calibration.h"
#include "meshwarper.h"
#include <algorithm>
#include <memory>

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/cudawarping.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/cudaarithm.hpp"

#include "Eigen/IterativeLinearSolvers"
#include "Eigen/SparseCholesky"

using namespace cv;
using namespace cv::detail;
using std::vector;
 

bool calibrateCameras(vector<CameraParams> &cameras, const cv::Size full_img_size,
                      const double work_scale) {
    cameras = vector<CameraParams>(NUM_IMAGES);
	double fov = 90.0*PI/180.0;
	double focal_tmp = 1.0/(tan(fov*0.5));

    for (int i = 0; i < cameras.size(); ++i) {
        float rot = static_cast<float>(2.0 * PI * static_cast<float>(i) / NUM_IMAGES); //kameroiden paikat ympyrän kehällä.

		Mat Rx = (Mat_<float>(3, 3) <<
			1, 0, 0,
			0, cos(0), -sin(0),
			0, sin(0), cos(0)); //Only for example as we do not have this rotation the matrix becomes an identity matrix --> no need

		Mat Ry = (Mat_<float>(3, 3) <<
			cos(rot), 0, sin(rot),
			0, 1, 0,
			-sin(rot), 0, cos(rot)); //The cameras in the grid are rotated around y-axis. No other rotation is present.

		Mat Rz = (Mat_<float>(3, 3) <<
			cos(0), -sin(0), 0,
			sin(0), cos(0), 0,
			0, 0, 1); //Only for example as we do not have this rotation the matrix becomes an identity matrix --> no need

		Mat rotMat = Rz * Ry * Rx; //the order to combine euler angle matrixes to rotation matrix is ZYX!

        cameras[i].R = rotMat;
		cameras[i].ppx = (full_img_size.width * work_scale) / 2.0;
        cameras[i].ppy = (full_img_size.height * work_scale) / 2.0; // principal point y
		cameras[i].aspect = 16.0 / 9.0; //as it is known the cameras have 1080p resolution, the aspect ratio is known to be 16:9
		//in 1080p resolution with medium fov (127 degrees) the focal lengt is 21mm.
		// with fov 170 degrees the focal lenth is 14mm and with fov 90 degrees the focal length is 28mm 
		// This information from gopro cameras is found from internet may be different for the model used in the grid!!!
		
		cameras[i].focal = focal_tmp * cameras[i].ppx;
		std::cout << "Focal " << i << ": " << cameras[i].focal << std::endl;
    }

    return true;
}


// Precalculates warp maps for images so only precalculated maps are used to warp online
void warpImages(vector<Mat> full_img, Size full_img_size, vector<CameraParams> cameras,
                Ptr<Blender> blender, Ptr<ExposureCompensator> compensator, double work_scale,
                double seam_scale, double seam_work_aspect, vector<cuda::GpuMat> &x_maps,
                vector<cuda::GpuMat> &y_maps, double &compose_scale, float &warped_image_scale,
                float &blend_width) {
    // STEP 3: warping images // ------------------------------------------------------------------------------------------------
    cuda::Stream stream;
    vector<cuda::GpuMat> gpu_imgs(NUM_IMAGES);
    vector<cuda::GpuMat> gpu_seam_imgs(NUM_IMAGES);
    vector<cuda::GpuMat> gpu_seam_imgs_warped(NUM_IMAGES);
    vector<cuda::GpuMat> gpu_masks(NUM_IMAGES);
    vector<cuda::GpuMat> gpu_masks_warped(NUM_IMAGES);
    vector<UMat> masks_warped(NUM_IMAGES);
    vector<UMat> images_warped(NUM_IMAGES);
    vector<UMat> masks(NUM_IMAGES);
    vector<Size> sizes(NUM_IMAGES);
    vector<Mat> images(NUM_IMAGES);

    // Create masks for warping
#pragma omp parallel for
    for (int i = 0; i < NUM_IMAGES; i++)
    {
        gpu_imgs[i].upload(full_img[i], stream);
        cuda::resize(gpu_imgs[i], gpu_seam_imgs[i], Size(), seam_scale, seam_scale, 1, stream);
        gpu_masks[i].create(gpu_seam_imgs[i].size(), CV_8U);
        gpu_masks[i].setTo(Scalar::all(255), stream);
    }

    Ptr<WarperCreator> warper_creator = makePtr<cv::CylindricalWarperGpu>();
    Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale * seam_work_aspect));
    cv::detail::CylindricalWarperGpu* gpu_warper = dynamic_cast<cv::detail::CylindricalWarperGpu*>(warper.get());

    vector<UMat> images_warped_f(NUM_IMAGES);
    vector<Point> corners(NUM_IMAGES);

    // Warp image
    for (int i = 0; i < NUM_IMAGES; i++)
    {
        Mat_<float> K;
        cameras[i].K().convertTo(K, CV_32F);
        float swa = (float)seam_work_aspect;
        K(0, 0) *= swa;
        K(0, 2) *= swa;
        K(1, 1) *= swa;
        K(1, 2) *= swa;

        corners[i] = gpu_warper->warp(gpu_seam_imgs[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, gpu_seam_imgs_warped[i]);
        sizes[i] = gpu_seam_imgs_warped[i].size();
        gpu_seam_imgs_warped[i].download(images_warped[i], stream);

        gpu_warper->warp(gpu_masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, gpu_masks_warped[i]);
        gpu_masks_warped[i].download(masks_warped[i]);

        gpu_seam_imgs_warped[i].convertTo(gpu_seam_imgs_warped[i], CV_32F, stream);
        gpu_seam_imgs_warped[i].download(images_warped_f[i], stream);
    }

    // STEP 4: compensating exposure and finding seams // -----------------------------------------------------------------------

    compensator->feed(corners, images_warped, masks_warped);
    GainCompensator* gain_comp = dynamic_cast<GainCompensator*>(compensator.get());

    Ptr<SeamFinder> seam_finder = makePtr<VoronoiSeamFinder>();
    seam_finder->find(images_warped_f, corners, masks_warped);

    // Release unused memory
    images.clear();
    images_warped.clear();
    images_warped_f.clear();
    masks.clear();

    //  STEP 5: composing panorama // -------------------------------------------------------------------------------------------
    double compose_work_aspect = 1;

    // Negative value means compose in original resolution
    if (COMPOSE_MEGAPIX > 0)
    {
        compose_scale = min(1.0, sqrt(COMPOSE_MEGAPIX * 1e6 / full_img[0].size().area()));
    }

    // Compute relative scale
    compose_work_aspect = compose_scale / work_scale;

    // Update warped image scale
    warper = warper_creator->create(warped_image_scale * static_cast<float>(compose_work_aspect));
    gpu_warper = dynamic_cast<cv::detail::CylindricalWarperGpu*>(warper.get());

    Size sz = full_img_size;

    if (std::abs(compose_scale - 1) > 1e-1)
    {
        sz.width = cvRound(full_img_size.width * compose_scale);
        sz.height = cvRound(full_img_size.height * compose_scale);
    }

    // Update corners and sizes
    for (int i = 0; i < NUM_IMAGES; i++)
    {
        // Update intrinsics
        cameras[i].focal *= compose_work_aspect;
        cameras[i].ppx *= compose_work_aspect;
        cameras[i].ppy *= compose_work_aspect;

        // Update corners and sizes
        Mat K;
        cameras[i].K().convertTo(K, CV_32F);
        Rect roi = warper->warpRoi(sz, K, cameras[i].R);
        corners[i] = roi.tl();
        sizes[i] = roi.size();
    }

    Size dst_sz = resultRoi(corners, sizes).size();
    blend_width = sqrt(static_cast<float>(dst_sz.area())) * BLEND_STRENGTH / 100.f;

    if (blend_width < 1.f)
    {
        blender = Blender::createDefault(Blender::NO, true);
    }
    else
    {
        MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(blender.get());
        mb->setNumBands(static_cast<int>(ceil(log(blend_width) / log(2.)) - 1.));
    }

    blender->prepare(corners, sizes);


    MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(blender.get());
    int64 start = getTickCount();
    Size img_size;
    cuda::GpuMat gpu_img;
    cuda::GpuMat gpu_mask;
    cuda::GpuMat gpu_mask_warped;
    cuda::GpuMat gpu_seam_mask;
    Mat mask_warped;
    // Dilation filter for local warping, without dilation local warping will cause black borders between seams
    // Better solution might be needed in the future
    Ptr<cuda::Filter> dilation_filter = cuda::createMorphologyFilter(MORPH_DILATE, CV_8U, Mat(), {-1,-1}, 1);
    for (int img_idx = 0; img_idx < NUM_IMAGES; img_idx++)
    {
        gpu_mask_warped.release();
        img_size = full_img[img_idx].size();
        img_size = Size((int)(img_size.width * compose_scale), (int)(img_size.height * compose_scale));
        full_img[img_idx].release();

        Mat K;
        cameras[img_idx].K().convertTo(K, CV_32F);

        // Create warping map for online process
        gpu_warper->buildMaps(img_size, K, cameras[img_idx].R, x_maps[img_idx], y_maps[img_idx]);

        // Warp the current image mask
        gpu_mask.create(img_size, CV_8U);
        gpu_mask.setTo(Scalar::all(255));

        gpu_warper->warp(gpu_mask, K, cameras[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, gpu_mask_warped);

        gpu_masks_warped[img_idx].upload(masks_warped[img_idx]);

        if (enable_local)
            dilation_filter->apply(gpu_masks_warped[img_idx], gpu_seam_mask);
        else
            gpu_seam_mask = gpu_masks_warped[img_idx];

        cuda::resize(gpu_seam_mask, gpu_seam_mask, gpu_mask_warped.size(), 0.0, 0.0, 1);
        cuda::bitwise_and(gpu_seam_mask, gpu_mask_warped, gpu_mask_warped, noArray());

        // Calculate mask pyramid for online process
        mb->init_gpu(gpu_img, gpu_mask_warped, corners[img_idx]);
    }

    Mat result;
    Mat result_mask;

    blender->blend(result, result_mask);

}

// Takes in maps for 3D remapping, compose scale for sizing final panorama, blender and image size.
// Returns true if all the phases of calibration are successful.
bool stitch_calib(LockableVector<Mat> &full_img, vector<CameraParams> &cameras, vector<cuda::GpuMat> &x_maps,
                  vector<cuda::GpuMat> &y_maps, LockableVector<cuda::GpuMat> &x_mesh, LockableVector<cuda::GpuMat> &y_mesh,
                  double &work_scale, double &seam_scale, double &seam_work_aspect, double &compose_scale,
                  Ptr<Blender> &blender, Ptr<ExposureCompensator> compensator, float &warped_image_scale,
                  float &blend_width, Size &full_img_size, std::shared_ptr<MeshWarper> &mw)
{
    // STEP 1: reading images, feature finding and matching // ------------------------------------------------------------------

    for (int i = 0; i < NUM_IMAGES; ++i) {
        //CAPTURES[i].read(full_img[i]);
        if (full_img[i].empty()) {
            LOGLN("Can't read frame from camera/file nro " << i << "...");
            return false;
        }
    }
    full_img_size = full_img[0].size();

	// Negative value means processing images in the original size
	if (WORK_MEGAPIX < 0)
	{
		work_scale = 1;
	}
	// Else downscale images to speed up the process
	else
	{
		work_scale = min(1.0, sqrt(WORK_MEGAPIX * 1e6 / full_img_size.area()));
	}
	// Calculate scale for downscaling done in seam finding process
	seam_scale = min(1.0, sqrt(SEAM_MEAGPIX * 1e6 / full_img_size.area()));
	seam_work_aspect = seam_scale / work_scale;


	vector<ImageFeatures> features(NUM_IMAGES);
	vector<MatchesInfo> pairwise_matches(NUM_IMAGES - 1 + (int)wrapAround);

    // STEP 2: estimating homographies // ---------------------------------------------------------------------------------------
    if (!calibrateCameras(cameras, full_img_size, work_scale)) {
        return false;
    }
	warped_image_scale = static_cast<float>(cameras[0].focal);
	std::cout << "Warped image scale: " << warped_image_scale << std::endl;


    warpImages(full_img, full_img_size, cameras, blender, compensator,
               work_scale, seam_scale, seam_work_aspect, x_maps, y_maps, compose_scale,
               warped_image_scale, blend_width);

    if (enable_local) {
        mw.reset(new MeshWarper(full_img.size(), MESH_WIDTH, MESH_HEIGHT, cameras[0].focal, compose_scale, work_scale));
        mw->calibrateMeshWarp(full_img, features, pairwise_matches, x_mesh, y_mesh,
                          x_maps, y_maps);
        MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(blender.get());
        cuda::Stream stream;
        for (int i = 0; i < full_img.size(); ++i) {
            //Disabled, causes black seams
            //mb->update_mask(i, x_mesh[i], y_mesh[i], stream);
        }
    }
    return true;
}
