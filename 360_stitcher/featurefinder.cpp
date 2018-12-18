#include "featurefinder.h"
#include "defs.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/calib3d.hpp"

using namespace cv;
using namespace cv::detail;
using std::vector;

void featurefinder::findFeatures(vector<Mat> &full_img, vector<ImageFeatures> &features,
                  const double &work_scale) {
    Ptr<cuda::ORB> d_orb = cuda::ORB::create(2500, 1.2f, 8);
    Ptr<SurfFeaturesFinderGpu> surf = makePtr<SurfFeaturesFinderGpu>(HESS_THRESH, NOCTAVES, NOCTAVESLAYERS);
    Mat image;
    cuda::GpuMat gpu_img;
    cuda::GpuMat descriptors;
    // Read images from file and resize if necessary
    for (int i = 0; i < NUM_IMAGES; i++) {
        // Negative value means processing images in the original size
        if (WORK_MEGAPIX < 0 || work_scale < 0)
        {
            image = full_img[i];
        }
        // Else downscale images to speed up the process
        else
        {
            cv::resize(full_img[i], image, Size(), work_scale, work_scale);
        }

        if (use_surf) {
            // Find features with SURF feature finder
            (*surf)(image, features[i]);
        }
        else
        {
            // Find features with ORB feature finder
            gpu_img.upload(image);
            cuda::cvtColor(gpu_img, gpu_img, CV_BGR2GRAY);
            features[i].img_size = image.size();
            d_orb->detectAndCompute(gpu_img, noArray(), features[i].keypoints, descriptors);
            descriptors.download(features[i].descriptors);
        }
        features[i].img_idx = i;
    }
}

void featurefinder::matchFeatures(vector<ImageFeatures> &features, vector<MatchesInfo> &pairwise_matches) {
    // Use different way of matching features for SURF and ORB
    if (use_surf) {
        Ptr<FeaturesMatcher> fm = makePtr<BestOf2NearestMatcher>(true, MATCH_CONF);
        (*fm)(features, pairwise_matches);
        return;
    }
    Ptr<DescriptorMatcher> dm = DescriptorMatcher::create("BruteForce-Hamming");

    // Match features
    for (int i = 0; i < pairwise_matches.size(); ++i) {
        int idx1 = (i - 1 == -1) ? i : i;
        int idx2 = (i - 1 == -1) ? NUM_IMAGES - 1 : i-1;
        pairwise_matches[i].src_img_idx = idx1;
        pairwise_matches[i].dst_img_idx = idx2;
        ImageFeatures f1 = features[idx1];
        ImageFeatures f2 = features[idx2];
        vector<vector<DMatch>> matches;
        dm->knnMatch(f1.descriptors, f2.descriptors, matches, 2);
        for (int j = 0; j < matches.size(); ++j) {
            if (matches[j][0].distance < 0.7 * matches[j][1].distance) {
                pairwise_matches[i].matches.push_back(matches[j][0]);
            }
        }
        Mat src_points(1, static_cast<int>(pairwise_matches[i].matches.size()), CV_32FC2);
        Mat dst_points(1, static_cast<int>(pairwise_matches[i].matches.size()), CV_32FC2);
        if (pairwise_matches[i].matches.size()) {
            for (int j = 0; j < pairwise_matches[i].matches.size(); ++j) {
                const DMatch& m = pairwise_matches[i].matches[j];

                Point2f p = f1.keypoints[m.queryIdx].pt;
                p.x -= f1.img_size.width * 0.5f;
                p.y -= f1.img_size.height * 0.5f;
                src_points.at<Point2f>(0, static_cast<int>(j)) = p;

                p = f2.keypoints[m.trainIdx].pt;
                p.x -= f2.img_size.width * 0.5f;
                p.y -= f2.img_size.height * 0.5f;
                dst_points.at<Point2f>(0, static_cast<int>(j)) = p;
            }

            // Find pair-wise motion
            pairwise_matches[i].H = findHomography(src_points, dst_points, pairwise_matches[i].inliers_mask, RANSAC);

            // Find number of inliers
            pairwise_matches[i].num_inliers = 0;
            for (size_t idx = 0; idx < pairwise_matches[i].inliers_mask.size(); ++idx)
                if (pairwise_matches[i].inliers_mask[idx])
                    pairwise_matches[i].num_inliers++;

            // Confidence calculation copied from opencv feature matching code
            // These coeffs are from paper M. Brown and D. Lowe. "Automatic Panoramic Image Stitching
            // using Invariant Features"
            pairwise_matches[i].confidence = pairwise_matches[i].num_inliers /
                                             (8 + 0.3 * pairwise_matches[i].matches.size());

            // Set zero confidence to remove matches between too close images, as they don't provide
            // additional information anyway. The threshold was set experimentally.
            pairwise_matches[i].confidence = pairwise_matches[i].confidence > 3. ?
                                             0. : pairwise_matches[i].confidence;
        }
        pairwise_matches[i].confidence = 1;
    }
}



