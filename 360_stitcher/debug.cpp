#include "debug.h"

bool showMat(const char *name, const cv::Mat &m)
{
	if (m.empty()) {
		std::cout << "mat " << name << " is empty!" << std::endl;
		return false;
	}
	cv::imshow(name, m);
	cv::waitKey(0);
	return true;
}

bool showMat(const char *name, const cv::UMat &m)
{
	if (m.empty()) {
		std::cout << "mat " << name << " is empty!" << std::endl;
		return false;
	}
	cv::imshow(name, m);
	cv::waitKey(0);
	return true;
}

bool showMat(const char *name, const cv::cuda::GpuMat &m)
{
    if (m.empty()) {
		std::cout << "mat " << name << " is empty!" << std::endl;
		return false;
    }
	cv::Mat temp(m);
	bool not_empty = showMat(name, temp);
	temp.release();
	return not_empty;
}

void showMats(const char *name, const std::vector<cv::Mat> &mats)
{
	for (int i = 0; i < mats.size(); i++) {
		if (mats[i].empty()) {
			std::cout << i << " mat " << name << " is empty!" << std::endl;
			continue;
		}
		cv::imshow(std::to_string(i) + std::string(" ") + std::string(name), mats[i]);
	}
	cv::waitKey(0);
}

void showMats(const char *name, const std::vector<cv::UMat> &mats)
{
	for (int i = 0; i < mats.size(); i++) {
		if (mats[i].empty()) {
			std::cout << i << " mat " << name << " is empty!" << std::endl;
			continue;
		}
		cv::imshow(std::to_string(i) + std::string(" ") + std::string(name), mats[i]);
	}
	cv::waitKey(0);
}

void showMats(const char *name, const std::vector<cv::cuda::GpuMat> &mats)
{
	for (int i = 0; i < mats.size(); i++) {
		if (mats[i].empty()) {
			std::cout << i << " mat " << name << " is empty!" << std::endl;
			continue;
		}
		cv::Mat m(mats[i]);
		cv::imshow(std::to_string(i) + std::string(" ") + std::string(name), m);
		m.release();
	}
	cv::waitKey(0);
}
