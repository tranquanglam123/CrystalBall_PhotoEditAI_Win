#pragma once

#include <vector>
#include <string>
#ifndef WIN32
#include<android/log.h>
#endif
#include "net.h"
#include "Matting.h"
#include "vl/globalmatting.h"
#include "vl/fastguidedfilter.h"

using namespace std;
using namespace cv;

const cv::String PATH_TEST = "C:/Users/kki/Documents/DB/test";

namespace CrystalBall {
	class MagicEngine {
		public:
			MagicEngine();
			MagicEngine(const string& model_path);
			~MagicEngine();

			void init(const string& model_path);
			void trimapMask(ncnn::Mat& inputImage, cv::Mat& outImage);

			Mat getMaskImage(Mat srcImage, Mat trimapImage);
			void preprocTrimapImage(Mat trimapImage);
			Mat preprocTrimapImage(Mat& trimapImage, int cxNor, int cyNor);

			cv::Mat mattingImageColorLineLaplace2(cv::Mat& srcImage, cv::Mat& maskImage);
			cv::Mat mattingImageColorLineLaplace(cv::Mat& srcImage, cv::Mat& trimapImage);
			cv::Mat mattingImageInformationFlowDouble(cv::Mat& srcImage, cv::Mat& trimapImage);
			cv::Mat mattingImageInformationFlowFloat(cv::Mat& srcImage, cv::Mat& trimapImage);
			cv::Mat mattingImageGuideFilter(cv::Mat& srcImage, cv::Mat& trimapImage);
			cv::Mat mattingImageInformationFlowFloatGuideFilter(cv::Mat& srcImage, cv::Mat& trimapImage);

			cv::Mat mattingImageForEffectImage(cv::Mat& srcImage, cv::Mat& trimapImage);

			//background : blur (seamlessClone)
			cv::Mat effectImageBackBlur(cv::Mat& srcImage, cv::Mat& trimapImage, bool bGrayBackgound);

			//background : gray & blur (infoflow matting)
			cv::Mat effectImageColorOnYou(cv::Mat& srcImage, cv::Mat& trimapImage);

			//romantic
			cv::Mat effectImageRomantic(cv::Mat& frmImage, cv::Mat& srcImage1, cv::Mat& srcImage2, Point2f  pt1[4], Point2f  pt2[4]);

			//face beauty
			cv::Mat effectImageFaceBeauty(cv::Mat& srcImage, int smoothParam);

			//test for windows
			void ShowImage(const String& winname, InputArray mat);
			void SaveImage(const String& winname, InputArray mat);

		private:
			void Visualization(cv::Mat prediction_map, cv::Mat& output_image);
			void GetMonoImage(unsigned char *pGrayImage, int cx, int cy);
			void Dilation(cv::Mat src, cv::Mat& dilation_dst, int dilation_size = 3);
			void Erosion(cv::Mat src, cv::Mat& erosion_dst, int erosion_size = 3);
			void GetSegmentImage(unsigned char* pGrayImage, int cx, int cy, unsigned char* pSegmentForeground, unsigned char* pSegmentBackground);
			void getActualSizeofBitmap(unsigned char *pAlphaImage, int wid, int hei, int &x_pos, int &y_pos, int &pic_wid, int &pic_hei);
			void makePNG(unsigned char*pData, int wid, int hei);
			void applyAlpha(unsigned char* pData, unsigned char* pAlpha, unsigned char* pOut, int wid, int hei);

			string LUT_file;
			ncnn::Net m_netSegmentation;
	};
} //namespace CrystalBall