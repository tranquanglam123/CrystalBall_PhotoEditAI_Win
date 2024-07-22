#include "MagicEngine.h"

// using namespace cv;

namespace cv {
	namespace alphamat {
		//! @addtogroup alphamat
		//! @{

		/**
		* @brief Compute alpha matte of an object in an image
		* @param image Input RGB image
		* @param tmap Input greyscale trimap image
		* @param result Output alpha matte image
		*
		* The function infoFlow performs alpha matting on a RGB image using a greyscale trimap image, and outputs a greyscale alpha matte image. The output alpha matte can be used to softly extract the foreground object from a background image. Examples can be found in the samples directory.
		*
		*/
		void infoFlowDouble(InputArray image, InputArray tmap, OutputArray result);
		void infoFlowFloat(InputArray image, InputArray tmap, OutputArray result);

		//! @}
	}
}  // namespace

namespace CrystalBall {
	MagicEngine::MagicEngine() {

	}

	MagicEngine::MagicEngine(const string &model_path) {
		init(model_path);
	}

	MagicEngine::~MagicEngine() {

	}

	void MagicEngine::init(const string &model_path) {
		std::string filepath(model_path);
		const string &model_file = filepath + "/seg.proto";
		const string &trained_file = filepath + "/seg.bin";
		LUT_file = filepath + "/LutFile.png";
		m_netSegmentation.load_param(model_file.c_str());
		m_netSegmentation.load_model(trained_file.c_str());
	}

	void MagicEngine::trimapMask(ncnn::Mat &inputImage, cv::Mat &outImage) {
		ncnn::Extractor ex = m_netSegmentation.create_extractor();
		ex.set_num_threads(1);
		ex.set_light_mode(true);

		const float mean_vals[3] = {128, 128, 128};
		const float norm_vals[3] = {0.0078125, 0.0078125, 0.0078125};

		inputImage.substract_mean_normalize(mean_vals, norm_vals);
		ex.input("data", inputImage);

		ncnn::Mat outData;
		int nRet = ex.extract("deconv6_0_0", outData);
		cv::Mat class_each_row(3, 224 * 224, CV_32FC1, outData.data);
		class_each_row = class_each_row.t();

		cv::Point maxId;    // point [x,y] values for index of max
		double maxValue;    // the holy max value itself
		cv::Mat prediction_map(224, 224, CV_8UC1);

		for (int i = 0; i < class_each_row.rows; i++) {
			cv::minMaxLoc(class_each_row.row(i), 0, &maxValue, 0, &maxId);
			prediction_map.at<uchar>(i) = maxId.x;
		}

		cv::Mat output_image;
		Visualization(prediction_map, output_image);
		cv::cvtColor(output_image, output_image, COLOR_BGR2GRAY);
		outImage = output_image;
	}

	void MagicEngine::Visualization(cv::Mat prediction_map, cv::Mat &output_image) {
		cv::cvtColor(prediction_map.clone(), prediction_map, cv::COLOR_GRAY2BGR);
		cv::Mat label_colours = cv::imread(LUT_file, 1);
		// 	cv::cvtColor(label_colours, label_colours, cv::COLOR_RGB2BGR);
		cv::LUT(prediction_map, label_colours, output_image);
	}

	void MagicEngine::Dilation(cv::Mat src, cv::Mat &dilation_dst, int dilation_size) {
		cv::Mat element = getStructuringElement(MORPH_ELLIPSE, cv::Size(2 * dilation_size + 1,
																		2 * dilation_size + 1),
												cv::Point(dilation_size, dilation_size));
		cv::dilate(src, dilation_dst, element);
		//imshow("Dilation Demo", dilation_dst);
	}

	/**  @function Erosion  */
	void MagicEngine::Erosion(cv::Mat src, cv::Mat &erosion_dst, int erosion_size) {
		cv::Mat element = getStructuringElement(MORPH_ELLIPSE, cv::Size(2 * erosion_size + 1,
																		2 * erosion_size + 1),
												cv::Point(erosion_size, erosion_size));
		cv::erode(src, erosion_dst, element);
		//imshow("Erosion Demo", erosion_dst);
	}

	void MagicEngine::GetSegmentImage(unsigned char *pGrayImage, int cx, int cy,
									  unsigned char *pSegmentForeground,
									  unsigned char *pSegmentBackground) {
		for (int i = 0; i < cx * cy; i++) {
			pSegmentBackground[i] = 0;
			pSegmentForeground[i] = 0;

			if (pGrayImage[i] == 128) {
				pSegmentBackground[i] = 128;
			} else if (pGrayImage[i] == 255) {
				pSegmentForeground[i] = 255;
			}
		}
	}

	void MagicEngine::GetMonoImage(unsigned char *pGrayImage, int cx, int cy) {
		for (int i = 0; i < cx * cy; i++) {
			if (pGrayImage[i] >= 128)
				pGrayImage[i] = 255;
			else
				pGrayImage[i] = 0;

		}
	}

	void MagicEngine::makePNG(unsigned char *pData, int wid, int hei) {
		for (int i = 0; i < wid * hei; i++) {
			float alpha = (float) (*(pData + i * 4 + 3)) * (1.f / 255.f);
			*(pData + i * 4 + 2) = (float) (*(pData + i * 4 + 2)) * alpha;
			*(pData + i * 4 + 1) = (float) (*(pData + i * 4 + 1)) * alpha;
			*(pData + i * 4 + 0) = (float) (*(pData + i * 4 + 0)) * alpha;
//		if(*(pData+i*4+3) ==0)
//		{
//			*(pData+i*4+2)=*(pData+i*4+1)=*(pData+i*4+0)=0;
//		}
		}
	}

	void MagicEngine::applyAlpha(unsigned char *pData, unsigned char *pAlpha, unsigned char *pOut,
								 int wid, int hei) {
		for (int i = 0; i < wid * hei; i++) {
			float alpha = pAlpha[i] * (1.f / 255.f);
			*(pOut + i * 3 + 2) = (float) (*(pData + i * 3 + 2)) * alpha;
			*(pOut + i * 3 + 1) = (float) (*(pData + i * 3 + 1)) * alpha;
			*(pOut + i * 3 + 0) = (float) (*(pData + i * 3 + 0)) * alpha;
		}
	}

	void
	MagicEngine::getActualSizeofBitmap(unsigned char *pAlphaImage, int wid, int hei, int &x_pos,
									   int &y_pos, int &pic_wid, int &pic_hei) {
		int min_x = wid;
		int min_y = hei;
		int max_x = 0;
		int max_y = 0;

		for (int i = 0; i < hei; i++) {
			for (int j = 0; j < wid; j++) {
				int val = pAlphaImage[i * wid + j];
				if (val != 0) {
					if (i < min_y)
						min_y = i;

					if (j < min_x)
						min_x = j;

					break;
				}
			}
		}

		for (int i = hei - 1; i >= 0; i--) {
			for (int j = wid - 1; j >= 0; j--) {
				int val = pAlphaImage[i * wid + j];
				if (val != 0) {
					if (i > max_y)
						max_y = i;

					if (j > max_x)
						max_x = j;

					break;
				}
			}
		}
		x_pos = min_x;
		y_pos = min_y;
		pic_wid = max_x - min_x + 1;
		pic_hei = max_y - min_y + 1;

        if(pic_wid <1 || pic_hei < 1) {
            x_pos = 0, y_pos = 0;
            pic_wid = wid; pic_hei = hei;
        }
	}

	//Pre-process trimap
	void MagicEngine::preprocTrimapImage(Mat trimapImage) {
		int nRows = trimapImage.rows;
		int nCols = trimapImage.cols;

		for (int i = 0; i < nRows; ++i) {
			for (int j = 0; j < nCols; ++j) {
				uchar &pix = trimapImage.at<uchar>(i, j);
				if (pix <= 0.2f * 255) {
					pix = 0;
				} else if (pix >= 0.8f * 255) {
					pix = 255;
				} else {
					pix = 128;
				}

				trimapImage.at<uchar>(i, j) = pix;
			}
		}
	}

	//Pre-process trimap
	Mat MagicEngine::preprocTrimapImage(Mat &trimapImage, int cxNor, int cyNor) {
		Mat foreground = Mat::zeros(cv::Size(cxNor, cyNor), CV_8UC1);
		Mat background = Mat::zeros(cv::Size(cxNor, cyNor), CV_8UC1);
		Mat preprocTri = Mat::zeros(cv::Size(cxNor, cyNor), CV_8UC1);

		cv::resize(trimapImage, trimapImage, cv::Size(cxNor, cyNor));
// 		imwrite("D:/crystal/trimapImageNor.png", trimapImage);

		int nRows = trimapImage.rows;
		int nCols = trimapImage.cols;

		for (int i = 0; i < nRows; ++i) {
			for (int j = 0; j < nCols; ++j) {
				uchar &pix = trimapImage.at<uchar>(i, j);
				if (pix <= 0.2f * 255) {
					pix = 0;
				} else if (pix >= 0.8f * 255) {
					pix = 255;
					foreground.at<uchar>(i, j) = pix;
				} else {
					pix = 128;
					background.at<uchar>(i, j) = pix;
					foreground.at<uchar>(i, j) = pix;
				}

				trimapImage.at<uchar>(i, j) = pix;
			}
		}

// 		Dilation(foreground, foreground, 2);
// 		Erosion(foreground, foreground, 4);
// 		Dilation(background, background, 2);
// 		Erosion(background, background, 4);

		Erosion(foreground, foreground, 3);
		Erosion(background, background, 9);

		cv::bitwise_or(foreground, background, preprocTri);

		imwrite("D:/crystal/preprocTri.png", preprocTri);

		return preprocTri;
	}

	
	Mat MagicEngine::getMaskImage(Mat srcImage, Mat trimapImage) {
		Mat mask = trimapImage.clone();

		resize(mask, mask, srcImage.size());

		int nRows = mask.rows;
		int nCols = mask.cols;

		for (int i = 0; i < nRows; ++i) {
			for (int j = 0; j < nCols; ++j) {
				uchar &pix = mask.at<uchar>(i, j);
				if (pix <= 0.2f * 255) {
					pix = 0;
					mask.at<uchar>(i, j) = pix;
				}
				else if (pix >= 0.8f * 255) {
					pix = 255;
					mask.at<uchar>(i, j) = pix;
				}
				else {
					pix = 255;
					mask.at<uchar>(i, j) = pix;
				}

				if (i == nRows-1) {
					pix = 0;
					mask.at<uchar>(i, j) = pix;
				}
			}
		}

		Erosion(mask, mask, 9);

		imwrite("D:/crystal/mask.png", mask);

		return mask;
	}

	cv::Mat MagicEngine::mattingImageColorLineLaplace2(cv::Mat &srcImage, cv::Mat &maskImage) {
		int64 begin = cv::getTickCount();

		resize(maskImage, maskImage, cv::Size(srcImage.cols, srcImage.rows));

		maskImage = maskImage / 255 - 0.4;
		maskImage = maskImage * 255;
		cv::imshow("mask", maskImage);
		imwrite("D:/crystal/mask.png", maskImage);

		Mat img224;
		int wid = srcImage.cols;
		int hei = srcImage.rows;

		int cxNor = 360;
		int cyNor = 360;

		cv::resize(srcImage, srcImage, cv::Size(1 * cxNor, 1 * cyNor));
		cv::resize(srcImage, img224, cv::Size(1 * cxNor, 1 * cyNor));
		cv::resize(maskImage, maskImage, cv::Size(cxNor, cyNor));


		GetMonoImage(maskImage.data, maskImage.cols, maskImage.rows);
		imshow("mono", maskImage);
		imwrite("D:/crystal/mono.png", maskImage);

		Mat dilation, border, preproc;
		Dilation(maskImage, dilation, 7);
		imshow("Dilation", dilation);
		imwrite("D:/crystal/Dilation.png", dilation);

		cv::bitwise_or(maskImage, dilation, border);
		border = border / 2;
		cv::bitwise_or(maskImage, border, preproc);

		maskImage = preproc;

		imwrite("D:/crystal/imgNor.png", img224);
		imshow("mask", maskImage);
		imwrite("D:/crystal/preproc.png", maskImage);

		Mat SegmentDilation;
		Mat SegmentForeground(maskImage.rows, maskImage.cols, CV_8UC1);
		Mat SegmentBackground(maskImage.rows, maskImage.cols, CV_8UC1);

		GetSegmentImage(maskImage.data, maskImage.cols, maskImage.rows, SegmentForeground.data,
						SegmentBackground.data);

		Dilation(SegmentForeground, SegmentForeground, 1);
		Erosion(SegmentForeground, SegmentForeground, 1);
		Dilation(SegmentBackground, SegmentBackground, 3);
		Erosion(SegmentBackground, SegmentBackground, 1);
		cv::bitwise_or(SegmentForeground, SegmentBackground, SegmentDilation);

		cv::resize(SegmentDilation, SegmentDilation, cv::Size(1 * cxNor, 1 * cyNor), 0.0, 0.0,
				   INTER_NEAREST);
		imshow("SegmentDilation", SegmentDilation);
		imwrite("D:/crystal/SegmentDilation.png", SegmentDilation);

#if 0 //test
        cv::Mat tmpTri[3];
        tmpTri[0] = SegmentDilation.clone();
        tmpTri[1] = SegmentDilation.clone();
        tmpTri[2] = SegmentDilation.clone();

        Mat synImg;
        merge(tmpTri, 3, synImg);
        synImg = 0.5*img224 + 0.5*synImg;

        cv::imshow("synImg2", synImg);	imwrite("D:/crystal/synImg2.png", synImg);
#endif

		Mat trimap = labelExpansion(img224, SegmentDilation);
		imshow("trimap-le", trimap);
		imwrite("D:/crystal/trimap-le.png", trimap);
// 		imshow("trimap-l", trimap); imwrite("D:/crystal/trimap-l.png", trimap);

		float *pAlpha = (float *) malloc(img224.rows * img224.cols * sizeof(float));

		/////Mat Three layer///
		Matting(&img224, trimap, pAlpha, 0);
		Mat mask(img224.rows, img224.cols, CV_32FC1, pAlpha);

		// 		cv::cvtColor(mask, mask, COLOR_GRAY2BGR);
		mask = mask * 255;
		mask.convertTo(mask,
					   CV_8UC1); //mask = convertTypeAndSize(mask, CV_8UC1, cv::Size(img224.cols, img224.rows));

		Mat mattingImage = img224.clone();
		applyAlpha(img224.data, mask.data, mattingImage.data, img224.cols, img224.rows);
		imshow("mattingImage", mask);
		imwrite("D:/crystal/mattingImage.png", mattingImage);


		cv::resize(mask, mask, srcImage.size());

#if GUIDE_FILTER
        //Mat cut = srcImage.clone();
        Mat cut(srcImage.size(), CV_8UC3, 255);

        applyAlpha(srcImage.data, mask.data, cut.data, mask.cols, mask.rows);

        imwrite("D:/trimap.png", trimap);
        imwrite("D:/mask.png", mask);
        imwrite("D:/cut.png", cut);

        //////////////////////////////////////////////////////////////////////////

        int r = 8;
        double eps = 1e-4;
        eps *= 255 * 255;
        Mat processImg = fastGuidedFilter(srcImage, cut, r, eps, 4);

        imwrite("D:/processImg.png", processImg);

        Mat srcImageGray, processImgGray;
        cvtColor(srcImage, srcImageGray, COLOR_BGR2GRAY);
        cvtColor(processImg, processImgGray, COLOR_BGR2GRAY);
        mask = 255 * (srcImageGray / processImgGray);
        mask.convertTo(mask, CV_8UC1);

        //		cv::seamlessClone()
        //////////////////////////////////////////////////////////////////////////
#endif

		int x, y, pic_wid, pic_hei;
		getActualSizeofBitmap(mask.data, mask.cols, mask.rows, x, y, pic_wid, pic_hei);

		Mat newImage = srcImage.clone();

		cv::Mat tmp, split[4];
		cv::Rect actualRect(x, y, pic_wid, pic_hei);
		cv::split(newImage, split);
// 		tmp = split[0].clone();
// 		split[0] = split[2].clone();
// 		split[2] = tmp;
		split[3] = mask;
		cv::merge(split, 4, newImage);

		newImage = newImage(actualRect);
		Mat retImg;
		retImg.create(newImage.size(), CV_8UC4);
		newImage.copyTo(retImg);

		makePNG(retImg.data, retImg.cols, retImg.rows);

		free(pAlpha);


		double elapsed_secs = ((double) (getTickCount() - begin)) / getTickFrequency();

		FILE *fp = fopen("D:/time.txt", "ab+");
		fprintf(fp, "colorline-2: %f\n", elapsed_secs);
		fclose(fp);

		return retImg;
	}

	cv::Mat MagicEngine::mattingImageColorLineLaplace(cv::Mat &srcImage, cv::Mat &trimapImage) {
		int64 begin = cv::getTickCount();

		cv::resize(srcImage, srcImage, cv::Size(360, 360));

		trimapImage = preprocTrimapImage(trimapImage, 360, 360);
		cv::imshow("src", trimapImage);
		imwrite("D:/crystal/trimapImage.png", trimapImage);

#if 1 //test
		cv::Mat tmpTri[3];
		tmpTri[0] = trimapImage.clone();
		tmpTri[1] = trimapImage.clone();
		tmpTri[2] = trimapImage.clone();

		Mat synImg;
		merge(tmpTri, 3, synImg);
		synImg = 0.5 * srcImage + 0.5 * synImg;

		cv::imshow("synImg2", synImg);
		imwrite("D:/crystal/synImg2.png", synImg);
#endif

		Mat trimap = labelExpansion(srcImage, trimapImage);

		imshow("trimap-le", trimap);
		imwrite("D:/crystal/trimap-le.png", trimap);

		float *pAlpha = (float *) malloc(srcImage.rows * srcImage.cols * sizeof(float));

		/////Mat Three layer///
		Matting(&srcImage, trimap, pAlpha, 0);
		Mat mask(srcImage.rows, srcImage.cols, CV_32FC1, pAlpha);

		mask = mask * 255;
		mask.convertTo(mask, CV_8UC1);
		imshow("srcImage", srcImage);
		imwrite("D:/crystal/srcImage.png", srcImage);
		imshow("trimap", trimap);
		imwrite("D:/crystal/trimap.png", trimap);
		imshow("alpha", mask);
		imwrite("D:/crystal/alpha.png", mask);

		Mat mattingImage = srcImage.clone();
		applyAlpha(srcImage.data, mask.data, mattingImage.data, srcImage.cols, srcImage.rows);
		imshow("mattingImage", mask);
		imwrite("D:/crystal/mattingImageTrimap.png", mattingImage);

		cv::resize(mask, mask, srcImage.size());

#if GUIDE_FILTER
        //Mat cut = srcImage.clone();
        Mat cut(srcImage.size(), CV_8UC3, 255);

        applyAlpha(srcImage.data, mask.data, cut.data, mask.cols, mask.rows);

        imwrite("D:/trimap.png", trimap);
        imwrite("D:/mask.png", mask);
        imwrite("D:/cut.png", cut);

        //////////////////////////////////////////////////////////////////////////

        int r = 8;
        double eps = 1e-4;
        eps *= 255 * 255;
        Mat processImg = fastGuidedFilter(srcImage, cut, r, eps, 4);

        imwrite("D:/processImg.png", processImg);

        Mat srcImageGray, processImgGray;
        cvtColor(srcImage, srcImageGray, COLOR_BGR2GRAY);
        cvtColor(processImg, processImgGray, COLOR_BGR2GRAY);
        mask = 255*(srcImageGray / processImgGray);
        mask.convertTo(mask, CV_8UC1);

//		cv::seamlessClone()
        //////////////////////////////////////////////////////////////////////////
#endif

		int x, y, pic_wid, pic_hei;
		getActualSizeofBitmap(mask.data, mask.cols, mask.rows, x, y, pic_wid, pic_hei);

		Mat newImage = srcImage.clone();

		cv::Mat tmp, split[4];
		cv::Rect actualRect(x, y, pic_wid, pic_hei);
		cv::split(newImage, split);
// 		tmp = split[0].clone();
// 		split[0] = split[2].clone();
// 		split[2] = tmp;
		split[3] = mask;
		cv::merge(split, 4, newImage);

		newImage = newImage(actualRect);
		Mat retImg;
		retImg.create(newImage.size(), CV_8UC4);
		newImage.copyTo(retImg);

		makePNG(retImg.data, retImg.cols, retImg.rows);

		free(pAlpha);


		double elapsed_secs = ((double) (getTickCount() - begin)) / getTickFrequency();

		FILE *fp = fopen("D:/time.txt", "ab+");
		fprintf(fp, "colorline: %f\n", elapsed_secs);
		fclose(fp);

		return retImg;
	}

	cv::Mat
	MagicEngine::mattingImageInformationFlowDouble(cv::Mat &srcImage, cv::Mat &trimapImage) {
		cv::Mat matImage;

		cv::resize(srcImage, srcImage, cv::Size(360, 360));

		trimapImage = preprocTrimapImage(trimapImage, 360, 360);
		cv::imshow("src", trimapImage);
		imwrite("D:/crystal/trimapImage.png", trimapImage);

#if 1 //test
		cv::Mat tmpTri[3];
		tmpTri[0] = trimapImage.clone();
		tmpTri[1] = trimapImage.clone();
		tmpTri[2] = trimapImage.clone();

		Mat synImg;
		merge(tmpTri, 3, synImg);
		synImg = 0.5 * srcImage + 0.5 * synImg;

		cv::imshow("synImg2", synImg);
		imwrite("D:/crystal/synImg2.png", synImg);
#endif

		Mat trimap = trimapImage; // labelExpansion(srcImage, trimapImage);

		imshow("trimap-le", trimap);
		imwrite("D:/crystal/trimap-le.png", trimap);


		cv::alphamat::infoFlowDouble(srcImage, trimap, matImage);
		imshow("mattingImage", matImage);
		imwrite("D:/crystal/mattingImageInfo.png", matImage);


		int x, y, pic_wid, pic_hei;
		getActualSizeofBitmap(matImage.data, matImage.cols, matImage.rows, x, y, pic_wid, pic_hei);

		Mat newImage = srcImage.clone();

		cv::Mat tmp, split[4];
		cv::Rect actualRect(x, y, pic_wid, pic_hei);
		cv::split(newImage, split);
		// 		tmp = split[0].clone();
		// 		split[0] = split[2].clone();
		// 		split[2] = tmp;
		split[3] = matImage;
		cv::merge(split, 4, newImage);

		newImage = newImage(actualRect);
		Mat retImg;
		retImg.create(newImage.size(), CV_8UC4);
		newImage.copyTo(retImg);

		makePNG(retImg.data, retImg.cols, retImg.rows);

		return retImg;
	}

	cv::Mat MagicEngine::mattingImageInformationFlowFloat(cv::Mat &srcImage, cv::Mat &trimapImage) {
		cv::Mat oriImage = srcImage.clone();
		cv::Mat matImage;

		cv::resize(srcImage, srcImage, cv::Size(360, 360));

		trimapImage = preprocTrimapImage(trimapImage, 360, 360);
		cv::imshow("src", trimapImage);
		imwrite("D:/crystal/trimapImage.png", trimapImage);

#if 1 //test
		cv::Mat tmpTri[3];
		tmpTri[0] = trimapImage.clone();
		tmpTri[1] = trimapImage.clone();
		tmpTri[2] = trimapImage.clone();

		Mat synImg;
		merge(tmpTri, 3, synImg);
		synImg = 0.5 * srcImage + 0.5 * synImg;

		cv::imshow("synImg2", synImg);
		imwrite("D:/crystal/synImg2.png", synImg);
#endif

		Mat trimap = labelExpansion(srcImage, trimapImage);

		imshow("trimap-le", trimap);
		imwrite("D:/crystal/trimap-le.png", trimap);


		cv::alphamat::infoFlowFloat(srcImage, trimap, matImage);
		imshow("mattingImage", matImage);
		imwrite("D:/crystal/mattingImageInfo.png", matImage);


#if 1 //GUIDE_FILTER
		int64 begin = cv::getTickCount();

		Mat gray, cut;
		cvtColor(oriImage, gray, COLOR_BGR2GRAY);
		resize(matImage, matImage, Size(gray.cols, gray.rows));
		cut = gray.mul(matImage / 255.0);

		imshow("gray", gray);
		imwrite("D:/gray.png", gray);
		imshow("matI", cut);
		imwrite("D:/matI.png", cut);

		//////////////////////////////////////////////////////////////////////////
		int r = 1;
		double eps = 1e-4;
		eps *= 255 * 255;
		Mat filterImg = fastGuidedFilter(gray, cut, r, eps, 1);

		imshow("filterImg", filterImg);
		imwrite("D:/filterImg.png", filterImg);

		for (int i = 0; i < filterImg.rows; ++i) {
			for (int j = 0; j < filterImg.cols; ++j) {
				if (i == 96 && j == 91) {
					i = i;
				}
				int pix_filter = filterImg.at<uchar>(i, j);
				int pix_gray = gray.at<uchar>(i, j);
				int pix = (255.0f * pix_filter / pix_gray);
				if (pix > 255) {
					matImage.at<uchar>(i, j) = 255;
				} else {
					matImage.at<uchar>(i, j) = (uchar) pix;
				}
			}
		}
// 		matImage = (255.0 * filterImg / cut);
// 		matImage.convertTo(matImage, CV_8UC1);

		imshow("matting", matImage);
		imwrite("D:/matting.png", matImage);

		double elapsed_secs = ((double) (getTickCount() - begin)) / getTickFrequency();

		FILE *fp = fopen("D:/time.txt", "ab+");
		fprintf(fp, "guide filter: %f\n", elapsed_secs);
		fclose(fp);

		//		cv::seamlessClone()
		//////////////////////////////////////////////////////////////////////////
#endif

		int x, y, pic_wid, pic_hei;
		getActualSizeofBitmap(matImage.data, matImage.cols, matImage.rows, x, y, pic_wid, pic_hei);

		Mat newImage = srcImage.clone();

		cv::Mat tmp, split[4];
		cv::Rect actualRect(x, y, pic_wid, pic_hei);
		cv::split(newImage, split);
		// 		tmp = split[0].clone();
		// 		split[0] = split[2].clone();
		// 		split[2] = tmp;
		split[3] = matImage;
		cv::merge(split, 4, newImage);

		newImage = newImage(actualRect);
		Mat retImg;
		retImg.create(newImage.size(), CV_8UC4);
		newImage.copyTo(retImg);

		makePNG(retImg.data, retImg.cols, retImg.rows);

		return retImg;
	}

	cv::Mat MagicEngine::mattingImageGuideFilter(cv::Mat &srcImage, cv::Mat &trimapImage) {
		int64 begin = cv::getTickCount();

		cv::Mat matImage;

		trimapImage = preprocTrimapImage(trimapImage, srcImage.cols, srcImage.rows);
		cv::imshow("src", trimapImage);
		imwrite("D:/crystal/trimapImage.png", trimapImage);

#if 1 //test
		cv::Mat tmpTri[3];
		tmpTri[0] = trimapImage.clone();
		tmpTri[1] = trimapImage.clone();
		tmpTri[2] = trimapImage.clone();

		Mat synImg;
		merge(tmpTri, 3, synImg);
		synImg = 0.5 * srcImage + 0.5 * synImg;

		cv::imshow("synImg5", synImg);
		imwrite("D:/crystal/synImg5.png", synImg);
#endif

		Mat gray, cut;
		cvtColor(srcImage, gray, COLOR_BGR2GRAY);
		resize(trimapImage, matImage, Size(gray.cols, gray.rows));
		cut = gray.mul(matImage / 255.0);

		imshow("gray", gray);
		imwrite("D:/gray.png", gray);
		imshow("matI", cut);
		imwrite("D:/matI.png", cut);

		//////////////////////////////////////////////////////////////////////////
		int r = 1;
		double eps = 1e-4;
		eps *= 255 * 255;
		Mat filterImg = fastGuidedFilter(gray, cut, r, eps, 1);

		imshow("filterImg", filterImg);
		imwrite("D:/filterImg.png", filterImg);

		for (int i = 0; i < filterImg.rows; ++i) {
			for (int j = 0; j < filterImg.cols; ++j) {
				if (i == 96 && j == 91) {
					i = i;
				}
				int pix_filter = filterImg.at<uchar>(i, j);
				int pix_gray = gray.at<uchar>(i, j);
				int pix = (255.0f * pix_filter / pix_gray);
				if (pix > 255) {
					matImage.at<uchar>(i, j) = 255;
				} else {
					matImage.at<uchar>(i, j) = (uchar) pix;
				}
			}
		}
		// 		matImage = (255.0 * filterImg / cut);
		// 		matImage.convertTo(matImage, CV_8UC1);

		imshow("matting", matImage);
		imwrite("D:/matting.png", matImage);

		double elapsed_secs = ((double) (getTickCount() - begin)) / getTickFrequency();

		FILE *fp = fopen("D:/time.txt", "ab+");
		fprintf(fp, "guide filter: %f\n", elapsed_secs);
		fclose(fp);

		//		cv::seamlessClone()
		//////////////////////////////////////////////////////////////////////////

		return matImage;
	}

	cv::Mat MagicEngine::mattingImageInformationFlowFloatGuideFilter(cv::Mat &srcImage,
																	 cv::Mat &trimapImage) {

        int64 begin = cv::getTickCount();

		cv::Mat oriImage = srcImage.clone();
		cv::Mat matImage;

		cv::resize(srcImage, srcImage, cv::Size(trimapImage.rows, trimapImage.cols));

		trimapImage = preprocTrimapImage(trimapImage, trimapImage.rows, trimapImage.cols);
//		cv::imshow("trimapImage", trimapImage);
//		imwrite("D:/crystal/trimapImage.png", trimapImage);

#if 0 //test
		cv::Mat tmpTri[3];
		tmpTri[0] = trimapImage.clone();
		tmpTri[1] = trimapImage.clone();
		tmpTri[2] = trimapImage.clone();

		Mat synImg;
		merge(tmpTri, 3, synImg);
		synImg = 0.5 * srcImage + 0.5 * synImg;

//		cv::imshow("synImg6", synImg);
//		imwrite("D:/crystal/synImg6.png", synImg);
#endif

		Mat trimap = labelExpansion(srcImage, trimapImage);

//		imshow("trimap-le", trimap);
//		imwrite("D:/crystal/trimap-le.png", trimap);

        double elapsed_secs = ((double) (getTickCount() - begin)) / getTickFrequency();
		#ifndef WIN32
        char str[256];
        sprintf(str, "%f", elapsed_secs);
        __android_log_write(ANDROID_LOG_DEBUG, "---prepare---", str);
		#endif

		cv::alphamat::infoFlowFloat(srcImage, trimap, matImage);


#if 1 //GUIDE_FILTER
#define MAX_GUIDE_FILTER_SIZE        1200
		int cxNor = oriImage.cols;
		int cyNor = oriImage.rows;

		if (oriImage.cols >= oriImage.rows) {
			if (oriImage.cols > MAX_GUIDE_FILTER_SIZE) {
				cxNor = MAX_GUIDE_FILTER_SIZE;
				cyNor = MAX_GUIDE_FILTER_SIZE * oriImage.rows / oriImage.cols;
			}
		} else {
			if (oriImage.rows > MAX_GUIDE_FILTER_SIZE) {
				cxNor = MAX_GUIDE_FILTER_SIZE * oriImage.cols / oriImage.rows;
				cyNor = MAX_GUIDE_FILTER_SIZE;
			}
		}

		cv::resize(oriImage, srcImage, cv::Size(cxNor, cyNor));

		begin = cv::getTickCount();

		Mat gray, gray_filter, cut;
		cvtColor(srcImage, gray, COLOR_BGR2GRAY);
		ShowImage("gray", gray);
		SaveImage(PATH_TEST + "/gray.png", gray);

		resize(matImage, matImage, Size(srcImage.cols, srcImage.rows));

		ShowImage("mattingImage", matImage);
		SaveImage(PATH_TEST + "/mattingImageInfo.png", matImage);

		Mat splits[3];
		cv::split(srcImage, splits);
		splits[0] = splits[0].mul(matImage / 255.0);
		splits[1] = splits[1].mul(matImage / 255.0);
		splits[2] = splits[2].mul(matImage / 255.0);
		cv::merge(splits, 3, cut);
// 		cut = srcImage.mul(matImage / 255.0);

// 		imshow("gray", gray); imwrite("D:/gray.png", gray);
		ShowImage("matI", cut);
		SaveImage(PATH_TEST + "/matI.png", cut);
//		imshow("resize mat", matImage);
//		imwrite("D:/resize_mat.png", matImage);

		//////////////////////////////////////////////////////////////////////////
		int r = 8;
		double eps = 1e-4;
		eps *= 255 * 255;
		Mat filterImg = fastGuidedFilter(srcImage, cut, r, eps, 1); //1

		ShowImage("filterImg", filterImg);
		SaveImage(PATH_TEST + "/filterImg.png", filterImg);

		cvtColor(filterImg, gray_filter, COLOR_BGR2GRAY);

		ShowImage("gray_filter", gray_filter);
		SaveImage(PATH_TEST + "/gray_filter.png", gray_filter);

		for (int i = 0; i < filterImg.rows; ++i) {
			for (int j = 0; j < filterImg.cols; ++j) {
				if (i == 71 && j == 398) {
					i = i;
				}

				int val = matImage.at<uchar>(i, j);
				if (val >= 250) {
					continue;
				}

				int pix_filter = gray_filter.at<uchar>(i, j);
				int pix_gray = gray.at<uchar>(i, j);

				if (pix_filter == 0 || pix_gray == 0) {
					continue;
				}

				int pix = (255.0f * pix_filter / pix_gray);
				if (abs(val - pix) > 200) {
					continue;
				}

				if (pix > 255) {
					matImage.at<uchar>(i, j) = 255;
				} else {
					matImage.at<uchar>(i, j) = (uchar) pix;
				}
			}
		}

		ShowImage("matting", matImage);
		SaveImage(PATH_TEST + "/matting.png", matImage);

		cv::resize(matImage, matImage, oriImage.size());
		// 		matImage = (255.0 * filterImg / cut);
		// 		matImage.convertTo(matImage, CV_8UC1);

		ShowImage("mattingO", matImage);
		SaveImage(PATH_TEST + "/mattingO.png", matImage);

		elapsed_secs = ((double) (getTickCount() - begin)) / getTickFrequency();
        begin = getTickCount();
		#ifndef WIN32
        memset(str, 0, 256);
        sprintf(str, "%f", elapsed_secs);
        __android_log_write(ANDROID_LOG_DEBUG, "---guide filter---", str);
		#endif
//		FILE *fp = fopen("D:/time.txt", "ab+");
//		fprintf(fp, "guide filter: %f\n", elapsed_secs);
//		fclose(fp);

		//		cv::seamlessClone()
		//////////////////////////////////////////////////////////////////////////
#endif

		int x, y, pic_wid, pic_hei;
		#ifndef WIN32
		sprintf(str, "%d   %d", matImage.cols, matImage.rows);
		__android_log_write(ANDROID_LOG_DEBUG, "---last---", str);
		#endif
		
		getActualSizeofBitmap(matImage.data, matImage.cols, matImage.rows, x, y, pic_wid, pic_hei);

		imshow("matImageDefault2", matImage);
		imwrite("D:/matImageDefault2.png", matImage);

//		Mat newImage = srcImage.clone();
		Mat newImage = oriImage.clone();

		cv::Mat tmp, split[4];
		cv::Rect actualRect(x, y, pic_wid, pic_hei);
		cv::split(newImage, split);
        tmp = split[0].clone();
        split[0] = split[2].clone();
        split[2] = tmp;
		split[3] = matImage;
		cv::merge(split, 4, newImage);

		newImage = newImage(actualRect);
		Mat retImg;
		retImg.create(newImage.size(), CV_8UC4);
		newImage.copyTo(retImg);

		makePNG(retImg.data, retImg.cols, retImg.rows);

        elapsed_secs = ((double) (getTickCount() - begin)) / getTickFrequency();
		
		#ifndef WIN32
        memset(str, 0, 256);
        sprintf(str, "%f", elapsed_secs);
        __android_log_write(ANDROID_LOG_DEBUG, "---last---", str);
		#endif

		return retImg;
	}

	cv::Mat MagicEngine::mattingImageForEffectImage(cv::Mat &srcImage,
		cv::Mat &trimapImage) {

		int64 begin = cv::getTickCount();

		cv::Mat oriImage = srcImage.clone();
		cv::Mat matImage;

		cv::resize(oriImage, oriImage, cv::Size(trimapImage.rows, trimapImage.cols));

		trimapImage = preprocTrimapImage(trimapImage, trimapImage.rows, trimapImage.cols);
		//		cv::imshow("trimapImage", trimapImage);
		//		imwrite("D:/crystal/trimapImage.png", trimapImage);

#if 1 //test
		cv::Mat tmpTri[3];
		tmpTri[0] = trimapImage.clone();
		tmpTri[1] = trimapImage.clone();
		tmpTri[2] = trimapImage.clone();

		Mat synImg;
		merge(tmpTri, 3, synImg);
		synImg = 0.5 * oriImage + 0.5 * synImg;

		//		cv::imshow("synImg6", synImg);
		//		imwrite("D:/crystal/synImg6.png", synImg);
#endif

		Mat trimap = labelExpansion(oriImage, trimapImage);

		//		imshow("trimap-le", trimap);
		//		imwrite("D:/crystal/trimap-le.png", trimap);

		double elapsed_secs = ((double)(getTickCount() - begin)) / getTickFrequency();
#ifndef WIN32
		char str[256];
		sprintf(str, "%f", elapsed_secs);
		__android_log_write(ANDROID_LOG_DEBUG, "---prepare---", str);
#endif

		cv::alphamat::infoFlowFloat(oriImage, trimap, matImage);
		//		imshow("mattingImage", matImage);
		//		imwrite("D:/crystal/mattingImageInfo.png", matImage);


#if 1 //GUIDE_FILTER
#define MAX_GUIDE_FILTER_SIZE        1200
		int cxNor = srcImage.cols;
		int cyNor = srcImage.rows;

		if (srcImage.cols >= srcImage.rows) {
			if (srcImage.cols > MAX_GUIDE_FILTER_SIZE) {
				cxNor = MAX_GUIDE_FILTER_SIZE;
				cyNor = MAX_GUIDE_FILTER_SIZE * srcImage.rows / srcImage.cols;
			}
		}
		else {
			if (srcImage.rows > MAX_GUIDE_FILTER_SIZE) {
				cxNor = MAX_GUIDE_FILTER_SIZE * srcImage.cols / srcImage.rows;
				cyNor = MAX_GUIDE_FILTER_SIZE;
			}
		}

		cv::resize(srcImage, oriImage, cv::Size(cxNor, cyNor));

		begin = cv::getTickCount();

		Mat gray, gray_filter, cut;
		cvtColor(oriImage, gray, COLOR_BGR2GRAY);
		//		imshow("gray", gray);
		//		imwrite("D:/gray.png", gray);

		resize(matImage, matImage, Size(oriImage.cols, oriImage.rows));

		Mat splits[3];
		cv::split(oriImage, splits);
		splits[0] = splits[0].mul(matImage / 255.0);
		splits[1] = splits[1].mul(matImage / 255.0);
		splits[2] = splits[2].mul(matImage / 255.0);
		cv::merge(splits, 3, cut);
		// 		cut = oriImage.mul(matImage / 255.0);

		// 		imshow("gray", gray); imwrite("D:/gray.png", gray);
		//		imshow("matI", cut);
		//		imwrite("D:/matI.png", cut);
		//		imshow("resize mat", matImage);
		//		imwrite("D:/resize_mat.png", matImage);

		//////////////////////////////////////////////////////////////////////////
		int r = 8;
		double eps = 1e-4;
		eps *= 255 * 255;
		Mat filterImg = fastGuidedFilter(oriImage, cut, r, eps, 2); //1

																	//		imshow("filterImg", filterImg);
																	//		imwrite("D:/filterImg.png", filterImg);
		cvtColor(filterImg, gray_filter, COLOR_BGR2GRAY);
		//		imshow("gray_filter", gray_filter);
		//		imwrite("D:/gray_filter.png", gray_filter);

		for (int i = 0; i < filterImg.rows; ++i) {
			for (int j = 0; j < filterImg.cols; ++j) {
				if (i == 71 && j == 398) {
					i = i;
				}

				int val = matImage.at<uchar>(i, j);
				if (val >= 250) {
					continue;
				}

				int pix_filter = gray_filter.at<uchar>(i, j);
				int pix_gray = gray.at<uchar>(i, j);

				if (pix_filter == 0 || pix_gray == 0) {
					continue;
				}

				int pix = (255.0f * pix_filter / pix_gray);
				if (abs(val - pix) > 200) {
					continue;
				}

				if (pix > 255) {
					matImage.at<uchar>(i, j) = 255;
				}
				else {
					matImage.at<uchar>(i, j) = (uchar)pix;
				}
			}
		}
		cv::resize(matImage, matImage, srcImage.size());
		// 		matImage = (255.0 * filterImg / cut);
		// 		matImage.convertTo(matImage, CV_8UC1);

		//		imshow("matting", matImage);
		//		imwrite("D:/matting.png", matImage);

		elapsed_secs = ((double)(getTickCount() - begin)) / getTickFrequency();
		begin = getTickCount();
#ifndef WIN32
		memset(str, 0, 256);
		sprintf(str, "%f", elapsed_secs);
		__android_log_write(ANDROID_LOG_DEBUG, "---guide filter---", str);
#endif
		//		FILE *fp = fopen("D:/time.txt", "ab+");
		//		fprintf(fp, "guide filter: %f\n", elapsed_secs);
		//		fclose(fp);

		//		cv::seamlessClone()
		//////////////////////////////////////////////////////////////////////////
#endif

#ifndef WIN32
		sprintf(str, "%d   %d", matImage.cols, matImage.rows);
		__android_log_write(ANDROID_LOG_DEBUG, "---last---", str);
#endif

#ifndef WIN32
		memset(str, 0, 256);
		sprintf(str, "%f", elapsed_secs);
		__android_log_write(ANDROID_LOG_DEBUG, "---last---", str);
#endif

		return matImage;
	}

	cv::Mat MagicEngine::effectImageBackBlur(cv::Mat &srcImage,
		cv::Mat &trimapImage, bool bGrayBackgound) {
		int64 begin = cv::getTickCount();


		cv::Mat oriImage = srcImage.clone();

		cv::resize(oriImage, oriImage, cv::Size(trimapImage.rows, trimapImage.cols));

		Mat mask = getMaskImage(srcImage, trimapImage);


		cv::Mat retImg = srcImage.clone();;
		cv::Mat gryImage;
		cv::Mat oneImage(srcImage.rows, srcImage.cols, CV_8UC1, 255);
		cv::Mat oneImage3(srcImage.rows, srcImage.cols, CV_8UC3, Scalar(255, 255, 255));

		if (bGrayBackgound == true) {
			cvtColor(srcImage, gryImage, COLOR_BGR2GRAY);
			cv::blur(gryImage, gryImage, cv::Size(12, 12));
			cvtColor(gryImage, gryImage, COLOR_GRAY2BGR);
		}
		else {
			cv::blur(srcImage, gryImage, cv::Size(12, 12));
		}

		Mat result = gryImage.clone();

		imshow("oneImage3", oneImage3);
		imwrite("D:/oneImage3.png", oneImage3);

		imshow("mask", mask);
		imwrite("D:/mask.png", mask);
		

		imshow("gryImage", gryImage);
		imwrite("D:/gryImage.png", gryImage);

		imshow("srcImage", srcImage);
		imwrite("D:/srcImage.png", srcImage);


		Rect r = boundingRect(mask);
		Mat img1WarpedSub = srcImage(r);
		Mat img2Sub = gryImage(r);
		Mat maskSub = mask(r);

		Point center(r.width / 2, r.height / 2);

		cv::seamlessClone(img1WarpedSub, img2Sub, maskSub, center, retImg, NORMAL_CLONE);
		retImg.copyTo(result(r));

		imshow("result", result);
		imwrite("D:/result.png", result);

		imshow("effect", retImg);
		imwrite("D:/effect.png", retImg);


		cv::Mat tmp, split[4];
		cv::split(result, split);
		tmp = split[0].clone();
		split[0] = split[2].clone();
		split[2] = tmp;
		split[3] = oneImage;
		cv::merge(split, 4, result);

		return result;
	}

	cv::Mat MagicEngine::effectImageColorOnYou(cv::Mat &srcImage,
		cv::Mat &trimapImage) {

		int64 begin = cv::getTickCount();

		cv::Mat retImg;
		cv::Mat gryImage;
		cv::Mat oneImage(srcImage.rows, srcImage.cols, CV_8UC1, 255);
		cv::Mat oneImage3(srcImage.rows, srcImage.cols, CV_8UC3, Scalar(255, 255, 255));
		
		cvtColor(srcImage, gryImage, COLOR_BGR2GRAY);
		cv::blur(gryImage, gryImage, cv::Size(12, 12));
		cvtColor(gryImage, gryImage, COLOR_GRAY2BGR);

// 		cv::Mat matImage2 = mattingImageInformationFlowFloatGuideFilter(srcImage, trimapImage);
		cv::Mat matImage = mattingImageForEffectImage(srcImage, trimapImage);

		imshow("matImageEffect", matImage);
		imwrite("D:/matImageEffect.png", matImage);

// 		imshow("matImageDefault", matImage2);
// 		imwrite("D:/matImageDefault.png", matImage2);

		cv::Mat matImageF;
		matImage.convertTo(matImageF, CV_32FC1);

		cv::Mat weights2 = matImageF.mul(1 / 255.f);
		cv::Mat weights1 = Scalar::all(1.f) - weights2;
		cvtColor(matImage, matImage, COLOR_GRAY2BGR);
		
// 		retImg = gryImage.mul((oneImage3 - matImage) / 255.0) + srcImage.mul(matImage / 255.0);

		blendLinear(gryImage, srcImage, weights1, weights2, retImg);

		imshow("retImg", retImg);
		imwrite("D:/retImg.png", retImg);

// 		cv::Mat alphaImg = gryImage.mul((oneImage3 - matImage) / 255.0);
// 		cv::Mat betaImg = srcImage.mul(matImage / 255.0);
// 
// 		cv::Mat minus = oneImage3 - matImage;


// 		int r = 8;
// 		double eps = 1e-4;
// 		eps *= 255 * 255;
// 		Mat filterImg = fastGuidedFilter(retImg, betaImg, r, eps, 2); //1

// 		imshow("retImg", retImg);
// 		imwrite("D:/retImg.png", retImg);

// 		imshow("minus", minus);
// 		imwrite("D:/minus.png", minus);
// 
// 		imshow("oneImage3", oneImage3);
// 		imwrite("D:/oneImage3.png", oneImage3);
// 
// 		imshow("alphaImg", alphaImg);
// 		imwrite("D:/alphaImg.png", alphaImg);
// 
// 		imshow("betaImg", betaImg);
// 		imwrite("D:/betaImg.png", betaImg);



		imshow("gryImage", gryImage);
		imwrite("D:/gryImage.png", gryImage);

		imshow("matImage", matImage);
		imwrite("D:/matImage.png", matImage);

		imshow("srcImage", srcImage);
		imwrite("D:/srcImage.png", srcImage);

// 		imshow("effect", filterImg);
// 		imwrite("D:/effect.png", filterImg);

		cv::Mat tmp, split[4];
		cv::split(retImg, split);
		tmp = split[0].clone();
		split[0] = split[2].clone();
		split[2] = tmp;
		split[3] = oneImage;
		cv::merge(split, 4, retImg);

		return retImg;
	}

	cv::Mat MagicEngine::effectImageRomantic(cv::Mat& frmImage, cv::Mat& srcImage1, cv::Mat& srcImage2, Point2f pt1[4], Point2f pt2[4])
	{
		Mat retImg(frmImage.rows, frmImage.cols, srcImage1.type());

		if (frmImage.channels() < 4) {
			return retImg;
		}

		Point2f ptSrc1[4] = { { 0.f, 0.f },{ 0.f, srcImage1.rows - 1.f },{ srcImage1.cols - 1.f, srcImage1.rows - 1.f },{ srcImage1.cols - 1.f, 0.f } };
		Point2f ptSrc2[4] = { { 0.f, 0.f },{ 0.f, srcImage2.rows - 1.f },{ srcImage2.cols - 1.f, srcImage2.rows - 1.f },{ srcImage2.cols - 1.f, 0.f } };

		float nMaxX1 = 0.f, nMaxY1 = 0.f;
		float nMaxX2 = 0.f, nMaxY2 = 0.f;

		vector<Point2f> vecPt1;
		vector<Point2f> vecPt2;

		for (int i = 0; i < 4; i++) {
			vecPt1.push_back(pt1[i]);
			vecPt2.push_back(pt2[i]);

			nMaxX1 = cv::max(nMaxX1, pt1[i].x);
			nMaxY1 = cv::max(nMaxY1, pt1[i].y);
			nMaxX2 = cv::max(nMaxX2, pt2[i].x);
			nMaxY2 = cv::max(nMaxY2, pt2[i].y);
		}

		Rect r1 = boundingRect(vecPt1);
		Rect r2 = boundingRect(vecPt2);

// 		Size warped_image_size1 = Size(cvRound(nMaxX1), cvRound(nMaxY1));
// 		Size warped_image_size2 = Size(cvRound(nMaxX2), cvRound(nMaxY2));
		Size warped_image_size1 = Size(frmImage.cols, frmImage.rows);
		Size warped_image_size2 = Size(frmImage.cols, frmImage.rows);

		Mat M1 = getPerspectiveTransform(ptSrc1, pt1);
		Mat M2 = getPerspectiveTransform(ptSrc2, pt2);

		Mat warped_image1;
		Mat warped_image2;

		warpPerspective(srcImage1, warped_image1, M1, warped_image_size1); // do perspective transformation
		warpPerspective(srcImage2, warped_image2, M2, warped_image_size2); // do perspective transformation

		imwrite("D:/warped_image1.png", warped_image1);
		imshow("Warped Image1", warped_image1);
		imwrite("D:/warped_image2.png", warped_image2);
		imshow("Warped Image2", warped_image2);
		
		warped_image2(r2).copyTo(retImg(r2));
		warped_image1(r1).copyTo(retImg(r1));

// 		retImg.convertTo(retImg, CV_8UC4);
// 
// 		for (int y = 0; y < frmImage.rows; y++) {
// 			for (int x = 0; x < frmImage.cols; x++) {
// 				retImg.at<Vec4b>(y, x)[3] = 255;
// 
// 				int alpha = frmImage.at<Vec4b>(y, x)[3];
// 				if (alpha == 0) {
// 					continue;
// 				}
// 
// 				retImg.at<Vec4b>(y, x) = frmImage.at<Vec4b>(y, x);
// 				retImg.at<Vec4b>(y, x)[3] = 255;
// 			}
// 		}

		imshow("Romantic Image", retImg);

		return retImg;
	}

	cv::Mat MagicEngine::effectImageFaceBeauty(cv::Mat& srcImage, int smoothParam)
	{
		Mat bilateral, canny;
		cv::Canny(srcImage, canny, 10, 100);
		cv::bilateralFilter(srcImage, bilateral, 9, 75, 75);
		Mat pImageResult = (1.0 - smoothParam / 100.f) * (srcImage - bilateral) + bilateral;
		imshow("canny filter", canny);
		imshow("bilateral filter", pImageResult);

		return pImageResult;
	}

	//test for windows
	void MagicEngine::ShowImage(const String& winname, InputArray mat)
	{
#ifdef WIN32
		imshow(winname, mat);
#endif
	}

	void MagicEngine::SaveImage(const String& winname, InputArray mat)
	{
#ifdef WIN32
		imwrite(winname, mat);
#endif
	}

}