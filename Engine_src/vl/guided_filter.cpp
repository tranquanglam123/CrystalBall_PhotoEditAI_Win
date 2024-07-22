#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgproc/types_c.h"
#include "opencv2/highgui.hpp"
#include <opencv2/core/utility.hpp>
#include <opencv2/opencv.hpp>
//#include <caffe/guided_filter.hpp>
#include <iostream>
#include <string.h>

#include "Calc.h"
#include "nrutil.h"
#include "../Matting.h"

using namespace std;
//using namespace std::tr1;
using namespace cv;
//using namespace cv::ximgproc;

#ifndef SQR
#define SQR(x) ((x)*(x))
#endif

//#include "caffe/export.h"
//
//class GuidedFilterRefImpl : public GuidedFilter
//{
//    int height, width, rad, chNum;
//    Mat det;
//    Mat *channels, *exps, **vars, **A;
//    double eps;
//
//    void meanFilter(const Mat &src, Mat & dst);
//
//    void computeCovGuide();
//
//    void computeCovGuideInv();
//
//    void applyTransform(int cNum, Mat *Ichannels, Mat *beta, Mat **alpha, int dDepth);
//
//    void computeCovGuideAndSrc(int cNum, Mat **vars_I, Mat *Ichannels, Mat *exp_I);
//
//    void computeBeta(int cNum, Mat *beta, Mat *exp_I, Mat **alpha);
//
//    void computeAlpha(int cNum, Mat **alpha, Mat **vars_I);
//
//public:
//
//    GuidedFilterRefImpl(InputArray guide_, int rad, double eps);
//
//    void filter(InputArray src, OutputArray dst, int dDepth = -1);
//
//    ~GuidedFilterRefImpl();
//};
//
//void GuidedFilterRefImpl::meanFilter(const Mat &src, Mat & dst)
//{
//    boxFilter(src, dst, CV_32F, Size(2 * rad + 1, 2 * rad + 1), Point(-1, -1), true, BORDER_REFLECT);
//}
//
//GuidedFilterRefImpl::GuidedFilterRefImpl(InputArray _guide, int _rad, double _eps) :
//  height(_guide.rows()), width(_guide.cols()), rad(_rad), chNum(_guide.channels()), eps(_eps)
//{
//    Mat guide = _guide.getMat();
//    CV_Assert(chNum > 0 && chNum <= 3);
//
//    channels = new Mat[chNum];
//    exps     = new Mat[chNum];
//
//    A    = new Mat *[chNum];
//    vars = new Mat *[chNum];
//    for (int i = 0; i < chNum; ++i)
//    {
//        A[i]    = new Mat[chNum];
//        vars[i] = new Mat[chNum];
//    }
//
//    split(guide, channels);
//    for (int i = 0; i < chNum; ++i)
//    {
//        channels[i].convertTo(channels[i], CV_32F);
//        meanFilter(channels[i], exps[i]);
//    }
//
//    computeCovGuide();
//
//    computeCovGuideInv();
//}
//
//void GuidedFilterRefImpl::computeCovGuide()
//{
//    static const int pY[] = { 0, 0, 1, 0, 1, 2 };
//    static const int pX[] = { 0, 1, 1, 2, 2, 2 };
//
//    int numOfIterations = (SQR(chNum) - chNum) / 2 + chNum;
//    for (int k = 0; k < numOfIterations; ++k)
//    {
//        int i = pY[k], j = pX[k];
//
//        vars[i][j] = channels[i].mul(channels[j]);
//        meanFilter(vars[i][j], vars[i][j]);
//        vars[i][j] -= exps[i].mul(exps[j]);
//
//        if (i == j)
//            vars[i][j] += eps * Mat::ones(height, width, CV_32F);
//        else
//            vars[j][i] = vars[i][j];
//    }
//}
//
//void GuidedFilterRefImpl::computeCovGuideInv()
//{
//    static const int pY[] = { 0, 0, 1, 0, 1, 2 };
//    static const int pX[] = { 0, 1, 1, 2, 2, 2 };
//
//    int numOfIterations = (SQR(chNum) - chNum) / 2 + chNum;
//    if (chNum == 3)
//    {
//        for (int k = 0; k < numOfIterations; ++k){
//            int i = pY[k], i1 = (pY[k] + 1) % 3, i2 = (pY[k] + 2) % 3;
//            int j = pX[k], j1 = (pX[k] + 1) % 3, j2 = (pX[k] + 2) % 3;
//
//            A[i][j] = vars[i1][j1].mul(vars[i2][j2])
//                - vars[i1][j2].mul(vars[i2][j1]);
//        }
//    }
//    else if (chNum == 2)
//    {
//        A[0][0] = vars[1][1];
//        A[1][1] = vars[0][0];
//        A[0][1] = -vars[0][1];
//    }
//    else if (chNum == 1)
//        A[0][0] = Mat::ones(height, width, CV_32F);
//
//    for (int i = 0; i < chNum; ++i)
//        for (int j = 0; j < i; ++j)
//            A[i][j] = A[j][i];
//
//    det = vars[0][0].mul(A[0][0]);
//    for (int k = 0; k < chNum - 1; ++k)
//        det += vars[0][k + 1].mul(A[0][k + 1]);
//}
//
//GuidedFilterRefImpl::~GuidedFilterRefImpl(){
//    delete [] channels;
//    delete [] exps;
//
//    for (int i = 0; i < chNum; ++i)
//    {
//        delete [] A[i];
//        delete [] vars[i];
//    }
//
//    delete [] A;
//    delete [] vars;
//}
//
//void GuidedFilterRefImpl::filter(InputArray src_, OutputArray dst_, int dDepth)
//{
//    if (dDepth == -1) dDepth = src_.depth();
//    dst_.create(height, width, src_.type());
//    Mat src = src_.getMat();
//    Mat dst = dst_.getMat();
//    int cNum = src.channels();
//
//    CV_Assert(height == src.rows && width == src.cols);
//
//    Mat *Ichannels, *exp_I, **vars_I, **alpha, *beta;
//    Ichannels = new Mat[cNum];
//    exp_I     = new Mat[cNum];
//    beta      = new Mat[cNum];
//
//    vars_I = new Mat *[chNum];
//    alpha  = new Mat *[chNum];
//    for (int i = 0; i < chNum; ++i){
//        vars_I[i] = new Mat[cNum];
//        alpha[i]  = new Mat[cNum];
//    }
//
//    split(src, Ichannels);
//    for (int i = 0; i < cNum; ++i)
//    {
//        Ichannels[i].convertTo(Ichannels[i], CV_32F);
//        meanFilter(Ichannels[i], exp_I[i]);
//    }
//
//    computeCovGuideAndSrc(cNum, vars_I, Ichannels, exp_I);
//
//    computeAlpha(cNum, alpha, vars_I);
//
//    computeBeta(cNum, beta, exp_I, alpha);
//
//    for (int i = 0; i < chNum + 1; ++i)
//        for (int j = 0; j < cNum; ++j)
//            if (i < chNum)
//                meanFilter(alpha[i][j], alpha[i][j]);
//            else
//                meanFilter(beta[j], beta[j]);
//
//    applyTransform(cNum, Ichannels, beta, alpha, dDepth);
//    merge(Ichannels, cNum, dst);
//
//    delete [] Ichannels;
//    delete [] exp_I;
//    delete [] beta;
//
//    for (int i = 0; i < chNum; ++i)
//    {
//        delete [] vars_I[i];
//        delete [] alpha[i];
//    }
//    delete [] vars_I;
//    delete [] alpha;
//}
//
//void GuidedFilterRefImpl::computeAlpha(int cNum, Mat **alpha, Mat **vars_I)
//{
//    for (int i = 0; i < chNum; ++i)
//        for (int j = 0; j < cNum; ++j)
//        {
//            alpha[i][j] = vars_I[0][j].mul(A[i][0]);
//            for (int k = 1; k < chNum; ++k)
//                alpha[i][j] += vars_I[k][j].mul(A[i][k]);
//            alpha[i][j] /= det;
//        }
//}
//
//void GuidedFilterRefImpl::computeBeta(int cNum, Mat *beta, Mat *exp_I, Mat **alpha)
//{
//    for (int i = 0; i < cNum; ++i)
//    {
//        beta[i] = exp_I[i];
//        for (int j = 0; j < chNum; ++j)
//            beta[i] -= alpha[j][i].mul(exps[j]);
//    }
//}
//
//void GuidedFilterRefImpl::computeCovGuideAndSrc(int cNum, Mat **vars_I, Mat *Ichannels, Mat *exp_I)
//{
//    for (int i = 0; i < chNum; ++i)
//        for (int j = 0; j < cNum; ++j)
//        {
//            vars_I[i][j] = channels[i].mul(Ichannels[j]);
//            meanFilter(vars_I[i][j], vars_I[i][j]);
//            vars_I[i][j] -= exp_I[j].mul(exps[i]);
//        }
//}
//
//void GuidedFilterRefImpl::applyTransform(int cNum, Mat *Ichannels, Mat *beta, Mat **alpha, int dDepth)
//{
//    for (int i = 0; i < cNum; ++i)
//    {
//        Ichannels[i] = beta[i];
//        for (int j = 0; j < chNum; ++j)
//            Ichannels[i] += alpha[j][i].mul(channels[j]);
//        Ichannels[i].convertTo(Ichannels[i], dDepth);
//    }
//}

typedef tuple<int, string, string> GFParams;


// Alpha blending using multiply and add functions
Mat& blend(Mat& alpha, Mat& foreground, Mat& background, Mat& outImage)
{
	Mat fore, back;
	multiply(alpha, foreground, fore);
	multiply(Scalar::all(1.0) - alpha, background, back);
	add(fore, back, outImage);

	return outImage;
}

// Alpha Blending using direct pointer access
Mat& alphaBlendDirectAccess(Mat& alpha, Mat& foreground, Mat& background, Mat& outImage)
{
	int numberOfPixels = foreground.rows * foreground.cols * foreground.channels();

	unsigned char* fptr = reinterpret_cast<unsigned char*>(foreground.data);
	unsigned char* bptr = reinterpret_cast<unsigned char*>(background.data);
	float* aptr = reinterpret_cast<float*>(alpha.data);
	float* outImagePtr = reinterpret_cast<float*>(outImage.data);

	int i, j;
	for (j = 0; j < numberOfPixels; ++j, outImagePtr++, fptr++, aptr++, bptr++) {
		*outImagePtr = (*fptr)*(*aptr) + (*bptr)*(1 - *aptr);
	}

	return outImage;
}

Mat& alphaBlendByPos(Mat& alpha, Mat& foreground, Mat& background, Mat& outImage, int cx, int cy)
{
    int numberOfPixels = background.rows * background.cols * background.channels();

    int rx, by;
    if((cx + foreground.cols) < background.cols )
        rx = cx +foreground.cols;
    else
        rx = background.cols;

    if((cy + foreground.rows) < background.rows )
        by = cy +foreground.rows;
    else
        by = background.rows;

    int xdelta = 0;
    int ydelta = 0;
    if(cx <0) {xdelta = -cx; cx = 0;}
    if(cy <0) {ydelta = -cy; cy = 0;}
    unsigned char* fptr = reinterpret_cast<unsigned char*>(foreground.data);
    unsigned char* bptr = reinterpret_cast<unsigned char*>(background.data);
    float* aptr = reinterpret_cast<float*>(alpha.data);
    float* outImagePtr = reinterpret_cast<float*>(outImage.data);

    int i, j;
//    for (j = 0; j < numberOfPixels; ++j, outImagePtr++, fptr++, aptr++, bptr++) {
//        *outImagePtr = (*fptr)*(*aptr) + (*bptr)*(1 - *aptr);
//    }
    for(i = cy; i<by; i++)
    {
        for(j=cx; j<rx; j++)
        {
//            unsigned char alpha = aptr[(i-cy + ydelta)*(rx-cx + xdelta)*3+(j-cx + xdelta)*3];
            unsigned char alpha = aptr[(i-cy + ydelta)*(foreground.cols)*3+(j-cx + xdelta)*3];
            outImagePtr[i*outImage.cols*3 + j*3] = fptr[(i-cy + ydelta)*(foreground.cols)*3+(j-cx+xdelta)*3] * alpha + bptr[i*background.cols*3 + j*3]*(1-alpha);
            outImagePtr[i*outImage.cols*3 + j*3+1] = fptr[(i-cy + ydelta)*(foreground.cols)*3+(j-cx+xdelta)*3+1] * alpha + bptr[i*background.cols*3 + j*3+1]*(1-alpha);
            outImagePtr[i*outImage.cols*3 + j*3+2] = fptr[(i-cy + ydelta)*(foreground.cols)*3+(j-cx+xdelta)*3+2] * alpha + bptr[i*background.cols*3 + j*3+2]*(1-alpha);
        }
    }
    return outImage;
}

//Mat AlphaBlendImage(Mat guide, Mat mask)
//{
//	int guideCnNum = 3;
//	int srcCnNum = 3;
//
//	srcCnNum = mask.channels();
//	//ASSERT_TRUE(!guide.empty() && !src.empty());
//
//	Size dstSize(guide.cols, guide.rows);
//	guide = convertTypeAndSize(guide, CV_MAKE_TYPE(guide.depth(), guideCnNum), dstSize);
//	mask = convertTypeAndSize(mask, CV_MAKE_TYPE(mask.depth(), srcCnNum), dstSize);
//	mask = mask * 255;
//	Mat output;
//
//	ximgproc::guidedFilter(guide, mask, output, 3, 1e-6);
//
//	size_t whitePixels = 0;
//	for (int i = 0; i < output.rows; i++) {
//		for (int j = 0; j < output.cols; j++) {
//			if (output.channels() == 1) {
//				if (output.ptr<uchar>(i)[j] == 255)
//					whitePixels++;
//			}
//			else if (output.channels() == 3) {
//				Vec3b currentPixel = output.ptr<Vec3b>(i)[j];
//				if (currentPixel == Vec3b(255, 255, 255))
//					whitePixels++;
//			}
//		}
//	}
//	double whiteRate = whitePixels / (double)output.total();
//	imwrite("d:\\mask2.bmp", output);
//	//EXPECT_LE(whiteRate, 0.1);
//
//
//	//Alpha blending
//	Mat alpha = output.clone();
//
//	// Read background image
//	Mat background = imread("d:/test/back.png");
//	Mat foreground = guide.clone();
//	resize(background, background, Size(foreground.cols, foreground.rows));
//
//	imwrite("d:\\foreground.bmp", foreground);
//	imwrite("d:\\background.bmp", background);
//	imwrite("d:\\alpha.bmp", alpha);
//
//	// Convert Mat to float data type
//	foreground.convertTo(foreground, CV_32FC3);
//	background.convertTo(background, CV_32FC3);
//	alpha.convertTo(alpha, CV_32FC3, 1.0 / 255); // keeps the alpha values betwen 0 and 1
//												 // Number of iterations to average the performane over
//	int numOfIterations = 1; //1000;
//							 // Alpha blending using functions multiply and add
//	Mat outImage = Mat::zeros(foreground.size(), foreground.type());
//
//	for (int i = 0; i < numOfIterations; i++) {
//		outImage = blend(alpha, foreground, background, outImage);
//	}
//	imwrite("d:\\outImage1.bmp", outImage);
//
//	// Alpha blending using direct Mat access with for loop
//	outImage = Mat::zeros(foreground.size(), foreground.type());
//
//	for (int i = 0; i < numOfIterations; i++) {
//		outImage = alphaBlendDirectAccess(alpha, foreground, background, outImage);
//	}
//
//	imwrite("d:\\outImage.bmp", outImage);
//
//	return outImage;
//}

Mat AlphaBlendImage(Mat guide, Mat mask, Mat background)
{
    int guideCnNum = 3;
    int srcCnNum = 3;

    srcCnNum = mask.channels();
    //ASSERT_TRUE(!guide.empty() && !src.empty());

    Size dstSize(guide.cols, guide.rows);
    guide = convertTypeAndSize(guide, CV_MAKE_TYPE(guide.depth(), guideCnNum), dstSize);
    mask = convertTypeAndSize(mask, CV_MAKE_TYPE(mask.depth(), srcCnNum), dstSize);
    mask = mask * 255;
#if 0
    Mat output;

    ximgproc::guidedFilter(guide, mask, output, 3, 1e-6);

    size_t whitePixels = 0;
    for (int i = 0; i < output.rows; i++) {
        for (int j = 0; j < output.cols; j++) {
            if (output.channels() == 1) {
                if (output.ptr<uchar>(i)[j] == 255)
                    whitePixels++;
            }
            else if (output.channels() == 3) {
                Vec3b currentPixel = output.ptr<Vec3b>(i)[j];
                if (currentPixel == Vec3b(255, 255, 255))
                    whitePixels++;
            }
        }
    }
    double whiteRate = whitePixels / (double)output.total();
    //EXPECT_LE(whiteRate, 0.1);


    //Alpha blending
    Mat alpha = output.clone();
#else
    Mat alpha = mask;//.clone();
#endif
    // Read background image
    Mat foreground = guide;//.clone();
    resize(background, background, Size(foreground.cols, foreground.rows));

    // Convert Mat to float data type
   // foreground.convertTo(foreground, CV_32FC3);
    //background.convertTo(background, CV_32FC3);
    alpha.convertTo(alpha, CV_32FC3, 1.0 / 255); // keeps the alpha values betwen 0 and 1
    // Number of iterations to average the performane over
//    int numOfIterations = 1; //1000;
    // Alpha blending using functions multiply and add
    Mat outImage = Mat::zeros(foreground.size(), CV_32FC3);
//    Mat outImage = Mat::zeros(foreground.size(), foreground.type());

//    for (int i = 0; i < numOfIterations; i++) {
//        outImage = blend(alpha, foreground, background, outImage);
  //  }
//    imwrite("d:\\outImage1.bmp", outImage);

    // Alpha blending using direct Mat access with for loop
  //  outImage = Mat::zeros(foreground.size(), foreground.type());

 //   for (int i = 0; i < numOfIterations; i++) {
        outImage = alphaBlendDirectAccess(alpha, foreground, background, outImage);
   // }

//    imwrite("d:\\outImage.bmp", outImage);

    return outImage;
}

Mat AlphaBlendImageByPos(Mat guide, Mat mask, Mat background, int cx, int cy)
{
    int guideCnNum = 3;
    int srcCnNum = 3;

    srcCnNum = mask.channels();
    //ASSERT_TRUE(!guide.empty() && !src.empty());

    Size dstSize(guide.cols, guide.rows);
    guide = convertTypeAndSize(guide, CV_MAKE_TYPE(guide.depth(), guideCnNum), dstSize);
    mask = convertTypeAndSize(mask, CV_MAKE_TYPE(mask.depth(), srcCnNum), dstSize);
    mask = mask * 255;

    Mat alpha = mask;//.clone();

    Mat foreground = guide;//.clone();
    alpha.convertTo(alpha, CV_32FC3, 1.0 / 255); // keeps the alpha values betwen 0 and 1
    Mat outImage = background.clone();
    outImage.convertTo(outImage, CV_32FC3);
    outImage = alphaBlendByPos(alpha, foreground, background, outImage, cx, cy);

    return outImage;
}