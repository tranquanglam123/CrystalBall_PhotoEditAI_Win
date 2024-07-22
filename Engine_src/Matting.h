#pragma once

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgproc/types_c.h"
#include "opencv2/highgui.hpp"
#include <opencv2/core/utility.hpp>

#include <iostream>
#include <string.h>

#include "vl/Calc.h"
#include "vl/nrutil.h"
#include "vl/kdtree.h"

using namespace std;
using namespace cv;

typedef unsigned char       BYTE;

#define SafeMemFree(x) {											\
	if (x) {														\
		free(x); x = NULL;											\
	}																\
}

typedef struct {
	unsigned int *pIDX;
	double *pD;
} VLKDTREEQUERY_RESULT, *LPKDTREEQUERY_REUSLT;

VlKDForest* vl_kdtreebuild(double* pData, int m, int n);
VLKDTREEQUERY_RESULT* vl_kdtreequery(VlKDForest* forest, double* pQuery, unsigned int Param0, unsigned int m, unsigned int n);

Mat convertTypeAndSize(Mat src, int dstType, cv::Size dstSize);
Mat labelExpansion(Mat inPut, Mat trimap);
int getColorLineLaplace(float *pfImage, BYTE *consts, int width, int height, int BitCount, int &Num);
int RunMattingThreeLayer(Mat image, Mat &mask, float *alpha, BYTE *OutImage, int flag);
int Matting(Mat* image, Mat &mask, float *alpha, int flag);
Mat AlphaBlendImage(Mat guide, Mat mask, Mat back);
Mat AlphaBlendImageByPos(Mat guide, Mat mask, Mat back, int cx, int cy);
