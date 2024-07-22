// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef WIN32
#include <android/log.h>
#endif
#include "precomp.hpp"

#include <Eigen/Sparse>

using namespace Eigen;

namespace cv { namespace alphamat {

static
void solve_double(SparseMatrix<double> Wcm, SparseMatrix<double> Wuu, SparseMatrix<double> Wl, SparseMatrix<double> Dcm,
        SparseMatrix<double> Duu, SparseMatrix<double> Dl, SparseMatrix<double> T,
        Mat& wf, Mat& alpha)
{
    float suu = 0.01, sl = 0.1, lamd = 100;

    SparseMatrix<double> Lifm = ((Dcm - Wcm).transpose()) * (Dcm - Wcm) + sl * (Dl - Wl) + suu * (Duu - Wuu);

    SparseMatrix<double> A;
    int n = wf.rows;
    VectorXd b(n), x(n);

    Eigen::VectorXd wf_;
    cv2eigen(wf, wf_);

    A = Lifm + lamd * T;
    b = (lamd * T) * (wf_);

    ConjugateGradient<SparseMatrix<double>, Lower | Upper> cg;

	cg.setTolerance(0.000001);
    cg.setMaxIterations(500);
    cg.compute(A);
    x = cg.solve(b);
    CV_LOG_INFO(NULL, "ALPHAMAT: #iterations:     " << cg.iterations());
    CV_LOG_INFO(NULL, "ALPHAMAT: estimated error: " << cg.error());

    int nRows = alpha.rows;
    int nCols = alpha.cols;
    float pix_alpha;
    for (int j = 0; j < nCols; ++j)
    {
        for (int i = 0; i < nRows; ++i)
        {
            pix_alpha = x(i + j * nRows);
            if (pix_alpha < 0)
                pix_alpha = 0;
            if (pix_alpha > 1)
                pix_alpha = 1;
            alpha.at<uchar>(i, j) = uchar(pix_alpha * 255);
        }
    }
}

static
void solve_float(SparseMatrix<float> Wcm, SparseMatrix<float> Wuu, SparseMatrix<float> Wl, SparseMatrix<float> Dcm,
	SparseMatrix<float> Duu, SparseMatrix<float> Dl, SparseMatrix<float> T,
	Mat& wf, Mat& alpha)
{
	float suu = 0.01, sl = 0.1, lamd = 100;

	SparseMatrix<float> Lifm = ((Dcm - Wcm).transpose()) * (Dcm - Wcm) + sl * (Dl - Wl) + suu * (Duu - Wuu);

	SparseMatrix<float> A;
	int n = wf.rows;
	VectorXf b(n), x(n);

	Eigen::VectorXf wf_;
	cv2eigen(wf, wf_);

	A = Lifm + lamd * T;
	b = (lamd * T) * (wf_);

	ConjugateGradient<SparseMatrix<float>, Lower | Upper> cg;

	cg.setTolerance(0.00001);
	cg.setMaxIterations(500);
	cg.compute(A);
	x = cg.solve(b);
	CV_LOG_INFO(NULL, "ALPHAMAT: #iterations:     " << cg.iterations());
	CV_LOG_INFO(NULL, "ALPHAMAT: estimated error: " << cg.error());

	int nRows = alpha.rows;
	int nCols = alpha.cols;
	float pix_alpha;
	for (int j = 0; j < nCols; ++j)
	{
		for (int i = 0; i < nRows; ++i)
		{
			pix_alpha = x(i + j * nRows);
			if (pix_alpha < 0)
				pix_alpha = 0;
			if (pix_alpha > 1)
				pix_alpha = 1;
			alpha.at<uchar>(i, j) = uchar(pix_alpha * 255);
		}
	}
}

void infoFlowDouble(InputArray image_ia, InputArray tmap_ia, OutputArray result)
{
    Mat image = image_ia.getMat();
    Mat tmap = tmap_ia.getMat();

    int64 begin = cv::getTickCount();

    int nRows = image.rows;
    int nCols = image.cols;
    int N = nRows * nCols;

    SparseMatrix<double> T(N, N);
    typedef Triplet<double> Tr;
    std::vector<Tr> triplets;

    //Pre-process trimap
    for (int i = 0; i < nRows; ++i)
    {
        for (int j = 0; j < nCols; ++j)
        {
            uchar& pix = tmap.at<uchar>(i, j);
            if (pix <= 0.2f * 255)
                pix = 0;
            else if (pix >= 0.8f * 255)
                pix = 255;
            else
                pix = 128;
        }
    }

    Mat wf = Mat::zeros(nRows * nCols, 1, CV_8U);

    // Column Major Interpretation for working with SparseMatrix
    for (int i = 0; i < nRows; ++i)
    {
        for (int j = 0; j < nCols; ++j)
        {
            uchar pix = tmap.at<uchar>(i, j);

            // collection of known pixels samples
            triplets.push_back(Tr(i + j * nRows, i + j * nRows, (pix != 128) ? 1 : 0));

            // foreground pixel
            wf.at<uchar>(i + j * nRows, 0) = (pix > 200) ? 1 : 0;
        }
    }

    SparseMatrix<double> Wl(N, N), Dl(N, N);
    local_info_double(image, tmap, Wl, Dl);

    SparseMatrix<double> Wcm(N, N), Dcm(N, N);
    cm_double(image, tmap, Wcm, Dcm);

    Mat new_tmap = tmap.clone();

    SparseMatrix<double> Wuu(N, N), Duu(N, N);
    Mat image_t = image.t();
    Mat tmap_t = tmap.t();
    UU_double(image, tmap, Wuu, Duu);

    double elapsed_secs = ((double)(getTickCount() - begin)) / getTickFrequency();

//	FILE* fp = fopen("D:/time.txt", "ab+");
//	fprintf(fp, "infoflow double: %f\n", elapsed_secs);

    T.setFromTriplets(triplets.begin(), triplets.end());

    Mat alpha = Mat::zeros(nRows, nCols, CV_8UC1);
    solve_double(Wcm, Wuu, Wl, Dcm, Duu, Dl, T, wf, alpha);

    alpha.copyTo(result);

    elapsed_secs = ((double)(getTickCount() - begin)) / getTickFrequency();
    CV_LOG_INFO(NULL, "ALPHAMAT: total time: " << elapsed_secs);

//	fprintf(fp, "infoflow double: %f\n", elapsed_secs);
//	fclose(fp);
}

void infoFlowFloat(InputArray image_ia, InputArray tmap_ia, OutputArray result)
{
	Mat image = image_ia.getMat();
	Mat tmap = tmap_ia.getMat();

	int64 begin = cv::getTickCount();

	int nRows = image.rows;
	int nCols = image.cols;
	int N = nRows * nCols;

	SparseMatrix<float> T(N, N);
	typedef Triplet<float> Tr;
	std::vector<Tr> triplets;

	//Pre-process trimap
	for (int i = 0; i < nRows; ++i) {
		for (int j = 0; j < nCols; ++j) {
			uchar& pix = tmap.at<uchar>(i, j);
			if (pix <= 0.2f * 255)
				pix = 0;
			else if (pix >= 0.8f * 255)
				pix = 255;
			else
				pix = 128;
		}
	}

	Mat wf = Mat::zeros(nRows * nCols, 1, CV_8U);

	// Column Major Interpretation for working with SparseMatrix
	for (int i = 0; i < nRows; ++i) {
		for (int j = 0; j < nCols; ++j) {
			uchar pix = tmap.at<uchar>(i, j);

			// collection of known pixels samples
			triplets.push_back(Tr(i + j * nRows, i + j * nRows, (pix != 128) ? 1 : 0));

			// foreground pixel
			wf.at<uchar>(i + j * nRows, 0) = (pix > 200) ? 1 : 0;
		}
	}

	SparseMatrix<float> Wl(N, N), Dl(N, N);
	local_info_float(image, tmap, Wl, Dl);

	SparseMatrix<float> Wcm(N, N), Dcm(N, N);
	cm_float(image, tmap, Wcm, Dcm);

	Mat new_tmap = tmap.clone();

	SparseMatrix<float> Wuu(N, N), Duu(N, N);
	Mat image_t = image.t();
	Mat tmap_t = tmap.t();
	UU_float(image, tmap, Wuu, Duu);

	double elapsed_secs = ((double)(getTickCount() - begin)) / getTickFrequency();

#ifndef WIN32
    char str[256];
    sprintf(str, "%f", elapsed_secs);
    __android_log_write(ANDROID_LOG_DEBUG, "---infoflow---1", str);
#endif
//	FILE* fp = fopen("D:/time.txt", "ab+");
//	fprintf(fp, "infoflow float: %f\n", elapsed_secs);

	T.setFromTriplets(triplets.begin(), triplets.end());

	Mat alpha = Mat::zeros(nRows, nCols, CV_8UC1);
	solve_float(Wcm, Wuu, Wl, Dcm, Duu, Dl, T, wf, alpha);

	alpha.copyTo(result);

	elapsed_secs = ((double)(getTickCount() - begin)) / getTickFrequency();
//	CV_LOG_INFO(NULL, "ALPHAMAT: total time: " << elapsed_secs);
#ifndef WIN32
    memset(str, 0, 256);
    sprintf(str, "%f", elapsed_secs);
    __android_log_write(ANDROID_LOG_DEBUG, "---infoflow---2", str);
#endif

//	fprintf(fp, "infoflow float: %f\n", elapsed_secs);
//	fclose(fp);
}

}}  // namespace cv::alphamat
