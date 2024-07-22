#include "Matting.h"
using namespace cv;
extern unsigned long *g_ija;
extern double *g_sa;
extern float *gf_sa;
float *Alpha;

cv::Mat convertTypeAndSize(cv::Mat src, int dstType, Size dstSize)
{
	cv::Mat dst;
	int srcCnNum = src.channels();
	int dstCnNum = CV_MAT_CN(dstType);
	CV_Assert(srcCnNum == 3);

	if (srcCnNum == dstCnNum)
	{
		src.copyTo(dst);
	}
	else if (dstCnNum == 1 && srcCnNum == 3)
	{
		cv::cvtColor(src, dst, COLOR_BGR2GRAY);
	}
	else if (dstCnNum == 1 && srcCnNum == 4)
	{
		cv::cvtColor(src, dst, COLOR_BGRA2GRAY);
	}
	else
	{
		vector<Mat> srcCn;
		cv::split(src, srcCn);
		srcCn.resize(dstCnNum);

		uint64 seed = 10000 * src.rows + 1000 * src.cols + 100 * dstSize.height + 10 * dstSize.width + dstType;
		RNG rnd(seed);

		for (int i = srcCnNum; i < dstCnNum; i++)
		{
			Mat& donor = srcCn[i % srcCnNum];

			double minVal, maxVal;
			cv::minMaxLoc(donor, &minVal, &maxVal);

			Mat randItem(src.size(), CV_MAKE_TYPE(src.depth(), 1));
			cv::randn(randItem, 0, (maxVal - minVal) / 100);

			cv::add(donor, randItem, srcCn[i]);
		}

		cv::merge(srcCn, dst);
	}

	dst.convertTo(dst, dstType);
	cv::resize(dst, dst, dstSize, 0, 0, INTER_LINEAR/*_EXACT*/);

	return dst;
}

cv::Mat matlab_reshape(const cv::Mat &m, int new_row, int new_col, int new_ch)
{
	int old_row, old_col, old_ch;
	old_row = m.size().height;
	old_col = m.size().width;
	old_ch = m.channels();

	cv::Mat m1(1, new_row*new_col*new_ch, m.depth());

	vector <Mat> p(old_ch);
	cv::split(m, p);
	for (int i = 0; i < p.size(); ++i) {
		Mat t(p[i].size().height, p[i].size().width, m1.type());
		t = p[i].t();
		Mat aux = m1.colRange(i*old_row*old_col, (i + 1)*old_row*old_col).rowRange(0, 1);
		t.reshape(0, 1).copyTo(aux);
	}

	vector <Mat> r(new_ch);
	for (int i = 0; i < r.size(); ++i) {
		Mat aux = m1.colRange(i*new_row*new_col, (i + 1)*new_row*new_col).rowRange(0, 1);
		r[i] = aux.reshape(0, new_col);
		r[i] = r[i].t();
	}

	Mat result;
	cv::merge(r, result);
	return result;
}

double H[9 * 9] = {
	0.00000000000,	0.00000000000,	0.00000000000,	0.00000000000,	0.00000000000,	0.00000000000,	0.00000000000,	0.00000000000,	0.00000000000,
	0.00000000000,	0.00000000000,	0.00000000000,	0.00000000128,	0.00000000942,	0.00000000128,	0.00000000000,	0.00000000000,	0.00000000000,
	0.00000000000,	0.00000000000,	0.00000006962,	0.00002808864,	0.00020754854,	0.00002808864,	0.00000006962,	0.00000000000,	0.00000000000,
	0.00000000000,	0.00000000128,	0.00002808864,	0.01133176631,	0.08373105697,	0.01133176631,	0.00002808864,	0.00000000128,	0.00000000000,
	0.00000000000,	0.00000000942,	0.00020754854,	0.08373105697,	0.61869347718,	0.08373105697,	0.00020754854,	0.00000000942,	0.00000000000,
	0.00000000000,	0.00000000128,	0.00002808864,	0.01133176631,	0.08373105697,	0.01133176631,	0.00002808864,	0.00000000128,	0.00000000000,
	0.00000000000,	0.00000000000,	0.00000006962,	0.00002808864,	0.00020754854,	0.00002808864,	0.00000006962,	0.00000000000,	0.00000000000,
	0.00000000000,	0.00000000000,	0.00000000000,	0.00000000128,	0.00000000942,	0.00000000128,	0.00000000000,	0.00000000000,	0.00000000000,
	0.00000000000,	0.00000000000,	0.00000000000,	0.00000000000,	0.00000000000,	0.00000000000,	0.00000000000,	0.00000000000,	0.00000000000
};

template<class T>
cv::Mat     POW2(const cv::Mat& a)
{
	cv::Mat t(a.size(), a.type(), cv::Scalar(0));
	for (int j = 0; j < a.rows; j++) {
		const T* ap = a.ptr<T>(j);
		T* tp = t.ptr<T>(j);
		for (int i = 0; i < a.cols; i++)    tp[i] = ap[i] * ap[i];
	}
	return t;
}

template<class T>
cv::Mat     SQRT(const cv::Mat& a)
{
	cv::Mat t(a.size(), a.type(), cv::Scalar(0));
	for (int j = 0; j < a.rows; j++) {
		const T* ap = a.ptr<T>(j);
		T* tp = t.ptr<T>(j);
		for (int i = 0; i < a.cols; i++)    tp[i] = sqrt(ap[i]);
	}
	return t;
}

Mat stdfilt(Mat_<double> const& I, InputArray kernel, int borderType)
{
	Mat G1, G2;
	Mat I2 = POW2<double>(I);
	Scalar n = sum(kernel);
	filter2D(I2, G2, CV_64F, kernel, Point(-1, -1), 0, borderType);
	G2 = G2 / (n[0] - 1.);
	filter2D(I, G1, CV_64F, kernel, Point(-1, -1), 0, borderType);
	G1 = POW2<double>(G1) / (n[0] * (n[0] - 1.));
	return SQRT<double>(max<double>(G2 - G1, 0.));
}

Mat stdfilt(Mat_<double> const& image32f, int ksize, int borderType)
{
	int kernel_size = 1 + 2 * ksize;
	Mat kernel = Mat::ones(kernel_size, kernel_size, CV_64F);
	return stdfilt(image32f, kernel, borderType);
}

double findMinD(double *I, double *D, double *J, int m, int n, int ak[8], int i, int j)
{
	double minD, gradient, gamma, temp;
	int k, x, y, NN;

	NN = m*n;
	minD = D[i + j*m];
	for (k = 1; k <= 4; k++)
	{
		x = j + ak[2 * k - 2];
		y = i + ak[2 * k - 1];
		if (y < 0 || x < 0 || y >= m || x >= n)
			continue;
		else
		{
			// 			gradient = pow(I[i + j*m] - I[y + x*m], 2) + pow(I[i + j*m + N] - I[y + x*m + N], 2) + pow(I[i + j*m + 2 * N] - I[y + x*m + 2 * N], 2);
			gradient = pow(I[3 * i + 3 * j*m + 0] - I[3 * y + 3 * x*m + 0], 2) +
				pow(I[3 * i + 3 * j*m + 1] - I[3 * y + 3 * x*m + 1], 2) +
				pow(I[3 * i + 3 * j*m + 2] - I[3 * y + 3 * x*m + 2], 2);
			gamma = 1 / (J[i + j*m] + 0.01);
			temp = D[y + x*m] + pow(ak[2 * k - 2] * ak[2 * k - 2] + ak[2 * k - 1] * ak[2 * k - 1] + gamma*gamma*gradient, 0.5);
			if (temp < minD)
				minD = temp;
		}
	}
	return minD;
}

void rasterScan(double *I, double *D, double *J, int m, int n)
{
	int Ak[4][8] = { { -1,-1,-1,0,-1,1,0,-1 },{ 1,1,1,0,1,-1,0,1 },{ 1,-1,0,-1,-1,-1,1,0 },{ -1,1,0,1,1,1,-1,0 } };
	int i, j, k, ak[8];
	double minD;

	for (k = 0; k <= 7; k++)
		ak[k] = Ak[0][k];
	for (j = 0; j < n; j++)
		for (i = 0; i < m; i++)
		{
			D[i + j*m] = findMinD(I, D, J, m, n, ak, i, j);
		};

	for (k = 0; k <= 7; k++)
		ak[k] = Ak[1][k];
	for (j = n - 1; j >= 0; j--)
		for (i = m - 1; i >= 0; i--)
		{
			D[i + j*m] = findMinD(I, D, J, m, n, ak, i, j);
		};

	for (k = 0; k <= 7; k++)
		ak[k] = Ak[2][k];
	for (i = 0; i <= m - 1; i++)
		for (j = n - 1; j >= 0; j--)
		{
			D[i + j*m] = findMinD(I, D, J, m, n, ak, i, j);
		};

	for (k = 0; k <= 7; k++)
		ak[k] = Ak[3][k];
	for (i = m - 1; i >= 0; i--)
		for (j = 0; j <= n - 1; j++)
		{
			D[i + j*m] = findMinD(I, D, J, m, n, ak, i, j);
		};

}

cv::Mat GetGeodesicDis(cv::Mat img, cv::Mat ordfiltImg, cv::Mat t_mono)
{
	int m = img.rows;
	int n = img.cols;
	Mat D(m, n, CV_64FC1);
	int v = 10000;
	D = v * t_mono;

	double* I = img.ptr<double>();
	double* J = ordfiltImg.ptr<double>();
	double* d = D.ptr<double>();
	rasterScan(I, d, J, n, m);
	return D;
}

cv::Mat labelExpansion(cv::Mat inPut, cv::Mat trimap)
{
	Mat I = convertTypeAndSize(inPut, CV_64FC3, Size(inPut.cols, inPut.rows));
	Mat grayI = convertTypeAndSize(inPut, CV_64FC1, Size(inPut.cols, inPut.rows));
	Mat J = stdfilt(grayI, 3, BORDER_REFLECT);
//	imwrite("d:/test/grayI.bmp", grayI);
//	imwrite("d:/test/J.bmp", J);
	Mat J_ordfilt2;
	Mat kernel = Mat::ones(9, 9, CV_64F);
	// 	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	erode(J, J_ordfilt2, kernel, cv::Point(-1, -1), 1, BORDER_CONSTANT, 0);
	int ch = J_ordfilt2.channels();
//	imwrite("d:/test/J_ordfilt2.bmp", J_ordfilt2);

	// 	Mat upper = Mat::ones(J_ordfilt2.rows, J_ordfilt2.cols, CV_8UC1);
	// 	upper = upper * 254;
	// 	Mat M, M1;
	// 
	// 	Mat trimap_gray = convertTypeAndSize(trimap, CV_8UC1, Size(trimap.cols, trimap.rows));
	// 	cv::inRange(trimap_gray, Mat::zeros(trimap.rows,trimap.cols, CV_8UC1), upper, M);
	// 	M = M / 255;
	// 	M.convertTo(M, CV_64FC1);
	// 	Mat DF = GetGeodesicDis(I, J_ordfilt2, M);
	// 
	// 	cv::inRange(trimap_gray, upper / 254, trimap_gray, M1);
	// 	M1 = M1 / 255;
	// 	M1.convertTo(M1, CV_64FC1);
	// 	Mat DB = GetGeodesicDis(I, J_ordfilt2, M1);

	Mat M0 = Mat::zeros(trimap.size(), CV_8UC1);
	for (int y = 0; y < trimap.rows; y++) {
		for (int x = 0; x < trimap.cols; x++) {
			if (trimap.data[y*trimap.cols + x] < 255) {
				M0.data[y*trimap.cols + x] = 1;
			}
		}
	}
	// 	Mat M = trimap < 255;
	Mat MM;// = convertTypeAndSize(M, CV_64FC3, Size(M.cols, M.rows));
	M0.convertTo(MM, CV_64FC1);
//	imwrite("d:/test/M.bmp", M0);

	Mat DF = GetGeodesicDis(I, J_ordfilt2, MM);

	Mat B = Mat::zeros(trimap.size(), CV_8UC1);
	for (int y = 0; y < trimap.rows; y++) {
		for (int x = 0; x < trimap.cols; x++) {
			if (trimap.data[y*trimap.cols + x] > 0) {
				B.data[y*trimap.cols + x] = 1;
			}
		}
	}
	// 	Mat B = trimap > 0;
	Mat BB;// = convertTypeAndSize(B, CV_64FC3, Size(B.cols, B.rows));
	B.convertTo(BB, CV_64FC1);
	Mat DB = GetGeodesicDis(I, J_ordfilt2, BB);
//	imwrite("d:/test/MM.bmp", MM);
//	imwrite("d:/test/B.bmp", B);
//	imwrite("d:/test/DF.bmp", DF);
//	imwrite("d:/test/DB.bmp", DB);


	//compute threshold
	int k;
	Mat MF = (trimap == 255);
	Mat MB = (trimap == 0);
	double dMin, dMax;
	Mat SE = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
	for (k = 1; k < 100; k++) {
		//SE = [0, 1, 0;1, 1, 1;0, 1, 0];
		//MF = imdilate(MF, SE);
		dilate(MF, MF, SE);
		minMaxLoc(MF & MB, &dMin, &dMax);
		if (dMax == 255) {
			break;
		}
	}
	int threshold = 20;
	threshold = min((k - 1) / 2, 20);

	////////////////////
	Mat EF = DF < threshold;
	Mat EB = DB < threshold;

	unsigned char* pTrimap = trimap.ptr<unsigned char>();
	unsigned char* pEF = EF.ptr<unsigned char>();
	unsigned char* pEB = EB.ptr<unsigned char>();

	for (int i = 0; i < inPut.cols* inPut.rows; i++) {
		if (pEF[i] > 0) pTrimap[i] = 255;
		if (pEB[i] > 0) pTrimap[i] = 0;
	}

	return trimap;
}

cv::Mat colSums(Mat src)
{
	Mat ret(src.rows, 1, CV_64FC1);
	double *pData = (double*)malloc(sizeof(double)*src.rows);
	for (int i = 0; i < src.rows; i++)
	{
		double sum = 0;
		for (int j = 0; j < src.cols; j++)
		{
			sum += src.ptr<double>(i)[j];
		}
		pData[i] = sum;
	}
	memcpy(ret.data, pData, sizeof(double)*src.rows);
	free(pData);
	return ret;
}

cv::Mat colSumsSQR(Mat src)
{
	Mat ret(src.rows, 1, CV_64FC1);
	double *pData = (double*)malloc(sizeof(double)*src.rows);

	for (int i = 0; i < src.rows; i++) {
		double sum = 0;
		for (int j = 0; j < src.cols; j++) {
			sum += src.ptr<double>(i)[j] * src.ptr<double>(i)[j];
		}
		pData[i] = sum;
	}
	memcpy(ret.data, pData, sizeof(double)*src.rows);
	free(pData);

	return ret;
}

cv::Mat sumDDDD(Mat src, Mat idx)
{
	Mat ret(idx.rows, 1, CV_64FC1);

	for (int i = 0; i < idx.rows; i++)
	{
		int index = idx.ptr<int>(i)[0] - 1;
		ret.ptr<double>(i)[0] = src.ptr<double>(index)[0];
	}
	return ret;
}

cv::Mat getsumD(Mat D, Mat IDX, int K)
{
	int layernum = 120;
	Mat sumD = colSums(D);

	for (int k = 0; k < layernum; k++) {
		Mat D_dot, D_reshape;
		// 		cv::transpose(D, D_dot);
		D_reshape = D.reshape(1, D.cols*D.rows);
		// 		D_reshape = D_dot.reshape(1, D_dot.cols*D_dot.rows);

		Mat IDX_dot, IDX_reshape;
		// 		cv::transpose(IDX, IDX_dot);
		// 		Mat B = IDX_dot.reshape(1, IDX_dot.cols*IDX_dot.rows);
		IDX_reshape = IDX.reshape(1, IDX.cols*IDX.rows);

		Mat tmp = sumDDDD(sumD, IDX_reshape);

		cv::add(D_reshape, tmp, sumD);

		Mat sumD_dot, sumD_dot_reshape, sumD_dot_reshape_dot;
		// 		cv::transpose(sumD, sumD_dot);
		sumD_dot_reshape = sumD.reshape(1, sumD.rows / K);
		//		cv::transpose(sumD_dot_reshape, sumD_dot_reshape_dot);
		sumD = colSums(sumD_dot_reshape);
	}

	double minVal = 0, maxVal = 0;
	minMaxLoc(sumD, &minVal, &maxVal);
	sumD = (sumD - minVal) / (maxVal - minVal);

	double stdVal = 0;
	Mat meanMat, stdMat;
	meanStdDev(sumD, meanMat, stdMat);
	stdVal = stdMat.ptr<double>(0)[0];
	sumD = sumD * 255 / stdVal;
	return sumD;
}

cv::Mat MakeIndArray(Mat trimap, Mat flag)
{
	int row = trimap.rows;
	int col = trimap.cols;

	double* pTmp = (double*)malloc(sizeof(double) * row * col);
	int nCnt = 0;

	for (int i = 0; i < col; i++) {
		for (int j = 0; j < row; j++) {
			if (!((trimap.ptr<unsigned char>(j)[i] == 255) ||
				(trimap.ptr<unsigned char>(j)[i] == 0) ||
				(flag.ptr<unsigned char>(j)[i] == 0)))
			{
				pTmp[nCnt] = i*row + j;
				nCnt++;
			}
			else {
				continue;
			}
		}
	}

	Mat ret(nCnt, 1, CV_64FC1);
	memcpy(ret.data, pTmp, sizeof(double)*nCnt);
	free(pTmp);

	return ret;
}

cv::Mat MakeIndexedMatByRow(Mat src, Mat idx)
{
	Mat ret(idx.cols * idx.rows, src.cols, CV_64FC1);
	for (int i = 0; i < ret.cols; i++) {
		for (int j = 0; j < ret.rows; j++) {
			int index = idx.ptr<unsigned int>(0)[j] - 1;
			double a = src.ptr<double>(index)[i];
			ret.ptr<double>(j)[i] = src.ptr<double>(index)[i];
		}
	}
	return ret;
}

// Multiply a matrix in row - index sparse storage arrays sa and ija by a vector x[1..n], giving a vector b[1..n].
void sprsax(double sa[], unsigned long ija[], double x[], double b[], unsigned long n)
{
	unsigned long i, k;
	if (ija[1] != n + 2) {
		printf("sprsax: mismatched vector and matrix\n");
	}
	for (i = 1; i <= n; i++) {
		b[i] = sa[i] * x[i]; //Start with diagonal term.
		for (k = ija[i]; k <= ija[i + 1] - 1; k++) { //Loop over off - diagonal terms.			
			b[i] += sa[k] * x[ija[k]];
		}
	}
}

#if 1
void MultiplySparse(unsigned char* pDiag, double* pMul, int nDim, double* pV, double* pI, double* pJ, int nCount)
{
	// 	if (ija[1] != n + 2) {
	// 		printf("sprsax: mismatched vector and matrix\n");
	// 	}
	int nIndex = 0;
	for (int i = 0; i < nCount; i++) {
		if (pV[i] != 0) {
			i = i;
		}
		else {
			continue;
		}
		if ((int)pI[i] - 1 == (int)pJ[i] - 1) {
			pMul[(int)pI[i] - 1] += pDiag[(int)pI[i] - 1] * (1 - pV[i]);
		}
		else {
			pMul[(int)pI[i] - 1] += pDiag[(int)pI[i] - 1] * (0 - pV[i]);
		}
	}
}
#else
void MultiplySparse(unsigned char* pDiag, double* pMul, int nDim, double* pV, double* pI, int* pJ, int nCount)
{
	// 	if (ija[1] != n + 2) {
	// 		printf("sprsax: mismatched vector and matrix\n");
	// 	}
	int nIndex = 0;
	for (int i = 0; i < nDim; i++) {
		// 		if (pDiag[i] == 0) {
		// 			pMul[i] = 0;
		// 			continue;
		// 		}

		while (pI[nIndex] == 1) {
			nIndex++;
		}



		while ((pI[nIndex] - 1) == i) {
			pMul[i] += pDiag[i] * (1 - pV[nIndex]);
			nIndex++;
		}
		if (nIndex == 21700) {
			nIndex = nIndex;
		}

		while (pI[nIndex] < i) {
			nIndex++;
		}
	}
}
#endif

double* GetGraph(Mat img, Mat trimap, Mat neighborsArray, int zz, Mat flag)
{
	int m = img.rows;
	int n = img.cols;
	int z = img.channels();
	int NN = m*n;
	int K = zz;
	Mat featherArray = img.clone();
	cv::transpose(featherArray, featherArray);

	Mat X = featherArray.reshape(1, NN);
	Mat w_array = Mat::zeros(K, NN, CV_64FC1);
	Mat i_array = Mat::ones(K, NN, CV_64FC1);
	Mat j_array = Mat::ones(K, NN, CV_64FC1);

	double tol = 1.000000000000000e-03;
	Mat indArray = MakeIndArray(trimap, flag);
	Mat neighborsIndex;
	//	cv::transpose(neighborsArray, neighborsIndex);
	Mat neighborsIndex_tmp = /*neighborsIndex*/neighborsArray.reshape(1, 1);
	Mat neighbors = MakeIndexedMatByRow(X, neighborsIndex_tmp);
	// 	cv::transpose(neighbors, neighbors);

	vector<Mat> vecMat_A;
	for (int i = 0; i < NN; i++) {
		Mat neighbor_piece(K, X.cols, CV_64FC1);
		Mat tmp0 = neighbors.rowRange(i*K, i*K + K);
		cv::transpose(tmp0, tmp0);
		tmp0.copyTo(neighbor_piece);

		Mat tmp(X.cols, 1, CV_64FC1);
		for (int ii = 0; ii < X.cols; ii++) {
			tmp.ptr<double>(ii)[0] = X.ptr<double>(i)[ii];
		}

		tmp = cv::repeat(tmp, 1, K);
		vecMat_A.push_back(neighbor_piece - tmp);
	}
	Mat w = Mat::zeros(K, 1, CV_64FC1);

	Mat indArray_dot;
	Mat onesK = Mat::ones(K, 1, CV_64FC1);
	Mat indArrayT = indArray.t();
	Mat tmp = onesK * indArrayT;
	for (int i = 0; i < indArray.rows; i++) {
		int idx = indArray.ptr<double>(i)[0];
		X = vecMat_A.at(idx);
		Mat X_dot;
		transpose(X, X_dot);
		Mat C = X_dot * X;
		Scalar sc = cv::sum(C.diag());
		double trace_C = sc[0];
		if (trace_C == 0) {
			w = Mat::ones(K, 1, CV_64FC1);
		}
		else {
			C = C + Mat::eye(K, K, CV_64FC1)*tol*trace_C;
			cv::solve(C, Mat::ones(K, 1, CV_64FC1), w);
		}

		double sum = cv::sum(w)[0];
		w = w / sum;

		Mat w_ddd = w_array.clone();
		w.copyTo(w_array.col(idx));

		tmp.col(i).copyTo(i_array.col(idx));

		Mat tmp1 = neighborsArray.row(idx);
		transpose(tmp1, tmp1);
		tmp1.copyTo(j_array.col(idx));
	}

	i_array = i_array.reshape(1, i_array.rows*i_array.cols);
	j_array = j_array.reshape(1, j_array.rows*j_array.cols);
	Mat o_array = flag.reshape(1, flag.rows*flag.cols);

	w_array = w_array.reshape(1, w_array.rows*w_array.cols);

	int non_zeros = 0;
	for (int i = 0; i < w_array.rows; i++) {
		for (int j = 0; j < w_array.cols; j++) {
			if (w_array.ptr<double>(i)[j] != 0) {
				non_zeros++;
			}
		}
	}

	g_sa = dvector(1, non_zeros+w_array.rows*w_array.cols);
	g_ija = lvector(1, non_zeros+w_array.rows*w_array.cols);

	int* pI = ivector(1, i_array.rows*i_array.cols);
	int* pJ = ivector(1, j_array.rows*j_array.cols);

	double* pTempI = (double*)i_array.data;
	double* pTempJ = (double*)j_array.data;
	for (int i = 0; i < i_array.rows*i_array.cols; i++) {
		pI[i] = pTempI[i];
		pJ[i] = pTempJ[i];
	}

	double* pMul = dvector(1, w_array.rows*w_array.cols);
	sprsin_FromCordiToRow((double*)w_array.data, pI, pJ, non_zeros, flag.rows*flag.cols, g_sa, g_ija);
	
	
// 	o_array = flag(:);
// 	O = spdiags(o_array, 0, N, N);
// 	W = speye(N) - sparse(i_array, j_array, w_array, N, N);
// 	W = O*W;

	dsprsax(g_sa, g_ija, (double*)o_array.data, pMul, flag.rows*flag.cols);

	double* pMult = dvector(1, w_array.rows*w_array.cols);
	dsprstx(pMul, g_ija, pMul, pMult, flag.rows*flag.cols);
	memcpy(g_sa, pMult, w_array.rows*w_array.cols*sizeof(double));

// 	double* pMul = (double*)calloc(flag.rows*flag.cols, sizeof(double));
// 	MultiplySparse(o_array.data, pMul, flag.rows*flag.cols, (double*)w_array.data, (double*)i_array.data, (double*)j_array.data, i_array.rows*i_array.cols);

	return pMult;
}

cv::Mat selectPix(cv::Mat imag, cv::Mat D, double selectedRatio)
{
	Mat IX, ret;
	Mat D_Sum = colSumsSQR(D);
	cv::sortIdx(D_Sum, IX, CV_SORT_EVERY_COLUMN | CV_SORT_ASCENDING);

	return ret;
}

Mat GetThreeGraph(Mat img, Mat trimap, int K0, int K1, int K2, int K3)
{
	Mat ret;

	int m = img.rows;
	int n = img.cols;
	int z = img.channels();
	int NN = m*n;

	//Computing W1
	Mat matX_1(1, n, CV_64FC1), matY_1(m, 1, CV_64FC1);
	for (int i = 0; i < n; i++) {
		matX_1.ptr<double>(0)[i] = i + 1;
	}
	for (int i = 0; i < m; i++) {
		matY_1.ptr<double>(i)[0] = i + 1;
	}
	Mat matX = cv::repeat(matX_1, m, 1);
	Mat matY = cv::repeat(matY_1, 1, n);

	matX = matX / 12;
	matY = matY / 12;

	Mat bgra[5];
	split(img, bgra);
	Mat tmp = bgra[0];
	bgra[0] = bgra[2];
	bgra[2] = tmp;

	matY.copyTo(bgra[3]);
	matX.copyTo(bgra[4]);

	Mat X_mat, X_dot_mat;
	cv::merge(bgra, 5, X_mat);

	X_mat = matlab_reshape(X_mat, NN, 5, 1);
	//	X_mat = X_mat.reshape(1, NN);
	// 	cv::transpose(X_mat, X_dot_mat);


	int K = K0;
	VlKDForest *pforest = vl_kdtreebuild((double*)X_mat.data, 5, NN);
	double* pData = (double*)pforest->data;
	VLKDTREEQUERY_RESULT *pQueryResult = vl_kdtreequery(pforest, (double*)X_mat.data, K + 1, 5, NN);

	Mat IDX(K + 1, NN, CV_32SC1), D(K + 1, NN, CV_64FC1);
	memcpy(IDX.data, pQueryResult->pIDX, sizeof(unsigned int) * NN * (K + 1));
	memcpy(D.data, pQueryResult->pD, sizeof(double) * NN * (K + 1));
	free(pQueryResult->pD);
	free(pQueryResult->pIDX);
	free(pQueryResult);

	// 	cv::transpose(IDX, IDX);
	// 	cv::transpose(D, D);
	Mat IDX_reshape = IDX.reshape(IDX.channels(), IDX.cols);
	Mat D_reshape = D.reshape(D.channels(), D.cols);

	Mat IDX1 = IDX_reshape.rowRange(0, IDX_reshape.rows).colRange(1, IDX_reshape.cols);
	Mat D1 = D_reshape.rowRange(0, D_reshape.rows).colRange(1, D_reshape.cols);
	// 	imwrite("d:/xxx.bmp", D);
	Mat IDX12 = IDX1.clone();
	Mat D12 = D1.clone();
	Mat sumD = getsumD(D12, IDX12, K);


	cv::rotate(X_mat, X_mat, ROTATE_90_CLOCKWISE);
	cv::rotate(sumD, sumD, ROTATE_90_CLOCKWISE);
	X_mat.push_back(sumD);
	cv::rotate(X_mat, X_mat, ROTATE_90_COUNTERCLOCKWISE);

	K = K1;
	// 	cv::transpose(X_mat, X_dot_mat);

	pforest = vl_kdtreebuild((double*)X_mat.data, X_mat.cols, X_mat.rows);
	pQueryResult = vl_kdtreequery(pforest, (double*)X_mat.data, K + 1, X_mat.cols, X_mat.rows);

	IDX.release(); D.release();
	IDX.create(K + 1, X_mat.rows, CV_32SC1);
	D.create(K + 1, X_mat.rows, CV_64FC1);
	memcpy(IDX.data, pQueryResult->pIDX, sizeof(unsigned int) * NN * (K + 1));
	memcpy(D.data, pQueryResult->pD, sizeof(double) * NN * (K + 1));
	free(pQueryResult->pD);
	free(pQueryResult->pIDX);
	free(pQueryResult);

	// 	cv::transpose(IDX, IDX);
	// 	cv::transpose(D, D);
	// 	IDX = IDX.colRange(1, IDX.cols);
	// 	D = D.colRange(1, D.cols);
	IDX_reshape = IDX.reshape(IDX.channels(), IDX.cols);
	D_reshape = D.reshape(D.channels(), D.cols);

	IDX1 = IDX_reshape.rowRange(0, IDX_reshape.rows).colRange(1, IDX_reshape.cols);
	D1 = D_reshape.rowRange(0, D_reshape.rows).colRange(1, D_reshape.cols);
	// 	imwrite("d:/xxx.bmp", D);
	IDX12 = IDX1.clone();
	D12 = D1.clone();



	Mat IX;
	Mat D_Sum = colSumsSQR(D12);
	cv::sortIdx(D_Sum, IX, CV_SORT_EVERY_COLUMN | CV_SORT_ASCENDING);

	// 	cv::sortIdx(colSums(D12 ^ 2), IX, CV_SORT_EVERY_COLUMN | CV_SORT_ASCENDING);
	IX = IX.rowRange(0, floor(0.9 * NN));
	Mat flag = Mat::zeros(m, n, CV_64FC1/*CV_8UC1*/);
	Mat flag2 = Mat::zeros(n, m, CV_64FC1/*CV_8UC1*/);
	for (int i = 0; i < IX.rows; i++) {
		int idx = IX.ptr<unsigned int>(i)[0];
		int nrow = idx / flag.cols;
		int ncol = idx % flag.cols;
		flag.ptr<double/*unsigned char*/>(nrow)[ncol] = 1;

		nrow = idx / flag2.cols;
		ncol = idx % flag2.cols;
		flag2.ptr<double/*unsigned char*/>(nrow)[ncol] = 1;
	}
	cv::transpose(flag2, flag2);

	Mat imgX;
	cv::merge(bgra, 3, imgX);

	double* pW1 = GetGraph(imgX, trimap, IDX12, K, flag2);

#if 0
	//Computing W2 
	sumD = getsumD(D12, IDX12, K);

	K = K2;
	// 	cv::transpose(X_mat, X_dot_mat);

	pforest = vl_kdtreebuild((double*)X_mat.data, X_mat.cols, X_mat.rows);
	pQueryResult = vl_kdtreequery(pforest, (double*)X_mat.data, K + 1, X_mat.cols, X_mat.rows);

	IDX.release(); D.release();
	IDX.create(K + 1, X_mat.rows, CV_32SC1);
	D.create(K + 1, X_mat.rows, CV_64FC1);
	memcpy(IDX.data, pQueryResult->pIDX, sizeof(unsigned int) * NN * (K + 1));
	memcpy(D.data, pQueryResult->pD, sizeof(double) * NN * (K + 1));
	free(pQueryResult->pD);
	free(pQueryResult->pIDX);
	free(pQueryResult);

	flag = selectPix(img, D, 0.1);
	double* pW2 = GetGraph(imgX, trimap, IDX12, K, flag2);

	//Computing W3 
	sumD = getsumD(D12, IDX12, K);

	K = K3;
	// 	cv::transpose(X_mat, X_dot_mat);

	pforest = vl_kdtreebuild((double*)X_mat.data, X_mat.cols, X_mat.rows);
	pQueryResult = vl_kdtreequery(pforest, (double*)X_mat.data, K + 1, X_mat.cols, X_mat.rows);

	IDX.release(); D.release();
	IDX.create(K + 1, X_mat.rows, CV_32SC1);
	D.create(K + 1, X_mat.rows, CV_64FC1);
	memcpy(IDX.data, pQueryResult->pIDX, sizeof(unsigned int) * NN * (K + 1));
	memcpy(D.data, pQueryResult->pD, sizeof(double) * NN * (K + 1));
	free(pQueryResult->pD);
	free(pQueryResult->pIDX);
	free(pQueryResult);

	flag = selectPix(img, D, 0.1);
	double* pW3 = GetGraph(imgX, trimap, IDX12, K, flag2);

	//L1 = 0.1*(W3'*W3) + 0.3*(W2'*W2) + W1'*W1;
#endif

	return ret;
}

cv::Mat getRowMeanMat(cv::Mat src)
{
	Mat ret(1, src.cols, CV_64FC1);
	double sum = 0;
	for (int i = 0; i < src.cols; i++)
	{
		sum = 0;
		for (int j = 0; j < src.rows; j++)
		{
			sum += src.ptr<double>(j)[i];
		}
		ret.ptr<double>(0)[i] = sum / src.cols;
	}
	return ret;
}

void copyMatByIndex(cv::Mat &src, cv::Mat &copyData, int startIndex)
{
	memcpy(src.ptr<double>() + startIndex, copyData.ptr<double>(), copyData.cols*copyData.rows*sizeof(double));
}

cv::Mat fspecial(int WinSize, double sigma)
{
	// I wrote this only for square kernels as I have no need for kernels that aren't square
	cv::Mat xx(WinSize, WinSize, CV_64F);
	for (int i = 0; i < WinSize; i++) {
		for (int j = 0; j < WinSize; j++) {
			xx.at<double>(j, i) = (i - (WinSize - 1) / 2)*(i - (WinSize - 1) / 2);
		}
	}
	cv::Mat yy;
	cv::transpose(xx, yy);
	cv::Mat arg = -(xx + yy) / (2 * pow(sigma, 2));
	cv::Mat h(WinSize, WinSize, CV_64F);
	for (int i = 0; i < WinSize; i++) {
		for (int j = 0; j < WinSize; j++) {
			h.at<double>(j, i) = pow(exp(1), (arg.at<double>(j, i)));
		}
	}
	double minimalVal, maximalVal;
	minMaxLoc(h, &minimalVal, &maximalVal);
	cv::Mat tempMask = (h > DBL_EPSILON*maximalVal) / 255;
	tempMask.convertTo(tempMask, h.type());
	cv::multiply(tempMask, h, h);

	if (cv::sum(h)[0] != 0) { h = h / cv::sum(h)[0]; }

	// 	cv::Mat h1 = (xx + yy - 2 * (pow(sigma, 2))) / (pow(sigma, 4));
	// 	cv::multiply(h, h1, h1);
	// 	h = h1 - cv::sum(h1)[0] / (WinSize*WinSize);
	return h;
}

// return NxN (square kernel) of Laplacian of Gaussian as is returned by     Matlab's: fspecial(Winsize,sigma)
cv::Mat fspecialLoG(int WinSize, double sigma)
{
	// I wrote this only for square kernels as I have no need for kernels that aren't square
	cv::Mat xx(WinSize, WinSize, CV_64F);
	for (int i = 0; i < WinSize; i++) {
		for (int j = 0; j < WinSize; j++) {
			xx.at<double>(j, i) = (i - (WinSize - 1) / 2)*(i - (WinSize - 1) / 2);
		}
	}
	cv::Mat yy;
	cv::transpose(xx, yy);
	cv::Mat arg = -(xx + yy) / (2 * pow(sigma, 2));
	cv::Mat h(WinSize, WinSize, CV_64F);
	for (int i = 0; i < WinSize; i++) {
		for (int j = 0; j < WinSize; j++) {
			h.at<double>(j, i) = pow(exp(1), (arg.at<double>(j, i)));
		}
	}
	double minimalVal, maximalVal;
	minMaxLoc(h, &minimalVal, &maximalVal);
	cv::Mat tempMask = (h > DBL_EPSILON*maximalVal) / 255;
	tempMask.convertTo(tempMask, h.type());
	cv::multiply(tempMask, h, h);

	if (cv::sum(h)[0] != 0) { h = h / cv::sum(h)[0]; }

	cv::Mat h1 = (xx + yy - 2 * (pow(sigma, 2))) / (pow(sigma, 4));
	cv::multiply(h, h1, h1);
	h = h1 - cv::sum(h1)[0] / (WinSize*WinSize);
	return h;
}

int bin_erode(BYTE *im, int width, int height)
{
	int i, j, i1, j1, adr, st_x, st_y, end_x, end_y;

	if (im == 0) return 0;
	BYTE *im_tmp = (BYTE *)calloc(width * height, sizeof(BYTE));
	memset(im_tmp, 0, width * height * sizeof(BYTE));
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			adr = i * width + j;
			if (im[adr] == 0) {
				continue;
			}
			st_x = max(0, j - 1); st_y = max(0, i - 1);
			end_x = min(width - 1, j + 1); end_y = min(height - 1, i + 1);
			for (i1 = st_y; i1 <= end_y; i1++) {
				for (j1 = st_x; j1 <= end_x; j1++) {
					if (im[i1 * width + j1] == 0) {
						goto TT;
					}
				}
			}
			im_tmp[adr] = 1;
		TT:         j = j;
		}
	}
	memcpy(im, im_tmp, width * height * sizeof(BYTE));
	SafeMemFree(im_tmp);

	return 1;
}

void GetMean(double **Mat, int row, int col, double **Mean)
{
	int i, j;
	double s = 0;

	for (j = 1; j <= col; j++) {
		s = 0;
		for (i = 1; i <= row; i++) {
			s += Mat[i][j];
		}
		Mean[j][1] = s / row;
	}
}

cv::Mat getColorLineLaplace(cv::Mat img_input, cv::Mat trimap)
{
	double epsilon = 0.0000001;
	int win_size = 1;
	img_input = img_input / 255;
	int neb_size = (win_size * 2 + 1) * (win_size * 2 + 1);
	int h = img_input.rows;
	int w = img_input.cols;
	int c = img_input.channels();
	int img_size = w * h;

	cv::Mat consts((trimap == 0 | trimap == 255));
	cv::Mat kernel = Mat::ones(win_size * 2 + 1, win_size * 2 + 1, CV_64F);

	cv::erode(consts, consts, kernel);

	cv::Mat indsM(h, w, CV_64FC1);
	int idx = 0;
	for (int i = 0; i < w; i++)
		for (int j = 0; j < h; j++)
			indsM.ptr<double>(j)[i] = idx++;

	//	tlen = sum(sum(1 - consts(win_size + 1:end - win_size, win_size + 1 : end - win_size)))*(neb_size ^ 2);
	Mat constsTmp = consts.colRange(win_size, w - win_size);
	constsTmp = constsTmp.rowRange(win_size, h - win_size);
	Scalar sc = cv::sum(1 - constsTmp);
	double tlen = sc[0];
	tlen = tlen * (neb_size * neb_size);
	///////////////////////////////////////////////

	Mat row_inds = Mat::zeros(tlen, 1, CV_64FC1);
	Mat col_inds = Mat::zeros(tlen, 1, CV_64FC1);
	Mat vals = Mat::zeros(tlen, 1, CV_64FC1);
	int len = 0;

	for (int j = 0 + win_size; j < w - win_size; j++)
	{
		for (int i = 0 + win_size; i < h - win_size; i++)
		{
			if (consts.ptr<unsigned char>(i)[j] > 0)
				continue;
			Mat win_inds_ = indsM.colRange(j - win_size, j + win_size + 1);
			win_inds_ = win_inds_.rowRange(i - win_size, i + win_size + 1);
			Mat win_inds;
			win_inds_.copyTo(win_inds);
			win_inds = win_inds.reshape(1, win_inds.rows*win_inds.cols);
			Mat win_inds_dot;
			transpose(win_inds, win_inds_dot);

			Mat winI = img_input.colRange(j - win_size, j + win_size + 1);
			winI = winI.rowRange(i - win_size, i + win_size + 1);
			Mat winI_;
			winI.copyTo(winI_);
			winI = winI_.reshape(1, neb_size);
			winI.convertTo(winI, CV_64FC1);
			Mat winMu = getRowMeanMat(winI);

			transpose(winMu, winMu);
			//win_var=inv(winI'*winI/neb_size-win_mu*win_mu' +epsilon/neb_size*eye(c));
			Mat winI_dot;
			transpose(winI, winI_dot);
			Mat tmpA, tmpB, tmpC;
			tmpC = Mat::eye(c, c, CV_64FC1);
			tmpA = winI_dot * winI;
			Mat winMu_dot;
			transpose(winMu, winMu_dot);
			tmpB = winMu * winMu_dot;
			Mat win_var = tmpA - tmpB + (epsilon / neb_size)*tmpC;
			cv::invert(win_var, win_var);
			//////////////////////////////////////////winI=winI-repmat(win_mu',neb_size,1);
			tmpC = cv::repeat(winMu_dot, 9, 1);
			winI = winI - tmpC;
			Mat tvars = (1 + winI*win_var*winI_dot) / neb_size;
			/////////////////row_inds(1+len:neb_size^2+len)=reshape(repmat(win_inds,1,neb_size),neb_size ^ 2, 1);
			tmpA = cv::repeat(win_inds, 1, neb_size);
			tmpA = tmpA.reshape(1, neb_size*neb_size);
			copyMatByIndex(row_inds, tmpA, len);
			tmpB = cv::repeat(win_inds_dot, neb_size, 1);
			tmpB = tmpB.reshape(1, neb_size*neb_size);
			copyMatByIndex(col_inds, tmpB, len);
			copyMatByIndex(vals, tvars, len);
			len += neb_size * neb_size;
		}
	}

	vals = vals.rowRange(0, len);
	row_inds = row_inds.rowRange(0, len);
	col_inds = col_inds.rowRange(0, len);

	//A = sparse(row_inds, col_inds, vals, img_size, img_size);

	//sumA = sum(A, 2);
	//A = spdiags(sumA(:), 0, img_size, img_size) - A;
	return trimap;
}

int getColorLineLaplace(float *pfImage, BYTE *consts, int width, int height, int BitCount, int &Num)
{
	int i, j, adr, adr1, k, l, m, c, n, image_size = width * height;
	int win_size = 1, neb_size, tlen, len;
	double epsilon = 0.0000001;
	int **indsM = imatrix(1, height, 1, width);

	neb_size = (int)SQR(win_size * 2 + 1);
	double **tvals = dmatrix(1, neb_size, 1, neb_size);
	double **TmpMat1 = dmatrix(1, neb_size, 1, neb_size);
	double **TmpMat2 = dmatrix(1, neb_size, 1, neb_size);
	double **win_var = dmatrix(1, 3, 1, 3);
	double **win_mu = dmatrix(1, 3, 1, 1);

	int *Ind_Tbl_row = ivector(1, SQR(neb_size));
	int *Ind_Tbl_col = ivector(1, SQR(neb_size));

	for (i = 1; i <= SQR(neb_size); i++) {
		adr = i % neb_size;
		if (adr == 0) {
			adr = neb_size;
		}
		Ind_Tbl_row[i] = adr;
		Ind_Tbl_col[i] = (i - 1) / neb_size + 1;
	}

	double ttt = 0;
	n = height;
	m = width;
	bin_erode(consts, width, height);
	Mat erodeimg(height, width, CV_8UC1, consts);
	//imwrite("d:/test/erodeimg.bmp", erodeimg);

	for (i = 1; i <= height; i++) {
		for (j = 1; j <= width; j++) {
			indsM[i][j] = (j - 1) * height + i;
		}
	}
	tlen = 0;
	for (i = win_size; i < height - win_size; i++) {
		for (j = win_size; j < width - win_size; j++) {
			tlen += 1 - consts[i * width + j];
		}
	}
	tlen *= SQR(neb_size);

	if (tlen <= 0) {
		free_imatrix(indsM, 1, height, 1, width);
		free_dmatrix(tvals, 1, neb_size, 1, neb_size);
		free_dmatrix(TmpMat1, 1, neb_size, 1, neb_size);
		free_dmatrix(TmpMat2, 1, neb_size, 1, neb_size);
		free_ivector(Ind_Tbl_row, 1, neb_size);
		free_ivector(Ind_Tbl_col, 1, neb_size);
		free_dmatrix(win_var, 1, 3, 1, 3);
		free_dmatrix(win_mu, 1, 3, 1, 1);

		return -1;
	}

	int *win_inds = ivector(1, neb_size);
	double* tvals_1 = dvector(1, SQR(neb_size));
	double **winI = dmatrix(1, neb_size, 1, 3);

	len = 0;
	int *Num_PerRow = (int *)calloc((image_size + 1), sizeof(int));
	ItemSort **Cols_PerRow = (ItemSort **)malloc((image_size + 1) * sizeof(ItemSort *));
	double **Dta_PerRow = (double **)malloc((image_size + 1) * sizeof(double *));
	for (i = 0; i <= image_size; i++) {
		Cols_PerRow[i] = (ItemSort *)malloc(90 * sizeof(ItemSort));
		Dta_PerRow[i] = (double *)malloc(90 * sizeof(double));
	}
	for (j = 1 + win_size; j <= width - win_size; j++) {
		for (i = win_size + 1; i <= height - win_size; i++) {
			if (consts[(i - 1) * width + j - 1] == 1) {
				continue;
			}
			adr = 1;
			for (l = j - win_size; l <= j + win_size; l++) {
				for (k = i - win_size; k <= i + win_size; k++) {
					win_inds[adr] = indsM[k][l];
					adr++;
				}
			}
			adr = 1;
			for (l = j - win_size; l <= j + win_size; l++) {
				for (k = i - win_size; k <= i + win_size; k++) {
					adr1 = (k - 1) * width + l - 1;
					for (c = 1; c <= BitCount; c++) {
						winI[adr][c] = pfImage[BitCount * adr1 + c - 1];
					}
					adr++;
				}
			}
			GetMean(winI, neb_size, BitCount, win_mu);
			TranseMatMul(winI, winI, BitCount, neb_size, BitCount, TmpMat1);
			MatMulTranse(win_mu, win_mu, BitCount, 1, BitCount, TmpMat2);

			for (k = 1; k <= BitCount; k++) {
				for (l = 1; l <= BitCount; l++) {
					if (k == l) {
						win_var[k][l] = (double)TmpMat1[k][l] / neb_size - (double)TmpMat2[k][l] + (double)epsilon / neb_size;
					}
					else {
						win_var[k][l] = (double)TmpMat1[k][l] / neb_size - (double)TmpMat2[k][l];
					}
				}
			}
			InvMatrix(win_var, BitCount, TmpMat1);

			for (k = 1; k <= neb_size; k++) {
				for (c = 1; c <= BitCount; c++) {
					winI[k][c] -= win_mu[c][1];
				}
			}
			Mat_Mul(winI, TmpMat1, neb_size, BitCount, BitCount, TmpMat2);
			MatMulTranse(TmpMat2, winI, neb_size, BitCount, neb_size, tvals);
			adr = 1;
			for (k = 1; k <= neb_size; k++) {
				for (l = 1; l <= neb_size; l++) {
					tvals_1[adr] = (tvals[k][l] + 1) / neb_size;
					adr++;
				}
			}
			for (k = 1; k <= SQR(neb_size); k++) {
				int row = win_inds[Ind_Tbl_row[k]];
				Num_PerRow[row]++;
				Cols_PerRow[row][Num_PerRow[row]].colIndex = win_inds[Ind_Tbl_col[k]];
				Cols_PerRow[row][Num_PerRow[row]].oldIndex = Num_PerRow[row];
				Dta_PerRow[row][Num_PerRow[row]] = tvals_1[k];
			}
			len += SQR(neb_size);
		}
	}
	// 	int yyy = 0;
	// 	for (i = 0; i <= image_size; i++) {
	// 		yyy = max(yyy, Num_PerRow[i]);
	// 	}
	Num = tlen + image_size;
	g_sa = dvector(1, Num);
	g_ija = lvector(1, Num);
	sprsin_FromCordiToRow(Dta_PerRow, Cols_PerRow, Num_PerRow, image_size, g_sa, g_ija);
	double *Diag = dvector(1, image_size);
	sprsum_col(g_sa, g_ija, Diag, image_size);
	for (i = 1; i <= image_size; i++) {
		g_sa[i] = Diag[i] - g_sa[i];
	}
	for (i = image_size + 1; i <= g_ija[image_size + 1] - 1; i++) {
		g_sa[i] = -g_sa[i];
	}

	// 	for (int i = 0; i < height; i++) {
	// 	 	for (int j = 0; j < width; j++)	{
	// 	 		double a;
	// 	 		a = consts[i * width + j];
	// 	 		fwrite(&a, 1, sizeof(double), fp);
	// 	 	}
	// 	}	
	// 	fclose(fp);


	free_imatrix(indsM, 1, height, 1, width);
	free_ivector(win_inds, 1, neb_size);
	free_dmatrix(winI, 1, neb_size, 1, 3);
	free_dmatrix(tvals, 1, neb_size, 1, neb_size);
	free_dmatrix(TmpMat1, 1, neb_size, 1, neb_size);
	free_dmatrix(TmpMat2, 1, neb_size, 1, neb_size);
	free_ivector(Ind_Tbl_row, 1, neb_size);
	free_ivector(Ind_Tbl_col, 1, neb_size);
	free_dmatrix(win_var, 1, 3, 1, 3);
	free_dmatrix(win_mu, 1, 3, 1, 1);
	free_dvector(tvals_1, 1, neb_size);
	free_dvector(Diag, 1, image_size);
	for (i = 0; i <= image_size; i++) {
		SafeMemFree(Cols_PerRow[i]);
		SafeMemFree(Dta_PerRow[i]);
	}
	SafeMemFree(Cols_PerRow);
	SafeMemFree(Dta_PerRow);
	SafeMemFree(Num_PerRow);

	return 0;
}

int RunMattingThreeLayer(Mat img_input, Mat &trimap, float *alpha, BYTE *OutImage, int flag)
{
	// 	trimap.convertTo(trimap, CV_8UC1);

	int nInputChannels = img_input.channels();
	int nWid = img_input.cols; int nHei = img_input.rows;
	Size dstSize(img_input.cols, img_input.rows);
	Mat I;// = convertTypeAndSize(img_input, CV_64FC3, dstSize);
	img_input.convertTo(I, CV_64FC3);
	if (I.type() != CV_64FC3) {
		return 1;
	}

	int WinSize(9);
	double sigma(0.5); // can be changed to other odd-sized WinSize and different sigma values
	cv::Mat h = fspecial(WinSize, sigma);
	Mat img_fil;
	filter2D(I, img_fil, I.depth(), h, cv::Point(-1, -1), 0.0, BORDER_REPLICATE);
	Mat trimap_new = labelExpansion(img_input, trimap);
	//imwrite("d:/le.bmp", trimap_new);

	int K0 = 12, K1 = 12, K2 = 7, K3 = 7;
	int lamda = 1000, delta = 7;
	I = I + (I - img_fil);

	//imwrite("d:/I.bmp", I);

	Mat L1 = GetThreeGraph(I, trimap_new, K0, K1, K2, K3);
	// 	Mat L2 = getColorLineLaplace(I, trimap_new);
	// 	Mat L = L1 + delta* L2;
	// 
	// 	Mat M(trimap_new == 0 | trimap_new == 255);
	// 	M.convertTo(M, CV_64FC1);
	// 	M = M / 255;
	// 	Mat G(trimap_new == 255);
	// 	G.convertTo(G, CV_64FC1);
	// 	G = G / 255;
	// 	G = G.reshape(1, G.rows * G.cols);
	// 
	// 	double tol = 1.000000000000000e-07;
	// 	int maxit = 2000;

}

int GetAlpha(Mat* image, Mat &mask, int flag, float *pfImage, BYTE *consts_map, BYTE *consts_vals, int width, int height, int BitCount, float *alpha, float *InitVal = NULL)
{
	int    i, j, adr, ind, Num;
	int    img_size = width * height;
	double lambda = 10;
	float  *Diag = fvector(1, img_size);
	float  *Bias = fvector(1, img_size);

	adr = 1;
	for (j = 1; j <= width; j++) {
		for (i = 1; i <= height; i++) {
			ind = (i - 1) * width + j - 1;
			Diag[adr] = consts_map[ind];
			Bias[adr] = lambda * consts_map[ind] * consts_vals[ind];
			adr++;
		}
	}

// 	RunMattingThreeLayer(*image, mask, alpha, OutImage, flag);
	if (getColorLineLaplace(pfImage, consts_map, width, height, BitCount, Num) < 0) {
		free_vector(Diag, 1, img_size);
		free_vector(Bias, 1, img_size);
		return -1;
	}

	gf_sa = fvector(1, Num/*width*height*/);
	for (i = 1; i <= img_size; i++) {
		g_sa[i] = g_sa[i] + lambda * Diag[i];
	}
	for (i = 1; i <= Num/*width*height*/; i++) {
		gf_sa[i] = (float)g_sa[i];
	}
	int itmax = 1000, iter = 0, itol = 1;
	float ferror = 0, ftol = 0.0001f;

	if (InitVal == NULL) {
		memset(Diag + 1, 0, img_size * sizeof(float));
	} else {
		memcpy(Diag + 1, InitVal, img_size * sizeof(float));
	}

	ConjugateGradient_DiagPreco(img_size, Bias, Diag, ftol, itmax, &iter, &ferror);

	adr = 0; int tmp = 0;
	for (i = 1; i <= height; i++) {
		for (j = 1; j <= width; j++) {
			int ind = (j - 1) * height + i - 1 + 1;
			if (Diag[ind] < 0.02) {
				Diag[ind] = 0;
			}
			if (Diag[ind] > 1) {
				Diag[ind] = 1;
			}
			alpha[adr] = Diag[ind];
			adr++;
		}
	}

	free_dvector(g_sa, 1, Num);
	free_lvector(g_ija, 1, Num);
	free_vector(gf_sa, 1, Num);
	free_vector(Diag, 1, img_size);
	free_vector(Bias, 1, img_size);

	return 0;
}

int Matting(Mat* image, Mat &mask, float *alpha, int flag)
{
	int i, j, k, adr, bit_count = image->channels(), ind;
	int cx = image->cols;
	int cy = image->rows;

	float* pfImage= (float*)malloc(cx * cy * bit_count * sizeof(float));
	float* pfInit = (float*)malloc(cx * cy * bit_count * sizeof(float));

	BYTE* consts_map = (BYTE*)calloc(cx * cy, sizeof(BYTE));
	BYTE* consts_vals= (BYTE*)calloc(cx * cy, sizeof(BYTE));

	adr = 0; ind = 0;
	for (i = 0; i < cy; i++) {
		for (j = 0; j < cx; j++) {
			Vec3b bbb = image->at<Vec3b>(i, j);
			for (k = 0; k < bit_count; k++) {
				pfImage[adr + k] = (float)bbb[2 - k] / 255;
			}

			BYTE val = mask.at<char>(i, j);
			if (val == 0) {
				val = GC_BGD;
			} else if (val == 128) {
				val = GC_PR_BGD;
			} else if (val == 255) {
				val = GC_FGD;
			}

			if (val == GC_FGD || val == GC_BGD) {
				consts_map[ind] = 1;
				if (val == 1) {
					consts_vals[ind] = 1;
				}
			} else {
				consts_map[ind] = 0;
			}
			if (flag == 0) {
				if (val == GC_BGD || val == GC_PR_BGD) {
					pfInit[i * cx + j/*j * cy + i*/] = 0;
				}
				if (val == GC_FGD || val == GC_PR_FGD) {
					pfInit[i * cx + j/*j * cy + i*/] = 1;
				}
			} else {
				pfInit[i * cx + j/*j * cy + i*/] = alpha[i * cx + j];
			}
			adr += bit_count;
			ind++;
		}
	}

	int ch = mask.channels();
	Mat consts_map0(cy, cx, CV_8UC1, consts_map);
//	imwrite("d:/test/maskconsts_map.bmp", mask);
//	imwrite("d:/test/consts_map.bmp", consts_map0);

	int rgb[3] = { 255, 255, 255 };
	if (GetAlpha(image, mask, flag, pfImage, consts_map, consts_vals, cx, cy, bit_count, alpha, pfInit) < 0) {
		SafeMemFree(consts_map);
		SafeMemFree(consts_vals);
		SafeMemFree(pfImage);
		SafeMemFree(pfInit);
		
		return -1;
	}

	float alpha1 = 0;
	adr = 0;
	for (i = 0; i < cy; i++) {
		for (j = 0; j < cx; j++) {
			alpha1 = alpha[adr];
			BYTE val = mask.at<char>(i, j);
			if (val != GC_FGD && val != GC_BGD) {
				if (alpha1 > 0.2) {
					mask.at<char>(i, j) = GC_PR_FGD;
				}
				if (alpha1 < 0.1) {
					mask.at<char>(i, j) = GC_PR_BGD;
				}
			}
			adr++;
		}
	}

//	Mat outimg(cy, cx, CV_8UC3, OutImage);
//	imwrite("d:/test/mat0.bmp", outimg);

	SafeMemFree(consts_map);
	SafeMemFree(consts_vals);
	SafeMemFree(pfImage);
	SafeMemFree(pfInit);

	return 0;
}
