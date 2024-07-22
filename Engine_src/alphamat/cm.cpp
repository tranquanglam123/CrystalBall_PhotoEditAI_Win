// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "intraU.hpp"
#include "cm.hpp"

namespace cv { namespace alphamat {

static
void generateFVectorCMDouble(my_vector_of_vectors_t_double& samples, Mat& img)
{
    int nRows = img.rows;
    int nCols = img.cols;

    samples.resize(nRows * nCols);

    int i, j;

    for (i = 0; i < nRows; ++i)
    {
        for (j = 0; j < nCols; ++j)
        {
            samples[i * nCols + j].resize(ALPHAMAT_DIM);
            samples[i * nCols + j][0] = img.at<cv::Vec3b>(i, j)[0] / 255.0;
            samples[i * nCols + j][1] = img.at<cv::Vec3b>(i, j)[1] / 255.0;
            samples[i * nCols + j][2] = img.at<cv::Vec3b>(i, j)[2] / 255.0;
            samples[i * nCols + j][3] = double(i) / nRows;
            samples[i * nCols + j][4] = double(j) / nCols;
        }
    }
}

static
void generateFVectorCMFloat(my_vector_of_vectors_t_float& samples, Mat& img)
{
	int nRows = img.rows;
	int nCols = img.cols;

	samples.resize(nRows * nCols);

	int i, j;

	for (i = 0; i < nRows; ++i)
	{
		for (j = 0; j < nCols; ++j)
		{
			samples[i * nCols + j].resize(ALPHAMAT_DIM);
			samples[i * nCols + j][0] = img.at<cv::Vec3b>(i, j)[0] / 255.0;
			samples[i * nCols + j][1] = img.at<cv::Vec3b>(i, j)[1] / 255.0;
			samples[i * nCols + j][2] = img.at<cv::Vec3b>(i, j)[2] / 255.0;
			samples[i * nCols + j][3] = float(i) / nRows;
			samples[i * nCols + j][4] = float(j) / nCols;
		}
	}
}

static
void kdtree_CM_double(Mat& img, my_vector_of_vectors_t_double& indm, my_vector_of_vectors_t_double& samples, std::unordered_set<int>& unk)
{
    // Generate feature vectors for intra U:
    generateFVectorCMDouble(samples, img);

    // Query point: same as samples from which KD tree is generated

    // construct a kd-tree index:
    // Dimensionality set at run-time (default: L2)
    // ------------------------------------------------------------
    typedef KDTreeVectorOfVectorsAdaptor<my_vector_of_vectors_t_double, double> my_kd_tree_t;
    my_kd_tree_t mat_index(ALPHAMAT_DIM /*dim*/, samples, 10 /* max leaf */);
    mat_index.index->buildIndex();

    // do a knn search with cm = 20
    const size_t num_results = 20 + 1;

    int N = unk.size();

    std::vector<size_t> ret_indexes(num_results);
    std::vector<double> out_dists_sqr(num_results);
    nanoflann::KNNResultSet<double> resultSet(num_results);

    indm.resize(N);
    int i = 0;
    for (std::unordered_set<int>::iterator it = unk.begin(); it != unk.end(); it++)
    {
        resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);
        mat_index.index->findNeighbors(resultSet, &samples[*it][0], nanoflann::SearchParams(10));

        indm[i].resize(num_results - 1);
        for (std::size_t j = 1; j < num_results; j++)
        {
            indm[i][j - 1] = ret_indexes[j];
        }
        i++;
    }
}

static
void kdtree_CM_float(Mat& img, my_vector_of_vectors_t_float& indm, my_vector_of_vectors_t_float& samples, std::unordered_set<int>& unk)
{
	// Generate feature vectors for intra U:
	generateFVectorCMFloat(samples, img);

	// Query point: same as samples from which KD tree is generated

	// construct a kd-tree index:
	// Dimensionality set at run-time (default: L2)
	// ------------------------------------------------------------
	typedef KDTreeVectorOfVectorsAdaptor<my_vector_of_vectors_t_float, float> my_kd_tree_t;
	my_kd_tree_t mat_index(ALPHAMAT_DIM /*dim*/, samples, 10 /* max leaf */);
	mat_index.index->buildIndex();

	// do a knn search with cm = 20
	const size_t num_results = 20 + 1;

	int N = unk.size();

	std::vector<size_t> ret_indexes(num_results);
	std::vector<float> out_dists_sqr(num_results);
	nanoflann::KNNResultSet<float> resultSet(num_results);

	indm.resize(N);
	int i = 0;
	for (std::unordered_set<int>::iterator it = unk.begin(); it != unk.end(); it++)
	{
		resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);
		mat_index.index->findNeighbors(resultSet, &samples[*it][0], nanoflann::SearchParams(10));

		indm[i].resize(num_results - 1);
		for (std::size_t j = 1; j < num_results; j++)
		{
			indm[i][j - 1] = ret_indexes[j];
		}
		i++;
	}
}

static
void lle_double(my_vector_of_vectors_t_double& indm, my_vector_of_vectors_t_double& samples, float eps, std::unordered_set<int>& unk,
        SparseMatrix<double>& Wcm, SparseMatrix<double>& Dcm, Mat& img)
{
    CV_LOG_INFO(NULL, "ALPHAMAT: In cm's lle function");
    int k = indm[0].size();  //number of neighbours that we are considering
    int n = indm.size();  //number of unknown pixels

    typedef Triplet<double> T;
    std::vector<T> triplets, td;

    my_vector_of_vectors_t_double wcm;
    wcm.resize(n);

    Mat C(20, 20, DataType<float>::type), rhs(20, 1, DataType<float>::type), Z(3, 20, DataType<float>::type), weights(20, 1, DataType<float>::type), pt(3, 1, DataType<float>::type);
    Mat ptDotN(20, 1, DataType<float>::type), imd(20, 1, DataType<float>::type);
    Mat Cones(20, 1, DataType<float>::type), Cinv(20, 1, DataType<float>::type);
    float alpha, beta, lagrangeMult;
    Cones = 1;

    C = 0;
    rhs = 1;

    int i, ind = 0;
    for (std::unordered_set<int>::iterator it = unk.begin(); it != unk.end(); it++)
    {
        // filling values in Z
        i = *it;

        int index_nbr;
        for (int j = 0; j < k; j++)
        {
            index_nbr = indm[ind][j];
            for (int p = 0; p < ALPHAMAT_DIM - 2; p++)
            {
                Z.at<float>(p, j) = samples[index_nbr][p];
            }
        }
        pt.at<float>(0, 0) = samples[i][0];
        pt.at<float>(1, 0) = samples[i][1];
        pt.at<float>(2, 0) = samples[i][2];

        C = Z.t() * Z;
        for (int p = 0; p < k; p++)
        {
            C.at<float>(p, p) += eps;
        }

        ptDotN = Z.t() * pt;
        solve(C, ptDotN, imd);
        alpha = 1 - cv::sum(imd)[0];
        solve(C, Cones, Cinv);
        beta = cv::sum(Cinv)[0];  //% sum of elements of inv(corr)
        lagrangeMult = alpha / beta;
        solve(C, ptDotN + lagrangeMult * Cones, weights);

        float sum = cv::sum(weights)[0];
        weights = weights / sum;

        int cMaj_i = findColMajorInd(i, img.rows, img.cols);

        for (int j = 0; j < k; j++)
        {
            int cMaj_ind_j = findColMajorInd(indm[ind][j], img.rows, img.cols);
            triplets.push_back(T(cMaj_i, cMaj_ind_j, weights.at<float>(j, 0)));
            td.push_back(T(cMaj_i, cMaj_i, weights.at<float>(j, 0)));
        }
        ind++;
    }

    Wcm.setFromTriplets(triplets.begin(), triplets.end());
    Dcm.setFromTriplets(td.begin(), td.end());
}

static
void lle_float(my_vector_of_vectors_t_float& indm, my_vector_of_vectors_t_float& samples, float eps, std::unordered_set<int>& unk,
	SparseMatrix<float>& Wcm, SparseMatrix<float>& Dcm, Mat& img)
{
	CV_LOG_INFO(NULL, "ALPHAMAT: In cm's lle function");
	int k = indm[0].size();  //number of neighbours that we are considering
	int n = indm.size();  //number of unknown pixels

	typedef Triplet<float> T;
	std::vector<T> triplets, td;

	my_vector_of_vectors_t_float wcm;
	wcm.resize(n);

	Mat C(20, 20, DataType<float>::type), rhs(20, 1, DataType<float>::type), Z(3, 20, DataType<float>::type), weights(20, 1, DataType<float>::type), pt(3, 1, DataType<float>::type);
	Mat ptDotN(20, 1, DataType<float>::type), imd(20, 1, DataType<float>::type);
	Mat Cones(20, 1, DataType<float>::type), Cinv(20, 1, DataType<float>::type);
	float alpha, beta, lagrangeMult;
	Cones = 1;

	C = 0;
	rhs = 1;

	int i, ind = 0;
	for (std::unordered_set<int>::iterator it = unk.begin(); it != unk.end(); it++)
	{
		// filling values in Z
		i = *it;

		int index_nbr;
		for (int j = 0; j < k; j++)
		{
			index_nbr = indm[ind][j];
			for (int p = 0; p < ALPHAMAT_DIM - 2; p++)
			{
				Z.at<float>(p, j) = samples[index_nbr][p];
			}
		}
		pt.at<float>(0, 0) = samples[i][0];
		pt.at<float>(1, 0) = samples[i][1];
		pt.at<float>(2, 0) = samples[i][2];

		C = Z.t() * Z;
		for (int p = 0; p < k; p++)
		{
			C.at<float>(p, p) += eps;
		}

		ptDotN = Z.t() * pt;
		solve(C, ptDotN, imd);
		alpha = 1 - cv::sum(imd)[0];
		solve(C, Cones, Cinv);
		beta = cv::sum(Cinv)[0];  //% sum of elements of inv(corr)
		lagrangeMult = alpha / beta;
		solve(C, ptDotN + lagrangeMult * Cones, weights);

		float sum = cv::sum(weights)[0];
		weights = weights / sum;

		int cMaj_i = findColMajorInd(i, img.rows, img.cols);

		for (int j = 0; j < k; j++)
		{
			int cMaj_ind_j = findColMajorInd(indm[ind][j], img.rows, img.cols);
			triplets.push_back(T(cMaj_i, cMaj_ind_j, weights.at<float>(j, 0)));
			td.push_back(T(cMaj_i, cMaj_i, weights.at<float>(j, 0)));
		}
		ind++;
	}

	Wcm.setFromTriplets(triplets.begin(), triplets.end());
	Dcm.setFromTriplets(td.begin(), td.end());
}

void cm_double(Mat& image, Mat& tmap, SparseMatrix<double>& Wcm, SparseMatrix<double>& Dcm)
{
    my_vector_of_vectors_t_double samples, indm, Euu;

    int i, j;
    std::unordered_set<int> unk;
    for (i = 0; i < tmap.rows; i++)
    {
        for (j = 0; j < tmap.cols; j++)
        {
            uchar pix = tmap.at<uchar>(i, j);
            if (pix == 128)
                unk.insert(i * tmap.cols + j);
        }
    }

    kdtree_CM_double(image, indm, samples, unk);
    float eps = 0.00001;
    lle_double(indm, samples, eps, unk, Wcm, Dcm, image);
    CV_LOG_INFO(NULL, "ALPHAMAT: cm DONE");
}

void cm_float(Mat& image, Mat& tmap, SparseMatrix<float>& Wcm, SparseMatrix<float>& Dcm)
{
	my_vector_of_vectors_t_float samples, indm, Euu;

	int i, j;
	std::unordered_set<int> unk;
	for (i = 0; i < tmap.rows; i++)
	{
		for (j = 0; j < tmap.cols; j++)
		{
			uchar pix = tmap.at<uchar>(i, j);
			if (pix == 128)
				unk.insert(i * tmap.cols + j);
		}
	}

	kdtree_CM_float(image, indm, samples, unk);
	float eps = 0.00001;
	lle_float(indm, samples, eps, unk, Wcm, Dcm, image);
	CV_LOG_INFO(NULL, "ALPHAMAT: cm DONE");
}

}}  // namespace cv::alphamat
