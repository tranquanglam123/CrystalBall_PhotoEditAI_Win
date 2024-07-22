// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_ALPHAMAT_INTRAU_H__
#define __OPENCV_ALPHAMAT_INTRAU_H__

namespace cv { namespace alphamat {

const int ALPHAMAT_DIM = 5;  // dimension of feature vectors

using namespace Eigen;
using namespace nanoflann;

typedef std::vector<std::vector<double>> my_vector_of_vectors_t_double;
typedef std::vector<std::vector<float>> my_vector_of_vectors_t_float;

int findColMajorInd(int rowMajorInd, int nRows, int nCols);

void UU_double(Mat& image, Mat& tmap, SparseMatrix<double>& Wuu, SparseMatrix<double>& Duu);
void UU_float(Mat& image, Mat& tmap, SparseMatrix<float>& Wuu, SparseMatrix<float>& Duu);

}}  // namespace

#endif
