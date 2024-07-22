/** @internal
 ** @file     vl_kdetreebuild.c
 ** @brief    vl_KDForestbuild MEX implementation
 ** @author   Andrea Vedaldi
 **/

/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/


#include "../Matting.h"
#include "kdtree.h"

#if 0
VlKDForest* vl_kdtreebuild(double* pData, int m, int n)
{
  enum {IN_DATA = 0, IN_END} ;
  enum {OUT_TREE = 0} ;

  int            verbose = 0 ;
  int            opt ;
  int            next = IN_END ;

  VlKDForest * forest ;
  void * data = pData;
  vl_size numData =n ;
  vl_size dimension =m ;
  vl_type dataType = VL_TYPE_DOUBLE;
  int thresholdingMethod = VL_KDTREE_MEDIAN ;
  VlVectorComparisonType distance = VlDistanceL2;
  vl_size numTrees = 1 ;


  //data = mxGetData (IN(DATA)) ;
  //numData = mxGetN (IN(DATA)) ;
  //dimension = mxGetM (IN(DATA)) ;

  
  forest = vl_kdforest_new (dataType, dimension, numTrees, distance) ;
  vl_kdforest_set_thresholding_method (forest, thresholdingMethod) ;


  vl_kdforest_build (forest, numData, data) ;

  return forest;
}

//void
//vl_kdtreequery(int nout, mxArray *out[],
//	int nin, const mxArray *in[])

VLKDTREEQUERY_RESULT* vl_kdtreequery(VlKDForest* forest, double* pQuery, unsigned int Param0, unsigned int m, unsigned int n)
{
	enum { IN_FOREST = 0, IN_DATA, IN_QUERY, IN_END };
	enum { OUT_INDEX = 0, OUT_DISTANCE };

	int verbose = 0;

	void * query = pQuery;
	//vl_uint32 * index;
	void * distance;
	vl_size numNeighbors = (vl_size)Param0;
	vl_size numQueries;
	unsigned int numComparisons = 0;
	unsigned int maxNumComparisons = 0;
	vl_index i;

	/* -----------------------------------------------------------------
	*                                               Check the arguments
	* -------------------------------------------------------------- */

	vl_kdforest_set_max_num_comparisons(forest, maxNumComparisons);

	numQueries = n;
	VLKDTREEQUERY_RESULT* out = (VLKDTREEQUERY_RESULT*)calloc(sizeof(VLKDTREEQUERY_RESULT), 1);
	out->pIDX = (unsigned int *)calloc(sizeof(unsigned int), m*n);
	out->pD = (double *)calloc(sizeof(double), m*n);


	numComparisons = vl_kdforest_query_with_array(forest, out->pIDX, numNeighbors, numQueries, out->pD, query);

	vl_kdforest_delete(forest);

	/* adjust for MATLAB indexing */
	//for (i = 0; i < (signed)(numNeighbors * numQueries); ++i) { index[i] ++; }

	return out;

}
#endif

VlKDForest* vl_kdtreebuild(double* pData, int m, int n)
{
	enum { IN_DATA = 0, IN_END };
	enum { OUT_TREE = 0 };

	int            verbose = 0;
	int            opt;
	int            next = IN_END;

	VlKDForest * forest;
	void * data = pData;
	vl_size numData = n;
	vl_size dimension = m;
	vl_type dataType = VL_TYPE_DOUBLE;
	int thresholdingMethod = VL_KDTREE_MEDIAN;
	VlVectorComparisonType distance = VlDistanceL2;
	vl_size numTrees = 1;


	//data = mxGetData (IN(DATA)) ;
	//numData = mxGetN (IN(DATA)) ;
	//dimension = mxGetM (IN(DATA)) ;


	forest = vl_kdforest_new(dataType, dimension, numTrees, distance);
	vl_kdforest_set_thresholding_method(forest, (VlKDTreeThresholdingMethod)thresholdingMethod);


	vl_kdforest_build(forest, numData, data);

	return forest;
}

VLKDTREEQUERY_RESULT* vl_kdtreequery(VlKDForest* forest, double* pQuery, unsigned int Param0, unsigned int m, unsigned int n)
{
	enum { IN_FOREST = 0, IN_DATA, IN_QUERY, IN_END };
	enum { OUT_INDEX = 0, OUT_DISTANCE };

	int verbose = 0;

	void * query = pQuery;
	//vl_uint32 * index;
	void * distance;
	vl_size numNeighbors = (vl_size)Param0;
	vl_size numQueries;
	unsigned int numComparisons = 0;
	unsigned int maxNumComparisons = 0;
	vl_index i;

	/* -----------------------------------------------------------------
	*                                               Check the arguments
	* -------------------------------------------------------------- */
	//	VlKDForest * forest = new_kdforest_from_array(forestArr, pQuery);

	vl_kdforest_set_max_num_comparisons(forest, maxNumComparisons);

	numQueries = n;
	VLKDTREEQUERY_RESULT* out = (VLKDTREEQUERY_RESULT*)calloc(sizeof(VLKDTREEQUERY_RESULT), 1);
	out->pIDX = (unsigned int *)calloc(sizeof(unsigned int), numNeighbors*numQueries);
	out->pD = (double *)calloc(sizeof(double), numNeighbors*numQueries);


	numComparisons = vl_kdforest_query_with_array(forest, out->pIDX, numNeighbors, numQueries, out->pD, query);


	vl_kdforest_delete(forest);

	for (i = 0; i < (signed)(numNeighbors * numQueries); i++) { out->pIDX[i] ++; }

	/* adjust for MATLAB indexing */
	//for (i = 0; i < (signed)(numNeighbors * numQueries); ++i) { index[i] ++; }

	return out;
}
