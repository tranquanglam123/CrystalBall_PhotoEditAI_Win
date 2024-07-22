#pragma once

#ifndef VL_KDTREE_USR_H
#define VL_KDTREE_USR_H
//#include <Windows.h>
#include "../vl/kdtree.h"

typedef struct {
	unsigned int *pIDX;
	double *pD;
}VLKDTREEQUERY_RESULT, *LPKDTREEQUERY_REUSLT;

VlKDForest* vl_kdtreebuild(double* pData, int m, int n);
VLKDTREEQUERY_RESULT* vl_kdtreequery(VlKDForest* forest, double* pQuery, unsigned int Param0, unsigned int m, unsigned int n);

//
//#define VL_OS_WIN
//
//#define VL_TYPE_FLOAT   1     /**< @c float type */
//#define VL_TYPE_DOUBLE  2     /**< @c double type */
//#define VL_TYPE_INT8    3     /**< @c ::vl_int8 type */
//#define VL_TYPE_UINT8   4     /**< @c ::vl_uint8 type */
//#define VL_TYPE_INT16   5     /**< @c ::vl_int16 type */
//#define VL_TYPE_UINT16  6     /**< @c ::vl_uint16 type */
//#define VL_TYPE_INT32   7     /**< @c ::vl_int32 type */
//#define VL_TYPE_UINT32  8     /**< @c ::vl_uint32 type */
//#define VL_TYPE_INT64   9     /**< @c ::vl_int64 type */
//#define VL_TYPE_UINT64  10    /**< @c ::vl_uint64 type */
//
//#define VL_THREADS_WIN 1
//#define VL_KDTREE_SPLIT_HEAP_SIZE 5
//
//#define VL_TRUE 1   /**< @brief @c true (1) constant */
//#define VL_FALSE 0  /**< @brief @c false (0) constant */
//
////#if defined(VL_COMPILER_LP64) || defined(VL_COMPILER_LLP64)
//typedef long long           vl_int64;   /**< @brief Signed 64-bit integer. */
//typedef int                 vl_int32;   /**< @brief Signed 32-bit integer. */
//typedef short               vl_int16;   /**< @brief Signed 16-bit integer. */
//typedef char                vl_int8;   /**< @brief Signed  8-bit integer. */
//
//typedef long long unsigned  vl_uint64;  /**< @brief Unsigned 64-bit integer. */
//typedef int       unsigned  vl_uint32;  /**< @brief Unsigned 32-bit integer. */
//typedef short     unsigned  vl_uint16;  /**< @brief Unsigned 16-bit integer. */
//typedef char      unsigned  vl_uint8;   /**< @brief Unsigned  8-bit integer. */
//
//typedef int                 vl_int;     /**< @brief Same as @c int. */
//typedef unsigned int        vl_uint;    /**< @brief Same as <code>unsigned int</code>. */
//
//typedef int                 vl_bool;    /**< @brief Boolean. */
//typedef vl_int64            vl_intptr;  /**< @brief Integer holding a pointer. */
//typedef vl_uint64           vl_uintptr; /**< @brief Unsigned integer holding a pointer. */
//typedef vl_uint64           vl_size;    /**< @brief Unsigned integer holding the size of a memory block. */
//typedef vl_int64            vl_index;   /**< @brief Signed version of ::vl_size and ::vl_uindex */
//typedef vl_uint64           vl_uindex;  /**< @brief Same as ::vl_size */
////#endif
//
///** @brief Random numbber generator state */
//typedef struct _VlRand {
//	vl_uint32 mt[624];
//	vl_uint32 mti;
//} VlRand;
//
//typedef vl_uint32 vl_type;
//
///** @brief Vector comparison types */
//enum _VlVectorComparisonType {
//	VlDistanceL1,        /**< l1 distance (squared intersection metric) */
//	VlDistanceL2,        /**< squared l2 distance */
//	VlDistanceChi2,      /**< squared Chi2 distance */
//	VlDistanceHellinger, /**< squared Hellinger's distance */
//	VlDistanceJS,        /**< squared Jensen-Shannon distance */
//	VlDistanceMahalanobis,     /**< squared mahalanobis distance */
//	VlKernelL1,          /**< intersection kernel */
//	VlKernelL2,          /**< l2 kernel */
//	VlKernelChi2,        /**< Chi2 kernel */
//	VlKernelHellinger,   /**< Hellinger's kernel */
//	VlKernelJS           /**< Jensen-Shannon kernel */
//};
//
///** @brief Vector comparison types */
//typedef enum _VlVectorComparisonType VlVectorComparisonType;
//
///** @brief Thresholding method */
//typedef enum _VlKDTreeThresholdingMethod
//{
//	VL_KDTREE_MEDIAN,
//	VL_KDTREE_MEAN
//} VlKDTreeThresholdingMethod;
//
//
//typedef struct _VlKDTreeNode VlKDTreeNode;
//typedef struct _VlKDTreeSplitDimension VlKDTreeSplitDimension;
//typedef struct _VlKDTreeDataIndexEntry VlKDTreeDataIndexEntry;
//typedef struct _VlKDForestSearchState VlKDForestSearchState;
//
//
//struct _VlKDTreeNode
//{
//	vl_uindex parent;
//	vl_index lowerChild;
//	vl_index upperChild;
//	unsigned int splitDimension;
//	double splitThreshold;
//	double lowerBound;
//	double upperBound;
//};
//
//struct _VlKDTreeSplitDimension
//{
//	unsigned int dimension;
//	double mean;
//	double variance;
//};
//
//struct _VlKDTreeDataIndexEntry
//{
//	vl_index index;
//	double value;
//};
//
//
///** @brief Neighbor of a query point */
//typedef struct _VlKDForestNeighbor {
//	double distance;   /**< distance to the query point */
//	vl_uindex index;   /**< index of the neighbor in the KDTree data */
//} VlKDForestNeighbor;
//
//typedef struct _VlKDTree
//{
//	VlKDTreeNode * nodes;
//	vl_size numUsedNodes;
//	vl_size numAllocatedNodes;
//	VlKDTreeDataIndexEntry * dataIndex;
//	unsigned int depth;
//} VlKDTree;
//
//struct _VlKDForestSearchState
//{
//	VlKDTree * tree;
//	vl_uindex nodeIndex;
//	double distanceLowerBound;
//};
//
//typedef struct _VlThreadState
//{
//	/* errors */
//	int lastError;
//	char lastErrorMessage[1024];
//
//	/* random number generator */
//	VlRand rand;
//
//	/* time */
//#if defined(VL_OS_WIN)
//	LARGE_INTEGER ticFreq;
//	LARGE_INTEGER ticMark;
//#else
//	clock_t ticMark;
//#endif
//} VlThreadState;
//
///* Gobal state */
//typedef struct _VlState
//{
//	/* The thread state uses either a mutex (POSIX)
//	or a critical section (Win) */
//#if defined(VL_DISABLE_THREADS)
//	VlThreadState * threadState;
//#else
//#if defined(VL_THREADS_POSIX)
//	pthread_key_t threadKey;
//	pthread_mutex_t mutex;
//	pthread_t mutexOwner;
//	pthread_cond_t mutexCondition;
//	size_t mutexCount;
//#elif defined(VL_THREADS_WIN)
//	DWORD tlsIndex;
//	CRITICAL_SECTION mutex;
//#endif
//#endif /* VL_DISABLE_THREADS */
//
//	/* Configurable functions */
//	int(*printf_func)  (char const * format, ...);
//	void *(*malloc_func)  (size_t);
//	void *(*realloc_func) (void*, size_t);
//	void *(*calloc_func)  (size_t, size_t);
//	void(*free_func)    (void*);
//
//#if defined(VL_ARCH_IX86) || defined(VL_ARCH_X64) || defined(VL_ARCH_IA64)
//	VlX86CpuInfo cpuInfo;
//#endif
//	vl_size numCPUs;
//	vl_bool simdEnabled;
//	vl_size numThreads;
//} VlState;
//
///** @brief KDForest object */
//typedef struct _VlKDForest
//{
//	vl_size dimension;
//
//	/* random number generator */
//	VlRand * rand;
//
//	/* indexed data */
//	vl_type dataType;
//	void const * data;
//	vl_size numData;
//	VlVectorComparisonType distance;
//	void(*distanceFunction)(void);
//
//	/* tree structure */
//	VlKDTree ** trees;
//	vl_size numTrees;
//
//	/* build */
//	VlKDTreeThresholdingMethod thresholdingMethod;
//	VlKDTreeSplitDimension splitHeapArray[VL_KDTREE_SPLIT_HEAP_SIZE];
//	vl_size splitHeapNumNodes;
//	vl_size splitHeapSize;
//	vl_size maxNumNodes;
//
//	/* query */
//	vl_size searchMaxNumComparisons;
//	vl_size numSearchers;
//	struct _VlKDForestSearcher * headSearcher;  /* head of the double linked list with searchers */
//
//} VlKDForest;
//
//typedef struct _VlKDForestSearcher
//{
//	/* maintain a linked list of searchers for later disposal*/
//	struct _VlKDForestSearcher * next;
//	struct _VlKDForestSearcher * previous;
//
//	vl_uindex * searchIdBook;
//	VlKDForestSearchState * searchHeapArray;
//	VlKDForest * forest;
//
//	vl_size searchNumComparisons;
//	vl_size searchNumRecursions;
//	vl_size searchNumSimplifications;
//
//	vl_size searchHeapNumNodes;
//	vl_uindex searchId;
//} VlKDForestSearcher;
//
//#define VL_UINT32_C(x) x ## U
//#define MATRIX_A VL_UINT32_C(0x9908b0df)   /* constant vector a */
//#define UPPER_MASK VL_UINT32_C(0x80000000) /* most asignificant w-r bits */
//#define LOWER_MASK VL_UINT32_C(0x7fffffff) /* least significant r bits */
//#define N 624
//#define M 397
//
///** @internal @brief IEEE single precision quiet NaN constant */
//static union { vl_uint32 raw; float value; }
//const vl_nan_f =
//{ 0x7FC00000UL };
//
///** @internal @brief IEEE single precision infinity constant */
//static union { vl_uint32 raw; float value; }
//const vl_infinity_f =
//{ 0x7F800000UL };
//
///** @internal @brief IEEE double precision quiet NaN constant */
//static union { vl_uint64 raw; double value; }
//const vl_nan_d =
//#ifdef VL_COMPILER_MSC
//{ 0x7FF8000000000000ui64 };
//#else
//{ 0x7FF8000000000000ULL };
//#endif
//
//#define VL_INFINITY_F (vl_infinity_f.value)
//#define VL_NAN_F (vl_nan_f.value)

#endif
