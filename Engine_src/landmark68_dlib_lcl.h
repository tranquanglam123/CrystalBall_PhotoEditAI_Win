#pragma once
#ifndef _LANDMARK68_DLIB_LCL_H_
#define _LANDMARK68_DLIB_LCL_H_

#include "common.h"

static double __maxarg1, __maxarg2;
#define FMAX(a, b) (__maxarg1 = (a), __maxarg2 = (b), (__maxarg1) > (__maxarg2) ?\
	(__maxarg1) : (__maxarg2))

static float __minarg1, __minarg2;
#define FMIN(a, b) (__minarg1 = (a), __minarg2 = (b), (__minarg1) < (__minarg2) ?\
	(__minarg1) : (__minarg2))

#define STANDARD_CX							480
#define STANDARD_CY							640

#define ITERATE_NUM			15
#define SHAPE_FEATURE_NUM	500

#define LANDMARK_COUNT			68
#define MAX_FACENUM				30

typedef struct _NRECT NRECT, *LPNRECT;
struct _NRECT {
	union {
		int x;
		int l;
	};
	union {
		int y;
		int t;
	};
	union {
		int w;
		int r;
	};
	union {
		int h;
		int b;
	};
	LPNRECT pNext;
};

typedef struct {
	int x;
	int y;
	int width;
	int height;
} CF_Rect;

typedef struct {
	int x;
	int y;
} CF_Point;

typedef struct {
	float x;
	float y;
} FPoint, *LPFPoint;

typedef struct {
	float dblYaw;
	float dblRoll;
	float dblPitch;
} SPoseAngle;

typedef struct {
	CF_Point     ptCord[LANDMARK_COUNT];
	FPoint       LEye;
	FPoint       REye;
	SPoseAngle   pose;
} SFaceCordInfo;

typedef struct
{
	unsigned char			bEyeDetect;				// eye detect success flag
	CF_Rect			rcFace;					// face region
	CF_Point		ptLEye, ptREye;			// left, right eye position
	SFaceCordInfo	ptLMCode;				// landmark points
	SPoseAngle		PoseAngle;				// Pose Angle
}SFaceItem;

typedef struct
{
	int			nFaces;							// face count
	SFaceItem	pFace[MAX_FACENUM];				// face structure
	int			nMaxFaceIdx;					// max face index
} SFaceInfo;

typedef struct{
	float point[2];	
}ShapePoint, *LPShapePoint;

typedef struct{
	ShapePoint data[LANDMARK_COUNT];	
	float data_x[LANDMARK_COUNT];
	float data_y[LANDMARK_COUNT];
}ShapeInfo, *LPShapeInfo;

typedef struct{
	int idx1;
	int idx2;
	float thresh;
}SpliteInfo, *LPSpliteInfo;

typedef struct{
	SpliteInfo splite[15];
	ShapeInfo leaf_value[16];
}ForestInfo, *LPForestInfo;

typedef struct{
	ForestInfo forestinfo[SHAPE_FEATURE_NUM];	
}ForestStru, *LPForestStru;

typedef struct{
	int idx[SHAPE_FEATURE_NUM];
}Anchor_idx;

typedef struct{
	ShapePoint delta_value[SHAPE_FEATURE_NUM];
}Delta, *LPDelta;

typedef struct{
	ShapeInfo initShape;
	ForestStru forest[ITERATE_NUM];
	Anchor_idx anchor[ITERATE_NUM];
	Delta delta[ITERATE_NUM];
}ShapeModel, *LPShapeModel;

static uint32_t g_dwNormalLandmarkVert[56] = {
	0xBE37DF3D, 0x3E37DF3D, 0x247A9427, 0x23C43914,
	0xA385364F, 0xA45D6F66, 0xBE9670D5, 0x3E9670D5,
	0xBF272809, 0xBE8E153B, 0x3E8E153B, 0x3F272809,
	0xBEE1A93F, 0x3EE1A93F, 0x3EA6C297, 0x3EA6C297,
	0x3E59CD3F, 0x3DD58360, 0xBC51BE97, 0xBDFC0C0B,
	0xBE809C87, 0xBE809C87, 0x3E339920, 0x3E2BB65C,
	0x3E2BB65C, 0x3E339920, 0xBF021488, 0xBF021488,
	0x3E16415C, 0x3E16415C, 0x3E761835, 0x3F17CF0E,
	0x3F8631FB, 0x3FA612A0, 0xBD71542C, 0xBD71542C,
	0xBF31B70A, 0xBF0A2748, 0xBF0A2748, 0xBF31B70A,
	0xBEE317D6, 0xBEE317D6, 0x3D80E2E5, 0x3D80E2E5,
	0x3DD1C138, 0x3E53BD9F, 0x3EAD37CC, 0x3ED55D22,
	0x3DBBCC19, 0x3DBBCC19, 0xBE03BF35, 0xBDB54228,
	0xBDB54228, 0xBE03BF35, 0x3CF925F7, 0x3CF925F7
};
static uint32_t g_dwNormalLandmarkHorz[56] = {
	0xBDE147AE, 0x3EE66666, 0xBE6B851F, 0x3F800000,
	0x3DE147AE, 0x3EE66666, 0xBE6B851F, 0x3F800000,
	0x00000000, 0x3EAE147B, 0xBE570A3D, 0x3F800000,
	0x00000000, 0x3E6B851F, 0xBE0F5C29, 0x3F800000,
	0x00000000, 0x3DE147AE, 0xBD4CCCCD, 0x3F800000,
	0x00000000, 0x00000000, 0x00000000, 0x3F800000,
	0xBE3851EC, 0xBDCCCCCD, 0xBE851EB8, 0x3F800000,
	0x3E3851EC, 0xBDCCCCCD, 0xBE851EB8, 0x3F800000,
	0xBECCCCCD, 0x3EA3D70A, 0xBEC7AE14, 0x3F800000,
	0xBE2E147B, 0x3E9EB852, 0xBEB851EC, 0x3F800000,
	0x3E2E147B, 0x3E9EB852, 0xBEB851EC, 0x3F800000,
	0x3ECCCCCD, 0x3EA3D70A, 0xBEC7AE14, 0x3F800000,
	0xBE8A3D71, 0xBEAE147B, 0xBEA8F5C3, 0x3F800000,
	0x3E8A3D71, 0xBEAE147B, 0xBEA8F5C3, 0x3F800000
};
static unsigned char g_pbyUsefulLandmarkIndexs[14] = { 0x15, 0x16, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x23, 0x24, 0x27, 0x2A, 0x2D, 0x30, 0x36 };

int LM68_dlib_finish();
int LM68_dlib_init(const char* model_path);
int LM68_dlib_do_extract(const unsigned char* image, int width, int height, int size_of_pixel, const Detection face_location, FaceLocation& landmarks);

#endif//#ifdef FACEX_LM68_AVAILABLE