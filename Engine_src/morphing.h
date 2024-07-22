#pragma once
#include "landmark68_dlib_lcl.h"
using namespace cv;
int get_edge_point(Point2f &p1, Point2f &p2, Rect bound);
int StableLandmark(FaceLocation& landmark);
int warpPointArray(const cv::Mat &i_imgSrc, cv::Mat &i_imgDst, vector<Point2f> ptSrc, vector<Point2f> ptDst, vector<Point2f> ptCenter);
int GetLandmark(const Mat& inImage, Detection &faceLocationSet, FaceLocation &faceLandMarkSet);
int MorphingOne(const cv::Mat &inImage, const FaceLocation& landmark, int faceParam, int eyeParam, Mat &outImage);
int MorphingAll(const Mat& inImage, const vector<FaceLocation>& aryLandmarks, int faceParam, int eyeParam, Mat &outImage);
int TwinFromFaceInfo(const Mat& inImage, const Detection& faceLocation, Mat& outImage);
int JawThinFromFaceInfo(const Mat& inImage, const FaceLocation& landmarks, int jawParam, Mat &outImage);
int LipThinFromFaceInfo(const Mat& inImage, const FaceLocation& landmarks, int lipParam, Mat &outImage);
int NoseSharpenFromFaceInfo(const Mat& inImage, const FaceLocation& landmarks, int noseParam, Mat &outImage);
int ForeheadHighFromFaceInfo(const Mat& inImage, const FaceLocation& landmark, int foreheadParam, Mat& outImage);
int EyeGlitterFromFaceInfo(const Mat& inImage, const FaceLocation& landmark, const int glitterParam, Mat& outImage);
