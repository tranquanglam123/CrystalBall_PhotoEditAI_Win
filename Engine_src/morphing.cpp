#include "morphing.h"
#define  WARP_PAR		0
using namespace cv;
Point2f GetNormalVector(Point2f a, Point2f basept)
{
	Point2f unittangent = a / LENGTH(a);
	Point2f unitnormalvector1(unittangent.y, -unittangent.x);
	Point2f unitnormalvector2(-unittangent.y, unittangent.x);
	if ((basept.x * unitnormalvector1.x + basept.y * unitnormalvector1.y) < 0) {
		return unitnormalvector1;
	}
	return unitnormalvector2;
}
// calculate distance between from point p to line (a, b)
double get_distance_to_line(Point2f a, Point2f b, Point2f p) {
	Point2f offca = p - a, offcb = p - b, offab = a - b;
	double dca = SQR(offca.x) + SQR(offca.y);
	double dcb = SQR(offcb.x) + SQR(offcb.y);
	double dab = SQR(offab.x) + SQR(offab.y);
	if (dab == 0) return 0;
	double z = SQR(dca - dcb + dab) / (4 * dab);
	if (dca < z) return 0;
	double x = sqrt(dca - z);
	return x;
}

inline Point2f get_edge_point_one(const Point2f &p1, const Point2f &p2, Point2f pt, int nValidAxis) {
	if (nValidAxis == 0) {//x Axis, so get pt.y from pt.x
		pt.y = (pt.x - p1.x) / (p2.x - p1.x) * (p2.y - p1.y) + p1.y;
	}
	else {// y Axis, so get pt.x from pt.y
		pt.x = (pt.y - p1.y) / (p2.y - p1.y) * (p2.x - p1.x) + p1.x;
	}
	return pt;
}

int get_edge_point(Point2f &p1, Point2f &p2, Rect bound) {
	Point2f pt;
	if (p2.y < bound.tl().y || p2.y >= bound.br().y) {
		if (p2.y < bound.tl().y) {
			pt.y = bound.tl().y;
		}
		else if (p2.y >= bound.br().y) {
			pt.y = bound.br().y - 1;
		}
		p2 = get_edge_point_one(p1, p2, pt, 1);
	}
	if (p2.x < bound.tl().x || p2.x >= bound.br().x) {
		if (p2.x < bound.tl().x) {
			pt.x = bound.tl().x;
		}
		else if (p2.x >= bound.br().x) {
			pt.x = bound.br().x - 1;
		}
		p2 = get_edge_point_one(p1, p2, pt, 0);
	}
	if (p1.y < bound.tl().y || p1.y >= bound.br().y) {
		if (p1.y < bound.tl().y) {
			pt.y = bound.tl().y;
		}
		else if (p1.y >= bound.br().y) {
			pt.y = bound.br().y - 1;
		}
		p1 = get_edge_point_one(p2, p1, pt, 1);
	}
	if (p1.x < bound.tl().x || p1.x >= bound.br().x) {
		if (p1.x < bound.tl().x) {
			pt.x = bound.tl().x;
		}
		else if (p1.x >= bound.br().x) {
			pt.x = bound.br().x - 1;
		}
		p1 = get_edge_point_one(p2, p1, pt, 0);
	}
	return 0;
}

inline int get_projection_point(Point2f p0, Point2f p1, Point2f srcpt, Point2f& projectionpt)
{
	float alpha = ((srcpt.x - p1.x) * (p0.x - p1.x) + (srcpt.y - p1.y) * (p0.y - p1.y)) /
		((p0.x - p1.x) * (p0.x - p1.x) + (p0.y - p1.y) * (p0.y - p1.y));
	projectionpt = alpha * p0 + (1 - alpha) * p1;
	return 0;
}

inline int get_crossing_point(Point2f p0, Point2f p1, Point2f q0, Point2f q1, Point2f& crossingpt)
{
	float alpha;
	alpha = (p0.x - q0.x) * (p1.y - p0.y) - (p0.y - q0.y) * (p1.x - p0.x);
	alpha /= ((q1.x - q0.x) * (p1.y - p0.y) - (q1.y - q0.y) * (p1.x - p0.x));
	crossingpt = q0 + alpha * (q1 - q0);
	return 0;
}

int get_symmetrical_point(Point2f pcenter, Point2f p0, Point2f p1, Point2f srcpt, Point2f &SymmetricalPt,
	float fsymratio = 2.0f, float fcenterratio = 1.0)
{
	Point2f projectionpt;
	get_projection_point(p0, p1, srcpt, projectionpt);
	SymmetricalPt = (1 + fsymratio / 2) * projectionpt - fsymratio / 2 * srcpt;
	SymmetricalPt = pcenter + (SymmetricalPt - pcenter) * fcenterratio;
	return 0;
}

// calculate symmetric point p by line (a, b)
Point2f get_symmetrical_point(Point2f a, Point2f b, Point2f p) {
	Point2f vecAP = p - a;
	float dist = get_distance_to_line(a, b, p);
	Point2f vecAB = b - a;
	float distAP_P = sqrt(SQR_LENGTH(vecAP) - dist * dist);
	Point2f vecC = (vecAB / LENGTH(vecAB)) * distAP_P * 2;
	Point2f res = (vecC - vecAP) + a;
	return res;
}

inline int IsSameSide(Point2f tangent, Point2f a, Point2f b)
{
	float cosa, cosb;
	cosa = tangent.y * a.x - tangent.x * a.y;
	cosb = tangent.y * b.x - tangent.x * b.y;
	if (SIGN(cosa) == SIGN(cosb)) {
		return 1;
	}
	return 0;
}

inline bool IsLine(vector<Point2f> a)
{
	Point2f v1 = a[0] - a[1];
	Point2f v2 = a[1] - a[2];

	if (abs(v1.cross(v2)) < 0.001) {
		return true;
	}
	return false;
}

inline bool IsOutPoint(Point2f a, int nWidth, int nHeight)
{
	if(a.x < 0 || a.x >= nWidth || a.y < 0 || a.y >= nHeight) {
		return true;
	}
	return false;
}

// Apply affine transform calculated using srcTri and dstTri to src
void applyAffineTransform(cv::Mat &warpImage, cv::Mat &src, vector<Point2f> &srcTri, vector<Point2f> &dstTri)
{
	// Given a pair of triangles, find the affine transform.
	cv::Mat warpMat = cv::getAffineTransform(srcTri, dstTri);
	// Apply the Affine Transform just found to the src image
	cv::warpAffine(src, warpImage, warpMat, warpImage.size(), INTER_LINEAR, BORDER_REFLECT_101);
}

// Warps and alpha blends triangular regions from img1 and img2 to img
#if !WARP_PAR
int warpTriangle(const cv::Mat &i_imgSrc, cv::Mat &i_imgDst, vector<Point2f> &ptSrc, vector<Point2f> &ptDst)
{
	if (IsLine(ptSrc) || IsLine(ptDst)) {
		return 0;
	}
	for(int i=0; i<3; i++) {
		if(IsOutPoint(ptSrc[i], i_imgSrc.cols, i_imgSrc.rows) == true || IsOutPoint(ptDst[i], i_imgSrc.cols, i_imgSrc.rows) == true) {
			return 0;
		}
	}
	//LogI("Face Beauty-Nose------{(%.3f,%.3f),(%.3f,%.3f),(%.3f,%.3f) -- (%.3f,%.3f),(%.3f,%.3f),(%.3f,%.3f)}", ptSrc[0].x, ptSrc[0].y,ptSrc[1].x, ptSrc[1].y,ptSrc[2].x, ptSrc[2].y,ptDst[0].x,ptDst[0].y,ptDst[1].x,ptDst[1].y,ptDst[2].x,ptDst[2].y);
	cv::Rect rtSrc = boundingRect(ptSrc);
	cv::Rect rtDst = boundingRect(ptDst);
	vector<Point2f> t1Rect, t2Rect;
	vector<Point> t2RectInt;
	for (int i = 0; i < 3; i++) {
		t1Rect.push_back(Point2f(ptSrc[i].x - rtSrc.x, ptSrc[i].y - rtSrc.y));
		t2Rect.push_back(Point2f(ptDst[i].x - rtDst.x, ptDst[i].y - rtDst.y));
		t2RectInt.push_back(Point((int)(ptDst[i].x - rtDst.x), (int)(ptDst[i].y - rtDst.y))); // for fillConvexPoly
	}

	// Get mask by filling triangle
	cv::Mat mask = cv::Mat::zeros(rtDst.height, rtDst.width, i_imgSrc.type());
	cv::Scalar maskValue(1.0, 1.0, 1.0, 1.0);
	cv::fillConvexPoly(mask, t2RectInt, maskValue, 16, 0);

	// Apply warpImage to small rectangular patches
	cv::Mat imgSrc;
	i_imgSrc(rtSrc).copyTo(imgSrc);

	cv::Mat imgDst = cv::Mat::zeros(rtDst.height, rtDst.width, imgSrc.type());

	applyAffineTransform(imgDst, imgSrc, t1Rect, t2Rect);
	cv::multiply(imgDst, mask, imgDst);
	cv::multiply(i_imgDst(rtDst), maskValue - mask, i_imgDst(rtDst));

	i_imgDst(rtDst) = i_imgDst(rtDst) + imgDst;
	return 0;
}
#else
double multi(Point2f &p1, Point2f &p2, Point2f &p0) {
	return ((p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y));
}

int isInConvexPolygon(vector<Point2f> polygon, Point2f p) {
	double s = 0, tri, area = 0;
	int n = polygon.size();
	for (int i = 1; i <= n; i++) {
		tri = fabs(multi(polygon[i - 1], polygon[i % n], p));
		if (tri == 0) return 1;
		s += tri;
		area += fabs(multi(polygon[i - 1], polygon[i % n], polygon[0]));
	}
	return (fabs(s - area) < 0.9 ? 0 : 2);
}

inline void get_AffinMat(vector<Point2f> p, vector<Point2f> q, float* AffineMat)
{
	float MatrixVal = -p[0].x * p[2].y + p[1].x * p[2].y - p[1].x * p[0].y - p[2].x * p[1].y + p[2].x * p[0].y + p[0].x * p[1].y;
	AffineMat[0] = (-p[1].y * q[2].x + p[0].y * q[2].x - p[0].y * q[1].x - q[0].x * p[2].y + q[0].x * p[1].y + q[1].x * p[2].y) / MatrixVal;	///a11
	AffineMat[1] = (-p[1].x * q[0].x - p[0].x * q[2].x + p[2].x * q[0].x + p[0].x * q[1].x + p[1].x * q[2].x - p[2].x * q[1].x) / MatrixVal;	///a12
	AffineMat[2] = (p[1].y * p[0].x * q[2].x - q[1].x *p[0].x * p[2].y - p[0].y * p[1].x * q[2].x + p[0].y * p[2].x * q[1].x
		+ q[0].x * p[1].x * p[2].y - q[0].x * p[2].x * p[1].y) / MatrixVal;																		///b1

	AffineMat[3] = (p[2].y * q[1].y - p[0].y * q[1].y + p[0].y * q[2].y - p[2].y * q[0].y + p[1].y * q[0].y - p[1].y * q[2].y) / MatrixVal;		///a21
	AffineMat[4] = (-p[1].x * q[0].y + p[0].x * q[1].y + p[2].x * q[0].y - p[2].x * q[1].y - p[0].x * q[2].y + p[1].x * q[2].y) / MatrixVal;	///a22
	AffineMat[5] = (-p[2].y * p[0].x * q[1].y + q[2].y *p[0].x * p[1].y + p[0].y * p[2].x * q[1].y - p[0].y * p[1].x * q[2].y
		+ p[2].y * p[1].x * q[0].y - q[0].y * p[2].x * p[1].y) / MatrixVal;																		///b2
}

int warpTriangle(const cv::Mat i_imgSrc, cv::Mat &i_imgDst, vector<Point2f> ptSrc, vector<Point2f> ptDst)
{
	if (IsLine(ptDst)) {
		return 0;
	}
	int start_x, start_y, end_x, end_y, in_x, in_y, adrin, adrout, x, y;
	float fin_x, fin_y, x_bi, y_bi;
	float AffineMat[6];
	get_AffinMat(ptDst, ptSrc, AffineMat); // inverse Affine Transformation
	int nBitSize = i_imgSrc.channels();
	Rect srcRt = boundingRect(ptSrc);
	Rect dstRt = boundingRect(ptDst);
	start_x = dstRt.x;
	start_y = dstRt.y;
	end_x = dstRt.br().x;
	end_y = dstRt.br().y;

#if OMP_USE
#pragma omp parallel for schedule(dynamic)
#endif
	for (int y = start_y; y < end_y; y++) {
		for (int x = start_x; x < end_x; x++) {
			if (isInConvexPolygon(ptDst, Point2f(x, y)) > 1) {
				continue;
			}
			fin_x = AffineMat[0] * x + AffineMat[1] * y + AffineMat[2];
			fin_y = AffineMat[3] * x + AffineMat[4] * y + AffineMat[5];
			in_x = fin_x;
			in_y = fin_y;
			x_bi = fin_x - in_x;
			y_bi = fin_y - in_y;
			adrin = (in_y * i_imgSrc.cols + in_x) * nBitSize;
			adrout = (y * i_imgDst.cols + x) * nBitSize;
			for (int n = 0; n < nBitSize; n++) {
				i_imgDst.data[adrout] = i_imgSrc.data[adrin] * (1 - x_bi) * (1 - y_bi) +
					i_imgSrc.data[adrin + nBitSize] * x_bi * (1 - y_bi) + 
					i_imgSrc.data[adrin + nBitSize * i_imgSrc.cols] * (1 - x_bi) * y_bi + 
					i_imgSrc.data[adrin + nBitSize * i_imgSrc.cols + nBitSize] * x_bi * y_bi;
				adrout++;
				adrin++;
			}
		}
	}
	return 0;
}
#endif

int MorphingOneOrigin(const cv::Mat &inImage, const FaceLocation& landmark, int faceParam, int eyeParam, Mat &outImage)
{
	faceParam = max(-100, min(100, faceParam));
	eyeParam = max(-100, min(100, eyeParam));
	outImage = inImage.clone();
	vector<Point2f> pointsFaceOriginFront(17);
	vector<Point2f> pointsFaceModelFront(17);
	// set face front points for origin, model, need only 0-17, 27 indices for face morphing
	for (int i = 0; i < 17; i++) {
		pointsFaceOriginFront[i] = Point2f((float)landmark.landmarks[i].x, (float)max(0, landmark.landmarks[i].y));
	}

	Rect imgRect(1, 1, inImage.cols - 2, inImage.rows - 2);
	cv::Rect rt = boundingRect(pointsFaceOriginFront);
	if (!imgRect.contains(rt.br()) || !imgRect.contains(rt.tl())) {
		return 0;
	}

	Point2f pointCenter = Point2f((float)landmark.landmarks[27].x, (float)max(0, landmark.landmarks[27].y));
	Point2f tangent;

	// calc face front points for model
	for (int i = 0; i < 17; i++) {
		double dis;
		if (i < 8) {
			dis = get_distance_to_line(pointsFaceOriginFront[0], pointsFaceOriginFront[8], pointsFaceOriginFront[i]);
			tangent = pointCenter - pointsFaceOriginFront[0];
		}
		else {
			dis = get_distance_to_line(pointsFaceOriginFront[8], pointsFaceOriginFront[16], pointsFaceOriginFront[i]);
			tangent = pointCenter - pointsFaceOriginFront[16];
		}
		
		double lengthTangent = LENGTH(tangent);
		tangent /= lengthTangent;
		pointsFaceModelFront[i] = pointsFaceOriginFront[i] + tangent * dis * (faceParam / 400.0f);
		get_edge_point(pointCenter, pointsFaceOriginFront[i], imgRect);
		get_edge_point(pointCenter, pointsFaceModelFront[i], imgRect);
	}

	// set face back points for model, origin
	vector<Point2f> pointsOriginBack(17);
	Point2f pt;
	for (int i = 0; i < 17; i++) {
		pt = pointCenter + 1.5 * (pointsFaceOriginFront[i] - pointCenter);
		get_edge_point(pointCenter, pt, imgRect);
		pointsOriginBack[i] = pt;
	}

	// calc eye front points for model
	vector<Point2f> pointsREyeOriginFront(12);
	vector<Point2f> pointsREyeModelFront(12);
	vector<Point2f> pointsLEyeOriginFront(12);
	vector<Point2f> pointsLEyeModelFront(12);

	Point2f LEyePt = Point2f((float)landmark.landmarks[LM68_LEFT_EYE_INDEX].x, (float)max(0, landmark.landmarks[LM68_LEFT_EYE_INDEX].y));
	Point2f REyePt = Point2f((float)landmark.landmarks[LM68_RIGHT_EYE_INDEX].x, (float)max(0, landmark.landmarks[LM68_RIGHT_EYE_INDEX].y));
	
	for (int i = 0; i < 6; i++) {
		pt = Point2f((float)landmark.landmarks[i + 36].x, (float)max(0, landmark.landmarks[i + 36].y));
		get_edge_point(LEyePt, pt, imgRect);
		pointsLEyeOriginFront[i * 2] = pt;

		pt = Point2f((float)landmark.landmarks[i + 42].x, (float)max(0, landmark.landmarks[i + 42].y));
		get_edge_point(REyePt, pt, imgRect);
		pointsREyeOriginFront[i * 2] = pt;
	}

	for (int i = 0; i < 6; i++) {
		pt = (pointsLEyeOriginFront[i * 2] + pointsLEyeOriginFront[(i * 2 + 2) % 12]) / 2.0f;
		get_edge_point(LEyePt, pt, imgRect);
		pointsLEyeOriginFront[i * 2 + 1] = pt;
		pt = (pointsREyeOriginFront[i * 2] + pointsREyeOriginFront[(i * 2 + 2) % 12]) / 2.0f;
		get_edge_point(REyePt, pt, imgRect);
		pointsREyeOriginFront[i * 2 + 1] = pt;
	}

	double LEyeLength = LENGTH((pointsLEyeOriginFront[0] - pointsLEyeOriginFront[6]));
	double REyeLength = LENGTH((pointsREyeOriginFront[0] - pointsREyeOriginFront[6]));

	for (int i = 0; i < 12; i++) {
		tangent = GET_UNIT_VECTOR((LEyePt - pointsLEyeOriginFront[i]));
		pt = pointsLEyeOriginFront[i] - tangent * LEyeLength / 10.0 * eyeParam / 100.0;
		get_edge_point(LEyePt, pt, imgRect);
		pointsLEyeModelFront[i] = pt;

		tangent = GET_UNIT_VECTOR((REyePt - pointsREyeOriginFront[i]));
		pt = pointsREyeOriginFront[i] - tangent * REyeLength / 10.0 * eyeParam / 100.0;
		get_edge_point(REyePt, pt, imgRect);
		pointsREyeModelFront[i] = pt;
	}
		
	// set eye back points for model, origin
	vector<Point2f> pointsLEyeOriginBack(12);
	vector<Point2f> pointsREyeOriginBack(12);
	for (int i = 0; i < 12; i++) {
		tangent = GET_UNIT_VECTOR((LEyePt - pointsLEyeOriginFront[i]));
		pt = pointsLEyeOriginFront[i] - tangent * LEyeLength * 2 / 10.0;
		get_edge_point(LEyePt, pt, imgRect);
		pointsLEyeOriginBack[i] = pt;

		tangent = GET_UNIT_VECTOR((REyePt - pointsREyeOriginFront[i]));
		pt = pointsREyeOriginFront[i] - tangent * REyeLength * 2 / 10.0;
		get_edge_point(REyePt, pt, imgRect);
		pointsREyeOriginBack[i] = pt;
	}
	
	vector<Point2f> facePointCenters(1);
	facePointCenters[0] = pointCenter;
	warpPointArray(inImage, outImage, pointsFaceOriginFront, pointsFaceModelFront, facePointCenters);
	warpPointArray(inImage, outImage, pointsFaceOriginFront, pointsFaceModelFront, pointsOriginBack);
	vector<Point2f> eyePointCenters(1);
	eyePointCenters[0] = LEyePt;
	warpPointArray(inImage, outImage, pointsLEyeOriginFront, pointsLEyeModelFront, eyePointCenters);
	warpPointArray(inImage, outImage, pointsLEyeOriginFront, pointsLEyeModelFront, pointsLEyeOriginBack);
	eyePointCenters[0] = REyePt;
	warpPointArray(inImage, outImage, pointsREyeOriginFront, pointsREyeModelFront, eyePointCenters);
	warpPointArray(inImage, outImage, pointsREyeOriginFront, pointsREyeModelFront, pointsREyeOriginBack);
	
	return 0;
}

int MorphingOne(const cv::Mat &inImage, const FaceLocation& landmark, int faceParam, int eyeParam, Mat &outImage)
{
	if (faceParam == 0 && eyeParam == 0) {
		return 0;
	}

	faceParam = max(-100, min(100, faceParam));
	eyeParam = max(-100, min(100, eyeParam));
	vector<Point2f> pointsFaceFront(17);
	vector<Point2f> pointsFaceOriginFront(17);
	vector<Point2f> pointsFaceModelFront(17);
	// set face front points for origin, model, need only 0-17, 27 indices for face morphing
	for (int i = 0; i < 17; i++) {
		pointsFaceOriginFront[i] = Point2f((float)landmark.landmarks[i].x, (float)max(0, landmark.landmarks[i].y));
	}

	Rect imgRect(1, 1, inImage.cols - 2, inImage.rows - 2);
	cv::Rect rt = boundingRect(pointsFaceOriginFront);
	if (!imgRect.contains(rt.br()) || !imgRect.contains(rt.tl())) {
		return 0;
	}

	Point2f pointCenter = Point2f((float)landmark.landmarks[27].x, (float)max(0, landmark.landmarks[27].y));
	Point2f tangent;

	// calc face front points for model
	for (int i = 0; i < 17; i++) {
		double dis;
		if (i < 8) {
			dis = get_distance_to_line(pointsFaceOriginFront[0], pointsFaceOriginFront[8], pointsFaceOriginFront[i]);
			tangent = pointCenter - pointsFaceOriginFront[0];
		}
		else {
			dis = get_distance_to_line(pointsFaceOriginFront[8], pointsFaceOriginFront[16], pointsFaceOriginFront[i]);
			tangent = pointCenter - pointsFaceOriginFront[16];
		}

		double lengthTangent = LENGTH(tangent);
		tangent /= lengthTangent;
		pointsFaceModelFront[i] = pointsFaceOriginFront[i] + tangent * dis * (faceParam / 400.0f);
		pointsFaceFront[i] = 0.05 * pointCenter + 0.95 * pointsFaceOriginFront[i];
		get_edge_point(pointCenter, pointsFaceOriginFront[i], imgRect);
		get_edge_point(pointCenter, pointsFaceFront[i], imgRect);
		get_edge_point(pointCenter, pointsFaceModelFront[i], imgRect);
	}

	// set face back points for model, origin
	vector<Point2f> pointsOriginBack(17);
	Point2f pt;
	for (int i = 0; i < 17; i++) {
		pt = pointCenter + 1.5 * (pointsFaceOriginFront[i] - pointCenter);
		get_edge_point(pointCenter, pt, imgRect);
		pointsOriginBack[i] = pt;
	}

	// calc eye front points for model
	vector<Point2f> pointsREyeOriginFront(13);
	vector<Point2f> pointsREyeModelFront(13);
	vector<Point2f> pointsLEyeOriginFront(13);
	vector<Point2f> pointsLEyeModelFront(13);

	Point2f LEyePt = Point2f((float)landmark.landmarks[LM68_LEFT_EYE_INDEX].x, (float)max(0, landmark.landmarks[LM68_LEFT_EYE_INDEX].y));
	Point2f REyePt = Point2f((float)landmark.landmarks[LM68_RIGHT_EYE_INDEX].x, (float)max(0, landmark.landmarks[LM68_RIGHT_EYE_INDEX].y));

	for (int i = 0; i < 6; i++) {
		pt = Point2f((float)landmark.landmarks[i + 36].x, (float)max(0, landmark.landmarks[i + 36].y));
		get_edge_point(LEyePt, pt, imgRect);
		pointsLEyeOriginFront[i * 2] = pt;

		pt = Point2f((float)landmark.landmarks[i + 42].x, (float)max(0, landmark.landmarks[i + 42].y));
		get_edge_point(REyePt, pt, imgRect);
		pointsREyeOriginFront[i * 2] = pt;
	}

	for (int i = 0; i < 6; i++) {
		pt = (pointsLEyeOriginFront[i * 2] + pointsLEyeOriginFront[(i * 2 + 2) % 12]) / 2.0f;
		get_edge_point(LEyePt, pt, imgRect);
		pointsLEyeOriginFront[i * 2 + 1] = pt;
		pt = (pointsREyeOriginFront[i * 2] + pointsREyeOriginFront[(i * 2 + 2) % 12]) / 2.0f;
		get_edge_point(REyePt, pt, imgRect);
		pointsREyeOriginFront[i * 2 + 1] = pt;
	}
	pointsLEyeOriginFront[12] = pointsLEyeOriginFront[0];
	pointsLEyeOriginFront[12] = pointsLEyeOriginFront[0];

	pointsREyeOriginFront[12] = pointsREyeOriginFront[0];
	pointsREyeModelFront[12] = pointsREyeModelFront[0];

	double LEyeLength = LENGTH((pointsLEyeOriginFront[0] - pointsLEyeOriginFront[6]));
	double REyeLength = LENGTH((pointsREyeOriginFront[0] - pointsREyeOriginFront[6]));

	for (int i = 0; i < 13; i++) {
		tangent = GET_UNIT_VECTOR((LEyePt - pointsLEyeOriginFront[i]));
		pt = pointsLEyeOriginFront[i] - tangent * LEyeLength / 10.0 * eyeParam / 100.0;
		get_edge_point(LEyePt, pt, imgRect);
		pointsLEyeModelFront[i] = pt;

		tangent = GET_UNIT_VECTOR((REyePt - pointsREyeOriginFront[i]));
		pt = pointsREyeOriginFront[i] - tangent * REyeLength / 10.0 * eyeParam / 100.0;
		get_edge_point(REyePt, pt, imgRect);
		pointsREyeModelFront[i] = pt;
	}

	// set eye back points for model, origin
	vector<Point2f> pointsLEyeOriginBack(13);
	vector<Point2f> pointsREyeOriginBack(13);
	for (int i = 0; i < 13; i++) {
		tangent = GET_UNIT_VECTOR((LEyePt - pointsLEyeOriginFront[i]));
		pt = pointsLEyeOriginFront[i] - tangent * LEyeLength * 2 / 10.0;
		get_edge_point(LEyePt, pt, imgRect);
		pointsLEyeOriginBack[i] = pt;

		tangent = GET_UNIT_VECTOR((REyePt - pointsREyeOriginFront[i]));
		pt = pointsREyeOriginFront[i] - tangent * REyeLength * 2 / 10.0;
		get_edge_point(REyePt, pt, imgRect);
		pointsREyeOriginBack[i] = pt;
	}
	outImage = inImage.clone();
	warpPointArray(inImage, outImage, pointsFaceOriginFront, pointsFaceModelFront, pointsFaceFront);
	warpPointArray(inImage, outImage, pointsFaceOriginFront, pointsFaceModelFront, pointsOriginBack);
	vector<Point2f> eyePointCenters(1);
	eyePointCenters[0] = LEyePt;
	warpPointArray(inImage, outImage, pointsLEyeOriginFront, pointsLEyeModelFront, eyePointCenters);
	warpPointArray(inImage, outImage, pointsLEyeOriginFront, pointsLEyeModelFront, pointsLEyeOriginBack);
	eyePointCenters[0] = REyePt;
	warpPointArray(inImage, outImage, pointsREyeOriginFront, pointsREyeModelFront, eyePointCenters);
	warpPointArray(inImage, outImage, pointsREyeOriginFront, pointsREyeModelFront, pointsREyeOriginBack);
	return 0;
}

int MorphingAll(const Mat& inImage, const vector<FaceLocation>& aryLandmarks, int faceParam, int eyeParam, Mat &outImage)
{
	for (int i = 0; i < aryLandmarks.size(); i++) {
		MorphingOne(inImage, aryLandmarks[i], faceParam, eyeParam, outImage);
	}
	return 0;
}

int StableLandmark(FaceLocation& landmark) {
	static FaceLocation prevLandmark1 = {};
	static FaceLocation prevLandmark2 = {};
	static int weight = 0;
	static Vec2f weightLandmark[70];
	Vec2f velocity(0, 0);
	Vec2f cur, prev;
	for (int i = 0; i < 70; i++) {
		cur = Vec2f(landmark.landmarks[i].x, landmark.landmarks[i].y);
		prev = Vec2f(prevLandmark1.landmarks[i].x, prevLandmark1.landmarks[i].y);
		velocity += (cur - prev);
	}
	velocity /= 70;

	float distVelocity = sqrt((velocity[0] * velocity[0]) + (velocity[1] * velocity[1]));
	if (distVelocity < 8.0f) {
		for (int i = 0; i < 70; i++) {
			weightLandmark[i][0] = (weightLandmark[i][0] * weight + landmark.landmarks[i].x) / (float)(weight + 1);
			weightLandmark[i][1] = (weightLandmark[i][1] * weight + landmark.landmarks[i].y) / (float)(weight + 1);

			landmark.landmarks[i].x = (int)weightLandmark[i][0];
			landmark.landmarks[i].y = (int)weightLandmark[i][1];
		}
		weight++;
	}
	else {
		weight = 0;
	}

	//save prev state
	prevLandmark2 = prevLandmark1;
	prevLandmark1 = landmark;
	return 0;
}

int RotatePoint(const Point &pos1, const int nWidth, const int nHeight, int nRotate, Point &pos2) {
	Point temp = pos1;
	switch (nRotate) {
		case 270:
			pos2.x = temp.y;
			pos2.y = nWidth - 1  - temp.x;
			break;
		case 90:
			pos2.x = nHeight - 1  - temp.y;
			pos2.y = temp.x;
			break;
		case 180:
			pos2.x = nHeight -1 - temp.y;
			pos2.y = nWidth - 1  - temp.x;
			break;
		default :
			pos2.x = temp.x;
			pos2.y = temp.y;
			break;
	}
	return 0 ;
}

int RotateFaceLocation(const Detection &pos1, const int nWidth, const int nHeight, int nRotate, Detection &pos2) {
	Point temp1 = Point(pos1.xmin, pos1.ymin);
	Point temp2 = Point(pos1.xmax, pos1.ymax);
	switch (nRotate) {
		case 270:
			pos2.xmin = temp1.y;
			pos2.ymin = nWidth - 1  - temp1.x;
			pos2.xmax = temp2.y;
			pos2.ymax = nWidth - 1 - temp2.x;
			break;
		case 90:
			pos2.xmin = nHeight - 1 - temp1.y;
			pos2.ymin = temp1.x;
			pos2.xmax = nHeight - 1 - temp2.y;
			pos2.ymax = temp2.x;
			break;
		case 180:
			pos2.xmin = nHeight - 1 - temp1.y;
			pos2.ymin = nWidth - 1 - temp1.x;
			pos2.xmax = nHeight - 1 - temp2.y;
			pos2.ymax = nWidth - 1 - temp2.x;
			break;
		default :
			pos2.xmin = temp1.x;
			pos2.ymin = temp1.y;
			pos2.xmax = temp2.x;
			pos2.ymax = temp2.y;
			break;
	}
	return 0 ;
}

int RotateFaceLocationAndLandmark(Detection &faceLocation, FaceLocation & faceLandmark, const int nWidth, const int nHeight, const int nRotate) {
	RotateFaceLocation(faceLocation, nWidth, nHeight, nRotate, faceLocation);
	for(int i = 0; i < 70;  i++) {
		RotatePoint(faceLandmark.landmarks[i], nWidth, nHeight, nRotate, faceLandmark.landmarks[i]);
	}
	return 0;
}

int GetLandmark(const Mat& inColorImage, Detection &faceLocationSet, FaceLocation &faceLandmarkSet)
{
	int res;
	Mat grayImage;
	int nMaxDistance = min(inColorImage.cols, inColorImage.rows) / 2, nDistance;
	if(inColorImage.channels() == 4) {
		cv::cvtColor(inColorImage, grayImage, cv::COLOR_RGBA2GRAY);
	} else if(inColorImage.channels() == 1){
		grayImage = inColorImage.clone();
	} else if(inColorImage.channels() == 3){
		cv::cvtColor(inColorImage, grayImage, cv::COLOR_RGB2GRAY);
	}
	res = LM68_dlib_do_extract(grayImage.data, grayImage.cols, grayImage.rows, 8, faceLocationSet, faceLandmarkSet);
	if (res == 0) {
		StableLandmark(faceLandmarkSet);
		nDistance = sqrt(SQR(faceLandmarkSet.landmarks[LM68_LEFT_EYE_INDEX].x - faceLandmarkSet.landmarks[LM68_RIGHT_EYE_INDEX].x) +
			SQR(faceLandmarkSet.landmarks[LM68_LEFT_EYE_INDEX].y - faceLandmarkSet.landmarks[LM68_RIGHT_EYE_INDEX].y));
		if (nDistance >= nMaxDistance) {
			return -1;
		}
		return 0;
	}
	return 0;
}

///Warps and alpha blends triangular regions from img1 and img2 to img
inline void DrawTriangle(cv::Mat &io_img, vector<Point2f> aryPt, Scalar color) {
	cv::line(io_img, aryPt[0], aryPt[1], color);
	cv::line(io_img, aryPt[2], aryPt[1], color);
	cv::line(io_img, aryPt[0], aryPt[2], color);
}

int warpPointArray(const cv::Mat &i_imgSrc, cv::Mat &i_imgDst, vector<Point2f> ptSrc, vector<Point2f> ptDst, vector<Point2f> ptCenter)
{
	if (ptSrc.size() != ptDst.size()) {
		return -1;
	}
	if (ptCenter.size() != ptSrc.size()) {
		if (ptCenter.size() != 1 && ptCenter.size() != ptSrc.size() + 2) {
			return -1;
		}
	}

	int nPointCount = ptSrc.size();
	vector<Point2f> originpt(3), modelpt(3);
	if (ptCenter.size() == nPointCount + 2) {
		originpt[0] = ptCenter[0];
		originpt[1] = ptCenter[1];
		originpt[2] = ptSrc[0];
		modelpt[0] = ptCenter[0];
		modelpt[1] = ptCenter[1];
		modelpt[2] = ptDst[0];
		warpTriangle(i_imgSrc, i_imgDst, originpt, modelpt);		

		for (int VertexNum = 0; VertexNum < nPointCount - 1; VertexNum++) {
			originpt[0] = ptCenter[VertexNum + 1];
			originpt[1] = ptSrc[VertexNum];
			originpt[2] = ptSrc[VertexNum + 1];
			modelpt[0] = ptCenter[VertexNum + 1];
			modelpt[1] = ptDst[VertexNum];
			modelpt[2] = ptDst[VertexNum + 1];
			warpTriangle(i_imgSrc, i_imgDst, originpt, modelpt);
			
			originpt[0] = ptCenter[VertexNum + 1];
			originpt[1] = ptCenter[VertexNum + 2];
			originpt[2] = ptSrc[VertexNum + 1];
			modelpt[0] = ptCenter[VertexNum + 1];
			modelpt[1] = ptCenter[VertexNum + 2];
			modelpt[2] = ptDst[VertexNum + 1];
			warpTriangle(i_imgSrc, i_imgDst, originpt, modelpt);			
		}
		
		originpt[0] = ptCenter[nPointCount + 1];
		originpt[1] = ptCenter[nPointCount];
		originpt[2] = ptSrc[nPointCount - 1];
		modelpt[0] = ptCenter[nPointCount + 1];
		modelpt[1] = ptCenter[nPointCount];
		modelpt[2] = ptDst[nPointCount - 1];
		warpTriangle(i_imgSrc, i_imgDst, originpt, modelpt);		
	} if (ptCenter.size() != 1) {
		for (int VertexNum = 0; VertexNum < nPointCount - 1; VertexNum++) {
			originpt[0] = ptCenter[VertexNum];
			originpt[1] = ptSrc[VertexNum];
			originpt[2] = ptSrc[VertexNum + 1];
			modelpt[0] = ptCenter[VertexNum];
			modelpt[1] = ptDst[VertexNum];
			modelpt[2] = ptDst[VertexNum + 1];

			warpTriangle(i_imgSrc, i_imgDst, originpt, modelpt);
						
			originpt[0] = ptCenter[VertexNum];
			originpt[1] = ptCenter[VertexNum + 1];
			originpt[2] = ptSrc[VertexNum + 1];
			modelpt[0] = ptCenter[VertexNum];
			modelpt[1] = ptCenter[VertexNum + 1];
			modelpt[2] = ptDst[VertexNum + 1];
			warpTriangle(i_imgSrc, i_imgDst, originpt, modelpt);			
		}
	} else {
		originpt[0] = ptCenter[0];
		modelpt[0] = ptCenter[0];
		for (int VertexNum = 0; VertexNum < nPointCount - 1; VertexNum++) {
			originpt[1] = ptSrc[VertexNum];
			originpt[2] = ptSrc[VertexNum + 1];
			modelpt[1] = ptDst[VertexNum];
			modelpt[2] = ptDst[VertexNum + 1];
			warpTriangle(i_imgSrc, i_imgDst, originpt, modelpt);			
		}
	}
	return 0;
}

int TwinFromFaceInfo(const Mat& inImage, const Detection& faceLocation, Mat& outImage)
{
	int nWidth = inImage.cols;
	int nHeight = inImage.rows;
	outImage = inImage.clone();
	int res = -1;

	if (faceLocation.xmax > faceLocation.xmin && faceLocation.ymax > faceLocation.ymin) {
		if (((faceLocation.xmax + faceLocation.xmin) / 2) < (nWidth / 2)) {
			for (int i = 0; i < nHeight; i++) {
				for (int j = nWidth / 2; j < nWidth; j++) {
					outImage.at<Vec3b>(i, j)[0] = outImage.at<Vec3b>(i, nWidth - j - 1)[0];
					outImage.at<Vec3b>(i, j)[1] = outImage.at<Vec3b>(i, nWidth - j - 1)[1];
					outImage.at<Vec3b>(i, j)[2] = outImage.at<Vec3b>(i, nWidth - j - 1)[2];
				}
			}
		}
		else {
			for (int i = 0; i < nHeight; i++) {
				for (int j = 0; j < nWidth / 2; j++) {
					outImage.at<Vec3b>(i, j)[0] = outImage.at<Vec3b>(i, nWidth - j - 1)[0];
					outImage.at<Vec3b>(i, j)[1] = outImage.at<Vec3b>(i, nWidth - j - 1)[1];
					outImage.at<Vec3b>(i, j)[2] = outImage.at<Vec3b>(i, nWidth - j - 1)[2];
				}
			}
		}
		res = 1;
	}
	return res;
}

int JawThinFromFaceInfo(const Mat& inImage, const FaceLocation& landmarks, int jawParam, Mat &outImage)
{
	if (jawParam == JAW_DEFAULT_PARAM) {
		outImage = inImage.clone();
		return 1;
	}
	Mat img;
	jawParam = max(0, min(jawParam, 100));
	float fJawScale = (jawParam - 30) / 600.0f;
	Rect imgRect(1, 1, inImage.cols - 2, inImage.rows - 2);
	Point2f pt8 = landmarks.landmarks[8], pt5 = landmarks.landmarks[5], pt11 = landmarks.landmarks[11], pt27 = landmarks.landmarks[27];
	get_edge_point(pt27, pt8, imgRect);

	vector<Point2f> pointsJawContour(7);
	vector<Point2f> pointsJawContourModel(7);
	vector<Point2f> pointsJawContourBack(7);

	Point2f pointCenter(0, 0), pointJawBottom, pointJawBottomBack, pt;
	int i;
	pointCenter = (pt11 + pt5) / 2.0f;
	get_edge_point(pt27, pointCenter, imgRect);
	pointJawBottom = pt8 * (1.0f + fJawScale) - pointCenter * fJawScale;

	for (i = 0; i < 7; i++) {
		pointsJawContour[i] = landmarks.landmarks[i + 5];
	}
	double lengthTangentPtBase1 = LENGTH((pt5 - pt8));
	double lengthTangentPtBase2 = LENGTH((pt11 - pt8));
	Point2f normalPtModel1 = GetNormalVector(pt5 - pointJawBottom, SIGN(fJawScale) * (pointCenter - pt5));
	Point2f normalPtModel2 = GetNormalVector(pt11 - pointJawBottom, SIGN(fJawScale) * (pointCenter - pt11));

	double dis, lengthBase, ratioProjection;
	Point2f projectionPtBase, projectionPtModel, projectionPtBack, projectionPtBack2;
	for (i = 0; i < 7; i++) {
		if (i < 3) {
			get_projection_point(pt5, pt8, pointsJawContour[i], projectionPtBase);
			dis = LENGTH((pointsJawContour[i] - projectionPtBase));

			lengthBase = LENGTH((pt5 - projectionPtBase));
			ratioProjection = lengthBase / lengthTangentPtBase1;
			projectionPtModel = (1 - ratioProjection) * pt5 + ratioProjection * pointJawBottom;
			pointsJawContourModel[i].x = ROUND((projectionPtModel + SIGN(fJawScale) * (1.0f - fJawScale) * dis * normalPtModel1).x);
			pointsJawContourModel[i].y = ROUND((projectionPtModel + SIGN(fJawScale) * (1.0f - fJawScale) * dis * normalPtModel1).y);

		}
		else {
			get_projection_point(pt8, pt11, pointsJawContour[i], projectionPtBase);
			dis = LENGTH((pointsJawContour[i] - projectionPtBase));

			lengthBase = LENGTH((pt11 - projectionPtBase));
			ratioProjection = lengthBase / lengthTangentPtBase2;
			projectionPtModel = (1 - ratioProjection) * pt11 + ratioProjection * pointJawBottom;
			pointsJawContourModel[i].x = ROUND((projectionPtModel + SIGN(fJawScale) * (1.0f - fJawScale / 2.0f) * dis * normalPtModel2).x);
			pointsJawContourModel[i].y = ROUND((projectionPtModel + SIGN(fJawScale) * (1.0f - fJawScale / 2.0f) * dis * normalPtModel2).y);

		}
		pointsJawContourBack[i] = pointCenter + 2.0 * (pointsJawContourModel[i] - pointCenter);
	}

	cv::Rect rt = boundingRect(pointsJawContour);
	if (!imgRect.contains(rt.br()) || !imgRect.contains(rt.tl())) {
		return -1;
	}

	for (i = 0; i < 7; i++) {
		get_edge_point(pointCenter, pointsJawContour[i], imgRect);
		get_edge_point(pointCenter, pointsJawContourModel[i], imgRect);
		get_edge_point(pointCenter, pointsJawContourBack[i], imgRect);
	}
	vector<Point2f> pointCenters(1);
	pointCenters[0].x = ROUND(pointCenter.x);
	pointCenters[0].y = ROUND(pointCenter.y);
	clock_t tm = clock();
	outImage = inImage.clone();
	warpPointArray(inImage, outImage, pointsJawContour, pointsJawContourModel, pointCenters);
	warpPointArray(inImage, outImage, pointsJawContour, pointsJawContourModel, pointsJawContourBack);
	tm = clock() - tm;
	cout << "draw time = " << tm << endl;
	return 1;
}

int LipThinFromFaceInfo(const Mat& inImage, const FaceLocation& landmarks, int lipParam, Mat &outImage)
{
	if (lipParam == LIP_DEFAULT_PARAM) {
		outImage = inImage.clone();
		return 1;
	}
	lipParam = max(0, min(lipParam, 100));
	float fLipScale = 1.0f + (lipParam - 50) / 500.0f;
	Rect imgRect(1, 1, inImage.cols - 2, inImage.rows - 2);
	Point2f pointCenter(0.0f, 0.0f), pt;
	vector<Point2f> pointsLipContour(13);
	vector<Point2f> pointsLipContourModel(13);
	vector<Point2f> pointsLipContourBack(13);
	Point2f pt3 = landmarks.landmarks[3], pt13 = landmarks.landmarks[13], pt33 = landmarks.landmarks[33];

	int i;
	for (i = 48; i < 68; i++) {
		pointCenter.x += (float)landmarks.landmarks[i].x;
		pointCenter.y += (float)landmarks.landmarks[i].y;
		if (i < 60) {
			pointsLipContour[i - 48].x = (float)landmarks.landmarks[i].x;
			pointsLipContour[i - 48].y = (float)landmarks.landmarks[i].y;
		}
	}
	pointCenter /= 20.0f;
	pointsLipContour[12] = pointsLipContour[0];

	for (i = 5; i < 12; i++) {
		pointsLipContourBack[11 - (i + 6) % 12].x = (float)landmarks.landmarks[i].x;
		pointsLipContourBack[11 - (i + 6) % 12].y = (float)landmarks.landmarks[i].y;
	}

	for (i = 1; i < 6; i++) {
		if (i < 3) {      
			get_crossing_point(pt3, pt33, pointCenter, pointsLipContour[i], pointsLipContourBack[i]);
		}
		else {
			get_crossing_point(pt13, pt33, pointCenter, pointsLipContour[i], pointsLipContourBack[i]);
		}
	}
	pointsLipContourBack[12] = pointsLipContourBack[0];

	for (i = 0; i < 12; i++) {
		pt = pointCenter + fLipScale * (pointsLipContour[i] - pointCenter);
		if (i > 0 && i < 3) {
			if (!IsSameSide(pt3 - pt33, pt, pointCenter)) {
				pointsLipContourModel[i] = pointsLipContourBack[i];
				continue;
			}
		}
		if (i > 2 && i < 6) {
			if (!IsSameSide(pt13 - pt33, pt, pointCenter)) {
				pointsLipContourModel[i] = pointsLipContourBack[i];
				continue;
			}
		}
		pointsLipContourModel[i] = pt;
	}
	pointsLipContourModel[12] = pointsLipContourModel[0];

	cv::Rect rt = boundingRect(pointsLipContour);
	if (!imgRect.contains(rt.br()) || !imgRect.contains(rt.tl())) {
		return -1;
	}

	for (i = 0; i < 13; i++) {
		get_edge_point(pointCenter, pointsLipContour[i], imgRect);
		get_edge_point(pointCenter, pointsLipContourModel[i], imgRect);
		get_edge_point(pointCenter, pointsLipContourBack[i], imgRect);
	}
	vector<Point2f> pointCenters(1);
	pointCenters[0].x = ROUND(pointCenter.x);
	pointCenters[0].y = ROUND(pointCenter.y);
	outImage = inImage.clone();
	warpPointArray(inImage, outImage, pointsLipContour, pointsLipContourModel, pointCenters);
	warpPointArray(inImage, outImage, pointsLipContour, pointsLipContourModel, pointsLipContourBack);
	return 1;
}

int NoseSharpenFromFaceInfo(const Mat& inImage, const FaceLocation& landmarks, int noseParam, Mat &outImage)
{
	if (noseParam == NOSE_DEFAULT_PARAM) {
		outImage = inImage.clone();
		return 1;
	}
	noseParam = max(0, min(noseParam, 100));
	float fNoseScale = (noseParam - 50) / 800.0f;
	Rect imgRect(1, 1, inImage.cols - 2, inImage.rows - 2);
	Point2f pointCenter(0.0f, 0.0f), pt;
	vector<Point2f> pointsCenter(1);
	vector<Point2f> pointsLNoseContour(6);
	vector<Point2f> pointsRNoseContour(6);
	vector<Point2f> pointsLNoseContourModel(6);
	vector<Point2f> pointsRNoseContourModel(6);
	vector<Point2f> pointsLNoseContourBack(6);
	vector<Point2f> pointsRNoseContourBack(6);

	int i;
	Point2f pt31 = landmarks.landmarks[31], pt27 = landmarks.landmarks[27], pt29 = landmarks.landmarks[29], pt35 = landmarks.landmarks[35],
			pt39 = landmarks.landmarks[39], pt42 = landmarks.landmarks[42];
	for (i = 0; i < 3; i++) {
		pointsLNoseContour[i] = landmarks.landmarks[33 - i];
		pointsRNoseContour[i] = landmarks.landmarks[33 + i];
	}
	pointsLNoseContour[3].x = ROUND((pointsLNoseContour[2] * 1.5f - pointsLNoseContour[0] * 0.5f).x);
	pointsLNoseContour[3].y = ROUND((pointsLNoseContour[2] * 1.5f - pointsLNoseContour[0] * 0.5f).y);
	pointsRNoseContour[3].x = ROUND((pointsRNoseContour[2] * 1.5f - pointsRNoseContour[0] * 0.5f).x);
	pointsRNoseContour[3].y = ROUND((pointsRNoseContour[2] * 1.5f - pointsRNoseContour[0] * 0.5f).y);

	pointsLNoseContour[4].x = ROUND(((pt27 + pt39) / 2.0f).x);
	pointsLNoseContour[4].y = ROUND(((pt27 + pt39) / 2.0f).y);
	pointsRNoseContour[4].x = ROUND(((pt27 + pt42) / 2.0f).x);
	pointsRNoseContour[4].y = ROUND(((pt27 + pt42) / 2.0f).y);

	pt.x = ROUND((landmarks.landmarks[21].x + landmarks.landmarks[22].x) / 2.0f);
	pt.y = ROUND((landmarks.landmarks[21].y + landmarks.landmarks[22].y) / 2.0f);

	pointsLNoseContour[5].x = ROUND(pt.x);
	pointsLNoseContour[5].y = ROUND(pt.y);
	pointsRNoseContour[5].x = ROUND(pt.x);
	pointsRNoseContour[5].y = ROUND(pt.y);

	pointsCenter[0] = pt29;

	pt.x = ROUND((landmarks.landmarks[30].x + landmarks.landmarks[33].x) / 2.0f);
	pt.y = ROUND((landmarks.landmarks[30].y + landmarks.landmarks[33].y) / 2.0f);

	Point2f pointLNoseNormalVector;
	pointLNoseNormalVector = -GetNormalVector(pointsCenter[0] - pt, pt35);
	Point2f pointRNoseNormalVector = -pointLNoseNormalVector;
	Point2f pointLQuadVector, pointRQuadVector;
	
	double ltotaldist = LENGTH((pointsLNoseContour[0] - pointsLNoseContour[3])), 
		rtotaldist = LENGTH((pointsRNoseContour[0] - pointsRNoseContour[3])), dist, ratio;

	for (i = 0; i < 4; i++) {
		dist = LENGTH((pointsLNoseContour[i] - pointsLNoseContour[0]));
		ratio = dist / ltotaldist;
		pointsLNoseContourModel[i].x = ROUND((pointsLNoseContour[i] + pointLNoseNormalVector * 1.5f * ratio * fNoseScale * dist).x);
		pointsLNoseContourModel[i].y = ROUND((pointsLNoseContour[i] + pointLNoseNormalVector * 1.5f * ratio * fNoseScale * dist).y);

		dist = LENGTH((pointsRNoseContour[i] - pointsRNoseContour[0]));
		ratio = dist / rtotaldist;
		pointsRNoseContourModel[i].x = ROUND((pointsRNoseContour[i] + pointRNoseNormalVector * 1.5f * ratio * fNoseScale * dist).x);
		pointsRNoseContourModel[i].y = ROUND((pointsRNoseContour[i] + pointRNoseNormalVector * 1.5f * ratio * fNoseScale * dist).y);
	}

	pointsLNoseContourModel[4].x = ROUND((pt27 * (0.5f - fNoseScale) + (0.5f + fNoseScale) * pt39).x);
	pointsLNoseContourModel[4].y = ROUND((pt27 * (0.5f - fNoseScale) + (0.5f + fNoseScale) * pt39).y);
	pointsRNoseContourModel[4].x = ROUND((pt27 * (0.5f - fNoseScale) + (0.5f + fNoseScale) * pt42).x);
	pointsRNoseContourModel[4].y = ROUND((pt27 * (0.5f - fNoseScale) + (0.5f + fNoseScale) * pt42).y);

	pointsLNoseContourModel[5] = pointsLNoseContour[5];
	pointsRNoseContourModel[5] = pointsRNoseContour[5];
	for (i = 0; i < 6; i++) {
		pointsLNoseContourBack[i] = pt29 + (pointsLNoseContour[i] - pt29) * 2;
		pointsRNoseContourBack[i] = pt29 + (pointsRNoseContour[i] - pt29) * 2;
	}

	cv::Rect rt = boundingRect(pointsLNoseContour);
	if (!imgRect.contains(rt.br()) || !imgRect.contains(rt.tl())) {
		return -1;
	}

	for (i = 0; i < 3; i++) {
		get_edge_point(pointsCenter[0], pointsLNoseContour[i], imgRect);
		get_edge_point(pointsCenter[0], pointsLNoseContourModel[i], imgRect);
		get_edge_point(pointsCenter[0], pointsLNoseContourBack[i], imgRect);
		get_edge_point(pointsCenter[0], pointsRNoseContour[i], imgRect);
		get_edge_point(pointsCenter[0], pointsRNoseContourModel[i], imgRect);
		get_edge_point(pointsCenter[0], pointsRNoseContourBack[i], imgRect);
	}
	outImage = inImage.clone();

	warpPointArray(inImage, outImage, pointsLNoseContour, pointsLNoseContourModel, pointsCenter);
	warpPointArray(inImage, outImage, pointsRNoseContour, pointsRNoseContourModel, pointsCenter);

	warpPointArray(inImage, outImage, pointsLNoseContour, pointsLNoseContourModel, pointsLNoseContourBack);
	warpPointArray(inImage, outImage, pointsRNoseContour, pointsRNoseContourModel, pointsRNoseContourBack);
	return 1;
}

int ForeheadHighFromFaceInfo(const Mat& inImage, const FaceLocation& landmark, const int foreheadParam, Mat& outImage)
{
	if (foreheadParam == FOREHEAD_DEFAULT_PARAM) {
		outImage = inImage.clone();
		return 1;
	}

	int realParam;
	realParam = (foreheadParam - 50) * 1.3;
	
	cv::Rect bound(1, 1, inImage.cols - 2, inImage.rows - 2);
	Point2f ptA = landmark.landmarks[0];
	Point2f ptB = landmark.landmarks[16];
	Point2f pt27 = landmark.landmarks[27];
	Point2f pt33 = landmark.landmarks[33];
	Point2f ptDir = pt27 - pt33;
	Point2f ptFront, pt;
	Point2f ptSrc;
	Point2f ptDst;
	Point2f ptBack;
	vector<Point2f> vecFront;
	vector<Point2f> vecSrc;
	vector<Point2f> vecDst;
	vector<Point2f> vecBack;

	for (int i = 17; i < 27; i++) {
		pt = landmark.landmarks[i];
		ptFront = pt + ptDir * 0.05f;// +ptDir * len / 5.0f;
		get_edge_point(pt27, ptFront, bound);
		vecFront.push_back(ptFront);
		ptSrc = pt + ptDir * 0.5f;
		get_edge_point(pt, ptSrc, bound);
		vecSrc.push_back(ptSrc);
		ptDst = pt + ptDir * (5 + realParam / 50.0f) / 10.0f;
		get_edge_point(pt, ptDst, bound);
		vecDst.push_back(ptDst);
		ptBack = pt + ptDir * 1.5f;
		get_edge_point(pt, ptBack, bound);
		vecBack.push_back(ptBack);
	}
	Point2f ptFirst, ptEnd;
	ptFirst.x = ptA.x;
	if (fabs(ptA.x - vecFront[0].x) < 5) {
		ptFirst.x = ptA.x - (vecFront[2].x - vecFront[0].x) / 10.0f;
	}
	ptFirst.y = (vecFront[0].y + vecSrc[0].y) / 2.0f;

	ptEnd.x = ptB.x;
	if (fabs(ptB.x - vecFront[5].x) < 5) {
		ptEnd.x = ptB.x + (vecFront[5].x - vecFront[3].x) / 10.0f;
	}
	ptEnd.y = (vecFront[5].y + vecSrc[5].y) / 2.0f;

	get_edge_point(pt27, ptFirst, bound);
	get_edge_point(pt27, ptEnd, bound);
	vecFront.insert(vecFront.begin(), ptFirst);
	vecBack.insert(vecBack.begin(), ptFirst);
	vecFront.push_back(ptEnd);
	vecBack.push_back(ptEnd);
	outImage = inImage.clone();

	warpPointArray(inImage, outImage, vecSrc, vecDst, vecFront);
	warpPointArray(inImage, outImage, vecSrc, vecDst, vecBack);
	return 1;
}

int EyeGlitterFromFaceInfo(const Mat& inImage, const FaceLocation& landmark, const int glitterParam, Mat& outImage)
{
	float fGlitterParam = max(0, min(glitterParam, 100)) / 100.0f;
	if (glitterParam == EYE_DEFAULT_PARAM) {
		outImage = inImage.clone();
		return 1;
	}
	Rect bound(1, 1, inImage.cols - 1, inImage.rows - 1);
	vector<Point> pointsREye(6);
	vector<Point> pointsLEye(6);

	Point LEyePt = Point((float)landmark.landmarks[LM68_LEFT_EYE_INDEX].x, (float)max(0, landmark.landmarks[LM68_LEFT_EYE_INDEX].y));
	Point REyePt = Point((float)landmark.landmarks[LM68_RIGHT_EYE_INDEX].x, (float)max(0, landmark.landmarks[LM68_RIGHT_EYE_INDEX].y));

	for (int i = 0; i < 6; i++) {
		pointsLEye[i] = Point((float)landmark.landmarks[i + 36].x, (float)max(0, landmark.landmarks[i + 36].y));
		pointsREye[i] = Point((float)landmark.landmarks[i + 42].x, (float)max(0, landmark.landmarks[i + 42].y));
	}
	Rect LEyeRect = boundingRect(pointsLEye);
	Rect REyeRect = boundingRect(pointsREye);
	if(IsOutPoint(LEyeRect.tl(), outImage.cols, outImage.rows) == true || IsOutPoint(LEyeRect.br(), outImage.cols, outImage.rows) == true
	   || IsOutPoint(REyeRect.tl(), outImage.cols, outImage.rows) == true || IsOutPoint(REyeRect.tl(), outImage.cols, outImage.rows) == true) {
		return -1;
	}
	for (int i = 0; i < 6; i++) {
		pointsLEye[i] = pointsLEye[i] - LEyeRect.tl();
		pointsREye[i] = pointsREye[i] - REyeRect.tl();
	}

	Mat LEyeImg = outImage(LEyeRect).clone(), LYuv;
	Mat REyeImg = outImage(REyeRect).clone(), RYuv;
	LEyeImg.convertTo(LEyeImg, CV_32FC3);
	REyeImg.convertTo(REyeImg, CV_32FC3);

	cv::cvtColor(LEyeImg, LYuv, COLOR_BGR2YUV);
	cv::cvtColor(REyeImg, RYuv, COLOR_BGR2YUV);
	float* Lyuvdata = (float*)LYuv.data;
	float* Ryuvdata = (float*)RYuv.data;

	Mat LEyeMask, REyeMask;
	LEyeMask = Mat::zeros(LEyeImg.size(), LEyeImg.type());
	REyeMask = Mat::zeros(REyeImg.size(), REyeImg.type());
	fillConvexPoly(LEyeMask, pointsLEye, Scalar::all(1), LINE_AA, 0);
	fillConvexPoly(REyeMask, pointsREye, Scalar::all(1), LINE_AA, 0);

	float fAverageLEye = 0, fAverageREye = 0;
	int adr = 0;
	for (int idx = 0; idx < LYuv.rows * LYuv.cols; idx++) {
		fAverageLEye += Lyuvdata[adr];
		adr += 3;
	}
	fAverageLEye /= (LYuv.cols * LYuv.rows);

	adr = 0;
	for (int idx = 0; idx < LYuv.rows * LYuv.cols; idx++) {
		Lyuvdata[adr] = max(0.0f, min(255.0f, Lyuvdata[adr] + (Lyuvdata[adr] - fAverageLEye) * fGlitterParam));
		adr += 3;
	}

	adr = 0;
	for (int idx = 0; idx < RYuv.rows * RYuv.cols; idx++) {
		fAverageREye += Ryuvdata[adr];
		adr += 3;
	}
	fAverageREye /= (RYuv.cols * RYuv.rows);

	adr = 0;
	for (int idx = 0; idx < RYuv.rows * RYuv.cols; idx++) {
		Ryuvdata[adr] = max(0.0f, min(255.0f, Ryuvdata[adr] + (Ryuvdata[adr] - fAverageREye) * fGlitterParam));
		adr += 3;
	}

	cv::cvtColor(LYuv, LEyeImg, COLOR_YUV2BGR);
	cv::cvtColor(RYuv, REyeImg, COLOR_YUV2BGR);

	cv::multiply(LEyeImg, LEyeMask, LEyeImg);
	LEyeImg.convertTo(LEyeImg, CV_8UC3);
	cv::multiply(outImage(LEyeRect), Scalar(1.0, 1.0, 1.0) - LEyeMask, outImage(LEyeRect));
	outImage(LEyeRect) = outImage(LEyeRect) + LEyeImg;

	cv::multiply(REyeImg, REyeMask, REyeImg);
	REyeImg.convertTo(REyeImg, CV_8UC3);
	cv::multiply(outImage(REyeRect), Scalar(1.0, 1.0, 1.0) - REyeMask, outImage(REyeRect));
	outImage(REyeRect) = outImage(REyeRect) + REyeImg;
	return 1;
}
