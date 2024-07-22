#pragma once
//#include "filtering/relative_velocity_filter.h"
#include <vector>
#include <opencv2/opencv.hpp>
#include <cmath>

#define DELAY_TEST			1

using namespace std;
using namespace cv;

#ifndef MIN
#define MIN(a,b)				(((a)<(b))?(a):(b))
#endif
#ifndef MAX
#define MAX(a,b)				(((a)>(b))?(a):(b))
#endif
#ifndef ABS
#define ABS(a) 					((a)>0?(a):(-(a)))
#endif

#define SQR(a)					((a) * (a))
#define SQR_LENGTH(a)				(SQR(a.x) + SQR(a.y))
#define LENGTH(a)				(sqrt(SQR(a.x) + SQR(a.y)))
#define GET_UNIT_VECTOR(a)		(a / LENGTH(a))	

#define IMIN(a, b)				((a) ^ (((a)^(b)) & (((a) < (b)) - 1)))
#define IMAX(a, b)				((a) ^ (((a)^(b)) & (((a) > (b)) - 1)))

#ifndef SafeMemFree
#define SafeMemFree(x)			{ if (x) { free(x); x = NULL;} }
#endif

#ifndef SafeMemDelete
#define SafeMemDelete(x)		{ if (x) {	delete x; x = NULL;	} }
#endif

#define ROUND(x)				( x >= 0 ? (int)(x+0.5) : (int)(x-0.5) )
#define SIGN(x)					(x >= 0 ? 1 : -1)
#define LM68_LEFT_EYE_INDEX   68
#define LM68_RIGHT_EYE_INDEX  69

#define SQR(a)					((a) * (a))
#define IMIN(a, b)				((a) ^ (((a)^(b)) & (((a) < (b)) - 1)))
#define IMAX(a, b)				((a) ^ (((a)^(b)) & (((a) > (b)) - 1)))
#define ROUND(x)				( x >= 0 ? (int)(x+0.5) : (int)(x-0.5) )
#define FACE_DEFAULT_PARAM		/*50*/50
#define EYE_DEFAULT_PARAM		/*50*/50
#define SOFT_DEFAULT_PARAM		/*0*/50
#define WHITE_DEFAULT_PARAM		/*0*/30

#define JAW_DEFAULT_PARAM		/*30*/30
#define LIP_DEFAULT_PARAM		/*50*/50
#define NOSE_DEFAULT_PARAM		/*50*/50
#define FOREHEAD_DEFAULT_PARAM	50/*100*/
#define SMOOTH_DEFAULT_PARAM	/*0*/50

struct Anchor {
	float x;
	float y;
	float width;
	float height;
};

struct AnchorOptions {
	int				input_size_width;
	int				input_size_height;
	float			min_scale;
	float			max_scale;
	float			anchor_offset_x = 0.5;
	float			anchor_offset_y = 0.5;
	int				num_layers;
	vector<int>		feature_map_width;
	vector<int>		feature_map_height;
	vector<int>		strides;
	vector<float>	aspect_ratios;	
	bool			reduce_boxes_in_lowest_layer = false;
	float			interpolated_scale_aspect_ratio = 1.0f;
	bool			fixed_anchor_size = false;
};

struct TensorsToDetectionsOptions {
	int num_classes;
	int num_boxes;
	int num_coords;
	int keypoint_coord_offset;
	int num_keypoints = 0;
	int num_values_per_keypoint = 2;
	int box_coord_offset = 0;
	float x_scale = 0.0f;
	float y_scale = 0.0f;
	float h_scale = 0.0f;
	float w_scale = 0.0f;
	bool apply_exponential_on_box_size = false;
	bool reverse_output_order = false;
	int ignore_classes;
	bool sigmoid_score = false;
	float score_clipping_thresh;
	bool has_score_clipping_thresh;
	bool flip_vertically = false;
	float min_score_thresh;
	bool has_min_score_thresh;
};

enum OverlapType {
	UNSPECIFIED_OVERLAP_TYPE = 0,
	JACCARD = 1,
	MODIFIED_JACCARD = 2,
	INTERSECTION_OVER_UNION = 3
};

enum NmsAlgorithm {
	DEFAULTALGORITHM = 0,
// Only supports relative bounding box for weighted NMS.
	WEIGHTED = 1
};

struct NonMaxSuppressionCalculatorOptions
{
	int num_detection_streams = 1;
	int max_num_detections = -1;
	float min_score_threshold = -1.0f;
	float min_suppression_threshold = 1.0f;
	bool return_empty_detections;
	int frameWidth;
	int frameHeight;
	OverlapType overlap_type = JACCARD;
	NmsAlgorithm algorithm = DEFAULTALGORITHM;
};

struct Detection {
	int label_id = -1;
	float score = 0;
	float ymin = 0.0;
	float xmin = 0.0;
	float ymax = 0.0;
	float xmax = 0.0;
	vector<Point2f> keyPoints;
	string getLabel(vector<string> strLabels) {
		return strLabels[label_id];
	};
};

enum ConversionMode {
	DEFAULT = 0,
	USE_BOUNDING_BOX = 1,
	USE_KEYPOINTS = 2
};

struct DetectionsToRectsOptions
{
	int rotation_vector_start_keypoint_index;
	int rotation_vector_end_keypoint_index;
	float rotation_vector_target_angle;
	float rotation_vector_target_angle_degrees;
	bool output_zero_rect_for_empty_detections;
	ConversionMode conversion_mode;
};

struct RectTransformationOptions
{
	float scale_x = 1.0f;
	float scale_y = 1.0f;
	float rotation;
	int rotation_degree;
	float shift_x;
	float shitf_y;
	bool square_long;
	bool square_short;
};

struct RotRect {
	float center_x;
	float center_y;
	float width;
	float height;
	float rotation;
};

struct ImageToTensorOptions
{
	float minval;
	float maxval;
	int output_tensor_width;
	int output_tensor_height;
	bool keep_aspect_ratio;
};

struct Landmark
{
	float x_;
	float y_;
	float z_;
	float visibility_;
	float presence_;
};

struct LandmarkList
{
	vector<Landmark> vecLandmark;
};

struct TensorsToLandmarksOptions
{
	int num_landmarks;
	int input_image_width;
	int input_image_height;
	bool flip_vertically = false;
	bool flip_horizontally = false;
	float normalize_z = 1.0f;
};

struct FaceLocation 
{
	Rect faceRt;
	Point landmarks[70];
};

typedef vector<Detection> Detections;

enum {
	NMS_UNION = 0,
	NMS_MIN = 1
};

int GeneratePadding(const int nInWidth, const int nInHeight, const int nOutWidth, const int nOutHeight, 
	int& nPaddingX, int& nPaddingY);

RotRect GetRoi(int input_width, int input_height, RotRect* normRect = NULL);
array<float, 4> PadRoi(int input_tensor_width, int input_tensor_height, bool keep_aspect_ratio, RotRect* roi);
void GetRotatedSubRectToRectTransformMatrix(const RotRect& sub_rect, int rect_width, int rect_height,
	bool flip_horizontaly, std::array<float, 16>* matrix_ptr);
Mat Convert(const Mat& src, const RotRect& roi, const Size& output_dims, float range_min, float range_max);
Mat InvConvert(const Mat& src, const RotRect& roi, const Size& input_dims, const Size &output_dims);

vector<Detection> NMS(vector<Detection> &winList, float threshold, int type);
int NMSProcess(Detections& input_detections, const NonMaxSuppressionCalculatorOptions &options_, char* kTag,
	Detections& outputput_detections);

int DecodeBoxes(int num_boxes_, int num_coords_, const TensorsToDetectionsOptions &options_, const float* raw_boxes, const std::vector<Anchor>& anchors, std::vector<float>* boxes);
float CalculateScale(float min_scale, float max_scale, int stride_index, int num_strides);
int GenerateAnchors(std::vector<Anchor>* anchors, const AnchorOptions& options);
int ConvertToDetections(const TensorsToDetectionsOptions& tensorDetectorCalcOption, const float* detection_boxes, const float* detection_scores, const int* detection_classes, std::vector<Detection>& output_detections);
vector<Detection> ProcessDetection(const TensorsToDetectionsOptions& tensorDetectorCalcOption, vector<Anchor> anchors_, float* raw_boxes, float* raw_scores);
LandmarkList ProcessLandmark(const TensorsToLandmarksOptions& tensorfToLandmarksCalculatorOptions, char* kTag, float* raw_locations, int num_dimensions);
int RemovalPadding(vector<Detection> &input_detections, array<float, 4> letterbox_padding);
int RemovalPadding(LandmarkList &input_landmarks, array<float, 4> letterbox_padding);
int LandmarkProjection(LandmarkList &input_landmarks, RotRect &Roi);
Detection ConvertLandmarksToDetection(const LandmarkList& landmarks);

inline float NormalizeRadians(float angle) {
	return angle - 2 * M_PI * std::floor((angle - (-M_PI)) / (2 * M_PI));
}

int DetectionToRectsCalculator(vector<Detection>& detections, char* strTag, const Size image_size,
	const DetectionsToRectsOptions options_, vector<RotRect> &vec_outRect);
int RectTransformationCalculator(vector<RotRect>& rotRect, char* strTag, const RectTransformationOptions options_,
	const Size image_size);
int DetectionToNormalizedRect(const Detection& detection, const DetectionsToRectsOptions& options_, const Size& image_size, RotRect* rect);
int DetectionToNormalizedRect(const Detection& detection, const Size& image_size, RotRect* rect);

//
//class VelocityFilter {
//public:
//	VelocityFilter(int window_size, float velocity_scale, float min_allowed_object_scale);
//	void Reset();
//	int Apply(LandmarkList& in_landmarks, const std::pair<int, int>& image_size, const int64_t timestamp);
//private:
//	// Initializes filters for the first time or after Reset. If initialized then
//	// check the size.
//	int InitializeFiltersIfEmpty(const int n_landmarks);
//	int window_size_;
//	float velocity_scale_;
//	float min_allowed_object_scale_;
//
//	std::vector<RelativeVelocityFilter> x_filters_;
//	std::vector<RelativeVelocityFilter> y_filters_;
//	std::vector<RelativeVelocityFilter> z_filters_;
//};