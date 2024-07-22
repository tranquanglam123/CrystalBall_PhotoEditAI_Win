#include "common.h"
using namespace cv;
#if DELAY_TEST
TickMeter g_timer;
#endif

int GeneratePadding(const int nInWidth, const int nInHeight, const int nOutWidth, const int nOutHeight, int& nPaddingX, int& nPaddingY)
{
	const float input_aspect_ratio = static_cast<float>(nInWidth) / nInHeight;
	const float output_aspect_ratio = static_cast<float>(nOutWidth) / nOutHeight;

	if (input_aspect_ratio < output_aspect_ratio) {
		// Compute left and right padding.
		nPaddingX = nInWidth * (1.f - input_aspect_ratio / output_aspect_ratio) / 2.f / input_aspect_ratio;
	}
	else if (output_aspect_ratio < input_aspect_ratio) {
		// Compute top and bottom padding.
		nPaddingY = nInHeight * (1.f - output_aspect_ratio / input_aspect_ratio) / 2.f * input_aspect_ratio;
	}
	return 0;
}

constexpr char kDetectionTag[] = "DETECTION";
constexpr char kDetectionsTag[] = "DETECTIONS";
constexpr char kImageSizeTag[] = "IMAGE_SIZE";
constexpr char kRectTag[] = "RECT";
constexpr char kNormRectTag[] = "NORM_RECT";
constexpr char kRectsTag[] = "RECTS";
constexpr char kNormRectsTag[] = "NORM_RECTS";

constexpr float kMinFloat = std::numeric_limits<float>::lowest();
constexpr float kMaxFloat = std::numeric_limits<float>::max();

constexpr int kWristJoint = 0;
constexpr int kMiddleFingerPIPJoint = 6;
constexpr int kIndexFingerPIPJoint = 4;
constexpr int kRingFingerPIPJoint = 8;
constexpr float kTargetAngle = M_PI_2;

int NormRectFromKeyPoints(const Detection& location_data, RotRect* rect)
{
	if (location_data.keyPoints.size() < 2) {
		return -1;
	}
	float xmin = kMaxFloat;
	float ymin = kMaxFloat;
	float xmax = kMinFloat;
	float ymax = kMinFloat;
	for (int i = 0; i < location_data.keyPoints.size(); ++i) {
		const auto& kp = location_data.keyPoints[i];
		xmin = std::min(xmin, kp.x);
		ymin = std::min(ymin, kp.y);
		xmax = std::max(xmax, kp.x);
		ymax = std::max(ymax, kp.y);
	}
	rect->center_x = (xmin + xmax) / 2;
	rect->center_y = (ymin + ymax) / 2;
	rect->width = (xmax - xmin);
	rect->height = (ymax - ymin);
	return 0;
}

template <class B, class R>
void RectFromBox(B box, R* rect)
{
	rect->center_x = (box.xmin + box.xmax) / 2;
	rect->center_y = (box.ymin + box.ymax) / 2;
	rect->width = (box.xmax - box.xmin);
	rect->height = (box.ymax - box.ymin);
}

int DetectionToRect(const Detection& detection, const Size image_size,
	const DetectionsToRectsOptions options_, RotRect* rect)
{
	switch (options_.conversion_mode) {
	case DEFAULT:
	case USE_BOUNDING_BOX: {
		RectFromBox(detection, rect);
		break;
	}
	case USE_KEYPOINTS: {
		const int width = image_size.width;
		const int height = image_size.height;
		RotRect norm_rect;
		NormRectFromKeyPoints(detection, &norm_rect);
		rect->center_x = round(norm_rect.center_x * width);
		rect->center_y = round(norm_rect.center_y * height);
		rect->width = round(norm_rect.width * width);
		rect->height = round(norm_rect.height * height);
		break;
	}
	}
	return 0;
}

int DetectionToNormalizedRect(const Detection& detection,
	const DetectionsToRectsOptions options_, RotRect* rect)
{
	switch (options_.conversion_mode) {
	case DEFAULT:
	case USE_BOUNDING_BOX: {
		RectFromBox(detection, rect);
		break;
	}
	case USE_KEYPOINTS: {
		NormRectFromKeyPoints(detection, rect);
		break;
	}
	}
	return 0;
}

int ComputeRotation(const Detection& detection, const Size image_size,
	const DetectionsToRectsOptions options_, float* rotation)
{
	const float x0 = detection.keyPoints[options_.rotation_vector_start_keypoint_index].x *image_size.width;
	const float y0 = detection.keyPoints[options_.rotation_vector_start_keypoint_index].y *image_size.height;
	const float x1 = detection.keyPoints[options_.rotation_vector_end_keypoint_index].x * image_size.width;
	const float y1 = detection.keyPoints[options_.rotation_vector_end_keypoint_index].y * image_size.height;

	*rotation = NormalizeRadians(options_.rotation_vector_target_angle - std::atan2(-(y1 - y0), x1 - x0));
	return 0;
}

int DetectionToRectsCalculator(vector<Detection>& detections, char* strTag, const Size image_size,
	const DetectionsToRectsOptions options_, vector<RotRect> &vec_outRect)
{
	bool rotate_ = true;

	if (strcmp(strTag, kRectTag) == 0) {
		vec_outRect.resize(1);
		DetectionToRect(detections[0], image_size, options_, &vec_outRect[0]);
		if (rotate_) {
			float rotation;
			ComputeRotation(detections[0], image_size, options_, &rotation);
			vec_outRect[0].rotation = rotation;
		}
	}
	else if (strcmp(strTag, kRectsTag) == 0) {
		vec_outRect.resize(detections.size());
		for (int i = 0; i < detections.size(); ++i) {
			DetectionToRect(detections[i], image_size, options_, &vec_outRect[i]);
			if (rotate_) {
				float rotation;
				ComputeRotation(detections[i], image_size, options_, &rotation);
				vec_outRect[i].rotation = rotation;
			}
		}
	}
	else if (strcmp(strTag, kNormRectTag) == 0) {
		vec_outRect.resize(1);
		DetectionToNormalizedRect(detections[0], options_, &vec_outRect[0]);
		if (rotate_) {
			float rotation;
			ComputeRotation(detections[0], image_size, options_, &rotation);
			vec_outRect[0].rotation = rotation;
		}
	}
	else if (strcmp(strTag, kNormRectsTag) == 0) {
		vec_outRect.resize(detections.size());
		for (int i = 0; i < detections.size(); ++i) {
			DetectionToNormalizedRect(detections[i], options_, &vec_outRect[i]);
			if (rotate_) {
				float rotation;
				ComputeRotation(detections[i], image_size, options_, &rotation);
				vec_outRect[i].rotation = rotation;
			}
		}
	}
	return 0;
}

float ComputeNewRotation(float rotation, const RectTransformationOptions options_) {
	if ((options_.rotation != 0)) {
		rotation += options_.rotation;
	}
	else if ((options_.rotation_degree != 0)) {
		rotation += M_PI * options_.rotation_degree / 180.f;
	}
	return NormalizeRadians(rotation);
}

void TransformRect(RotRect* rect, const RectTransformationOptions options_) {
	float width = rect->width;
	float height = rect->height;
	float rotation = rect->rotation;

	if ((options_.rotation != 0) || (options_.rotation_degree != 0)) {
		rotation = ComputeNewRotation(rotation, options_);
	}
	if (rotation == 0.f) {
		rect->center_x = (rect->center_x + width * options_.shift_x);
		rect->center_y = (rect->center_y + height * options_.shitf_y);
	}
	else {
		const float x_shift = width * options_.shift_x * std::cos(rotation) -
			height * options_.shitf_y * std::sin(rotation);
		const float y_shift = width * options_.shift_x * std::sin(rotation) +
			height * options_.shitf_y * std::cos(rotation);
		rect->center_x = (rect->center_x + x_shift);
		rect->center_y = (rect->center_y + y_shift);
	}

	if (options_.square_long) {
		const float long_side = std::max(width, height);
		width = long_side;
		height = long_side;
	}
	else if (options_.square_short) {
		const float short_side = std::min(width, height);
		width = short_side;
		height = short_side;
	}
	rect->width = (width * options_.scale_x);
	rect->height = (height * options_.scale_y);
}

void TransformNormalizedRect(RotRect* rect, const RectTransformationOptions options_, int image_width, int image_height)
{
	float width = rect->width;
	float height = rect->height;
	float rotation = rect->rotation;

	if ((options_.rotation != 0) || (options_.rotation_degree != 0)) {
		rotation = ComputeNewRotation(rotation, options_);
	}
	if (rotation == 0.f) {
		rect->center_x = (rect->center_x + width * options_.shift_x);
		rect->center_y = (rect->center_y + height * options_.shitf_y);
	}
	else {
		const float x_shift = (image_width * width * options_.shift_x * std::cos(rotation) -
			image_height * height * options_.shitf_y * std::sin(rotation)) / image_width;
		const float y_shift = (image_width * width * options_.shift_x * std::sin(rotation) +
			image_height * height * options_.shitf_y * std::cos(rotation)) / image_height;
		rect->center_x = (rect->center_x + x_shift);
		rect->center_y = (rect->center_y + y_shift);
	}

	if (options_.square_long) {
		const float long_side = std::max(width * image_width, height * image_height);
		width = long_side / image_width;
		height = long_side / image_height;
	}
	else if (options_.square_short) {
		const float short_side = std::min(width * image_width, height * image_height);
		width = short_side / image_width;
		height = short_side / image_height;
	}
	rect->width = (width * options_.scale_x);
	rect->height = (height * options_.scale_y);
}

int RectTransformationCalculator(vector<RotRect>& rotRect, char* strTag, const RectTransformationOptions options_, const Size image_size)
{
	if (strcmp(strTag, kRectTag) == 0) {
		TransformRect(&rotRect[0], options_);
	}
	else if (strcmp(strTag, kRectsTag) == 0) {
		for (int i = 0; i < rotRect.size(); ++i) {
			TransformRect(&rotRect[i], options_);
		}
	}
	else if (strcmp(strTag, kNormRectTag) == 0) {
		TransformNormalizedRect(&rotRect[0], options_, image_size.width, image_size.height);
	}
	else if (strcmp(strTag, kNormRectsTag) == 0) {
		for (int i = 0; i < rotRect.size(); ++i) {
			TransformNormalizedRect(&rotRect[i], options_, image_size.width, image_size.height);
		}
	}
	return 0;
}

int DetectionToNormalizedRect(const Detection& detection, const DetectionsToRectsOptions& options_, const Size& image_size, RotRect* rect)
{
	const float x_center = detection.keyPoints[options_.rotation_vector_start_keypoint_index].x * image_size.width;
	const float y_center = detection.keyPoints[options_.rotation_vector_start_keypoint_index].y * image_size.height;

	const float x_scale = detection.keyPoints[options_.rotation_vector_end_keypoint_index].x * image_size.width;
	const float y_scale = detection.keyPoints[options_.rotation_vector_end_keypoint_index].y * image_size.height;

	const float box_size = sqrt((x_scale - x_center) * (x_scale - x_center) + (y_scale - y_center) * (y_scale - y_center)) * 2.0;

	rect->center_x = x_center / image_size.width;
	rect->center_y = y_center / image_size.height;
	rect->width = box_size / image_size.width;
	rect->height = box_size / image_size.height;

	return 0;
}

int DetectionToNormalizedRect(const Detection& detection, const Size& image_size, RotRect* rect)
{
	const float x_center = (detection.xmax + detection.xmin) / 2 * image_size.width;
	const float y_center = (detection.ymax + detection.ymin) / 2 * image_size.height;

	const float x_scale = (detection.xmax - detection.xmin) / 2 * image_size.width;
	const float y_scale = (detection.ymax - detection.ymin) / 2 * image_size.height;

	const float box_size = sqrt((x_scale - x_center) * (x_scale - x_center) + (y_scale - y_center) * (y_scale - y_center)) * 2.0;

	rect->center_x = x_center / image_size.width;
	rect->center_y = y_center / image_size.height;
	rect->width = box_size / image_size.width;
	rect->height = box_size / image_size.height;

	return 0;
}


RotRect GetRoi(int input_width, int input_height, RotRect* normRect)
{
	RotRect rt;
	if (normRect != NULL) {
		rt.center_x = normRect->center_x * input_width;
		rt.center_y = normRect->center_y * input_height;
		rt.width = static_cast<float>(normRect->width * input_width);
		rt.height = static_cast<float>(normRect->height * input_height);
		rt.rotation = normRect->rotation;
	}
	else {
		rt.center_x = 0.5f * input_width;
		rt.center_y = 0.5f * input_height;
		rt.width = static_cast<float>(input_width);
		rt.height = static_cast<float>(input_height);
		rt.rotation = 0;
	}
	return rt;
}

array<float, 4> PadRoi(int input_tensor_width, int input_tensor_height, bool keep_aspect_ratio, RotRect* roi)
{
	if (!keep_aspect_ratio) {
		return std::array<float, 4>{0.0f, 0.0f, 0.0f, 0.0f};
	}

	if (input_tensor_width <= 0 || input_tensor_height <= 0) {
		return array<float, 4>{0, 0, 0, 0};
	}
	const float tensor_aspect_ratio =
		static_cast<float>(input_tensor_height) / input_tensor_width;

	if (roi->width <= 0 || roi->height <= 0) {
		return array<float, 4>{0, 0, 0, 0};
	}

	const float roi_aspect_ratio = roi->height / roi->width;

	float vertical_padding = 0.0f;
	float horizontal_padding = 0.0f;
	float new_width;
	float new_height;
	if (tensor_aspect_ratio > roi_aspect_ratio) {
		new_width = roi->width;
		new_height = roi->width * tensor_aspect_ratio;
		vertical_padding = (1.0f - roi_aspect_ratio / tensor_aspect_ratio) / 2.0f;
	}
	else {
		new_width = roi->height / tensor_aspect_ratio;
		new_height = roi->height;
		horizontal_padding = (1.0f - tensor_aspect_ratio / roi_aspect_ratio) / 2.0f;
	}

	roi->width = new_width;
	roi->height = new_height;

	return array<float, 4>{horizontal_padding, vertical_padding,
		horizontal_padding, vertical_padding};
}

void GetRotatedSubRectToRectTransformMatrix(const RotRect& sub_rect, int rect_width, int rect_height,
	bool flip_horizontaly, std::array<float, 16>* matrix_ptr)
{
	std::array<float, 16>& matrix = *matrix_ptr;
	const float a = sub_rect.width;
	const float b = sub_rect.height;
	const float flip = flip_horizontaly ? -1 : 1;
	const float c = std::cos(sub_rect.rotation);
	const float d = std::sin(sub_rect.rotation);
	const float e = sub_rect.center_x;
	const float f = sub_rect.center_y;
	const float g = 1.0f / rect_width;
	const float h = 1.0f / rect_height;
	// row 1
	matrix[0] = a * c * flip * g;
	matrix[1] = -b * d * g;
	matrix[2] = 0.0f;
	matrix[3] = (-0.5f * a * c * flip + 0.5f * b * d + e) * g;

	// row 2
	matrix[4] = a * d * flip * h;
	matrix[5] = b * c * h;
	matrix[6] = 0.0f;
	matrix[7] = (-0.5f * b * c - 0.5f * a * d * flip + f) * h;

	// row 3
	matrix[8] = 0.0f;
	matrix[9] = 0.0f;
	matrix[10] = a * g;
	matrix[11] = 0.0f;

	// row 4
	matrix[12] = 0.0f;
	matrix[13] = 0.0f;
	matrix[14] = 0.0f;
	matrix[15] = 1.0f;
}

Mat Convert(const Mat& src, const RotRect& roi, const Size& output_dims, float range_min, float range_max)
{
	constexpr int kNumChannels = 3;
	Mat dst(output_dims.height, output_dims.width, CV_32FC3);

	const RotatedRect rotated_rect(Point2f(roi.center_x, roi.center_y), Size2f(roi.width, roi.height), roi.rotation * 180.f / CV_PI);
	Mat src_points;
	boxPoints(rotated_rect, src_points);

	const float dst_width = output_dims.width;
	const float dst_height = output_dims.height;
	/* clang-format off */
	float dst_corners[8] = { 0.0f,      dst_height,
		0.0f,      0.0f,
		dst_width, 0.0f,
		dst_width, dst_height };
	/* clang-format on */

	Mat dst_points = Mat(4, 2, CV_32F, dst_corners);
	Mat projection_matrix = getPerspectiveTransform(src_points, dst_points);
	Mat transformed;
	warpPerspective(src, transformed, projection_matrix, Size(dst_width, dst_height), INTER_LINEAR, BORDER_REPLICATE);
	if (transformed.channels() > kNumChannels) {
		Mat proper_channels_mat;
		cvtColor(transformed, proper_channels_mat, COLOR_RGBA2RGB);
		transformed = proper_channels_mat;
	}
	Mat proper_channels_mat;
	cvtColor(transformed, proper_channels_mat, COLOR_BGR2RGB);
	transformed = proper_channels_mat;
	constexpr float kInputImageRangeMin = 0.0f;
	constexpr float kInputImageRangeMax = 255.0f;
	const float scale = (range_max - range_min) / (kInputImageRangeMax - kInputImageRangeMin);
	const float offset = range_min - kInputImageRangeMin * scale;

	transformed.convertTo(dst, CV_32FC3, scale, offset);
	return dst;
}

Mat InvConvert(const Mat& src, const RotRect& roi, const Size& input_dims, const Size &output_dims)
{
	constexpr int kNumChannels = 3;
	Mat dst;

	const RotatedRect rotated_rect(Point2f(roi.center_x, roi.center_y), Size2f(roi.width, roi.height), roi.rotation * 180.f / CV_PI);
	Mat dst_points;
	boxPoints(rotated_rect, dst_points);

	const float src_width = input_dims.width;
	const float src_height = input_dims.height;
	/* clang-format off */
	float src_corners[8] = { 0.0f,      src_height,
		0.0f,      0.0f,
		src_width, 0.0f,
		src_width, src_height };

	Mat src_points = Mat(4, 2, CV_32F, src_corners);
	Mat projection_matrix = getPerspectiveTransform(src_points, dst_points);
	warpPerspective(src, dst, projection_matrix, output_dims, INTER_LINEAR, BORDER_CONSTANT);
	return dst;
}


typedef vector<pair<int, float>> IndexedScores;
constexpr char kImageTag[] = "IMAGE";

float IoU(Detection &window1, Detection &window2, int type)
{
	float xOverlap = max(0.f, min(window1.xmax, window2.xmax) - max(window1.xmin, window2.xmin));
	float yOverlap = max(0.f, min(window1.ymax, window2.ymax) - max(window1.ymin, window2.ymin));
	float intersection = xOverlap * yOverlap;
	float s1 = (window1.xmax - window1.xmin) * (window1.ymax - window1.ymin);
	float s2 = (window2.xmax - window2.xmin) * (window2.ymax - window2.ymin);


	float union_ = s1 + s2 - intersection;
	if (type == NMS_UNION) {
		return float(intersection) / union_;
	}
	else {
		return (intersection / min(s1, s2));
	}
}

bool CompareWin(const Detection &w1, const Detection &w2)
{
	return w1.score > w2.score;
}

bool Inside(int x, int y, const Detection& rect)
{
	if (x >= rect.xmin && y >= rect.ymin && x < rect.xmax && y < rect.ymax)
		return true;
	else
		return false;
}

vector<Detection> NMS(vector<Detection> &winList, float threshold, int type)
{
	if (winList.size() == 0) {
		return winList;
	}

	sort(winList.begin(), winList.end(), CompareWin);

	bool* flag = (bool*)malloc(winList.size() * sizeof(bool));
	memset(flag, 0, winList.size());

	for (int i = 0; i < winList.size(); i++) {
		if (flag[i])
			continue;
		for (int j = i + 1; j < winList.size(); j++) {
			if ((IoU(winList[i], winList[j], type) > threshold) && (winList[i].label_id == winList[j].label_id))
				flag[j] = 1;
		}
	}

	vector<Detection> ret;
	for (int i = 0; i < winList.size(); i++) {
		if (!flag[i]) ret.push_back(winList[i]);
	}

	free(flag);
	flag = NULL;

	return ret;
}

bool Intersects(const Rect2f& r, const Rect2f &l)
{
	if ((r.x + r.width) < l.x || (l.x + l.width) < r.x || (r.y + r.height) < l.y || (l.y + l.height) < r.y) {
		return false;
	}
	return true;
}

Rect2f Intersect(const Rect2f& r, const Rect2f& l)
{
	Point2f pmin(std::max(l.x, r.x), std::max(l.y, r.y));
	Point2f pmax(std::min(l.x + l.width, r.x + r.width), std::min(l.y + l.height, r.y + r.height));

	if (pmin.x > pmax.x || pmin.y > pmax.y)
		return Rect2f();
	else
		return Rect2f(pmin, pmax);
}

Rect2f Union(const Rect2f& r, const Rect2f& l)
{
	return Rect2f(Point2f(std::min(l.x, r.x), std::min(l.y, r.y)),
		Point2f(std::max(l.x + l.width, r.x + r.width), std::max(l.y + l.height, r.y + r.height)));
}

Rect2f ConvertToRelativeBBox(const Detection& location, int image_width, int image_height)
{
	return Rect2f(Point2f(location.xmin / image_width, location.ymin / image_width),
		Point2f(location.xmax / image_width, location.ymax / image_height));
}

bool SortBySecond(const std::pair<int, float>& indexed_score_0, const std::pair<int, float>& indexed_score_1)
{
	return (indexed_score_0.second > indexed_score_1.second);
}

float OverlapSimilarity(const OverlapType overlap_type, const Rect2f& rect1, const Rect2f& rect2)
{
	if (!Intersects(rect1, rect2)) return 0.0f;
	const float intersection_area = Intersect(rect1, rect2).area();
	float normalization;
	switch (overlap_type) {
	case JACCARD:
		normalization = Union(rect1, rect2).area();
		break;
	case MODIFIED_JACCARD:
		normalization = rect2.area();
		break;
	case INTERSECTION_OVER_UNION:
		normalization = rect1.area() + rect2.area() - intersection_area;
		break;
	}
	return normalization > 0.0f ? intersection_area / normalization : 0.0f;
}

float OverlapSimilarity(const int frame_width, const int frame_height, const OverlapType overlap_type,
	const Detection& location1, const Detection& location2)
{
	const auto rect1 = ConvertToRelativeBBox(location1, frame_width, frame_height);
	const auto rect2 = ConvertToRelativeBBox(location2, frame_width, frame_height);
	return OverlapSimilarity(overlap_type, rect1, rect2);
}

float OverlapSimilarity(const OverlapType overlap_type, const Detection& location1, const Detection& location2) {
	Rect2f rect1(Point2f(location1.xmin, location1.ymin), Point2f(location1.xmax, location1.ymax));
	Rect2f rect2(Point2f(location2.xmin, location2.ymin), Point2f(location2.xmax, location2.ymax));
	return OverlapSimilarity(overlap_type, rect1, rect2);
}

void NonMaxSuppression(const IndexedScores& indexed_scores, const Detections& detections, const int max_num_detections,
	const NonMaxSuppressionCalculatorOptions &options_, char* kTag, Detections& output_detections)
{
	for (const auto& indexed_score : indexed_scores) {
		const auto& detection = detections[indexed_score.first];
		if (options_.min_score_threshold > 0 &&
			detection.score < options_.min_score_threshold) {
			break;
		}
		bool suppressed = false;
		for (const auto& retained_location : output_detections) {
			float similarity;
			if (strcmp(kTag, kImageTag) == 0) {
				similarity = OverlapSimilarity(options_.frameWidth, options_.frameHeight,
					options_.overlap_type, retained_location, detection);
			}
			else {
				similarity = OverlapSimilarity(options_.overlap_type, retained_location, detection);
			}
			if (similarity > options_.min_suppression_threshold) {
				suppressed = true;
				break;
			}
		}
		if (!suppressed) {
			output_detections.push_back(detection);
		}
		if (output_detections.size() >= max_num_detections) {
			break;
		}
	}
}

void WeightedNonMaxSuppression(const IndexedScores& indexed_scores, const Detections& detections, const int max_num_detections,
	const NonMaxSuppressionCalculatorOptions &options_, char* kTag, Detections& output_detections)
{
	IndexedScores remained_indexed_scores;
	remained_indexed_scores.assign(indexed_scores.begin(),
		indexed_scores.end());

	IndexedScores remained;
	IndexedScores candidates;
	while (!remained_indexed_scores.empty()) {
		const int original_indexed_scores_size = remained_indexed_scores.size();
		const auto& detection = detections[remained_indexed_scores[0].first];
		if (options_.min_score_threshold > 0 && detection.score < options_.min_score_threshold) {
			break;
		}
		remained.clear();
		candidates.clear();
		// This includes the first box.
		for (const auto& indexed_score : remained_indexed_scores) {
			float similarity = OverlapSimilarity(options_.overlap_type, detections[indexed_score.first], detection);
			if (similarity > options_.min_suppression_threshold) {
				candidates.push_back(indexed_score);
			}
			else {
				remained.push_back(indexed_score);
			}
		}
		auto weighted_detection = detection;
		if (!candidates.empty()) {
			const int num_keypoints = detection.keyPoints.size();
			std::vector<float> keypoints(num_keypoints * 2);
			float w_xmin = 0.0f;
			float w_ymin = 0.0f;
			float w_xmax = 0.0f;
			float w_ymax = 0.0f;
			float total_score = 0.0f;
			for (const auto& candidate : candidates) {
				total_score += candidate.second;
				const auto& location_data = detections[candidate.first];
				w_xmin += location_data.xmin * candidate.second;
				w_ymin += location_data.ymin * candidate.second;
				w_xmax += location_data.xmax * candidate.second;
				w_ymax += location_data.ymax * candidate.second;

				for (int i = 0; i < num_keypoints; ++i) {
					keypoints[i * 2] += location_data.keyPoints[i].x * candidate.second;
					keypoints[i * 2 + 1] += location_data.keyPoints[i].y * candidate.second;
				}
			}
			weighted_detection.xmin = w_xmin / total_score;
			weighted_detection.ymin = w_ymin / total_score;
			weighted_detection.xmax = w_xmax / total_score;
			weighted_detection.ymax = w_ymax / total_score;
			for (int i = 0; i < num_keypoints; ++i) {
				weighted_detection.keyPoints[i].x = keypoints[i * 2] / total_score;
				weighted_detection.keyPoints[i].y = keypoints[i * 2 + 1] / total_score;
			}
		}
		output_detections.push_back(weighted_detection);
		if (original_indexed_scores_size == remained.size()) {
			break;
		}
		else {
			remained_indexed_scores = std::move(remained);
		}
	}
}

int NMSProcess(Detections& input_detections, const NonMaxSuppressionCalculatorOptions &options_, char* kTag,
	Detections& outputput_detections)
{
	// Check if there are any detections at all.
	if (input_detections.empty()) {
		return -1;
	}
	IndexedScores indexed_scores;
	indexed_scores.reserve(input_detections.size());
	for (int index = 0; index < input_detections.size(); ++index) {
		indexed_scores.push_back(std::make_pair(index, input_detections[index].score));
	}
	std::sort(indexed_scores.begin(), indexed_scores.end(), SortBySecond);

	const int max_num_detections = (options_.max_num_detections > -1) ? options_.max_num_detections : static_cast<int>(indexed_scores.size());
	outputput_detections.reserve(max_num_detections);

	if (options_.algorithm == WEIGHTED) {
		WeightedNonMaxSuppression(indexed_scores, input_detections, max_num_detections, options_, kTag, outputput_detections);
	}
	else {
		NonMaxSuppression(indexed_scores, input_detections, max_num_detections, options_, kTag, outputput_detections);
	}
	return 0;
}


int DecodeBoxes(int num_boxes_, int num_coords_, const TensorsToDetectionsOptions &options_, const float* raw_boxes, const std::vector<Anchor>& anchors, std::vector<float>* boxes)
{
	for (int i = 0; i < num_boxes_; ++i) {
		const int box_offset = i * num_coords_ + options_.box_coord_offset;

		float y_center = raw_boxes[box_offset];
		float x_center = raw_boxes[box_offset + 1];
		float h = raw_boxes[box_offset + 2];
		float w = raw_boxes[box_offset + 3];
		if (options_.reverse_output_order) {
			x_center = raw_boxes[box_offset];
			y_center = raw_boxes[box_offset + 1];
			w = raw_boxes[box_offset + 2];
			h = raw_boxes[box_offset + 3];
		}

		x_center =
			x_center / options_.x_scale * anchors[i].width + anchors[i].x;
		y_center =
			y_center / options_.y_scale * anchors[i].height + anchors[i].y;

		if (options_.apply_exponential_on_box_size) {
			h = std::exp(h / options_.h_scale) * anchors[i].height;
			w = std::exp(w / options_.w_scale) * anchors[i].width;
		}
		else {
			h = h / options_.h_scale * anchors[i].height;
			w = w / options_.w_scale * anchors[i].width;
		}

		const float ymin = y_center - h / 2.f;
		const float xmin = x_center - w / 2.f;
		const float ymax = y_center + h / 2.f;
		const float xmax = x_center + w / 2.f;

		(*boxes)[i * num_coords_ + 0] = ymin;
		(*boxes)[i * num_coords_ + 1] = xmin;
		(*boxes)[i * num_coords_ + 2] = ymax;
		(*boxes)[i * num_coords_ + 3] = xmax;

		if (options_.num_keypoints) {
			for (int k = 0; k < options_.num_keypoints; ++k) {
				const int offset = i * num_coords_ + options_.keypoint_coord_offset +
					k * options_.num_values_per_keypoint;

				float keypoint_y = raw_boxes[offset];
				float keypoint_x = raw_boxes[offset + 1];
				if (options_.reverse_output_order) {
					keypoint_x = raw_boxes[offset];
					keypoint_y = raw_boxes[offset + 1];
				}

				(*boxes)[offset] = keypoint_x / options_.x_scale * anchors[i].width + anchors[i].x;
				(*boxes)[offset + 1] = keypoint_y / options_.y_scale * anchors[i].height + anchors[i].y;
			}
		}
	}
	return 0;
}

float CalculateScale(float min_scale, float max_scale, int stride_index, int num_strides)
{
	if (num_strides == 1) {
		return (min_scale + max_scale) * 0.5f;
	}
	else {
		return min_scale +
			(max_scale - min_scale) * 1.0 * stride_index / (num_strides - 1.0f);
	}
}

int GenerateAnchors(std::vector<Anchor>* anchors, const AnchorOptions& options)
{
	// Verify the options.
	if (options.feature_map_height.size() == 0 && options.strides.size() == 0) {
		return -1;
	}
	if (options.feature_map_height.size()) {
		if (options.feature_map_height.size() != options.num_layers) {
			return -1;
		}
		if (options.feature_map_height.size() != options.feature_map_width.size()) {
			return -1;
		}
	}
	else {
		if (options.strides.size() != options.num_layers) {
			return -1;
		}
	}

	int layer_id = 0;
	while (layer_id < options.num_layers) {
		vector<float> anchor_height;
		vector<float> anchor_width;
		vector<float> aspect_ratios;
		vector<float> scales;

		// For same strides, we merge the anchors in the same order.
		int last_same_stride_layer = layer_id;
		while (last_same_stride_layer < options.strides.size()
			&& options.strides[last_same_stride_layer] == options.strides[layer_id]) {
			const float scale =
				CalculateScale(options.min_scale, options.max_scale,
					last_same_stride_layer, options.strides.size());
			if (last_same_stride_layer == 0 &&
				options.reduce_boxes_in_lowest_layer) {
				// For first layer, it can be specified to use predefined anchors.
				aspect_ratios.push_back(1.0);
				aspect_ratios.push_back(2.0);
				aspect_ratios.push_back(0.5);
				scales.push_back(0.1);
				scales.push_back(scale);
				scales.push_back(scale);
			}
			else {
				for (int aspect_ratio_id = 0; aspect_ratio_id < options.aspect_ratios.size(); ++aspect_ratio_id) {
					aspect_ratios.push_back(options.aspect_ratios[aspect_ratio_id]);
					scales.push_back(scale);
				}
				if (options.interpolated_scale_aspect_ratio > 0.0) {
					const float scale_next =
						last_same_stride_layer == options.strides.size() - 1
						? 1.0f
						: CalculateScale(options.min_scale, options.max_scale,
							last_same_stride_layer + 1, options.strides.size());
					scales.push_back(sqrt(scale * scale_next));
					aspect_ratios.push_back(options.interpolated_scale_aspect_ratio);
				}
			}
			last_same_stride_layer++;
		}

		for (int i = 0; i < aspect_ratios.size(); ++i) {
			const float ratio_sqrts = sqrt(aspect_ratios[i]);
			anchor_height.push_back(scales[i] / ratio_sqrts);
			anchor_width.push_back(scales[i] * ratio_sqrts);
		}

		int feature_map_height = 0;
		int feature_map_width = 0;
		if (options.feature_map_height.size()) {
			feature_map_height = options.feature_map_height[layer_id];
			feature_map_width = options.feature_map_width[layer_id];
		}
		else {
			const int stride = options.strides[layer_id];
			feature_map_height = ceil(1.0f * options.input_size_height / stride);
			feature_map_width = ceil(1.0f * options.input_size_width / stride);
		}

		for (int y = 0; y < feature_map_height; ++y) {
			for (int x = 0; x < feature_map_width; ++x) {
				for (int anchor_id = 0; anchor_id < anchor_height.size(); ++anchor_id) {
					// TODO: Support specifying anchor_offset_x, anchor_offset_y.
					const float x_center = (x + options.anchor_offset_x) * 1.0f / feature_map_width;
					const float y_center = (y + options.anchor_offset_y) * 1.0f / feature_map_height;

					Anchor new_anchor;
					new_anchor.x = x_center;
					new_anchor.y = y_center;

					if (options.fixed_anchor_size) {
						new_anchor.width = 1.0f;
						new_anchor.height = 1.0f;
					}
					else {
						new_anchor.width = anchor_width[anchor_id];
						new_anchor.height = anchor_height[anchor_id];
					}
					anchors->push_back(new_anchor);
				}
			}
		}
		layer_id = last_same_stride_layer;
	}
	return 0;
}

int ConvertToDetections(
	const TensorsToDetectionsOptions& tensorDetectorCalcOption,
	const float* detection_boxes, const float* detection_scores,
	const int* detection_classes, std::vector<Detection>& output_detections)
{
	for (int i = 0; i < tensorDetectorCalcOption.num_boxes; ++i) {
		if (tensorDetectorCalcOption.has_min_score_thresh &&
			detection_scores[i] < tensorDetectorCalcOption.min_score_thresh) {
			continue;
		}
		const int box_offset = i * tensorDetectorCalcOption.num_coords;
		Detection detection;
		detection.score = detection_scores[i];
		detection.label_id = detection_classes[i];

		detection.xmin = detection_boxes[box_offset + 1];
		detection.ymin = tensorDetectorCalcOption.flip_vertically ? 1.0f - detection_boxes[box_offset + 2] : detection_boxes[box_offset + 0];
		detection.xmax = detection_boxes[box_offset + 3];
		detection.ymax = detection.ymin + (detection_boxes[box_offset + 2] - detection_boxes[box_offset + 0]);


		//Add keypoints.
		if (tensorDetectorCalcOption.num_keypoints > 0) {
			for (int kp_id = 0;
				kp_id < tensorDetectorCalcOption.num_keypoints * tensorDetectorCalcOption.num_values_per_keypoint;
				kp_id += tensorDetectorCalcOption.num_values_per_keypoint)
			{
				const int keypoint_index = box_offset + tensorDetectorCalcOption.keypoint_coord_offset + kp_id;
				cv::Point2f keyPoint;
				keyPoint.x = detection_boxes[keypoint_index + 0];
				keyPoint.y = tensorDetectorCalcOption.flip_vertically ? 1.f - detection_boxes[keypoint_index + 1] : detection_boxes[keypoint_index + 1];
				detection.keyPoints.push_back(keyPoint);
			}
		}
		output_detections.emplace_back(detection);
	}
	return 0;
}

vector<Detection> ProcessDetection(const TensorsToDetectionsOptions& tensorDetectorCalcOption, std::vector<Anchor> anchors_, float* raw_boxes, float* raw_scores)
{
	const int num_boxes_ = tensorDetectorCalcOption.num_boxes;
	const int num_classes_ = tensorDetectorCalcOption.num_classes;
	const int num_coords_ = tensorDetectorCalcOption.num_coords;

	std::vector<float> boxes(num_boxes_ * num_coords_);
	DecodeBoxes(num_boxes_, num_coords_, tensorDetectorCalcOption, raw_boxes, anchors_, &boxes);

	std::vector<float> detection_scores(num_boxes_);
	std::vector<int> detection_classes(num_boxes_);

	// Filter classes by scores.
	for (int i = 0; i < num_boxes_; ++i) {
		int class_id = -1;
		float max_score = -std::numeric_limits<float>::max();
		// Find the top score for box i.
		for (int score_idx = 0; score_idx < num_classes_; ++score_idx) {
			auto score = raw_scores[i * num_classes_ + score_idx];
			if (tensorDetectorCalcOption.sigmoid_score) {
				if (tensorDetectorCalcOption.has_score_clipping_thresh) {
					score = score < -tensorDetectorCalcOption.score_clipping_thresh
						? -tensorDetectorCalcOption.score_clipping_thresh
						: score;
					score = score > tensorDetectorCalcOption.score_clipping_thresh
						? tensorDetectorCalcOption.score_clipping_thresh
						: score;
				}
				score = 1.0f / (1.0f + std::exp(-score));
			}
			if (max_score < score) {
				max_score = score;
				class_id = score_idx;
			}
		}
		detection_scores[i] = max_score;
		detection_classes[i] = class_id;
	}

	vector<Detection> output_detections;
	ConvertToDetections(tensorDetectorCalcOption, boxes.data(), detection_scores.data(), detection_classes.data(), output_detections);
	return output_detections;
}

int RemovalPadding(vector<Detection> &input_detections, array<float, 4> letterbox_padding)
{
	// Only process if there's input detections.
	const float left = letterbox_padding[0];
	const float top = letterbox_padding[1];
	const float left_and_right = letterbox_padding[0] + letterbox_padding[2];
	const float top_and_bottom = letterbox_padding[1] + letterbox_padding[3];

	int numKeys = input_detections[0].keyPoints.size();
	for (int i = 0; i < input_detections.size(); i++) {
		input_detections[i].xmin = (input_detections[i].xmin - left) / (1.0f - left_and_right);
		input_detections[i].ymin = (input_detections[i].ymin - top) / (1.0f - top_and_bottom);
		input_detections[i].xmax = (input_detections[i].xmax - left) / (1.0f - left_and_right);
		input_detections[i].ymax = (input_detections[i].ymax - top) / (1.0f - top_and_bottom);
		// Adjust keypoints as well.
		for (int j = 0; j < numKeys; j++) {
			input_detections[i].keyPoints[j].x = (input_detections[i].keyPoints[j].x - left) / (1.0f - left_and_right);
			input_detections[i].keyPoints[j].y = (input_detections[i].keyPoints[j].y - top) / (1.0f - top_and_bottom);
		}
	}
	return 0;
}

int RemovalPadding(LandmarkList &input_landmarks, array<float, 4> letterbox_padding)
{
	const float left = letterbox_padding[0];
	const float top = letterbox_padding[1];
	const float left_and_right = letterbox_padding[0] + letterbox_padding[2];
	const float top_and_bottom = letterbox_padding[1] + letterbox_padding[3];

	// Number of inputs and outpus is the same according to the contract.
	for (int i = 0; i < input_landmarks.vecLandmark.size(); ++i) {
		Landmark& landmark = input_landmarks.vecLandmark[i];
		landmark.x_ = (landmark.x_ - left) / (1.0f - left_and_right);
		landmark.y_ = (landmark.y_ - top) / (1.0f - top_and_bottom);
		landmark.z_ = landmark.z_ / (1.0f - left_and_right);  // Scale Z coordinate as X.
	}

	return 0;
}

int LandmarkProjection(LandmarkList &input_landmarks, RotRect &Roi)
{
	for (int i = 0; i < input_landmarks.vecLandmark.size(); ++i) {
		Landmark& landmark = input_landmarks.vecLandmark[i];

		const float x = landmark.x_ - 0.5f;
		const float y = landmark.y_ - 0.5f;
		const float angle = Roi.rotation;
		float new_x = cos(angle) * x - sin(angle) * y;
		float new_y = sin(angle) * x + cos(angle) * y;

		landmark.x_ = new_x * Roi.width + Roi.center_x;
		landmark.y_ = new_y * Roi.height + Roi.center_y;
		landmark.z_ *= Roi.width;  // Scale Z coordinate as X.
	}
	return 0;
}

Detection ConvertLandmarksToDetection(const LandmarkList& landmarks)
{
	Detection detection;
	float x_min = std::numeric_limits<float>::max();
	float x_max = std::numeric_limits<float>::min();
	float y_min = std::numeric_limits<float>::max();
	float y_max = std::numeric_limits<float>::min();
	for (int i = 0; i < landmarks.vecLandmark.size(); ++i) {
		const Landmark& landmark = landmarks.vecLandmark[i];
		x_min = std::min(x_min, landmark.x_);
		x_max = std::max(x_max, landmark.x_);
		y_min = std::min(y_min, landmark.y_);
		y_max = std::max(y_max, landmark.y_);

		Point2f keypoint;
		keypoint.x = landmark.x_;
		keypoint.y = landmark.y_;
		detection.keyPoints.push_back(keypoint);
	}
	detection.xmax = x_max;
	detection.xmin = x_min;
	detection.ymax = y_max;
	detection.ymin = y_min;
	return detection;
}

LandmarkList ProcessLandmark(const TensorsToLandmarksOptions& tensorfToLandmarksCalculatorOptions, char* kTag,
	float* raw_locations, int num_dimensions)
{
	LandmarkList outLandmarks;
	int num_landmarks_ = tensorfToLandmarksCalculatorOptions.num_landmarks;
	Landmark landmark;
	for (int ld = 0; ld < num_landmarks_; ++ld) {
		const int offset = ld * num_dimensions;
		if (tensorfToLandmarksCalculatorOptions.flip_horizontally) {
			landmark.x_ = (tensorfToLandmarksCalculatorOptions.input_image_width - raw_locations[offset]);
		}
		else {
			landmark.x_ = raw_locations[offset];
		}
		if (num_dimensions > 1) {
			if (tensorfToLandmarksCalculatorOptions.flip_vertically) {
				landmark.y_ = (tensorfToLandmarksCalculatorOptions.input_image_height - raw_locations[offset + 1]);
			}
			else {
				landmark.y_ = raw_locations[offset + 1];
			}
		}
		if (num_dimensions > 2) {
			landmark.z_ = raw_locations[offset + 2];
		}
		if (num_dimensions > 3) {
			landmark.visibility_ = raw_locations[offset + 3];
		}
		if (num_dimensions > 4) {
			landmark.presence_ = raw_locations[offset + 4];
		}
		outLandmarks.vecLandmark.push_back(landmark);
	}
	if (strcmp(kTag, "NORM_LANDMARKS") == 0) {
		for (int i = 0; i < num_landmarks_; i++) {
			outLandmarks.vecLandmark[i].x_ /= tensorfToLandmarksCalculatorOptions.input_image_width;
			outLandmarks.vecLandmark[i].y_ /= tensorfToLandmarksCalculatorOptions.input_image_height;
			outLandmarks.vecLandmark[i].z_ /= (tensorfToLandmarksCalculatorOptions.input_image_width * tensorfToLandmarksCalculatorOptions.normalize_z);
		}
	}
	return outLandmarks;
}
float GetObjectScale(const LandmarkList& landmarks, int image_width, int image_height)
{
	float x_min = FLT_MAX, x_max = -FLT_MAX, y_min = FLT_MAX, y_max = -FLT_MAX;
	for (int i = 0; i < landmarks.vecLandmark.size(); i++) {
		x_min = min(x_min, landmarks.vecLandmark[i].x_);
		y_min = min(y_min, landmarks.vecLandmark[i].y_);
		x_max = max(x_max, landmarks.vecLandmark[i].x_);
		x_max = max(y_max, landmarks.vecLandmark[i].y_);
	}

	const float object_width = (x_max - x_min) * image_width;
	const float object_height = (y_max - y_min) * image_height;

	return (object_width + object_height) / 2.0f;
}
//
//VelocityFilter::VelocityFilter(int window_size, float velocity_scale, float min_allowed_object_scale)
//	: window_size_(window_size),
//	velocity_scale_(velocity_scale),
//	min_allowed_object_scale_(min_allowed_object_scale)
//{
//}
//
//void VelocityFilter::Reset()
//{
//	x_filters_.clear();
//	y_filters_.clear();
//	z_filters_.clear();
//}
//
//int VelocityFilter::Apply(LandmarkList& in_landmarks, const std::pair<int, int>& image_size, const int64_t timestamp)
//{
//	// Get image size.
//	int image_width;
//	int image_height;
//	std::tie(image_width, image_height) = image_size;
//
//	const float object_scale = GetObjectScale(in_landmarks, image_width, image_height);
//	if (object_scale < min_allowed_object_scale_) {
//		//*out_landmarks = in_landmarks;
//		return 0;
//	}
//	const float value_scale = 1.0f / object_scale;
//
//	// Initialize filters once.
//	InitializeFiltersIfEmpty(in_landmarks.vecLandmark.size());
//
//	// Filter landmarks. Every axis of every landmark is filtered separately.
//	for (int i = 0; i < in_landmarks.vecLandmark.size(); ++i) {
//		in_landmarks.vecLandmark[i].x_ = x_filters_[i].Apply(timestamp, value_scale, in_landmarks.vecLandmark[i].x_ * image_width) / image_width;
//		in_landmarks.vecLandmark[i].y_ = y_filters_[i].Apply(timestamp, value_scale, in_landmarks.vecLandmark[i].y_ * image_height) / image_height;
//		// Scale Z the save was as X (using image width).
//		in_landmarks.vecLandmark[i].z_ = z_filters_[i].Apply(timestamp, value_scale, in_landmarks.vecLandmark[i].z_ * image_width) / image_width;
//	}
//
//	return 0;
//}
//
//int VelocityFilter::InitializeFiltersIfEmpty(const int n_landmarks)
//{
//	x_filters_.resize(n_landmarks, RelativeVelocityFilter(window_size_, velocity_scale_));
//	y_filters_.resize(n_landmarks, RelativeVelocityFilter(window_size_, velocity_scale_));
//	z_filters_.resize(n_landmarks, RelativeVelocityFilter(window_size_, velocity_scale_));
//
//	return 0;
//}
