
// CrystalBall_WinTesterDlg.cpp : implementation file
//

#include "stdafx.h"
#include "CrystalBall_WinTester.h"
#include "CrystalBall_WinTesterDlg.h"
//#include "afxdialogex.h"

#include <vector>
#include <opencv2/imgcodecs.hpp>
#include "opencv2/core.hpp";
#include "opencv2/photo.hpp"

using namespace std;
using namespace cv;
using namespace CrystalBall;

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CCrystalBall_WinTesterDlg dialog


CCrystalBall_WinTesterDlg::CCrystalBall_WinTesterDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(IDD_CRYSTALBALL_WINTESTER_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
	char strDataPath[MAX_PATH], *p;
	GetModuleFileNameA(NULL, strDataPath, MAX_PATH);
	p = strrchr(strDataPath, '\\');
	strcpy(p + 1, "data");

	m_MagicEngine.init(std::string(strDataPath));

}

void CCrystalBall_WinTesterDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CCrystalBall_WinTesterDlg, CDialogEx)
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BUTTON_VISIONMIX, &CCrystalBall_WinTesterDlg::OnBnClickedButtonVisionmix)
	ON_BN_CLICKED(IDC_BUTTON_EFFECT, &CCrystalBall_WinTesterDlg::OnBnClickedButtonEffect)
END_MESSAGE_MAP()


// CCrystalBall_WinTesterDlg message handlers

BOOL CCrystalBall_WinTesterDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// Set the icon for this dialog.  The framework does this automatically
	//  when the application's main window is not a dialog
	SetIcon(m_hIcon, TRUE);			// Set big icon
	SetIcon(m_hIcon, FALSE);		// Set small icon

	// TODO: Add extra initialization here

	return TRUE;  // return TRUE  unless you set the focus to a control
}

// If you add a minimize button to your dialog, you will need the code below
//  to draw the icon.  For MFC applications using the document/view model,
//  this is automatically done for you by the framework.

void CCrystalBall_WinTesterDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // device context for painting

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// Center icon in client rectangle
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// Draw the icon
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

// The system calls this function to obtain the cursor to display while the user drags
//  the minimized window.
HCURSOR CCrystalBall_WinTesterDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

double myinf = std::numeric_limits<double>::infinity();

class Domain_Filter
{
public:
	Mat ct_H, ct_V, horiz, vert, O, O_t, lower_idx, upper_idx;
	void init(const Mat &img, int flags, float sigma_s, float sigma_r);
	void getGradientx(const Mat &img, Mat &gx);
	void getGradienty(const Mat &img, Mat &gy);
	void diffx(const Mat &img, Mat &temp);
	void diffy(const Mat &img, Mat &temp);
	void find_magnitude(Mat &img, Mat &mag);
	void compute_boxfilter(Mat &output, Mat &hz, Mat &psketch, float radius);
	void compute_Rfilter(Mat &O, Mat &horiz, float sigma_h);
	void compute_NCfilter(Mat &O, Mat &horiz, Mat &psketch, float radius);
	void filter(const Mat &img, Mat &res, float sigma_s, float sigma_r, int flags);
	void pencil_sketch(const Mat &img, Mat &sketch, Mat &color_res, float sigma_s, float sigma_r, float shade_factor);
	void Depth_of_field(const Mat &img, Mat &img1, float sigma_s, float sigma_r);
};

void Domain_Filter::diffx(const Mat &img, Mat &temp)
{
	int channel = img.channels();

	for (int i = 0; i < img.size().height; i++)
		for (int j = 0; j < img.size().width - 1; j++)
		{
			for (int c = 0; c < channel; c++)
			{
				temp.at<float>(i, j*channel + c) =
					img.at<float>(i, (j + 1)*channel + c) - img.at<float>(i, j*channel + c);
			}
		}
}

void Domain_Filter::diffy(const Mat &img, Mat &temp)
{
	int channel = img.channels();

	for (int i = 0; i < img.size().height - 1; i++)
		for (int j = 0; j < img.size().width; j++)
		{
			for (int c = 0; c < channel; c++)
			{
				temp.at<float>(i, j*channel + c) =
					img.at<float>((i + 1), j*channel + c) - img.at<float>(i, j*channel + c);
			}
		}
}

void Domain_Filter::getGradientx(const Mat &img, Mat &gx)
{
	int w = img.cols;
	int h = img.rows;
	int channel = img.channels();

	for (int i = 0; i < h; i++)
		for (int j = 0; j < w; j++)
			for (int c = 0; c < channel; ++c)
			{
				gx.at<float>(i, j*channel + c) =
					img.at<float>(i, (j + 1)*channel + c) - img.at<float>(i, j*channel + c);
			}
}

void Domain_Filter::getGradienty(const Mat &img, Mat &gy)
{
	int w = img.cols;
	int h = img.rows;
	int channel = img.channels();

	for (int i = 0; i < h; i++)
		for (int j = 0; j < w; j++)
			for (int c = 0; c < channel; ++c)
			{
				gy.at<float>(i, j*channel + c) =
					img.at<float>(i + 1, j*channel + c) - img.at<float>(i, j*channel + c);

			}
}

void Domain_Filter::find_magnitude(Mat &img, Mat &mag)
{
	int h = img.rows;
	int w = img.cols;

	vector <Mat> planes;
	split(img, planes);

	Mat magXR = Mat(h, w, CV_32FC1);
	Mat magYR = Mat(h, w, CV_32FC1);

	Mat magXG = Mat(h, w, CV_32FC1);
	Mat magYG = Mat(h, w, CV_32FC1);

	Mat magXB = Mat(h, w, CV_32FC1);
	Mat magYB = Mat(h, w, CV_32FC1);

	Sobel(planes[0], magXR, CV_32FC1, 1, 0, 3);
	Sobel(planes[0], magYR, CV_32FC1, 0, 1, 3);

	Sobel(planes[1], magXG, CV_32FC1, 1, 0, 3);
	Sobel(planes[1], magYG, CV_32FC1, 0, 1, 3);

	Sobel(planes[2], magXB, CV_32FC1, 1, 0, 3);
	Sobel(planes[2], magYB, CV_32FC1, 0, 1, 3);

	Mat mag1 = Mat(h, w, CV_32FC1);
	Mat mag2 = Mat(h, w, CV_32FC1);
	Mat mag3 = Mat(h, w, CV_32FC1);

	magnitude(magXR, magYR, mag1);
	magnitude(magXG, magYG, mag2);
	magnitude(magXB, magYB, mag3);

	mag = mag1 + mag2 + mag3;
	mag = 1.0f - mag;
}

void Domain_Filter::compute_Rfilter(Mat &output, Mat &hz, float sigma_h)
{
	int h = output.rows;
	int w = output.cols;
	int channel = output.channels();

	float a = (float)exp((-1.0 * sqrt(2.0)) / sigma_h);

	Mat temp = Mat(h, w, CV_32FC3);

	output.copyTo(temp);
	Mat V = Mat(h, w, CV_32FC1);

	for (int i = 0; i < h; i++)
		for (int j = 0; j < w; j++)
			V.at<float>(i, j) = pow(a, hz.at<float>(i, j));

	for (int i = 0; i < h; i++)
	{
		for (int j = 1; j < w; j++)
		{
			for (int c = 0; c < channel; c++)
			{
				temp.at<float>(i, j*channel + c) = temp.at<float>(i, j*channel + c) +
					(temp.at<float>(i, (j - 1)*channel + c) - temp.at<float>(i, j*channel + c)) * V.at<float>(i, j);
			}
		}
	}

	for (int i = 0; i < h; i++)
	{
		for (int j = w - 2; j >= 0; j--)
		{
			for (int c = 0; c < channel; c++)
			{
				temp.at<float>(i, j*channel + c) = temp.at<float>(i, j*channel + c) +
					(temp.at<float>(i, (j + 1)*channel + c) - temp.at<float>(i, j*channel + c))*V.at<float>(i, j + 1);
			}
		}
	}

	temp.copyTo(output);
}

void Domain_Filter::compute_boxfilter(Mat &output, Mat &hz, Mat &psketch, float radius)
{
	int h = output.rows;
	int w = output.cols;
	Mat lower_pos = Mat(h, w, CV_32FC1);
	Mat upper_pos = Mat(h, w, CV_32FC1);

	lower_pos = hz - radius;
	upper_pos = hz + radius;

	lower_idx = Mat::zeros(h, w, CV_32FC1);
	upper_idx = Mat::zeros(h, w, CV_32FC1);

	Mat domain_row = Mat::zeros(1, w + 1, CV_32FC1);

	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
			domain_row.at<float>(0, j) = hz.at<float>(i, j);
		domain_row.at<float>(0, w) = (float)myinf;

		Mat lower_pos_row = Mat::zeros(1, w, CV_32FC1);
		Mat upper_pos_row = Mat::zeros(1, w, CV_32FC1);

		for (int j = 0; j < w; j++)
		{
			lower_pos_row.at<float>(0, j) = lower_pos.at<float>(i, j);
			upper_pos_row.at<float>(0, j) = upper_pos.at<float>(i, j);
		}

		Mat temp_lower_idx = Mat::zeros(1, w, CV_32FC1);
		Mat temp_upper_idx = Mat::zeros(1, w, CV_32FC1);

		for (int j = 0; j<w; j++)
		{
			if (domain_row.at<float>(0, j) > lower_pos_row.at<float>(0, 0))
			{
				temp_lower_idx.at<float>(0, 0) = (float)j;
				break;
			}
		}
		for (int j = 0; j<w; j++)
		{
			if (domain_row.at<float>(0, j) > upper_pos_row.at<float>(0, 0))
			{
				temp_upper_idx.at<float>(0, 0) = (float)j;
				break;
			}
		}

		int temp = 0;
		for (int j = 1; j < w; j++)
		{
			int count = 0;
			for (int k = (int)temp_lower_idx.at<float>(0, j - 1); k < w + 1; k++)
			{
				if (domain_row.at<float>(0, k) > lower_pos_row.at<float>(0, j))
				{
					temp = count;
					break;
				}
				count++;
			}

			temp_lower_idx.at<float>(0, j) = temp_lower_idx.at<float>(0, j - 1) + temp;

			count = 0;
			for (int k = (int)temp_upper_idx.at<float>(0, j - 1); k < w + 1; k++)
			{


				if (domain_row.at<float>(0, k) > upper_pos_row.at<float>(0, j))
				{
					temp = count;
					break;
				}
				count++;
			}

			temp_upper_idx.at<float>(0, j) = temp_upper_idx.at<float>(0, j - 1) + temp;
		}

		for (int j = 0; j < w; j++)
		{
			lower_idx.at<float>(i, j) = temp_lower_idx.at<float>(0, j) + 1;
			upper_idx.at<float>(i, j) = temp_upper_idx.at<float>(0, j) + 1;
		}

	}
	psketch = upper_idx - lower_idx;
}
void Domain_Filter::compute_NCfilter(Mat &output, Mat &hz, Mat &psketch, float radius)
{
	int h = output.rows;
	int w = output.cols;
	int channel = output.channels();

	compute_boxfilter(output, hz, psketch, radius);

	Mat box_filter = Mat::zeros(h, w + 1, CV_32FC3);

	for (int i = 0; i < h; i++)
	{
		box_filter.at<float>(i, 1 * channel + 0) = output.at<float>(i, 0 * channel + 0);
		box_filter.at<float>(i, 1 * channel + 1) = output.at<float>(i, 0 * channel + 1);
		box_filter.at<float>(i, 1 * channel + 2) = output.at<float>(i, 0 * channel + 2);
		for (int j = 2; j < w + 1; j++)
		{
			for (int c = 0; c < channel; c++)
				box_filter.at<float>(i, j*channel + c) = output.at<float>(i, (j - 1)*channel + c) + box_filter.at<float>(i, (j - 1)*channel + c);
		}
	}

	Mat indices = Mat::zeros(h, w, CV_32FC1);
	Mat final = Mat::zeros(h, w, CV_32FC3);

	for (int i = 0; i < h; i++)
		for (int j = 0; j < w; j++)
			indices.at<float>(i, j) = (float)i + 1;

	Mat a = Mat::zeros(h, w, CV_32FC1);
	Mat b = Mat::zeros(h, w, CV_32FC1);

	// Compute the box filter using a summed area table.
	for (int c = 0; c < channel; c++)
	{
		Mat flag = Mat::ones(h, w, CV_32FC1);
		multiply(flag, c + 1, flag);

		Mat temp1, temp2;
		multiply(flag - 1, h*(w + 1), temp1);
		multiply(lower_idx - 1, h, temp2);
		a = temp1 + temp2 + indices;

		multiply(flag - 1, h*(w + 1), temp1);
		multiply(upper_idx - 1, h, temp2);
		b = temp1 + temp2 + indices;

		int p, q, r, rem;
		int p1, q1, r1, rem1;

		// Calculating indices
		for (int i = 0; i < h; i++)
		{
			for (int j = 0; j < w; j++)
			{

				r = (int)b.at<float>(i, j) / (h*(w + 1));
				rem = (int)b.at<float>(i, j) - r*h*(w + 1);
				q = rem / h;
				p = rem - q*h;
				if (q == 0)
				{
					p = h;
					q = w;
					r = r - 1;
				}
				if (p == 0)
				{
					p = h;
					q = q - 1;
				}

				r1 = (int)a.at<float>(i, j) / (h*(w + 1));
				rem1 = (int)a.at<float>(i, j) - r1*h*(w + 1);
				q1 = rem1 / h;
				p1 = rem1 - q1*h;
				if (p1 == 0)
				{
					p1 = h;
					q1 = q1 - 1;
				}

				final.at<float>(i, j*channel + 2 - c) = (box_filter.at<float>(p - 1, q*channel + (2 - r)) - box_filter.at<float>(p1 - 1, q1*channel + (2 - r1)))
					/ (upper_idx.at<float>(i, j) - lower_idx.at<float>(i, j));
			}
		}
	}

	final.copyTo(output);
}
void Domain_Filter::init(const Mat &img, int flags, float sigma_s, float sigma_r)
{
	int h = img.size().height;
	int w = img.size().width;
	int channel = img.channels();

	////////////////////////////////////     horizontal and vertical partial derivatives /////////////////////////////////

	Mat derivx = Mat::zeros(h, w - 1, CV_32FC3);
	Mat derivy = Mat::zeros(h - 1, w, CV_32FC3);

	diffx(img, derivx);
	diffy(img, derivy);

	Mat distx = Mat::zeros(h, w, CV_32FC1);
	Mat disty = Mat::zeros(h, w, CV_32FC1);

	//////////////////////// Compute the l1-norm distance of neighbor pixels ////////////////////////////////////////////////

	for (int i = 0; i < h; i++)
		for (int j = 0, k = 1; j < w - 1; j++, k++)
			for (int c = 0; c < channel; c++)
			{
				distx.at<float>(i, k) =
					distx.at<float>(i, k) + abs(derivx.at<float>(i, j*channel + c));
			}

	for (int i = 0, k = 1; i < h - 1; i++, k++)
		for (int j = 0; j < w; j++)
			for (int c = 0; c < channel; c++)
			{
				disty.at<float>(k, j) =
					disty.at<float>(k, j) + abs(derivy.at<float>(i, j*channel + c));
			}

	////////////////////// Compute the derivatives of the horizontal and vertical domain transforms. /////////////////////////////

	horiz = Mat(h, w, CV_32FC1);
	vert = Mat(h, w, CV_32FC1);

	Mat final = Mat(h, w, CV_32FC3);

	Mat tempx, tempy;
	multiply(distx, sigma_s / sigma_r, tempx);
	multiply(disty, sigma_s / sigma_r, tempy);

	horiz = 1.0f + tempx;
	vert = 1.0f + tempy;

	O = Mat(h, w, CV_32FC3);
	img.copyTo(O);

	O_t = Mat(w, h, CV_32FC3);

	if (flags == 2)
	{

		ct_H = Mat(h, w, CV_32FC1);
		ct_V = Mat(h, w, CV_32FC1);

		for (int i = 0; i < h; i++)
		{
			ct_H.at<float>(i, 0) = horiz.at<float>(i, 0);
			for (int j = 1; j < w; j++)
			{
				ct_H.at<float>(i, j) = horiz.at<float>(i, j) + ct_H.at<float>(i, j - 1);
			}
		}

		for (int j = 0; j < w; j++)
		{
			ct_V.at<float>(0, j) = vert.at<float>(0, j);
			for (int i = 1; i < h; i++)
			{
				ct_V.at<float>(i, j) = vert.at<float>(i, j) + ct_V.at<float>(i - 1, j);
			}
		}
	}

}

void Domain_Filter::filter(const Mat &img, Mat &res, float sigma_s = 60, float sigma_r = 0.4, int flags = 1)
{
	int no_of_iter = 3;
	int h = img.size().height;
	int w = img.size().width;
	float sigma_h = sigma_s;

	init(img, flags, sigma_s, sigma_r);

	if (flags == 1)
	{
		Mat vert_t = vert.t();

		for (int i = 0; i < no_of_iter; i++)
		{
			sigma_h = (float)(sigma_s * sqrt(3.0) * pow(2.0, (no_of_iter - (i + 1))) / sqrt(pow(4.0, no_of_iter) - 1));

			compute_Rfilter(O, horiz, sigma_h);

			O_t = O.t();

			compute_Rfilter(O_t, vert_t, sigma_h);

			O = O_t.t();

		}
	}
	else if (flags == 2)
	{

		Mat vert_t = ct_V.t();
		Mat temp = Mat(h, w, CV_32FC1);
		Mat temp1 = Mat(w, h, CV_32FC1);

		float radius;

		for (int i = 0; i < no_of_iter; i++)
		{
			sigma_h = (float)(sigma_s * sqrt(3.0) * pow(2.0, (no_of_iter - (i + 1))) / sqrt(pow(4.0, no_of_iter) - 1));

			radius = (float)sqrt(3.0) * sigma_h;

			compute_NCfilter(O, ct_H, temp, radius);

			O_t = O.t();

			compute_NCfilter(O_t, vert_t, temp1, radius);

			O = O_t.t();
		}
	}

	res = O.clone();
}

void Domain_Filter::pencil_sketch(const Mat &img, Mat &sketch, Mat &color_res, float sigma_s, float sigma_r, float shade_factor)
{

	int no_of_iter = 3;
	init(img, 2, sigma_s, sigma_r);
	int h = img.size().height;
	int w = img.size().width;

	/////////////////////// convert to YCBCR model for color pencil drawing //////////////////////////////////////////////////////

	Mat color_sketch = Mat(h, w, CV_32FC3);

	cvtColor(img, color_sketch, COLOR_BGR2YCrCb);

	vector <Mat> YUV_channel;
	Mat vert_t = ct_V.t();

	float sigma_h = sigma_s;

	Mat penx = Mat(h, w, CV_32FC1);

	Mat pen_res = Mat::zeros(h, w, CV_32FC1);
	Mat peny = Mat(w, h, CV_32FC1);

	Mat peny_t;

	float radius;

	for (int i = 0; i < no_of_iter; i++)
	{
		sigma_h = (float)(sigma_s * sqrt(3.0) * pow(2.0, (no_of_iter - (i + 1))) / sqrt(pow(4.0, no_of_iter) - 1));

		radius = (float)sqrt(3.0) * sigma_h;

		compute_boxfilter(O, ct_H, penx, radius);

		O_t = O.t();

		compute_boxfilter(O_t, vert_t, peny, radius);

		O = O_t.t();

		peny_t = peny.t();

		for (int k = 0; k < h; k++)
			for (int j = 0; j < w; j++)
				pen_res.at<float>(k, j) = (shade_factor * (penx.at<float>(k, j) + peny_t.at<float>(k, j)));

		if (i == 0)
		{
			sketch = pen_res.clone();
			split(color_sketch, YUV_channel);
			pen_res.copyTo(YUV_channel[0]);
			merge(YUV_channel, color_sketch);
			cvtColor(color_sketch, color_res, COLOR_YCrCb2BGR);
		}

	}
}

void detailEnhance2(InputArray _src, OutputArray dst, float sigma_s, float sigma_r)
{
// 	CV_INSTRUMENT_REGION();

	Mat I = _src.getMat();

	float factor = 2.0f;

	Mat lab;
	I.convertTo(lab, CV_32FC3, 1.0 / 255.0);

	vector <Mat> lab_channel;
	cvtColor(lab, lab, COLOR_BGR2Lab);
	split(lab, lab_channel);

	//////////////////////////////////////////////////////////////////////////
	Mat L;
	lab_channel[0].convertTo(lab_channel[0], CV_32FC1, 1.0 / 255.0);

	Domain_Filter obj;

	Mat res;
// 	resize(lab_channel[0], lab_channel[0], Size(lab_channel[0].cols/2, lab_channel[0].rows/2));
// 	obj.filter(lab_channel[0], res, sigma_s, sigma_r, 1);

// 	Mat bilateral, canny;
// 	cv::Canny(lab_channel[0], canny, 10, 100);

	cv::bilateralFilter(lab_channel[0], res, 5, 75, 75);

	Mat detail = lab_channel[0] - res;
	multiply(detail, factor, detail);
	lab_channel[0] = res + detail;

	lab_channel[0].convertTo(lab_channel[0], CV_32FC1, 255);
	//////////////////////////////////////////////////////////////////////////

// 	resize(lab_channel[0], lab_channel[0], Size(I.cols, I.rows));

	merge(lab_channel, lab);

	cvtColor(lab, lab, COLOR_Lab2BGR);
	lab.convertTo(dst, CV_8UC3, 255);
}

cv::Mat detailEnhance3(cv::Mat& srcImage, int smoothParam)
{
	Mat bilateral, canny;
	cv::Canny(srcImage, canny, 10, 100);
	cv::bilateralFilter(srcImage, bilateral, 9, 75, 75);
	Mat pImageResult = (1.0 - smoothParam / 100.f) * (srcImage - bilateral) + bilateral;
	imshow("canny filter", canny);
	imshow("bilateral filter", pImageResult);

	return pImageResult;
}

void CCrystalBall_WinTesterDlg::OnBnClickedButtonVisionmix()
{
	CFileDialog OpenDlg(TRUE, NULL, NULL, OFN_HIDEREADONLY | OFN_READONLY,
		_T("All Files (*.*)|*.*;\
		   |Bitmap Files(*.BMP) | *.BMP;\
		   |GIF(*.GIF) | *.GIF;\
		   |TIFF(*.TIFF) | *.TIFF;\
		   |PNG(*.PNG) | *.PNG;\
		   |JPEG(*.JPG; *.JPEG) | *.JPG; *.JPEG||"));


	LPCTSTR szTitle = _T("File Open");
	OpenDlg.m_ofn.lpstrTitle = szTitle;
	if (OpenDlg.DoModal() != IDOK) return;

	CString strPath = OpenDlg.GetPathName();
	int wid, hei, bitcnt;
	
	cv::Mat srcImg = cv::imread(std::string(strPath.GetBuffer()));
	wid = srcImg.cols;
	hei = srcImg.rows;
	cv::Mat maskImg[3];
	ncnn::Mat inputImage = ncnn::Mat::from_pixels_resize(srcImg.data, ncnn::Mat::PIXEL_RGB, wid, hei, 224, 224);
	m_MagicEngine.trimapMask(inputImage, maskImg[0]);

	cv::Mat read_mask = imread("C:/Users/kki/Documents/DB/1.png", IMREAD_UNCHANGED);
	cv::Mat splits[4];
	cv::split(read_mask, splits);

	resize(splits[3], maskImg[0], Size(224, 224));
	cv::imshow("read mask", maskImg[0]);		imwrite("C:/Users/kki/Documents/DB/test/maskImg[0].png", maskImg[0]);

	cv::Mat origin = srcImg.clone();

#if 1 //test for clarity
	Mat clarity;
	detailEnhance2(origin, clarity, (float)5, 0.15);

	cv::imshow("src", origin);  imwrite("C:/Users/kki/Documents/DB/test/origin.jpg", origin);
	cv::imshow("dst", clarity);	imwrite("C:/Users/kki/Documents/DB/test/clarity.jpg", clarity);

	clarity = detailEnhance3(origin, 50);
	cv::imshow("dst2", clarity);	imwrite("C:/Users/kki/Documents/DB/test/clarity2.jpg", clarity);

	return;
#endif

#if 1 //for test
	resize(srcImg, srcImg, cv::Size(224, 224));

	maskImg[1] = maskImg[0].clone();
	maskImg[2] = maskImg[0].clone();

	Mat synImg;
	merge(maskImg, 3, synImg);
	synImg = 0.5*srcImg + 0.5*synImg;

// 	cv::imshow("src", srcImg);			imwrite("C:/Users/kki/Documents/DB/test/src.png", srcImg);
	cv::imshow("trimap-0", maskImg[0]); imwrite("C:/Users/kki/Documents/DB/test/trimap-0.png", maskImg[0]);
	cv::imshow("synImg", synImg);		imwrite("C:/Users/kki/Documents/DB/test/synImg.png", synImg);
#endif

	cv::Mat tri = maskImg[0].clone();
// 	cv::Mat matImage1 = m_MagicEngine.mattingImageColorLineLaplace2(origin, tri);
// 	origin = srcImg.clone();
// 	tri = maskImg[0].clone();
// 	cv::Mat matImage2 = m_MagicEngine.mattingImageColorLineLaplace(origin, tri);
// 	origin = srcImg.clone();
// 	tri = maskImg[0].clone();
// 	cv::Mat matImage3 = m_MagicEngine.mattingImageInformationFlowDouble(origin, tri);
// 	origin = srcImg.clone();
// 	tri = maskImg[0].clone();
// 	cv::Mat matImage4 = m_MagicEngine.mattingImageInformationFlowFloat(origin, tri);
//  cv::Mat matImage5 = m_MagicEngine.mattingImageGuideFilter(origin, tri);
	origin = srcImg.clone();
	tri = maskImg[0].clone();
	cv::Mat matImage6 = m_MagicEngine.mattingImageInformationFlowFloatGuideFilter(origin, tri);

// 	cv::Mat splits[4];
// 	cv::split(matImage, splits);
// 	cv::Mat tmp;
// 	tmp = splits[0];
// 	splits[0] = splits[2];
// 	splits[2] = tmp;
// 	cv::merge(splits, 4, matImage);
// 
// 	cv::imshow("matting", matImage);
// 
// 	cv::imwrite("D:/mattingTrimap.png", matImage);


// 	cv::imshow("matting1", matImage1);
// 	cv::imshow("matting2", matImage2);
// 	cv::imshow("matting3", matImage3);
// 	cv::imshow("matting4", matImage4);
// 	cv::imshow("matting5", matImage5);
	cv::imwrite("C:/Users/kki/Documents/DB/test/matImage6.png", matImage6); cv::imshow("matting6", matImage6);
}


void CCrystalBall_WinTesterDlg::OnBnClickedButtonEffect()
{
	CFileDialog OpenDlg(TRUE, NULL, NULL, OFN_HIDEREADONLY | OFN_READONLY,
		_T("All Files (*.*)|*.*;\
		   |Bitmap Files(*.BMP) | *.BMP;\
		   |GIF(*.GIF) | *.GIF;\
		   |TIFF(*.TIFF) | *.TIFF;\
		   |PNG(*.PNG) | *.PNG;\
		   |JPEG(*.JPG; *.JPEG) | *.JPG; *.JPEG||"));


	LPCTSTR szTitle = _T("File Open");
	OpenDlg.m_ofn.lpstrTitle = szTitle;
	if (OpenDlg.DoModal() != IDOK) return;

	CString strPath = OpenDlg.GetPathName();
	int wid, hei, bitcnt;

	cv::Mat srcImg = cv::imread(std::string(strPath.GetBuffer()));
	wid = srcImg.cols;
	hei = srcImg.rows;
	cv::Mat maskImg[3];
	ncnn::Mat inputImage = ncnn::Mat::from_pixels_resize(srcImg.data, ncnn::Mat::PIXEL_RGB, wid, hei, 224, 224);
	m_MagicEngine.trimapMask(inputImage, maskImg[0]);

	cv::Mat origin = srcImg.clone();

#if 0 //for test
	resize(srcImg, srcImg, cv::Size(224, 224));

	maskImg[1] = maskImg[0].clone();
	maskImg[2] = maskImg[0].clone();

	Mat synImg;
	merge(maskImg, 3, synImg);
	synImg = 0.5*srcImg + 0.5*synImg;

	cv::imshow("src", srcImg);			imwrite("D:/crystal/src.png", srcImg);
	cv::imshow("trimap-0", maskImg[0]); imwrite("D:/crystal/trimap-0.png", maskImg[0]);
	cv::imshow("synImg", synImg);		imwrite("D:/crystal/synImg.png", synImg);
#endif

	cv::Mat tri = maskImg[0].clone();
	origin = srcImg.clone();
	tri = maskImg[0].clone();

#if 1
	// Read image
	Mat im = srcImg.clone();
	Mat imout, imout_gray;

	// Edge preserving filter with two different flags.
	edgePreservingFilter(im, imout, RECURS_FILTER);
	imshow("edge-preserving-recursive-filter", imout);
	imwrite("D:/edge-preserving-recursive-filter.jpg", imout);

	edgePreservingFilter(im, imout, NORMCONV_FILTER);
	imshow("edge-preserving-normalized-convolution-filter", imout);
	imwrite("D:/edge-preserving-normalized-convolution-filter.jpg", imout);

	// Detail enhance filter
	detailEnhance(im, imout);
	imshow("detail-enhance", imout);
	imwrite("D:/detail-enhance.jpg", imout);

	// Pencil sketch filter
	pencilSketch(im, imout_gray, imout);
	imshow("pencil-sketch", imout);
	imwrite("D:/pencil-sketch.jpg", imout_gray);

	// Stylization filter
	stylization(im, imout);
	imshow("stylization", imout);
	imwrite("D:/stylization.jpg", imout);

	return;
#endif

#if 1 //test done
	cv::Mat matImage6 = m_MagicEngine.effectImageBackBlur(origin, tri, true);
// 	cv::Mat matImage6 = m_MagicEngine.effectImageColorOnYou(origin, tri);
#endif

// 	cv::Mat matImage6 = m_MagicEngine.effectImageFaceBeauty(origin, 50);

#if 0
	cv::Mat frmImage  = cv::imread("D:/Repositories/CrystalBall-WinTester/Resource/theme_02-2.png", IMREAD_UNCHANGED);
	cv::Mat srcImage1 = cv::imread("D:/Repositories/CrystalBall-WinTester/Resource/theme_01.bmp");
	cv::Mat srcImage2 = cv::imread("D:/Repositories/CrystalBall-WinTester/Resource/theme_09.bmp");
	Point2f pt1[4] = { { 46, 407 }, { 74, 957 }, { 548, 913 }, { 510, 400 } };
	Point2f pt2[4] = { { 638, 498}, { 814, 998 }, { 1258, 838 }, { 1097, 364 } };
	cv::Mat matImage6 = m_MagicEngine.effectImageRomantic(frmImage, srcImage1, srcImage2, pt1, pt2);
#endif

#if 0
	cv::Mat matImage = m_MagicEngine.mattingImageInformationFlowFloatGuideFilter(origin, tri);

	cv::Mat splits[4];
	cv::split(matImage, splits);
	cv::Mat tmp;
	tmp = splits[0];
	splits[0] = splits[2];
	splits[2] = tmp;
	cv::merge(splits, 4, matImage);

	cv::imshow("matting", matImage);

	cv::imwrite("D:/mattingTrimap.png", matImage);
#endif

	// 	cv::imshow("matting1", matImage1);
	// 	cv::imshow("matting2", matImage2);
	// 	cv::imshow("matting3", matImage3);
	// 	cv::imshow("matting4", matImage4);
	// 	cv::imshow("matting5", matImage5);
	cv::imwrite("D:/matImage6.png", matImage6); cv::imshow("matting6", matImage6);
}
