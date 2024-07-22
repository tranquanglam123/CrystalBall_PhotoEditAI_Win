 #include "landmark68_dlib_lcl.h"
#include <math.h>
#include <memory.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

using namespace cv;

#define SIGN(a, b) ((b) >= 0.0 ? fabs(a) : -fabs(a))

const float coef[40] = { -0.091010,1.724049,1.037068,-0.443753,-0.339756,-0.201965,0.988396,-1.655543,-0.657768,0.609142,
						-0.002318,3.295552,-0.127733,-1.730172,-0.057874,0.062312,0.325592,-3.239637,-0.202194,2.630754,
						0.385387,-3.111044,0.863330,2.185272,-0.345650,0.415095,-0.327744,3.827341,0.498649,-3.359475,
						0.532066,-1.899927,-0.329757,1.059675,-0.062350,0.737815,-1.121765,2.645568,1.065651,-1.605761 };

static ShapeModel* shape = NULL;
static int count_of_init = 0;
int LM68_dlib_finish()
{
	if (count_of_init == 0) return 0;
	if (count_of_init == 1) {
		SafeMemDelete(shape);
		count_of_init = 0;
		return 0;
	}
	count_of_init --;
	return 0;
}

int LM68_dlib_init(const char* model_path)
{
	char ModelPath[260];
	if (count_of_init > 0) {
		count_of_init ++;
		return 0;
	}
	if(shape != NULL) return -1;
	sprintf(ModelPath, "%s/lm68_dlib.dat", model_path);
	FILE *fp = fopen(ModelPath, "rb");
	if(fp == NULL) {
		count_of_init = 0;
	    return -1;
    }
	shape = new ShapeModel;
	if (shape == NULL) {
		fclose(fp);
		count_of_init = 0;
		return -1;
	}
	for (int i = 0; i < LANDMARK_COUNT; i++) {
		fread(&shape->initShape.data[i].point[0], sizeof(float), 1, fp);
		fread(&shape->initShape.data[i].point[1], sizeof(float), 1, fp);
		shape->initShape.data_x[i] = shape->initShape.data[i].point[0];
		shape->initShape.data_y[i] = shape->initShape.data[i].point[1];
	}	
	for (int i = 0; i < ITERATE_NUM; i++) {
		fread(shape->anchor[i].idx, sizeof(int), SHAPE_FEATURE_NUM, fp);
		for (int j = 0; j < SHAPE_FEATURE_NUM; j++) {
			fread(&shape->delta[i].delta_value[j].point[0], sizeof(float), 1, fp);
			fread(&shape->delta[i].delta_value[j].point[1], sizeof(float), 1, fp);
			for (int n = 0; n < 15; n++) {
				fread(&shape->forest[i].forestinfo[j].splite[n].idx1, sizeof(int), 1, fp);
				fread(&shape->forest[i].forestinfo[j].splite[n].idx2, sizeof(int), 1, fp);
				fread(&shape->forest[i].forestinfo[j].splite[n].thresh, sizeof(float), 1, fp);
			}
			for (int n = 0; n < 16; n++) {
				for (int m = 0; m < LANDMARK_COUNT; m++) {
					fread(&shape->forest[i].forestinfo[j].leaf_value[n].data[m].point[0], sizeof(float), 1, fp);
					fread(&shape->forest[i].forestinfo[j].leaf_value[n].data[m].point[1], sizeof(float), 1, fp);
					shape->forest[i].forestinfo[j].leaf_value[n].data_x[m] = shape->forest[i].forestinfo[j].leaf_value[n].data[m].point[0];
					shape->forest[i].forestinfo[j].leaf_value[n].data_y[m] = shape->forest[i].forestinfo[j].leaf_value[n].data[m].point[1];
				}
			}
		}
	}
    count_of_init ++;
	return 0;
}

inline ShapePoint point_diff(ShapePoint sp1, ShapePoint sp2)
{
	ShapePoint pd;
	pd.point[0] = sp1.point[0] - sp2.point[0];
	pd.point[1] = sp1.point[1] - sp2.point[1];
	return pd;
}

inline float length_squared(ShapePoint sp)
{
	return sp.point[0]*sp.point[0] + sp.point[1]*sp.point[1];
}

inline float length_squared(ShapePoint sp1, ShapePoint sp2)
{
	return sp1.point[0]*sp2.point[0] + sp1.point[1]*sp2.point[1];
}

inline float pythag1(float a, float b)
{
	float absa,absb;
	absa=fabs(a);
	absb=fabs(b);
	if (absa > absb) return (float)(absa*sqrt(1.0+SQR(absb/absa)));
	else return (float)(absb == 0.0 ? 0.0 : absb*sqrt(1.0+SQR(absa/absb)));
}

inline void svdcmp(float **a, int m, int n, float w[], float **v)
{
	//float pythag(float a, float b);
	int flag,i,its,j,jj,k,l,nm;
	float anorm,c,f,g,h,s,scale,x,y,z,*rv1;
	rv1 = (float*)calloc(n+1, sizeof(float));
	g=scale=anorm=0.0;
	for (i=1;i<=n;i++) {
		l=i+1;
		rv1[i]=scale*g;
		g=s=scale=0.0;
		if (i <= m) {
			for (k=i;k<=m;k++) scale += fabs(a[k][i]);
			if (scale) {
				for (k=i;k<=m;k++) {
					a[k][i] /= scale;
					s += a[k][i]*a[k][i];
				}
				f=a[i][i];
				g = -SIGN(sqrt(s),f);
				h=f*g-s;
				a[i][i]=f-g;
				for (j=l;j<=n;j++) {
					for (s=0.0,k=i;k<=m;k++) s += a[k][i]*a[k][j];
					f=s/h;
					for (k=i;k<=m;k++) a[k][j] += f*a[k][i];
				}
				for (k=i;k<=m;k++) a[k][i] *= scale;
			}
		}
		w[i]=scale *g;
		g=s=scale=0.0;
		if (i <= m && i != n) {
			for (k=l;k<=n;k++) scale += fabs(a[i][k]);
			if (scale) {
				for (k=l;k<=n;k++) {
					a[i][k] /= scale;
					s += a[i][k]*a[i][k];
				}
				f=a[i][l];
				g = -SIGN(sqrt(s),f);
				h=f*g-s;
				a[i][l]=f-g;
				for (k=l;k<=n;k++) rv1[k]=a[i][k]/h;
				for (j=l;j<=m;j++) {
					for (s=0.0,k=l;k<=n;k++) s += a[j][k]*a[i][k];
					for (k=l;k<=n;k++) a[j][k] += s*rv1[k];
				}
				for (k=l;k<=n;k++) a[i][k] *= scale;
			}
		}
		anorm=(float)FMAX(anorm,(fabs(w[i])+fabs(rv1[i])));
	}
	for (i=n;i>=1;i--) { 
		if (i < n) {
			if (g) {
				for (j=l;j<=n;j++)
					v[j][i]=(a[i][j]/a[i][l])/g;
				for (j=l;j<=n;j++) {
					for (s=0.0,k=l;k<=n;k++) s += a[i][k]*v[k][j];
					for (k=l;k<=n;k++) v[k][j] += s*v[k][i];
				}
			}
			for (j=l;j<=n;j++) v[i][j]=v[j][i]=0.0;
		}
		v[i][i]=1.0;
		g=rv1[i];
		l=i;
	}
	for (i=IMIN(m,n);i>=1;i--) { 
		l=i+1;
		g=w[i];
		for (j=l;j<=n;j++) a[i][j]=0.0;
		if (g) {
			g=1.0f/g;
			for (j=l;j<=n;j++) {
				for (s=0.0,k=l;k<=m;k++) s += a[k][i]*a[k][j];
				f=(s/a[i][i])*g;
				for (k=i;k<=m;k++) a[k][j] += f*a[k][i];
			}
			for (j=i;j<=m;j++) a[j][i] *= g;
		} else 
			for (j=i;j<=m;j++) a[j][i]=0.0;
		++a[i][i];
	}
	for (k=n;k>=1;k--) {
		for (its=1;its<=30;its++) {
			flag=1;
			for (l=k;l>=1;l--) { 
				nm=l-1; 
				if ((float)(fabs(rv1[l])+anorm) == anorm) {
					flag=0;
					break;
				}
				if ((float)(fabs(w[nm])+anorm) == anorm) break;
			}
			if (flag) {
				c=0.0;
				s=1.0;
				for (i=l;i<=k;i++) {
					f=s*rv1[i];
					rv1[i]=c*rv1[i];
					if ((float)(fabs(f)+anorm) == anorm) break;
					g=w[i];
					h=pythag1(f,g);
					w[i]=h;
					h=1.0f/h;
					c=g*h;
					s = -f*h;
					for (j=1;j<=m;j++) {
						y=a[j][nm];
						z=a[j][i];
						a[j][nm]=y*c+z*s;
						a[j][i]=z*c-y*s;
					}
				}
			}
			z=w[k];
			if (l == k) { 
				if (z < 0.0) { 
					w[k] = -z;
					for (j=1;j<=n;j++) v[j][k] = -v[j][k];
				}
				break;
			}
			if (its == 30) return;
			x=w[l];
			nm=k-1;
			y=w[nm];
			g=rv1[nm];
			h=rv1[k];
			f=(float)(((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y));
			g=pythag1(f,1.0);
			f=((x-z)*(x+z)+h*((y/(f+SIGN(g,f)))-h))/x;
			c=s=1.0; 
			for (j=l;j<=nm;j++) {
				i=j+1;
				g=rv1[i];
				y=w[i];
				h=s*g;
				g=c*g;
				z=pythag1(f,h);
				rv1[j]=z;
				c=f/z;
				s=h/z;
				f=x*c+g*s;
				g = g*c-x*s;
				h=y*s;
				y *= c;
				for (jj=1;jj<=n;jj++) {
					x=v[jj][j];
					z=v[jj][i];
					v[jj][j]=x*c+z*s;
					v[jj][i]=z*c-x*s;
				}
				z=pythag1(f,h);
				w[j]=z; 
				if (z) {
					z=1.0f/z;
					c=f*z;
					s=h*z;
				}
				f=c*g+s*y;
				x=c*y-s*g;
				for (jj=1;jj<=m;jj++) {
					y=a[jj][j];
					z=a[jj][i];
					a[jj][j]=y*c+z*s;
					a[jj][i]=z*c-y*s;
				}
			}
			rv1[l]=0.0;
			rv1[k]=f;
			w[k]=x;
		}
	}
	free(rv1);
}

inline float det(float** mat){
	return mat[1][1]*mat[2][2] - mat[1][2]*mat[2][1];
}

void find_similarity_transform(ShapeInfo &from_info, ShapeInfo &to_info, float m[2][2])
{
	ShapePoint mean_from = {0, 0}, mean_to = {0, 0};
	float sigma_from = 0.0f, sigma_to = 0.0f;
	int i;	
	float** cov = (float**)calloc(3, sizeof(float*));
	float** u = (float**)calloc(3, sizeof(float*));
	float** v = (float**)calloc(3, sizeof(float*));	
	float** s = (float**)calloc(3, sizeof(float*));	
	float** r = (float**)calloc(3, sizeof(float*));	
	for (i = 0; i < 3; i++) {
		cov[i] = (float*)calloc(3, sizeof(float));
		v[i] = (float*)calloc(3, sizeof(float));
		u[i] = (float*)calloc(3, sizeof(float));
		s[i] = (float*)calloc(3, sizeof(float));
		r[i] = (float*)calloc(3, sizeof(float));
	}
	float w[3];
	for (i = 0; i < LANDMARK_COUNT; i++) {
		mean_from.point[0] += from_info.data_x[i];
		mean_from.point[1] += from_info.data_y[i];
		mean_to.point[0] += to_info.data_x[i];
		mean_to.point[1] += to_info.data_y[i];
	}
	mean_from.point[0] /= LANDMARK_COUNT;
	mean_from.point[1] /= LANDMARK_COUNT;
	mean_to.point[0] /= LANDMARK_COUNT;
	mean_to.point[1] /= LANDMARK_COUNT;

	for (i = 0; i < LANDMARK_COUNT; i++) {
		ShapePoint fm = point_diff(from_info.data[i], mean_from);
		ShapePoint tm = point_diff(to_info.data[i], mean_to);
		sigma_from += length_squared(fm);
		sigma_to += length_squared(tm);
		cov[1][1] += fm.point[0]*tm.point[0];
		cov[1][2] += fm.point[1]*tm.point[0];
		cov[2][1] += fm.point[0]*tm.point[1];
		cov[2][2] += fm.point[1]*tm.point[1];
	}
	sigma_from /= LANDMARK_COUNT;
	sigma_to /= LANDMARK_COUNT;
	cov[1][1] /= LANDMARK_COUNT;
	cov[1][2] /= LANDMARK_COUNT;
	cov[2][1] /= LANDMARK_COUNT;
	cov[2][2] /= LANDMARK_COUNT;
	u[1][1] = cov[1][1];
	u[1][2] = cov[1][2];
	u[2][1] = cov[2][1];
	u[2][2] = cov[2][2];
	svdcmp(u, 2, 2, w, v);
	s[1][1] = 1; s[2][2] = 1;
	if(det(cov) < 0 || det(cov) == 0 && det(u) * det(v) < 0){
		if(w[2] < w[1]) s[2][2] = -1;
		else s[1][1] = -1;
	}
	r[1][1] = u[1][1]*s[1][1]*v[1][1] + u[1][2]*s[2][2]*v[1][2];
	r[1][2] = u[1][1]*s[1][1]*v[2][1] + u[1][2]*s[2][2]*v[2][2];
	r[2][1] = u[2][1]*s[1][1]*v[1][1] + u[2][2]*s[2][2]*v[1][2];
	r[2][2] = u[2][1]*s[1][1]*v[2][1] + u[2][2]*s[2][2]*v[2][2];
	float c = 1; 
	if (sigma_from != 0){
		double trace_val = w[1]*s[1][1] + w[2]*s[2][2];
		c = (float)(1.0/sigma_from * trace_val);
	}
	float t[2];
	t[0] = mean_to.point[0] - c * (r[1][1] *mean_from.point[0] + r[1][2] * mean_from.point[1]);
	t[1] = mean_to.point[1] - c * (r[2][1] *mean_from.point[0] + r[2][2] * mean_from.point[1]);
	m[0][0] = c * r[1][1];
	m[0][1] = c * r[1][2];
	m[1][0] = c * r[2][1];
	m[1][1] = c * r[2][2];
	for (i = 0; i < 3; i++) {
		SafeMemFree(cov[i]);
		SafeMemFree(v[i]);
		SafeMemFree(u[i]);
		SafeMemFree(s[i]);
		SafeMemFree(r[i]);
	}
	SafeMemFree(cov);
	SafeMemFree(v);	
	SafeMemFree(u);
	SafeMemFree(s);
	SafeMemFree(r);
}

inline void find_affine_transform(CF_Rect rt, float m[2][2], float b[2])
{
	float P[3][3], Q[2][3];
	P[0][0] = -1; P[0][1] = 0;  P[0][2] = 1; 
	P[1][0] = 1;  P[1][1] = -1; P[1][2] = 0;
	P[2][0] = 0;  P[2][1] = 1;  P[2][2] = 0;
	Q[0][0] = (float)rt.x; Q[0][1] = (float)(rt.x + rt.width);	Q[0][2] = (float)(rt.x + rt.width);
	Q[1][0] = (float)rt.y;	Q[1][1] = (float)rt.y;				Q[1][2] = (float)(rt.y + rt.height);

	m[0][0] = Q[0][0]*P[0][0] + Q[0][1]*P[1][0] + Q[0][2]*P[2][0];
	m[0][1] = Q[0][0]*P[0][1] + Q[0][1]*P[1][1] + Q[0][2]*P[2][1];
	m[1][0] = Q[1][0]*P[0][0] + Q[1][1]*P[1][0] + Q[1][2]*P[2][0];
	m[1][1] = Q[1][0]*P[0][1] + Q[1][1]*P[1][1] + Q[1][2]*P[2][1];

	b[0] = Q[0][0]*P[0][2] + Q[0][1]*P[1][2] + Q[0][2]*P[2][2];
	b[1] = Q[1][0]*P[0][2] + Q[1][1]*P[1][2] + Q[1][2]*P[2][2];
}

void extractfeature(const unsigned char* inImage, int cx, int cy, float m[2][2], float n[2][2], float b[2], ShapeInfo &currentShape, float* feature, int idx)
{
	int i;	
#if OMP_USE
#pragma omp parallel for
#endif
	for (i = 0; i < SHAPE_FEATURE_NUM; i++) {
		ShapePoint sp1;
		int point_x, point_y;
		sp1.point[0] = m[0][0] * shape->delta[idx].delta_value[i].point[0] + m[0][1] * shape->delta[idx].delta_value[i].point[1] + currentShape.data_x[shape->anchor[idx].idx[i]];
		sp1.point[1] = m[1][0] * shape->delta[idx].delta_value[i].point[0] + m[1][1] * shape->delta[idx].delta_value[i].point[1] + currentShape.data_y[shape->anchor[idx].idx[i]];
		point_x = (int)(n[0][0] * sp1.point[0] + n[0][1] * sp1.point[1] + b[0] + 0.5f);
		point_y = (int)(n[1][0] * sp1.point[0] + n[1][1] * sp1.point[1] + b[1] + 0.5f);
		if(point_x < cx && point_x >= 0 && point_y >= 0 && point_y < cy){
			feature[i] = inImage[point_y * cx + point_x];
		}else{
			feature[i] = 0;
		}		
	}
}

void ShapeFeatureExtract(const unsigned char* inImage, int cx, int cy, CF_Rect rt, ShapeInfo &currentShape, SFaceItem* pFaceStru)
{
	int i, j, k, l;
	float* feature = (float*)calloc(SHAPE_FEATURE_NUM, sizeof(float));
	float m[2][2], n[2][2], b[2];
	find_affine_transform(rt, n, b);
	for (i = 0; i < ITERATE_NUM; i++) {
		find_similarity_transform(shape->initShape, currentShape, m);		
		extractfeature(inImage, cx, cy, m, n, b, currentShape, feature, i);
		for (j = 0; j < SHAPE_FEATURE_NUM; j++) {
			k = 0;
			while(k < 15) {
				if((feature[shape->forest[i].forestinfo[j].splite[k].idx1] - feature[shape->forest[i].forestinfo[j].splite[k].idx2]) > shape->forest[i].forestinfo[j].splite[k].thresh){
					k = 2*k + 1;
				}else{
					k = 2*k + 2;
				}
			}
			k = k - 15;
			for (l = 0; l < LANDMARK_COUNT; l++) {
				currentShape.data_x[l] += shape->forest[i].forestinfo[j].leaf_value[k].data_x[l];
				currentShape.data_y[l] += shape->forest[i].forestinfo[j].leaf_value[k].data_y[l];
				currentShape.data[l].point[0] = currentShape.data_x[l];
				currentShape.data[l].point[1] = currentShape.data_y[l];
			}
		}
	}
	for(i = 0; i < LANDMARK_COUNT; i++){
		pFaceStru->ptLMCode.ptCord[i].x = (int)(n[0][0] * currentShape.data[i].point[0] + n[0][1] * currentShape.data[i].point[1] + b[0] + 0.5f);
		pFaceStru->ptLMCode.ptCord[i].y = (int)(n[1][0] * currentShape.data[i].point[0] + n[1][1] * currentShape.data[i].point[1] + b[1] + 0.5f);
	}
	free(feature);
}

int DetectEye(CF_Point* lpLandmark, unsigned char *pbyEyeIndex, CF_Point& lpEyePoint)
{
	int nIndex = 4;
	int nX = 0, nY = 0, nActive= 0;
	for (int i = 0; i < nIndex; i++) {
		nX += lpLandmark[pbyEyeIndex[i]].x;
		nY += lpLandmark[pbyEyeIndex[i]].y;
	}
	lpEyePoint.x = (2*nX + nIndex)/ (2*nIndex);
	lpEyePoint.y = (2*nY +nIndex) / (2*nIndex);
	return 0;
}

int DetectRightEye(CF_Point* lpLandmark, CF_Point& lpRightEye)
{
	unsigned char lpEyeIndex[4] = {0x2B, 0x2C, 0x2E, 0x2F};
	return DetectEye(lpLandmark, lpEyeIndex, lpRightEye);
}

int DetectLeftEye(CF_Point* lpLandmark, CF_Point& lpLeftEye)
{
	unsigned char lpEyeIndex[4] = {0x25, 0x26, 0x28, 0x29};
	return DetectEye(lpLandmark, lpEyeIndex, lpLeftEye);
}

void ShapeDetect(const unsigned char* inImage, int cx, int cy, CF_Rect pFaceRect, SFaceItem* pFaceStru)
{
	ShapeInfo currentShape = shape->initShape;
	ShapeFeatureExtract(inImage, cx, cy, pFaceRect, currentShape, pFaceStru);
	DetectLeftEye(pFaceStru->ptLMCode.ptCord, pFaceStru->ptLEye);
	DetectRightEye(pFaceStru->ptLMCode.ptCord, pFaceStru->ptREye);
}
#define NORMAL_LANDMARK_HEIGHT /*150*/200
#define NORMAL_LANDMARK_WIDTH  /*150*/200
int RotateImageLandmark(const unsigned char *pImageGray, int width, int height, double ratio, int nMode, int center_x, int center_y, double alpha, unsigned char *pOutImg)
{
	if (pImageGray == NULL || pOutImg == NULL || width <= 0 || height <= 0) return -1;
	int x, y, px, py, adr, nWindow, val;
	double vx, vy, cos_val, sin_val;
	cos_val = cos(alpha);
	sin_val = sin(alpha);
	adr = 0;
	nWindow = /*nMode == 1 ? -60 : */-NORMAL_LANDMARK_WIDTH / 2;
	for (y = nWindow; y < nWindow + NORMAL_LANDMARK_HEIGHT; y++) {
		for (x = nWindow; x < nWindow + NORMAL_LANDMARK_WIDTH; x++) {
			vx = center_x + (x * cos_val + y * sin_val) / ratio;
			vy = center_y + (-x * sin_val + y * cos_val) / ratio;
			px = (int)(vx);
			py = (int)(vy);
			if (py < 0 || py >= height) { pOutImg[adr++] = 0; continue; }
			if (px < 0 || px >= width) { pOutImg[adr++] = 0; continue; }
			val = pImageGray[py * width + px];
			if (px >= width - 1 || py >= height - 1) { pOutImg[adr++] = val; continue; }
			val = (int)(val * (1 - (vx - px)) * (1 - (vy - py)) + pImageGray[py * width + px + 1] * (vx - px) * (1 - (vy - py)) +
				pImageGray[(py + 1) * width + px] * (1 - (vx - px)) * (vy - py) + pImageGray[(py + 1) * width + px + 1] * (vx - px) * (vy - py) + 0.5);
			pOutImg[adr++] = (unsigned char)(val);
		}
	}
	return 0;
}

int LM68_dlib_do_extract(const unsigned char* image, int width, int height, int size_of_pixel, const Detection face_location, FaceLocation& landmarks)
{
	if (image == NULL)
		return -1;
	CF_Rect rcFace;
	CF_Point cen;
	int BetEye = 0;
	BetEye = (int)((face_location.xmax - face_location.xmin) / 2.3 + 0.5);
	cen.x = (face_location.keyPoints[1].x + face_location.keyPoints[0].x) / 2;
	cen.y = (face_location.keyPoints[1].y + face_location.keyPoints[0].y) / 2;
	rcFace.x = (int)(cen.x - 1.2 * BetEye + 0.5);
	rcFace.y = (int)(cen.y - 0.8 * BetEye + 0.5);
	rcFace.width = (int)(BetEye * 2.4 + 0.5);
	rcFace.height = (int)(BetEye * 2.6 + 0.5);
	SFaceItem faceItem;
	ShapeDetect(image, width, height, rcFace, &faceItem);

	for (int i = 0; i < LANDMARK_COUNT; i++) {
		landmarks.landmarks[i].x = faceItem.ptLMCode.ptCord[i].x;
		landmarks.landmarks[i].y = faceItem.ptLMCode.ptCord[i].y;
	}
	landmarks.landmarks[LM68_LEFT_EYE_INDEX].x = faceItem.ptLEye.x;
	landmarks.landmarks[LM68_LEFT_EYE_INDEX].y = faceItem.ptLEye.y;
	landmarks.landmarks[LM68_RIGHT_EYE_INDEX].x = faceItem.ptREye.x;
	landmarks.landmarks[LM68_RIGHT_EYE_INDEX].y = faceItem.ptREye.y;
	
	landmarks.faceRt = cv::Rect(rcFace.x, rcFace.y, rcFace.width, rcFace.height);
	return 0;
}
