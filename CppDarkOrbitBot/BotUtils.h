#ifndef BOT_UTILS
#define BOT_UTILS

#include <opencv2/opencv.hpp>
#include <Windows.h>
#include <string>
#include <vector>

using namespace std;
using namespace cv;

void setConsoleStyle(int style);
vector<Mat> loadImages(vector<string> paths, vector<Mat>& grayscales, vector<Mat>& alphas);
void testConsoleColors();
void showImages(vector<Mat>& targetGrayImages, string name);
void extractPngNames(vector<string> pngPaths, vector<string>& targetNames);
Mat screenshotWindow(HWND hwnd);
double calculateIoU(const cv::Rect& a, const cv::Rect& b);
void applyNMS(const vector<Rect>& boxes, const vector<double>& scores, double nmsThreshold, vector<int>& indices);
long long getCurrentMillis();
int computeMillisPassed(long long start, long long finish);

#endif