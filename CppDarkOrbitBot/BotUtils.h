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

#endif