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
long long getCurrentMillis();
long long getCurrentMicros();
long long computeTimePassed(long long start, long long finish);
void computeFrameRate(int loopDuration, float &totalTime, float &totalFrames, string &currentFPSString, string &averageFPSString);

#endif