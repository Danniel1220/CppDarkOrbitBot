#ifndef BOT_UTILS
#define BOT_UTILS

#include <opencv2/opencv.hpp>
#include <Windows.h>
#include <string>
#include <vector>

using namespace std;
using namespace cv;

struct Template {
    string name;
    cv::TemplateMatchModes matchingMode;
    bool useDividedScreenshot;
    Mat grayscale;
    Mat alpha;
};

void setConsoleStyle(int style);
vector<Mat> loadImages(vector<string> paths, vector<Mat>& grayscales, vector<Mat>& alphas);
void loadImages(vector<Template> &templates);
void testConsoleColors();
void showImages(vector<Mat>& targetGrayImages, string name);
void extractPngNames(vector<string> pngPaths, vector<string>& targetNames);
void extractPngNames(vector<Template> &templates);
long long getCurrentMillis();
long long getCurrentMicros();
string millisToTimestamp(long long millis);
void printWithTimestamp(string message);
void printTimeProfiling(long long startMicros, string message);
long long computeTimePassed(long long start, long long finish);
void computeFrameRate(int loopDuration, float &totalTime, float &totalFrames, string &currentFPSString, string &averageFPSString);

#endif