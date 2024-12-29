#ifndef BOT_UTILS
#define BOT_UTILS

#include <opencv2/opencv.hpp>
#include <Windows.h>
#include <string>
#include <vector>

using namespace std;
using namespace cv;

enum BotStatus {
    SCANNING = 0,
    MOVING = 1,
    COLLECTING = 2
};

struct Template {
    string name;
    cv::TemplateMatchModes matchingMode;
    double confidenceThreshold;
    bool useDividedScreenshot;
    bool multipleMatches;
    Mat grayscale;
    Mat alpha;
};

enum TemplateIdentifier {
    //PALLADIUM = 0,
    PROMETIUM = 0,
    CARGO_ICON = 1,
    //ENDURIUM = 3
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
void printWithTimestamp(string message, int style);
void printTimeProfiling(long long startMicros, string message);
long long computeTimePassed(long long start, long long finish);
void computeFrameRate(int loopDuration, float &totalTime, float &totalFrames, string &currentFPSString, string &averageFPSString);
void clickAt(int x, int y);
string botStatusEnumToString(BotStatus status);

#endif