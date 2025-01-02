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
    COLLECTING = 2,
    TRAVELING = 3
};

enum TemplateIdentifier {
    NO_TEMPLATE = -1,
    PALLADIUM = 0,
    CARGO_ICON = 1,
    PROMETIUM = 2,
    ENDURIUM = 3,
    MINIMAP_ICON = 4,
    MINIMAP_BUTTONS = 5
};

struct Template {
    string name;
    TemplateIdentifier identifier;
    cv::TemplateMatchModes matchingMode;
    double confidenceThreshold;
    bool useDividedScreenshot;
    bool multipleMatches;
    Mat grayscale;
    Mat alpha;
};

struct TemplateMatch
{
    Rect rect;
    double confidence;
    TemplateIdentifier identifier;

    bool operator()(const TemplateMatch &a, const TemplateMatch &b) const 
    {
        return (a.rect.x == b.rect.x) ? a.rect.y < b.rect.y : a.rect.x < b.rect.x;
    }

    TemplateMatch(const Rect& rect, double confidence, const TemplateIdentifier& identifier)
        : rect(rect), confidence(confidence), identifier(identifier) {}
};

void setConsoleStyle(int style);
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
double distanceBetweenPoints(Point &a, Point &b);
double pointToScreenshotCenterDistance(int &x, int &y, int screenWidth, int screenHeight);

#endif