#ifndef BOT_CV
#define BOT_CV

#include "ThreadPool.h"
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <Windows.h>

using namespace std;
using namespace cv;

class ScreenshotManager {
public:
    explicit ScreenshotManager(HWND hwnd);

    ~ScreenshotManager();

    cv::Mat capture();

private:
    HWND hwnd_;

    int width_, height_;

    HDC hwindowDC_;
    HDC hwindowCompatibleDC_;
    HBITMAP hbwindow_;

    BITMAPINFOHEADER bi_;

    void initialize();

    void cleanup();
};

void drawMultipleTargets(Mat &screenshot, vector<TemplateMatch> &matches, string templateName);
void drawSingleTarget(Mat &screenshot, TemplateMatch target, string name, Scalar color);
void drawSingleTarget(Mat &screenshot, Rect target, string name, Scalar color);
void matchSingleTemplate(Mat screenshot, Mat templateGrayscale, Mat templateAlpha, string templateName, TemplateMatchModes matchMode, double confidenceThreshold,
    vector<double> &matchScores, vector<Rect> &matchRectangles, vector<int> &deduplicatedMatchIndexes);
void matchTemplatesParallel(Mat &screenshot, int screenshotOffset, vector<vector<Mat>> &screenshotGrid, vector<Template> &templates,
    ThreadPool &threadPool, vector<vector<TemplateMatch>> &resultMatches);
vector<vector<Mat>> divideImage(Mat image, int gridWidth, int gridHeight, int overlapAmount);
Mat screenshotWindow(HWND hwnd);
double calculateIoU(const cv::Rect& a, const cv::Rect& b);
void applyNMS(const vector<Rect>& boxes, const vector<double>& scores, double nmsThreshold, vector<int>& indices);
bool matchTemplateWithHighestScore(Mat screenshot, Mat templateGrayscale, Mat templateAlpha, string templateName, TemplateMatchModes matchMode, double confidenceThreshold,
    double &matchScore, Rect &matchRectangle);

#endif