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

void drawMatchedTargets(vector<Rect> &rectangles, vector<double> &confidences, Mat &screenshot, string templateName);
void drawSingleTargetOnScreenshot(Mat &screenshot, Rect rectangle, double confidence, string name, Scalar color);
void matchSingleTemplate(Mat screenshot, Mat templateGrayscale, Mat templateAlpha, string templateName, TemplateMatchModes matchMode, double confidenceThreshold,
    vector<double> &matchScores, vector<Rect> &matchRectangles, vector<int> &deduplicatedMatchIndexes);
void matchTemplatesParallel(Mat &screenshot, int screenshotOffset, vector<vector<Mat>> &screenshotGrid, vector<Template> &templates, double confidenceThreshold, 
    ThreadPool &threadPool, vector<vector<double>> &resultMatchedConfidences, vector<vector<Rect>> &resultMatchedRectangles);
vector<vector<Mat>> divideImage(Mat image, int gridWidth, int gridHeight, int overlapAmount);
Mat screenshotWindow(HWND hwnd);
double calculateIoU(const cv::Rect& a, const cv::Rect& b);
void applyNMS(const vector<Rect>& boxes, const vector<double>& scores, double nmsThreshold, vector<int>& indices);

#endif