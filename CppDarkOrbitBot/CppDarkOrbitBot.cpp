#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <filesystem>
#include <windows.h>
#include "CppDarkOrbitBot.h"
#include <regex>
#include <string>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <numeric>

#include "Constants.h"
#include "BotUtils.h"
#include "BotCV.h"
#include "ThreadPool.h"

using namespace std;
using namespace cv;
using namespace chrono;

HWND darkOrbitHandle;

enum TemplateIdentifier {
    //PALLADIUM = 0,
    PROMETIUM = 0,
    CARGO_ICON = 1
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
};

double distanceBetweenPoints(Point &a, Point &b) 
{
    return sqrt(pow((b.x - a.x), 2) + pow((b.y - a.y), 2));
}

double pointToOriginDistance(Point &a) 
{
    // only passing 1 point will be considered will return the distance between the point and the origin (0,0)
    return sqrt(pow((a.x), 2) + pow((a.y), 2));
}

double pointToScreenshotCenterDistance(int &x, int &y, int screenWidth, int screenHeight)
{
    Point screenshotCenter = Point(screenWidth / 2, screenHeight / 2);

    return sqrt(pow((screenshotCenter.x - x), 2) + pow((screenshotCenter.y - y), 2));
}

int main() 
{
    long long initialisationStart = getCurrentMillis();

    initializeConsoleHandle();

    darkOrbitHandle = FindWindow(NULL, L"DarkOrbit");

    if (darkOrbitHandle)
    {
        setConsoleStyle(GREEN_TEXT_BLACK_BACKGROUND);
        cout << "DarkOrbit handle found!" << endl;
    }
    else
    {
        setConsoleStyle(RED_TEXT_BLACK_BACKGROUND);
        cout << "DarkOrbit handle not found..." << endl;
        return -1;
    }
    setConsoleStyle(DEFAULT);

    vector<string> pngPaths = {
        //"C:\\Users\\climd\\source\\repos\\CppDarkOrbitBot\\pngs\\palladium1.png",
        "C:\\Users\\climd\\source\\repos\\CppDarkOrbitBot\\pngs\\prometium1.png",
        "C:\\Users\\climd\\source\\repos\\CppDarkOrbitBot\\pngs\\cargo_icon.png",
    };
    vector<Mat> templateGrayscales;
    vector<Mat> templateAlphas;
    vector<string> templateNames;

    int screenshotGridColumns = 5;
    int screenshotGridRows = 3;
    int screenshotOffset = 50;

    int threadCount = 15;

    double confidenceThreshold = 0.75;

    float totalTime = 0.0f;
    float totalFrames = 0.0f;
    int frameCount = 0;
    float averageMillis = 0.0f;
    float averageFPS = 0.0f;

    loadImages(pngPaths, templateGrayscales, templateAlphas);
    extractPngNames(pngPaths, templateNames);

    // templates - matches
    vector<vector<Rect>> matchedRectangles(templateGrayscales.size());
    vector<vector<double>> matchedConfidences(templateGrayscales.size());

    setConsoleStyle(YELLOW_TEXT_BLACK_BACKGROUND);
    cout << "Screenshot grid size: " << screenshotGridColumns << " columns x " << screenshotGridRows << " rows" << endl;
    cout << "Screenshot offset: " << screenshotOffset << endl;

    ThreadPool threadPool(threadCount);
    cout << "Started " << threadCount << " worker threads" << endl;
    setConsoleStyle(DEFAULT);

    long long initialisationDuration = computeTimePassed(initialisationStart, getCurrentMillis());
    setConsoleStyle(GREEN_TEXT_BLACK_BACKGROUND);
    cout << "Bot initialisation took " << initialisationDuration << "ms" << endl;
    setConsoleStyle(DEFAULT);

    ScreenshotManager screenshotManager(darkOrbitHandle);

    bool timeProfiling = false;

    while (true)
    {
        // keep track of when the loop starts
        long long frameStart = getCurrentMillis();
        long long timerStart;

        timerStart = getCurrentMicros();
        // clearing previous frame's matches
        for (vector<Rect> &v : matchedRectangles) v.clear();
        for (vector<double> &v : matchedConfidences) v.clear();
        if (timeProfiling) printTimeProfiling(timerStart, "Clearing previous frame matches");

        timerStart = getCurrentMicros();
        Mat screenshot = screenshotManager.capture();
        if (timeProfiling) printTimeProfiling(timerStart, "Taking screenshot");

        timerStart = getCurrentMicros();
        vector<vector<Mat>> dividedScreenshot = divideImage(screenshot, screenshotGridColumns, screenshotGridRows, screenshotOffset);
        if (timeProfiling) printTimeProfiling(timerStart, "Dividing screenshots");

        timerStart = getCurrentMicros();
        matchTemplatesParallel(screenshot, screenshotOffset, dividedScreenshot, templateGrayscales, templateAlphas, templateNames, confidenceThreshold, threadPool,
            matchedConfidences, matchedRectangles);
        if (timeProfiling) printTimeProfiling(timerStart, "Template matching");

        timerStart = getCurrentMicros();
        Rect closestResourceRect;
        double closestResourceConfidence = -1;
        double closestResourceDistance = screenshot.cols;
        int closestResourceIndex = -1;
        if (timeProfiling) printTimeProfiling(timerStart, "Declaring closest resource vars");


        timerStart = getCurrentMicros();
        // find the closest prometium match
        for (int i = 0; i < matchedRectangles[PROMETIUM].size(); i++)
        {
            // this returns distance between the point and the center of the screenshot where the ship is
            double distance = pointToScreenshotCenterDistance(matchedRectangles[PROMETIUM][i].x, matchedRectangles[PROMETIUM][i].y, screenshot.cols, screenshot.rows);
            if (distance < closestResourceDistance)
            {
                closestResourceDistance = distance;
                closestResourceRect = matchedRectangles[PROMETIUM][i];
                closestResourceConfidence = matchedConfidences[PROMETIUM][i];
                closestResourceIndex = i;
            }
        }
        if (timeProfiling) printTimeProfiling(timerStart, "Closest resource loop");
        timerStart = getCurrentMicros();
        if (closestResourceIndex != -1)
        {
            // removing the closest match from the vector so that it wont get drawn like the other matches
            matchedRectangles[PROMETIUM].erase(matchedRectangles[PROMETIUM].begin() + closestResourceIndex);
            matchedConfidences[PROMETIUM].erase(matchedConfidences[PROMETIUM].begin() + closestResourceIndex);

            // drawing closest resource separately to use a different color
            drawSingleTargetOnScreenshot(screenshot, closestResourceRect, closestResourceConfidence, templateNames[PROMETIUM], Scalar(255, 255, 255));

            // draw a line between the ship and the closest resource found
            line(screenshot, 
                Point(closestResourceRect.x + closestResourceRect.width / 2, closestResourceRect.y + closestResourceRect.height / 2), 
                Point(screenshot.cols / 2, screenshot.rows / 2), 
                Scalar(255, 255, 255), 1, LINE_4, 0);
        }
        if (timeProfiling) printTimeProfiling(timerStart, "Removing closest match from list and drawing it separately");

        timerStart = getCurrentMicros();
        // drawing matches
        for (int i = 0; i < templateNames.size(); i++) 
            drawMatchedTargets(matchedRectangles[i], matchedConfidences[i], screenshot, templateNames[i]);
        if (timeProfiling) printTimeProfiling(timerStart, "Drawing matches onto screen");

        // TODO: add a property on each of the pngs to tell the matching function used for each of them as well as wether it should use the
        // divided screenshot or not, additionally a thread counter for each frame that will be displayed as debug info on the bot screen


        // keep track of when the loop ends, to calculate how long the loop took and fps
        long long frameDuration = computeTimePassed(frameStart, getCurrentMillis());
        string frameRate;
        string averageFrameRate;
        computeFrameRate(frameDuration, totalTime, totalFrames, frameRate, averageFrameRate);
        
        cv::putText(screenshot, frameRate, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        cv::putText(screenshot, averageFrameRate, cv::Point(10, 70), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);


        // show the frame at the end
        cv::imshow("CppDarkOrbitBotView", screenshot);
        int key = cv::waitKey(10);

        if (timeProfiling) break;
    }

    cv::destroyAllWindows();

    return 0;
}