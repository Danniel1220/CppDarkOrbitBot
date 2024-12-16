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
    vector<vector<Point>> matchedLocations(templateGrayscales.size());
    vector<vector<double>> matchedConfidences(templateGrayscales.size());
    vector<vector<Rect>> matchedRectangles(templateGrayscales.size());

    setConsoleStyle(YELLOW_TEXT_BLACK_BACKGROUND);
    cout << "Screenshot grid size: " << screenshotGridColumns << " columns x " << screenshotGridRows << " rows" << endl;
    cout << "Screenshot offset: " << screenshotOffset << endl;

    ThreadPool threadPool(threadCount);
    
    cout << "Started " << threadCount << " worker threads" << endl;

    setConsoleStyle(DEFAULT);

    long long initialisationDuration = computeMillisPassed(initialisationStart, getCurrentMillis());
    setConsoleStyle(GREEN_TEXT_BLACK_BACKGROUND);
    cout << "Bot initialisation took " << initialisationDuration << "ms" << endl;
    setConsoleStyle(DEFAULT);

    while (true)
    {
        // keep track of when the loop starts
        long long start = getCurrentMillis();

        // clearing previous frame's matched
        for (vector<Point> &v : matchedLocations) v.clear();
        for (vector<double> &v : matchedConfidences) v.clear();
        for (vector<Rect> &v : matchedRectangles) v.clear();

        Mat screenshot = screenshotWindow(darkOrbitHandle);


        vector<vector<Mat>> dividedScreenshot = divideImage(screenshot, screenshotGridColumns, screenshotGridRows, screenshotOffset);

        matchTemplatesParallel(screenshot, screenshotOffset, dividedScreenshot, templateGrayscales, templateAlphas, templateNames, confidenceThreshold, threadPool,
            matchedLocations, matchedConfidences, matchedRectangles);

        // drawing all the matches
        for (int i = 0; i < templateNames.size(); i++) 
            drawMatchedTargets(matchedRectangles[i], matchedConfidences[i], screenshot, templateNames[i]);





        // keep track of when the loop ends, to calculate how long the loop took and fps
        long long duration = computeMillisPassed(start, getCurrentMillis());
        string frameRate;
        string averageFrameRate;
        computeFrameRate(duration, totalTime, totalFrames, frameRate, averageFrameRate);


        cv::putText(screenshot, frameRate, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        cv::putText(screenshot, averageFrameRate, cv::Point(10, 70), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

        // show the frame at the end
        cv::imshow("CppDarkOrbitBotView", screenshot);
        int key = cv::waitKey(10);
    }

    cv::destroyAllWindows();

    return 0;
}