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
    CARGO_ICON = 1,
    //ENDURIUM = 3
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

    vector<Template> templates = {
        //{"C:\\Users\\climd\\source\\repos\\CppDarkOrbitBot\\pngs\\palladium1.png", TM_CCOEFF_NORMED, 0.75, true, Mat(), Mat()},
        {"C:\\Users\\climd\\source\\repos\\CppDarkOrbitBot\\pngs\\prometium1.png", TM_CCOEFF_NORMED, 0.75, true, Mat(), Mat()},
        {"C:\\Users\\climd\\source\\repos\\CppDarkOrbitBot\\pngs\\cargo_icon.png", TM_SQDIFF_NORMED, 0.1, false, Mat(), Mat()},
        //{"C:\\Users\\climd\\source\\repos\\CppDarkOrbitBot\\pngs\\endurium2.png", TM_CCOEFF_NORMED, 0.7, true, Mat(), Mat()}
    };

    int screenshotGridColumns = 4;
    int screenshotGridRows = 3;
    int screenshotOffset = 50;

    int threadCount = 15;

    float totalTime = 0.0f;
    float totalFrames = 0.0f;
    float averageMillis = 0.0f;
    float averageFPS = 0.0f;

    loadImages(templates);
    extractPngNames(templates);

    // templates - matches
    vector<vector<Rect>> matchedRectangles(templates.size());
    vector<vector<double>> matchedConfidences(templates.size());

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

    vector<string> timeProfilerSteps = {
        "Clearing previous frames",
        "Taking screenshot",
        "Dividing screenshot",
        "Template matching",
        "Closest resource loop",
        "Closest match drawing",
        "Drawing matches"
    };
    vector<long long> timeProfilerTotalTimes(timeProfilerSteps.size(), 0);
    vector<float> timeProfilerAverageTimes(timeProfilerSteps.size(), 0);

    while (true)
    {
        // keep track of when the loop starts
        long long frameStart = getCurrentMillis();
        long long timeProfilerAux;
        int profilingStep = 0;


        // clearing previous frame's matches
        timeProfilerAux = getCurrentMicros();
        for (vector<Rect> &v : matchedRectangles) v.clear();
        for (vector<double> &v : matchedConfidences) v.clear();
        timeProfilerTotalTimes[profilingStep] += computeTimePassed(timeProfilerAux, getCurrentMicros());
        profilingStep++;


        // capturing screenshot
        timeProfilerAux = getCurrentMicros();
        Mat screenshot = screenshotManager.capture();
        timeProfilerTotalTimes[profilingStep] += computeTimePassed(timeProfilerAux, getCurrentMicros());
        profilingStep++;


        // dividing screenshot
        timeProfilerAux = getCurrentMicros();
        vector<vector<Mat>> dividedScreenshot = divideImage(screenshot, screenshotGridColumns, screenshotGridRows, screenshotOffset);
        timeProfilerTotalTimes[profilingStep] += computeTimePassed(timeProfilerAux, getCurrentMicros());
        profilingStep++;


        // template matching
        timeProfilerAux = getCurrentMicros();
        matchTemplatesParallel(screenshot, screenshotOffset, dividedScreenshot, templates, threadPool, matchedConfidences, matchedRectangles);
        timeProfilerTotalTimes[profilingStep] += computeTimePassed(timeProfilerAux, getCurrentMicros());
        profilingStep++;


        // figuring out which match is closest
        Rect closestResourceRect;
        double closestResourceConfidence = -1;
        double closestResourceDistance = screenshot.cols;
        int closestResourceIndex = -1;

        timeProfilerAux = getCurrentMicros();
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
        timeProfilerTotalTimes[profilingStep] += computeTimePassed(timeProfilerAux, getCurrentMicros());
        profilingStep++;


        // closest match drawing
        timeProfilerAux = getCurrentMicros();
        if (closestResourceIndex != -1)
        {
            // removing the closest match from the vector so that it wont get drawn like the other matches
            matchedRectangles[PROMETIUM].erase(matchedRectangles[PROMETIUM].begin() + closestResourceIndex);
            matchedConfidences[PROMETIUM].erase(matchedConfidences[PROMETIUM].begin() + closestResourceIndex);

            // drawing closest resource separately to use a different color
            drawSingleTargetOnScreenshot(screenshot, closestResourceRect, closestResourceConfidence, templates[PROMETIUM].name, Scalar(255, 255, 255));

            // draw a line between the ship and the closest resource found
            line(screenshot, 
                Point(closestResourceRect.x + closestResourceRect.width / 2, closestResourceRect.y + closestResourceRect.height / 2), 
                Point(screenshot.cols / 2, screenshot.rows / 2), 
                Scalar(255, 255, 255), 1, LINE_4, 0);
        }
        timeProfilerTotalTimes[profilingStep] += computeTimePassed(timeProfilerAux, getCurrentMicros());
        profilingStep++;


        // drawing matches
        timeProfilerAux = getCurrentMicros();
        for (int i = 0; i < templates.size(); i++) 
            drawMatchedTargets(matchedRectangles[i], matchedConfidences[i], screenshot, templates[i].name);
        timeProfilerTotalTimes[profilingStep] += computeTimePassed(timeProfilerAux, getCurrentMicros());
        profilingStep++;


        // TODO: add a property on each of the pngs to tell the matching function used for each of them as well as wether it should use the
        // divided screenshot or not, additionally a thread counter for each frame that will be displayed as debug info on the bot screen


        // keeping track of when the loop ends, to calculate how long the loop took and fps
        long long frameDuration = computeTimePassed(frameStart, getCurrentMillis());
        string frameRate;
        string averageFrameRate;
        computeFrameRate(frameDuration, totalTime, totalFrames, frameRate, averageFrameRate);
        

        // drawing debug information
        cv::putText(screenshot, frameRate, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        cv::putText(screenshot, averageFrameRate, cv::Point(10, 70), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

        for (int i = 0; i < timeProfilerSteps.size(); i++)
        {
            stringstream str;
            str << fixed << setprecision(4);
            timeProfilerAverageTimes[i] = timeProfilerTotalTimes[i] / totalFrames / 1000;

            str << timeProfilerAverageTimes[i];

            // spacing based on how many digits the avg time has
            if (int(timeProfilerAverageTimes[i]) % 10 > 0) str << " ";
            else str << "  ";

            str << "ms - " << timeProfilerSteps[i];

            cv::putText(screenshot, str.str(), cv::Point(10, 800 + i * 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }

        // showing the frame at the end
        cv::imshow("CppDarkOrbitBotView", screenshot);
        int key = cv::waitKey(10);
    }

    cv::destroyAllWindows();

    return 0;
}