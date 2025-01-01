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
        {"C:\\Users\\climd\\source\\repos\\CppDarkOrbitBot\\pngs\\palladium1.png", TM_CCOEFF_NORMED, 0.75, true, true, Mat(), Mat()},
        {"C:\\Users\\climd\\source\\repos\\CppDarkOrbitBot\\pngs\\cargo_icon.png", TM_SQDIFF_NORMED, 0.1, false, false, Mat(), Mat()},
        {"C:\\Users\\climd\\source\\repos\\CppDarkOrbitBot\\pngs\\prometium1.png", TM_CCOEFF_NORMED, 0.75, true, true, Mat(), Mat()},
        {"C:\\Users\\climd\\source\\repos\\CppDarkOrbitBot\\pngs\\endurium2.png", TM_CCOEFF_NORMED, 0.7, true, true, Mat(), Mat()},
        {"C:\\Users\\climd\\source\\repos\\CppDarkOrbitBot\\pngs\\minimap_icon.png", TM_SQDIFF_NORMED, 0.1, false, false, Mat(), Mat()},
        {"C:\\Users\\climd\\source\\repos\\CppDarkOrbitBot\\pngs\\minimap_buttons.png", TM_SQDIFF_NORMED, 0.1, false, false, Mat(), Mat()}
    };

    int screenshotGridColumns = 4;
    int screenshotGridRows = 3;
    int screenshotOffset = 50;

    int threadCount = 15;

    float totalTime = 0.0f;
    float totalFrames = 0.0f;
    float averageMillis = 0.0f;
    float averageFPS = 0.0f;

    long long collectingTimer;
    long long movingTimer;

    float minimumResourceDistance = 100.0;

    BotStatus status = BotStatus::SCANNING;

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
        "Drawing matches",
        "Bot decision logic"
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
        timeProfilerAux = getCurrentMicros();
        Rect closestResourceRect;
        double closestResourceConfidence = -1;
        double closestResourceDistance = screenshot.cols;
        int closestResourceIndex = -1;
        for (int i = 0; i < matchedRectangles[PALLADIUM].size(); i++)
        {
            // this returns distance between the point and the center of the screenshot where the ship is
            double distance = pointToScreenshotCenterDistance(matchedRectangles[PALLADIUM][i].x, matchedRectangles[PALLADIUM][i].y, screenshot.cols, screenshot.rows);
            if (distance < closestResourceDistance && distance > minimumResourceDistance)
            {
                closestResourceDistance = distance;
                closestResourceRect = matchedRectangles[PALLADIUM][i];
                closestResourceConfidence = matchedConfidences[PALLADIUM][i];
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
            matchedRectangles[PALLADIUM].erase(matchedRectangles[PALLADIUM].begin() + closestResourceIndex);
            matchedConfidences[PALLADIUM].erase(matchedConfidences[PALLADIUM].begin() + closestResourceIndex);

            // drawing closest resource separately to use a different color
            drawSingleTargetOnScreenshot(screenshot, closestResourceRect, closestResourceConfidence, templates[PALLADIUM].name, Scalar(255, 255, 255));

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


        // bot decision logic
        timeProfilerAux = getCurrentMicros();
        bool botON = true;
        if (botON)
        {
            // if the bot is scanning and a closest resource has been found
            if (status == SCANNING && closestResourceRect.width != 0)
            {
                clickAt(closestResourceRect.x + closestResourceRect.width / 2, closestResourceRect.y + closestResourceRect.height / 2);
                status = MOVING;
                movingTimer = getCurrentMillis();

                printWithTimestamp("BOT_STATUS: MOVING");

            }
            else if (status == MOVING) 
            {
                // if 4 seconds of moving havent passed yet
                if (computeTimePassed(movingTimer, getCurrentMillis()) > 4000)
                {
                    status = SCANNING;
                    printWithTimestamp("Fallback to scanning after 4s passed", RED_TEXT_BLACK_BACKGROUND);
                }
                // if 4 seconds have passed we are probably stuck so we go back to scanning
                else 
                {
                    Mat screenshotROI = screenshot(Rect(935, 615, 50, 50));
                    imshow("test", screenshotROI);
                    double score;
                    Rect rectangle;
                    bool matchFound = matchTemplateWithHighestScore(screenshotROI,
                        templates[PALLADIUM].grayscale, templates[PALLADIUM].alpha, templates[PALLADIUM].name, templates[PALLADIUM].matchingMode,
                        0.5, score, rectangle);

                    if (matchFound)
                    {
                        status = COLLECTING;
                        printWithTimestamp("Found collecting match with score: " + to_string(score));
                        printWithTimestamp("BOT_STATUS: COLLECTING");
                        collectingTimer = getCurrentMillis();
                    }
                }

                
            }
            else if (status == COLLECTING)
            {
                if (computeTimePassed(collectingTimer, getCurrentMillis()) > 1000)
                {
                    status = SCANNING;
                    printWithTimestamp("Collected resource");
                    printWithTimestamp("BOT_STATUS: SCANNING");
                }
            }
        }

        timeProfilerTotalTimes[profilingStep] += computeTimePassed(timeProfilerAux, getCurrentMicros());
        profilingStep++;



        // keeping track of when the loop ends, to calculate how long the loop took and fps
        long long frameDuration = computeTimePassed(frameStart, getCurrentMillis());
        string frameRate;
        string averageFrameRate;
        computeFrameRate(frameDuration, totalTime, totalFrames, frameRate, averageFrameRate);
        

        // drawing debug information
        cv::putText(screenshot, frameRate, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        cv::putText(screenshot, averageFrameRate, cv::Point(10, 70), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        cv::putText(screenshot, "BOT_STATUS: " + botStatusEnumToString(status), cv::Point(800, 1040), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 0), 2);

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