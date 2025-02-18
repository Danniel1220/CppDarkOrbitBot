﻿#include <iostream>
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
#include <random>

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
        printWithTimestamp("DarkOrbit handle found!", GREEN_TEXT_BLACK_BACKGROUND);
    }
    else
    {
        printWithTimestamp("DarkOrbit handle not found...", RED_TEXT_BLACK_BACKGROUND);
        return -1;
    }

    vector<Template> templates = {
        {"C:\\Users\\climd\\source\\repos\\CppDarkOrbitBot\\pngs\\palladium1.png", PALLADIUM, TM_CCOEFF_NORMED, 0.75, true, true, Mat(), Mat()},
        {"C:\\Users\\climd\\source\\repos\\CppDarkOrbitBot\\pngs\\cargo_icon.png", CARGO_ICON, TM_SQDIFF_NORMED, 0.1, false, false, Mat(), Mat()},
        {"C:\\Users\\climd\\source\\repos\\CppDarkOrbitBot\\pngs\\prometium1.png", PROMETIUM, TM_CCOEFF_NORMED, 0.75, true, true, Mat(), Mat()},
        {"C:\\Users\\climd\\source\\repos\\CppDarkOrbitBot\\pngs\\endurium2.png", ENDURIUM, TM_CCOEFF_NORMED, 0.7, true, true, Mat(), Mat()},
        {"C:\\Users\\climd\\source\\repos\\CppDarkOrbitBot\\pngs\\minimap_icon.png", MINIMAP_ICON, TM_SQDIFF_NORMED, 0.1, false, false, Mat(), Mat()},
        {"C:\\Users\\climd\\source\\repos\\CppDarkOrbitBot\\pngs\\minimap_buttons.png", MINIMAP_BUTTONS, TM_SQDIFF_NORMED, 0.1, false, false, Mat(), Mat()}
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
    long long travellingTimer;

    float minimumResourceDistance = 100.0;

    BotStatus status = BotStatus::SCANNING;

    loadImages(templates);
    extractPngNames(templates);

    vector<Template> resourceTemplates;
    resourceTemplates.emplace_back(templates[PALLADIUM]);

    // templates - matches
    vector<vector<TemplateMatch>> matchedTemplates(templates.size());

    printWithTimestamp("Screenshot grid size: " + to_string(screenshotGridColumns) + " columns x " + to_string(screenshotGridRows) + " rows", YELLOW_TEXT_BLACK_BACKGROUND);
    printWithTimestamp("Screenshot offset: " + to_string(screenshotOffset), YELLOW_TEXT_BLACK_BACKGROUND);

    ThreadPool threadPool(threadCount);
    printWithTimestamp("Started " + to_string(threadCount) + " worker threads", YELLOW_TEXT_BLACK_BACKGROUND);

    ScreenshotManager screenshotManager(darkOrbitHandle);

    // finding the location and size of the minimap

    // grabbing the templates for the minimap
    vector<Template> minimapTemplates;
    minimapTemplates.emplace_back(templates[MINIMAP_ICON]);
    minimapTemplates.emplace_back(templates[MINIMAP_BUTTONS]);
    // taking screenshot
    Mat screenshotForMinimap = screenshotManager.capture();
    vector<vector<Mat>> dividedScreenshotForMinimap = divideImage(screenshotForMinimap, screenshotGridColumns, screenshotGridRows, screenshotOffset);
    // performing template matching to find the minimap
    vector<vector<TemplateMatch>> minimapMatchedTemplates(minimapTemplates.size());
    matchTemplatesParallel(screenshotForMinimap, screenshotOffset, dividedScreenshotForMinimap, minimapTemplates, threadPool, minimapMatchedTemplates);
    if (minimapMatchedTemplates[0].size() == 0 || minimapMatchedTemplates[1].size() == 0)
    {
        printWithTimestamp("Could not find minimap...", RED_TEXT_BLACK_BACKGROUND);
        return -1;
    }
    int width = (minimapMatchedTemplates[1][0].rect.x + minimapMatchedTemplates[1][0].rect.width) - minimapMatchedTemplates[0][0].rect.x;
    int height = width / 1.408; // 1.408 is the ratio between width and height for the minimap
    // final minimap rect 
    Rect minimapRect = Rect(minimapMatchedTemplates[0][0].rect.x, minimapMatchedTemplates[0][0].rect.y, width, height);
    printWithTimestamp("Minimap found at [" + to_string(minimapRect.x) + ", " + to_string(minimapRect.y) 
        + "] with size " + to_string(minimapRect.width) + "x" + to_string(minimapRect.height),
        YELLOW_TEXT_BLACK_BACKGROUND);

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

    long long initialisationDuration = computeTimePassed(initialisationStart, getCurrentMillis());
    printWithTimestamp("Bot initialisation took " + to_string(initialisationDuration) + "ms", GREEN_TEXT_BLACK_BACKGROUND);
    printWithTimestamp("Starting bot main loop");

    bool botON = false;
    bool toggleKeyPressed = false;

    while (true)
    {
        // keep track of when the loop starts
        long long frameStart = getCurrentMillis();
        long long timeProfilerAux;
        int profilingStep = 0;


        // clearing previous frame's matches
        timeProfilerAux = getCurrentMicros();
        for (vector<TemplateMatch> &v : matchedTemplates) v.clear();
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


        // resource template matching
        timeProfilerAux = getCurrentMicros();
        matchTemplatesParallel(screenshot, screenshotOffset, dividedScreenshot, resourceTemplates, threadPool, matchedTemplates);
        timeProfilerTotalTimes[profilingStep] += computeTimePassed(timeProfilerAux, getCurrentMicros());
        profilingStep++;


        // figuring out which match is closest
        timeProfilerAux = getCurrentMicros();
        TemplateMatch closestResource = TemplateMatch(Rect(), -1, NO_TEMPLATE);
        double closestResourceDistance = screenshot.cols;
        int closestResourceIndex = -1;
        for (int i = 0; i < matchedTemplates[0].size(); i++)
        {
            // this returns distance between the point and the center of the screenshot where the ship is
            double distance = pointToScreenshotCenterDistance(matchedTemplates[0][i].rect.x, matchedTemplates[0][i].rect.y, screenshot.cols, screenshot.rows);
            if (distance < closestResourceDistance && distance > minimumResourceDistance)
            {
                closestResource.rect = matchedTemplates[0][i].rect;
                closestResource.confidence = matchedTemplates[0][i].confidence;
                closestResourceDistance = distance;
                closestResourceIndex = i;
            }
        }
        timeProfilerTotalTimes[profilingStep] += computeTimePassed(timeProfilerAux, getCurrentMicros());
        profilingStep++;

        if (closestResourceIndex == -1 && matchedTemplates[0].size() == 1)
        {
            closestResource.rect = matchedTemplates[0][0].rect;
            closestResource.confidence = matchedTemplates[0][0].confidence;
            closestResourceIndex = 0;
        }


        // closest match drawing
        timeProfilerAux = getCurrentMicros();
        if (closestResourceIndex != -1)
        {
            // removing the closest match from the vector so that it wont get drawn like the other matches
            matchedTemplates[PALLADIUM].erase(matchedTemplates[0].begin() + closestResourceIndex);

            // drawing closest resource separately to use a different color
            drawSingleTarget(screenshot, closestResource, templates[0].name, Scalar(255, 255, 255));

            // draw a line between the ship and the closest resource found
            line(screenshot, 
                Point(closestResource.rect.x + closestResource.rect.width / 2, closestResource.rect.y + closestResource.rect.height / 2), 
                Point(screenshot.cols / 2, screenshot.rows / 2), 
                Scalar(255, 255, 255), 1, LINE_4, 0);
        }
        timeProfilerTotalTimes[profilingStep] += computeTimePassed(timeProfilerAux, getCurrentMicros());
        profilingStep++;


        // drawing matches
        timeProfilerAux = getCurrentMicros();
        for (int i = 0; i < templates.size(); i++) 
            drawMultipleTargets(screenshot, matchedTemplates[i], templates[i].name);
        timeProfilerTotalTimes[profilingStep] += computeTimePassed(timeProfilerAux, getCurrentMicros());
        profilingStep++;

        // drawing minimap rect
        drawSingleTarget(screenshot, minimapRect, "Minimap", Scalar(0, 255, 0));


        // bot logic on-off toggle
        if (GetAsyncKeyState(0x70) & 0x8000) // 0x54 is the virtual key code for 'F1'
        {  
            if (!toggleKeyPressed) 
            {
                botON = !botON;
                toggleKeyPressed = true;
                string msg = "Bot turned ";
                if (botON) msg = msg + "ON";
                else msg = msg + "OFF";
                printWithTimestamp(msg);
            }
        } 
        else 
        {
            toggleKeyPressed = false;
        }

        // bot decision logic
        timeProfilerAux = getCurrentMicros();
        if (botON)
        {
            // if the bot is ready to collect and a closest resource has been found
            if ((status == SCANNING || status == TRAVELING) && closestResource.rect.width != 0)
            {
                clickAt(closestResource.rect.x + closestResource.rect.width / 2, closestResource.rect.y + closestResource.rect.height / 2);
                status = MOVING;
                movingTimer = getCurrentMillis();

                printWithTimestamp("BOT_STATUS: MOVING");

            }
            else if (status == MOVING) 
            {
                // if 4 seconds of moving havent passed yet
                if (computeTimePassed(movingTimer, getCurrentMillis()) > 2500)
                {
                    status = SCANNING;
                    printWithTimestamp("Fallback to scanning after 4s passed", YELLOW_TEXT_BLACK_BACKGROUND);
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
                if (computeTimePassed(collectingTimer, getCurrentMillis()) > 50)
                {
                    status = SCANNING;
                    printWithTimestamp("Collected resource");
                    printWithTimestamp("BOT_STATUS: SCANNING");
                }
            }
            // if we are scanning but no matches have been found
            else if (status == SCANNING && matchedTemplates[0].size() == 0)
            {
                printWithTimestamp("Cannot find any resources, changing location");
                status = TRAVELING;
                printWithTimestamp("BOT_STATUS: TRAVELLING");
                travellingTimer = getCurrentMillis();
                Point topLeft = Point(minimapRect.x + minimapRect.width / 3.13, minimapRect.y + minimapRect.height / 1.42);
                Point bottomRight = Point(minimapRect.x + minimapRect.width / 1.32, minimapRect.y + minimapRect.height / 1.07);

                // generating a random location inside the palladium field
                random_device rd;
                mt19937 gen(rd());
                uniform_int_distribution<int> rdX(topLeft.x, bottomRight.x);
                uniform_int_distribution<int> rdY(topLeft.y, bottomRight.y);
                //clickAt(rdX(gen), rdY(gen));
                clickAt(bottomRight.x -12, rdY(gen));
            }
            else if (status == TRAVELING)
            {
                if (computeTimePassed(travellingTimer, getCurrentMillis()) > 10000)
                {
                    status = SCANNING;
                    printWithTimestamp("Travelling for too long...", YELLOW_TEXT_BLACK_BACKGROUND);
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