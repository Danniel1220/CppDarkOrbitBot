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

using namespace std;
using namespace cv;
using namespace chrono;

HWND darkOrbitHandle;

void computeFrameRate(milliseconds loopDuration, float &totalTime, float &totalFrames, string &currentFPSString, string &averageFPSString)
{
    float millis = loopDuration.count();
    float currentFPS = 1 / (millis / 1000);

    totalTime += millis;
    totalFrames++;

    float averageMillis = totalTime / totalFrames;
    float averageFPS = 1 / (averageMillis / 1000);

    stringstream frameRateStream;
    stringstream averageFrameRateStream;
    frameRateStream << fixed << setprecision(2);
    averageFrameRateStream << fixed << setprecision(2);
    frameRateStream << millis << " ms | " << currentFPS << " FPS";
    averageFrameRateStream << averageMillis << " ms | " << averageFPS << " FPS | avg";

    currentFPSString = frameRateStream.str();
    averageFPSString = averageFrameRateStream.str();
}

void drawOnMatchedTarget(vector<int> selectedIndices, vector<Rect> boxes, vector<double> matchScores, Mat &screenshot, string templateName)
{
    // Draw the final matches with labels
    for (int idx : selectedIndices) {
        const Rect& box = boxes[idx];
        double confidence = matchScores[idx];

        // Draw rectangle
        cv::rectangle(screenshot, box, Scalar(0, 255, 0), 2);

        // Create label with confidence score
        std::ostringstream labelStream;
        labelStream << std::fixed << std::setprecision(2) << confidence;
        std::string label = templateName + " | " + labelStream.str();

        // Calculate position for the label
        int baseLine = 0;
        Size labelSize = cv::getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        Point labelPos(box.x, box.y - 10); // Position above the rectangle
        if (labelPos.y < 0) labelPos.y = box.y + labelSize.height + 10; // Adjust if too close to top edge

        // Draw background rectangle for the label
        cv::rectangle(screenshot, labelPos + Point(0, baseLine), labelPos + Point(labelSize.width, -labelSize.height), Scalar(0, 255, 0), FILLED);

        // Put the label text
        cv::putText(screenshot, label, labelPos, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
    }
}

void matchSingleTemplate(Mat &screenshot, Mat templateGrayscale, Mat templateAlpha, string templateName)
{

}

void matchTemplates(Mat &screenshot, vector<Mat> templateGrayscales, vector<Mat> templateAlphas, vector<string> templateNames)
{
    Mat grayscaleScreenshot;
    cv::cvtColor(screenshot, grayscaleScreenshot, cv::COLOR_BGR2GRAY);

    // Threshold for match confidence
    double confidenceThreshold = 0.75;

    vector<Point> matchLocations;
    vector<double> matchScores;

    for (int i = 0; i < templateGrayscales.size(); i++)
    {
        int result_cols = grayscaleScreenshot.cols - templateGrayscales[i].cols + 1;
        int result_rows = grayscaleScreenshot.rows - templateGrayscales[i].rows + 1;

        Mat result;
        result.create(result_rows, result_cols, CV_32FC1);

        cv::matchTemplate(grayscaleScreenshot, templateGrayscales[i], result, TM_CCOEFF_NORMED, templateAlphas[i]);
        //normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());

        // finding matches above threshold
        vector<cv::Point> matchLocations;
        vector<double> matchScores;

        for (int y = 0; y < result.rows; y++) {
            for (int x = 0; x < result.cols; x++) {
                double score = result.at<float>(y, x);
                if (score >= confidenceThreshold && !isinf(score)) {
                    matchLocations.push_back(Point(x, y));
                    matchScores.push_back(score);
                }
            }
        }

        // converting locations to rectangles
        vector<Rect> boxes;
        for (const auto& loc : matchLocations) {
            boxes.emplace_back(Rect(loc, templateGrayscales[i].size()));
        }

        // applying Non-Maximum Suppression
        double nmsThreshold = 0.3;  // overlap threshold for NMS
        vector<int> deduplicatedMatchIndexes;
        applyNMS(boxes, matchScores, nmsThreshold, deduplicatedMatchIndexes);

        drawOnMatchedTarget(deduplicatedMatchIndexes, boxes, matchScores, screenshot, templateNames[i]);
    }
}


int main() 
{
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
        "C:\\Users\\climd\\source\\repos\\CppDarkOrbitBot\\pngs\\palladium1.png",
        "C:\\Users\\climd\\source\\repos\\CppDarkOrbitBot\\pngs\\prometium1.png",
        "C:\\Users\\climd\\source\\repos\\CppDarkOrbitBot\\pngs\\cargo_icon.png",
    };
    vector<Mat> templateGrayscales;
    vector<Mat> templateAlphas;
    vector<string> templateNames;

    loadImages(pngPaths, templateGrayscales, templateAlphas);
    extractPngNames(pngPaths, templateNames);

    //showImages(templateGrayscales, "gray");
    //showImages(templateAlphas, "alpha");
    //cv::waitKey();

    float totalTime = 0.0f;
    float totalFrames = 0.0f;
    int frameCount = 0;
    float averageMillis = 0.0f;
    float averageFPS = 0.0f;

    while (true)
    {
        // keep track of when the loop starts
        time_point start = high_resolution_clock::now();

        Mat screenshot = screenshotWindow(darkOrbitHandle);
        if (screenshot.empty()) {
            setConsoleStyle(RED_TEXT_BLACK_BACKGROUND);
            cout << "Failed to capture the window as Mat." << endl;
            return -1;
        }

        matchTemplates(screenshot, templateGrayscales, templateAlphas, templateNames);











        // keep track of when the loop ends, to calculate how long the loop took and fps
        time_point end = high_resolution_clock::now();
        milliseconds duration = duration_cast<milliseconds>(end - start);
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