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

void matchSingleTemplate(Mat screenshot, Mat templateGrayscale, Mat templateAlpha, string templateName, TemplateMatchModes matchMode, double confidenceThreshold,
    vector<Point> &matchLocations, vector<double> &matchScores, vector<Rect> &matchRectangles, vector<int> &deduplicatedMatchIndexes)
{
    Mat grayscaleScreenshot;
    cv::cvtColor(screenshot, grayscaleScreenshot, cv::COLOR_BGR2GRAY);

    int result_cols = grayscaleScreenshot.cols - templateGrayscale.cols + 1;
    int result_rows = grayscaleScreenshot.rows - templateGrayscale.rows + 1;

    Mat result;
    result.create(result_rows, result_cols, CV_32FC1);

    cv::matchTemplate(grayscaleScreenshot, templateGrayscale, result, matchMode, templateAlpha);

    // finding matches above threshold
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
    for (const auto& loc : matchLocations) 
    {
        matchRectangles.emplace_back(Rect(loc, templateGrayscale.size()));
    }

    // applying Non-Maximum Suppression
    double nmsThreshold = 0.3;  // overlap threshold for NMS
    applyNMS(matchRectangles, matchScores, nmsThreshold, deduplicatedMatchIndexes);
}

void matchTemplates(Mat &screenshot, vector<Mat> &templateGrayscales, vector<Mat> &templateAlphas, vector<string> &templateNames)
{
    Mat grayscaleScreenshot;
    cv::cvtColor(screenshot, grayscaleScreenshot, cv::COLOR_BGR2GRAY);

    // Threshold for match confidence
    double confidenceThreshold = 0.75;

    vector<vector<Point>> matchedLocations(templateGrayscales.size());
    vector<vector<double>> matchedConfidences(templateGrayscales.size());
    vector<vector<Rect>> matchedRectangles(templateGrayscales.size());
    vector<vector<int>> deduplicatedMatchIndexes(templateGrayscales.size());

    vector<thread> matchingThreads;

    for (int i = 0; i < templateGrayscales.size(); i++)
    {
        cout << "Starting thread " << i << endl;
        matchingThreads.emplace_back(matchSingleTemplate, screenshot, templateGrayscales[i], templateAlphas[i], templateNames[i], TM_CCOEFF_NORMED, confidenceThreshold,
            ref(matchedLocations[i]), ref(matchedConfidences[i]), ref(matchedRectangles[i]), ref(deduplicatedMatchIndexes[i]));
    }

    cout << endl;

    int i = 0;
    for (thread &t : matchingThreads) {
        if (t.joinable()) {
            cout << "Joined thread " << i << endl;;
            t.join();
            i++;
        }
        else
        {
            setConsoleStyle(RED_TEXT_BLACK_BACKGROUND);
            cout << "COULDNT JOIN THREAD???";
        }
    }

    cout << endl;

    for (int i = 0; i < templateGrayscales.size(); i++)
    {
        drawOnMatchedTarget(deduplicatedMatchIndexes[i], matchedRectangles[i], matchedConfidences[i], screenshot, templateNames[i]);
    }
}

vector<Mat> divideImage(Mat image, int divideAmount, int overlapAmount) 
{
    int imageWidth = image.cols;
    int imageHeight = image.rows;
    tuple<int, int> gridSize = { imageWidth / divideAmount, imageHeight / divideAmount };

    cout << get<0>(gridSize) << " " << get<1>(gridSize) << endl;

    for (int i = 0; i < divideAmount; i++)
    {
        for (int j = 0; j < divideAmount; j++)
        {
            Rect gridRect = Rect(i * get<0>(gridSize), j * get<1>(gridSize), get<0>(gridSize), get<1>(gridSize));
            Mat grid = image(gridRect);
            imshow("grid" + to_string(i) + to_string(j), grid);
            moveWindow("grid" + to_string(i) + to_string(j), gridRect.x - 1920, gridRect.y);
        }
    }

    return vector<Mat>();
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

        //matchTemplates(screenshot, templateGrayscales, templateAlphas, templateNames);

        vector<Mat> dividedScreenshot = divideImage(screenshot, 3, 0);









        // keep track of when the loop ends, to calculate how long the loop took and fps
        time_point end = high_resolution_clock::now();
        milliseconds duration = duration_cast<milliseconds>(end - start);
        string frameRate;
        string averageFrameRate;
        computeFrameRate(duration, totalTime, totalFrames, frameRate, averageFrameRate);

        cv::putText(screenshot, frameRate, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        cv::putText(screenshot, averageFrameRate, cv::Point(10, 70), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

        // show the frame at the end
        //cv::imshow("CppDarkOrbitBotView", screenshot);
        int key = cv::waitKey(10);
    }

    cv::destroyAllWindows();

    return 0;
}