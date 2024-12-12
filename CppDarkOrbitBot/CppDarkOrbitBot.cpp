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

void computeFrameRate(int loopDuration, float &totalTime, float &totalFrames, string &currentFPSString, string &averageFPSString)
{
    float currentFPS = 1 / (loopDuration / 1000);

    totalTime += loopDuration;
    totalFrames++;

    float averageMillis = totalTime / totalFrames;
    float averageFPS = 1 / (averageMillis / 1000);

    stringstream frameRateStream;
    stringstream averageFrameRateStream;
    frameRateStream << fixed << setprecision(2);
    averageFrameRateStream << fixed << setprecision(2);
    frameRateStream << loopDuration << " ms | " << currentFPS << " FPS";
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
    cout << "Starting match single template function" << endl;

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

void matchTemplates2(Mat& screenshot, vector<vector<Mat>> &screenshotGrid, vector<Mat> &templateGrayscales, vector<Mat> &templateAlphas, vector<string> &templateNames)
{
    vector<vector<Mat>> grayscaleScreenshotGrid;

    for (int i = 0; i < screenshotGrid.size(); i++)
    {
        vector<Mat> grayscaleScreenshotGridRow;
        for (int j = 0; j < screenshotGrid[i].size(); j++)
        {
            Mat grayscaleGridCell;
            cvtColor(screenshotGrid[i][j], grayscaleGridCell, cv::COLOR_BGR2GRAY);
            grayscaleScreenshotGridRow.emplace_back(grayscaleGridCell);
        }
        grayscaleScreenshotGrid.emplace_back(grayscaleScreenshotGridRow);
    }

    double confidenceThreshold = 0.75;

    vector<vector<vector<vector<Point>>>> matchedLocations;
    vector<vector<vector<vector<double>>>> matchedConfidences;
    vector<vector<vector<vector<Rect>>>> matchedRectangles;
    vector<vector<vector<vector<int>>>> deduplicatedMatchIndexes;

    vector<thread> matchingThreads;
    vector<string> threadNames;

    // for each row of the grid
    for (int gridRow = 0; gridRow < grayscaleScreenshotGrid.size(); gridRow++)
    {
        // for each grid cell in the row
        for (int gridColumn = 0; gridColumn < grayscaleScreenshotGrid[gridRow].size(); gridColumn++)
        {
            // for each template needed to be matched for each grid
            for (int i = 0; i < templateGrayscales.size(); i++)
            {
                string threadName = "[" + to_string(gridRow) + "][" + to_string(gridColumn) + "] - " + to_string(i);
                threadNames.emplace_back(threadName);
                cout << "Creating thread: " << threadName << endl;
                
                /*matchingThreads.emplace_back(matchSingleTemplate, grayscaleScreenshotGrid[gridRow][gridColumn], templateGrayscales[i], templateAlphas[i], templateNames[i], TM_CCOEFF_NORMED, confidenceThreshold,
                    ref(matchedLocations[gridRow][gridColumn][i]),
                    ref(matchedConfidences[gridRow][gridColumn][i]),
                    ref(matchedRectangles[gridRow][gridColumn][i]),
                    ref(deduplicatedMatchIndexes[gridRow][gridColumn][i]));*/

                matchSingleTemplate(grayscaleScreenshotGrid[gridRow][gridColumn], templateGrayscales[i], templateAlphas[i], templateNames[i], TM_CCOEFF_NORMED, confidenceThreshold,
                    ref(matchedLocations[gridRow][gridColumn][i]),
                    ref(matchedConfidences[gridRow][gridColumn][i]),
                    ref(matchedRectangles[gridRow][gridColumn][i]),
                    ref(deduplicatedMatchIndexes[gridRow][gridColumn][i]));

                cout << "Created thread!" << endl;
            }
        }
    }

    cout << "da";

    int i = 0;
    for (thread& t : matchingThreads)
    {
        if (t.joinable())
        {
            t.join();
            cout << "Joined thread: " << threadNames[i];
            i++;
        }
        else
        {
            setConsoleStyle(RED_TEXT_BLACK_BACKGROUND);
            cout << "COULDNT JOIN THREAD???";
        }
    }

    this_thread::sleep_for(seconds(2));
}

vector<vector<Mat>> divideImage(Mat image, int gridWidth, int gridHeight, int overlapAmount) 
{
    int imageWidth = image.cols;
    int imageHeight = image.rows;
    int gridCellWidth = imageWidth / gridWidth;
    int gridCellHeight = imageHeight / gridHeight;

    vector<vector<Mat>> imageGrid;

    cout << gridCellWidth << " " << gridCellHeight << endl;

    for (int i = 0; i < gridHeight; i++)
    {
        vector<Mat> gridRow;
        for (int j = 0; j < gridWidth; j++)
        {
            // creating a rectangle for the region of screen to crop
            // the rectangles are extended in every direction to overlap eachother by a fixed amount
            // to prevent the loss of possible matches that happen to be on the edges of a grid cell
            // also the grids will not be extended outside the image if that side of the grid is on the edge
            Rect gridCellRect = Rect(
                i * gridCellWidth - (i == 0 ? 0 : overlapAmount),
                j * gridCellHeight - (j == 0 ? 0 : overlapAmount),
                gridCellWidth + (i == gridWidth - 1 ? overlapAmount : overlapAmount * 2),
                gridCellHeight + (j == gridWidth - 1 ? overlapAmount : overlapAmount * 2));
            Mat gridCell = image(gridCellRect);
            
            gridRow.emplace_back(gridCell);

            //imshow("grid" + to_string(i) + to_string(j), gridCell);
            //moveWindow("grid" + to_string(i) + to_string(j), gridCellRect.x - 1920, gridCellRect.y);
        }
        imageGrid.emplace_back(gridRow);
    }

    for (int i = 0; i < imageGrid.size(); i++)
    {
        for (int j = 0; j < imageGrid[i].size(); j++)
        {
            if (imageGrid[i][j].empty()) cout << "IMAGE GRID " << i << " " << j << " EMPTY IN IMAGE DIVIDE FUNCTION";
        }
    }

    return imageGrid;
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
        long long start = getCurrentMillis();

        Mat screenshot = screenshotWindow(darkOrbitHandle);
        if (screenshot.empty()) {
            setConsoleStyle(RED_TEXT_BLACK_BACKGROUND);
            cout << "Failed to capture the window as Mat." << endl;
            return -1;
        }

        //matchTemplates(screenshot, templateGrayscales, templateAlphas, templateNames);

        vector<vector<Mat>> dividedScreenshot = divideImage(screenshot, 2, 2, 0);

        matchTemplates2(screenshot, dividedScreenshot, templateGrayscales, templateAlphas, templateNames);








        // keep track of when the loop ends, to calculate how long the loop took and fps
        int duration = computeMillisPassed(start, getCurrentMillis());
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