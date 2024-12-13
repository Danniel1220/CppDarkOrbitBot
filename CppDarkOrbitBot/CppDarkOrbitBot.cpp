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
    float currentFPS = 1 / (float(loopDuration) / 1000);

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

void drawMatchedTargets(vector<int> selectedIndices, vector<Rect> boxes, vector<double> matchScores, Mat &screenshot, string templateName)
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

void drawMatchedTargets2(vector<Rect> rectangles, vector<double> confidences, Mat &screenshot, string templateName)
{
    for (int i = 0; i < rectangles.size(); i++)
    {
        cout << "Drawing " << templateName << " at " << rectangles[i].x << " " << rectangles[i].y << endl;

        // drawing rectangle
        cv::rectangle(screenshot, rectangles[i], Scalar(0, 255, 0), 2);

        // creating label with confidence score
        ostringstream labelStream;
        labelStream << std::fixed << std::setprecision(2) << confidences[i];
        string label = templateName + " | " + labelStream.str();

        // calculating position for the label (so it doesnt go off screen
        int baseLine = 0;
        Size labelSize = cv::getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        Point labelPos(rectangles[i].x, rectangles[i].y - 10); // position above the rectangle
        if (labelPos.y < 0) labelPos.y = rectangles[i].y + labelSize.height + 10; // adjust if too close to top edge

        // drawing background rectangle for the label
        cv::rectangle(screenshot, labelPos + Point(0, baseLine), labelPos + Point(labelSize.width, -labelSize.height), Scalar(0, 255, 0), FILLED);

        // drawing the label text
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

    setConsoleStyle(YELLOW_TEXT_BLACK_BACKGROUND);
    string msg = "Thread found " + to_string(matchRectangles.size()) + " matches for " + templateName;
    cout << msg << endl;
    setConsoleStyle(DEFAULT);
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
        drawMatchedTargets(deduplicatedMatchIndexes[i], matchedRectangles[i], matchedConfidences[i], screenshot, templateNames[i]);
    }
}

void matchTemplates2(Mat& screenshot, int screenshotOffset, vector<vector<Mat>> &screenshotGrid, vector<Mat> &templateGrayscales, vector<Mat> &templateAlphas, vector<string> &templateNames)
{
    double confidenceThreshold = 0.75;

    // rows - columns - templates - matches
    vector<vector<vector<vector<Point>>>> matchedLocations(screenshotGrid.size(), vector<vector<vector<Point>>>(screenshotGrid[0].size(), vector<vector<Point>>(templateGrayscales.size())));
    vector<vector<vector<vector<double>>>> matchedConfidences(screenshotGrid.size(), vector<vector<vector<double>>>(screenshotGrid[0].size(), vector<vector<double>>(templateGrayscales.size())));
    vector<vector<vector<vector<Rect>>>> matchedRectangles(screenshotGrid.size(), vector<vector<vector<Rect>>>(screenshotGrid[0].size(), vector<vector<Rect>>(templateGrayscales.size())));
    vector<vector<vector<vector<int>>>> deduplicatedMatchIndexes(screenshotGrid.size(), vector<vector<vector<int>>>(screenshotGrid[0].size(), vector<vector<int>>(templateGrayscales.size())));

    vector<thread> matchingThreads;
    vector<string> threadNames;

    // for each row of the grid
    for (int gridRow = 0; gridRow < screenshotGrid.size(); gridRow++)
    {
        // for each grid cell in the row
        for (int gridColumn = 0; gridColumn < screenshotGrid[gridRow].size(); gridColumn++)
        {
            // for each template needed to be matched for each grid cell
            for (int i = 0; i < templateGrayscales.size(); i++)
            {
                string threadName = "[" + to_string(gridRow) + "][" + to_string(gridColumn) + "]/" + "t" + to_string(i);
                threadNames.emplace_back(threadName);
                cout << "Creating thread: " << threadName << endl;
                
                matchingThreads.emplace_back(matchSingleTemplate, screenshotGrid[gridRow][gridColumn], templateGrayscales[i], templateAlphas[i], templateNames[i], TM_CCOEFF_NORMED, confidenceThreshold,
                    ref(matchedLocations[gridRow][gridColumn][i]),
                    ref(matchedConfidences[gridRow][gridColumn][i]),
                    ref(matchedRectangles[gridRow][gridColumn][i]),
                    ref(deduplicatedMatchIndexes[gridRow][gridColumn][i]));
            }
        }
    }

    int i = 0;
    for (thread& t : matchingThreads)
    {
        if (t.joinable())
        {
            t.join();
            cout << "Joined thread: " << threadNames[i] << endl;
            i++;
        }
        else
        {
            setConsoleStyle(RED_TEXT_BLACK_BACKGROUND);
            cout << "COULDNT JOIN THREAD???";
        }
    }

    // templates - matches
    vector<vector<Point>> aggregatedMatchedLocations(templateGrayscales.size());
    vector<vector<double>> aggregatedMatchedConfidences(templateGrayscales.size());
    vector<vector<Rect>> aggregatedMatchedRectangles(templateGrayscales.size());
    vector<vector<int>> aggregatedDeduplicatedMatchIndexes(templateGrayscales.size());

    // grabbing the size of the grid
    int gridSizeX = screenshotGrid[0][0].cols - screenshotOffset;
    int gridSizeY = screenshotGrid[0][0].rows - screenshotOffset;

    // going through all the matches from each thread and aggregating them into one structure
    // also adjusting the location of the matched points and rects to match real the coordinates on the full screenshot

    // for each row of the grid
    for (int gridRow = 0; gridRow < screenshotGrid.size(); gridRow++)
    {
        // for each grid cell in the row
        for (int gridColumn = 0; gridColumn < screenshotGrid[gridRow].size(); gridColumn++)
        {
            // for each template matched for each grid cell
            for (int i = 0; i < templateGrayscales.size(); i++)
            {        
                // only for the deduplicated matches
                for (int j = 0; j < deduplicatedMatchIndexes[gridRow][gridColumn][i].size(); j++)
                {
                    int overlapOffsetX;

                    if (gridColumn == 0) overlapOffsetX = 0;
                    else overlapOffsetX = -screenshotOffset;
                    
                    int overlapOffsetY;

                    if (gridRow == 0) overlapOffsetY = 0;
                    else overlapOffsetY = -screenshotOffset;


                    int xOffset = gridColumn * gridSizeX + overlapOffsetX;
                    int yOffset = gridRow * gridSizeY + overlapOffsetY;

                    int deduplicatedMatchIndex = deduplicatedMatchIndexes[gridRow][gridColumn][i][j];

                    Point initialPoint = Point(
                        matchedLocations[gridRow][gridColumn][i][deduplicatedMatchIndex].x,
                        matchedLocations[gridRow][gridColumn][i][deduplicatedMatchIndex].y);

                    Point adjustedPoint = Point(
                        matchedLocations[gridRow][gridColumn][i][deduplicatedMatchIndex].x + xOffset,
                        matchedLocations[gridRow][gridColumn][i][deduplicatedMatchIndex].y + yOffset);

                    Rect adjustedRect = Rect(
                        matchedLocations[gridRow][gridColumn][i][deduplicatedMatchIndex].x + xOffset, 
                        matchedLocations[gridRow][gridColumn][i][deduplicatedMatchIndex].y + yOffset,
                        matchedRectangles[gridRow][gridColumn][i][deduplicatedMatchIndex].width, 
                        matchedRectangles[gridRow][gridColumn][i][deduplicatedMatchIndex].height);

                    aggregatedMatchedLocations[i].emplace_back(adjustedPoint);
                    aggregatedMatchedConfidences[i].emplace_back(matchedConfidences[gridRow][gridColumn][i][deduplicatedMatchIndex]);
                    aggregatedMatchedRectangles[i].emplace_back(adjustedRect);
                }
            }
        }
    }

    // drawing all the matches
    for (int i = 0; i < templateNames.size(); i++)
    {
        drawMatchedTargets2(aggregatedMatchedRectangles[i], aggregatedMatchedConfidences[i], screenshot, templateNames[i]);
    }

    //this_thread::sleep_for(seconds(5));
}

vector<vector<Mat>> divideImage(Mat image, int gridWidth, int gridHeight, int overlapAmount) 
{
    int imageWidth = image.cols;
    int imageHeight = image.rows;
    int gridCellWidth = imageWidth / gridWidth;
    int gridCellHeight = imageHeight / gridHeight;

    vector<vector<Mat>> imageGrid;

    //cout << "Screenshot grid size: " << gridCellWidth << " x " << gridCellHeight << endl;

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
                j * gridCellWidth - (j == 0 ? 0 : overlapAmount),
                i * gridCellHeight - (i == 0 ? 0 : overlapAmount),
                gridCellWidth + (j == 0 || j == gridWidth - 1 ? overlapAmount : overlapAmount * 2),
                gridCellHeight + (i == 0 || i == gridHeight - 1 ? overlapAmount : overlapAmount * 2));
            Mat gridCell = image(gridCellRect);
            
            gridRow.emplace_back(gridCell);

            //imshow("grid" + to_string(i) + to_string(j), gridCell);
            //moveWindow("grid" + to_string(i) + to_string(j), gridCellRect.x - 1920, gridCellRect.y);
        }
        imageGrid.emplace_back(gridRow);
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

        int screenshotOffset = 50;

        vector<vector<Mat>> dividedScreenshot = divideImage(screenshot, 3, 3, screenshotOffset);

        //matchTemplates(screenshot, templateGrayscales, templateAlphas, templateNames);

        matchTemplates2(screenshot, screenshotOffset, dividedScreenshot, templateGrayscales, templateAlphas, templateNames);






        // keep track of when the loop ends, to calculate how long the loop took and fps
        int duration = computeMillisPassed(start, getCurrentMillis());
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