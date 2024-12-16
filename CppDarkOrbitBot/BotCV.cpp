#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>

#include "ThreadPool.h"
#include "BotUtils.h"

using namespace std;
using namespace cv;

void drawMatchedTargets(vector<Rect> rectangles, vector<double> confidences, Mat &screenshot, string templateName)
{
    Scalar color;

    if (templateName == "prometium1.png") color = Scalar(0, 255, 0); // green
    else if (templateName == "cargo_icon.png") color = Scalar(0, 0, 255); // red
    else color = Scalar(255, 255, 255); //fallback to white in case something went wrong

    for (int i = 0; i < rectangles.size(); i++)
    {
        // drawing rectangle
        cv::rectangle(screenshot, rectangles[i], color, 2);

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
        cv::rectangle(screenshot, labelPos + Point(0, baseLine), labelPos + Point(labelSize.width, -labelSize.height), color, FILLED);

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
    for (int y = 0; y < result.rows; y++) 
    {
        for (int x = 0; x < result.cols; x++) 
        {
            double score = result.at<float>(y, x);
            if (score >= confidenceThreshold && !isinf(score)) 
            {
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

    // applying Non-Maximum Suppression to remove duplicate matches
    double nmsThreshold = 0.3;  // overlap threshold for NMS
    applyNMS(matchRectangles, matchScores, nmsThreshold, deduplicatedMatchIndexes);
}

void matchTemplatesParallel(Mat &screenshot, int screenshotOffset, vector<vector<Mat>> &screenshotGrid, vector<Mat> &templateGrayscales, vector<Mat> &templateAlphas,
    vector<string> &templateNames, double confidenceThreshold, ThreadPool &threadPool, 
    vector<vector<Point>> &resultMatchedLocations, vector<vector<double>> &resultMatchedConfidences, vector<vector<Rect>> &resultMatchedRectangles)
{
    // rows - columns - templates - matches
    vector<vector<vector<vector<Point>>>> matchedLocations(screenshotGrid.size(), vector<vector<vector<Point>>>(screenshotGrid[0].size(), vector<vector<Point>>(templateGrayscales.size())));
    vector<vector<vector<vector<double>>>> matchedConfidences(screenshotGrid.size(), vector<vector<vector<double>>>(screenshotGrid[0].size(), vector<vector<double>>(templateGrayscales.size())));
    vector<vector<vector<vector<Rect>>>> matchedRectangles(screenshotGrid.size(), vector<vector<vector<Rect>>>(screenshotGrid[0].size(), vector<vector<Rect>>(templateGrayscales.size())));
    vector<vector<vector<vector<int>>>> deduplicatedMatchIndexes(screenshotGrid.size(), vector<vector<vector<int>>>(screenshotGrid[0].size(), vector<vector<int>>(templateGrayscales.size())));

    // for each row of the grid
    for (int gridRow = 0; gridRow < screenshotGrid.size(); gridRow++)
    {
        // for each grid cell in the row
        for (int gridColumn = 0; gridColumn < screenshotGrid[gridRow].size(); gridColumn++)
        {
            // for each template needed to be matched for each grid cell
            for (int i = 0; i < templateGrayscales.size(); i++)
            {
                threadPool.enqueue(std::bind(matchSingleTemplate, 
                    screenshotGrid[gridRow][gridColumn], 
                    templateGrayscales[i], 
                    templateAlphas[i], 
                    templateNames[i], 
                    TM_CCOEFF_NORMED, 
                    confidenceThreshold,
                    ref(matchedLocations[gridRow][gridColumn][i]),
                    ref(matchedConfidences[gridRow][gridColumn][i]),
                    ref(matchedRectangles[gridRow][gridColumn][i]),
                    ref(deduplicatedMatchIndexes[gridRow][gridColumn][i])));
            }
        }
    }
    threadPool.waitForCompletion();

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
                    int overlapOffsetX = gridColumn == 0 ? 0 : -screenshotOffset;
                    int overlapOffsetY = gridRow == 0 ? 0 : -screenshotOffset;

                    int xOffset = gridColumn * gridSizeX + overlapOffsetX;
                    int yOffset = gridRow * gridSizeY + overlapOffsetY;

                    int deduplicatedMatchIndex = deduplicatedMatchIndexes[gridRow][gridColumn][i][j];

                    Point adjustedPoint = Point(
                        matchedLocations[gridRow][gridColumn][i][deduplicatedMatchIndex].x + xOffset,
                        matchedLocations[gridRow][gridColumn][i][deduplicatedMatchIndex].y + yOffset);

                    Rect adjustedRect = Rect(
                        matchedLocations[gridRow][gridColumn][i][deduplicatedMatchIndex].x + xOffset, 
                        matchedLocations[gridRow][gridColumn][i][deduplicatedMatchIndex].y + yOffset,
                        matchedRectangles[gridRow][gridColumn][i][deduplicatedMatchIndex].width, 
                        matchedRectangles[gridRow][gridColumn][i][deduplicatedMatchIndex].height);

                    resultMatchedLocations[i].emplace_back(adjustedPoint);
                    resultMatchedConfidences[i].emplace_back(matchedConfidences[gridRow][gridColumn][i][deduplicatedMatchIndex]);
                    resultMatchedRectangles[i].emplace_back(adjustedRect);
                }
            }
        }
    }
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
                gridCellWidth + (j == 0 && j == gridWidth - 1 ? 0 : (j == 0 || j == gridWidth - 1 ? overlapAmount : overlapAmount * 2)),
                gridCellHeight + (i == 0 && i == gridHeight - 1 ? 0 : (i == 0 || i == gridHeight - 1 ? overlapAmount : overlapAmount * 2)));
            Mat gridCell = image(gridCellRect);

            gridRow.emplace_back(gridCell);

            //imshow("grid" + to_string(i) + to_string(j), gridCell);
            //moveWindow("grid" + to_string(i) + to_string(j), gridCellRect.x - 1920, gridCellRect.y);
        }
        imageGrid.emplace_back(gridRow);
    }

    return imageGrid;
}