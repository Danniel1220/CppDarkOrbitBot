#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <numeric>

#include "ThreadPool.h"
#include "BotUtils.h"
#include "BotCV.h"

using namespace std;
using namespace cv;

void drawMatchedTargets(vector<Rect> &rectangles, vector<double> &confidences, Mat &screenshot, string templateName)
{
    Scalar color;

    if (templateName == "prometium1.png") color = Scalar(0, 255, 0); // green
    else if (templateName == "cargo_icon.png") color = Scalar(0, 0, 255); // red
    else color = Scalar(255, 255, 255); //fallback to white in case something went wrong

    for (int i = 0; i < rectangles.size(); i++)
    {
        drawSingleTargetOnScreenshot(screenshot, rectangles[i], confidences[i], templateName, color);

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

void drawSingleTargetOnScreenshot(Mat &screenshot, Rect rectangle, double confidence, string name, Scalar color)
{
    // drawing rectangle
    cv::rectangle(screenshot, rectangle, color, 2);

    // creating label with confidence score
    ostringstream labelStream;
    labelStream << std::fixed << std::setprecision(2) << confidence;
    string label = name + " | " + labelStream.str();

    // calculating position for the label (so it doesnt go off screen
    int baseLine = 0;
    Size labelSize = cv::getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    Point labelPos(rectangle.x, rectangle.y - 10); // position above the rectangle
    if (labelPos.y < 0) labelPos.y = rectangle.y + labelSize.height + 10; // adjust if too close to top edge

    // drawing background rectangle for the label
    cv::rectangle(screenshot, labelPos + Point(0, baseLine), labelPos + Point(labelSize.width, -labelSize.height), color, FILLED);

    // drawing the label text
    cv::putText(screenshot, label, labelPos, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
}

void matchSingleTemplate(Mat screenshot, Mat templateGrayscale, Mat templateAlpha, string templateName, TemplateMatchModes matchMode, double confidenceThreshold,
    vector<double> &matchScores, vector<Rect> &matchRectangles, vector<int> &deduplicatedMatchIndexes)
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
                matchRectangles.emplace_back(Point(x, y), templateGrayscale.size());
                matchScores.emplace_back(score);
            }
        }
    }

    // applying Non-Maximum Suppression to remove duplicate matches
    double nmsThreshold = 0.3;  // overlap threshold for NMS
    applyNMS(matchRectangles, matchScores, nmsThreshold, deduplicatedMatchIndexes);
}

void matchTemplatesParallel(Mat &screenshot, int screenshotOffset, vector<vector<Mat>> &screenshotGrid, vector<Mat> &templateGrayscales, vector<Mat> &templateAlphas,
    vector<string> &templateNames, double confidenceThreshold, ThreadPool &threadPool, 
    vector<vector<double>> &resultMatchedConfidences, vector<vector<Rect>> &resultMatchedRectangles)
{
    // rows - columns - templates - matches
    vector<vector<vector<vector<double>>>> matchedConfidences(screenshotGrid.size(), vector<vector<vector<double>>>(screenshotGrid[0].size(), vector<vector<double>>(templateGrayscales.size())));
    vector<vector<vector<vector<Rect>>>> matchedRectangles(screenshotGrid.size(), vector<vector<vector<Rect>>>(screenshotGrid[0].size(), vector<vector<Rect>>(templateGrayscales.size())));
    vector<vector<vector<vector<int>>>> firstNMSPassDeduplicatedIndexes(screenshotGrid.size(), vector<vector<vector<int>>>(screenshotGrid[0].size(), vector<vector<int>>(templateGrayscales.size())));

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
                    ref(matchedConfidences[gridRow][gridColumn][i]),
                    ref(matchedRectangles[gridRow][gridColumn][i]),
                    ref(firstNMSPassDeduplicatedIndexes[gridRow][gridColumn][i])));
            }
        }
    }
    threadPool.waitForCompletion();

    // grabbing the size of the grid
    int gridSizeX = screenshotGrid[0][0].cols - screenshotOffset;
    int gridSizeY = screenshotGrid[0][0].rows - screenshotOffset;

    // templates - matches
    vector<vector<Rect>> firstNMSPassMatchedRectangles(templateGrayscales.size());
    vector<vector<double>> firstNMSPassMatchedConfidences(templateGrayscales.size());

    // going through all the matches from each thread and aggregating the deduplicated ones on the first pass of NMS into one structure
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
                for (int j = 0; j < firstNMSPassDeduplicatedIndexes[gridRow][gridColumn][i].size(); j++)
                {
                    int overlapOffsetX = gridColumn == 0 ? 0 : -screenshotOffset;
                    int overlapOffsetY = gridRow == 0 ? 0 : -screenshotOffset;

                    int xOffset = gridColumn * gridSizeX + overlapOffsetX;
                    int yOffset = gridRow * gridSizeY + overlapOffsetY;

                    int deduplicatedMatchIndex = firstNMSPassDeduplicatedIndexes[gridRow][gridColumn][i][j];

                    Rect adjustedRect = Rect(
                        matchedRectangles[gridRow][gridColumn][i][deduplicatedMatchIndex].x + xOffset, 
                        matchedRectangles[gridRow][gridColumn][i][deduplicatedMatchIndex].y + yOffset,
                        matchedRectangles[gridRow][gridColumn][i][deduplicatedMatchIndex].width, 
                        matchedRectangles[gridRow][gridColumn][i][deduplicatedMatchIndex].height);

                    firstNMSPassMatchedRectangles[i].emplace_back(adjustedRect);
                    firstNMSPassMatchedConfidences[i].emplace_back(matchedConfidences[gridRow][gridColumn][i][deduplicatedMatchIndex]);
                }
            }
        }
    }

    // applying a second pass of NMS for each template because there might still be duplicates caused by the overlapping screenshot grid cells
    vector<vector<int>> secondNMSPassDeduplicatedIndexes(templateGrayscales.size());
    // for each template
    for (int i = 0; i < firstNMSPassMatchedRectangles.size(); i++)
    {
        applyNMS(firstNMSPassMatchedRectangles[i], firstNMSPassMatchedConfidences[i], 0.3, secondNMSPassDeduplicatedIndexes[i]);
        // placing the deduplicated matches into the final result vectors
        for (int j = 0; j < secondNMSPassDeduplicatedIndexes[i].size(); j++)
        {
            resultMatchedRectangles[i].emplace_back(firstNMSPassMatchedRectangles[i][secondNMSPassDeduplicatedIndexes[i][j]]);
            resultMatchedConfidences[i].emplace_back(firstNMSPassMatchedConfidences[i][secondNMSPassDeduplicatedIndexes[i][j]]);
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

Mat screenshotWindow(HWND hwnd)
{
    HDC hscreenDC, hwindowCompatibleDC;
    int height, width, srcheight, srcwidth;
    HBITMAP hbwindow;
    cv::Mat src;
    BITMAPINFOHEADER bi;

    // Capture the entire screen using screen device context
    hscreenDC = GetDC(0); // Get the screen device context (0 means the entire screen)
    hwindowCompatibleDC = CreateCompatibleDC(hscreenDC);
    SetStretchBltMode(hwindowCompatibleDC, COLORONCOLOR);

    // Get screen dimensions
    srcheight = GetSystemMetrics(SM_CYSCREEN);
    srcwidth = GetSystemMetrics(SM_CXSCREEN);
    height = srcheight;
    width = srcwidth;

    // Create a bitmap compatible with the screen
    hbwindow = CreateCompatibleBitmap(hscreenDC, width, height);  // Create a bitmap for the full screen
    bi.biSize = sizeof(BITMAPINFOHEADER);
    bi.biWidth = width;
    bi.biHeight = -height;  // Negative height for correct orientation
    bi.biPlanes = 1;
    bi.biBitCount = 32;  // RGBA format
    bi.biCompression = BI_RGB;
    bi.biSizeImage = 0;
    bi.biXPelsPerMeter = 1;
    bi.biYPelsPerMeter = 2;
    bi.biClrUsed = 3;
    bi.biClrImportant = 4;

    // Select the newly created compatible bitmap into the DC
    SelectObject(hwindowCompatibleDC, hbwindow);

    // Capture the entire screen into the bitmap
    if (!StretchBlt(hwindowCompatibleDC, 0, 0, width, height, hscreenDC, 0, 0, srcwidth, srcheight, SRCCOPY)) {
        cerr << "Failed to capture the full screen!" << endl;
        return cv::Mat();  // Return empty matrix on failure
    }

    // Create an empty matrix to store the captured image (RGBA)
    src.create(height, width, CV_8UC4);  // RGBA format
    if (GetDIBits(hwindowCompatibleDC, hbwindow, 0, height, src.data, (BITMAPINFO*)&bi, DIB_RGB_COLORS) == 0) {
        cerr << "Failed to retrieve bitmap data!" << endl;
        return cv::Mat();  // Return empty matrix on failure
    }

    // Release resources
    DeleteObject(hbwindow);
    DeleteDC(hwindowCompatibleDC);
    ReleaseDC(0, hscreenDC);

    // Get the window's position and size
    RECT windowRect;
    GetWindowRect(hwnd, &windowRect);  // Get the full window rectangle (including title bar)

    // Calculate the width and height of the client area
    int windowX = windowRect.left;   // Left position of the window
    int windowY = windowRect.top;    // Top position of the window
    int windowWidth = windowRect.right - windowRect.left;  // Window width
    int windowHeight = windowRect.bottom - windowRect.top; // Window height

    // Ensure the crop region is within bounds of the captured image
    if (windowX < 0) windowX = 0;
    if (windowY < 0) windowY = 0;
    if (windowX + windowWidth > src.cols) windowWidth = src.cols - windowX;
    if (windowY + windowHeight > src.rows) windowHeight = src.rows - windowY;

    // Now crop the screen capture to the window's size and position
    cv::Rect cropRegion(windowX, windowY, windowWidth, windowHeight);

    // Perform the crop operation
    cv::Mat windowCapture = src(cropRegion);  // Crop the relevant portion

    return windowCapture;
}

double calculateIoU(const cv::Rect& a, const cv::Rect& b) {
    int x1 = max(a.x, b.x);
    int y1 = max(a.y, b.y);
    int x2 = min(a.x + a.width, b.x + b.width);
    int y2 = min(a.y + a.height, b.y + b.height);

    int intersection = max(0, x2 - x1) * max(0, y2 - y1);
    int unionArea = a.area() + b.area() - intersection;

    return static_cast<double>(intersection) / unionArea;
}

void applyNMS(const vector<Rect> &boxes, const vector<double> &scores, double nmsThreshold, vector<int> &indices) {
    vector<int> sortedIndices(boxes.size());
    iota(sortedIndices.begin(), sortedIndices.end(), 0);

    // Sort indices by score in descending order
    sort(sortedIndices.begin(), sortedIndices.end(), [&](int i1, int i2) {
        return scores[i1] > scores[i2];
        });

    vector<bool> suppressed(boxes.size(), false);

    for (size_t i = 0; i < sortedIndices.size(); ++i) {
        int idx = sortedIndices[i];
        if (suppressed[idx]) continue;

        indices.push_back(idx);

        for (size_t j = i + 1; j < sortedIndices.size(); ++j) {
            int otherIdx = sortedIndices[j];
            if (suppressed[otherIdx]) continue;

            if (calculateIoU(boxes[idx], boxes[otherIdx]) > nmsThreshold) {
                suppressed[otherIdx] = true;
            }
        }
    }
}