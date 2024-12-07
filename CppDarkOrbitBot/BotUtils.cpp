#include "BotUtils.h"
#include "Constants.h"

#include <opencv2/opencv.hpp>
#include <Windows.h>
#include <string>
#include <vector>
#include <regex>
#include <numeric>

using namespace cv;
using namespace std;

void setConsoleStyle(int style)
{
    SetConsoleTextAttribute(consoleHandle, style);
}

vector<Mat> loadImages(vector<string> paths, vector<Mat>& grayscales, vector<Mat>& alphas)
{
    setConsoleStyle(YELLOW_TEXT_BLACK_BACKGROUND);

    cout << "Loading images..." << endl;

    vector<Mat> png_images;
    bool loadingFailed = false;

    for (int i = 0; i < paths.size(); i++)
    {
        Mat png = cv::imread(paths[i], IMREAD_UNCHANGED);

        if (png.empty())
        {
            setConsoleStyle(RED_TEXT_BLACK_BACKGROUND);
            cout << "Error: Could not load image: " << paths[i] << endl;
            loadingFailed = true;
            setConsoleStyle(YELLOW_TEXT_BLACK_BACKGROUND);
        }
        else
        {
            Mat targetBase;
            Mat targetGrayBase;
            Mat targetAlpha;
            vector<Mat> channels;

            // split the image into the 4 channels (RGB + A)
            cv::split(png, channels);

            cv::merge(vector<Mat>{channels[0], channels[1], channels[2]}, targetBase);  // merge the first 3 channels (RGB) into one mat
            cv::cvtColor(targetBase, targetGrayBase, cv::COLOR_BGR2GRAY);               // convert the colored Mat to grayscale

            targetAlpha = channels[3]; // grab the last channel (alpha)


            grayscales.push_back(targetGrayBase);
            alphas.push_back(targetAlpha);

            cout << "Successfully loaded image: " << paths[i] << endl;
        }
    }

    if (loadingFailed)
    {
        setConsoleStyle(RED_TEXT_BLACK_BACKGROUND);
        cout << "One or more errors occured while loading images..." << endl;
    }
    else
    {
        setConsoleStyle(GREEN_TEXT_BLACK_BACKGROUND);
        cout << "Successfully loaded all images!" << endl;
    }

    setConsoleStyle(DEFAULT);

    return png_images;
}

void testConsoleColors() {
    for (int k = 1; k < 255; k++)
    {
        // pick the colorattribute k you want
        SetConsoleTextAttribute(consoleHandle, k);
        cout << k << " POGGERS" << endl;
    }
    // 0 = Black     8 = Gray
    // 1 = Blue      9 = Light Blue
    // 2 = Green     a = Light Green
    // 3 = Aqua      b = Light Aqua
    // 4 = Red       c = Light Red
    // 5 = Purple    d = Light Purple
    // 6 = Yellow    e = Light Yellow
    // 7 = White     f = Bright White

    // system("color <backgroundColor><textColor>");
}

void showImages(vector<Mat>& targetGrayImages, string name)
{
    for (int i = 0; Mat target : targetGrayImages)
    {
        cv::imshow(name + to_string(i), target);
        i++;
    }
}

void extractPngNames(vector<string> pngPaths, vector<string>& targetNames)
{
    setConsoleStyle(YELLOW_TEXT_BLACK_BACKGROUND);
    cout << "Extracting png file names..." << endl;

    regex pngPathRegex(R"([^\\/:*?"<>|]+\.png$)");
    smatch regexMatch;

    bool extractionFailed = false;

    for (string pngPath : pngPaths)
    {
        if (regex_search(pngPath, regexMatch, pngPathRegex)) {
            setConsoleStyle(YELLOW_TEXT_BLACK_BACKGROUND);
            targetNames.push_back(regexMatch[0]);
            cout << "Extracted file name: " << regexMatch[0] << endl;
        }
        else {
            setConsoleStyle(RED_TEXT_BLACK_BACKGROUND);
            cout << "No regex match found for path: \"" << pngPath << "\"" << endl;
            extractionFailed = true;
        }
    }

    if (extractionFailed)
    {
        setConsoleStyle(RED_TEXT_BLACK_BACKGROUND); // red text
        cout << "Regex extraction of the png file names failed..." << endl;
    }
    else
    {
        setConsoleStyle(GREEN_TEXT_BLACK_BACKGROUND);
        cout << "Successfully extracted all png file names!" << endl;
    }

    setConsoleStyle(DEFAULT);
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

void applyNMS(const vector<Rect>& boxes, const vector<double>& scores, double nmsThreshold, vector<int>& indices) {
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