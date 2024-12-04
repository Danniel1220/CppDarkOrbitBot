#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <filesystem>
#include <windows.h>
#include "CppDarkOrbitBot.h"
#include <regex>
#include <string>
#include <chrono>


using namespace std;
using namespace cv;
using namespace std::chrono;

HANDLE consoleHandle;
HWND darkOrbitHandle;

const int GREEN_TEXT_BLACK_BACKGROUND = 2;
const int RED_TEXT_BLACK_BACKGROUND = 4;
const int YELLOW_TEXT_BLACK_BACKGROUND = 6;
const int DEFAULT = 15;

void setConsoleStyle(int style)
{
    SetConsoleTextAttribute(consoleHandle, style);
}

void testConsoleColors(HANDLE handle) {
    for (int k = 1; k < 255; k++)
    {
        // pick the colorattribute k you want
        SetConsoleTextAttribute(handle, k);
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

vector<Mat> loadImages(vector<string> paths, vector<Mat> &grayscales, vector<Mat> &alphas)
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

void showImages(vector<Mat> &targetGrayImages, string name)
{
    for (int i = 0; Mat target : targetGrayImages)
    {
        cv::imshow(name + to_string(i), target);
        i++;
    }
}

void extractPngNames(vector<string> pngPaths, vector<string> &targetNames)
{
    setConsoleStyle(YELLOW_TEXT_BLACK_BACKGROUND);
    cout << "Extracting png file names..." << endl;

    regex pngPathRegex(R"([^\\/:*?"<>|]+\.png$)");
    smatch regexMatch;

    bool extractionFailed = false;

    for (string pngPath: pngPaths)
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
        std::cerr << "Failed to capture the full screen!" << std::endl;
        return cv::Mat();  // Return empty matrix on failure
    }

    // Create an empty matrix to store the captured image (RGBA)
    src.create(height, width, CV_8UC4);  // RGBA format
    if (GetDIBits(hwindowCompatibleDC, hbwindow, 0, height, src.data, (BITMAPINFO*)&bi, DIB_RGB_COLORS) == 0) {
        std::cerr << "Failed to retrieve bitmap data!" << std::endl;
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

void matchTemplate(Mat &screenshot, vector<Mat> templateGrayscales, vector<Mat> templateAlphas, vector<string> templateNames)
{
    Mat grayscaleScreenshot;
    cv::cvtColor(screenshot, grayscaleScreenshot, cv::COLOR_BGR2GRAY);

    for (int i = 0; i < templateGrayscales.size(); i++)
    {
        int result_cols = grayscaleScreenshot.cols - templateGrayscales[i].cols + 1;
        int result_rows = grayscaleScreenshot.rows - templateGrayscales[i].rows + 1;

        Mat result;
        result.create(result_rows, result_cols, CV_32FC1);

        cv::matchTemplate(grayscaleScreenshot, templateGrayscales[i], result, TM_CCORR_NORMED, templateAlphas[i]);
        normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());

        double minVal; double maxVal; Point minLoc; Point maxLoc;
        Point matchLoc;
        minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
        
        matchLoc = maxLoc;
        cout << templateNames[i] << " " << matchLoc << endl;

        // draw to the screenshot provided by refference that will later get shown
        rectangle(screenshot, matchLoc, Point(matchLoc.x + templateGrayscales[i].cols, matchLoc.y + templateGrayscales[i].rows), Scalar(0, 255, 0), 2, 8, 0);
        rectangle(result, matchLoc, Point(matchLoc.x + templateGrayscales[i].cols, matchLoc.y + templateGrayscales[i].rows), Scalar(0, 255, 0), 2, 8, 0);
    }
}


int main() 
{
    consoleHandle = GetStdHandle(STD_OUTPUT_HANDLE);
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
            cout << "Failed to capture the window as Mat." << std::endl;
            return -1;
        }

        matchTemplate(screenshot, templateGrayscales, templateAlphas, templateNames);











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