#include "BotUtils.h"
#include "Constants.h"

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <regex>
#include <numeric>
#include <chrono>

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

long long getCurrentMillis()
{
    std::chrono::time_point now = std::chrono::high_resolution_clock::now();
    std:chrono::milliseconds duration = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch());

    return duration.count();
}

long long getCurrentMicros()
{
    std::chrono::time_point now = std::chrono::high_resolution_clock::now();
    std:chrono::microseconds duration = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());

    return duration.count();
}

string millisToTimestamp(long long millis) 
{
    time_t seconds = millis / 1000;

    // converting to `std::tm` structure (local time)
    tm timeinfo;
    localtime_s(&timeinfo, &seconds);

    // formating the timestamp
    ostringstream oss;
    oss << put_time(&timeinfo, "%H:%M:%S");

    return oss.str();
}

void printWithTimestamp(string message)
{
    string currentTimestamp = millisToTimestamp(getCurrentMillis());
    cout << "[" << currentTimestamp << "] " << message << "\n";
}

void printTimeProfiling(long long startMicros, string message)
{
    long long currentMicros = getCurrentMicros();
    long long microsPassed = computeTimePassed(startMicros, currentMicros);

    // this means the time passed is more than a millisecond so we print the milliseconds
    if (microsPassed > 999)
    {
        long long millisPassed = microsPassed / 1000;
        stringstream str;
        str << to_string(millisPassed) << " ms - " << message;
        printWithTimestamp(str.str());
    }
    // else print as microseconds
    else
    {
        // 230 is ascii for μ
        stringstream str;
        str << to_string(microsPassed) << " " << char(230) << "s - " << message;
        printWithTimestamp(str.str());
    }
}

long long computeTimePassed(long long start, long long finish)
{
    return finish - start;
}

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