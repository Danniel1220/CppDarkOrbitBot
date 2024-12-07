#include "BotUtils.h"
#include "Constants.h"

#include <opencv2/opencv.hpp>
#include <Windows.h>
#include <string>
#include <vector>

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