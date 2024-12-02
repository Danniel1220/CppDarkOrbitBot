#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <windows.h>

using namespace std;
using namespace cv;

HANDLE consoleHandle;

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

vector<Mat> loadImages(vector<string> png_paths)
{
    SetConsoleTextAttribute(consoleHandle, 6); // yellow text
    cout << "Loading images..." << endl;

    vector<Mat> png_images;
    bool loadingFailed = 0;

    for (int i = 0; i < png_paths.size(); i++)
    {
        Mat img = cv::imread(png_paths[i], IMREAD_GRAYSCALE);
        if (img.empty())
        {
            SetConsoleTextAttribute(consoleHandle, 4); // red text
            cout << "Error: Could not load image: " << png_paths[i] << endl;
            loadingFailed = 1;
            SetConsoleTextAttribute(consoleHandle, 6); // yellow text
        }
        else
        {
            png_images.push_back(img);
            cout << "Successfully loaded image: " << png_paths[i] << endl;
        }
    }

    if (loadingFailed) {
        SetConsoleTextAttribute(consoleHandle, 4); // red text
        cout << "One or more errors occured while loading images..." << endl;
    }
    else
    {
        SetConsoleTextAttribute(consoleHandle, 2); // green text
        cout << "Successfully loaded all images!" << endl;
    }

    SetConsoleTextAttribute(consoleHandle, 15); // white (regular) text

    return png_images;
}

int main()
{
    consoleHandle = GetStdHandle(STD_OUTPUT_HANDLE);

    vector<string> pngPaths = {
        "C:\\Users\\climd\\source\\repos\\CppDarkOrbitBot\\pngs\\palladium1.png",
        "C:\\Users\\climd\\source\\repos\\CppDarkOrbitBot\\pngs\\prometium1.png",
        "C:\\Users\\climd\\source\\repos\\CppDarkOrbitBot\\pngs\\cargo_icon.png",
    };
    vector<Mat> pngImages = loadImages(pngPaths);
    vector<Mat> targetImages;
    vector<Mat> targetAlphas;

    for (Mat png : pngImages)
    {
        vector<Mat> pngChannels;
        cv::split(png, pngChannels);

        targetImages.push_back(pngChannels[0]); // extracting the base channel (only 1 needed because of grayscale images)
        
        // vector out of range?
        //targetAlphas.push_back(pngChannels[3]); // extracting the alpha channel

    }


    return 0;
}


