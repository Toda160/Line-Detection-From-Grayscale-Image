#include <iostream>
#include <opencv2/opencv.hpp>
#include "src/project.h"  // Make sure this matches your actual header filename
using namespace std;
using namespace cv;

int main() {
    // Load the image in grayscale
    Mat source = imread("C:/Users/Toda/CLionProjects/untitled8/images/grayscale_road.jpg", IMREAD_GRAYSCALE);
    if (source.empty()) {
        cerr << "Error: Could not read the image." << endl;
        return -1;
    }

    imshow("Original", source);

    Mat edges = apply_Canny(source, 50, 150, "Sobel", true);
    imshow("Canny Edge Detection", edges);

    int hough_threshold = 50;
    vector<line_structure> detected_lines = apply_hough_transform(edges, hough_threshold, true);

    cout << "Detected " << detected_lines.size() << " lines" << endl;

    Mat result_image = draw_detected_lines(source, detected_lines, 10, Scalar(0, 0, 255));

    imshow("Detected Lines", result_image);

    waitKey(0);

    return 0;
}