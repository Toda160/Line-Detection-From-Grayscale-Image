#include <iostream>
#include <opencv2/opencv.hpp>
#include "src/project.h"
using namespace std;
using namespace cv;

int main() {
    // Read the image
    Mat source = imread("C:/Users/Toda/CLionProjects/untitled8/images/grayscale_road.jpg", IMREAD_GRAYSCALE);
    if(source.empty()) {
        cout << "Error: Could not read the image." << endl;
        return -1;
    }

    // Apply Canny edge detection
    Mat edges = apply_Canny(source, 0, 0, "Sobel", true);
    
    // Apply Hough transform to detect lines
    vector<line_structure> lines = apply_hough_transform(edges, 100, true);
    
    // Draw detected lines on the original image
    Mat result = draw_detected_lines(source, lines, 10); // Draw top 10 lines
    
    // Display results
    imshow("Original Image", source);
    imshow("Edge Detection", edges);
    imshow("Detected Lines", result);
    
    // Wait for user input
    waitKey(0);
    return 0;
}