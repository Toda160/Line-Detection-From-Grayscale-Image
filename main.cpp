#include <iostream>
#include <opencv2/opencv.hpp>
#include "src/project.h"
using namespace std;
using namespace cv;

int main() {
    // Read the image
    Mat source = imread("C:/Users/Toda/Documents/GitHub/Line-Detection-From-Grayscale-Image/images/grayscale_pod.jpg", IMREAD_GRAYSCALE);
    if (source.empty()) {
        cout << "Error: Could not read the image." << endl;
        return -1;
    }

    // Apply Canny edge detection with adjusted thresholds
    Mat edges = apply_Canny(source, 50, 100, "Sobel", false); // Adjusted to 30, 100

    // Apply Probabilistic Hough transform to detect lines with tuned parameters
    vector<line_structure_prob> lines = apply_probabilistic_hough_transform(edges, 30, 70, 20, false); // Adjusted to 30, 70, 20

    // Draw detected lines on the original image
    Mat result = draw_detected_lines(source, lines);
    
    // Display results
    imshow("Original Image", source);
    imshow("Edge Detection", edges);
    imshow("Detected Lines", result);
    
    // Wait for user input
    waitKey(0);
    return 0;
}