#include <iostream>
#include <opencv2/opencv.hpp>
#include "src/project.h"
#include <chrono>
using namespace std;
using namespace cv;

int main() {
    Mat source = imread("C:/Users/Toda/Documents/GitHub/Line-Detection-From-Grayscale-Image/images/grayscale_road.jpg", IMREAD_GRAYSCALE);

    auto start_custom = chrono::high_resolution_clock::now();

    Mat edges_custom = apply_Canny(source, 30, 100, "Sobel", false);

    vector<line_structure_prob> lines_custom = apply_probabilistic_hough_transform(edges_custom, 30, 70, 20, false);

    Mat result_custom = draw_detected_lines(source, lines_custom, Scalar(0, 0, 255), 3);
    auto end_custom = chrono::high_resolution_clock::now();
    auto duration_custom = chrono::duration_cast<chrono::milliseconds>(end_custom - start_custom).count();

    auto start_opencv = chrono::high_resolution_clock::now();
    Mat blurred;
    GaussianBlur(source, blurred, Size(7, 7), 1.5);

    Mat edges_opencv;
    Canny(blurred, edges_opencv, 30, 100);

    vector<Vec4i> lines_opencv;
    HoughLinesP(edges_opencv, lines_opencv, 1, CV_PI / 180, 30, 70, 20);

    Mat result_opencv;
    cvtColor(source, result_opencv, COLOR_GRAY2BGR);
    for (size_t i = 0; i < lines_opencv.size(); ++i) {
        Vec4i l = lines_opencv[i];
        double length = norm(Point(l[0], l[1]) - Point(l[2], l[3]));
        if (length > 70) {
            line(result_opencv, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3);
        }
    }
    auto end_opencv = chrono::high_resolution_clock::now();
    auto duration_opencv = chrono::duration_cast<chrono::milliseconds>(end_opencv - start_opencv).count();

    int edge_pixels_custom = countNonZero(edges_custom);
    int edge_pixels_opencv = countNonZero(edges_opencv);

    double avg_length_custom = 0.0;
    for (const auto& line : lines_custom) {
        double length = norm(line.end - line.start);
        if (length > 70) avg_length_custom += length;
    }
    int num_lines_custom = lines_custom.size();
    if (num_lines_custom > 0) avg_length_custom /= num_lines_custom;

    double avg_length_opencv = 0.0;
    int num_lines_opencv_filtered = 0;
    for (const auto& l : lines_opencv) {
        double length = norm(Point(l[0], l[1]) - Point(l[2], l[3]));
        if (length > 70) {
            avg_length_opencv += length;
            num_lines_opencv_filtered++;
        }
    }
    if (num_lines_opencv_filtered > 0) avg_length_opencv /= num_lines_opencv_filtered;

    cout << "Custom Implementation:" << endl;
    cout << "  Edge Pixels: " << edge_pixels_custom << endl;
    cout << "  Number of Lines: " << num_lines_custom << endl;
    cout << "  Average Line Length: " << avg_length_custom << " pixels" << endl;
    cout << "  Execution Time: " << duration_custom << " ms" << endl;

    cout << "OpenCV Implementation:" << endl;
    cout << "  Edge Pixels: " << edge_pixels_opencv << endl;
    cout << "  Number of Lines (after filter): " << num_lines_opencv_filtered << endl;
    cout << "  Average Line Length: " << avg_length_opencv << " pixels" << endl;
    cout << "  Execution Time: " << duration_opencv << " ms" << endl;

    Mat edge_diff;
    absdiff(edges_custom, edges_opencv, edge_diff);
    normalize(edge_diff, edge_diff, 0, 255, NORM_MINMAX, CV_8U);

    Mat overlay_comparison;
    cvtColor(source, overlay_comparison, COLOR_GRAY2BGR);
    for (const auto& line : lines_custom) {
        double length = norm(line.end - line.start);
        if (length > 70) {
            cv::line(overlay_comparison, line.start, line.end, Scalar(0, 0, 255), 6);
        }
    }

    for (const auto& l : lines_opencv) {
        double length = norm(Point(l[0], l[1]) - Point(l[2], l[3]));
        if (length > 70) {
            cv::line(overlay_comparison, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255, 0, 0), 3); // Reduced thickness to 3
        }
    }

    imshow("Original Image", source);
    imshow("Edge Detection (Custom)", edges_custom);
    imshow("Edge Detection (OpenCV)", edges_opencv);
    imshow("Edge Detection Difference", edge_diff);
    imshow("Detected Lines (Custom)", result_custom);
    imshow("Detected Lines (OpenCV)", result_opencv);
    imshow("Overlay Comparison (Custom Red, OpenCV Blue)", overlay_comparison);

    waitKey(0);
    return 0;
}