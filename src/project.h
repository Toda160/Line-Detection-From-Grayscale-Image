#ifndef PROJECT_H
#define PROJECT_H
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

// Global arrays for neighbor offsets
const int di[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
const int dj[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };

// Structure for gradient components
typedef struct {
    Mat x;
    Mat y;
    Mat magnitude;
    Mat direction;
} gradients_structure;

// Structure for filter kernels
typedef struct {
    int* filter_x;
    int* filter_y;
    int* di;
    int* dj;
} filter_structure;

// Structure for Hough lines
typedef struct {
    float rho;       // Distance from origin
    float theta;     // Angle in radians
    int votes;       // Accumulator value (number of votes)
} line_structure;

// Function declarations
int* compute_histogram_naive(Mat source);
vector<float> compute_kernel_1D(int kernel_size);
Mat apply_gaussian_filtering_1D(Mat source, int kernel_size);
filter_structure get_filter(string filter_type);
gradients_structure compute_gradients(Mat source, const int* filter_x, const int* filter_y, const int* di, const int* dj);
Mat non_maxima_gradient_suppression(gradients_structure gradient);
Mat normalize_suppression(Mat suppression, string filter_type);
int adaptive_threshold(Mat magnitude, float p, bool verbose);
Mat histeresis_thresholding(Mat source, int th);
Mat apply_Canny(Mat source, int low_threshold, int high_threshold, string filter_type, bool verbose);
vector<line_structure> apply_hough_transform(Mat edges, int threshold, bool verbose = false);
Mat draw_detected_lines(Mat original, const vector<line_structure>& lines, int max_lines = -1, Scalar color = Scalar(0, 0, 255));

#endif