#ifndef LAB10_H
#define LAB10_H
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

const int di[8] = { 0,-1,-1, -1, 0, 1, 1, 1 };
const int dj[8] = { 1, 1, 0, -1, -1,-1, 0, 1 };

Mat apply_Canny(Mat source, int low_threshold, int high_threshold, string filter_type, bool verbose);

typedef struct{
    Mat x;
    Mat y;
    Mat magnitude;
    Mat direction;
} gradients_structure;

vector<float> compute_kernel_1D(int kernel_size);

Mat apply_gaussian_filtering_1D(Mat source, int kernel_size);

typedef struct{
    int* filter_x;
    int* filter_y;
    int* di;
    int* dj;
}filter_structure;

filter_structure get_filter(string filter_type);

Mat non_maxima_gradient_supression(gradients_structure gradient);

gradients_structure compute_gradients(Mat source, const int* filter_x, const int* filter_y, const int* di, const int* dj);

// New structure for Hough lines
typedef struct{
    float rho;       // Distance from origin
    float theta;     // Angle in radians
    int votes;       // Accumulator value (number of votes)
} line_structure;

// New function declarations for Hough transform
vector<line_structure> apply_hough_transform(Mat edges, int threshold, bool verbose = false);
Mat draw_detected_lines(Mat original, const vector<line_structure>& lines, int max_lines = -1, Scalar color = Scalar(0, 0, 255));

#endif