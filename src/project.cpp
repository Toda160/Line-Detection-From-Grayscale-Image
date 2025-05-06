#include <iostream>
#include <opencv2/opencv.hpp>
#include "project.h"
#include <fstream>
#include <queue>
using namespace std;
using namespace cv;

int* compute_histogram_naive(Mat source) {
    int* histogram = (int*)calloc(256, sizeof(int));
    for (int y = 0; y < source.rows; y++) {
        for (int x = 0; x < source.cols; x++) {
            uchar intensity = source.at<uchar>(y, x);
            histogram[intensity]++;
        }
    }
    return histogram;
}

vector<float> compute_kernel_1D(int kernel_size) {
    vector<float> kernel(kernel_size);
    float sigma = kernel_size / 6.0f;
    float sigma2 = sigma * sigma;
    int center = (kernel_size - 1) / 2;
    float sum = 0.0f;
    for (int i = 0; i < kernel_size; i++) {
        int x = i - center;
        kernel[i] = exp(- (x * x) / (2.0f * sigma2));
        sum += kernel[i];
    }
    for (int i = 0; i < kernel_size; i++) {
        kernel[i] /= sum;
    }
    return kernel;
}

Mat apply_gaussian_filtering_1D(Mat source, int kernel_size) {
    Mat result;
    vector<float> kernel = compute_kernel_1D(kernel_size);
    Mat source_float;
    source.convertTo(source_float, CV_32F);
    Mat intermediate = Mat::zeros(source.size(), CV_32F);
    int radius = kernel_size / 2;
    for (int y = 0; y < source_float.rows; y++) {
        for (int x = 0; x < source_float.cols; x++) {
            float sum = 0.0f;
            for (int k = -radius; k <= radius; k++) {
                int x_k = min(max(x + k, 0), source_float.cols - 1);
                sum += source_float.at<float>(y, x_k) * kernel[k + radius];
            }
            intermediate.at<float>(y, x) = sum;
        }
    }
    result = Mat::zeros(source.size(), CV_32F);
    for (int y = 0; y < intermediate.rows; y++) {
        for (int x = 0; x < intermediate.cols; x++) {
            float sum = 0.0f;
            for (int k = -radius; k <= radius; k++) {
                int y_k = min(max(y + k, 0), intermediate.rows - 1);
                sum += intermediate.at<float>(y_k, x) * kernel[k + radius];
            }
            result.at<float>(y, x) = sum;
        }
    }
    result.convertTo(result, source.type());
    return result;
}

gradients_structure compute_gradients(Mat source, const int* filter_x, const int* filter_y, const int* di, const int* dj) {
    gradients_structure gradients;
    gradients.x = Mat::zeros(source.size(), CV_32F);
    gradients.y = Mat::zeros(source.size(), CV_32F);
    gradients.magnitude = Mat::zeros(source.size(), CV_32F);
    gradients.direction = Mat::zeros(source.size(), CV_32F);
    Mat source_float;
    source.convertTo(source_float, CV_32F);
    for (int y = 0; y < source.rows; y++) {
        for (int x = 0; x < source.cols; x++) {
            float sum_x = 0.0f, sum_y = 0.0f;
            for (int k = 0; k < 8; k++) {
                int ny = min(max(y + di[k], 0), source.rows - 1);
                int nx = min(max(x + dj[k], 0), source.cols - 1);
                sum_x += source_float.at<float>(ny, nx) * filter_x[k];
                sum_y += source_float.at<float>(ny, nx) * filter_y[k];
            }
            gradients.x.at<float>(y, x) = sum_x;
            gradients.y.at<float>(y, x) = sum_y;
            gradients.magnitude.at<float>(y, x) = sqrt(sum_x * sum_x + sum_y * sum_y);
            gradients.direction.at<float>(y, x) = atan2(sum_y, sum_x);
        }
    }
    return gradients;
}

Mat non_maxima_gradient_suppression(gradients_structure gradient) {
    Mat result = Mat::zeros(gradient.magnitude.size(), CV_32F);
    for (int y = 1; y < gradient.magnitude.rows - 1; y++) {
        for (int x = 1; x < gradient.magnitude.cols - 1; x++) {
            float mag = gradient.magnitude.at<float>(y, x);
            float angle = gradient.direction.at<float>(y, x);
            float angle_deg = angle * 180.0f / CV_PI;
            if (angle_deg < 0) angle_deg += 180.0f;
            int direction;
            if ((angle_deg >= 0 && angle_deg < 22.5) || (angle_deg >= 157.5 && angle_deg < 180))
                direction = 0;
            else if (angle_deg >= 22.5 && angle_deg < 67.5)
                direction = 1;
            else if (angle_deg >= 67.5 && angle_deg < 112.5)
                direction = 2;
            else
                direction = 3;
            float mag1, mag2;
            switch (direction) {
                case 0:
                    mag1 = gradient.magnitude.at<float>(y, x + 1);
                    mag2 = gradient.magnitude.at<float>(y, x - 1);
                    break;
                case 1:
                    mag1 = gradient.magnitude.at<float>(y - 1, x + 1);
                    mag2 = gradient.magnitude.at<float>(y + 1, x - 1);
                    break;
                case 2:
                    mag1 = gradient.magnitude.at<float>(y - 1, x);
                    mag2 = gradient.magnitude.at<float>(y + 1, x);
                    break;
                case 3:
                    mag1 = gradient.magnitude.at<float>(y - 1, x - 1);
                    mag2 = gradient.magnitude.at<float>(y + 1, x + 1);
                    break;
                default:
                    mag1 = mag2 = 0.0f;
            }
            if (mag >= mag1 && mag >= mag2) {
                result.at<float>(y, x) = mag;
            } else {
                result.at<float>(y, x) = 0.0f;
            }
        }
    }
    return result;
}

filter_structure get_filter(string filter_type) {
    filter_structure filter;
    filter.filter_x = new int[8];
    filter.filter_y = new int[8];
    filter.di = (int*)di;
    filter.dj = (int*)dj;
    if (filter_type == "Sobel") {
        int sobel_x[8] = {-1, -2, -1, 0, 0, 1, 2, 1};
        int sobel_y[8] = {1, 2, 1, 0, 0, -1, -2, -1};
        copy(sobel_x, sobel_x + 8, filter.filter_x);
        copy(sobel_y, sobel_y + 8, filter.filter_y);
    }
    else if (filter_type == "Prewitt") {
        int prewitt_x[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
        int prewitt_y[8] = {1, 1, 1, 0, 0, -1, -1, -1};
        copy(prewitt_x, prewitt_x + 8, filter.filter_x);
        copy(prewitt_y, prewitt_y + 8, filter.filter_y);
    }
    else if (filter_type == "Roberts") {
        int roberts_x[8] = {0, 0, 0, 0, -1, 0, 0, 1};
        int roberts_y[8] = {0, 0, -1, 0, 0, 0, 0, 1};
        copy(roberts_x, roberts_x + 8, filter.filter_x);
        copy(roberts_y, roberts_y + 8, filter.filter_y);
    }
    else {
        int sobel_x[8] = {-1, -2, -1, 0, 0, 1, 2, 1};
        int sobel_y[8] = {1, 2, 1, 0, 0, -1, -2, -1};
        copy(sobel_x, sobel_x + 8, filter.filter_x);
        copy(sobel_y, sobel_y + 8, filter.filter_y);
    }
    return filter;
}

Mat normalize_supression(Mat supression, string filter_type) {
    Mat result = Mat::zeros(supression.size(), CV_32F);
    double min_val, max_val;
    minMaxLoc(supression, &min_val, &max_val);
    if (max_val > 0) {
        for (int y = 0; y < supression.rows; y++) {
            for (int x = 0; x < supression.cols; x++) {
                float value = supression.at<float>(y, x);
                if (value > 0) {
                    result.at<float>(y, x) = (value / max_val) * 255.0f;
                }
                else {
                    result.at<float>(y, x) = 0.0f;
                }
            }
        }
    }
    Mat result_8u;
    result.convertTo(result_8u, CV_8U);
    result = result_8u;
    return result;
}

int adaptive_threshold(Mat magnitude, float p, bool verbose) {
    int th;
    int histogram[256] = {0};
    for (int y = 0; y < magnitude.rows; y++) {
        for (int x = 0; x < magnitude.cols; x++) {
            uchar intensity = magnitude.at<uchar>(y, x);
            histogram[intensity]++;
        }
    }
    int total_pixels = magnitude.rows * magnitude.cols;
    int zero_pixels = histogram[0];
    int non_zero_pixels = total_pixels - zero_pixels;
    int nr_non_muchie = static_cast<int>((1.0f - p) * non_zero_pixels);
    int sum = 0;
    th = 0;
    for (int i = 1; i < 256; i++) {
        sum += histogram[i];
        if (sum > nr_non_muchie) {
            th = i;
            break;
        }
    }
    if (verbose) {
        std::cout << "Adaptive Threshold: " << th << std::endl;
        std::cout << "Non-zero pixels: " << non_zero_pixels << ", Non-edge pixels: " << nr_non_muchie << std::endl;
    }
    return th;
}

Mat histeresis_thresholding(Mat source, int th) {
    Mat result = Mat::zeros(source.size(), CV_8U);
    float k = 0.4f;
    int prag_inalt = th;
    int prag_coborat = static_cast<int>(k * th);
    for (int y = 0; y < source.rows; y++) {
        for (int x = 0; x < source.cols; x++) {
            uchar value = source.at<uchar>(y, x);
            if (value > prag_inalt) {
                result.at<uchar>(y, x) = 255;
            }
            else if (value > prag_coborat) {
                result.at<uchar>(y, x) = 128;
            }
            else {
                result.at<uchar>(y, x) = 0;
            }
        }
    }
    queue<Point> strong_edges;
    for (int y = 0; y < result.rows; y++) {
        for (int x = 0; x < result.cols; x++) {
            if (result.at<uchar>(y, x) == 255) {
                strong_edges.push(Point(x, y));
                while (!strong_edges.empty()) {
                    Point p = strong_edges.front();
                    strong_edges.pop();
                    for (int k = 0; k < 8; k++) {
                        int ny = p.y + di[k];
                        int nx = p.x + dj[k];
                        if (ny >= 0 && ny < result.rows && nx >= 0 && nx < result.cols &&
                            result.at<uchar>(ny, nx) == 128) {
                            result.at<uchar>(ny, nx) = 255;
                            strong_edges.push(Point(nx, ny));
                        }
                    }
                }
            }
        }
    }
    for (int y = 0; y < result.rows; y++) {
        for (int x = 0; x < result.cols; x++) {
            if (result.at<uchar>(y, x) == 128) {
                result.at<uchar>(y, x) = 0;
            }
        }
    }
    return result;
}

Mat histeresis(Mat source) {
    Mat result = Mat::zeros(source.size(), CV_8U);
    float p = 0.1f;
    int histogram[256] = {0};
    for (int y = 0; y < source.rows; y++) {
        for (int x = 0; x < source.cols; x++) {
            uchar intensity = source.at<uchar>(y, x);
            histogram[intensity]++;
        }
    }
    int total_pixels = source.rows * source.cols;
    int zero_pixels = histogram[0];
    int non_zero_pixels = total_pixels - zero_pixels;
    int nr_non_muchie = static_cast<int>((1.0f - p) * non_zero_pixels);
    int th = 0;
    int sum = 0;
    for (int i = 1; i < 256; i++) {
        sum += histogram[i];
        if (sum > nr_non_muchie) {
            th = i;
            break;
        }
    }
    float k = 0.4f;
    int prag_inalt = th;
    int prag_coborat = static_cast<int>(k * th);
    for (int y = 0; y < source.rows; y++) {
        for (int x = 0; x < source.cols; x++) {
            uchar value = source.at<uchar>(y, x);
            if (value > prag_inalt) {
                result.at<uchar>(y, x) = 255;
            }
            else if (value > prag_coborat) {
                result.at<uchar>(y, x) = 128;
            }
            else {
                result.at<uchar>(y, x) = 0;
            }
        }
    }
    queue<Point> strong_edges;
    for (int y = 0; y < result.rows; y++) {
        for (int x = 0; x < result.cols; x++) {
            if (result.at<uchar>(y, x) == 255) {
                strong_edges.push(Point(x, y));
                while (!strong_edges.empty()) {
                    Point p = strong_edges.front();
                    strong_edges.pop();
                    for (int k = 0; k < 8; k++) {
                        int ny = p.y + di[k];
                        int nx = p.x + dj[k];
                        if (ny >= 0 && ny < result.rows && nx >= 0 && nx < result.cols &&
                            result.at<uchar>(ny, nx) == 128) {
                            result.at<uchar>(ny, nx) = 255;
                            strong_edges.push(Point(nx, ny));
                        }
                    }
                }
            }
        }
    }
    for (int y = 0; y < result.rows; y++) {
        for (int x = 0; x < result.cols; x++) {
            if (result.at<uchar>(y, x) == 128) {
                result.at<uchar>(y, x) = 0;
            }
        }
    }
    return result;
}

Mat apply_Canny(Mat source, int low_threshold, int high_threshold, string filter_type, bool verbose) {
    Mat result;
    if (source.empty() || source.channels() != 1) {
        std::cerr << "Error: Input image must be single-channel grayscale" << std::endl;
        return result;
    }
    int kernel_size = 5;
    Mat smoothed = apply_gaussian_filtering_1D(source, kernel_size);
    filter_structure filter = get_filter(filter_type);
    gradients_structure gradients = compute_gradients(smoothed, filter.filter_x, filter.filter_y, filter.di, filter.dj);
    Mat suppressed = non_maxima_gradient_suppression(gradients);
    Mat normalized = normalize_supression(suppressed, filter_type);
    float p = 0.1f;
    int computed_high_threshold = adaptive_threshold(normalized, p, verbose);
    result = histeresis_thresholding(normalized, computed_high_threshold);
    if (verbose) {
        imshow("Canny Edges", result);
        waitKey(1);
    }
    delete[] filter.filter_x;
    delete[] filter.filter_y;
    return result;
}

vector<line_structure> apply_hough_transform(Mat edges, int threshold, bool verbose) {
    vector<line_structure> detected_lines;
    const double rho_resolution = 1.0;
    const double theta_resolution = CV_PI / 180.0;
    int max_distance = cvRound(sqrt(edges.rows * edges.rows + edges.cols * edges.cols));
    int rho_bins = 2 * max_distance + 1;
    int theta_bins = 180;
    Mat accumulator = Mat::zeros(rho_bins, theta_bins, CV_32S);
    for (int y = 0; y < edges.rows; y++) {
        for (int x = 0; x < edges.cols; x++) {
            if (edges.at<uchar>(y, x) > 0) {
                for (int theta_idx = 0; theta_idx < theta_bins; theta_idx++) {
                    double theta = theta_idx * theta_resolution;
                    double rho = x * cos(theta) + y * sin(theta);
                    int rho_idx = cvRound(rho + max_distance);
                    accumulator.at<int>(rho_idx, theta_idx)++;
                }
            }
        }
    }
    for (int rho_idx = 0; rho_idx < rho_bins; rho_idx++) {
        for (int theta_idx = 0; theta_idx < theta_bins; theta_idx++) {
            int votes = accumulator.at<int>(rho_idx, theta_idx);
            if (votes > threshold) {
                bool is_max = true;
                for (int dr = -1; dr <= 1 && is_max; dr++) {
                    for (int dt = -1; dt <= 1 && is_max; dt++) {
                        int nr = rho_idx + dr;
                        int nt = theta_idx + dt;
                        if (nr >= 0 && nr < rho_bins && nt >= 0 && nt < theta_bins && (dr != 0 || dt != 0)) {
                            if (accumulator.at<int>(nr, nt) > votes) {
                                is_max = false;
                            }
                        }
                    }
                }
                if (is_max) {
                    float rho = (rho_idx - max_distance) * rho_resolution;
                    float theta = theta_idx * theta_resolution;
                    line_structure line;
                    line.rho = rho;
                    line.theta = theta;
                    line.votes = votes;
                    detected_lines.push_back(line);
                }
            }
        }
    }
    sort(detected_lines.begin(), detected_lines.end(),
         [](const line_structure& a, const line_structure& b) {
             return a.votes > b.votes;
         });
    if (verbose) {
        Mat acc_norm;
        normalize(accumulator, acc_norm, 0, 255, NORM_MINMAX, CV_8U);
        Mat acc_color;
        applyColorMap(acc_norm, acc_color, COLORMAP_JET);
    }
    return detected_lines;
}

Mat draw_detected_lines(Mat original, const vector<line_structure>& lines, int max_lines, Scalar color) {
    Mat result;
    if (original.channels() == 1) {
        cvtColor(original, result, COLOR_GRAY2BGR);
    } else {
        original.copyTo(result);
    }
    int num_lines = (max_lines < 0 || max_lines > lines.size()) ? lines.size() : max_lines;
    for (int i = 0; i < num_lines; i++) {
        float rho = lines[i].rho;
        float theta = lines[i].theta;
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));
        line(result, pt1, pt2, color, 2);
    }
    return result;
}