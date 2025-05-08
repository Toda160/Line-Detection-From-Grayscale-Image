#include <iostream>
#include <opencv2/opencv.hpp>
#include "project.h"
#include <fstream>
using namespace std;
using namespace cv;

int* compute_histogram_naive(Mat source) {
    // Compute the naive histogram of an image
    int* histogram = (int*)calloc(256, sizeof(int));

    // Iterate through the image pixels and count intensities
    for (int i = 0; i < source.rows; i++) {
        for (int j = 0; j < source.cols; j++) {
            histogram[source.at<uchar>(i, j)]++;
        }
    }

    return histogram;
}

vector<float> compute_kernel_1D(int kernel_size) {
    // Compute a 1D Gaussian kernel of size kernel_size
    vector<float> kernel(kernel_size);
    float sigma = kernel_size / 6.0;
    float sum = 0.0;
    int half_size = kernel_size / 2;

    // Compute Gaussian values
    for (int i = 0; i < kernel_size; i++) {
        int x = i - half_size;
        kernel[i] = exp(-(x * x) / (2 * sigma * sigma));
        sum += kernel[i];
    }

    // Normalize the kernel
    for (int i = 0; i < kernel_size; i++) {
        kernel[i] /= sum;
    }

    return kernel;
}

Mat apply_gaussian_filtering_1D(Mat source, int kernel_size) {
    // Apply 1D Gaussian filtering using separable convolution
    Mat result = source.clone();
    vector<float> kernel = compute_kernel_1D(kernel_size);

    // Temporary matrix for horizontal filtering
    Mat temp(source.size(), CV_32F);

    // Horizontal convolution
    for (int i = 0; i < source.rows; i++) {
        for (int j = 0; j < source.cols; j++) {
            float sum = 0.0;
            for (int k = 0; k < kernel_size; k++) {
                int nj = j + (k - kernel_size / 2);
                if (nj >= 0 && nj < source.cols) {
                    sum += kernel[k] * source.at<uchar>(i, nj);
                }
            }
            temp.at<float>(i, j) = sum;
        }
    }

    // Vertical convolution
    for (int i = 0; i < source.rows; i++) {
        for (int j = 0; j < source.cols; j++) {
            float sum = 0.0;
            for (int k = 0; k < kernel_size; k++) {
                int ni = i + (k - kernel_size / 2);
                if (ni >= 0 && ni < source.rows) {
                    sum += kernel[k] * temp.at<float>(ni, j);
                }
            }
            result.at<uchar>(i, j) = static_cast<uchar>(sum);
        }
    }

    return result;
}

gradients_structure compute_gradients(Mat source, const int* filter_x, const int* filter_y, const int* di, const int* dj) {
    // Compute gradients, magnitude, and direction
    gradients_structure gradients;
    int height = source.rows;
    int width = source.cols;

    gradients.x = Mat(height, width, CV_32S);
    gradients.y = Mat(height, width, CV_32S);
    gradients.magnitude = Mat(height, width, CV_8UC1);
    gradients.direction = Mat(height, width, CV_8UC1);

    // Compute gradients using convolution
    for (int i = 1; i < height - 1; i++) {
        for (int j = 1; j < width - 1; j++) {
            int sum_x = 0, sum_y = 0;
            for (int k = 0; k < 9; k++) {
                int ni = i + di[k];
                int nj = j + dj[k];
                if (ni >= 0 && ni < height && nj >= 0 && nj < width) {
                    sum_x += filter_x[k] * source.at<uchar>(ni, nj);
                    sum_y += filter_y[k] * source.at<uchar>(ni, nj);
                }
            }
            gradients.x.at<int>(i, j) = sum_x;
            gradients.y.at<int>(i, j) = sum_y;

            // Compute magnitude (normalized by 4√2 for Sobel)
            float mag = sqrt(sum_x * sum_x + sum_y * sum_y) / (4 * sqrt(2));
            gradients.magnitude.at<uchar>(i, j) = static_cast<uchar>(min(max(mag, 0.0f), 255.0f));

            // Compute direction
            float dir_rad = atan2(sum_y, sum_x);
            uchar dirn = 0;
            if (fabs(dir_rad) < CV_PI/8 || fabs(dir_rad) > 7*CV_PI/8)
                dirn = 2; // horizontal
            else if ((dir_rad > CV_PI/8 && dir_rad < 3*CV_PI/8) ||
                     (dir_rad < -5*CV_PI/8 && dir_rad > -7*CV_PI/8))
                dirn = 1; // 45 degrees
            else if ((dir_rad > 5*CV_PI/8 && dir_rad < 7*CV_PI/8) ||
                     (dir_rad < -CV_PI/8 && dir_rad > -3*CV_PI/8))
                dirn = 3; // 135 degrees
            else
                dirn = 0; // vertical
            gradients.direction.at<uchar>(i, j) = dirn;
        }
    }

    return gradients;
}

Mat non_maxima_gradient_supression(gradients_structure gradient) {
    // Apply non-maxima suppression to gradient magnitude
    Mat result = Mat::zeros(gradient.magnitude.size(), CV_8UC1);
    int height = gradient.magnitude.rows;
    int width = gradient.magnitude.cols;

    for (int i = 1; i < height - 1; i++) {
        for (int j = 1; j < width - 1; j++) {
            uchar mag = gradient.magnitude.at<uchar>(i, j);
            uchar dir = gradient.direction.at<uchar>(i, j);
            bool is_max = true;

            switch (dir) {
                case 2: // horizontal
                    if (mag < gradient.magnitude.at<uchar>(i, j+1) ||
                        mag < gradient.magnitude.at<uchar>(i, j-1))
                        is_max = false;
                    break;
                case 0: // vertical
                    if (mag < gradient.magnitude.at<uchar>(i-1, j) ||
                        mag < gradient.magnitude.at<uchar>(i+1, j))
                        is_max = false;
                    break;
                case 1: // 45 degrees
                    if (mag < gradient.magnitude.at<uchar>(i-1, j+1) ||
                        mag < gradient.magnitude.at<uchar>(i+1, j-1))
                        is_max = false;
                    break;
                case 3: // 135 degrees
                    if (mag < gradient.magnitude.at<uchar>(i-1, j-1) ||
                        mag < gradient.magnitude.at<uchar>(i+1, j+1))
                        is_max = false;
                    break;
            }
            if (is_max) {
                result.at<uchar>(i, j) = mag;
            }
        }
    }

    return result;
}

filter_structure get_filter(string filter_type) {
    // Return the corresponding filter for the given filter_type
    filter_structure filter;
    filter.filter_x = new int[9];
    filter.filter_y = new int[9];
    filter.di = new int[9];
    filter.dj = new int[9];

    if (filter_type == "Sobel") {
        int fx[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
        int fy[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
        copy(fx, fx + 9, filter.filter_x);
        copy(fy, fy + 9, filter.filter_y);
    } else if (filter_type == "Prewitt") {
        int fx[] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
        int fy[] = {1, 1, 1, 0, 0, 0, -1, -1, -1};
        copy(fx, fx + 9, filter.filter_x);
        copy(fy, fy + 9, filter.filter_y);
    } else { // Default to Sobel
        int fx[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
        int fy[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
        copy(fx, fx + 9, filter.filter_x);
        copy(fy, fy + 9, filter.filter_y);
    }

    // Set di, dj for 3x3 filter
    int di_vals[] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
    int dj_vals[] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
    copy(di_vals, di_vals + 9, filter.di);
    copy(dj_vals, dj_vals + 9, filter.dj);

    return filter;
}

Mat normalize_supression(Mat supression, string filter_type) {
    // Normalize the non-maxima suppression result
    Mat result = supression.clone();
    double max_val;
    minMaxLoc(supression, nullptr, &max_val);
    if (max_val > 0) {
        result.convertTo(result, CV_8U, 255.0 / max_val);
    }
    return result;
}

int adaptive_threshold(Mat magnitude, float p, bool verbose) {
    // Apply adaptive thresholding to the gradient magnitude image
    int histogram[256] = {0};
    for (int i = 0; i < magnitude.rows; i++) {
        for (int j = 0; j < magnitude.cols; j++) {
            histogram[magnitude.at<uchar>(i, j)]++;
        }
    }

    int noNonEdge = (1 - p) * (magnitude.rows * magnitude.cols - histogram[0]);
    int sum = 0;
    int th = 0;
    for (int i = 1; i < 256; i++) {
        sum += histogram[i];
        if (sum > noNonEdge) {
            th = i;
            break;
        }
    }

    if (verbose) {
        cout << "Adaptive threshold: " << th << endl;
    }

    return th;
}

Mat histeresis_thresholding(Mat source, int th) {
    // Apply hysteresis thresholding to the gradient magnitude image
    Mat result = Mat::zeros(source.size(), CV_8UC1);
    int tl = 0.4 * th; // k = 0.4 as specified

    for (int i = 1; i < source.rows - 1; i++) {
        for (int j = 1; j < source.cols - 1; j++) {
            if (source.at<uchar>(i, j) > th)
                result.at<uchar>(i, j) = 255; // strong edge
            else if (source.at<uchar>(i, j) > tl)
                result.at<uchar>(i, j) = 128; // weak edge
            else
                result.at<uchar>(i, j) = 0;   // no edge
        }
    }

    return result;
}

Mat histeresis(Mat source) {
    // Apply edge tracking by hysteresis
    Mat result = source.clone();
    int height = source.rows;
    int width = source.cols;
    vector<int> Qi, Qj;
    Qi.reserve(height * width);
    Qj.reserve(height * width);

    // Collect strong edges
    for (int i = 1; i < height - 1; i++) {
        for (int j = 1; j < width - 1; j++) {
            if (result.at<uchar>(i, j) == 255) {
                Qi.push_back(i);
                Qj.push_back(j);
            }
        }
    }

    // Track edges
    for (size_t idx = 0; idx < Qi.size(); idx++) {
        int i = Qi[idx];
        int j = Qj[idx];

        // Check 8-connected neighbors
        for (int k = 0; k < 8; k++) {
            int ni = i + di[k];
            int nj = j + dj[k];
            if (ni >= 1 && ni < height - 1 && nj >= 1 && nj < width - 1) {
                if (result.at<uchar>(ni, nj) == 128) {
                    result.at<uchar>(ni, nj) = 255;
                    Qi.push_back(ni);
                    Qj.push_back(nj);
                }
            }
        }
    }

    // Clean up weak edges
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (result.at<uchar>(i, j) == 128)
                result.at<uchar>(i, j) = 0;
        }
    }

    return result;
}

Mat apply_Canny(Mat source, int low_threshold, int high_threshold, string filter_type, bool verbose) {
    // Apply the Canny edge detection algorithm
    Mat result;

    // Step 1: Gaussian filtering
    Mat smoothed = apply_gaussian_filtering_1D(source, 5);

    // Step 2: Compute gradients
    filter_structure filter = get_filter(filter_type);
    gradients_structure gradients = compute_gradients(smoothed, filter.filter_x, filter.filter_y, filter.di, filter.dj);

    // Step 3: Non-maxima suppression
    Mat suppressed = non_maxima_gradient_supression(gradients);
    Mat normalized = normalize_supression(suppressed, filter_type);

    // Step 4: Apply adaptive thresholding
    float p = 0.1f;
    int th = adaptive_threshold(normalized, p, verbose);
    if (high_threshold > 0) th = high_threshold; // Override if provided
    if (low_threshold <= 0) low_threshold = 0.4 * th;

    // Step 5: Apply hysteresis thresholding
    Mat thresholded = histeresis_thresholding(normalized, th);

    // Step 6: Apply hysteresis
    result = histeresis(thresholded);

    // Clean up filter
    delete[] filter.filter_x;
    delete[] filter.filter_y;
    delete[] filter.di;
    delete[] filter.dj;

    if (verbose) {
        imshow("a) Original", source);
        imshow("b) After Gaussian", smoothed);
        imshow("c) Gradient Magnitude", gradients.magnitude);
        imshow("d) After Non-maxima Suppression", normalized);
        Mat temp = thresholded.clone();
        threshold(temp, temp, low_threshold, 255, THRESH_BINARY);
        imshow("e) After Adaptive Thresholding", temp);
        imshow("f) Final Result", result);
    }

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
    }
    else {
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