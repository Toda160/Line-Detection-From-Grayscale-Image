#include <iostream>
#include <opencv2/opencv.hpp>
#include "project.h"
#include <fstream>
using namespace std;
using namespace cv;

int* compute_histogram_naive(Mat source){

    // TODO: Compute the naive histogram of an image

    int* histogram = (int*)calloc(256, sizeof(int));

    //*****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    for (int y = 0; y < source.rows; y++) {
        for (int x = 0; x < source.cols; x++) {
            // Get the pixel intensity (assuming single-channel image, uchar type)
            uchar intensity = source.at<uchar>(y, x);
            // Increment the corresponding histogram bin
            histogram[intensity]++;
        }
    }

    //*****END OF YOUR CODE(DO NOT DELETE / MODIFY THIS LINE) *****

    return histogram;

}

vector<float> compute_kernel_1D(int kernel_size){
    // This function should return a Gaussian kernel of size (kernel_size, 1)
    // The std should be kernel_size/6.0
    vector<float> kernel(kernel_size);

    //*****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    // Use sigma = 0.5 as per reference image
    float sigma = 0.5f;
    float sigma2 = sigma * sigma;

    // Center of the kernel
    int center = (kernel_size - 1) / 2;

    // Compute Gaussian values
    float sum = 0.0f;
    for (int i = 0; i < kernel_size; i++) {
        int x = i - center;
        kernel[i] = exp(-(x * x) / (2.0f * sigma2));
        sum += kernel[i];
    }

    // Normalize the kernel
    for (int i = 0; i < kernel_size; i++) {
        kernel[i] /= sum;
    }

    //*****END OF YOUR CODE(DO NOT DELETE / MODIFY THIS LINE) *****

    return kernel;
}


Mat apply_gaussian_filtering_1D(Mat source, int kernel_size){
    // This function should apply a Gaussian filter to the source image using the kernel computed in compute_kernel_1D
    // by applying successive 1D convolutions in the x and y directions.

    Mat result;

    //*****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    // Compute the 1D Gaussian kernel
    vector<float> kernel = compute_kernel_1D(kernel_size);

    // Convert the source image to float for precise convolution
    Mat source_float;
    source.convertTo(source_float, CV_32F);

    // Step 1: Apply 1D Gaussian kernel horizontally (along rows)
    Mat intermediate = Mat::zeros(source.size(), CV_32F);
    int radius = kernel_size / 2;

    // Use OpenCV's filter2D for more efficient convolution
    Mat kernel_mat = Mat(1, kernel_size, CV_32F);
    for(int i = 0; i < kernel_size; i++) {
        kernel_mat.at<float>(0, i) = kernel[i];
    }

    // Apply horizontal convolution
    filter2D(source_float, intermediate, CV_32F, kernel_mat, Point(-1, -1), 0, BORDER_REFLECT);

    // Step 2: Apply 1D Gaussian kernel vertically (along columns)
    result = Mat::zeros(source.size(), CV_32F);
    Mat kernel_mat_t = kernel_mat.t(); // Transpose for vertical convolution
    filter2D(intermediate, result, CV_32F, kernel_mat_t, Point(-1, -1), 0, BORDER_REFLECT);

    // Convert the result back to the source type
    result.convertTo(result, source.type());

    //*****END OF YOUR CODE(DO NOT DELETE / MODIFY THIS LINE) *****

    return result;
}

gradients_structure compute_gradients(Mat source, const int* filter_x, const int* filter_y, const int* di, const int* dj) {
    gradients_structure gradients;

    gradients.x = Mat::zeros(source.size(), CV_32F);
    gradients.y = Mat::zeros(source.size(), CV_32F);
    gradients.magnitude = Mat::zeros(source.size(), CV_32F);
    gradients.direction = Mat::zeros(source.size(), CV_32F);

    // Compute gradients using convolution
    for (int y = 1; y < source.rows - 1; y++) {
        for (int x = 1; x < source.cols - 1; x++) {
            float sum_x = 0.0f, sum_y = 0.0f;

            // Apply 3x3 Sobel operators
            for (int k = 0; k < 8; k++) {
                int ny = y + di[k];
                int nx = x + dj[k];
                float pixel_val = source.at<float>(ny, nx);
                sum_x += pixel_val * filter_x[k];
                sum_y += pixel_val * filter_y[k];
            }

            // Store gradients (invert x and y for correct orientation)
            gradients.x.at<float>(y, x) = -sum_x;  // Invert x gradient
            gradients.y.at<float>(y, x) = -sum_y;  // Invert y gradient

            // Compute magnitude
            gradients.magnitude.at<float>(y, x) = sqrt(sum_x * sum_x + sum_y * sum_y);

            // Compute direction in range [-π, π]
            gradients.direction.at<float>(y, x) = atan2(-sum_y, -sum_x);  // Use inverted gradients
        }
    }

    // Handle border pixels by replication
    for(int y = 0; y < source.rows; y++) {
        for(int x = 0; x < source.cols; x++) {
            if(y == 0 || y == source.rows-1 || x == 0 || x == source.cols-1) {
                int ny = y == 0 ? 1 : (y == source.rows-1 ? source.rows-2 : y);
                int nx = x == 0 ? 1 : (x == source.cols-1 ? source.cols-2 : x);
                gradients.x.at<float>(y, x) = gradients.x.at<float>(ny, nx);
                gradients.y.at<float>(y, x) = gradients.y.at<float>(ny, nx);
                gradients.magnitude.at<float>(y, x) = gradients.magnitude.at<float>(ny, nx);
                gradients.direction.at<float>(y, x) = gradients.direction.at<float>(ny, nx);
            }
        }
    }

    return gradients;
}

Mat non_maxima_gradient_suppression(gradients_structure gradient) {
    Mat result = Mat::zeros(gradient.magnitude.size(), CV_32F);

    // Implementation according to section 11.2.3
    for (int y = 1; y < gradient.magnitude.rows - 1; y++) {
        for (int x = 1; x < gradient.magnitude.cols - 1; x++) {
            float angle = gradient.direction.at<float>(y, x) * 180.0f / CV_PI;
            // Ensure angle is in [0, 180)
            while (angle < 0) angle += 180.0f;
            while (angle >= 180) angle -= 180.0f;

            float mag = gradient.magnitude.at<float>(y, x);
            float mag1 = 0.0f, mag2 = 0.0f;

            // Interpolate between pixels for more accurate edge detection
            if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180)) {
                // Horizontal direction
                mag1 = gradient.magnitude.at<float>(y, x+1);
                mag2 = gradient.magnitude.at<float>(y, x-1);
            }
            else if (angle >= 22.5 && angle < 67.5) {
                // 45 degree direction
                mag1 = gradient.magnitude.at<float>(y-1, x+1);
                mag2 = gradient.magnitude.at<float>(y+1, x-1);
            }
            else if (angle >= 67.5 && angle < 112.5) {
                // Vertical direction
                mag1 = gradient.magnitude.at<float>(y-1, x);
                mag2 = gradient.magnitude.at<float>(y+1, x);
            }
            else {
                // 135 degree direction
                mag1 = gradient.magnitude.at<float>(y-1, x-1);
                mag2 = gradient.magnitude.at<float>(y+1, x+1);
            }

            // Non-maximum suppression check with a small threshold for noise reduction
            if (mag > mag1 && mag > mag2 && mag > 0.01f) {
                result.at<float>(y, x) = mag;
            }
        }
    }

    return result;
}


filter_structure get_filter(string filter_type) {
    // return the corresponding filter for the given filter_type
    filter_structure filter;

    //*****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    // Allocate memory for filter_x and filter_y (8 elements for 8 neighbors)
    filter.filter_x = new int[8];
    filter.filter_y = new int[8];

    // Assign global di and dj arrays
    filter.di = (int*)di; // Global di from header
    filter.dj = (int*)dj; // Global dj from header

    // Populate filters based on filter_type
    if (filter_type == "Sobel") {
        // Sobel kernels (3x3, excluding center)
        // filter_x: [-1, -2, -1, 0, 0, 1, 2, 1] (left-to-right, top-to-bottom, excluding (0,0))
        // filter_y: [1, 2, 1, 0, 0, -1, -2, -1]
        int sobel_x[8] = {-1, -2, -1, 0, 0, 1, 2, 1};
        int sobel_y[8] = {1, 2, 1, 0, 0, -1, -2, -1};
        copy(sobel_x, sobel_x + 8, filter.filter_x);
        copy(sobel_y, sobel_y + 8, filter.filter_y);
    }
    else if (filter_type == "Prewitt") {
        // Prewitt kernels (3x3, excluding center)
        // filter_x: [-1, -1, -1, 0, 0, 1, 1, 1]
        // filter_y: [1, 1, 1, 0, 0, -1, -1, -1]
        int prewitt_x[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
        int prewitt_y[8] = {1, 1, 1, 0, 0, -1, -1, -1};
        copy(prewitt_x, prewitt_x + 8, filter.filter_x);
        copy(prewitt_y, prewitt_y + 8, filter.filter_y);
    }
    else if (filter_type == "Roberts") {
        // Roberts kernels (2x2, adapted to 8 neighbors)
        // Original: filter_x = [1, 0; 0, -1], filter_y = [0, -1; 1, 0]
        // Map to 8 neighbors (di, dj), set unused positions to 0
        // Approximate by placing weights at relevant corners
        int roberts_x[8] = {0, 0, 0, 0, -1, 0, 0, 1}; // Approximate positions
        int roberts_y[8] = {0, 0, -1, 0, 0, 0, 0, 1};
        copy(roberts_x, roberts_x + 8, filter.filter_x);
        copy(roberts_y, roberts_y + 8, filter.filter_y);
    }
    else {
        // Default to Sobel if filter_type is unrecognized
        int sobel_x[8] = {-1, -2, -1, 0, 0, 1, 2, 1};
        int sobel_y[8] = {1, 2, 1, 0, 0, -1, -2, -1};
        copy(sobel_x, sobel_x + 8, filter.filter_x);
        copy(sobel_y, sobel_y + 8, filter.filter_y);
    }

    //*****END OF YOUR CODE(DO NOT DELETE / MODIFY THIS LINE) *****

    return filter;
}

Mat normalize_supression(Mat supression, string filter_type) {
    Mat result;

    // Convert to float for normalization
    Mat float_img;
    supression.convertTo(float_img, CV_32F);

    // Normalize by 4√2 as specified in the lab
    float normalization_factor = 4.0f * sqrt(2.0f);
    float_img /= normalization_factor;

    // Scale to [0, 255] range with improved contrast
    double min_val, max_val;
    minMaxLoc(float_img, &min_val, &max_val);
    if(max_val > 0) {
        float_img = float_img * (255.0 / max_val);
    }

    // Convert back to uchar
    float_img.convertTo(result, CV_8U);

    return result;
}

int adaptive_threshold(Mat magnitude, float p, bool verbose) {
    // Calculate histogram of non-zero pixels
    int histogram[256] = {0};
    int non_zero_count = 0;

    for(int y = 0; y < magnitude.rows; y++) {
        for(int x = 0; x < magnitude.cols; x++) {
            uchar pixel = magnitude.at<uchar>(y, x);
            if(pixel > 0) {
                histogram[pixel]++;
                non_zero_count++;
            }
        }
    }

    // Calculate number of pixels to be considered edges
    // p should be between 0.01 and 0.1 (using 0.1 as specified)
    int nr_edge = static_cast<int>(p * non_zero_count);

    // Find threshold by counting from highest intensity
    int sum = 0;
    int threshold = 255;

    // Start from high intensities and count down
    for(int i = 255; i >= 1; i--) {
        sum += histogram[i];
        if(sum > nr_edge) {
            threshold = i;
            break;
        }
    }

    // Ensure threshold is not too low
    threshold = max(threshold, 20);  // Minimum threshold to avoid noise

    return threshold;
}

Mat histeresis_thresholding(Mat source, int th) {
    Mat result = Mat::zeros(source.size(), CV_8U);

    // Define thresholds
    int high_threshold = th;
    float k = 0.4f; // As specified in the lab
    int low_threshold = static_cast<int>(k * high_threshold);

    // First pass: mark strong and weak edges
    for (int y = 0; y < source.rows; y++) {
        for (int x = 0; x < source.cols; x++) {
            uchar pixel = source.at<uchar>(y, x);
            if (pixel >= high_threshold)
                result.at<uchar>(y, x) = 255; // Strong edge
            else if (pixel >= low_threshold)
                result.at<uchar>(y, x) = 128; // Weak edge
        }
    }

    // Second pass: hysteresis with multiple iterations for better connectivity
    bool changed;
    int max_iterations = 8;  // Limit iterations to avoid infinite loops
    int iteration = 0;

    do {
        changed = false;
        Mat temp = result.clone();

        for (int y = 1; y < result.rows - 1; y++) {
            for (int x = 1; x < result.cols - 1; x++) {
                if (temp.at<uchar>(y, x) == 128) { // Weak edge
                    // Check 8-connected neighbors
                    bool hasStrongNeighbor = false;
                    for (int dy = -1; dy <= 1 && !hasStrongNeighbor; dy++) {
                        for (int dx = -1; dx <= 1 && !hasStrongNeighbor; dx++) {
                            if (dx == 0 && dy == 0) continue;
                            if (temp.at<uchar>(y + dy, x + dx) == 255) {
                                hasStrongNeighbor = true;
                            }
                        }
                    }

                    if (hasStrongNeighbor) {
                        result.at<uchar>(y, x) = 255;
                        changed = true;
                    }
                }
            }
        }
        iteration++;
    } while (changed && iteration < max_iterations);

    // Final pass: remove remaining weak edges
    for (int y = 0; y < result.rows; y++) {
        for (int x = 0; x < result.cols; x++) {
            if (result.at<uchar>(y, x) != 255) {
                result.at<uchar>(y, x) = 0;
            }
        }
    }

    return result;
}

Mat apply_Canny(Mat source, int low_threshold, int high_threshold, string filter_type, bool verbose) {
    Mat result = Mat::zeros(source.size(), CV_8UC1);
    int height = source.rows;
    int width = source.cols;

    // Step 1: Gaussian filtering
    Mat smoothed;
    Mat kernel = getGaussianKernel(5, 0.5);
    sepFilter2D(source, smoothed, CV_8U, kernel, kernel.t(), Point(-1,-1), 0, BORDER_REFLECT);

    // Step 2: Gradient computation
    Mat Gx(height, width, CV_32S);  // gradient along x
    Mat Gy(height, width, CV_32S);  // gradient along y
    Mat G(height, width, CV_8UC1);  // gradient magnitude
    Mat dir(height, width, CV_8UC1); // gradient direction
    G.setTo(0);
    Mat Gmax(height, width, CV_8UC1);  // non maxima suppressed image
    Gmax.setTo(0);

    // Compute intensity gradient using Sobel
    for (int i = 1; i < height - 1; i++) {
        for (int j = 1; j < width - 1; j++) {
            // Sobel masks
            Gx.at<int>(i, j) =
                (smoothed.at<uchar>(i-1, j+1) + 2*smoothed.at<uchar>(i, j+1) + smoothed.at<uchar>(i+1, j+1)) -
                (smoothed.at<uchar>(i-1, j-1) + 2*smoothed.at<uchar>(i, j-1) + smoothed.at<uchar>(i+1, j-1));

            Gy.at<int>(i, j) =
                (smoothed.at<uchar>(i-1, j-1) + 2*smoothed.at<uchar>(i-1, j) + smoothed.at<uchar>(i-1, j+1)) -
                (smoothed.at<uchar>(i+1, j-1) + 2*smoothed.at<uchar>(i+1, j) + smoothed.at<uchar>(i+1, j+1));

            // Gradient magnitude normalized by 4√2
            G.at<uchar>(i, j) = sqrt(Gx.at<int>(i,j) * Gx.at<int>(i,j) +
                                   Gy.at<int>(i,j) * Gy.at<int>(i,j)) / (4 * sqrt(2));

            // Compute direction
            float dir_rad = atan2(Gy.at<int>(i,j), Gx.at<int>(i,j));
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

            dir.at<uchar>(i, j) = dirn;
        }
    }

    // Non-maxima suppression
    for (int i = 1; i < height - 1; i++) {
        for (int j = 1; j < width - 1; j++) {
            switch (dir.at<uchar>(i, j)) {
                case 2: // horizontal
                    if (G.at<uchar>(i,j) < G.at<uchar>(i,j+1) ||
                        G.at<uchar>(i,j) < G.at<uchar>(i,j-1))
                        Gmax.at<uchar>(i,j) = 0;
                    else
                        Gmax.at<uchar>(i,j) = G.at<uchar>(i,j);
                    break;
                case 0: // vertical
                    if (G.at<uchar>(i,j) < G.at<uchar>(i-1,j) ||
                        G.at<uchar>(i,j) < G.at<uchar>(i+1,j))
                        Gmax.at<uchar>(i,j) = 0;
                    else
                        Gmax.at<uchar>(i,j) = G.at<uchar>(i,j);
                    break;
                case 1: // 45 degrees
                    if (G.at<uchar>(i,j) < G.at<uchar>(i-1,j+1) ||
                        G.at<uchar>(i,j) < G.at<uchar>(i+1,j-1))
                        Gmax.at<uchar>(i,j) = 0;
                    else
                        Gmax.at<uchar>(i,j) = G.at<uchar>(i,j);
                    break;
                case 3: // 135 degrees
                    if (G.at<uchar>(i,j) < G.at<uchar>(i-1,j-1) ||
                        G.at<uchar>(i,j) < G.at<uchar>(i+1,j+1))
                        Gmax.at<uchar>(i,j) = 0;
                    else
                        Gmax.at<uchar>(i,j) = G.at<uchar>(i,j);
                    break;
            }
        }
    }

    // Compute adaptive threshold
    int histogram[256] = {0};
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            histogram[Gmax.at<uchar>(i,j)]++;
        }
    }

    float p = 0.1f; // as specified
    int noNonEdge = (1 - p) * (height * width - histogram[0]);

    int sum = 0;
    int TH = 0;
    for(int i = 1; i < 256; i++) {
        sum += histogram[i];
        if(sum > noNonEdge) {
            TH = i;
            break;
        }
    }
    int TL = 0.4f * TH;  // k = 0.4 as specified

    // Initial hysteresis thresholding
    for(int i = 1; i < height-1; i++) {
        for(int j = 1; j < width-1; j++) {
            if(Gmax.at<uchar>(i,j) > TH)
                result.at<uchar>(i,j) = 255;  // strong edge
            else if(Gmax.at<uchar>(i,j) < TL)
                result.at<uchar>(i,j) = 0;    // no edge
            else
                result.at<uchar>(i,j) = 128;  // weak edge
        }
    }

    // Edge tracking by hysteresis
    vector<int> Qi, Qj;
    Qi.reserve(height * width);
    Qj.reserve(height * width);

    // First pass: collect strong edges
    for(int i = 1; i < height-1; i++) {
        for(int j = 1; j < width-1; j++) {
            if(result.at<uchar>(i,j) == 255) {
                Qi.push_back(i);
                Qj.push_back(j);
            }
        }
    }

    // Second pass: track edges
    for(size_t idx = 0; idx < Qi.size(); idx++) {
        int i = Qi[idx];
        int j = Qj[idx];

        // Check 8-connected neighbors
        for(int di = -1; di <= 1; di++) {
            for(int dj = -1; dj <= 1; dj++) {
                if(di == 0 && dj == 0) continue;

                int ni = i + di;
                int nj = j + dj;

                if(ni >= 1 && ni < height-1 && nj >= 1 && nj < width-1) {
                    if(result.at<uchar>(ni,nj) == 128) {
                        result.at<uchar>(ni,nj) = 255;
                        Qi.push_back(ni);
                        Qj.push_back(nj);
                    }
                }
            }
        }
    }

    // Final cleanup of weak edges
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            if(result.at<uchar>(i,j) == 128)
                result.at<uchar>(i,j) = 0;
        }
    }

    return result;
}

vector<line_structure> apply_hough_transform(Mat edges, int threshold, bool verbose) {
    int minLineLength = 200; // Tune as needed
    int numRhos = 400;
    int numThetas = 360;
    int edgeHeight = edges.rows;
    int edgeWidth = edges.cols;
    double edgeHeightHalf = edgeHeight / 2.0;
    double edgeWidthHalf = edgeWidth / 2.0;
    double d = sqrt(pow(edgeHeight, 2) + pow(edgeWidth, 2));
    double dTheta = CV_PI / numThetas; // theta in radians
    double dRho = (2.0 * d) / numRhos;

    std::vector<double> thetas(numThetas);
    std::vector<double> rhos(numRhos);
    for (int i = 0; i < numThetas; i++) {
        thetas[i] = i * dTheta;
    }
    for (int i = 0; i < numRhos; i++) {
        rhos[i] = -d + i * dRho;
    }

    Mat accumulator = Mat::zeros(numRhos, numThetas, CV_32S);
    std::vector<Point> edgePoints;
    findNonZero(edges, edgePoints);

    // Fill accumulator
    for (const Point& edgePoint : edgePoints) {
        double x = edgePoint.x - edgeWidthHalf;
        double y = edgePoint.y - edgeHeightHalf;
        for (int t = 0; t < numThetas; t++) {
            double rhoValue = x * cos(thetas[t]) + y * sin(thetas[t]);
            int rhoIndex = static_cast<int>((rhoValue + d) / dRho);
            if (rhoIndex >= 0 && rhoIndex < numRhos)
                accumulator.at<int>(rhoIndex, t)++;
        }
    }

    // Non-maximum suppression in accumulator
    std::vector<line_structure> detected_lines;
    for (int r = 1; r < numRhos - 1; r++) {
        for (int t = 1; t < numThetas - 1; t++) {
            int count = accumulator.at<int>(r, t);
            if (count > threshold) {
                bool is_local_max = true;
                for (int dr = -1; dr <= 1; dr++) {
                    for (int dt = -1; dt <= 1; dt++) {
                        if (dr == 0 && dt == 0) continue;
                        if (accumulator.at<int>(r + dr, t + dt) > count) {
                            is_local_max = false;
                            break;
                        }
                    }
                    if (!is_local_max) break;
                }
                if (is_local_max) {
                    double rho = rhos[r];
                    double theta = thetas[t];
                    // Calculate endpoints for line length
                    double a = cos(theta);
                    double b = sin(theta);
                    double x0 = a * rho + edgeWidthHalf;
                    double y0 = b * rho + edgeHeightHalf;
                    double x1 = x0 - 1000 * (-b);
                    double y1 = y0 - 1000 * (a);
                    double x2 = x0 + 1000 * (-b);
                    double y2 = y0 + 1000 * (a);
                    double lineLength = sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
                    if (lineLength > minLineLength) {
                        line_structure line = { (float)rho, (float)theta, count };
                        detected_lines.push_back(line);
                    }
                }
            }
        }
    }
    // Merge similar lines (if two lines are close in rho and theta, keep the one with more votes)
    std::vector<line_structure> merged_lines;
    const double rho_thresh = 10; // pixels
    const double theta_thresh = CV_PI / 180; // 1 degree
    for (const auto& l : detected_lines) {
        bool merged = false;
        for (auto& ml : merged_lines) {
            if (fabs(l.rho - ml.rho) < rho_thresh && fabs(l.theta - ml.theta) < theta_thresh) {
                if (l.votes > ml.votes) ml = l;
                merged = true;
                break;
            }
        }
        if (!merged) merged_lines.push_back(l);
    }
    // Sort by votes
    sort(merged_lines.begin(), merged_lines.end(), [](const line_structure& a, const line_structure& b) {
        return a.votes > b.votes;
    });
    if (verbose) {
        Mat hough_space;
        normalize(accumulator, hough_space, 0, 255, NORM_MINMAX);
        hough_space.convertTo(hough_space, CV_8U);
        applyColorMap(hough_space, hough_space, COLORMAP_JET);
        imshow("Hough Space", hough_space);
    }
    return merged_lines;
}

Mat draw_detected_lines(Mat original, const vector<line_structure>& lines, int max_lines, Scalar color) {
    Mat result;
    cvtColor(original, result, COLOR_GRAY2BGR);
    int edgeHeight = original.rows;
    int edgeWidth = original.cols;
    double edgeHeightHalf = edgeHeight / 2.0;
    double edgeWidthHalf = edgeWidth / 2.0;
    int num_lines = (max_lines == -1) ? lines.size() : std::min(max_lines, (int)lines.size());
    for (int i = 0; i < num_lines; i++) {
        const line_structure& detected_line = lines[i];
        double rho = detected_line.rho;
        double theta = detected_line.theta;
        double a = cos(theta);
        double b = sin(theta);
        double x0 = a * rho + edgeWidthHalf;
        double y0 = b * rho + edgeHeightHalf;
        double x1 = x0 - 1000 * (-b);
        double y1 = y0 - 1000 * (a);
        double x2 = x0 + 1000 * (-b);
        double y2 = y0 + 1000 * (a);
        cv::line(result, Point(cvRound(x1), cvRound(y1)), Point(cvRound(x2), cvRound(y2)), color, 2, LINE_AA);
    }
    return result;
}