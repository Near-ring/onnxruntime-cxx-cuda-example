//
// Created by semigroup on 10/4/24.
//

#include "transfroms.hpp"

void resize_fast_u8c3(const u8* __restrict input, u8* __restrict output, u32 in_h, u32 in_w, u32 out_h, u32 out_w)
{
    if (!input || !output) {
        throw std::runtime_error("Null Pointer Exception");
    }

    const f32 scale_y = static_cast<f32>(in_h) / out_h;
    const f32 scale_x = static_cast<f32>(in_w) / out_w;

    Vec<u32> y_map(out_h), x_map(out_w);
    for (u32 y = 0; y < out_h; ++y) {
        u32 yy = static_cast<u32>(y * scale_y);
        y_map[y] = std::min(yy, in_h - 1);
    }
    for (u32 x = 0; x < out_w; ++x) {
        u32 xx = static_cast<u32>(x * scale_x);
        x_map[x] = std::min(xx, in_w - 1);
    }

    const size_t in_row_bytes = static_cast<size_t>(in_w) * 3u;
    const size_t out_row_bytes = static_cast<size_t>(out_w) * 3u;

#pragma omp parallel for schedule(static) num_threads(4)
    for (int y = 0; y < static_cast<int>(out_h); ++y) {
        const u8* in_row = input + size_t(y_map[y]) * in_row_bytes;
        u8* out_row = output + size_t(y) * out_row_bytes;

        for (u32 x = 0; x < out_w; ++x) {
            const u8* pix = in_row + size_t(x_map[x]) * 3u;
            // copy R, G, B
            out_row[3 * x] = pix[0];
            out_row[3 * x + 1] = pix[1];
            out_row[3 * x + 2] = pix[2];
        }
    }
}

auto boxes2yolo_str(const Vec<array<f32, 6>>& boxes) -> String
{
    String ret;
    for (auto&& box: boxes) {
        f32 center_x = (box[0] + box[2]) / 2.0f;
        f32 center_y = (box[1] + box[3]) / 2.0f;
        f32 width = box[2] - box[0];
        f32 height = box[3] - box[1];
        ret += std::format("{} {:.7f} {:.7f} {:.7f} {:.7f}\n", box[5], center_x, center_y, width, height);
    }
    return ret;
}

auto preprocess_image_u8c3(const cv::Mat& img, array<i32, 2> target_shape) -> Vec<f32>
{
	if (img.empty()) {
		throw std::runtime_error("Input image is empty");
	}
	if (img.type() != CV_8UC3) {
		throw std::runtime_error("Input image must be of type CV_8UC3");
	}

	u8* p_resize = (u8*)malloc_aligned(64, target_shape[0] * target_shape[1] * 3 * sizeof(u8));

    u8* p_channels[3];
    for (int i = 0; i < 3; i++) {
        p_channels[i] = static_cast<u8*>(malloc_aligned(64, target_shape[0] * target_shape[1] * sizeof(u8)));
    }

    resize_fast_u8c3(img.data, p_resize, img.rows, img.cols, target_shape[0], target_shape[1]);
    cv::hal::split8u(p_resize, p_channels, gsl::narrow_cast<i32>(target_shape[0] * target_shape[1]), 3);

    Vec<f32> data(target_shape[0] * target_shape[1] * 3);
    const u64 len = target_shape[0] * target_shape[1];
    
	// planar to CHW and normalize
    for (u64 i = 0; i < 3; ++i) {
        for (u64 j = 0; j < len; ++j) {
            data[i * len + j] = p_channels[2 - i][j] / 255.0f;
        }
    }

	free_aligned(p_resize);
	for (int i = 0; i < 3; i++) {
		free_aligned(p_channels[i]);
	}
    return data;
}

inline f32 calculate_iou(const array<f32, 4>& box1, const array<f32, 4>& box2)
{
    const f32 x1 = std::max(box1[0], box2[0]);
    const f32 y1 = std::max(box1[1], box2[1]);
    const f32 x2 = std::min(box1[2], box2[2]);
    const f32 y2 = std::min(box1[3], box2[3]);

    const f32 inter_area = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);

    const f32 box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1]);
    const f32 box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1]);

    const f32 union_area = box1_area + box2_area - inter_area;

    return inter_area / union_area;
}

// NMS (Non-Maximum Suppression)
auto apply_nms(const Vec<array<f32, 6>>& detections, const f32 iou_threshold) -> Vec<array<f32, 6>>
{
    if (detections.empty()) return {};
    const u64 d_size = detections.size();
    Vec<array<f32, 4>> boxes(d_size);
    Vec<bool> suppressed(d_size, false);

    Vec<i32> keep(d_size, -1);
    for (u64 i = 0; i < d_size; ++i) {
        boxes[i] = {{detections[i][0], detections[i][1], detections[i][2], detections[i][3]}};
    }

    for (u64 i = 0, k = 0; i < d_size; ++i) {
        if (suppressed[i]) continue;

        keep[k++] = gsl::narrow_cast<i32>(i);

        for (u64 j = i + 1; j < d_size; ++j) {
            if (detections[i][5] == detections[j][5] && calculate_iou(boxes[i], boxes[j]) > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }

    Vec<array<f32, 6>> final_output;
    final_output.reserve(d_size);

    for (const i32 i: keep) {
        if (i == -1) [[unlikely]]
            break;
        final_output.push_back(detections[static_cast<u64>(i)]);
    }

    return final_output;
}

auto yolo11_process_detect_output(Vec<f32>& yolo_detect_output, u8 CLASS_NUM, f32 image_size) -> Vec<array<f32, 6>>
{
    const i32 dim_1 = 4 + CLASS_NUM;
    const i32 dim_0 = gsl::narrow_cast<i32>(yolo_detect_output.size()) / dim_1;

    constexpr f32 confidence_threshold = 0.25; // NOLINT
    constexpr f32 iou_threshold = 0.5f;

    Eigen::Map<Eigen::Matrix<float, -1, -1, Eigen::ColMajor>> mat_yolo(yolo_detect_output.data(), dim_0, dim_1);

    mat_yolo.block(0, 0, dim_0, 4) /= image_size;

    Vec<array<f32, 6>> output;
    output.reserve(dim_0);

    Eigen::VectorXf x1 = mat_yolo.col(0).array() - 0.5f * mat_yolo.col(2).array();
    Eigen::VectorXf x2 = x1.array() + mat_yolo.col(2).array();
    Eigen::VectorXf y1 = mat_yolo.col(1).array() - 0.5f * mat_yolo.col(3).array();
    Eigen::VectorXf y2 = y1.array() + mat_yolo.col(3).array();

    const auto& confidences = mat_yolo.block(0, 4, dim_0, dim_1 - 4);
    for (i32 i = 0; i < dim_0; ++i) {
        i32 max_index = 0;
        const f32 max_conf = confidences.row(i).maxCoeff(&max_index);
        if (max_conf > confidence_threshold) {
            output.emplace_back(std::array<f32, 6>{x1(i), y1(i), x2(i), y2(i), max_conf, static_cast<f32>(max_index)});
        }
    }

    output.shrink_to_fit();

    std::sort(std::execution::par, output.begin(), output.end(),
              [](const array<f32, 6>& a, const array<f32, 6>& b) { return a[4] > b[4]; });

    return apply_nms(output, iou_threshold);
}

auto uniform_color(u32 n) -> Vec<cv::Scalar>
{
    Vec<cv::Scalar> colors;
    for (u32 i = 0; i < n; ++i) {
        const u32 hue = (i * 180 / n);
        const cv::Scalar color = cv::Scalar(hue, 255, 255);
        cv::Mat hsv(1, 1, CV_8UC3, color);
        cv::Mat bgr;
        cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
        colors.emplace_back(bgr.at<cv::Vec3b>(0, 0)[0], bgr.at<cv::Vec3b>(0, 0)[1], bgr.at<cv::Vec3b>(0, 0)[2]);
    }
    return colors;
}

auto render_inference_result(const cv::Mat& img_, const Vec<array<f32, 6>>& boxes, Vec<cv::Scalar> colors) -> cv::Mat
{
    cv::Mat img = img_.clone();
    if (boxes.empty()) return img;

    for (const auto& box : boxes) {
        int class_id = static_cast<int>(box[5]);
        float conf = box[4];

        // Coordinates are normalized, scale to image size
        int x1 = static_cast<int>(box[0] * img.cols);
        int y1 = static_cast<int>(box[1] * img.rows);
        int x2 = static_cast<int>(box[2] * img.cols);
        int y2 = static_cast<int>(box[3] * img.rows);

        // Clamp coordinates to image bounds
        x1 = std::clamp(x1, 0, img.cols - 1);
        y1 = std::clamp(y1, 0, img.rows - 1);
        x2 = std::clamp(x2, 0, img.cols - 1);
        y2 = std::clamp(y2, 0, img.rows - 1);

        cv::Scalar color = colors[class_id % colors.size()];
        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), color, 4); // Thicker box

        // Prepare label: class id and confidence
        std::string label = std::format("{} {:.2f}", class_id, conf);

        int baseLine = 0;
        double font_scale = 0.8; // Larger text
        int text_thickness = 2;  // Thicker text
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, font_scale, text_thickness, &baseLine);
        int top = std::max(y1, label_size.height);

        // Draw filled rectangle for label background
        cv::rectangle(img, cv::Point(x1, top - label_size.height - 2),
            cv::Point(x1 + label_size.width, top + baseLine - 2),
            color, cv::FILLED);

        // Draw label text
        cv::putText(img, label, cv::Point(x1, top - 2),
            cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(0, 0, 0), text_thickness, cv::LINE_AA);
    }
    return img;
}

