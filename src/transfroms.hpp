//
// Created by semigroup on 10/4/24.
//

#pragma once
#include "types.hpp"
#include "utils.hpp"

auto yolo11_process_detect_output(Vec<f32>& yolo_detect_output, u8 CLASS_NUM = 1, f32 image_size = 320) -> Vec<array<f32, 6>>;

auto render_inference_result(const cv::Mat& img_, const Vec<array<f32, 6>>& boxes, Vec<cv::Scalar> colors) -> cv::Mat;

auto preprocess_image_u8c3(const cv::Mat& img, array<i32, 2> target_shape) -> Vec<f32>;

void resize_fast_u8c3(const u8* __restrict input, u8* __restrict output, u32 in_h, u32 in_w, u32 out_h, u32 out_w);

[[nodiscard]] auto uniform_color(u32 n) -> Vec<cv::Scalar>;
[[nodiscard]] auto boxes2yolo_str(const Vec<array<f32, 6>>& boxes) -> String;
