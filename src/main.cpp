#include <onnxruntime_c_api.h>

#include "types.hpp"
#include "transfroms.hpp"

#define USE_CUDA
constexpr u8 CLASS_NUM = 1; // Set according to your model
constexpr i32 INPUT_W = 640;
constexpr i32 INPUT_H = 640;

int main() {
    namespace fs = std::filesystem;
    const String test_dir = "d:/code/onnx_demo/imgs";
    const String result_dir = "d:/code/onnx_demo/result";

    const OrtApi* ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    OrtEnv* env = nullptr;
    ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "LOG_0", &env);

    OrtSessionOptions* sess_opts = nullptr;
    ort->CreateSessionOptions(&sess_opts);
    ort->SetSessionGraphOptimizationLevel(sess_opts, ORT_ENABLE_EXTENDED);
    OrtStatus* status;
#ifdef USE_CUDA
    OrtCUDAProviderOptions cuda_options;
    memset(&cuda_options, 0, sizeof(cuda_options));
    cuda_options.device_id = 0;
    //cuda_options.arena_extend_strategy = 0;
    //cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
    //cuda_options.do_copy_in_default_stream = 1;
    status = ort->SessionOptionsAppendExecutionProvider_CUDA(sess_opts, &cuda_options);
	if (status) {
		spdlog::warn("Failed to append CUDA execution provider: {}", ort->GetErrorMessage(status));
	}
#endif

    OrtSession* session = nullptr;

#ifdef _WIN32
    const wchar_t* model_path = L"d:/code/onnx_demo/model/circle_model.onnx";
    status = ort->CreateSession(env, model_path, sess_opts, &session);
#else
	const char* model_path = "d:/code/onnx_demo/model/circle_model.onnx";
    status = ort->CreateSession(env, model_path, sess_opts, &session);
#endif

    if (status) {
        spdlog::critical("Failed to create session: {}", ort->GetErrorMessage(status));
        ort->ReleaseStatus(status);
        return -1;
    }

    // Prepare allocator for input/output names
    OrtAllocator* allocator = nullptr;
    ort->GetAllocatorWithDefaultOptions(&allocator);

    // Get input/output names
    char* input_name = nullptr;
    ort->SessionGetInputName(session, 0, allocator, &input_name);
    char* output_name = nullptr;
    ort->SessionGetOutputName(session, 0, allocator, &output_name);

    fs::create_directory(result_dir);
    for (auto& e : fs::directory_iterator(test_dir)) {
        String fn = e.path().string();
        String ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext != ".jpg" && ext != ".jpeg" && ext != ".png") continue;

        cv::Mat img = cv::imread(fn, cv::ImreadModes::IMREAD_COLOR_BGR); //BGR image

        if (img.empty()) {
			spdlog::warn("Cannot read image: {}", fn);
			continue;
        }
        const i32 w = img.cols;
        const i32 h = img.rows;

        // Preprocess image
        auto input_tensor_data = preprocess_image_u8c3(img, {INPUT_H, INPUT_W});

        // Prepare input tensor shape: {1, 3, INPUT_H, INPUT_W}
        std::array<i64, 4> input_shape = {1, 3, INPUT_H, INPUT_W};
        size_t input_tensor_size = 3 * INPUT_H * INPUT_W;

        OrtMemoryInfo* mem_info = nullptr;
        ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info);

        OrtValue* input_tensor = nullptr;
        ort->CreateTensorWithDataAsOrtValue(
            mem_info,
            input_tensor_data.data(),
            input_tensor_size * sizeof(float),
            input_shape.data(),
            input_shape.size(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &input_tensor
        );

        // Prepare input/output names
        const char* input_names[] = { input_name };
        const char* output_names[] = { output_name };

        // Run inference
        OrtValue* output_tensor = nullptr;
        ort->Run(
            session, nullptr,
            input_names, &input_tensor, 1,
            output_names, 1, &output_tensor
        );

        // Get output tensor data
        float* output_data = nullptr;
        ort->GetTensorMutableData(output_tensor, (void**)&output_data);

        // Get output tensor shape
        OrtTensorTypeAndShapeInfo* output_info = nullptr;
        ort->GetTensorTypeAndShape(output_tensor, &output_info);
        size_t output_count = 0;
        ort->GetTensorShapeElementCount(output_info, &output_count);

        // Copy output to Vec<f32>
		f32* iter_first = output_data;
		f32* iter_last = output_data + output_count;
        Vec<f32> yolo_output(iter_first, iter_last);

        // Postprocess: get boxes
        f32 image_size = static_cast<f32>(INPUT_W); // Assuming square input
        auto boxes = yolo11_process_detect_output(yolo_output, CLASS_NUM, image_size);

        // Render results
		auto colors = uniform_color(CLASS_NUM);
        cv::Mat vis = render_inference_result(img, boxes, colors);

        // Save result
        String out_path = result_dir + "/"  + e.path().filename().string();
        cv::imwrite(out_path, vis);

        // Release resources
        ort->ReleaseValue(output_tensor);
        ort->ReleaseValue(input_tensor);
        ort->ReleaseTensorTypeAndShapeInfo(output_info);
        ort->ReleaseMemoryInfo(mem_info);
    }

    // Cleanup
    ort->ReleaseSession(session);
    ort->ReleaseSessionOptions(sess_opts);
    ort->ReleaseEnv(env);
    allocator->Free(allocator, input_name);
    allocator->Free(allocator, output_name);

    return 0;
}