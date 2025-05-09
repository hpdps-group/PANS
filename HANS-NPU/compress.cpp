#include <torch/script.h>
#include <torch_npu/aten/npu_native_functions.h>  // NPU 扩展头文件
#include <acl/acl.h>

int main() {
    // 初始化昇腾 NPU 环境
    aclInit(nullptr);
    int32_t deviceId = 0;
    aclrtSetDevice(deviceId);

    // 加载模型
    torch::Device device(torch::kNPU, deviceId);  // 指定 NPU 设备
    torch::jit::script::Module module;
    try {
        module = torch::jit::load("model_npu.pt");
        module.to(device);  // 将模型移动到 NPU
    } catch (const c10::Error& e) {
        std::cerr << "加载模型失败: " << e.what() << std::endl;
        return -1;
    }

    // 准备输入数据（需在 NPU 上分配）
    torch::Tensor input = torch::ones({1, 10}).to(device);
    
    // 执行推理
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);
    torch::Tensor output = module.forward(inputs).toTensor();
    
    std::cout << "输出结果:\n" << output << std::endl;

    // 清理资源
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}