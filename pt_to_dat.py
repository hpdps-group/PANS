import torch
import numpy as np
import argparse

def pt_to_bin(input_path, output_path):
    try:
        # 加载.pt文件
        tensor = torch.load(input_path, map_location=torch.device('cpu'))
        
        # 验证是否为Tensor
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("Input file is not a PyTorch tensor")
        
        # 检查是否需要梯度
        if tensor.requires_grad:
            tensor = tensor.detach()  # 分离张量，避免梯度计算
        
        # 处理不同数据类型
        dtype = tensor.dtype
        print(f"Detected tensor dtype: {dtype}")
        
        # 转换为numpy数组并处理二进制
        np_tensor = tensor.numpy()
        
        # 根据数据类型选择转换方式
        if dtype == torch.bfloat16:
            # BFloat16转换为uint16
            uint_arr = np_tensor.view(np.uint16)
            item_size = 2
        elif dtype == torch.float16:
            # FP16转换为uint16
            uint_arr = np_tensor.view(np.uint16)
            item_size = 2
        elif dtype == torch.float32:
            # FP32转换为uint32
            uint_arr = np_tensor.view(np.uint32)
            item_size = 4
        else:
            raise ValueError(f"Unsupported data type: {dtype}")
        
        # 写入二进制文件
        with open(output_path, 'wb') as f:
            uint_arr.tofile(f)
        
        print(f"Successfully saved {uint_arr.size} elements ({item_size} bytes each) to {output_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert .pt tensor to raw binary .dat')
    parser.add_argument('input', help='Input .pt file path')
    parser.add_argument('output', help='Output .dat file path')
    args = parser.parse_args()
    
    pt_to_bin(args.input, args.output)
    # import torch

    # 加载.pt文件
    tensor = torch.load('tensor_rank_0.pt', map_location=torch.device('cpu'))

    # 确保张量是fp16类型
    if tensor.dtype != torch.float16:
        tensor = tensor.to(torch.float16)

    # 获取第一个元素的二进制表示
    first_element = tensor.view(torch.int16)[0][0][0]
    binary_representation = bin(first_element.item())[2:].zfill(16)  # 转换为16位二进制字符串

    print(f"First fp16 element in tensor_rank_0.pt: {tensor[0][0][0]}")
    print(f"Binary representation: {binary_representation}")

    with open('tensor_rank_0.dat', 'rb') as f:
        first_byte = f.read(2)  # 读取2个字节，因为fp16是16位
        first_dat_element = int.from_bytes(first_byte, byteorder='little')  # 假设小端字节序
        binary_representation_dat = bin(first_dat_element)[2:].zfill(16)  # 转换为16位二进制字符串

    print(f"First element in tensor_rank_0.dat: {first_dat_element}")
    print(f"Binary representation: {binary_representation_dat}")