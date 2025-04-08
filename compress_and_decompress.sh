#!/bin/bash

# 定义压缩器路径
compressors=(
    "$HOME/PANS_test/PANS-BASIC/build/cpuans_compress"
    "$HOME/PANS_test/PANS-step1/build/cpuans_compress"
    "$HOME/PANS_test/PANS-step2/build/cpuans_compress"
    "$HOME/PANS_test/PANS-main/build/cpuans_compress"
)

# 定义解压器路径
decompressors=(
    "$HOME/PANS_test/PANS-BASIC/build/cpuans_decompress"
    "$HOME/PANS_test/PANS-step1/build/cpuans_decompress"
    "$HOME/PANS_test/PANS-step2/build/cpuans_decompress"
    "$HOME/PANS_test/PANS-main/build/cpuans_decompress"
)

# 定义输入文件目录
input_dir="$HOME/HWJ/mans/testdata/u2/haac"

# 定义输出目录
output_dir="$HOME/HWJ/mans/testdata/u2/haac_output"

# 定义表格输出文件
table_file="$HOME/HWJ/mans/testdata/u2/haac_output/results.csv"

# 确保输出目录存在
mkdir -p "$output_dir"

# 初始化表格文件
echo "File,Compressor,Compression Time (ms),Compression BW (MB/s),Compression Ratio,Decompression Time (ms),Decompression BW (MB/s),Success" > "$table_file"

# 遍历每个文件
for file in "$input_dir"/*; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        echo "Processing file: $filename"

        # 遍历每个压缩器
        for i in "${!compressors[@]}"; do
            compressor="${compressors[$i]}"
            decompressor="${decompressors[$i]}"
            echo "Using compressor: $compressor"

            # 压缩文件
            echo "Compressing..."
            if "$compressor" "$file" "$output_dir/${filename}.compressed" 2>&1 | tee compress_output.txt; then
                echo "Compression successful"
            else
                echo "Compression failed"
                continue
            fi

            # 解压文件
            echo "Decompressing..."
            if "$decompressor" "$output_dir/${filename}.compressed" "$output_dir/${filename}.decompressed" 2>&1 | tee decompress_output.txt; then
                echo "Decompression successful"
            else
                echo "Decompression failed"
                rm -f "$output_dir/${filename}.compressed" "$output_dir/${filename}.decompressed"
                continue
            fi

            # 提取压缩和解压信息
            comp_time=$(grep "comp   time" compress_output.txt | awk '{print $3}')
            comp_bw=$(grep "B/W" compress_output.txt | awk '{print $4}')
            comp_ratio=$(grep "compress ratio" compress_output.txt | awk '{print $3}')
            decomp_time=$(grep "decomp time" decompress_output.txt | awk '{print $3}')
            decomp_bw=$(grep "B/W" decompress_output.txt | awk '{print $4}')
            success="Yes"

            # 写入表格
            echo "$filename,$(basename "$compressor"),$comp_time,$comp_bw,$comp_ratio,$decomp_time,$decomp_bw,$success" >> "$table_file"

            # 清理临时文件
            rm -f "$output_dir/${filename}.compressed" "$output_dir/${filename}.decompressed" compress_output.txt decompress_output.txt

            # 在使用完一个压缩器后输出一个空行
            echo
        done

        # 在处理完一个文件后输出一个空行
        echo
    fi
done

echo "Results written to $table_file"
