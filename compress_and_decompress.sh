#!/bin/bash

compressors=(
    "$HOME/PANS_test/PANS-BASIC/build/cpuans_compress"
    "$HOME/PANS_test/PANS-step1/build/cpuans_compress"
    "$HOME/PANS_test/PANS-step2/build/cpuans_compress"
    "$HOME/PANS_test/PANS-main/build/cpuans_compress"
)

decompressors=(
    "$HOME/PANS_test/PANS-BASIC/build/cpuans_decompress"
    "$HOME/PANS_test/PANS-step1/build/cpuans_decompress"
    "$HOME/PANS_test/PANS-step2/build/cpuans_decompress"
    "$HOME/PANS_test/PANS-main/build/cpuans_decompress"
)

input_dir="$HOME/HWJ/mans/testdata/u2/haac"

output_dir="$HOME/HWJ/mans/testdata/u2/haac_output"

mkdir -p "$output_dir"

for file in "$input_dir"/*; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        echo "Processing file: $filename"

        for i in "${!compressors[@]}"; do
            compressor="${compressors[$i]}"
            decompressor="${decompressors[$i]}"
            echo "Using compressor: $compressor"

            echo "Compressing..."
            if "$compressor" "$file" "$output_dir/${filename}.compressed"; then
                echo "Compression successful"
            else
                echo "Compression failed"
                continue
            fi

            echo "Decompressing..."
            if "$decompressor" "$output_dir/${filename}.compressed" "$output_dir/${filename}.decompressed"; then
                echo "Decompression successful"
            else
                echo "Decompression failed"
                rm -f "$output_dir/${filename}.compressed" "$output_dir/${filename}.decompressed"
                continue
            fi

            echo "Comparing..."
            if diff -q "$file" "$output_dir/${filename}.decompressed" >/dev/null; then
                echo "Compression and decompression successful for $filename using $compressor"
            else
                echo "Compression and decompression failed for $filename using $compressor"
                rm -f "$output_dir/${filename}.compressed" "$output_dir/${filename}.decompressed"
                continue
            fi

            rm -f "$output_dir/${filename}.compressed" "$output_dir/${filename}.decompressed"

            echo
        done

        echo
    fi
done
