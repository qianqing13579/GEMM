nvcc -std=c++14 \
    -I=./,${CUDA_PATH}/include/,${OpenCV_PATH}/include/ \
    -L=${CUDA_PATH}/lib64/,${OpenCV_PATH}/lib/ \
    -l=opencv_core,opencv_imgproc,opencv_imgcodecs,opencv_dnn,cublas \
    GEMM.cu
./a.out