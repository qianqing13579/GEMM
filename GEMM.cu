/* GEMM优化示例

C=αA∗B+βC

示例程序中：alpha=1,beta=0
*/

#include <sys/time.h>
#include <cuda.h>
#include <cublas.h>
#include <opencv2/opencv.hpp>

using namespace cv;

// 计时
static double seconds()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

/* 分块参数设置
*/
#define BM 128 // block子块大小
#define BN 128
#define BK 8
#define TM 8 // thread子块大小
#define TN 8

/* 矩阵类

Matrix是一个view（类似于Pytorch中的Tensor类型），与原矩阵共享数据，作为原矩阵的一个视图，主要用来访问矩阵数据
*/
template<typename T>
class Matrix
{
public:
    __device__ __host__ Matrix() = default;
    __device__ __host__ Matrix(const Matrix &) = default;
    __device__ __host__ Matrix& operator=(const Matrix &) = default;
    __device__ __host__ Matrix(T *_data,int _rows,int _cols,int _strideOfRow,int _strideOfCol):
                                    data(_data),
                                    rows(_rows),
                                    cols(_cols),
                                    strideOfRow(_strideOfRow),
                                    strideOfCol(_strideOfCol){}

    // 返回该矩阵所有字节数
    constexpr __device__ __host__ int GetNumberOfBytes() const
    {
        return rows*cols*sizeof(T);
    }

    // 返回该矩阵元素个数
    constexpr __device__ __host__ int GetNumberOfElements() const
    {
        return rows*cols;
    }

    // 访问某个元素，该元素的索引为二维逻辑索引：(rowIndex,colIndex)
    __device__ __host__ float &operator()(int rowIndex,int colIndex)
    {
        // 计算内存索引
        int memoryIndex=rowIndex*strideOfRow+colIndex*strideOfCol;

        return data[memoryIndex];
    }

    // 访问某个元素，该元素的索引为一维逻辑索引：(Index)
    __device__ __host__ float &operator()(int index)
    {
        // 转换为二维逻辑索引
        int colIndex=index%cols;
        int rowIndex=index/cols;

        // 计算内存索引
        int memoryIndex=rowIndex*strideOfRow+colIndex*strideOfCol;

        return data[memoryIndex];
    }



public:
    T *data = nullptr;// 数据指针
    int rows = 0;// 矩阵的行数
    int cols = 0;// 矩阵的列数
    int strideOfRow = 0;// 行步长
    int strideOfCol = 0;// 列步长

};

/*  NaiveGEMM
*/
__global__ void NaiveGEMM(Matrix<float> A,Matrix<float> B,Matrix<float> C) // 注意，这里传参的时候不能传引用，而是要传值
{
    
    // 获取线程在网格内的索引
    int row = blockIdx.y * blockDim.y + threadIdx.y;// 定位到行
    int col = blockIdx.x * blockDim.x + threadIdx.x;// 定位到列

    // 每个线程计算矩阵C的一个元素
    if(row<C.rows&&col<C.cols)
    {
        float c = 0;
        for (int i = 0; i < A.cols; ++i)
        {
            c += A(row,i)*B(i,col);// 使用A的第row行乘以B的第col列
        }
        C(row,col) = c;
    }
    
}

/* BlockGEMM_V1
*/
__global__ void BlockGEMM_V1(Matrix<float> A,Matrix<float> B,Matrix<float> C)
{
    // 注意命名不要与前面的宏定义重名
    const int BLOCK_M=16;// block的行数
    const int BLOCK_N=16;// block的列数
    const int BLOCK_K=16;

    // 沿着K维度循环加载一个block中对应的A和B的数据到共享内存
    float c=0.0;
    for(int i=0;i<A.cols/BLOCK_K;++i)
    {
        // 每个block对应的全局内存中的A,B子块，即创建全局内存中A,B的view
        Matrix<float> ASub(A.data+blockIdx.y*BLOCK_M*A.strideOfRow+i*BLOCK_K,BLOCK_M,BLOCK_K,A.strideOfRow,A.strideOfCol);
        Matrix<float> BSub(B.data+i*BLOCK_K*B.strideOfRow+blockIdx.x*BLOCK_N,BLOCK_K,BLOCK_N,B.strideOfRow,B.strideOfCol);

        // 将Asub,BSub加载到共享内存
        // 注意：这里需要将一维逻辑索引转换为多维逻辑索引：stardIndex->(stardIndex/cols,stardIndex%cols)
        __shared__ float A_Shared[BLOCK_M][BLOCK_K];
        __shared__ float B_Shared[BLOCK_K][BLOCK_N];
        int numberOfElementsPerThread=(BLOCK_K*BLOCK_M)/(blockDim.x*blockDim.y);// 每个线程需要读取多少数据
        int stardIndex=numberOfElementsPerThread*(threadIdx.y*blockDim.x+threadIdx.x);// stardIndex为每个线程读取的起始索引
        for(int threadIndex=0;threadIndex<numberOfElementsPerThread;++threadIndex)
        {
            int logicalIndex=stardIndex+threadIndex;
            A_Shared[logicalIndex/BLOCK_K][logicalIndex%BLOCK_K]=ASub(logicalIndex/BLOCK_K,logicalIndex%BLOCK_K);
            B_Shared[logicalIndex/BLOCK_N][logicalIndex%BLOCK_N]=BSub(logicalIndex/BLOCK_N,logicalIndex%BLOCK_N);
        }
        __syncthreads();

        // 每个thread计算A的一行和B的一列
        for(int k=0;k<BLOCK_K;++k)
        {
            c+=A_Shared[threadIdx.y][k]*B_Shared[k][threadIdx.x];
        }
        __syncthreads();

    }

    // 将每个线程计算好的结果写回到C矩阵
    // CSub为每个线程对应的全局内存的C矩阵子块，创建C矩阵的view
    Matrix<float> CSub(C.data+(blockIdx.y*BLOCK_M*C.strideOfRow+blockIdx.x*BLOCK_N),BLOCK_M,BLOCK_N,C.strideOfRow,C.strideOfCol);
    CSub(threadIdx.y,threadIdx.x)=c;

}

/* BlockGEMM_V2
*/
__global__ void BlockGEMM_V2(Matrix<float> A,Matrix<float> B,Matrix<float> C)
{
    // 每个线程的计算结果
    float c[TM][TN]={0.0};
    float a[TM]={0.0};
    float b[TN]={0.0};

    // 沿着K维度循环加载一个block中对应的A和B的数据到共享内存
    for(int i=0;i<A.cols/BK;++i)
    {
        // 每个block对应的全局内存中的A,B子块，即创建全局内存中A,B的view
        Matrix<float> ASub(A.data+blockIdx.y*BM*A.strideOfRow+i*BK,BM,BK,A.strideOfRow,A.strideOfCol);
        Matrix<float> BSub(B.data+i*BK*B.strideOfRow+blockIdx.x*BN,BK,BN,B.strideOfRow,B.strideOfCol);

        // 将Asub,BSub加载到共享内存
        // 以block为128，thread为8为例：由于一个block有16x16=256个线程，而ASub和BSub中一共有1024个元素，所以每个线程加载4个元素
        // 注意：这里需要将一维逻辑索引转换为多维逻辑索引：stardIndex->(stardIndex/cols,stardIndex%cols)
        __shared__ float A_Shared[BM][BK];
        __shared__ float B_Shared[BK][BN];
        int numberOfElementsPerThread=(BK*BM)/(blockDim.x*blockDim.y);// 每个线程需要读取多少数据
        int stardIndex=numberOfElementsPerThread*(threadIdx.y*blockDim.x+threadIdx.x);// stardIndex为每个线程读取的起始索引
        for(int threadIndex=0;threadIndex<numberOfElementsPerThread;++threadIndex)
        {
            int logicalIndex=stardIndex+threadIndex;
            A_Shared[logicalIndex/BK][logicalIndex%BK]=ASub(logicalIndex/BK,logicalIndex%BK);
            B_Shared[logicalIndex/BN][logicalIndex%BN]=BSub(logicalIndex/BN,logicalIndex%BN);
        }
        __syncthreads();

        // 每个thread对应的共享内存中的A_Shared,B_Shared的子块，即创建A_Shared,B_Shared的view
        Matrix<float> ASub_Shared((float *)A_Shared+threadIdx.y*TM*BK,TM,BK,BK,1);// 每个线程对应的共享内存中A和B的子块
        Matrix<float> BSub_Shared((float *)B_Shared+threadIdx.x*TN,BK,TN,BN,1);

        // 每个线程执行计算
        for(int k=0;k<BK;++k)
        {
            // 先将A的一列和B的一行加载到寄存器
            for(int m=0;m<TM;++m)
            {
                a[m]=ASub_Shared(m,k);
            }
            for(int n=0;n<TN;++n)
            {
                b[n]=BSub_Shared(k,n);
            }

            // 使用寄存器计算
            for(int m=0;m<TM;++m)
            {
                for(int n=0;n<TN;++n)
                {
                    c[m][n]+=a[m]*b[n];
                }
            }
        }
        __syncthreads();

    }

    // 将每个线程计算好的结果写回到C矩阵
    // CSub为每个线程对应的全局内存的C矩阵子块，创建C矩阵的view
    Matrix<float> CSub(C.data+((blockIdx.y*BM+threadIdx.y*TM)*C.strideOfRow+blockIdx.x*BN+threadIdx.x*TN),TM,TN,C.strideOfRow,C.strideOfCol);
    for(int m=0;m<TM;++m)
    {
        for(int n=0;n<TN;++n)
        {
            CSub(m,n)=c[m][n];
        }
    }

}

/* BlockGEMM_V3
*/
__global__ void BlockGEMM_V3(Matrix<float> A,Matrix<float> B,Matrix<float> C)
{
    // 每个线程的计算结果
    float c[TM][TN]={0.0};
    float a[TM]={0.0};
    float b[TN]={0.0};

    // 此时需要的共享内存是原来的2倍
    // 注意：读取和写入的时候第一个维度的索引是交错进行的
    __shared__ float A_Shared[2][BM][BK];
    __shared__ float B_Shared[2][BK][BN];

    // 预取(先读取第一个BK)
    Matrix<float> ASub(A.data+blockIdx.y*BM*A.strideOfRow+0*BK,BM,BK,A.strideOfRow,A.strideOfCol);
    Matrix<float> BSub(B.data+0*BK*B.strideOfRow+blockIdx.x*BN,BK,BN,B.strideOfRow,B.strideOfCol);
    int numberOfElementsPerThread=(BK*BM)/(blockDim.x*blockDim.y);
    int stardIndex=numberOfElementsPerThread*(threadIdx.y*blockDim.x+threadIdx.x);// stardIndex为每个线程读取的起始索引
    for(int threadIndex=0;threadIndex<numberOfElementsPerThread;++threadIndex)
    {
        int logicalIndex=stardIndex+threadIndex;
        A_Shared[0][logicalIndex/BK][logicalIndex%BK]=ASub(logicalIndex/BK,logicalIndex%BK);
        B_Shared[0][logicalIndex/BN][logicalIndex%BN]=BSub(logicalIndex/BN,logicalIndex%BN);
    }
    __syncthreads();

    // 沿着K维度循环加载剩下的数据
    int indexOfRead,indexOfWrite;
    bool indexFlag=false;// 辅助变量，用来计算索引
    for(int i=1;i<A.cols/BK;++i)
    {
        // 计算索引，indexOfRead和indexOfWrite每次循环会交替变换，i=1时为indexOfRead=0,indexOfWrite=1，i=2时为indexOfRead=1,indexOfWrite=0
        indexOfRead = (int)indexFlag; // 读索引，即本次循环读取A_Shared[indexOfRead,:,:]和B_Shared[indexOfRead,:,:]中的数据执行计算
        indexOfWrite = 1-indexOfRead; // 写索引，即预取下一次计算需要的数据到A_Shared[indexOfWrite,:,:]和B_Shared[indexOfWrite,:,:]中

        // 每个线程执行计算
        Matrix<float> ASub_Shared(((float *)A_Shared+indexOfRead*BM*BK)+threadIdx.y*TM*BK,TM,BK,BK,1);// 每个线程对应的共享内存中A和B的子块
        Matrix<float> BSub_Shared(((float *)B_Shared+indexOfRead*BK*BN)+threadIdx.x*TN,BK,TN,BN,1);
        for(int k=0;k<BK;++k)
        {
            // 先将A的一列和B的一行加载到寄存器
            for(int m=0;m<TM;++m)
            {
                a[m]=ASub_Shared(m,k);
            }
            for(int n=0;n<TN;++n)
            {
                b[n]=BSub_Shared(k,n);
            }

            // 使用寄存器计算
            for(int m=0;m<TM;++m)
            {
                for(int n=0;n<TN;++n)
                {
                    c[m][n]+=a[m]*b[n];
                }
            }
        }

        // 预取下个循环的数据
        Matrix<float> ASub(A.data+blockIdx.y*BM*A.strideOfRow+i*BK,BM,BK,A.strideOfRow,A.strideOfCol);
        Matrix<float> BSub(B.data+i*BK*B.strideOfRow+blockIdx.x*BN,BK,BN,B.strideOfRow,B.strideOfCol);
        int numberOfElementsPerThread=(BK*BM)/(blockDim.x*blockDim.y);
        int stardIndex=numberOfElementsPerThread*(threadIdx.y*blockDim.x+threadIdx.x);// stardIndex为每个线程读取的起始索引
        for(int threadIndex=0;threadIndex<numberOfElementsPerThread;++threadIndex)
        {
            int logicalIndex=stardIndex+threadIndex;
            A_Shared[indexOfWrite][logicalIndex/BK][logicalIndex%BK]=ASub(logicalIndex/BK,logicalIndex%BK);
            B_Shared[indexOfWrite][logicalIndex/BN][logicalIndex%BN]=BSub(logicalIndex/BN,logicalIndex%BN);
        }
        __syncthreads();

        // 设置flag
        indexFlag=!indexFlag;
    }

    // 计算最后一个BK
    {
        Matrix<float> ASub_Shared(((float *)A_Shared+indexOfWrite*BM*BK)+threadIdx.y*TM*BK,TM,BK,BK,1);// 每个线程对应的共享内存中A和B的子块
        Matrix<float> BSub_Shared(((float *)B_Shared+indexOfWrite*BK*BN)+threadIdx.x*TN,BK,TN,BN,1);
        for(int k=0;k<BK;++k)
        {
            // 先将A的一列和B的一行加载到寄存器
            for(int m=0;m<TM;++m)
            {
                a[m]=ASub_Shared(m,k);
            }
            for(int n=0;n<TN;++n)
            {
                b[n]=BSub_Shared(k,n);
            }

            // 使用寄存器计算
            for(int m=0;m<TM;++m)
            {
                for(int n=0;n<TN;++n)
                {
                    c[m][n]+=a[m]*b[n];
                }
            }
        }
    }

    // 将每个线程计算好的结果写回到C矩阵
    // CSub为每个线程对应的全局内存的C矩阵子块，创建C矩阵的view
    Matrix<float> CSub(C.data+((blockIdx.y*BM+threadIdx.y*TM)*C.strideOfRow+blockIdx.x*BN+threadIdx.x*TN),TM,TN,C.strideOfRow,C.strideOfCol);
    for(int m=0;m<TM;++m)
    {
        for(int n=0;n<TN;++n)
        {
            CSub(m,n)=c[m][n];
        }
    }

}

int main(int argc,char *argv[])
{
    // 创建CPU A矩阵，这里使用OpenCV读取一张图像作为A矩阵
    cv::Mat A_Host=cv::imread("Test.jpg",0); // 读取为单通道灰度图
    cv::resize(A_Host,A_Host,cv::Size(512,512)); 
    cv::Canny(A_Host,A_Host,50,100,3,false); // 转为二值图，控制值范围
    A_Host.convertTo(A_Host,CV_32FC1); // 转换为FP32类型
    printf("A size:%d x %d\n",A_Host.cols,A_Host.rows);

    // 创建CPU B矩阵
    cv::Mat B_Host=A_Host.clone();

    // 计算CPU C矩阵(使用OpenCV计算矩阵乘)
    double time1,time2;
    cv::Mat C_Host;
    time1=seconds();
    C_Host=A_Host*B_Host;
    time2=seconds();
    printf("cpu elapsed:%f ms\n",(time2-time1)*1000);

    // 创建GPU A矩阵
    float *dataOfA_Device=nullptr;
    cudaMalloc((void **)&dataOfA_Device, A_Host.rows*A_Host.cols*sizeof(float));
    cudaMemcpy(dataOfA_Device, A_Host.data, A_Host.rows*A_Host.cols*sizeof(float), cudaMemcpyHostToDevice);
    Matrix<float> A_Device(dataOfA_Device,A_Host.rows,A_Host.cols,A_Host.cols,1);

    // 创建GPU B矩阵
    float *dataOfB_Device=nullptr;
    cudaMalloc((void **)&dataOfB_Device, B_Host.rows*B_Host.cols*sizeof(float));
    cudaMemcpy(dataOfB_Device, B_Host.data, B_Host.rows*B_Host.cols*sizeof(float), cudaMemcpyHostToDevice);
    Matrix<float> B_Device(dataOfB_Device,B_Host.rows,B_Host.cols,B_Host.cols,1);

    // 创建GPU C矩阵
    float *dataOfC_Device=nullptr;
    cudaMalloc((void **)&dataOfC_Device, A_Host.rows*B_Host.cols*sizeof(float));
    Matrix<float> C_Device(dataOfC_Device,A_Host.rows,B_Host.cols,B_Host.cols,1);
    
    ////////////////////////////// NaiveGEMM /////////////////////////////////////////////
    {
        int BLOCKX = 16;// 每个block的x方向线程数
        int BLOCKY = 16;// 每个block的y方向线程数
        dim3 block(BLOCKX,BLOCKY);
        dim3 grid((C_Device.cols+BLOCKX-1) / BLOCKX,(C_Device.rows+BLOCKY-1) / BLOCKY);
        for(int i=0;i<10;++i)
        {
            time1=seconds();
            NaiveGEMM<<<grid, block>>>(A_Device,B_Device,C_Device);
            cudaDeviceSynchronize();
            time2=seconds();
            printf("NaiveGEMM elapsed:%f ms\n",(time2-time1)*1000);
        }
        
    }

    //////////////////////////////// BlockGEMM_V1 /////////////////////////////////////////////
    {
        int BLOCKX = 16;// 每个block的x方向线程数
        int BLOCKY = 16;// 每个block的y方向线程数
        dim3 block(BLOCKX,BLOCKY);
        dim3 grid(C_Device.cols/BLOCKX,C_Device.rows/BLOCKY);
        for(int i=0;i<10;++i)
        {
            time1=seconds();
            BlockGEMM_V1<<<grid, block>>>(A_Device,B_Device,C_Device);
            cudaDeviceSynchronize();
            time2=seconds();
            printf("BlockGEMM_V1 elapsed:%f ms\n",(time2-time1)*1000);
        }

    }

    //////////////////////////////// BlockGEMM_V2 /////////////////////////////////////////////
    {
        int BLOCKX = 16;// 每个block的x方向线程数
        int BLOCKY = 16;// 每个block的y方向线程数
        dim3 block(BLOCKX,BLOCKY);
        dim3 grid(C_Device.cols/BN,C_Device.rows/BM);
        for(int i=0;i<10;++i)
        {
            time1=seconds();
            BlockGEMM_V2<<<grid, block>>>(A_Device,B_Device,C_Device);
            cudaDeviceSynchronize();
            time2=seconds();
            printf("BlockGEMM_V2 elapsed:%f ms\n",(time2-time1)*1000);
        }

    }

    //////////////////////////////// BlockGEMM_V3 /////////////////////////////////////////////
    {
        int BLOCKX = 16;// 每个block的x方向线程数
        int BLOCKY = 16;// 每个block的y方向线程数
        dim3 block(BLOCKX,BLOCKY);
        dim3 grid(C_Device.cols/BN,C_Device.rows/BM);
        for(int i=0;i<10;++i)
        {
            time1=seconds();
            BlockGEMM_V3<<<grid, block>>>(A_Device,B_Device,C_Device);
            cudaDeviceSynchronize();
            time2=seconds();
            printf("BlockGEMM_V3 elapsed:%f ms\n",(time2-time1)*1000);
        }

    }

    //////////////////////////////// cublas /////////////////////////////////////////////
    {
        cublasHandle_t handle;
        cublasCreate_v2(&handle);
	
	    cublasOperation_t  transA;
        cublasOperation_t  transB;
	
	    transA=CUBLAS_OP_N;
	    transB=CUBLAS_OP_N;// CUBLAS_OP_N,CUBLAS_OP_T

        float alpha=1.0;
	    float beta=0.0;

        
        for(int i=0;i<10;++i)
        {
            time1=seconds();
            cublasSgemm_v2(handle,transA,transB,A_Device.rows,B_Device.cols,A_Device.cols,&alpha,A_Device.data,A_Device.cols,B_Device.data,B_Device.cols,&beta,C_Device.data,C_Device.cols);
            cudaDeviceSynchronize();
            time2=seconds();
            printf("cublas elapsed:%f ms\n",(time2-time1)*1000);
        }

    }

    // 拷贝GPU结果
    float *dataOfC_DeviceToHost=nullptr;
    dataOfC_DeviceToHost=(float *)malloc(C_Device.GetNumberOfBytes());
    cudaMemcpy(dataOfC_DeviceToHost, C_Device.data, C_Device.GetNumberOfBytes(), cudaMemcpyDeviceToHost);

    // 验证结果的正确性
    float *resultOfCPU=(float*)C_Host.data;
    float *resultOfGPU=(float*)dataOfC_DeviceToHost;
    int numberOfError=0;
    for(int i=0;i<C_Device.GetNumberOfElements();++i)
    {
        if(fabs(resultOfCPU[i]-resultOfGPU[i])>1e-6)
        {
            ++numberOfError;
            printf("index: %d, error: %f\n",i,(float)(resultOfCPU[i]-resultOfGPU[i]));

        }
    }
    if(numberOfError==0)
    {
        printf("the result is OK!\n");
    }

    // free
    cudaFree(dataOfA_Device);
    cudaFree(dataOfB_Device);
    cudaFree(dataOfC_Device);
    free(dataOfC_DeviceToHost);

    return 0;
}
