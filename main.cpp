#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <time.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include  "opencv2/highgui.hpp"
#include <iostream>
#include <math.h>
using namespace std;
using namespace cv;

//输入图像为BGR图，将其转化为gray图
void rgb2grayInCuda(sycl::uchar3 *dataIn, unsigned char *dataOut, int imgHeight,
                    int imgWidth, sycl::nd_item<3> item_ct1)
{
    //图片二维扫描，分别有x方向，y方向的像素点
    int xIndex =
        item_ct1.get_local_id(2) +
        item_ct1.get_group(2) * item_ct1.get_local_range(2); // 表示x方向上的ID
    int yIndex =
        item_ct1.get_local_id(1) +
        item_ct1.get_group(1) * item_ct1.get_local_range(1); // 表示y方向上的ID
    //灰度变换操作
    if (xIndex < imgWidth && yIndex < imgHeight)
    {
        sycl::uchar3 rgb = dataIn[yIndex * imgWidth + xIndex];
        dataOut[yIndex * imgWidth + xIndex] =
            0.299f * rgb.x() + 0.587f * rgb.y() + 0.114f * rgb.z();
    }
}
//串行转换灰度图像
void rgb2grayincpu(unsigned char * const d_in, unsigned char * const d_out,uint imgheight, uint imgwidth)
{
    //使用两重循环嵌套实现x方向 y方向的变换
    for(int i = 0; i < imgheight; i++)
    {
        for(int j = 0; j < imgwidth; j++)
        {
            d_out[i * imgwidth + j] = 0.299f * d_in[(i * imgwidth + j)*3]
                                     + 0.587f * d_in[(i * imgwidth + j)*3 + 1]
                                     + 0.114f * d_in[(i * imgwidth + j)*3 + 2];
        }
    }
}

//灰度直方图统计
void imHistInCuda(unsigned char *dataIn, int *hist, sycl::nd_item<3> item_ct1)
{
    int threadIndex = item_ct1.get_local_id(2) +
                      item_ct1.get_local_id(1) * item_ct1.get_local_range(2);
    int blockIndex = item_ct1.get_group(2) +
                     item_ct1.get_group(1) * item_ct1.get_group_range(2);
    int index = threadIndex + blockIndex * item_ct1.get_local_range(2) *
                                  item_ct1.get_local_range(1);

    dpct::atomic_fetch_add<int, sycl::access::address_space::generic_space>(
        &hist[dataIn[index]], 1);
        //多个thread有序地对*dataIn地址加1
        //如果使用自加（++），会出现多个threads同步写竞争，造成数据出错
}

int main()
{
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
    //传入图片
    Mat srcImg = imread("testpic2.png");
    FILE* fp;//创建运行时间文件

    //读取图片像素值
    int imgHeight = srcImg.rows;  
    int imgWidth = srcImg.cols;  

    Mat grayImg(imgHeight, imgWidth, CV_8UC1, Scalar(0));	//输出灰度图
    int hist[256];	//灰度直方图统计数组
    memset(hist, 0, 256 * sizeof(int));	//对灰度直方图数组初始化为0
	
    //在GPU中开辟输入输出空间
    sycl::uchar3 *d_in;
    unsigned char *d_out;
    int *d_hist;

    //分配内存空间
    d_in = sycl::malloc_device<sycl::uchar3>(imgHeight * imgWidth, q_ct1);
    d_out = sycl::malloc_device<unsigned char>(imgHeight * imgWidth, q_ct1);
    d_hist = sycl::malloc_device<int>(256, q_ct1);

    //将图像数据传入GPU中
    q_ct1.memcpy(d_in, srcImg.data,
                 imgHeight * imgWidth * sizeof(sycl::uchar3));
    q_ct1.memcpy(d_hist, hist, 256 * sizeof(int)).wait();

    sycl::range<3> threadsPerBlock(1, 32, 32);
    sycl::range<3> blocksPerGrid(
        1, (imgHeight + threadsPerBlock[1] - 1) / threadsPerBlock[1],
        (imgWidth + threadsPerBlock[2] - 1) / threadsPerBlock[2]);

    //统计时间
    clock_t start, end;

    /*
    DPCT1008:2: clock function is not defined in the SYCL. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    start = clock();
    //cuda灰度化
    /*
    DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    q_ct1.parallel_for(
        sycl::nd_range<3>(blocksPerGrid * threadsPerBlock, threadsPerBlock),
        [=](sycl::nd_item<3> item_ct1) {
            rgb2grayInCuda(d_in, d_out, imgHeight, imgWidth, item_ct1);
        });

    dev_ct1
        .queues_wait_and_throw(); // 同步CPU和gpu，否则测速结果为cpu启动内核函数的速度
    /*
    DPCT1008:3: clock function is not defined in the SYCL. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    end = clock();
    double gputime =(double)(end-start)/CLOCKS_PER_SEC;

    //打印cuda并行执行时间
    printf("cuda exec time is %.20lf\n", gputime);

    //灰度直方图统计
    /*
    DPCT1049:1: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    q_ct1.parallel_for(
        sycl::nd_range<3>(blocksPerGrid * threadsPerBlock, threadsPerBlock),
        [=](sycl::nd_item<3> item_ct1) {
            imHistInCuda(d_out, d_hist, item_ct1);
        });

    //将数据从GPU传回CPU
    q_ct1.memcpy(hist, d_hist, 256 * sizeof(int));
    q_ct1
        .memcpy(grayImg.data, d_out,
                imgHeight * imgWidth * sizeof(unsigned char))
        .wait();

        vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9);
        sycl::free(d_in, q_ct1);
        sycl::free(d_out, q_ct1);
        sycl::free(d_hist, q_ct1);

    //串行灰度化
    /*
    DPCT1008:4: clock function is not defined in the SYCL. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    start = clock();
    rgb2grayincpu(srcImg.data, grayImg.data, imgHeight, imgWidth);
    /*
    DPCT1008:5: clock function is not defined in the SYCL. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    end = clock();

    double cputime =(double)(end-start)/CLOCKS_PER_SEC;
    
    //打印串行执行时间
    printf("cpu exec time is %.20lf\n",cputime );

    //将串行、并行执行时间记录到文件中，方便查看比对
    fp = fopen("time.txt","a");
    fprintf(fp,"cpu exec time is %.20lf s , cuda exec time is %.20lf s \n", cputime, gputime);
    fclose(fp);
	try  
   	{  
                imwrite("result.png",grayImg,compression_params);
		//在build文件夹中，生成灰度变换后的结果图片  
    	}  
    	catch (runtime_error& ex)  
    	{  
        	fprintf(stderr, "图像转换成PNG格式发生错误：%s\n", ex.what());  
        	return 1;  
    	}  
    return 0;
}