#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
// #include "extra.h" // use this if in OpenCV2 
#include <opencv2/xfeatures2d.hpp>

#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

// 定义点云类型
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud; 

//相机内参
const double camera_factor = 1000;
const double camera_cx = 979.674;
const double camera_cy = 535.383;
const double camera_fx = 1043.02;
const double camera_fy = 1047.78;
using namespace std;
using namespace cv;

/****************************************************
 * 本程序演示了如何使用2D-2D的特征匹配取得对应点区域点云
 * **************************************************/

int main ( int argc, char** argv )
{
    if ( argc != 3 )
    {
        cout<<"usage: image_feature2point img1 img2"<<endl;
        return 1;
    }
    //-- 读取图像
    Mat img_1 = imread ( argv[1], CV_LOAD_IMAGE_COLOR );
    Mat img_2 = imread ( argv[2], CV_LOAD_IMAGE_COLOR );
    // Mat img_1 = imread ( argv[1], IMREAD_GRAYSCALE );
    // Mat img_2 = imread ( argv[2], IMREAD_GRAYSCALE );
    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;

    Ptr<Feature2D> detector = xfeatures2d::SIFT::create();
    Ptr<DescriptorExtractor> extractor = xfeatures2d::SIFT::create();

    // Ptr<Feature2D> detector = xfeatures2d::SURF::create();
    // Ptr<DescriptorExtractor> extractor = xfeatures2d::SURF::create();

    detector->detect(img_1,keypoints_1);
    detector->detect(img_2,keypoints_2);

    Mat descriptor01,descriptor02;
    extractor->compute(img_1,keypoints_1,descriptor01);
    extractor->compute(img_2,keypoints_2,descriptor02);

    //匹配特征点，主要计算两个特征点特征向量的欧式距离，距离小于某个阈值则认为匹配
    //Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "FlannBased" );
    //Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-L1" );
     Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce" );
    //BruteForceMatcher<L2<float>> matcher;
    //vector<DMatch> matches;
    Mat img_matches;
    matcher->match(descriptor01,descriptor02,matches);
    drawMatches(img_1,keypoints_1,img_2,keypoints_2,matches,img_matches);
    cv::namedWindow("误匹配消除前",CV_WINDOW_NORMAL);
    imshow("误匹配消除前",img_matches);
    //////////////////////////////////////////
      //RANSAC 消除误匹配特征点 主要分为三个部分：
    //1）根据matches将特征点对齐,将坐标转换为float类型
    //2）使用求基础矩阵方法 findFundamentalMat,得到RansacStatus
    //3）根据RansacStatus来将误匹配的点也即RansacStatus[i]=0的点删除

    //根据matches将特征点对齐,将坐标转换为float类型
    vector<KeyPoint> R_keypoint01,R_keypoint02;
    for (size_t i=0;i<matches.size();i++)   
    {
        R_keypoint01.push_back(keypoints_1[matches[i].queryIdx]);
        R_keypoint02.push_back(keypoints_2[matches[i].trainIdx]);
        //这两句话的理解：R_keypoint1是要存储img01中能与img02匹配的特征点，
        //matches中存储了这些匹配点对的img01和img02的索引值
    }

    //坐标转换
    vector<Point2f>p01,p02;
    for (size_t i=0;i<matches.size();i++)
    {
        p01.push_back(R_keypoint01[i].pt);
        p02.push_back(R_keypoint02[i].pt);
    }

    //利用基础矩阵剔除误匹配点
    vector<uchar> RansacStatus;
    //Mat Fundamental= findFundamentalMat(p01,p02,RansacStatus,FM_RANSAC);
    Mat Fundamental= findHomography(p01,p02,CV_FM_RANSAC,3,RansacStatus);

    vector<KeyPoint> RR_keypoint01,RR_keypoint02;
    vector<DMatch> RR_matches;            //重新定义RR_keypoint 和RR_matches来存储新的关键点和匹配矩阵
    int index=0;
    for (size_t i=0;i<matches.size();i++)
    {
        if (RansacStatus[i]!=0)
        {
            RR_keypoint01.push_back(R_keypoint01[i]);
            RR_keypoint02.push_back(R_keypoint02[i]);
            matches[i].queryIdx=index;
            matches[i].trainIdx=index;
            RR_matches.push_back(matches[i]);
            index++;
        }
    }
    Mat img_RR_matches;
    drawMatches(img_1,RR_keypoint01,img_2,RR_keypoint02,RR_matches,img_RR_matches);
    cv::namedWindow("消除误匹配点后", CV_WINDOW_NORMAL); 
    imshow("消除误匹配点后",img_RR_matches);
    cv::imwrite("imgfeature2piont/消除误匹配点后.jpg",img_RR_matches);
   // waitKey(0);
    cout<<"一共找到了"<<matches.size() <<"组匹配点"<<endl;
    cout<<"ransac一共找到了"<<RR_matches.size() <<"组匹配点"<<endl;
////////////////////////////////////////////////////////////////
    int num =0 ;
    ////////////////////////////
    cv::Mat rgb1,rgb2,depth1,depth2;

    rgb1 = imread ( argv[1], CV_LOAD_IMAGE_COLOR );
    rgb2 = imread ( argv[2], CV_LOAD_IMAGE_COLOR );
    depth1 = cv::imread( "left_Depth.png", -1 );
    depth2 = cv::imread( "right_Depth.png", -1 );
    PointCloud::Ptr cloud1 ( new PointCloud );
    PointCloud::Ptr cloud2 ( new PointCloud );
//----------------------------------------------neighbourhood-----------------------------------
    for ( DMatch m: RR_matches )
    {
      
           int m1 = int(RR_keypoint01[ m.queryIdx ].pt.y);
           int n1 = int(RR_keypoint01[ m.queryIdx ].pt.x);
           int m2 = int(RR_keypoint02[ m.trainIdx ].pt.y);
           int n2 = int(RR_keypoint02[ m.trainIdx ].pt.x);
            PointT p1;
            // 计算这个点的空间坐标
            for(int b=m1-4;b<(m1+4);b++)
            {
                for(int a=n1-4;a<(n1+4);a++)
                {
                 ushort d1 = depth1.ptr<int>(b)[a];
                 if (d1 == 0)
                      continue;
           // cout << "d1== " << d1 << endl;
                p1.z = double(d1) / camera_factor;
                p1.x = (a - camera_cx) * p1.z / camera_fx;
                p1.y = -(b - camera_cy) * p1.z / camera_fy;

                p1.b = rgb1.ptr<uchar>(b)[a*3];
                p1.g = rgb1.ptr<uchar>(b)[a*3+1];
                p1.r = rgb1.ptr<uchar>(b)[a*3+2];
            // 把p加入到点云中
                }
            }
            cout <<""<<RR_keypoint01[ m.queryIdx ].pt <<""<<RR_keypoint02[ m.trainIdx ].pt<< "----"<<p1.x<<"---"<<p1.y<<"---"<<p1.z<<endl;
//-------------------------------------------------
            PointT p2;
            // 计算这个点的空间坐标
            for(int b=m2-4;b<(m2+4);b++)
            {
                for(int a=n2-4;a<(n2+4);a++)
                {
                    ushort d2 = depth2.ptr<int>(b)[a];
                   //ushort d2 = depth2.ptr<int>(m1)[n1];
                    if (d2 == 0)
                        continue;
                    p2.z = double(d2) / camera_factor;
                    p2.x = (a - camera_cx) * p2.z / camera_fx;
                    p2.y = -(b - camera_cy) * p2.z / camera_fy;

                    p2.b = rgb2.ptr<uchar>(b)[a*3];
                    p2.g = rgb2.ptr<uchar>(b)[a*3+1];
                    p2.r = rgb2.ptr<uchar>(b)[a*3+2];
                    // 把p加入到点云中
                    cloud1->points.push_back( p1 );
                    cloud2->points.push_back( p2 );

                }
            }
             num ++;
//          //}  
    }
//-----------------------------------------------matching keypoint---------------------------------
//     for ( DMatch m: RR_matches )
//     {
//     cout <<""<<RR_keypoint01[ m.queryIdx ].pt <<""<<RR_keypoint02[ m.trainIdx ].pt<< endl;
// // //---------------------------------------------------
//            int m1 = int(RR_keypoint01[ m.queryIdx ].pt.y);
//            int n1 = int(RR_keypoint01[ m.queryIdx ].pt.x);
//            int m2 = int(RR_keypoint02[ m.trainIdx ].pt.y);
//            int n2 = int(RR_keypoint02[ m.trainIdx ].pt.x);
//             PointT p1;
//             // 计算这个点的空间坐标
//             int b=m1;
//             int a=n1;
//             ushort d1 = depth1.ptr<int>(b)[a];
//             if (d1 == 0)
//                 continue;
//            // cout << "d1== " << d1 << endl;
//             p1.z = double(d1) / camera_factor;
//             p1.x = (a - camera_cx) * p1.z / camera_fx;
//             p1.y = -(b - camera_cy) * p1.z / camera_fy;

//             p1.b = rgb1.ptr<uchar>(b)[a*3];
//             p1.g = rgb1.ptr<uchar>(b)[a*3+1];
//             p1.r = rgb1.ptr<uchar>(b)[a*3+2];

// //-------------------------------------------------
//             PointT p2;
//             // 计算这个点的空间坐标
//             int b2=m2;
//             int a2=n2;
//             ushort d2 = depth2.ptr<int>(b2)[a2];
//            //ushort d2 = depth2.ptr<int>(m1)[n1];
//             if (d2 == 0)
//                 continue;
//             p2.z = double(d2) / camera_factor;
//             p2.x = (a2 - camera_cx) * p2.z / camera_fx;
//             p2.y = -(b2 - camera_cy) * p2.z / camera_fy;

//             p2.b = rgb2.ptr<uchar>(b2)[a2*3];
//             p2.g = rgb2.ptr<uchar>(b2)[a2*3+1];
//             p2.r = rgb2.ptr<uchar>(b2)[a2*3+2];
//             // 把p加入到点云中
//             cloud1->points.push_back( p1 );
//             // 把p加入到点云中
//             cloud2->points.push_back( p2 );
//              num ++;
// //          //}
       
//     }
// //-------------------------------------------------all keypoint--------------
//     for ( DMatch m: matches )
//     {
//     //cout <<""<<R_keypoint01[ m.queryIdx ].pt <<""<<R_keypoint02[ m.trainIdx ].pt<< endl;
// // //---------------------------------------------------
//            int m1 = int(R_keypoint01[ m.queryIdx ].pt.y);
//            int n1 = int(R_keypoint01[ m.queryIdx ].pt.x);
//            int m2 = int(R_keypoint02[ m.trainIdx ].pt.y);
//            int n2 = int(R_keypoint02[ m.trainIdx ].pt.x);

//             PointT p1;
//             // 计算这个点的空间坐标
//             int b=m1;
//             int a=n1;
//             ushort d1 = depth1.ptr<int>(b)[a];
//             if (d1 == 0)
//                 continue;
//            // cout << "d1== " << d1 << endl;
//             p1.z = double(d1) / camera_factor;
//             p1.x = (a - camera_cx) * p1.z / camera_fx;
//             p1.y = -(b - camera_cy) * p1.z / camera_fy;

//             p1.b = rgb1.ptr<uchar>(b)[a*3];
//             p1.g = rgb1.ptr<uchar>(b)[a*3+1];
//             p1.r = rgb1.ptr<uchar>(b)[a*3+2];

// //-------------------------------------------------
//             PointT p2;
//             // 计算这个点的空间坐标
//             int b2=m2;
//             int a2=n2;
//             ushort d2 = depth2.ptr<int>(b2)[a2];
//            //ushort d2 = depth2.ptr<int>(m1)[n1];
//             if (d2 == 0)
//                 continue;
//             p2.z = double(d2) / camera_factor;
//             p2.x = (a2 - camera_cx) * p2.z / camera_fx;
//             p2.y = -(b2 - camera_cy) * p2.z / camera_fy;

//             p2.b = rgb2.ptr<uchar>(b2)[a2*3];
//             p2.g = rgb2.ptr<uchar>(b2)[a2*3+1];
//             p2.r = rgb2.ptr<uchar>(b2)[a2*3+2];
//             // 把p加入到点云中
//             cloud1->points.push_back( p1 );
//             // 把p加入到点云中
//             cloud2->points.push_back( p2 );
//             // num ++;     
//     }

    cloud1->height = 1;
    cloud1->width = cloud1->points.size();
    cout<<"point1 cloud size = "<<cloud1->points.size()<<endl;
    cloud1->is_dense = false;
       cloud2->height = 1;
    cloud2->width = cloud2->points.size();
    cout<<"point2 cloud size = "<<cloud2->points.size()<<endl;
    cloud2->is_dense = false;
    pcl::io::savePLYFile( "imgfeature2piont/left.ply", *cloud1 );
    pcl::io::savePLYFile( "imgfeature2piont/right.ply", *cloud2 );
    //  printf("%d\n",num);
    return 0;
}

