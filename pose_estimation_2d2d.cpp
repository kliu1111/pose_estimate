#include <iostream>
#include <time.h> 
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
 * 本程序演示了如何使用2D-2D的特征匹配估计相机运动
 * **************************************************/

void find_feature_matches (
    const Mat& img_1, const Mat& img_2,
    std::vector<KeyPoint>& keypoints_1,
    std::vector<KeyPoint>& keypoints_2,
    std::vector< DMatch >& matches );

void pose_estimation_2d2d (
    std::vector<KeyPoint> keypoints_1,
    std::vector<KeyPoint> keypoints_2,
    std::vector< DMatch > matches,
    Mat& R, Mat& t );

// 像素坐标转相机归一化坐标
Point2d pixel2cam ( const Point2d& p, const Mat& K );

int main ( int argc, char** argv )
{
    if ( argc != 3 )
    {
        cout<<"usage: pose_estimation_2d2d img1 img2"<<endl;
        return 1;
    }

    //-- 读取图像
    Mat img_1 = imread ( argv[1], CV_LOAD_IMAGE_COLOR );
    Mat img_2 = imread ( argv[2], CV_LOAD_IMAGE_COLOR );
    // Mat img_1 = imread ( argv[1], IMREAD_GRAYSCALE );
    // Mat img_2 = imread ( argv[2], IMREAD_GRAYSCALE );
    clock_t start, end;
    start = clock();
    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    //find_feature_matches ( img_1, img_2, keypoints_1, keypoints_2, matches );

    // Ptr<Feature2D> detector = xfeatures2d::SIFT::create();
    // Ptr<DescriptorExtractor> extractor = xfeatures2d::SIFT::create();

    Ptr<Feature2D> detector = xfeatures2d::SURF::create();
    Ptr<DescriptorExtractor> extractor = xfeatures2d::SURF::create();


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
    cv::imwrite("误匹配消除前.jpg",img_matches);

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
    cv::imwrite("消除误匹配点后.jpg",img_RR_matches);



    cout<<"一共找到了"<<matches.size() <<"组匹配点"<<endl;
    cout<<"ransac一共找到了"<<RR_matches.size() <<"组匹配点"<<endl;

    //waitKey(0);

    //-- 估计两张图像间运动
    Mat R,t;
    pose_estimation_2d2d ( RR_keypoint01, RR_keypoint02, RR_matches, R, t );



    //-- 验证E=t^R*scale
    Mat t_x = ( Mat_<double> ( 3,3 ) <<
                0,                      -t.at<double> ( 2,0 ),     t.at<double> ( 1,0 ),
                t.at<double> ( 2,0 ),      0,                      -t.at<double> ( 0,0 ),
                -t.at<double> ( 1.0 ),     t.at<double> ( 0,0 ),      0 );

    cout<<"t^R="<<endl<<t_x*R<<endl;

    //-- 验证对极约束
    // Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );
    Mat K = ( Mat_<double> ( 3,3 )<< 1043.02, 0, 979.674, 0, 1047.78, 535.383, 0, 0, 1 );
    int num =0 ;
    ////////////////////////////
    // cv::Mat rgb1,rgb2,depth1,depth2;

    // rgb1 = imread ( argv[1], CV_LOAD_IMAGE_COLOR );
    // rgb2 = imread ( argv[2], CV_LOAD_IMAGE_COLOR );
    // depth1 = cv::imread( "test52_Depth.png", -1 );
    // depth2 = cv::imread( "test51_Depth.png", -1 );
    // PointCloud::Ptr cloud1 ( new PointCloud );
    // PointCloud::Ptr cloud2 ( new PointCloud );
    for ( DMatch m: RR_matches )
    {
        Point2d pt1 = pixel2cam ( RR_keypoint01[ m.queryIdx ].pt, K );
        Mat y1 = ( Mat_<double> ( 3,1 ) << pt1.x, pt1.y, 1 );
        Point2d pt2 = pixel2cam ( RR_keypoint02[ m.trainIdx ].pt, K );
        Mat y2 = ( Mat_<double> ( 3,1 ) << pt2.x, pt2.y, 1 );
        //cout << "y2 = " << y2 << endl;
        Mat d = y2.t() * t_x * R * y1;
        
        // double error = determinant(d);
        //  if (error >-0.009 && error < 0.009)
        //  {
        //      cout << "epipolar constraint = " << error << endl;
        //  }
            // cout <<"ul,vl"<<RR_keypoint01[ m.queryIdx ].pt <<"-----"<< pt1 << endl;
            // cout <<"ul,vl"<<RR_keypoint02[ m.queryIdx ].pt <<"-----"<< pt1 << endl;
   // cout <<""<<RR_keypoint01[ m.queryIdx ].pt <<""<<RR_keypoint02[ m.trainIdx ].pt<< endl;
    // cout <<""<< 1920*(int(RR_keypoint01[ m.queryIdx ].pt.y)-1) + int(RR_keypoint01[ m.queryIdx ].pt.x) <<"====="<< 1920*(int(RR_keypoint02[ m.queryIdx ].pt.y)-1) + int(RR_keypoint02[ m.queryIdx ].pt.x) << endl;
// //---------------------------------------------------
//            int m1 = int(RR_keypoint01[ m.queryIdx ].pt.y);
//            int n1 = int(RR_keypoint01[ m.queryIdx ].pt.x);
//            int m2 = int(RR_keypoint02[ m.trainIdx ].pt.y);
//            int n2 = int(RR_keypoint02[ m.trainIdx ].pt.x);
//            // ushort d1 = depth1.ptr<int>(m1)[n1];
//             // ushort d2 = depth2.ptr<int>(m1)[n1];
//             // d 可能没有值，若如此，跳过此点
//             // if (d1 == 0)
//             //     continue;
//             // if (d2 == 0)
//             //     continue;
//             // d 存在值，则向点云增加一个点
//             PointT p1;
//             // 计算这个点的空间坐标
//             for(int b=m1-20;b<(m1+20);b++)
//             {
//                 for(int a=n1-20;a<(n1+20);a++)
//                 {
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
//             // 把p加入到点云中
//             cloud1->points.push_back( p1 );

//                 }
//             }
//             // p1.z = double(d1) / camera_factor;
//             // p1.x = (n1 - camera_cx) * p1.z / camera_fx;
//             // p1.y = -(m1 - camera_cy) * p1.z / camera_fy;
//             // // 把p加入到点云中
//             // cloud1->points.push_back( p1 );
// //-------------------------------------------------
//             // if (d2 == 0)
//             //     continue;
//             //  PointT p2;
//             // p2.z = double(d2) / camera_factor;
//             // p2.x = (n1 - camera_cx) * p2.z / camera_fx;
//             // p2.y = -(m1 - camera_cy) * p2.z / camera_fy;
//             // // 把p加入到点云中
//             // cloud2->points.push_back( p2 );
//             PointT p2;
//             // 计算这个点的空间坐标
//             for(int b=m2-20;b<(m2+20);b++){
//                 for(int a=n2-20;a<(n2+20);a++){
//             ushort d2 = depth2.ptr<int>(b)[a];
//            //ushort d2 = depth2.ptr<int>(m1)[n1];
//             if (d2 == 0)
//                 continue;
//             p2.z = double(d2) / camera_factor;
//             p2.x = (a - camera_cx) * p2.z / camera_fx;
//             p2.y = -(b - camera_cy) * p2.z / camera_fy;

//             p2.b = rgb2.ptr<uchar>(b)[a*3];
//             p2.g = rgb2.ptr<uchar>(b)[a*3+1];
//             p2.r = rgb2.ptr<uchar>(b)[a*3+2];
//             // 把p加入到点云中
//             cloud2->points.push_back( p2 );

//                 }
//             }
             num ++;
//          //}
       
    }
    //     cloud1->height = 1;
    // cloud1->width = cloud1->points.size();
    // cout<<"point1 cloud size = "<<cloud1->points.size()<<endl;
    // cloud1->is_dense = false;
    //    cloud2->height = 1;
    // cloud2->width = cloud2->points.size();
    // cout<<"point2 cloud size = "<<cloud2->points.size()<<endl;
    // cloud2->is_dense = false;
    // // pcl::io::savePCDFile( "./59.pcd", *cloud );
    // pcl::io::savePLYFile( "./keypoints_52.ply", *cloud1 );
    // pcl::io::savePLYFile( "./keypoints_51.ply", *cloud2 );
    end = clock();
    cout << "calculate time is: " << (double)(end - start)/CLOCKS_PER_SEC << endl;
      printf("%d\n",num);
    return 0;
}
void find_feature_matches ( const Mat& img_1, const Mat& img_2,
                            std::vector<KeyPoint>& keypoints_1,
                            std::vector<KeyPoint>& keypoints_2,
                            std::vector< DMatch >& matches )
{
//读取图像
    // Mat img01=imread("57_Color.png");
    // Mat img02=imread("74_Color.png");
    // cv::namedWindow("original image1",CV_WINDOW_NORMAL);
    // cv::namedWindow("original image2",CV_WINDOW_NORMAL);
    // imshow("original image1",img_1);
    // imshow("original image2",img_2);

    Ptr<Feature2D> detector = xfeatures2d::SIFT::create();
    Ptr<DescriptorExtractor> extractor = xfeatures2d::SIFT::create();

    // Ptr<Feature2D> detector = xfeatures2d::SURF::create();
    // Ptr<DescriptorExtractor> extractor = xfeatures2d::SURF::create();
    //SIFT特征检测
    //SiftFeatureDetector detector;        //定义特点点检测器
    //vector<KeyPoint> keypoints_1,keypoints_2;//定义两个容器存放特征点
    //detector->detect ( img_1,keypoints_1 );
    detector->detect(img_1,keypoints_1);
    detector->detect(img_2,keypoints_2);

    //在两幅图中画出检测到的特征点
    Mat out_img01;
    Mat out_img02;
    drawKeypoints(img_1,keypoints_1,out_img01);
    drawKeypoints(img_2,keypoints_1,out_img02);
    // cv::namedWindow("特征点图01",CV_WINDOW_NORMAL);
    // cv::namedWindow("特征点图02",CV_WINDOW_NORMAL);
    // imshow("特征点图01",out_img01);
    // imshow("特征点图02",out_img02);

    //提取特征点的特征向量（128维）
 
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
    // drawMatches(img_1,keypoints_1,img_2,keypoints_2,matches,img_matches);
    // cv::namedWindow("误匹配消除前",CV_WINDOW_NORMAL);
    // imshow("误匹配消除前",img_matches);
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
    // drawMatches(img_1,RR_keypoint01,img_2,RR_keypoint02,RR_matches,img_RR_matches);
    // cv::namedWindow("消除误匹配点后", CV_WINDOW_NORMAL); 
    // imshow("消除误匹配点后",img_RR_matches);
    //cv::imwrite("消除误匹配点后.jpg",img_RR_matches);
    waitKey(0);
    cout<<"一共找到了"<<matches.size() <<"组匹配点"<<endl;
    cout<<"ransac一共找到了"<<RR_matches.size() <<"组匹配点"<<endl;
}

Point2d pixel2cam ( const Point2d& p, const Mat& K )
{
    return Point2d
           (
               ( p.x - K.at<double> ( 0,2 ) ) / K.at<double> ( 0,0 ),
               ( p.y - K.at<double> ( 1,2 ) ) / K.at<double> ( 1,1 )
           );
}


void pose_estimation_2d2d ( std::vector<KeyPoint> keypoints_1,
                            std::vector<KeyPoint> keypoints_2,
                            std::vector< DMatch > RR_matches,
                            Mat& R, Mat& t )
{
    // 相机内参,TUM Freiburg2
    Mat K = ( Mat_<double> ( 3,3 ) << 1043.02, 0, 979.674, 0, 1047.78, 535.383, 0, 0, 1 );

    //-- 把匹配点转换为vector<Point2f>的形式
    vector<Point2f> points1;
    vector<Point2f> points2;

    for ( int i = 0; i < ( int ) RR_matches.size(); i++ )
    {
        points1.push_back ( keypoints_1[RR_matches[i].queryIdx].pt );
        points2.push_back ( keypoints_2[RR_matches[i].trainIdx].pt );
    }

    //-- 计算基础矩阵
    Mat fundamental_matrix;
    fundamental_matrix = findFundamentalMat ( points1, points2,CV_FM_RANSAC, CV_FM_8POINT );
    //fundamental_matrix = findFundamentalMat ( points1, points2, CV_FM_8POINT );
    //fundamental_matrix = findFundamentalMat ( points1, points2,CV_FM_RANSAC );
    cout<<"fundamental_matrix is "<<endl<< fundamental_matrix<<endl;

    //-- 计算本质矩阵
    //Point2d principal_point ( 325.1, 249.7 );	//相机光心, TUM dataset标定值
    //double focal_length = 521;			//相机焦距, TUM dataset标定值
    double focal_length = 0.5*(K.at<double>(0) + K.at<double>(4));
    Point2d principal_point(K.at<double>(2), K.at<double>(5));
    Mat essential_matrix;
   essential_matrix = findEssentialMat ( points1, points2, focal_length, principal_point,RANSAC);
    //essential_matrix = findEssentialMat ( points1, points2, focal_length, principal_point);
    cout<<"essential_matrix is "<<endl<< essential_matrix<<endl;

    //-- 计算单应矩阵
    Mat homography_matrix;
    homography_matrix = findHomography ( points1, points2, RANSAC, 3 );
    cout<<"homography_matrix is "<<endl<<homography_matrix<<endl;

    //-- 从本质矩阵中恢复旋转和平移信息.
    recoverPose ( essential_matrix, points1, points2, R, t, focal_length, principal_point );
    cout<<"R is "<<endl<<R<<endl;
    cout<<"t is "<<endl<<t<<endl;
    
}
