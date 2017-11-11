#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <Eigen/Geometry>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
//#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <chrono>

using namespace std;
using namespace cv;

void find_feature_matches (
    const Mat& img_1, const Mat& img_2,
    std::vector<KeyPoint>& keypoints_1,
    std::vector<KeyPoint>& keypoints_2,
    std::vector< DMatch >& matches );

// 像素坐标转相机归一化坐标
Point2d pixel2cam ( const Point2d& p, const Mat& K );

void bundleAdjustment (
    const vector<Point3f> points_3d,
    const vector<Point2f> points_2d,
    const Mat& K,
    Mat& R, Mat& t
);

int main ( int argc, char** argv )
{
    if ( argc != 5 )
    {
        cout<<"usage: pose_estimation_3d2d img1 img2 depth1 depth2"<<endl;
        return 1;
    }
    clock_t start, end;
    start = clock();
    //-- 读取图像
    // Mat img_1 = imread ( argv[1], CV_LOAD_IMAGE_COLOR );
    // Mat img_2 = imread ( argv[2], CV_LOAD_IMAGE_COLOR );
    Mat img_1 = imread ( argv[1], IMREAD_GRAYSCALE );
    Mat img_2 = imread ( argv[2], IMREAD_GRAYSCALE );
    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    //find_feature_matches ( img_1, img_2, keypoints_1, keypoints_2, matches );

    // Ptr<Feature2D> detector = xfeatures2d::SIFT::create();
    // Ptr<DescriptorExtractor> extractor = xfeatures2d::SIFT::create();

    Ptr<Feature2D> detector = xfeatures2d::SURF::create();
    Ptr<DescriptorExtractor> extractor = xfeatures2d::SURF::create();

    //SIFT特征检测
    //SiftFeatureDetector detector;        //定义特点点检测器
    //vector<KeyPoint> keypoints_1,keypoints_2;//定义两个容器存放特征点
       //detector->detect ( img_1,keypoints_1 );
    detector->detect(img_1,keypoints_1);
    detector->detect(img_2,keypoints_2);

    //在两幅图中画出检测到的特征点
    // Mat out_img01;
    // Mat out_img02;
    // drawKeypoints(img_1,keypoints_1,out_img01);
    // drawKeypoints(img_2,keypoints_1,out_img02);
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
    cv::imwrite("消除误匹配点后.jpg",img_RR_matches);
    //waitKey(0);
    cout<<"一共找到了"<<matches.size() <<"组匹配点"<<endl;
    cout<<"ransac一共找到了"<<RR_matches.size() <<"组匹配点"<<endl;

    // 建立3D点
    Mat d1 = imread ( argv[3], CV_LOAD_IMAGE_UNCHANGED );       // 深度图为16位无符号数，单通道图像
    //Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );
    Mat K = ( Mat_<double> ( 3,3 )<< 1043.02, 0, 979.674, 0, 1047.78, 535.383, 0, 0, 1 );
    vector<Point3f> pts_3d;
    vector<Point2f> pts_2d;
    for ( DMatch m:RR_matches )
    {
        ushort d = d1.ptr<unsigned short> (int ( RR_keypoint01[m.queryIdx].pt.y )) [ int ( RR_keypoint01[m.queryIdx].pt.x ) ];
        if ( d == 0 )   // bad depth
            continue;
        float dd = d/1000.0;
        Point2d p1 = pixel2cam ( RR_keypoint01[m.queryIdx].pt, K );
        pts_3d.push_back ( Point3f ( p1.x*dd, p1.y*dd, dd ) );
        pts_2d.push_back ( RR_keypoint02[m.trainIdx].pt );
    }
    cout<<"3d-2d pairs: "<<pts_2d.size() <<endl;
    cout<<"3d-2d pairs: "<<pts_3d.size() <<endl;

    Mat r, t;
    solvePnP ( pts_3d, pts_2d, K, Mat(), r, t, false,SOLVEPNP_EPNP); // 调用OpenCV 的 PnP 求解，可选择EPNP，DLS等方法
    Mat R;
    cv::Rodrigues ( r, R ); // r为旋转向量形式，用Rodrigues公式转换为矩阵

    cout<<"R="<<endl<<R<<endl;
    cout<<"t="<<endl<<t<<endl;

    cout<<"calling bundle adjustment"<<endl;


    //bundleAdjustment ( pts_3d, pts_2d, K, R, t );
    end = clock();
    cout << "calculate time is: " << (double)(end - start)/CLOCKS_PER_SEC << endl;
}
// void find_feature_matches ( const Mat& img_1, const Mat& img_2,
//                             std::vector<KeyPoint>& keypoints_1,
//                             std::vector<KeyPoint>& keypoints_2,
//                             std::vector< DMatch >& matches )
// {
// //读取图像
//     // Mat img01=imread("57_Color.png");
//     // Mat img02=imread("74_Color.png");
//     // cv::namedWindow("original image1",CV_WINDOW_NORMAL);
//     // cv::namedWindow("original image2",CV_WINDOW_NORMAL);
//     // imshow("original image1",img_1);
//     // imshow("original image2",img_2);

//     Ptr<Feature2D> detector = xfeatures2d::SIFT::create();
//     Ptr<DescriptorExtractor> extractor = xfeatures2d::SIFT::create();

//     // Ptr<Feature2D> detector = xfeatures2d::SURF::create();
//     // Ptr<DescriptorExtractor> extractor = xfeatures2d::SURF::create();
//     //SIFT特征检测
//     //SiftFeatureDetector detector;        //定义特点点检测器
//     //vector<KeyPoint> keypoints_1,keypoints_2;//定义两个容器存放特征点
//        //detector->detect ( img_1,keypoints_1 );
//     detector->detect(img_1,keypoints_1);
//     detector->detect(img_2,keypoints_2);

//     //在两幅图中画出检测到的特征点
//     Mat out_img01;
//     Mat out_img02;
//     // drawKeypoints(img_1,keypoints_1,out_img01);
//     // drawKeypoints(img_2,keypoints_1,out_img02);
//     // cv::namedWindow("特征点图01",CV_WINDOW_NORMAL);
//     // cv::namedWindow("特征点图02",CV_WINDOW_NORMAL);
//     // imshow("特征点图01",out_img01);
//     // imshow("特征点图02",out_img02);

//     //提取特征点的特征向量（128维）
 
//     Mat descriptor01,descriptor02;
//     extractor->compute(img_1,keypoints_1,descriptor01);
//     extractor->compute(img_2,keypoints_2,descriptor02);

//     //匹配特征点，主要计算两个特征点特征向量的欧式距离，距离小于某个阈值则认为匹配
//      Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce" );
//     // Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-L1" );
//      //Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-L1" );
//     //BruteForceMatcher<L2<float>> matcher;
//     //vector<DMatch> matches;
//     Mat img_matches;
//     matcher->match(descriptor01,descriptor02,matches);
//     drawMatches(img_1,keypoints_1,img_2,keypoints_2,matches,img_matches);
//     cv::namedWindow("误匹配消除前",CV_WINDOW_NORMAL);
//     imshow("误匹配消除前",img_matches);
//     //////////////////////////////////////////
//       //RANSAC 消除误匹配特征点 主要分为三个部分：
//     //1）根据matches将特征点对齐,将坐标转换为float类型
//     //2）使用求基础矩阵方法 findFundamentalMat,得到RansacStatus
//     //3）根据RansacStatus来将误匹配的点也即RansacStatus[i]=0的点删除

//     //根据matches将特征点对齐,将坐标转换为float类型
//     vector<KeyPoint> R_keypoint01,R_keypoint02;
//     for (size_t i=0;i<matches.size();i++)   
//     {
//         R_keypoint01.push_back(keypoints_1[matches[i].queryIdx]);
//         R_keypoint02.push_back(keypoints_2[matches[i].trainIdx]);
//         //这两句话的理解：R_keypoint1是要存储img01中能与img02匹配的特征点，
//         //matches中存储了这些匹配点对的img01和img02的索引值
//     }

//     //坐标转换
//     vector<Point2f>p01,p02;
//     for (size_t i=0;i<matches.size();i++)
//     {
//         p01.push_back(R_keypoint01[i].pt);
//         p02.push_back(R_keypoint02[i].pt);
//     }

//     //利用基础矩阵剔除误匹配点
//     vector<uchar> RansacStatus;
//    // Mat Fundamental= findFundamentalMat(p01,p02,RansacStatus,FM_RANSAC);
//     Mat Fundamental= findHomography(p01,p02,CV_FM_RANSAC,3,RansacStatus);

//     vector<KeyPoint> RR_keypoint01,RR_keypoint02;
//     vector<DMatch> RR_matches;            //重新定义RR_keypoint 和RR_matches来存储新的关键点和匹配矩阵
//     int index=0;
//     for (size_t i=0;i<matches.size();i++)
//     {
//         if (RansacStatus[i]!=0)
//         {
//             RR_keypoint01.push_back(R_keypoint01[i]);
//             RR_keypoint02.push_back(R_keypoint02[i]);
//             matches[i].queryIdx=index;
//             matches[i].trainIdx=index;
//             RR_matches.push_back(matches[i]);
//             index++;
//         }
//     }
//     Mat img_RR_matches;
//     drawMatches(img_1,RR_keypoint01,img_2,RR_keypoint02,RR_matches,img_RR_matches);
//     cv::namedWindow("消除误匹配点后", CV_WINDOW_NORMAL); 
//     imshow("消除误匹配点后",img_RR_matches);
//     //cv::imwrite("消除误匹配点后.jpg",img_RR_matches);
//     //waitKey(0);
//     cout<<"一共找到了"<<matches.size() <<"组匹配点"<<endl;
//     cout<<"ransac一共找到了"<<RR_matches.size() <<"组匹配点"<<endl;
// }
// void find_feature_matches ( const Mat& img_1, const Mat& img_2,
//                             std::vector<KeyPoint>& keypoints_1,
//                             std::vector<KeyPoint>& keypoints_2,
//                             std::vector< DMatch >& matches )
// {
//     //-- 初始化
//     Mat descriptors_1, descriptors_2;
//     // used in OpenCV3
//     Ptr<FeatureDetector> detector = ORB::create();
//     Ptr<DescriptorExtractor> descriptor = ORB::create();
//     // use this if you are in OpenCV2
//     // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
//     // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
//     Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-Hamming" );
//     //-- 第一步:检测 Oriented FAST 角点位置
//     detector->detect ( img_1,keypoints_1 );
//     detector->detect ( img_2,keypoints_2 );

//     //-- 第二步:根据角点位置计算 BRIEF 描述子
//     descriptor->compute ( img_1, keypoints_1, descriptors_1 );
//     descriptor->compute ( img_2, keypoints_2, descriptors_2 );

//     //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
//     vector<DMatch> match;
//     // BFMatcher matcher ( NORM_HAMMING );
//     matcher->match ( descriptors_1, descriptors_2, match );

//     //-- 第四步:匹配点对筛选
//     double min_dist=10000, max_dist=0;

//     //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
//     for ( int i = 0; i < descriptors_1.rows; i++ )
//     {
//         double dist = match[i].distance;
//         if ( dist < min_dist ) min_dist = dist;
//         if ( dist > max_dist ) max_dist = dist;
//     }

//     printf ( "-- Max dist : %f \n", max_dist );
//     printf ( "-- Min dist : %f \n", min_dist );

//     //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
//     for ( int i = 0; i < descriptors_1.rows; i++ )
//     {
//         if ( match[i].distance <= max ( 1.3*min_dist, 30.0 ) )
//         {
//             matches.push_back ( match[i] );
//         }
//     }
// }

Point2d pixel2cam ( const Point2d& p, const Mat& K )
{
    return Point2d
           (
               ( p.x - K.at<double> ( 0,2 ) ) / K.at<double> ( 0,0 ),
               ( p.y - K.at<double> ( 1,2 ) ) / K.at<double> ( 1,1 )
           );
}

// void bundleAdjustment (
//     const vector< Point3f > points_3d,
//     const vector< Point2f > points_2d,
//     const Mat& K,
//     Mat& R, Mat& t )
// {
//     // 初始化g2o
//     typedef g2o::BlockSolver< g2o::BlockSolverTraits<6,3> > Block;  // pose 维度为 6, landmark 维度为 3
//     Block::LinearSolverType* linearSolver = new g2o::LinearSolverCSparse<Block::PoseMatrixType>(); // 线性方程求解器
//     Block* solver_ptr = new Block ( linearSolver );     // 矩阵块求解器

//     g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( solver_ptr );// 梯度下降方法，从GN, LM, DogLeg 中选
//     g2o::SparseOptimizer optimizer;      // 图模型
//     optimizer.setAlgorithm ( solver );    // 设置求解器

//     // vertex
//     g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap(); // camera pose
//     Eigen::Matrix3d R_mat;
//     R_mat <<
//           R.at<double> ( 0,0 ), R.at<double> ( 0,1 ), R.at<double> ( 0,2 ),
//                R.at<double> ( 1,0 ), R.at<double> ( 1,1 ), R.at<double> ( 1,2 ),
//                R.at<double> ( 2,0 ), R.at<double> ( 2,1 ), R.at<double> ( 2,2 );
//     pose->setId ( 0 );
//     pose->setEstimate ( g2o::SE3Quat (
//                             R_mat,
//                             Eigen::Vector3d ( t.at<double> ( 0,0 ), t.at<double> ( 1,0 ), t.at<double> ( 2,0 ) )
//                         ) );
//     optimizer.addVertex ( pose );

//     int index = 1;
//     for ( const Point3f p:points_3d )   // landmarks   三维点
//     {
//         g2o::VertexSBAPointXYZ* point = new g2o::VertexSBAPointXYZ();
//         point->setId ( index++ );
//         point->setEstimate ( Eigen::Vector3d ( p.x, p.y, p.z ) );
//         point->setMarginalized ( true ); // g2o 中必须设置 marg 参见第十讲内容
//         optimizer.addVertex ( point );
//     }

//     // parameter: camera intrinsics
//     g2o::CameraParameters* camera = new g2o::CameraParameters (
//         K.at<double> ( 0,0 ), Eigen::Vector2d ( K.at<double> ( 0,2 ), K.at<double> ( 1,2 ) ), 0
//     );
//     camera->setId ( 0 );
//     optimizer.addParameter ( camera );

//     // edges   
//     index = 1;
//     for ( const Point2f p:points_2d )    //像素点
//     {
//         g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
//         edge->setId ( index );
//         edge->setVertex ( 0, dynamic_cast<g2o::VertexSBAPointXYZ*> ( optimizer.vertex ( index ) ) );
//         edge->setVertex ( 1, pose );
//         edge->setMeasurement ( Eigen::Vector2d ( p.x, p.y ) );
//         edge->setParameterId ( 0,0 );
//         edge->setInformation ( Eigen::Matrix2d::Identity() );
//         optimizer.addEdge ( edge );
//         index++;
//     }

//     chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
//     optimizer.setVerbose ( true );
//     optimizer.initializeOptimization();
//     optimizer.optimize ( 100 );
//     chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
//     chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> ( t2-t1 );
//     cout<<"optimization costs time: "<<time_used.count() <<" seconds."<<endl;

//     cout<<endl<<"after optimization:"<<endl;
//     cout<<"T="<<endl<<Eigen::Isometry3d ( pose->estimate() ).matrix() <<endl;
// }
