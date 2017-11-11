#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <chrono>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
//#include "extra.h" // use this if in OpenCV2 
#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;

void find_feature_matches (
    const Mat& img_1, const Mat& img_2,
    std::vector<KeyPoint>& keypoints_1,
    std::vector<KeyPoint>& keypoints_2,
    std::vector< DMatch >& matches );

// 像素坐标转相机归一化坐标
Point2d pixel2cam ( const Point2d& p, const Mat& K );

void pose_estimation_3d3d (
    const vector<Point3f>& pts1,
    const vector<Point3f>& pts2,
    Mat& R, Mat& t
);

void bundleAdjustment(
    const vector<Point3f>& points_3d,
    const vector<Point3f>& points_2d,
    Mat& R, Mat& t
);

// g2o edge
class EdgeProjectXYZRGBDPoseOnly : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeProjectXYZRGBDPoseOnly( const Eigen::Vector3d& point ) : _point(point) {}

    virtual void computeError()
    {
        const g2o::VertexSE3Expmap* pose = static_cast<const g2o::VertexSE3Expmap*> ( _vertices[0] );
        // measurement is p, point is p'
        _error = _measurement - pose->estimate().map( _point );
    }
    
    virtual void linearizeOplus()
    {
        g2o::VertexSE3Expmap* pose = static_cast<g2o::VertexSE3Expmap *>(_vertices[0]);
        g2o::SE3Quat T(pose->estimate());
        Eigen::Vector3d xyz_trans = T.map(_point);
        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double z = xyz_trans[2];
        
        _jacobianOplusXi(0,0) = 0;
        _jacobianOplusXi(0,1) = -z;
        _jacobianOplusXi(0,2) = y;
        _jacobianOplusXi(0,3) = -1;
        _jacobianOplusXi(0,4) = 0;
        _jacobianOplusXi(0,5) = 0;
        
        _jacobianOplusXi(1,0) = z;
        _jacobianOplusXi(1,1) = 0;
        _jacobianOplusXi(1,2) = -x;
        _jacobianOplusXi(1,3) = 0;
        _jacobianOplusXi(1,4) = -1;
        _jacobianOplusXi(1,5) = 0;
        
        _jacobianOplusXi(2,0) = -y;
        _jacobianOplusXi(2,1) = x;
        _jacobianOplusXi(2,2) = 0;
        _jacobianOplusXi(2,3) = 0;
        _jacobianOplusXi(2,4) = 0;
        _jacobianOplusXi(2,5) = -1;
    }

    bool read ( istream& in ) {}
    bool write ( ostream& out ) const {}
protected:
    Eigen::Vector3d _point;
};

int main ( int argc, char** argv )
{
    if ( argc != 5 )
    {
        cout<<"usage: pose_estimation_3d3d img1 img2 depth1 depth2"<<endl;
        return 1;
    }
    //-- 读取图像
    clock_t start, end;
    start = clock();
    Mat img_1 = imread ( argv[1], CV_LOAD_IMAGE_COLOR );
    Mat img_2 = imread ( argv[2], CV_LOAD_IMAGE_COLOR );
    // Mat img_1 = imread ( argv[1], IMREAD_GRAYSCALE );
    // Mat img_2 = imread ( argv[2], IMREAD_GRAYSCALE );
    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    //find_feature_matches ( img_1, img_2, keypoints_1, keypoints_2, matches );
    // Ptr<Feature2D> detector = xfeatures2d::SIFT::create();
    // Ptr<DescriptorExtractor> extractor = xfeatures2d::SIFT::create();

    Ptr<Feature2D> detector = xfeatures2d::SURF::create();
    Ptr<DescriptorExtractor> extractor = xfeatures2d::SURF::create();
    
    detector->detect(img_1,keypoints_1);
    detector->detect(img_2,keypoints_2);



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
    
    cout<<"一共找到了"<<matches.size() <<"组匹配点"<<endl;
    cout<<"ransac一共找到了"<<RR_matches.size() <<"组匹配点"<<endl;
    waitKey(0);

    // 建立3D点
    Mat depth1 = imread ( argv[3], CV_LOAD_IMAGE_UNCHANGED );       // 深度图为16位无符号数，单通道图像
    Mat depth2 = imread ( argv[4], CV_LOAD_IMAGE_UNCHANGED );       // 深度图为16位无符号数，单通道图像
   // Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );
     Mat K = ( Mat_<double> ( 3,3 ) << 1043.02, 0, 979.674, 0, 1047.78, 535.383, 0, 0, 1 );
    vector<Point3f> pts1, pts2;

    for ( DMatch m:RR_matches )
    {
        ushort d1 = depth1.ptr<unsigned short> ( int ( RR_keypoint01[m.queryIdx].pt.y ) ) [ int ( RR_keypoint01[m.queryIdx].pt.x ) ];
        ushort d2 = depth2.ptr<unsigned short> ( int ( RR_keypoint02[m.trainIdx].pt.y ) ) [ int ( RR_keypoint02[m.trainIdx].pt.x ) ];
        if ( d1==0 || d2==0 )   // bad depth
            continue;
        Point2d p1 = pixel2cam ( RR_keypoint01[m.queryIdx].pt, K );
        Point2d p2 = pixel2cam ( RR_keypoint02[m.trainIdx].pt, K );
        float dd1 = float ( d1 ) /1000.0;
        float dd2 = float ( d2 ) /1000.0;
        pts1.push_back ( Point3f ( p1.x*dd1, p1.y*dd1, dd1 ) );
        pts2.push_back ( Point3f ( p2.x*dd2, p2.y*dd2, dd2 ) );
    }

    cout<<"3d-3d pairs: "<<pts1.size() <<endl;
    cout<<"3d-3d pairs: "<<pts2.size() <<endl;
    Mat R, t;
    pose_estimation_3d3d ( pts1, pts2, R, t );
    cout<<"ICP via SVD results: "<<endl;
    cout<<"R = "<<R<<endl;
    cout<<"t = "<<t<<endl;
    cout<<"R_inv = "<<R.t() <<endl;
    cout<<"t_inv = "<<-R.t() *t<<endl;

    end = clock();
    cout << "calculate time is: " << (double)(end - start)/CLOCKS_PER_SEC << endl;
    cout<<"calling bundle adjustment"<<endl;

   // bundleAdjustment( pts1, pts2, R, t );
    
    // verify p1 = R*p2 + t
    for ( int i=0; i<5; i++ )
    {
        cout<<"p1 = "<<pts1[i]<<endl;
        cout<<"p2 = "<<pts2[i]<<endl;
        cout<<"(R*p2+t) = "<< 
            R * (Mat_<double>(3,1)<<pts2[i].x, pts2[i].y, pts2[i].z) + t
            <<endl;
        cout<<endl;
    }
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
//     drawKeypoints(img_1,keypoints_1,out_img01);
//     drawKeypoints(img_2,keypoints_1,out_img02);
//     // cv::namedWindow("特征点图01",CV_WINDOW_NORMAL);
//     // cv::namedWindow("特征点图02",CV_WINDOW_NORMAL);
//     // imshow("特征点图01",out_img01);
//     // imshow("特征点图02",out_img02);

//     //提取特征点的特征向量（128维）
 
//     Mat descriptor01,descriptor02;
//     extractor->compute(img_1,keypoints_1,descriptor01);
//     extractor->compute(img_2,keypoints_2,descriptor02);

//     //匹配特征点，主要计算两个特征点特征向量的欧式距离，距离小于某个阈值则认为匹配
//      //Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "FlannBased" );
//     //Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-L1" );
//      Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce" );
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
//     //Mat Fundamental= findFundamentalMat(p01,p02,RansacStatus,FM_RANSAC);
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
//     waitKey(0);
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
//     Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create("BruteForce-Hamming");
//     //-- 第一步:检测 Oriented FAST 角点位置
//     detector->detect ( img_1,keypoints_1 );
//     detector->detect ( img_2,keypoints_2 );

//     //-- 第二步:根据角点位置计算 BRIEF 描述子
//     descriptor->compute ( img_1, keypoints_1, descriptors_1 );
//     descriptor->compute ( img_2, keypoints_2, descriptors_2 );

//     //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
//     vector<DMatch> match;
//    // BFMatcher matcher ( NORM_HAMMING );
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
//         if ( match[i].distance <= max ( 2*min_dist, 30.0 ) )
//         {
//             matches.push_back ( match[i] );
//         }
//     }
//     // Mat matches;
//     // drawMatches ( img_1, keypoints_1, img_2, keypoints_2, matches );
//     // cv::namedWindow("所有匹配点对", CV_WINDOW_NORMAL);  
//     // imshow ( "所有匹配点对", matches );
// }

Point2d pixel2cam ( const Point2d& p, const Mat& K )
{
    return Point2d
           (
               ( p.x - K.at<double> ( 0,2 ) ) / K.at<double> ( 0,0 ),
               ( p.y - K.at<double> ( 1,2 ) ) / K.at<double> ( 1,1 )
           );
}

void pose_estimation_3d3d (
    const vector<Point3f>& pts1,
    const vector<Point3f>& pts2,
    Mat& R, Mat& t
)
{
    Point3f p1, p2;     // center of mass
    int N = pts1.size();
    for ( int i=0; i<N; i++ )
    {
        p1 += pts1[i];
        p2 += pts2[i];
    }
    p1 = Point3f( Vec3f(p1) /  N);
    p2 = Point3f( Vec3f(p2) / N);
    vector<Point3f>     q1 ( N ), q2 ( N ); // remove the center
    for ( int i=0; i<N; i++ )
    {
        q1[i] = pts1[i] - p1;
        q2[i] = pts2[i] - p2;
    }

    // compute q1*q2^T
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for ( int i=0; i<N; i++ )
    {
        W += Eigen::Vector3d ( q1[i].x, q1[i].y, q1[i].z ) * Eigen::Vector3d ( q2[i].x, q2[i].y, q2[i].z ).transpose();
    }
    cout<<"W="<<W<<endl;

    // SVD on W
    Eigen::JacobiSVD<Eigen::Matrix3d> svd ( W, Eigen::ComputeFullU|Eigen::ComputeFullV );
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    cout<<"U="<<U<<endl;
    cout<<"V="<<V<<endl;

    Eigen::Matrix3d R_ = U* ( V.transpose() );
    Eigen::Vector3d t_ = Eigen::Vector3d ( p1.x, p1.y, p1.z ) - R_ * Eigen::Vector3d ( p2.x, p2.y, p2.z );

    // convert to cv::Mat
    R = ( Mat_<double> ( 3,3 ) <<
          R_ ( 0,0 ), R_ ( 0,1 ), R_ ( 0,2 ),
          R_ ( 1,0 ), R_ ( 1,1 ), R_ ( 1,2 ),
          R_ ( 2,0 ), R_ ( 2,1 ), R_ ( 2,2 )
        );
    t = ( Mat_<double> ( 3,1 ) << t_ ( 0,0 ), t_ ( 1,0 ), t_ ( 2,0 ) );
}

void bundleAdjustment (
    const vector< Point3f >& pts1,
    const vector< Point3f >& pts2,
    Mat& R, Mat& t )
{
    // 初始化g2o
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<6,3> > Block;  // pose维度为 6, landmark 维度为 3
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverEigen<Block::PoseMatrixType>(); // 线性方程求解器
    Block* solver_ptr = new Block( linearSolver );      // 矩阵块求解器
    g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( solver_ptr );
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm( solver );

    // vertex
    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap(); // camera pose
    pose->setId(0);
    pose->setEstimate( g2o::SE3Quat(
        Eigen::Matrix3d::Identity(),
        Eigen::Vector3d( 0,0,0 )
    ) );
    optimizer.addVertex( pose );

    // edges
    int index = 1;
    vector<EdgeProjectXYZRGBDPoseOnly*> edges;
    for ( size_t i=0; i<pts1.size(); i++ )
    {
        EdgeProjectXYZRGBDPoseOnly* edge = new EdgeProjectXYZRGBDPoseOnly( 
            Eigen::Vector3d(pts2[i].x, pts2[i].y, pts2[i].z) );
        edge->setId( index );
        edge->setVertex( 0, dynamic_cast<g2o::VertexSE3Expmap*> (pose) );
        edge->setMeasurement( Eigen::Vector3d( 
            pts1[i].x, pts1[i].y, pts1[i].z) );
        edge->setInformation( Eigen::Matrix3d::Identity()*1e4 );
        optimizer.addEdge(edge);
        index++;
        edges.push_back(edge);
    }

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.setVerbose( true );
    optimizer.initializeOptimization();
    optimizer.optimize(100);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
    cout<<"optimization costs time: "<<time_used.count()<<" seconds."<<endl;

    cout<<endl<<"after optimization:"<<endl;
    cout<<"T="<<endl<<Eigen::Isometry3d( pose->estimate() ).matrix()<<endl;
    
}
