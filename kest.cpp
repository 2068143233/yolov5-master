#include <ros/ros.h>
#include <std_msgs/String.h>
#include<car_sensor/image_detect.h>
#include<string>
#include <boost/thread.hpp>
#include <sstream>
#include<sensor_msgs/PointCloud2.h>
#include<opencv2/opencv.hpp>
#include<cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
//pcl and pointcloud2 to handle the lidar_point data
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <sensor_msgs/PointCloud2.h>

using namespace std;
using namespace cv;

class SubscribeAndPublish
{

private:
  cv::Mat image;
  ros::NodeHandle n;
  //ros::Publisher pub_;
  ros::Subscriber sub_camera;
  ros::Subscriber sub_lidar;
  //std_msgs::String output;
  //int count;
  std::vector<string> roi_label;
  std::vector<int> roi_leftX;
  std::vector<int> roi_leftY;
  std::vector<int> roi_width;
  std::vector<int> roi_height;
  vector<Point2f> projectedPoints;           
  vector<Point3f> point_data;//lidar data in <Point3f> data type
  pcl::PointCloud<pcl::PointXYZ> lidar_cloud;

public:
  
  SubscribeAndPublish()
  {
    sub_camera = n.subscribe("/daheng_camera_image", 10, &SubscribeAndPublish::callback_camera, this);
    sub_lidar = n.subscribe("/lslidar_point_cloud", 10, &SubscribeAndPublish::callback_lidar, this);  
  }

  ~SubscribeAndPublish()
  {

  }
//相机的回调函数，这里的car_sensor是我自己的package,image_detect是自己定义的msg消息
  void callback_camera(car_sensor::image_detect msg_camera)
  {
    handle_detect_image(msg_camera,roi_label,roi_leftX,roi_leftY,roi_width,roi_height,image);
    data_fusion();
  }
//雷达的回调函数
  void callback_lidar(sensor_msgs::PointCloud2 msg_lidar)
  {
    lidar_cloud.clear();
    point_data.clear();
    handle_lidar_data(msg_lidar,point_data,lidar_cloud);

  }
  //这里我们用来接收yolov3检测出来物体的坐标和label
  void handle_detect_image(car_sensor::image_detect msg_camera,std::vector<string> &roi_label,std::vector<int> &roi_leftX,
  std::vector<int> &roi_leftY,std::vector<int> &roi_width,std::vector<int> &roi_height,Mat &image)
  {
    if(msg_camera.label.size()>0)
    {
      roi_label.clear();
      roi_leftX.clear();
      roi_leftY.clear();
      roi_width.clear();
      roi_height.clear();
      for(int i=0;i<msg_camera.label.size();i++)
      {
        roi_label.push_back(msg_camera.label[i]);
        roi_leftX.push_back(msg_camera.X[i]);
        roi_leftY.push_back(msg_camera.Y[i]);
        roi_width.push_back(msg_camera.W[i]);
        roi_height.push_back(msg_camera.H[i]);
      }
      //from ROS image to opencv Mat image
      cv_bridge::CvImagePtr cv_ptr;
      cv_ptr = cv_bridge::toCvCopy(msg_camera.image_raw, sensor_msgs::image_encodings::BGR8);   
      image=cv_ptr->image;//transformed successfully from ROS to Mat
    }
    else
    {
      std::cout<<"We do not detetct any objects in the image!"<<endl;
    }

    //.... do something with the input and generate the output...
  }
//我们用来处理雷达发布的点云信息，保存到opencv中的vector<Point3f>中
  void handle_lidar_data(sensor_msgs::PointCloud2 msg_lidar,vector<Point3f> &point_data,pcl::PointCloud<pcl::PointXYZ> &lidar_cloud)
  {
    lidar_cloud.clear();
    point_data.clear();
    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(msg_lidar,pcl_pc2);  //sensor_msgs::PointCloud2->pcl::PCLPointCloud2
    pcl::fromPCLPointCloud2(pcl_pc2,lidar_cloud);//pcl::PCLPointCloud2->pcl::PointCloud<pcl::PointXYZ>
    //cout<<lidar_cloud.size()<<endl;

    if(lidar_cloud.size()>0)       //pcl::PointCloud<pcl::PointXYZ>->vector<Point3f>
    {
      for(int i=0;i<lidar_cloud.size();i++)
      {
        point_data.push_back(Point3f(lidar_cloud[i].x,lidar_cloud[i].y,lidar_cloud[i].z));
      }
    }
    else
    {
      std::cout<<"I have not receive the lidar data!"<<endl;
    }
  }
  //我们在这里处理雷达和点云数据的融合，主要是把雷达点云数据映射到图像上，并可视化出来~
  void data_fusion()
  {

    if(image.empty())
    {
      std::cout<<"There is no image data in data_fusion!"<<endl;
    }
    else if(point_data.size()==0)
    {
      cout<<"there is no lidar data in data_fusion!"<<endl;
    }
    //when we get both the image data and the lidar data.we begin to do data-fusion
    else
    {
      projectedPoints.clear();//first you need to clear the vector to avoid data accumulation
      //相机的内参矩阵
      cv::Mat distCoeffs(5, 1, cv::DataType<double>::type);   // Distortion vector
      distCoeffs.at<double>(0) =-0.0565;
      distCoeffs.at<double>(1) = 0.0643;
      distCoeffs.at<double>(2) = 0;
      distCoeffs.at<double>(3) =0;
      distCoeffs.at<double>(4) = 0;

      cv::Mat cameraMatrix(3, 3, cv::DataType<double>::type);
      //float tempMatrix[3][3] = { { 3522.3, 0, 0 }, { 0, 3538.4, 0 }, { 1968.9,1375.4,1.0 } };
      float tempMatrix[3][3] = { { 3522.3, 0, 1968.9 }, { 0, 3538.4, 1375.4 }, { 0, 0, 1.0 } };

      for (int i = 0; i < 3; i++)
      {
          for (int j = 0; j < 3; j++)
          {
              cameraMatrix.at<double>(i, j) = tempMatrix[i][j];
          }
      }
      //标定出的外参矩阵
      cv::Mat rvec(3, 3, cv::DataType<double>::type);

      rvec.at<double>(0,0)=0.9802;
      rvec.at<double>(0,1)=0.1979;
      rvec.at<double>(0,2)=0.01078;
      rvec.at<double>(1,0)=0.00999;
      rvec.at<double>(1,1)=0.00497;
      rvec.at<double>(1,2)=-0.9999;
      rvec.at<double>(2,0)=-0.198;
      rvec.at<double>(2,1)=0.9802;
      rvec.at<double>(2,2)=0.00289;

      cv::Mat tvec(3, 1, cv::DataType<double>::type);
      tvec.at<double>(0,0)=-1094.1;
      tvec.at<double>(1,0)=140.88;
      tvec.at<double>(2,0)=-203.2;
      cv::projectPoints(point_data, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);

      for (int i = 0; i<projectedPoints.size(); i++)
      {
          cv::Point2f p = projectedPoints[i];
          if (p.y<1080)
          {
              if(point_data[i].y<2000)
              {
                  circle(image, p, 5, Scalar(255, 255, 0), 1, 8, 0);
              }
              else
              {
                  circle(image, p, 5, Scalar(255, 0, 255), 1, 8, 0);
              }
           }
      }
      cv::namedWindow("lidar_camera",0);
      cv::imshow("lidar_camera",image);
      cv::waitKey(10);
    }

  }

};//End of class SubscribeAndPublish


int main(int argc, char **argv)
{
  //Initiate ROS
  ros::init(argc, argv, "subscribe_and_publish");
  //Create an object of class SubscribeAndPublish that will take care of everything
  SubscribeAndPublish fusiondata;
  ros::spin();

  return 0;
}
