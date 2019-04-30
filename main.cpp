#include "opencv2/core.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d.hpp"

#include <iostream>
#include <stdio.h>
#include <list>
#include <cmath>
#include <thread>
#include <utility>
#include <cassert>
#include <map>

#define MINIMUM_SIZE 55
#define SAMPLE_NUM 6
#define FAIL_MAX 3
#define EDGE_THRESH 200
#define ADJACENT_THRESH 5
#define RECAL_HEAD 10
#define CONTOUR_THRESH 150
#define BLOCKSIZE 5
#define APERTURESIZE 7
#define K_HC 0.02
#define CORNER_THRESH 70

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay(  VideoCapture &videoCap,
            VideoWriter &outputVideo);

bool adjacent(const Point2f& p1, const Point2f& p2);

bool isGoodDetection(const pair<Point, Point> &boundary, const Rect &largest, int sizeofList);

void detectSingleShape( int &failcount, list<Rect> &Rectlist, Mat &frame_gray, 
                        CascadeClassifier &shapeCascade, pair<Point, Point> &boundary );

double Distance(const Point2f& pt1, const Point2f& pt2);
double RealDistance(const pair<Point, Point> &boundary, int listsize);
void init_image3D( vector<Point3f> &original_image);

/** Global variables */
String rect_cascade_name = "cascade_square_11_16.xml";
String circ_cascade_name = "cascade_circle_11_16.xml";
String hshp_cascade_name = "cascade_Hshape_10_26.xml";
string window_name = "Capture - All shapes detection";
Mat cameraMatrix = ( Mat_<double>(3,3) << 226.595071864109800, 0, 367.463539836797280, 0, 206.377730618208860, 239.581855429149020, 0, 0, 1);
Mat distCoeff = ( Mat_<double>(1,4) << 0.093693958516561, -0.071304881560280, -0.003616384070080, -0.001498354440878, 0.000000000000000);


/** @function main */
int main( int argc, const char** argv )
{
  VideoCapture videoCap("circle.MPG");
  VideoWriter outputVideo;

  Size S=Size((int) videoCap.get(CV_CAP_PROP_FRAME_WIDTH),(int)videoCap.get(CV_CAP_PROP_FRAME_HEIGHT));

  outputVideo.open("result_neo.avi",CV_FOURCC('M','J','P','G'),30.0,S,true);
  detectAndDisplay(videoCap, outputVideo);
  return 0;
}

/************************************************************************************/
// tiny function figuring out if two points are adjacent
// version: 1.2
// Date:    2018.2.18
//
bool adjacent(const Point2f& p1, const Point2f& p2)
{
  return (Distance(p1, p2) <= ADJACENT_THRESH);
}

/************************************************************************************/
// tiny function calculating the distance between two points
// version: 1.0
// Date:    2018.1.14
//
double Distance(const Point2f& pt1, const Point2f& pt2)
{
  return sqrtf(powf((pt1.x - pt2.x), 2) + powf((pt1.y - pt2.y), 2));
}
/************************************************************************************/
// tiny function calculating the distance between two vertices of a boundary pair
// version: 1.1
// Date:    2018.1.9
//
double RealDistance(const pair<Point, Point> &boundary, int listsize)
{
  Point2f realpoint1(double(boundary.first.x)/listsize, double(boundary.first.y)/listsize);
  Point2f realpoint2(double(boundary.second.x)/listsize, double(boundary.second.y)/listsize);

  double distance;
  distance = Distance(realpoint1, realpoint2);
  return distance;
}

/************************************************************************************/
// see if the rectangle passed in matched the boundary we got before
// version: 1.5
// Date:    2018.1.9
//
bool isGoodDetection(const pair<Point, Point> &boundary, const Rect &largest, int sizeofList)
{
  //////////// ** for debug ** ////////////
  // cout << "boundary: " << boundary.first.x << ", " << boundary.first.y << '\t' << boundary.second.x << ", " << boundary.second.y << '\n';
  // cout << "largest: " << largest.x << ", " << largest.y << '\t' << largest.x + largest.width << ", " << largest.y + largest.height << endl;
  return (abs( boundary.first.x/sizeofList - largest.x ) + abs( boundary.first.y/sizeofList - largest.y ) <
          min( (boundary.second.x - boundary.first.x)/sizeofList, largest.width) ) && // centers match?
         (abs( (boundary.second.x - boundary.first.x)/sizeofList - largest.width ) + 
          abs( (boundary.second.y - boundary.first.y)/sizeofList - largest.height ) < 
          min( (boundary.second.x - boundary.first.x)/sizeofList, largest.width)/2); // widths match?
}

/************************************************************************************/
// initialize points in image plane
// version: 1.0
// Date:    2018.2.25
//
void init_image3D( vector<Point3f> &original_image)
{
  original_image.push_back(Point3f(     0,     0, 0));
  original_image.push_back(Point3f(     0,   208, 0));
  original_image.push_back(Point3f(   208,   208, 0));
  original_image.push_back(Point3f(   208,     0, 0));
  original_image.push_back(Point3f(  14.5,  14.5, 0));
  original_image.push_back(Point3f( 193.5,  14.5, 0));
  original_image.push_back(Point3f( 193.5, 193.5, 0));
  original_image.push_back(Point3f(  14.5, 193.5, 0));
  original_image.push_back(Point3f(    25,    25, 0));
  original_image.push_back(Point3f(   182,    25, 0));
  original_image.push_back(Point3f(   182,   182, 0));
  original_image.push_back(Point3f(   182,    25, 0));
  original_image.push_back(Point3f(  38.5,  38.5, 0));
  original_image.push_back(Point3f( 178.5,  38.5, 0));
  original_image.push_back(Point3f( 178.5, 178.5, 0));
  original_image.push_back(Point3f(  38.5, 178.5, 0));
  original_image.push_back(Point3f(  65.5,  65.5, 0));
  original_image.push_back(Point3f(   142,  65.5, 0));
  original_image.push_back(Point3f(   142,   142, 0));
  original_image.push_back(Point3f(  65.5,   142, 0));
}
/************************************************************************************/
// detect shapes in one frame using HAAR detection
// version: 1.8
// Date:    2018.1.9
//
void detectSingleShape( int &failcount, list<Rect> &Rectlist, Mat &frame_gray, 
                        CascadeClassifier &shapeCascade, pair<Point, Point> &boundary )
{
  // vector which stores the detected shapes
  vector<Rect> Rectvector;

  shapeCascade.detectMultiScale( frame_gray, Rectvector, 1.05, 6,
     0|CV_HAAR_SCALE_IMAGE, Size(MINIMUM_SIZE, MINIMUM_SIZE) );

  // store the largest rectangle
  Rect largest;

  // if the detection failed
  if (Rectvector.empty())
  {
    failcount++;
    return;
  }

  int index = 0;
  for( size_t i = 0; i < Rectvector.size(); i++ )
  {
    if (Rectvector[i].width * Rectvector[i].height > largest.width * largest.height )
      index = i;
  }
  largest = Rectvector[index];

  // save last MINIMUM_SIZE largest rectangles
  if (largest.height < MINIMUM_SIZE)
  {
    failcount++;
    return;
  }
  if (Rectlist.empty())
  {
    failcount = 0;
    // add the point to the filtering list
    Rectlist.push_back(largest);
    // up-left point of the rectangle
    boundary.first = Point( largest.x, largest.y );
    // down-right point of the rectangle
    boundary.second = Point(largest.x+largest.width, largest.y+largest.height);
  }
  else if (isGoodDetection(boundary, largest, Rectlist.size()))
  {
    failcount = 0;
    if (Rectlist.size() == SAMPLE_NUM)
    {
      boundary.first.x = boundary.first.x - Rectlist.front().x + largest.x;

      boundary.first.y = boundary.first.y - Rectlist.front().y + largest.y;

      boundary.second.x = boundary.second.x - (Rectlist.front().x + Rectlist.front().width) + largest.x + largest.width;

      boundary.second.y = boundary.second.y - (Rectlist.front().y + Rectlist.front().height) + largest.y + largest.height;
      Rectlist.pop_front();
      Rectlist.push_back(largest);
    }
    else
    {
      boundary.first.x = boundary.first.x + largest.x;

      boundary.first.y = boundary.first.y + largest.y;

      boundary.second.x = boundary.second.x + largest.x + largest.width;

      boundary.second.y = boundary.second.y + largest.y + largest.height;
      Rectlist.push_back(largest);
    }
  }
  else
  {
    //////////// ** for debug ** ////////////
    // cout << "Something Wrong" << endl;
    failcount++;
  }
  if(failcount > RECAL_HEAD)
  {
    Rectlist.clear();
    failcount = 0;
  }
}

/************************************************************************************/
// the function where we realize the continuous detection
// version: 2.1
// Date:    2018.1.27
//
void detectAndDisplay( VideoCapture &videoCap, VideoWriter &outputVideo )
{
  Mat frame;
  Mat frame_gray;

  CascadeClassifier rect_cascade;
  CascadeClassifier circ_cascade;
  CascadeClassifier hshp_cascade;

  // Load the cascades
  rect_cascade.load( rect_cascade_name );
  circ_cascade.load( circ_cascade_name );
  hshp_cascade.load( hshp_cascade_name );

  // some conatiners we are going to use
  list<Rect> some_rect;
  list<Rect> some_circ;
  list<Rect> some_hshp;

  // vector<vector<Point> > contours_rect;
  // vector<vector<Point> > contours_circ;
  // vector<vector<Point> > contours_hshp;

  int failcount_rect = 0;
  int failcount_circ = 0;
  int failcount_hshp = 0;

  // detection boundaries where shapes are held inside (probably)
  pair<Point, Point> boundary_rect;
  pair<Point, Point> boundary_circ;
  pair<Point, Point> boundary_hshp;

  // the final boundary for the shapes
  Point center_rect;
  int radius_rect;
  Point center_circ;
  int radius_circ;
  Point center_hshp;
  int radius_hshp;

  while( true )
  {
    videoCap >> frame;

    // Apply the classifier to the frame
    if( !frame.empty() )
    {
      // pre-processes
      cvtColor( frame, frame_gray, CV_BGR2GRAY );
      equalizeHist( frame_gray, frame_gray );


      // Detect shapes using threading
      // thread rectThread(detectSingleShape, ref(failcount_rect), ref(some_rect), 
      //                   ref(frame_gray), ref(rect_cascade), ref(boundary_rect));

      thread circThread(detectSingleShape, ref(failcount_circ), ref(some_circ), 
                        ref(frame_gray), ref(circ_cascade), ref(boundary_circ));

      // thread hshpThread(detectSingleShape, ref(failcount_hshp), ref(some_hshp), 
      //                   ref(frame_gray), ref(hshp_cascade), ref(boundary_hshp));
      // wait for threads to join
      // rectThread.join();
      circThread.join();
      // hshpThread.join();

      //////////// ** for debug ** ////////////
      // cout << "failcount_rect:  " << failcount_rect << endl;
      // cout << "failcount_circ:  " << failcount_circ << endl;
      // cout << "failcount_hshp:  " << failcount_hshp << endl;
      /* if (failcount_rect < FAIL_MAX && !some_rect.empty())
      {
        center_rect = Point((boundary_rect.first.x + boundary_rect.second.x)*0.5/some_rect.size(), 
                            (boundary_rect.first.y + boundary_rect.second.y)*0.5/some_rect.size());
        radius_rect = cvRound(0.5*RealDistance(boundary_rect, some_rect.size()));
        circle( frame, center_rect, radius_rect, Scalar( 255, 0, 0 ), 4, 8, 0 );
      } */
      if (failcount_circ < FAIL_MAX && !some_circ.empty())
      {
        center_circ = Point((boundary_circ.first.x + boundary_circ.second.x)*0.5/some_circ.size(), 
                            (boundary_circ.first.y + boundary_circ.second.y)*0.5/some_circ.size());
        radius_circ = cvRound(0.3*RealDistance(boundary_circ, some_circ.size()));
        circle( frame, center_circ, radius_circ, Scalar( 0, 255, 0 ), 4, 8, 0 );
      }
      /*
      if (failcount_hshp < FAIL_MAX && !some_hshp.empty())
      {
        center_hshp = Point((boundary_hshp.first.x + boundary_hshp.second.x)*0.5/some_hshp.size(), 
                            (boundary_hshp.first.y + boundary_hshp.second.y)*0.5/some_hshp.size());
        radius_hshp = cvRound(0.5*RealDistance(boundary_hshp, some_hshp.size()));
        circle( frame, center_hshp, radius_hshp, Scalar( 0, 0, 255 ), 4, 8, 0 );
      }*/
      if ( failcount_circ == 0 )
      { 
        // cout << failcount_rect << endl;
        // blur the frame
        blur( frame_gray, frame_gray, Size(5, 5));
        
        // some temporary containers we need to use
        Mat frame_binary;
        // Mat dst, dst_norm, dst_norm_scaled;
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        // create threshold to binarize the image
        threshold( frame_gray, frame_binary, CONTOUR_THRESH, 255, THRESH_BINARY );
        // map<float, Point> pointmap; // save the points in the order of their scale of being a corner
        // vector<Point2f> pointvec; // save the real corners
        // dst = Mat::zeros( frame_gray.size(), CV_32FC1 );
        // Detecting corners
        // cornerHarris( frame_gray, dst, BLOCKSIZE, APERTURESIZE, K_HC, BORDER_DEFAULT );
        // Normalizing
        // normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
        /*
        // save all possible points
        for( int j = std::max(0, center_rect.y - radius_rect); j < std::min(dst.rows, center_rect.y + radius_rect); j++ )
        { 
          for( int i = std::max(0, center_rect.x - radius_rect); i < std::min(dst.cols, center_rect.x + radius_rect); i++ )
          {
            if( (int) dst_norm.at<float>(j,i) > CORNER_THRESH )
            {
              if ( Distance(Point(i, j), center_rect) < 0.9*radius_rect)
              {
                pointmap.insert(make_pair(dst_norm.at<float>(j,i), Point(i, j)));
                // circle( frame, Point( i, j ), 5,  Scalar(0), 4, 8, 0 );
              }
            }
          }
        }

        for (map<float, Point>::reverse_iterator ritr = pointmap.rbegin(); ritr != pointmap.rend(); ritr++)
        {
          if(pointvec.size() == 0)
          {
            pointvec.push_back(Point2f(ritr->second.x, ritr->second.y));
          }

          // we need only 20 points given by the square shape
          else if (pointvec.size() == 20) { break; }

          // get rid of the duplicate corners
          else 
          {
            size_t i = 0;
            for ( i = 0; i<pointvec.size(); i++)
              if(adjacent(ritr->second, pointvec[i]))
                break;
            if (i == pointvec.size())
            {
              pointvec.push_back(Point2f(ritr->second.x, ritr->second.y));
            }
          }
        }
        for (size_t i = 0; i<pointvec.size(); i++)
        {
          circle( frame, pointvec[i], 3, Scalar(0), 4, 8, 0 );
          // cout << pointvec[i].x << '\t' << pointvec[i].y << endl;
        }*/

        // Find contours
        findContours( frame_binary, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
        vector<RotatedRect> minEllipse( contours.size() );
        for (int m = 0; m < hierarchy.size(); ++m)
        {
          cout << hierarchy[m] << endl;
        }
        cout << endl;
        // vector<vector<Point> > contours_poly( contours.size() );  // container containing the approx-ploygons
        // vector<Rect> boundRect( contours.size() );
        for( size_t i = 0; i < contours.size(); i++ )
        { 
          // minRect[i] = minAreaRect( Mat(contours[i]) );
          if( contours[i].size() > 4)
          { 
            minEllipse[i] = fitEllipse( Mat(contours[i]) ); 
          }
        }

        Point2f vertices[4];
        size_t chosen_one = 0;
        Size2f sizeofEllipse;
        for( size_t i = 0; i < minEllipse.size(); i++)
        {
          minEllipse[i].points(vertices);
          bool flag = true;
          for(int j=0; j<4; j++)
          {
            if(Distance(vertices[j], center_circ) > 1.41*radius_circ)
              flag=false;
          }
          if(flag)
          {
            if (minEllipse[i].size.area() > sizeofEllipse.area())
            {
              chosen_one = i;
              sizeofEllipse = minEllipse[i].size;
            }
          }
        }
        drawContours( frame, contours, chosen_one, Scalar( 0, 255, 0), 1, 8, vector<Vec4i>(), 0, Point() );
        // circle( frame, contours[chosen_one][0], 3, Scalar( 0, 0, 255), 2, 8, 0 );
        // rectangle( frame, boundRect[i], Scalar( 0, 0, 255), 2, 8, 0 );
        ellipse( frame, minEllipse[chosen_one], Scalar( 255, 0, 255), 2, 8 );
      
      /*
      vector<Point2f> floating_contours;
      if (contours[chosen_one].size() >= 4)
      {
        // cout << contours[chosen_one].size() << endl;
        vector<Point3f> original_image;
        for(size_t k = 0; k < contours[chosen_one].size(); k++)
        {
          floating_contours.push_back(Point2f(contours[chosen_one][k].x, contours[chosen_one][k].y));
          if (k < contours[chosen_one].size()*0.25)
          {
            size_t y = k*800/contours[chosen_one].size();
            original_image.push_back(Point3f(0, y, 0));
          }
          else if(k<contours[chosen_one].size()*0.5)
          {
            size_t y = fmax(0, (k-0.25*contours[chosen_one].size())*800/contours[chosen_one].size());
            original_image.push_back(Point3f(200, y, 0));
          }
          else if(k<contours[chosen_one].size()*0.75)
          {
            size_t x = fmax(0, (k-0.5*contours[chosen_one].size())*800/contours[chosen_one].size());
            original_image.push_back(Point3f(x, 0, 0));
          }
          else
          {
            size_t x = fmax(0, (k-0.75*contours[chosen_one].size())*800/contours[chosen_one].size());
            original_image.push_back(Point3f(x, 200, 0));
          }
        }
        vector<Point3f> original_image;
        init_image3D(original_image);
        Mat rvec(3,1,cv::DataType<double>::type);
        Mat tvec(3,1,cv::DataType<double>::type);
        // cout << original_image.size() - contours[chosen_one].size() << endl;
        // InputArray temp1(original_image);
        // InputArray temp2(contours[chosen_one]);
        // Mat temp3 = temp1.getMat();
        // Mat temp4 = temp2.getMat();
        // cout << temp3.checkVector(3, CV_32F);
        if(pointvec.size()==original_image.size())
        {
          solvePnPRansac(Mat(original_image), Mat(pointvec), cameraMatrix, distCoeff, rvec, tvec);
          cout << "rvec" << '\n' << rvec << '\n' << "tvec" << '\n' << tvec << endl;
          
          vector<Point2f> projectedPoints;
          projectPoints(original_image, rvec, tvec, cameraMatrix, distCoeff, projectedPoints);
  
          for(size_t i = 0; i < projectedPoints.size(); ++i)
          {
            std::cout << "Image point: " << pointvec[i] << " Projected to " << projectedPoints[i] << std::endl;
          }
        }
      }*/
      // for (size_t k=0; k<contours[chosen_one].size(); k++)
      // {
      //   Point3f temp(contours[chosen_one][k].x, contours[chosen_one][k].y, 0);
      //   contours_image.push_back(temp);
      // }
      // solvePnPRansac(original_image, contours[chosen_one], );
      // Draw contours
      // Mat drawing = Mat::zeros( threshold_output.size(), CV_8UC3 );
      // for( size_t i = 0; i< contours.size(); i++ )
      // {

      //   // rotated rectangle
      //   // Point2f rect_points[4]; minRect[i].points( rect_points );
      //   // for( int j = 0; j < 4; j++ )
      //   //   line( drawing, rect_points[j], rect_points[(j+1)%4], color, 1, 8 );
      //   // circle( frame, center[i], (int)radius[i], Scalar( 0, 0, 255), 2, 8, 0 );

      //   /*
      //     find the largest ellipse inside the circle
      //   */
      // }
        outputVideo.write(frame);
        
        // Show what you got
        cvWaitKey(1);
        // imshow( window_name, Mat(contours[chosen_one]) );
        imshow( window_name, frame );
      }
    }
    else
    {
      break;
    }
  }
}