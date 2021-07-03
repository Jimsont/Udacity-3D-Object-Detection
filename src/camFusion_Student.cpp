
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);
    

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(20); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // loop through all keypoint matches
    // if the boundingBox contains current keypoint, calculate distance and store into distance vector 
    // Then, calculate average distance shift of matches. 
    // 
    vector<double> disShiftMatches;
    std::vector<cv::DMatch> matchesInRoi;
    for (auto match : kptMatches)
    {
        // if(boundingBox.roi.contains(kptsCurr[match.trainIdx].pt))
        if(boundingBox.roi.contains(kptsCurr[match.trainIdx].pt))
        {
            double disShift = cv::norm(kptsCurr[match.trainIdx].pt - kptsPrev[match.queryIdx].pt);
            disShiftMatches.push_back(disShift);
            matchesInRoi.push_back(match);
        }
    }
    // calculate average distance shift and std of the shifts
    double disShiftMean = 0.0, disShiftStd = 0.0;
    int nMatch = disShiftMatches.size();
    disShiftMean = accumulate(disShiftMatches.begin(), disShiftMatches.end(), 0.0);
    disShiftMean = disShiftMean / ((double)nMatch);

    for (auto dist : disShiftMatches)
    {
        disShiftStd = disShiftStd + (dist-disShiftMean)*(dist-disShiftMean);
    }
    disShiftStd = sqrt(disShiftStd/((double)nMatch));
    // for (auto disShift : disShiftMatches)
    // {
    //     disShiftStd = disShiftStd + (disShift - disShiftMean)*(disShift - disShiftMean);
    // }
    // disShiftStd = sqrt(disShiftStd/((double)nMatch));

    // loop through disShiftMatches, if disshift is whithin average*ratio, insert the match into boundingBox
    // double utol = 0.5;
    // double ltol = 0.00;
    // double ratioL= 0.7;
    for (int i = 0; i<nMatch; i++)
    {
        // if (disShiftMatches[i]< utol+disShiftMean && disShiftMatches[i]>ltol*disShiftMean)
        if (disShiftMatches[i]<disShiftStd*2+disShiftMean)
        // if (disShiftMatches[i]<1.5*disShiftMean)
        {
            boundingBox.kptMatches.push_back(matchesInRoi[i]);
        }
    }
    cout<<"original match size = "<<nMatch<<", leftover match size = "<<boundingBox.kptMatches.size()<<'\n';
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // extract first match point (p1 current image, p1 previous image)
    // loop through all the other match to get (p2 current image, p2 previous image)
    // calculate ratio = norm(p1 current image - p2 current image) / norm(p1 previous image-p2 previous image)
    // store ratio > thresh into vector distRatios
    vector<double> distRatios;
    for (int i=0; i<kptMatches.size()-1; i++)
    {
        cv::KeyPoint pt1Prev, pt1Curr;
        pt1Prev = kptsPrev[kptMatches[i].queryIdx];
        pt1Curr = kptsCurr[kptMatches[i].trainIdx];

        for (int j = i+1; j<kptMatches.size(); j++)
        {
            double thresh = 35.0;
            cv::KeyPoint pt2Prev, pt2Curr;
            pt2Prev = kptsPrev[kptMatches[j].queryIdx];
            pt2Curr = kptsCurr[kptMatches[j].trainIdx];

            double disPrev, disCurr, disRatio;
            disPrev = cv::norm(pt1Prev.pt - pt2Prev.pt);
            disCurr = cv::norm(pt1Curr.pt - pt2Curr.pt);

            // avoiding division by zero and discard small distance
            if (disPrev > std::numeric_limits<double>::epsilon() && disCurr>thresh)
            {
                disRatio = disCurr/disPrev;
                distRatios.push_back(disRatio);
            }
        }
    }
    // restun NAN is no ratio is pushed into distRatios
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    // calculate median ratio
    sort(distRatios.begin(), distRatios.end());
    double medDistRatio = 0.0;
    if (distRatios.size()%2!=0)
    {
        medDistRatio = distRatios[distRatios.size()/2];
    }
    else
    {
        medDistRatio = (distRatios[distRatios.size()/2] + distRatios[distRatios.size()/2-1])/2;
    }
    // medDistRatio = accumulate(distRatios.begin(), distRatios.end(), 0.0);
    // medDistRatio = medDistRatio/((double)distRatios.size());
    // cout<<"size of raitos for camera TTC calculation = "<<distRatios.size()<<'\n';
    // calculate TTC
    double dt = 1/frameRate;
    TTC = -dt / (1-medDistRatio);
    cout<<" ttc camera ratio size = "<<distRatios.size()<<"\n";
}


double computeDisLidar(std::vector<LidarPoint> &lidarPoints, double cutRatioL,  double cutRatioU, int sizeThresh, double tolY, double tolZ)
{
    // if number of lidar points < kernal size and > 0, calculate mean distance as estimation
    // if number of lidar points > sizeThresh, sorted lidarPoints by x measurement
    // cut out the largest N points of sorted lidarPoints,
    // N = floor(number of lidar points * cutTatio)
    // calculate mean and STD of remaning lidarPoints 
    // update distance estimation by mean-2*STD
  
    double dis = 1e9;
    int nlidarPt = lidarPoints.size();
    // double tolY = 0.5;
    if (nlidarPt<=sizeThresh && nlidarPt>0)
    {
        double disAccu = 0.0;
        for (auto lidar : lidarPoints)
        {
            disAccu = disAccu + lidar.x;
        }
        dis = disAccu/((double)nlidarPt);
    }
    else if (nlidarPt > sizeThresh)
    {
        double disAccu = 0.0;
        double centerPosY = 0.0;
        double centerPosZ = 0.0;

        std::vector<double> lidarPtSortedy;
        std::vector<double> lidarPtSortedz;
        std::vector<double> lidarPtSorted;

        for (auto lidar : lidarPoints)
        {
            lidarPtSortedy.push_back(lidar.y);
            lidarPtSortedz.push_back(lidar.z);
        }

        // sort lidar point clouds Y, then calculate center of preceding car in Y axis
        sort(lidarPtSortedy.begin(), lidarPtSortedy.end());
        if (lidarPtSortedy.size()%2 != 0)
        {
            centerPosY = lidarPtSortedy[lidarPtSortedy.size()/2];
        }
        else
        {
            centerPosY = (lidarPtSortedy[lidarPtSortedy.size()/2] + lidarPtSortedy[lidarPtSortedy.size()/2-1])/2;
        }
        
        // centerPosY = accumulate(lidarPtSortedy.begin(), lidarPtSortedy.end(), 0.0)/((double)nlidarPt);
        // sort lidar point clouds Z, then calculate center of preceding car in Z axis
        sort(lidarPtSortedz.begin(), lidarPtSortedz.end());
        if (lidarPtSortedz.size()%2 != 0)
        {
            centerPosZ = lidarPtSortedz[lidarPtSortedz.size()/2];
        }
        else
        {
            centerPosZ = (lidarPtSortedz[lidarPtSortedz.size()/2] + lidarPtSortedz[lidarPtSortedz.size()/2-1])/2;
        }
        // centerPosZ = accumulate(lidarPtSortedz.begin(), lidarPtSortedz.end(), 0.0)/((double)nlidarPt);

        // extract lidar point that is within y+tolY;
        for (auto lidar : lidarPoints)
        {
            if (abs(lidar.y - centerPosY)<tolY && abs(lidar.z - centerPosZ)<tolZ)
            {
                lidarPtSorted.push_back(lidar.x);
            }
        } 
        sort(lidarPtSorted.begin(), lidarPtSorted.end());

        
        // only use nearest k points to calculate average x distance
        int cutNum1 = lidarPtSorted.size()*cutRatioU;
        int cutNum2 = lidarPtSorted.size()*cutRatioL;
        int leftNum = lidarPtSorted.size()-cutNum1-cutNum2;
        dis = accumulate(lidarPtSorted.begin()+cutNum2, lidarPtSorted.end()-cutNum1, 0.0)/((double)leftNum);
        
        /*
        // calculate std
        double disSTD = 0.0;
        for (int i = 0; i<leftNum; i++)
        {
            disSTD = disSTD + (lidarPtSorted[i]-dis)*(lidarPtSorted[i]-dis);
        }
        disSTD = sqrt(disSTD/((double)leftNum));

        // update mean with std adjustment.
        // dis = lidarPtSorted[0];
        */

        // calculate dis by median value
        int start = cutNum2;
        if (leftNum%2!=0)
        {
            dis = lidarPtSorted[start+leftNum/2];
        }
        else
        {
            dis = (lidarPtSorted[start+leftNum/2] + lidarPtSorted[start+leftNum/2-1])/2;
        }
        cout<<"original lidar num = "<<nlidarPt<<", filter lidar num = "<<leftNum<<'\n';
    }   
    return dis;
}

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    double disCurr = 1e9;
    double disPrev = 1e9;
    double cutRatioU = 0.2;
    double cutRatioL = 0.1;
    int sizeThresh = 10;
    double tolY = 0.5;
    double tolZ = 0.2;
    double dt = 1.0/frameRate;
    // estimate distance
    disCurr = computeDisLidar(lidarPointsCurr, cutRatioL, cutRatioU, sizeThresh, tolY, tolZ);
    disPrev = computeDisLidar(lidarPointsPrev, cutRatioL, cutRatioU, sizeThresh, tolY, tolZ);

    cout<<"disPrev = "<<disPrev<<", disCurr = "<<disCurr<<'\n';
    // calculate TTC
    if (abs(disPrev - disCurr) < 0.0000001)
    {
        TTC = NAN;
    }
    else
    {
        TTC = disCurr * dt/(disPrev - disCurr);
    }
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    // use size of previous bounding box and size of current bounding box to create occurnace table.
    // initialize ocurrance table to zero
    int prevBboxNum = prevFrame.boundingBoxes.size();
    int currBboxNum = currFrame.boundingBoxes.size();
    vector<vector<int>> occurTable(prevBboxNum, vector<int> (currBboxNum, 0));

    // lopp through matches
    // extract keypoints in each match
    // find prevBbox ID that encloses previous matched keypoint in the match
    // find currBbox ID that encloses current matched keypoint in the match
    // update occurance table element at [prevBbox_ID][currBbox_ID)]
    // loop through row(id of prevBbox) in occruance table and 
    // find the column(ID of currBbox) that has highest ocurrance number.
    // update (row index, column index) into bbBestMatches

    for (auto match : matches)
    {
        // extract keypoints in each match
        cv::KeyPoint prevKpt = prevFrame.keypoints[match.queryIdx];
        cv::KeyPoint currKpt = currFrame.keypoints[match.trainIdx];

        // find prevBbox ID
        int prevBboxID = -1;
        for (auto Bbox : prevFrame.boundingBoxes)
        {
            if (Bbox.roi.contains(prevKpt.pt))
            {
                prevBboxID = Bbox.boxID;
            }
        }

        // find currBbox ID
        int currBboxID = -1;
        for (auto Bbox : currFrame.boundingBoxes)
        {
            if (Bbox.roi.contains(currKpt.pt))
            {
                currBboxID = Bbox.boxID;
            }
        }

        // update occurance table
        if (prevBboxID!=-1 && currBboxID!=-1)
        {
            occurTable[prevBboxID][currBboxID] = occurTable[prevBboxID][currBboxID] + 1;
        }
    }

    // find [row][column] combination which has highest occurance rate for each row
    // insert [row][column] into bbBestMatches
    // loop through each row
    for (int i = 0; i<prevBboxNum; i++)
    {
        // find column which has the highest ocurrance rate
        int maxOcurr = 0;
        int maxOcurrCol= 0;
        for (int j = 0; j<currBboxNum; j++)
        {
            if (occurTable[i][j]>maxOcurr)
            {
                maxOcurr = occurTable[i][j];
                maxOcurrCol = j;
            }
        }
        // update bbBestMatches when maxOcurr > 0;
        // if maxOcurr = 0, there is no match for this particular Bbox in prevFrame,
        // put -1 instead of maxOcurrCol which is the matched Bbox in currFrame
        if (maxOcurr > 0)
        {
            bbBestMatches.insert({i, maxOcurrCol});
        }
    }
}
