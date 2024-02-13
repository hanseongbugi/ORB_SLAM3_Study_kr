/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/


#include "Tracking.h"

#include "ORBmatcher.h"
#include "FrameDrawer.h"
#include "Converter.h"
#include "G2oTypes.h"
#include "Optimizer.h"
#include "Pinhole.h"
#include "KannalaBrandt8.h"
#include "MLPnPsolver.h"
#include "GeometricTools.h"

#include <iostream>

#include <mutex>
#include <chrono>


using namespace std;

namespace ORB_SLAM3
{


Tracking::Tracking(System *pSys, ORBVocabulary* pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Atlas *pAtlas, KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor, Settings* settings, const string &_nameSeq):
    mState(NO_IMAGES_YET), mSensor(sensor), mTrackedFr(0), mbStep(false),
    mbOnlyTracking(false), mbMapUpdated(false), mbVO(false), mpORBVocabulary(pVoc), mpKeyFrameDB(pKFDB),
    mbReadyToInitializate(false), mpSystem(pSys), mpViewer(NULL), bStepByStep(false),
    mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpAtlas(pAtlas), mnLastRelocFrameId(0), time_recently_lost(5.0),
    mnInitialFrameId(0), mbCreatedMap(false), mnFirstFrameId(0), mpCamera2(nullptr), mpLastKeyFrame(static_cast<KeyFrame*>(NULL))
{
    // Load camera parameters from settings file
    if(settings){
        newParameterLoader(settings);
    }
    else{
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

        bool b_parse_cam = ParseCamParamFile(fSettings);
        if(!b_parse_cam)
        {
            std::cout << "*Error with the camera parameters in the config file*" << std::endl;
        }

        // Load ORB parameters
        bool b_parse_orb = ParseORBParamFile(fSettings);
        if(!b_parse_orb)
        {
            std::cout << "*Error with the ORB parameters in the config file*" << std::endl;
        }

        bool b_parse_imu = true;
        if(sensor==System::IMU_MONOCULAR || sensor==System::IMU_STEREO || sensor==System::IMU_RGBD)
        {
            b_parse_imu = ParseIMUParamFile(fSettings);
            if(!b_parse_imu)
            {
                std::cout << "*Error with the IMU parameters in the config file*" << std::endl;
            }

            mnFramesToResetIMU = mMaxFrames;
        }

        if(!b_parse_cam || !b_parse_orb || !b_parse_imu)
        {
            std::cerr << "**ERROR in the config file, the format is not correct**" << std::endl;
            try
            {
                throw -1;
            }
            catch(exception &e)
            {

            }
        }
    }

    initID = 0; lastID = 0;
    mbInitWith3KFs = false;
    mnNumDataset = 0;

    vector<GeometricCamera*> vpCams = mpAtlas->GetAllCameras();
    std::cout << "There are " << vpCams.size() << " cameras in the atlas" << std::endl;
    for(GeometricCamera* pCam : vpCams)
    {
        std::cout << "Camera " << pCam->GetId();
        if(pCam->GetType() == GeometricCamera::CAM_PINHOLE)
        {
            std::cout << " is pinhole" << std::endl;
        }
        else if(pCam->GetType() == GeometricCamera::CAM_FISHEYE)
        {
            std::cout << " is fisheye" << std::endl;
        }
        else
        {
            std::cout << " is unknown" << std::endl;
        }
    }

#ifdef REGISTER_TIMES
    vdRectStereo_ms.clear();
    vdResizeImage_ms.clear();
    vdORBExtract_ms.clear();
    vdStereoMatch_ms.clear();
    vdIMUInteg_ms.clear();
    vdPosePred_ms.clear();
    vdLMTrack_ms.clear();
    vdNewKF_ms.clear();
    vdTrackTotal_ms.clear();
#endif
}

#ifdef REGISTER_TIMES
double calcAverage(vector<double> v_times)
{
    double accum = 0;
    for(double value : v_times)
    {
        accum += value;
    }

    return accum / v_times.size();
}

double calcDeviation(vector<double> v_times, double average)
{
    double accum = 0;
    for(double value : v_times)
    {
        accum += pow(value - average, 2);
    }
    return sqrt(accum / v_times.size());
}

double calcAverage(vector<int> v_values)
{
    double accum = 0;
    int total = 0;
    for(double value : v_values)
    {
        if(value == 0)
            continue;
        accum += value;
        total++;
    }

    return accum / total;
}

double calcDeviation(vector<int> v_values, double average)
{
    double accum = 0;
    int total = 0;
    for(double value : v_values)
    {
        if(value == 0)
            continue;
        accum += pow(value - average, 2);
        total++;
    }
    return sqrt(accum / total);
}

void Tracking::LocalMapStats2File()
{
    ofstream f;
    f.open("LocalMapTimeStats.txt");
    f << fixed << setprecision(6);
    f << "#Stereo rect[ms], MP culling[ms], MP creation[ms], LBA[ms], KF culling[ms], Total[ms]" << endl;
    for(int i=0; i<mpLocalMapper->vdLMTotal_ms.size(); ++i)
    {
        f << mpLocalMapper->vdKFInsert_ms[i] << "," << mpLocalMapper->vdMPCulling_ms[i] << ","
          << mpLocalMapper->vdMPCreation_ms[i] << "," << mpLocalMapper->vdLBASync_ms[i] << ","
          << mpLocalMapper->vdKFCullingSync_ms[i] <<  "," << mpLocalMapper->vdLMTotal_ms[i] << endl;
    }

    f.close();

    f.open("LBA_Stats.txt");
    f << fixed << setprecision(6);
    f << "#LBA time[ms], KF opt[#], KF fixed[#], MP[#], Edges[#]" << endl;
    for(int i=0; i<mpLocalMapper->vdLBASync_ms.size(); ++i)
    {
        f << mpLocalMapper->vdLBASync_ms[i] << "," << mpLocalMapper->vnLBA_KFopt[i] << ","
          << mpLocalMapper->vnLBA_KFfixed[i] << "," << mpLocalMapper->vnLBA_MPs[i] << ","
          << mpLocalMapper->vnLBA_edges[i] << endl;
    }


    f.close();
}

void Tracking::TrackStats2File()
{
    ofstream f;
    f.open("SessionInfo.txt");
    f << fixed;
    f << "Number of KFs: " << mpAtlas->GetAllKeyFrames().size() << endl;
    f << "Number of MPs: " << mpAtlas->GetAllMapPoints().size() << endl;

    f << "OpenCV version: " << CV_VERSION << endl;

    f.close();

    f.open("TrackingTimeStats.txt");
    f << fixed << setprecision(6);

    f << "#Image Rect[ms], Image Resize[ms], ORB ext[ms], Stereo match[ms], IMU preint[ms], Pose pred[ms], LM track[ms], KF dec[ms], Total[ms]" << endl;

    for(int i=0; i<vdTrackTotal_ms.size(); ++i)
    {
        double stereo_rect = 0.0;
        if(!vdRectStereo_ms.empty())
        {
            stereo_rect = vdRectStereo_ms[i];
        }

        double resize_image = 0.0;
        if(!vdResizeImage_ms.empty())
        {
            resize_image = vdResizeImage_ms[i];
        }

        double stereo_match = 0.0;
        if(!vdStereoMatch_ms.empty())
        {
            stereo_match = vdStereoMatch_ms[i];
        }

        double imu_preint = 0.0;
        if(!vdIMUInteg_ms.empty())
        {
            imu_preint = vdIMUInteg_ms[i];
        }

        f << stereo_rect << "," << resize_image << "," << vdORBExtract_ms[i] << "," << stereo_match << "," << imu_preint << ","
          << vdPosePred_ms[i] <<  "," << vdLMTrack_ms[i] << "," << vdNewKF_ms[i] << "," << vdTrackTotal_ms[i] << endl;
    }

    f.close();
}

void Tracking::PrintTimeStats()
{
    // Save data in files
    TrackStats2File();
    LocalMapStats2File();


    ofstream f;
    f.open("ExecMean.txt");
    f << fixed;
    //Report the mean and std of each one
    std::cout << std::endl << " TIME STATS in ms (mean$\\pm$std)" << std::endl;
    f << " TIME STATS in ms (mean$\\pm$std)" << std::endl;
    cout << "OpenCV version: " << CV_VERSION << endl;
    f << "OpenCV version: " << CV_VERSION << endl;
    std::cout << "---------------------------" << std::endl;
    std::cout << "Tracking" << std::setprecision(5) << std::endl << std::endl;
    f << "---------------------------" << std::endl;
    f << "Tracking" << std::setprecision(5) << std::endl << std::endl;
    double average, deviation;
    if(!vdRectStereo_ms.empty())
    {
        average = calcAverage(vdRectStereo_ms);
        deviation = calcDeviation(vdRectStereo_ms, average);
        std::cout << "Stereo Rectification: " << average << "$\\pm$" << deviation << std::endl;
        f << "Stereo Rectification: " << average << "$\\pm$" << deviation << std::endl;
    }

    if(!vdResizeImage_ms.empty())
    {
        average = calcAverage(vdResizeImage_ms);
        deviation = calcDeviation(vdResizeImage_ms, average);
        std::cout << "Image Resize: " << average << "$\\pm$" << deviation << std::endl;
        f << "Image Resize: " << average << "$\\pm$" << deviation << std::endl;
    }

    average = calcAverage(vdORBExtract_ms);
    deviation = calcDeviation(vdORBExtract_ms, average);
    std::cout << "ORB Extraction: " << average << "$\\pm$" << deviation << std::endl;
    f << "ORB Extraction: " << average << "$\\pm$" << deviation << std::endl;

    if(!vdStereoMatch_ms.empty())
    {
        average = calcAverage(vdStereoMatch_ms);
        deviation = calcDeviation(vdStereoMatch_ms, average);
        std::cout << "Stereo Matching: " << average << "$\\pm$" << deviation << std::endl;
        f << "Stereo Matching: " << average << "$\\pm$" << deviation << std::endl;
    }

    if(!vdIMUInteg_ms.empty())
    {
        average = calcAverage(vdIMUInteg_ms);
        deviation = calcDeviation(vdIMUInteg_ms, average);
        std::cout << "IMU Preintegration: " << average << "$\\pm$" << deviation << std::endl;
        f << "IMU Preintegration: " << average << "$\\pm$" << deviation << std::endl;
    }

    average = calcAverage(vdPosePred_ms);
    deviation = calcDeviation(vdPosePred_ms, average);
    std::cout << "Pose Prediction: " << average << "$\\pm$" << deviation << std::endl;
    f << "Pose Prediction: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(vdLMTrack_ms);
    deviation = calcDeviation(vdLMTrack_ms, average);
    std::cout << "LM Track: " << average << "$\\pm$" << deviation << std::endl;
    f << "LM Track: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(vdNewKF_ms);
    deviation = calcDeviation(vdNewKF_ms, average);
    std::cout << "New KF decision: " << average << "$\\pm$" << deviation << std::endl;
    f << "New KF decision: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(vdTrackTotal_ms);
    deviation = calcDeviation(vdTrackTotal_ms, average);
    std::cout << "Total Tracking: " << average << "$\\pm$" << deviation << std::endl;
    f << "Total Tracking: " << average << "$\\pm$" << deviation << std::endl;

    // Local Mapping time stats
    std::cout << std::endl << std::endl << std::endl;
    std::cout << "Local Mapping" << std::endl << std::endl;
    f << std::endl << "Local Mapping" << std::endl << std::endl;

    average = calcAverage(mpLocalMapper->vdKFInsert_ms);
    deviation = calcDeviation(mpLocalMapper->vdKFInsert_ms, average);
    std::cout << "KF Insertion: " << average << "$\\pm$" << deviation << std::endl;
    f << "KF Insertion: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdMPCulling_ms);
    deviation = calcDeviation(mpLocalMapper->vdMPCulling_ms, average);
    std::cout << "MP Culling: " << average << "$\\pm$" << deviation << std::endl;
    f << "MP Culling: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdMPCreation_ms);
    deviation = calcDeviation(mpLocalMapper->vdMPCreation_ms, average);
    std::cout << "MP Creation: " << average << "$\\pm$" << deviation << std::endl;
    f << "MP Creation: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdLBA_ms);
    deviation = calcDeviation(mpLocalMapper->vdLBA_ms, average);
    std::cout << "LBA: " << average << "$\\pm$" << deviation << std::endl;
    f << "LBA: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdKFCulling_ms);
    deviation = calcDeviation(mpLocalMapper->vdKFCulling_ms, average);
    std::cout << "KF Culling: " << average << "$\\pm$" << deviation << std::endl;
    f << "KF Culling: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdLMTotal_ms);
    deviation = calcDeviation(mpLocalMapper->vdLMTotal_ms, average);
    std::cout << "Total Local Mapping: " << average << "$\\pm$" << deviation << std::endl;
    f << "Total Local Mapping: " << average << "$\\pm$" << deviation << std::endl;

    // Local Mapping LBA complexity
    std::cout << "---------------------------" << std::endl;
    std::cout << std::endl << "LBA complexity (mean$\\pm$std)" << std::endl;
    f << "---------------------------" << std::endl;
    f << std::endl << "LBA complexity (mean$\\pm$std)" << std::endl;

    average = calcAverage(mpLocalMapper->vnLBA_edges);
    deviation = calcDeviation(mpLocalMapper->vnLBA_edges, average);
    std::cout << "LBA Edges: " << average << "$\\pm$" << deviation << std::endl;
    f << "LBA Edges: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vnLBA_KFopt);
    deviation = calcDeviation(mpLocalMapper->vnLBA_KFopt, average);
    std::cout << "LBA KF optimized: " << average << "$\\pm$" << deviation << std::endl;
    f << "LBA KF optimized: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vnLBA_KFfixed);
    deviation = calcDeviation(mpLocalMapper->vnLBA_KFfixed, average);
    std::cout << "LBA KF fixed: " << average << "$\\pm$" << deviation << std::endl;
    f << "LBA KF fixed: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vnLBA_MPs);
    deviation = calcDeviation(mpLocalMapper->vnLBA_MPs, average);
    std::cout << "LBA MP: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    f << "LBA MP: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    std::cout << "LBA executions: " << mpLocalMapper->nLBA_exec << std::endl;
    std::cout << "LBA aborts: " << mpLocalMapper->nLBA_abort << std::endl;
    f << "LBA executions: " << mpLocalMapper->nLBA_exec << std::endl;
    f << "LBA aborts: " << mpLocalMapper->nLBA_abort << std::endl;

    // Map complexity
    std::cout << "---------------------------" << std::endl;
    std::cout << std::endl << "Map complexity" << std::endl;
    std::cout << "KFs in map: " << mpAtlas->GetAllKeyFrames().size() << std::endl;
    std::cout << "MPs in map: " << mpAtlas->GetAllMapPoints().size() << std::endl;
    f << "---------------------------" << std::endl;
    f << std::endl << "Map complexity" << std::endl;
    vector<Map*> vpMaps = mpAtlas->GetAllMaps();
    Map* pBestMap = vpMaps[0];
    for(int i=1; i<vpMaps.size(); ++i)
    {
        if(pBestMap->GetAllKeyFrames().size() < vpMaps[i]->GetAllKeyFrames().size())
        {
            pBestMap = vpMaps[i];
        }
    }

    f << "KFs in map: " << pBestMap->GetAllKeyFrames().size() << std::endl;
    f << "MPs in map: " << pBestMap->GetAllMapPoints().size() << std::endl;

    f << "---------------------------" << std::endl;
    f << std::endl << "Place Recognition (mean$\\pm$std)" << std::endl;
    std::cout << "---------------------------" << std::endl;
    std::cout << std::endl << "Place Recognition (mean$\\pm$std)" << std::endl;
    average = calcAverage(mpLoopClosing->vdDataQuery_ms);
    deviation = calcDeviation(mpLoopClosing->vdDataQuery_ms, average);
    f << "Database Query: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Database Query: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdEstSim3_ms);
    deviation = calcDeviation(mpLoopClosing->vdEstSim3_ms, average);
    f << "SE3 estimation: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "SE3 estimation: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdPRTotal_ms);
    deviation = calcDeviation(mpLoopClosing->vdPRTotal_ms, average);
    f << "Total Place Recognition: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    std::cout << "Total Place Recognition: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    f << std::endl << "Loop Closing (mean$\\pm$std)" << std::endl;
    std::cout << std::endl << "Loop Closing (mean$\\pm$std)" << std::endl;
    average = calcAverage(mpLoopClosing->vdLoopFusion_ms);
    deviation = calcDeviation(mpLoopClosing->vdLoopFusion_ms, average);
    f << "Loop Fusion: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Loop Fusion: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdLoopOptEss_ms);
    deviation = calcDeviation(mpLoopClosing->vdLoopOptEss_ms, average);
    f << "Essential Graph: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Essential Graph: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdLoopTotal_ms);
    deviation = calcDeviation(mpLoopClosing->vdLoopTotal_ms, average);
    f << "Total Loop Closing: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    std::cout << "Total Loop Closing: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    f << "Numb exec: " << mpLoopClosing->nLoop << std::endl;
    std::cout << "Num exec: " << mpLoopClosing->nLoop << std::endl;
    average = calcAverage(mpLoopClosing->vnLoopKFs);
    deviation = calcDeviation(mpLoopClosing->vnLoopKFs, average);
    f << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;

    f << std::endl << "Map Merging (mean$\\pm$std)" << std::endl;
    std::cout << std::endl << "Map Merging (mean$\\pm$std)" << std::endl;
    average = calcAverage(mpLoopClosing->vdMergeMaps_ms);
    deviation = calcDeviation(mpLoopClosing->vdMergeMaps_ms, average);
    f << "Merge Maps: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Merge Maps: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdWeldingBA_ms);
    deviation = calcDeviation(mpLoopClosing->vdWeldingBA_ms, average);
    f << "Welding BA: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Welding BA: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdMergeOptEss_ms);
    deviation = calcDeviation(mpLoopClosing->vdMergeOptEss_ms, average);
    f << "Optimization Ess.: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Optimization Ess.: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdMergeTotal_ms);
    deviation = calcDeviation(mpLoopClosing->vdMergeTotal_ms, average);
    f << "Total Map Merging: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    std::cout << "Total Map Merging: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    f << "Numb exec: " << mpLoopClosing->nMerges << std::endl;
    std::cout << "Num exec: " << mpLoopClosing->nMerges << std::endl;
    average = calcAverage(mpLoopClosing->vnMergeKFs);
    deviation = calcDeviation(mpLoopClosing->vnMergeKFs, average);
    f << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vnMergeMPs);
    deviation = calcDeviation(mpLoopClosing->vnMergeMPs, average);
    f << "Number of MPs: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Number of MPs: " << average << "$\\pm$" << deviation << std::endl;

    f << std::endl << "Full GBA (mean$\\pm$std)" << std::endl;
    std::cout << std::endl << "Full GBA (mean$\\pm$std)" << std::endl;
    average = calcAverage(mpLoopClosing->vdGBA_ms);
    deviation = calcDeviation(mpLoopClosing->vdGBA_ms, average);
    f << "GBA: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "GBA: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdUpdateMap_ms);
    deviation = calcDeviation(mpLoopClosing->vdUpdateMap_ms, average);
    f << "Map Update: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Map Update: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdFGBATotal_ms);
    deviation = calcDeviation(mpLoopClosing->vdFGBATotal_ms, average);
    f << "Total Full GBA: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    std::cout << "Total Full GBA: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    f << "Numb exec: " << mpLoopClosing->nFGBA_exec << std::endl;
    std::cout << "Num exec: " << mpLoopClosing->nFGBA_exec << std::endl;
    f << "Numb abort: " << mpLoopClosing->nFGBA_abort << std::endl;
    std::cout << "Num abort: " << mpLoopClosing->nFGBA_abort << std::endl;
    average = calcAverage(mpLoopClosing->vnGBAKFs);
    deviation = calcDeviation(mpLoopClosing->vnGBAKFs, average);
    f << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vnGBAMPs);
    deviation = calcDeviation(mpLoopClosing->vnGBAMPs, average);
    f << "Number of MPs: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Number of MPs: " << average << "$\\pm$" << deviation << std::endl;

    f.close();

}

#endif

Tracking::~Tracking()
{
    //f_track_stats.close();

}

void Tracking::newParameterLoader(Settings *settings) {
    mpCamera = settings->camera1();
    mpCamera = mpAtlas->AddCamera(mpCamera);

    if(settings->needToUndistort()){
        mDistCoef = settings->camera1DistortionCoef();
    }
    else{
        mDistCoef = cv::Mat::zeros(4,1,CV_32F);
    }

    //TODO: missing image scaling and rectification
    mImageScale = 1.0f;

    mK = cv::Mat::eye(3,3,CV_32F);
    mK.at<float>(0,0) = mpCamera->getParameter(0);
    mK.at<float>(1,1) = mpCamera->getParameter(1);
    mK.at<float>(0,2) = mpCamera->getParameter(2);
    mK.at<float>(1,2) = mpCamera->getParameter(3);

    mK_.setIdentity();
    mK_(0,0) = mpCamera->getParameter(0);
    mK_(1,1) = mpCamera->getParameter(1);
    mK_(0,2) = mpCamera->getParameter(2);
    mK_(1,2) = mpCamera->getParameter(3);

    if((mSensor==System::STEREO || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD) &&
        settings->cameraType() == Settings::KannalaBrandt){
        mpCamera2 = settings->camera2();
        mpCamera2 = mpAtlas->AddCamera(mpCamera2);

        mTlr = settings->Tlr();

        mpFrameDrawer->both = true;
    }

    if(mSensor==System::STEREO || mSensor==System::RGBD || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD ){
        mbf = settings->bf();
        mThDepth = settings->b() * settings->thDepth();
    }

    if(mSensor==System::RGBD || mSensor==System::IMU_RGBD){
        mDepthMapFactor = settings->depthMapFactor();
        if(fabs(mDepthMapFactor)<1e-5)
            mDepthMapFactor=1;
        else
            mDepthMapFactor = 1.0f/mDepthMapFactor;
    }

    mMinFrames = 0;
    mMaxFrames = settings->fps();
    mbRGB = settings->rgb();

    //ORB parameters
    int nFeatures = settings->nFeatures();
    int nLevels = settings->nLevels();
    int fIniThFAST = settings->initThFAST();
    int fMinThFAST = settings->minThFAST();
    float fScaleFactor = settings->scaleFactor();

    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(mSensor==System::STEREO || mSensor==System::IMU_STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR)
        mpIniORBextractor = new ORBextractor(5*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    //IMU parameters
    Sophus::SE3f Tbc = settings->Tbc();
    mInsertKFsLost = settings->insertKFsWhenLost();
    mImuFreq = settings->imuFrequency();
    mImuPer = 0.001; //1.0 / (double) mImuFreq;     //TODO: ESTO ESTA BIEN?
    float Ng = settings->noiseGyro();
    float Na = settings->noiseAcc();
    float Ngw = settings->gyroWalk();
    float Naw = settings->accWalk();

    const float sf = sqrt(mImuFreq);
    mpImuCalib = new IMU::Calib(Tbc,Ng*sf,Na*sf,Ngw/sf,Naw/sf);

    mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);
}

bool Tracking::ParseCamParamFile(cv::FileStorage &fSettings)
{
    mDistCoef = cv::Mat::zeros(4,1,CV_32F);
    cout << endl << "Camera Parameters: " << endl;
    bool b_miss_params = false;

    string sCameraName = fSettings["Camera.type"];
    if(sCameraName == "PinHole")
    {
        float fx, fy, cx, cy;
        mImageScale = 1.f;

        // Camera calibration parameters
        cv::FileNode node = fSettings["Camera.fx"];
        if(!node.empty() && node.isReal())
        {
            fx = node.real();
        }
        else
        {
            std::cerr << "*Camera.fx parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.fy"];
        if(!node.empty() && node.isReal())
        {
            fy = node.real();
        }
        else
        {
            std::cerr << "*Camera.fy parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.cx"];
        if(!node.empty() && node.isReal())
        {
            cx = node.real();
        }
        else
        {
            std::cerr << "*Camera.cx parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.cy"];
        if(!node.empty() && node.isReal())
        {
            cy = node.real();
        }
        else
        {
            std::cerr << "*Camera.cy parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        // Distortion parameters
        node = fSettings["Camera.k1"];
        if(!node.empty() && node.isReal())
        {
            mDistCoef.at<float>(0) = node.real();
        }
        else
        {
            std::cerr << "*Camera.k1 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.k2"];
        if(!node.empty() && node.isReal())
        {
            mDistCoef.at<float>(1) = node.real();
        }
        else
        {
            std::cerr << "*Camera.k2 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.p1"];
        if(!node.empty() && node.isReal())
        {
            mDistCoef.at<float>(2) = node.real();
        }
        else
        {
            std::cerr << "*Camera.p1 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.p2"];
        if(!node.empty() && node.isReal())
        {
            mDistCoef.at<float>(3) = node.real();
        }
        else
        {
            std::cerr << "*Camera.p2 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.k3"];
        if(!node.empty() && node.isReal())
        {
            mDistCoef.resize(5);
            mDistCoef.at<float>(4) = node.real();
        }

        node = fSettings["Camera.imageScale"];
        if(!node.empty() && node.isReal())
        {
            mImageScale = node.real();
        }

        if(b_miss_params)
        {
            return false;
        }

        if(mImageScale != 1.f)
        {
            // K matrix parameters must be scaled.
            fx = fx * mImageScale;
            fy = fy * mImageScale;
            cx = cx * mImageScale;
            cy = cy * mImageScale;
        }

        vector<float> vCamCalib{fx,fy,cx,cy};

        mpCamera = new Pinhole(vCamCalib);

        mpCamera = mpAtlas->AddCamera(mpCamera);

        std::cout << "- Camera: Pinhole" << std::endl;
        std::cout << "- Image scale: " << mImageScale << std::endl;
        std::cout << "- fx: " << fx << std::endl;
        std::cout << "- fy: " << fy << std::endl;
        std::cout << "- cx: " << cx << std::endl;
        std::cout << "- cy: " << cy << std::endl;
        std::cout << "- k1: " << mDistCoef.at<float>(0) << std::endl;
        std::cout << "- k2: " << mDistCoef.at<float>(1) << std::endl;


        std::cout << "- p1: " << mDistCoef.at<float>(2) << std::endl;
        std::cout << "- p2: " << mDistCoef.at<float>(3) << std::endl;

        if(mDistCoef.rows==5)
            std::cout << "- k3: " << mDistCoef.at<float>(4) << std::endl;

        mK = cv::Mat::eye(3,3,CV_32F);
        mK.at<float>(0,0) = fx;
        mK.at<float>(1,1) = fy;
        mK.at<float>(0,2) = cx;
        mK.at<float>(1,2) = cy;

        mK_.setIdentity();
        mK_(0,0) = fx;
        mK_(1,1) = fy;
        mK_(0,2) = cx;
        mK_(1,2) = cy;
    }
    else if(sCameraName == "KannalaBrandt8")
    {
        float fx, fy, cx, cy;
        float k1, k2, k3, k4;
        mImageScale = 1.f;

        // Camera calibration parameters
        cv::FileNode node = fSettings["Camera.fx"];
        if(!node.empty() && node.isReal())
        {
            fx = node.real();
        }
        else
        {
            std::cerr << "*Camera.fx parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }
        node = fSettings["Camera.fy"];
        if(!node.empty() && node.isReal())
        {
            fy = node.real();
        }
        else
        {
            std::cerr << "*Camera.fy parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.cx"];
        if(!node.empty() && node.isReal())
        {
            cx = node.real();
        }
        else
        {
            std::cerr << "*Camera.cx parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.cy"];
        if(!node.empty() && node.isReal())
        {
            cy = node.real();
        }
        else
        {
            std::cerr << "*Camera.cy parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        // Distortion parameters
        node = fSettings["Camera.k1"];
        if(!node.empty() && node.isReal())
        {
            k1 = node.real();
        }
        else
        {
            std::cerr << "*Camera.k1 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }
        node = fSettings["Camera.k2"];
        if(!node.empty() && node.isReal())
        {
            k2 = node.real();
        }
        else
        {
            std::cerr << "*Camera.k2 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.k3"];
        if(!node.empty() && node.isReal())
        {
            k3 = node.real();
        }
        else
        {
            std::cerr << "*Camera.k3 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.k4"];
        if(!node.empty() && node.isReal())
        {
            k4 = node.real();
        }
        else
        {
            std::cerr << "*Camera.k4 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.imageScale"];
        if(!node.empty() && node.isReal())
        {
            mImageScale = node.real();
        }

        if(!b_miss_params)
        {
            if(mImageScale != 1.f)
            {
                // K matrix parameters must be scaled.
                fx = fx * mImageScale;
                fy = fy * mImageScale;
                cx = cx * mImageScale;
                cy = cy * mImageScale;
            }

            vector<float> vCamCalib{fx,fy,cx,cy,k1,k2,k3,k4};
            mpCamera = new KannalaBrandt8(vCamCalib);
            mpCamera = mpAtlas->AddCamera(mpCamera);
            std::cout << "- Camera: Fisheye" << std::endl;
            std::cout << "- Image scale: " << mImageScale << std::endl;
            std::cout << "- fx: " << fx << std::endl;
            std::cout << "- fy: " << fy << std::endl;
            std::cout << "- cx: " << cx << std::endl;
            std::cout << "- cy: " << cy << std::endl;
            std::cout << "- k1: " << k1 << std::endl;
            std::cout << "- k2: " << k2 << std::endl;
            std::cout << "- k3: " << k3 << std::endl;
            std::cout << "- k4: " << k4 << std::endl;

            mK = cv::Mat::eye(3,3,CV_32F);
            mK.at<float>(0,0) = fx;
            mK.at<float>(1,1) = fy;
            mK.at<float>(0,2) = cx;
            mK.at<float>(1,2) = cy;

            mK_.setIdentity();
            mK_(0,0) = fx;
            mK_(1,1) = fy;
            mK_(0,2) = cx;
            mK_(1,2) = cy;
        }

        if(mSensor==System::STEREO || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD){
            // Right camera
            // Camera calibration parameters
            cv::FileNode node = fSettings["Camera2.fx"];
            if(!node.empty() && node.isReal())
            {
                fx = node.real();
            }
            else
            {
                std::cerr << "*Camera2.fx parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }
            node = fSettings["Camera2.fy"];
            if(!node.empty() && node.isReal())
            {
                fy = node.real();
            }
            else
            {
                std::cerr << "*Camera2.fy parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera2.cx"];
            if(!node.empty() && node.isReal())
            {
                cx = node.real();
            }
            else
            {
                std::cerr << "*Camera2.cx parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera2.cy"];
            if(!node.empty() && node.isReal())
            {
                cy = node.real();
            }
            else
            {
                std::cerr << "*Camera2.cy parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            // Distortion parameters
            node = fSettings["Camera2.k1"];
            if(!node.empty() && node.isReal())
            {
                k1 = node.real();
            }
            else
            {
                std::cerr << "*Camera2.k1 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }
            node = fSettings["Camera2.k2"];
            if(!node.empty() && node.isReal())
            {
                k2 = node.real();
            }
            else
            {
                std::cerr << "*Camera2.k2 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera2.k3"];
            if(!node.empty() && node.isReal())
            {
                k3 = node.real();
            }
            else
            {
                std::cerr << "*Camera2.k3 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera2.k4"];
            if(!node.empty() && node.isReal())
            {
                k4 = node.real();
            }
            else
            {
                std::cerr << "*Camera2.k4 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }


            int leftLappingBegin = -1;
            int leftLappingEnd = -1;

            int rightLappingBegin = -1;
            int rightLappingEnd = -1;

            node = fSettings["Camera.lappingBegin"];
            if(!node.empty() && node.isInt())
            {
                leftLappingBegin = node.operator int();
            }
            else
            {
                std::cout << "WARNING: Camera.lappingBegin not correctly defined" << std::endl;
            }
            node = fSettings["Camera.lappingEnd"];
            if(!node.empty() && node.isInt())
            {
                leftLappingEnd = node.operator int();
            }
            else
            {
                std::cout << "WARNING: Camera.lappingEnd not correctly defined" << std::endl;
            }
            node = fSettings["Camera2.lappingBegin"];
            if(!node.empty() && node.isInt())
            {
                rightLappingBegin = node.operator int();
            }
            else
            {
                std::cout << "WARNING: Camera2.lappingBegin not correctly defined" << std::endl;
            }
            node = fSettings["Camera2.lappingEnd"];
            if(!node.empty() && node.isInt())
            {
                rightLappingEnd = node.operator int();
            }
            else
            {
                std::cout << "WARNING: Camera2.lappingEnd not correctly defined" << std::endl;
            }

            node = fSettings["Tlr"];
            cv::Mat cvTlr;
            if(!node.empty())
            {
                cvTlr = node.mat();
                if(cvTlr.rows != 3 || cvTlr.cols != 4)
                {
                    std::cerr << "*Tlr matrix have to be a 3x4 transformation matrix*" << std::endl;
                    b_miss_params = true;
                }
            }
            else
            {
                std::cerr << "*Tlr matrix doesn't exist*" << std::endl;
                b_miss_params = true;
            }

            if(!b_miss_params)
            {
                if(mImageScale != 1.f)
                {
                    // K matrix parameters must be scaled.
                    fx = fx * mImageScale;
                    fy = fy * mImageScale;
                    cx = cx * mImageScale;
                    cy = cy * mImageScale;

                    leftLappingBegin = leftLappingBegin * mImageScale;
                    leftLappingEnd = leftLappingEnd * mImageScale;
                    rightLappingBegin = rightLappingBegin * mImageScale;
                    rightLappingEnd = rightLappingEnd * mImageScale;
                }

                static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[0] = leftLappingBegin;
                static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[1] = leftLappingEnd;

                mpFrameDrawer->both = true;

                vector<float> vCamCalib2{fx,fy,cx,cy,k1,k2,k3,k4};
                mpCamera2 = new KannalaBrandt8(vCamCalib2);
                mpCamera2 = mpAtlas->AddCamera(mpCamera2);

                mTlr = Converter::toSophus(cvTlr);

                static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[0] = rightLappingBegin;
                static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[1] = rightLappingEnd;

                std::cout << "- Camera1 Lapping: " << leftLappingBegin << ", " << leftLappingEnd << std::endl;

                std::cout << std::endl << "Camera2 Parameters:" << std::endl;
                std::cout << "- Camera: Fisheye" << std::endl;
                std::cout << "- Image scale: " << mImageScale << std::endl;
                std::cout << "- fx: " << fx << std::endl;
                std::cout << "- fy: " << fy << std::endl;
                std::cout << "- cx: " << cx << std::endl;
                std::cout << "- cy: " << cy << std::endl;
                std::cout << "- k1: " << k1 << std::endl;
                std::cout << "- k2: " << k2 << std::endl;
                std::cout << "- k3: " << k3 << std::endl;
                std::cout << "- k4: " << k4 << std::endl;

                std::cout << "- mTlr: \n" << cvTlr << std::endl;

                std::cout << "- Camera2 Lapping: " << rightLappingBegin << ", " << rightLappingEnd << std::endl;
            }
        }

        if(b_miss_params)
        {
            return false;
        }

    }
    else
    {
        std::cerr << "*Not Supported Camera Sensor*" << std::endl;
        std::cerr << "Check an example configuration file with the desired sensor" << std::endl;
    }

    if(mSensor==System::STEREO || mSensor==System::RGBD || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD )
    {
        cv::FileNode node = fSettings["Camera.bf"];
        if(!node.empty() && node.isReal())
        {
            mbf = node.real();
            if(mImageScale != 1.f)
            {
                mbf *= mImageScale;
            }
        }
        else
        {
            std::cerr << "*Camera.bf parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

    }

    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << "- fps: " << fps << endl;


    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    if(mSensor==System::STEREO || mSensor==System::RGBD || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD)
    {
        float fx = mpCamera->getParameter(0);
        cv::FileNode node = fSettings["ThDepth"];
        if(!node.empty()  && node.isReal())
        {
            mThDepth = node.real();
            mThDepth = mbf*mThDepth/fx;
            cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
        }
        else
        {
            std::cerr << "*ThDepth parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }


    }

    if(mSensor==System::RGBD || mSensor==System::IMU_RGBD)
    {
        cv::FileNode node = fSettings["DepthMapFactor"];
        if(!node.empty() && node.isReal())
        {
            mDepthMapFactor = node.real();
            if(fabs(mDepthMapFactor)<1e-5)
                mDepthMapFactor=1;
            else
                mDepthMapFactor = 1.0f/mDepthMapFactor;
        }
        else
        {
            std::cerr << "*DepthMapFactor parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

    }

    if(b_miss_params)
    {
        return false;
    }

    return true;
}

bool Tracking::ParseORBParamFile(cv::FileStorage &fSettings)
{
    bool b_miss_params = false;
    int nFeatures, nLevels, fIniThFAST, fMinThFAST;
    float fScaleFactor;

    cv::FileNode node = fSettings["ORBextractor.nFeatures"];
    if(!node.empty() && node.isInt())
    {
        nFeatures = node.operator int();
    }
    else
    {
        std::cerr << "*ORBextractor.nFeatures parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["ORBextractor.scaleFactor"];
    if(!node.empty() && node.isReal())
    {
        fScaleFactor = node.real();
    }
    else
    {
        std::cerr << "*ORBextractor.scaleFactor parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["ORBextractor.nLevels"];
    if(!node.empty() && node.isInt())
    {
        nLevels = node.operator int();
    }
    else
    {
        std::cerr << "*ORBextractor.nLevels parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["ORBextractor.iniThFAST"];
    if(!node.empty() && node.isInt())
    {
        fIniThFAST = node.operator int();
    }
    else
    {
        std::cerr << "*ORBextractor.iniThFAST parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["ORBextractor.minThFAST"];
    if(!node.empty() && node.isInt())
    {
        fMinThFAST = node.operator int();
    }
    else
    {
        std::cerr << "*ORBextractor.minThFAST parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    if(b_miss_params)
    {
        return false;
    }

    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(mSensor==System::STEREO || mSensor==System::IMU_STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR)
        mpIniORBextractor = new ORBextractor(5*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    cout << endl << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    return true;
}

bool Tracking::ParseIMUParamFile(cv::FileStorage &fSettings)
{
    bool b_miss_params = false;

    cv::Mat cvTbc;
    cv::FileNode node = fSettings["Tbc"];
    if(!node.empty())
    {
        cvTbc = node.mat();
        if(cvTbc.rows != 4 || cvTbc.cols != 4)
        {
            std::cerr << "*Tbc matrix have to be a 4x4 transformation matrix*" << std::endl;
            b_miss_params = true;
        }
    }
    else
    {
        std::cerr << "*Tbc matrix doesn't exist*" << std::endl;
        b_miss_params = true;
    }
    cout << endl;
    cout << "Left camera to Imu Transform (Tbc): " << endl << cvTbc << endl;
    Eigen::Matrix<float,4,4,Eigen::RowMajor> eigTbc(cvTbc.ptr<float>(0));
    Sophus::SE3f Tbc(eigTbc);

    node = fSettings["InsertKFsWhenLost"];
    mInsertKFsLost = true;
    if(!node.empty() && node.isInt())
    {
        mInsertKFsLost = (bool) node.operator int();
    }

    if(!mInsertKFsLost)
        cout << "Do not insert keyframes when lost visual tracking " << endl;



    float Ng, Na, Ngw, Naw;

    node = fSettings["IMU.Frequency"];
    if(!node.empty() && node.isInt())
    {
        mImuFreq = node.operator int();
        mImuPer = 0.001; //1.0 / (double) mImuFreq;
    }
    else
    {
        std::cerr << "*IMU.Frequency parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.NoiseGyro"];
    if(!node.empty() && node.isReal())
    {
        Ng = node.real();
    }
    else
    {
        std::cerr << "*IMU.NoiseGyro parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.NoiseAcc"];
    if(!node.empty() && node.isReal())
    {
        Na = node.real();
    }
    else
    {
        std::cerr << "*IMU.NoiseAcc parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.GyroWalk"];
    if(!node.empty() && node.isReal())
    {
        Ngw = node.real();
    }
    else
    {
        std::cerr << "*IMU.GyroWalk parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.AccWalk"];
    if(!node.empty() && node.isReal())
    {
        Naw = node.real();
    }
    else
    {
        std::cerr << "*IMU.AccWalk parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.fastInit"];
    mFastInit = false;
    if(!node.empty())
    {
        mFastInit = static_cast<int>(fSettings["IMU.fastInit"]) != 0;
    }

    if(mFastInit)
        cout << "Fast IMU initialization. Acceleration is not checked \n";

    if(b_miss_params)
    {
        return false;
    }

    const float sf = sqrt(mImuFreq);
    cout << endl;
    cout << "IMU frequency: " << mImuFreq << " Hz" << endl;
    cout << "IMU gyro noise: " << Ng << " rad/s/sqrt(Hz)" << endl;
    cout << "IMU gyro walk: " << Ngw << " rad/s^2/sqrt(Hz)" << endl;
    cout << "IMU accelerometer noise: " << Na << " m/s^2/sqrt(Hz)" << endl;
    cout << "IMU accelerometer walk: " << Naw << " m/s^3/sqrt(Hz)" << endl;

    mpImuCalib = new IMU::Calib(Tbc,Ng*sf,Na*sf,Ngw/sf,Naw/sf);

    mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);


    return true;
}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer=pViewer;
}

void Tracking::SetStepByStep(bool bSet)
{
    bStepByStep = bSet;
}

bool Tracking::GetStepByStep()
{
    return bStepByStep;
}



Sophus::SE3f Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp, string filename)
{
    //cout << "GrabImageStereo" << endl;

    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;
    mImRight = imRectRight;

    if(mImGray.channels()==3)
    {
        //cout << "Image with 3 channels" << endl;
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,cv::COLOR_RGB2GRAY);
            cvtColor(imGrayRight,imGrayRight,cv::COLOR_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,cv::COLOR_BGR2GRAY);
            cvtColor(imGrayRight,imGrayRight,cv::COLOR_BGR2GRAY);
        }
    }
    else if(mImGray.channels()==4)
    {
        //cout << "Image with 4 channels" << endl;
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,cv::COLOR_RGBA2GRAY);
            cvtColor(imGrayRight,imGrayRight,cv::COLOR_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,cv::COLOR_BGRA2GRAY);
            cvtColor(imGrayRight,imGrayRight,cv::COLOR_BGRA2GRAY);
        }
    }

    //cout << "Incoming frame creation" << endl;

    if (mSensor == System::STEREO && !mpCamera2)
        mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera);
    else if(mSensor == System::STEREO && mpCamera2)
        mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera,mpCamera2,mTlr);
    else if(mSensor == System::IMU_STEREO && !mpCamera2)
        mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera,&mLastFrame,*mpImuCalib);
    else if(mSensor == System::IMU_STEREO && mpCamera2)
        mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera,mpCamera2,mTlr,&mLastFrame,*mpImuCalib);

    //cout << "Incoming frame ended" << endl;

    mCurrentFrame.mNameFile = filename;
    mCurrentFrame.mnDataset = mnNumDataset;

#ifdef REGISTER_TIMES
    vdORBExtract_ms.push_back(mCurrentFrame.mTimeORB_Ext);
    vdStereoMatch_ms.push_back(mCurrentFrame.mTimeStereoMatch);
#endif

    //cout << "Tracking start" << endl;
    Track();
    //cout << "Tracking end" << endl;

    return mCurrentFrame.GetPose();
}


Sophus::SE3f Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp, string filename)
{
    mImGray = imRGB;
    cv::Mat imDepth = imD;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,cv::COLOR_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,cv::COLOR_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,cv::COLOR_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,cv::COLOR_BGRA2GRAY);
    }

    if((fabs(mDepthMapFactor-1.0f)>1e-5) || imDepth.type()!=CV_32F)
        imDepth.convertTo(imDepth,CV_32F,mDepthMapFactor);

    if (mSensor == System::RGBD)
        mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera);
    else if(mSensor == System::IMU_RGBD)
        mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera,&mLastFrame,*mpImuCalib);






    mCurrentFrame.mNameFile = filename;
    mCurrentFrame.mnDataset = mnNumDataset;

#ifdef REGISTER_TIMES
    vdORBExtract_ms.push_back(mCurrentFrame.mTimeORB_Ext);
#endif

    Track();

    return mCurrentFrame.GetPose();
}


Sophus::SE3f Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp, string filename)
{
    mImGray = im;
    if(mImGray.channels()==3) // 이미지를 흑백 영상으로 변환
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,cv::COLOR_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,cv::COLOR_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,cv::COLOR_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,cv::COLOR_BGRA2GRAY);
    }

    if (mSensor == System::MONOCULAR)
    {
        if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET ||(lastID - initID) < mMaxFrames) 
            mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth); 
        else
            mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth); 
    }
    else if(mSensor == System::IMU_MONOCULAR) // 초기화 상태가 아닌 경우 (센서 사용 시)
    {
        if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET) // frame 객체 생성 시 이미지, time stamp, ORB 추출 객체, Vocab, camera, 왜곡 계수, bf(출력해보니 항상 0), 깊이 값, 이전 프레임, IMU calib 객체 전달
        {
            mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth,&mLastFrame,*mpImuCalib); // 초기화용 Frame 객체 생성 (특징점를 더 많이 뽑음)
        }
        else
            mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth,&mLastFrame,*mpImuCalib); // 트레킹 중인 경우 Frame 객체 생성
    }
    // cout << "mbf = " << mbf << '\n';
    if (mState==NO_IMAGES_YET) // 아직 이미지가 없다면
        t0=timestamp; // 현재 time stamp를 t0로 설정

    mCurrentFrame.mNameFile = filename;
    mCurrentFrame.mnDataset = mnNumDataset;

#ifdef REGISTER_TIMES
    vdORBExtract_ms.push_back(mCurrentFrame.mTimeORB_Ext);
#endif

    lastID = mCurrentFrame.mnId; // 이전 Frame의 ID를 현재 Frame ID로 설정
    Track(); // Tracking 진행

    return mCurrentFrame.GetPose(); // 현재 프레임의 Pose를 반환
}


void Tracking::GrabImuData(const IMU::Point &imuMeasurement)
{
    unique_lock<mutex> lock(mMutexImuQueue);
    mlQueueImuData.push_back(imuMeasurement);
}

void Tracking::PreintegrateIMU()
{

    if(!mCurrentFrame.mpPrevFrame) // 이전 프레임이 존재하지 않는다면
    {
        Verbose::PrintMess("non prev frame ", Verbose::VERBOSITY_NORMAL);
        mCurrentFrame.setIntegrated(); // integration flag 설정
        return;
    }

    mvImuFromLastFrame.clear(); // 이전 프레임 보다 이후에 존재하는 IMU 데이터 배열 초기화
    mvImuFromLastFrame.reserve(mlQueueImuData.size()); // 이전 프레임 배열 크기 할당
    if(mlQueueImuData.size() == 0) // queue에 imu 데이터가 없다면
    {
        Verbose::PrintMess("Not IMU data in mlQueueImuData!!", Verbose::VERBOSITY_NORMAL);
        mCurrentFrame.setIntegrated(); // integration flag 설정
        return;
    }

    while(true)
    {
        bool bSleep = false;
        {
            unique_lock<mutex> lock(mMutexImuQueue);
            if(!mlQueueImuData.empty()) // queue가 비어있지 않는다면
            {
                IMU::Point* m = &mlQueueImuData.front(); // queue에 있는 가장 앞에있는 IMU데이터를 꺼낸다
                cout.precision(17);
                if(m->t<mCurrentFrame.mpPrevFrame->mTimeStamp-mImuPer) // IMU의 timestamp가 이전 프레임의 time stamp보다 작다면
                {
                    mlQueueImuData.pop_front(); // queue에서 데이터 제거
                }
                else if(m->t<mCurrentFrame.mTimeStamp-mImuPer) // IMU의 time stamp가 현재 프레임의 time stamp보다 작다면
                {
                    mvImuFromLastFrame.push_back(*m); // 이전 프레임 기준 이후에 존재하는 IMU 데이터로 저장
                    mlQueueImuData.pop_front();
                }
                else // IMU의 time stamp가 현재 프레임의 time stamp보다 크면
                {
                    mvImuFromLastFrame.push_back(*m); // 이전 프레임 기준 이후에 존재하는 IMU 데이터로 저장 
                    break; // 반복 종료
                }
            }
            else // queue에 IMU 데이터가 존재한다면
            {
                break; // 반복 종료
                bSleep = true; // sleep flag 설정
            }
        }
        if(bSleep)
            usleep(500);
    }

    const int n = mvImuFromLastFrame.size()-1; // 이전 프레임 기준 IMU 데이터 개수 초기화
    if(n==0){ // IMU 데이터가 없다면
        cout << "Empty IMU measurements vector!!!\n";
        return;
    }

    IMU::Preintegrated* pImuPreintegratedFromLastFrame = new IMU::Preintegrated(mLastFrame.mImuBias,mCurrentFrame.mImuCalib); // preintegration 객체 생성 (이전 프레임의 bias와 현재 프레임의 imuCalibration 객체)

    for(int i=0; i<n; i++) // IMU 데이터 수 만큼 loop
    {
        float tstep;
        Eigen::Vector3f acc, angVel;
        if((i==0) && (i<(n-1))) // 첫 데이터이고 이후 계산할 데이터가 남아있는 경우
        {
            float tab = mvImuFromLastFrame[i+1].t-mvImuFromLastFrame[i].t; // IMU 데이터간 시간 차이 계산
            float tini = mvImuFromLastFrame[i].t-mCurrentFrame.mpPrevFrame->mTimeStamp; // IMU 데이터와 이전 프레임과의 시간 차이 계산
            // 이후 데이터와 현재 데이터의 가속도 및 각속도의 평균을 계산
            // 두 데이터 간의 가속도 변화를 먼저 계산 (이후 데이터와 현재 데이터간의 차이)
            acc = (mvImuFromLastFrame[i].a+mvImuFromLastFrame[i+1].a-
                    (mvImuFromLastFrame[i+1].a-mvImuFromLastFrame[i].a)*(tini/tab))*0.5f;  // 두 데이터 간의 가속도 평균을 계산 (이때 프레임과 IMU 간의 시간 간격을 고려하여 측정된 값을 보정, (tinitab) * 0.5f )
            angVel = (mvImuFromLastFrame[i].w+mvImuFromLastFrame[i+1].w-
                    (mvImuFromLastFrame[i+1].w-mvImuFromLastFrame[i].w)*(tini/tab))*0.5f; // 두 데이터 간의 각속도 평균을 계산 (이때 프레임과 IMU 간의 시간 간격을 고려하여 측정된 값을 보정, (tini/tab) * 0.5f )
            tstep = mvImuFromLastFrame[i+1].t-mCurrentFrame.mpPrevFrame->mTimeStamp; // imu 데이터와 현재 프레임간 시간 차이 저장
        }
        else if(i<(n-1)) // 마지막 이전 데이터가 아닌 경우
        {
            // 두 개의 연속된 데이터 간의 가속도 및 각속도의 평균을 계산
            acc = (mvImuFromLastFrame[i].a+mvImuFromLastFrame[i+1].a)*0.5f; 
            angVel = (mvImuFromLastFrame[i].w+mvImuFromLastFrame[i+1].w)*0.5f;
            tstep = mvImuFromLastFrame[i+1].t-mvImuFromLastFrame[i].t; // 
        }
        else if((i>0) && (i==(n-1))) // 첫데이터가 아니고 마지막 이전 데이터인 경우
        {
            float tab = mvImuFromLastFrame[i+1].t-mvImuFromLastFrame[i].t; // 마지막 IMU 데이터와의 시간 차이 계산
            float tend = mvImuFromLastFrame[i+1].t-mCurrentFrame.mTimeStamp; // 마지막 IMU 데이터와 현재 frame과의 시간 차이 계산
            // 이후 데이터와 현재 데이터의 가속도 및 각속도의 평균을 계산
            // 두 데이터 간의 가속도 변화를 먼저 계산 (이후 데이터와 현재 데이터간의 차이)
            acc = (mvImuFromLastFrame[i].a+mvImuFromLastFrame[i+1].a-
                    (mvImuFromLastFrame[i+1].a-mvImuFromLastFrame[i].a)*(tend/tab))*0.5f; // 두 데이터 간의 가속도 평균을 계산 (이때 IMU 간의 시간 간격을 고려하여 측정된 값을 보정, (tend/tab) * 0.5f )
            angVel = (mvImuFromLastFrame[i].w+mvImuFromLastFrame[i+1].w-
                    (mvImuFromLastFrame[i+1].w-mvImuFromLastFrame[i].w)*(tend/tab))*0.5f; // 두 데이터 간의 각속도 평균을 계산 (이때 IMU 간의 시간 간격을 고려하여 측정된 값을 보정, (tend/tab) * 0.5f )
            tstep = mCurrentFrame.mTimeStamp-mvImuFromLastFrame[i].t; // 현재 프레임과 imu 데이터간 시간 차이 저장
        }
        else if((i==0) && (i==(n-1))) // 첫 데이터이고 마지막 이전 데이터인 경우 (데이터가 1개인 경우)
        {
            // 하나의 데이터의 가속도, 각속도를 사용
            acc = mvImuFromLastFrame[i].a;
            angVel = mvImuFromLastFrame[i].w;
            tstep = mCurrentFrame.mTimeStamp-mCurrentFrame.mpPrevFrame->mTimeStamp; // 현재 프레임의 time stamp와 이전 프레임의 time stamp간의 시간 차이 계산
        }

        if (!mpImuPreintegratedFromLastKF) // 이전 keyFrame 기준 Preintegration 객체가 존재하지 않는 경우
            cout << "mpImuPreintegratedFromLastKF does not exist" << endl;
        mpImuPreintegratedFromLastKF->IntegrateNewMeasurement(acc,angVel,tstep); // 이전 keyFrame 기준 Preintegration 객체에 새로운 가속도 배열, 자이로 배열, 시간차이를 통해 integration 진행
        pImuPreintegratedFromLastFrame->IntegrateNewMeasurement(acc,angVel,tstep); // 이전 프레임 기준 Preintegration 객체에 새로운 가속도 배열, 자이로 배열, 시간차이 통해 integration 진행
    }

    mCurrentFrame.mpImuPreintegratedFrame = pImuPreintegratedFromLastFrame; // 현재 프레임에 Preintegration 객체 저장
    mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;
    mCurrentFrame.mpLastKeyFrame = mpLastKeyFrame; // 이전 keyFrame 저장

    mCurrentFrame.setIntegrated(); // integration flag 설정

    //Verbose::PrintMess("Preintegration is finished!! ", Verbose::VERBOSITY_DEBUG);
}


bool Tracking::PredictStateIMU()
{
    if(!mCurrentFrame.mpPrevFrame) // 현재 Frame에 이전 Frame이 존재하지 않는 경우
    {
        Verbose::PrintMess("No last frame", Verbose::VERBOSITY_NORMAL);
        return false; // 예측 실패
    }

    if(mbMapUpdated && mpLastKeyFrame) // Map이 Update 되었고 이전 KeyFrame이 존재하는 경우
    {
        const Eigen::Vector3f twb1 = mpLastKeyFrame->GetImuPosition(); // 이전 KeyFrame에서 위치 정보를 가져옴
        const Eigen::Matrix3f Rwb1 = mpLastKeyFrame->GetImuRotation(); // 이전 KeyFrame에서 회전 값을 가져옴
        const Eigen::Vector3f Vwb1 = mpLastKeyFrame->GetVelocity(); // 이전 KeyFrame에서 속도 값을 가져옴

        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE); // 중력 가속도 행렬 생성
        const float t12 = mpImuPreintegratedFromLastKF->dT; // 이전 KeyFrame 기준 Preintegration 객체에서 시간 차이 총합을 가져옴

        // IMU 예측 진행 (회전, 위치, 속도)
        Eigen::Matrix3f Rwb2 = IMU::NormalizeRotation(Rwb1 * mpImuPreintegratedFromLastKF->GetDeltaRotation(mpLastKeyFrame->GetImuBias()));
        Eigen::Vector3f twb2 = twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*mpImuPreintegratedFromLastKF->GetDeltaPosition(mpLastKeyFrame->GetImuBias());
        Eigen::Vector3f Vwb2 = Vwb1 + t12*Gz + Rwb1 * mpImuPreintegratedFromLastKF->GetDeltaVelocity(mpLastKeyFrame->GetImuBias());
        mCurrentFrame.SetImuPoseVelocity(Rwb2,twb2,Vwb2); // 예측한 값을 현재 Frame에 저장

        mCurrentFrame.mImuBias = mpLastKeyFrame->GetImuBias(); // 현재 Frame의 Bias를 이전 KeyFrame의 Bias로 초기화
        mCurrentFrame.mPredBias = mCurrentFrame.mImuBias; // 현재 Frame의 예측 Bias를 현재 Frame의 Bias로 초기화
        return true; // 예측 성공
    }
    else if(!mbMapUpdated) // Map이 Update되지 않은 경우
    {
        const Eigen::Vector3f twb1 = mLastFrame.GetImuPosition(); // 이전 Frame에서 위치 정보를 가져옴
        const Eigen::Matrix3f Rwb1 = mLastFrame.GetImuRotation(); // 이전 Frame에서 회전 값을 가져옴
        const Eigen::Vector3f Vwb1 = mLastFrame.GetVelocity(); // 이전 Frame에서 속도 값을 가져옴
        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE); // 중력 가속도 행렬 생성
        const float t12 = mCurrentFrame.mpImuPreintegratedFrame->dT; // 현재 Frame 기준 Preintegration 객체에서 시간 차이의 총합을 가져옴

        // IMU 예측 진행 (회전, 위치, 속도)
        Eigen::Matrix3f Rwb2 = IMU::NormalizeRotation(Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaRotation(mLastFrame.mImuBias));
        Eigen::Vector3f twb2 = twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaPosition(mLastFrame.mImuBias);
        Eigen::Vector3f Vwb2 = Vwb1 + t12*Gz + Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaVelocity(mLastFrame.mImuBias);

        mCurrentFrame.SetImuPoseVelocity(Rwb2,twb2,Vwb2); // 예측한 값을 현재 Frame에 저장

        mCurrentFrame.mImuBias = mLastFrame.mImuBias; // 현재 Frame의 Bias를 이전 Frame의 Bias로 초기화
        mCurrentFrame.mPredBias = mCurrentFrame.mImuBias; // 현재 Frame의 예측 Bias를 현재 Frame의 Bias로 초기화
        return true; // 예측 성공
    }
    else
        cout << "not IMU prediction!!" << endl;

    return false; // 예측 실패
}

void Tracking::ResetFrameIMU()
{
    // TODO To implement...
}


void Tracking::Track()
{

    if (bStepByStep) // 현재 진행 상태를 step by step으로 선택했다면
    {
        std::cout << "Tracking: Waiting to the next step" << std::endl;
        while(!mbStep && bStepByStep)
            usleep(500);
        mbStep = false;
    }

    if(mpLocalMapper->mbBadImu) // IMU의 값이 Bad 상태인 경우
    {
        cout << "TRACK: Reset map because local mapper set the bad imu flag " << endl;
        mpSystem->ResetActiveMap(); // 현재 활성화된 Map을 초기화
        return;
    }

    Map* pCurrentMap = mpAtlas->GetCurrentMap(); // 현재 Map을 Atlas에서 얻음
    if(!pCurrentMap) // 존재하지 않는다면
    {
        cout << "ERROR: There is not an active map in the atlas" << endl;
    }

    if(mState!=NO_IMAGES_YET) // 아직 이미지가 없는 경우
    {
        if(mLastFrame.mTimeStamp>mCurrentFrame.mTimeStamp) // 이전 Frame의 time stamp가 현재 Frame의 time stamp보다 크면
        {
            // 에러 발생하여 맵과 지금까지 수집한 IMU 데이터를 초기화하고 함수 종료
            cerr << "ERROR: Frame with a timestamp older than previous frame detected!" << endl;
            unique_lock<mutex> lock(mMutexImuQueue);
            mlQueueImuData.clear();
            CreateMapInAtlas();
            return;
        }
        else if(mCurrentFrame.mTimeStamp>mLastFrame.mTimeStamp+1.0) // 이전 프레임과 현재 프레임의 time stamp와 차이가 1초 이상이면
        {
            // cout << mCurrentFrame.mTimeStamp << ", " << mLastFrame.mTimeStamp << endl;
            // cout << "id last: " << mLastFrame.mnId << "    id curr: " << mCurrentFrame.mnId << endl;
            // Frame간 차이가 너무 심하여 Map을 초기화
            if(mpAtlas->isInertial()) // Atlas가 센서를 사용하는 경우
            {

                if(mpAtlas->isImuInitialized()) // Atlas에 IMU가 초기화된 경우
                {
                    cout << "Timestamp jump detected. State set to LOST. Reseting IMU integration..." << endl;
                    if(!pCurrentMap->GetIniertialBA2())
                    {
                        mpSystem->ResetActiveMap(); // 활성화된 Map 리셋
                    }
                    else
                    {
                        CreateMapInAtlas(); // Map 생성
                    }
                }
                else
                {
                    cout << "Timestamp jump detected, before IMU initialization. Reseting..." << endl;
                    mpSystem->ResetActiveMap(); // 활성화된 Map 리셋
                }
                return;
            }

        }
    }

    // IMU를 사용하고 이전 keyFrame이 존재한다면
    if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && mpLastKeyFrame)
        mCurrentFrame.SetNewBias(mpLastKeyFrame->GetImuBias()); // 이전 keyFrame의 IMU bias를 현재 프레임의 bias로 초기화

    if(mState==NO_IMAGES_YET) // 아직 이미지가 없는 상태라면
    {
        mState = NOT_INITIALIZED; // 상태를 초기화 전으로 갱신
    }

    mLastProcessedState=mState; // 이전 상태를 현재 상태로 갱신

    // IMU를 사용하고, 현재 Map이 생성되지 않았다면
    if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && !mbCreatedMap)
    {
#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_StartPreIMU = std::chrono::steady_clock::now();
#endif
        PreintegrateIMU(); // IMU PreIntegration 진행
#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_EndPreIMU = std::chrono::steady_clock::now();

        double timePreImu = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndPreIMU - time_StartPreIMU).count();
        vdIMUInteg_ms.push_back(timePreImu);
#endif

    }
    mbCreatedMap = false; // Map 생성 flag 설정

    // Get Map Mutex -> Map cannot be changed
    unique_lock<mutex> lock(pCurrentMap->mMutexMapUpdate);

    mbMapUpdated = false; // Map Update flag 설정

    int nCurMapChangeIndex = pCurrentMap->GetMapChangeIndex(); // 현재 Map index 번호 초기화
    int nMapChangeIndex = pCurrentMap->GetLastMapChange();  // 이전 Map index 번호 초기화
    if(nCurMapChangeIndex>nMapChangeIndex) // 현재 Map index 번호가 더 크면
    {
        pCurrentMap->SetLastMapChange(nCurMapChangeIndex);
        mbMapUpdated = true; // Map Update flag 설정
    }


    if(mState==NOT_INITIALIZED) // 초기화가 이루어지지 않았을 경우
    {
        if(mSensor==System::STEREO || mSensor==System::RGBD || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD)
        {
            StereoInitialization();
        }
        else
        {
            MonocularInitialization(); // Monocular인 경우 초기화 진행
        }

        //mpFrameDrawer->Update(this);
        // Tracking 상태가 OK가 아닌 경우
        if(mState!=OK) // If rightly initialized, mState=OK
        {
            mLastFrame = Frame(mCurrentFrame); // 이전 Frame을 현재 Frame으로 초기화
            return; // 함수 종료
        }

        if(mpAtlas->GetAllMaps().size() == 1) // Atlas의 모든 Map 배열의 크기가 1인 경우
        {
            mnFirstFrameId = mCurrentFrame.mnId; // 첫 번째 Frame ID를 현재 Frame의 ID로 초기화
        }
    }
    else
    {
        // System is initialized. Track Frame.
        bool bOK; // Tracking 결과 Flag 변수 생성

#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_StartPosePred = std::chrono::steady_clock::now();
#endif

        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        if(!mbOnlyTracking) // LocalMapping을 사용하는 경우
        {

            // State OK
            // Local Mapping is activated. This is the normal behaviour, unless
            // you explicitly activate the "only tracking" mode.
            if(mState==OK) // Tracking 상태가 OK인 경우
            {

                // Local Mapping might have changed some MapPoints tracked in last frame
                CheckReplacedInLastFrame(); // 이전 Frame에서 MapPoint 변경 확인

                // (속도가 측정되지 않았고 현재 Map에서 IMU가 초기화되지 않았다면) 또는 현재 Frame ID가 이전 Relocalization Frame ID보다 작은 경우
                if((!mbVelocity && !pCurrentMap->isImuInitialized()) || mCurrentFrame.mnId<mnLastRelocFrameId+2)
                {
                    Verbose::PrintMess("TRACK: Track with respect to the reference KF ", Verbose::VERBOSITY_DEBUG);
                    bOK = TrackReferenceKeyFrame(); // Referance KeyFrame으로부터 Tracking 진행
                }
                else
                {
                    Verbose::PrintMess("TRACK: Track with motion model", Verbose::VERBOSITY_DEBUG);
                    bOK = TrackWithMotionModel(); // MotionModel로부터 Tracking 진행
                    if(!bOK) // Tracking 결과 Flag가 false인 경우
                        bOK = TrackReferenceKeyFrame(); // Referance KeyFrame으로부터 Tracking 진행 
                }


                if (!bOK) // Tracking 결과 Flag가 false인 경우
                {
                    // 현재 Frame의 ID가 이전 Relocalization Frame의 ID + ResetIMU의 수보다 작거나 같거나
                    // 현재 IMU Monocular를 사용하는 경우
                    if ( mCurrentFrame.mnId<=(mnLastRelocFrameId+mnFramesToResetIMU) &&
                         (mSensor==System::IMU_MONOCULAR || mSensor==System::IMU_STEREO || mSensor == System::IMU_RGBD))
                    {
                        mState = LOST; // Tracking 상태를 LOST로 설정
                    }
                    else if(pCurrentMap->KeyFramesInMap()>10) // 현재 Map에 포함된 KeyFrame의 수가 10보다 큰 경우
                    {
                        // cout << "KF in map: " << pCurrentMap->KeyFramesInMap() << endl;
                        mState = RECENTLY_LOST; // Tracking 상태를 RECENTLY_LOST로 설정
                        mTimeStampLost = mCurrentFrame.mTimeStamp; // Lost Timestamp를 현재 Frame의 Timestamp로 설정
                    }
                    else
                    {
                        mState = LOST; // Tracking 상태를 LOST로 설정
                    }
                }
            }
            else
            {

                if (mState == RECENTLY_LOST) // Tracking 상태가 RECENTLY_LOST인 경우
                {
                    Verbose::PrintMess("Lost for a short time", Verbose::VERBOSITY_NORMAL);

                    bOK = true; // Tracking 결과를 true로 초기화
                    if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)) // IMU를 사용하는 경우
                    {
                        if(pCurrentMap->isImuInitialized()) // 현재 Map에서 IMU가 초기화된 경우
                            PredictStateIMU(); // IMU로 상태 예측
                        else
                            bOK = false; // Tracking 결과를 false로 초기화

                        if (mCurrentFrame.mTimeStamp-mTimeStampLost>time_recently_lost) // 현재 Frame의 Timestamp와 Lost일 때 Timestamp와의 차이가 recently_lost(5)보다 큰 경우
                        {
                            mState = LOST; // Tracking 상태를 LOST로 설정
                            Verbose::PrintMess("Track Lost...", Verbose::VERBOSITY_NORMAL);
                            bOK=false; // Tracking 결과를 flase로 초기화
                        }
                    }
                    else
                    {
                        // Relocalization
                        bOK = Relocalization();
                        //std::cout << "mCurrentFrame.mTimeStamp:" << to_string(mCurrentFrame.mTimeStamp) << std::endl;
                        //std::cout << "mTimeStampLost:" << to_string(mTimeStampLost) << std::endl;
                        if(mCurrentFrame.mTimeStamp-mTimeStampLost>3.0f && !bOK)
                        {
                            mState = LOST;
                            Verbose::PrintMess("Track Lost...", Verbose::VERBOSITY_NORMAL);
                            bOK=false;
                        }
                    }
                }
                else if (mState == LOST) // Tracking 상태가 LOST인 경우
                {

                    Verbose::PrintMess("A new map is started...", Verbose::VERBOSITY_NORMAL);

                    if (pCurrentMap->KeyFramesInMap()<10) // 현재 Map안에 있는 KeyFrame의 수가 10 미만인 경우
                    {
                        mpSystem->ResetActiveMap(); // 활성화된 Map Reset
                        Verbose::PrintMess("Reseting current map...", Verbose::VERBOSITY_NORMAL);
                    }else
                        CreateMapInAtlas(); // Atlas에 Map을 생성

                    if(mpLastKeyFrame) // 이전 KeyFrame이 존재하는 경우
                        mpLastKeyFrame = static_cast<KeyFrame*>(NULL); // 이전 KeyFrame을 NULL로 초기화

                    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

                    return; // 함수 종료
                }
            }

        }
        else
        {
            // Localization Mode: Local Mapping is deactivated (TODO Not available in inertial mode)
            if(mState==LOST) // Tracking 상태가 LOST인 경우
            {
                if(mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
                    Verbose::PrintMess("IMU. State LOST", Verbose::VERBOSITY_NORMAL);
                bOK = Relocalization(); // Relocalization을 진행하고 결과 반환
            }
            else
            {
                if(!mbVO) // Map에 충분한 MapPoint가 모였을 경우
                {
                    // In last frame we tracked enough MapPoints in the map
                    if(mbVelocity) // 속도 정보가 존재하는 경우
                    {
                        bOK = TrackWithMotionModel(); // MotionModel을 통한 추적 진행
                    }
                    else
                    {
                        bOK = TrackReferenceKeyFrame(); // 레퍼런스 KeyFrame을 통한 추적 진행
                    }
                }
                else
                {
                    // In last frame we tracked mainly "visual odometry" points.

                    // We compute two camera poses, one from motion model and one doing relocalization.
                    // If relocalization is sucessfull we choose that solution, otherwise we retain
                    // the "visual odometry" solution.

                    bool bOKMM = false;
                    bool bOKReloc = false;
                    vector<MapPoint*> vpMPsMM;
                    vector<bool> vbOutMM;
                    Sophus::SE3f TcwMM;
                    if(mbVelocity) // 속도 정보가 존재하는 경우
                    {
                        bOKMM = TrackWithMotionModel(); // MotionModel을 통한 추적 진행
                        vpMPsMM = mCurrentFrame.mvpMapPoints; // 현재 Frame의 MapPoint 배열 저장
                        vbOutMM = mCurrentFrame.mvbOutlier; // 현재 Frame의 아웃라이너 배열 저장
                        TcwMM = mCurrentFrame.GetPose(); // 현재 Frame의 pose를 변수에 저장
                    }
                    bOKReloc = Relocalization(); // Relocalization을 진행하고 결과 반환

                    if(bOKMM && !bOKReloc) // 추적에 성공하였고 Relocalization에 실패한 경우
                    {
                        // 변수에 저장한 값을 현재 Frame에 저장
                        mCurrentFrame.SetPose(TcwMM); 
                        mCurrentFrame.mvpMapPoints = vpMPsMM;
                        mCurrentFrame.mvbOutlier = vbOutMM;

                        if(mbVO) // Map에 충분한 MapPoint가 모이지 않았을 경우
                        {
                            for(int i =0; i<mCurrentFrame.N; i++)
                            {
                                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i]) // 현재 Frame의 MapPoint가 존재하고 아웃라이너가 아닌 경우
                                {
                                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound(); // MapPoint의 Found 수 증가
                                }
                            }
                        }
                    }
                    else if(bOKReloc) // Relocalization에 성공한 경우
                    {
                        mbVO = false; // Map에 충분한 MapPoint가 모였다고 설정
                    }

                    bOK = bOKReloc || bOKMM; // Tracking 결과를 Relocalization과 MotionModel을 통한 추적의 결과에 따라 변경함
                }
            }
        }

        if(!mCurrentFrame.mpReferenceKF) // 현재 Frame에 레퍼런스 KeyFrame이 존재하지 않는 경우
            mCurrentFrame.mpReferenceKF = mpReferenceKF; // 현재 Frame에 레퍼런스 KeyFrame을 저장

#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_EndPosePred = std::chrono::steady_clock::now();

        double timePosePred = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndPosePred - time_StartPosePred).count();
        vdPosePred_ms.push_back(timePosePred);
#endif


#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_StartLMTrack = std::chrono::steady_clock::now();
#endif
        // If we have an initial estimation of the camera pose and matching. Track the local map.
        if(!mbOnlyTracking) // LocalMapping을 사용하는 경우
        {
            if(bOK) // Tracking에 성공한 경우
            {
                bOK = TrackLocalMap(); // LocalMap을 통한 추적 진행

            }
            if(!bOK)
                cout << "Fail to track local map!" << endl;
        }
        else
        {
            // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
            // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
            // the camera we will use the local map again.
            if(bOK && !mbVO) // 추적에 성공하였고 map에 충분한 MapPoint가 있는 경우
                bOK = TrackLocalMap(); // LocalMap을 통한 추적 진행
        }

        if(bOK) // 추적에 성공한 경우
            mState = OK; // Tracking 상태 변경
        else if (mState == OK) // Tracking 상태가 OK인 경우
        {
            if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) // 센서를 사용하는 경우
            {
                Verbose::PrintMess("Track lost for less than one second...", Verbose::VERBOSITY_NORMAL);
                if(!pCurrentMap->isImuInitialized() || !pCurrentMap->GetIniertialBA2()) // 현재 Map에 IMU가 초기화되지 않았거나 IniertialBA2가 일어나지 않았을 경우
                {
                    cout << "IMU is not or recently initialized. Reseting active map..." << endl;
                    mpSystem->ResetActiveMap(); // 활성화된 Map 리셋
                }

                mState=RECENTLY_LOST; // Tracking 상태 변경
            }
            else
                mState=RECENTLY_LOST; // visual to lost

            /*if(mCurrentFrame.mnId>mnLastRelocFrameId+mMaxFrames)
            {*/
                mTimeStampLost = mCurrentFrame.mTimeStamp; // Lost TImestamp를 현재 Frame의 Timestamp로 변경
            //}
        }

        // Save frame if recent relocalization, since they are used for IMU reset (as we are making copy, it shluld be once mCurrFrame is completely modified)
        // 현재 Frame의 ID가 Relocalization Frame ID보다 작고 ResetIMU 수보다 크고
        // IMU를 사용하고 현재 Map에 IMU가 초기화 된 경우
        if((mCurrentFrame.mnId<(mnLastRelocFrameId+mnFramesToResetIMU)) && (mCurrentFrame.mnId > mnFramesToResetIMU) &&
           (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && pCurrentMap->isImuInitialized())
        {
            // TODO check this situation
            Verbose::PrintMess("Saving pointer to frame. imu needs reset...", Verbose::VERBOSITY_NORMAL);
            // 현재 Frame을 복사
            Frame* pF = new Frame(mCurrentFrame);
            pF->mpPrevFrame = new Frame(mLastFrame);

            // Load preintegration
            pF->mpImuPreintegratedFrame = new IMU::Preintegrated(mCurrentFrame.mpImuPreintegratedFrame);
        }

        if(pCurrentMap->isImuInitialized()) // 현재 Map에 IMU가 초기화된 경우
        {
            if(bOK) // Tracking에 성공한 경우
            {
                if(mCurrentFrame.mnId==(mnLastRelocFrameId+mnFramesToResetIMU)) // 현재 Frame의 ID가 Relocalization Frame ID와 같을 경우
                {
                    cout << "RESETING FRAME!!!" << endl;
                    ResetFrameIMU(); // 빈 함수 호출
                }
                else if(mCurrentFrame.mnId>(mnLastRelocFrameId+30)) // 현재 Frame의 ID가 Relocalization Frame ID보다 클 경우
                    mLastBias = mCurrentFrame.mImuBias; // 이전 Bias를 현재 Frame의 Bias로 변경
            }
        }

#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_EndLMTrack = std::chrono::steady_clock::now();

        double timeLMTrack = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndLMTrack - time_StartLMTrack).count();
        vdLMTrack_ms.push_back(timeLMTrack);
#endif

        // Update drawer
        mpFrameDrawer->Update(this); // FrameDrawer에 Tracking 객체 전달
        if(mCurrentFrame.isSet()) // 현재 Frame에 pose 정보가 있다면
            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.GetPose()); // MapDrawer에 현재 Frame의 pose를 전달

        if(bOK || mState==RECENTLY_LOST) // Tracking에 성공했거나 Tracking 상태가 RECENTLY_LOST인 경우
        {
            // Update motion model
            if(mLastFrame.isSet() && mCurrentFrame.isSet()) // 이전 Frame에 pose 정보가 있고 현재 Frame에 pose 정보가 있다면
            {
                Sophus::SE3f LastTwc = mLastFrame.GetPose().inverse(); // 이전 Frame의 pose를 가져옴
                mVelocity = mCurrentFrame.GetPose() * LastTwc; // 현재 Frame의 pose와 이전 Frame의 pose를 곱해 속도를 구함
                mbVelocity = true; // 속도 정보가 있다고 flag 설정
            }
            else {
                mbVelocity = false; // 속도 정보가 없다고 flag 설정
            }

            if(mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) // IMU를 사용하는 경우
                mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.GetPose()); // MapDrawer에 현재 Frame의 pose를 전달

            // Clean VO matches
            for(int i=0; i<mCurrentFrame.N; i++)
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i]; // 현재 Frame의 MapPoint를 가져옴
                if(pMP) // MapPoint가 존재하는 경우
                    if(pMP->Observations()<1) // MapPoint의 추적 대상이 없는 경우
                    {
                        mCurrentFrame.mvbOutlier[i] = false; // 아웃라이너가 없다고 설정
                        mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL); // MapPoint 제거
                    }
            }

            // Delete temporal MapPoints
            // 임시 MapPoint 배열의 크기만큼 loop
            for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++)
            {
                MapPoint* pMP = *lit;
                delete pMP; // 임시 MapPoint 제거
            }
            mlpTemporalPoints.clear(); // 임시 MapPoint 배열 지우기

#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_StartNewKF = std::chrono::steady_clock::now();
#endif
            bool bNeedKF = NeedNewKeyFrame(); // 새로운 KeyFrame이 필요한지 결정

            // Check if we need to insert a new keyframe
            // if(bNeedKF && bOK)
            // 새로운 keyFrame이 필요하고 
            // Tracking에 성공하거나 InsertKFsLost가 true이고(디폴트 값이 true) IMU를 사용하는 경우
            if(bNeedKF && (bOK || (mInsertKFsLost && mState==RECENTLY_LOST &&
                                   (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD))))
                CreateNewKeyFrame();

#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndNewKF = std::chrono::steady_clock::now();

            double timeNewKF = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndNewKF - time_StartNewKF).count();
            vdNewKF_ms.push_back(timeNewKF);
#endif

            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame. Only has effect if lastframe is tracked
            for(int i=0; i<mCurrentFrame.N;i++)
            {
                if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i]) // 현재 Frame에 MapPoint가 존재하고 아웃라이너인 경우 
                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL); // 현재 Frame의 MapPoint를 제거
            }
        }

        // Reset if the camera get lost soon after initialization
        if(mState==LOST) // Tracking 상태가 LOST인 경우
        {
            if(pCurrentMap->KeyFramesInMap()<=10) // 현재 Map에 keyFrame이 10개 이하 존재하는 경우
            {
                mpSystem->ResetActiveMap(); // 활성화 Map을 Reset
                return; // 함수 종료
            }
            if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) // IMU를 사용하는 경우
                if (!pCurrentMap->isImuInitialized()) // 현재 Map에 IMU가 초기화 되지 않았을 경우
                {
                    Verbose::PrintMess("Track lost before IMU initialisation, reseting...", Verbose::VERBOSITY_QUIET);
                    mpSystem->ResetActiveMap(); // 활성화 Map을 Reset
                    return; // 함수 종료
                }

            CreateMapInAtlas(); // Atlas에 Map 생성

            return; // 함수 종료
        }

        if(!mCurrentFrame.mpReferenceKF) // 현재 Frame에 레퍼런스 KeyFrame이 존재하지 않는 경우
            mCurrentFrame.mpReferenceKF = mpReferenceKF; // 현재 Frame에 레퍼런스 KeyFrame 저장

        mLastFrame = Frame(mCurrentFrame); // 이전 프레임을 현재 Frame으로 변경
    }




    if(mState==OK || mState==RECENTLY_LOST) // Tracking 상태가 OK 이거나 RECENTLY_LOST인 경우
    {
        // Store frame pose information to retrieve the complete camera trajectory afterwards.
        if(mCurrentFrame.isSet()) // 현재 Frame의 Pose가 있는 set된 경우
        {
            Sophus::SE3f Tcr_ = mCurrentFrame.GetPose() * mCurrentFrame.mpReferenceKF->GetPoseInverse(); // 현재 Frame의 Pose와 현재 Frame의 레퍼런스 KeyFrame의 Pose를 곱해 값 저장
            mlRelativeFramePoses.push_back(Tcr_); // Relative Frame Pose 배열에 삽입
            mlpReferences.push_back(mCurrentFrame.mpReferenceKF); // 현재 Frame의 레퍼런스 KeyFrame을 배열에 삽입
            mlFrameTimes.push_back(mCurrentFrame.mTimeStamp); // 현재 Frame의 Timestamp를 배열에 삽입
            mlbLost.push_back(mState==LOST); // 현재 Tracking 상태를 배열에 삽입
        }
        else
        {
            // This can happen if tracking is lost
            mlRelativeFramePoses.push_back(mlRelativeFramePoses.back()); // Relative Frame Pose 배열에 마지막 값을 삽입
            mlpReferences.push_back(mlpReferences.back()); // 레퍼런스 배열에 마지막 값 삽입
            mlFrameTimes.push_back(mlFrameTimes.back()); // Timestamp 배열에 마지막 값 삽입
            mlbLost.push_back(mState==LOST); // 현재 Tracking 상태를 배열에 삽입
        }

    }

#ifdef REGISTER_LOOP
    if (Stop()) {

        // Safe area to stop
        while(isStopped())
        {
            usleep(3000);
        }
    }
#endif
}


void Tracking::StereoInitialization()
{
    if(mCurrentFrame.N>500)
    {
        if (mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        {
            if (!mCurrentFrame.mpImuPreintegrated || !mLastFrame.mpImuPreintegrated)
            {
                cout << "not IMU meas" << endl;
                return;
            }

            if (!mFastInit && (mCurrentFrame.mpImuPreintegratedFrame->avgA-mLastFrame.mpImuPreintegratedFrame->avgA).norm()<0.5)
            {
                cout << "not enough acceleration" << endl;
                return;
            }

            if(mpImuPreintegratedFromLastKF)
                delete mpImuPreintegratedFromLastKF;

            mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);
            mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;
        }

        // Set Frame pose to the origin (In case of inertial SLAM to imu)
        if (mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        {
            Eigen::Matrix3f Rwb0 = mCurrentFrame.mImuCalib.mTcb.rotationMatrix();
            Eigen::Vector3f twb0 = mCurrentFrame.mImuCalib.mTcb.translation();
            Eigen::Vector3f Vwb0;
            Vwb0.setZero();
            mCurrentFrame.SetImuPoseVelocity(Rwb0, twb0, Vwb0);
        }
        else
            mCurrentFrame.SetPose(Sophus::SE3f());

        // Create KeyFrame
        KeyFrame* pKFini = new KeyFrame(mCurrentFrame,mpAtlas->GetCurrentMap(),mpKeyFrameDB);

        // Insert KeyFrame in the map
        mpAtlas->AddKeyFrame(pKFini);

        // Create MapPoints and asscoiate to KeyFrame
        if(!mpCamera2){
            for(int i=0; i<mCurrentFrame.N;i++)
            {
                float z = mCurrentFrame.mvDepth[i];
                if(z>0)
                {
                    Eigen::Vector3f x3D;
                    mCurrentFrame.UnprojectStereo(i, x3D);
                    MapPoint* pNewMP = new MapPoint(x3D, pKFini, mpAtlas->GetCurrentMap());
                    pNewMP->AddObservation(pKFini,i);
                    pKFini->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpAtlas->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                }
            }
        } else{
            for(int i = 0; i < mCurrentFrame.Nleft; i++){
                int rightIndex = mCurrentFrame.mvLeftToRightMatch[i];
                if(rightIndex != -1){
                    Eigen::Vector3f x3D = mCurrentFrame.mvStereo3Dpoints[i];

                    MapPoint* pNewMP = new MapPoint(x3D, pKFini, mpAtlas->GetCurrentMap());

                    pNewMP->AddObservation(pKFini,i);
                    pNewMP->AddObservation(pKFini,rightIndex + mCurrentFrame.Nleft);

                    pKFini->AddMapPoint(pNewMP,i);
                    pKFini->AddMapPoint(pNewMP,rightIndex + mCurrentFrame.Nleft);

                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpAtlas->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    mCurrentFrame.mvpMapPoints[rightIndex + mCurrentFrame.Nleft]=pNewMP;
                }
            }
        }

        Verbose::PrintMess("New Map created with " + to_string(mpAtlas->MapPointsInMap()) + " points", Verbose::VERBOSITY_QUIET);

        //cout << "Active map: " << mpAtlas->GetCurrentMap()->GetId() << endl;

        mpLocalMapper->InsertKeyFrame(pKFini);

        mLastFrame = Frame(mCurrentFrame);
        mnLastKeyFrameId = mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;
        //mnLastRelocFrameId = mCurrentFrame.mnId;

        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints=mpAtlas->GetAllMapPoints();
        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

        mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.push_back(pKFini);

        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.GetPose());

        mState=OK;
    }
}


void Tracking::MonocularInitialization()
{

    if(!mbReadyToInitializate) // 초기화할 준비가 되지 않았을 경우 (초기 frame이 존재하지 않는 경우)
    {
        // Set Reference Frame
        if(mCurrentFrame.mvKeys.size()>100) // 현재 Frame의 keyPoint의 수가 100개 초과인 경우
        {

            mInitialFrame = Frame(mCurrentFrame); // 초기 Frame 생성
            mLastFrame = Frame(mCurrentFrame); // 이전 Frame 생성
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size()); // keyPoint 배열 크기 할당
            for(size_t i=0; i<mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i]=mCurrentFrame.mvKeysUn[i].pt; // 현재 Frame의 투영되지 않은 keyPoint를 통해 배열 초기화

            fill(mvIniMatches.begin(),mvIniMatches.end(),-1); // 초기 매칭 배열을 -1로 초기화

            if (mSensor == System::IMU_MONOCULAR) // Monocular인 경우
            {
                if(mpImuPreintegratedFromLastKF) // 이전 keyFrame에 대해 Preintegration을 진행하였다면
                {
                    delete mpImuPreintegratedFromLastKF; // 객체 삭제
                }
                mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib); // 이전 keyFrame에 대한 Preintegration 객체 생성
                mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF; // 현재 frame의 preintgration객체 초기화

            }

            mbReadyToInitializate = true; // 초기화할 준비가 되었다고 설정

            return;
        }
    }
    else // 초기화할 준비가 되었을 경우
    {
        // 현재 Frame의 keyPoint 수가 100개 이하이거나 (Monocular를 사용하고 이전 프레임의 time stamp와 초기화용 Frame의 time stamp와의 시간 차이가 1초를 넘은) 경우
        if (((int)mCurrentFrame.mvKeys.size()<=100)||((mSensor == System::IMU_MONOCULAR)&&(mLastFrame.mTimeStamp-mInitialFrame.mTimeStamp>1.0)))
        {
            mbReadyToInitializate = false; // 초기화할 준비 취소

            return; 
        }

        // Find correspondences
        ORBmatcher matcher(0.9,true); // ORB 특징점 추출 객체 생성
        int nmatches = matcher.SearchForInitialization(mInitialFrame,mCurrentFrame,mvbPrevMatched,mvIniMatches,100);

        // Check if there are enough correspondences
        if(nmatches<100) // 매칭이 100개 미만인 경우
        {
            mbReadyToInitializate = false; // 초기화할 준비 취소
            return;
        }

        Sophus::SE3f Tcw;
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

        // Homography나 Fundamental를 통해 parallax를 구하고 Tcw를 초기화함 
        if(mpCamera->ReconstructWithTwoViews(mInitialFrame.mvKeysUn,mCurrentFrame.mvKeysUn,mvIniMatches,Tcw,mvIniP3D,vbTriangulated))
        {
            for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++) 
            {
                if(mvIniMatches[i]>=0 && !vbTriangulated[i]) // 매칭수가 0이상이고 Triangulated 대상이 아닌 경우
                {
                    mvIniMatches[i]=-1; // 매칭수 1감소
                    nmatches--; 
                }
            }

            // Set Frame Poses
            mInitialFrame.SetPose(Sophus::SE3f()); // 초기화용 Frame의 pose를 초기화
            mCurrentFrame.SetPose(Tcw); // Tcw 초기화

            CreateInitialMapMonocular(); // 맵 생성
        }
    }
}



void Tracking::CreateInitialMapMonocular()
{
    // Create KeyFrames
    KeyFrame* pKFini = new KeyFrame(mInitialFrame,mpAtlas->GetCurrentMap(),mpKeyFrameDB); // 초기화 keyFrame 생성
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame,mpAtlas->GetCurrentMap(),mpKeyFrameDB); // 현재 keyFrame 생성

    if(mSensor == System::IMU_MONOCULAR) // Monocular인 경우
        pKFini->mpImuPreintegrated = (IMU::Preintegrated*)(NULL); // 초기화 keyFrame의 preintegration 객체 제거


    pKFini->ComputeBoW(); // 초기화 keyFrame의 bag of word 계산
    pKFcur->ComputeBoW(); // 햔재 keyFrame의 bag of word 계산

    // Insert KFs in the map
    mpAtlas->AddKeyFrame(pKFini); // Atlas에 초가화 KeyFrame 추가
    mpAtlas->AddKeyFrame(pKFcur); // Atlas에 현재 KeyFrame 추가

    for(size_t i=0; i<mvIniMatches.size();i++) // 매칭점 배열 크기만큼 loop
    {
        if(mvIniMatches[i]<0) // 매칭 점수가 0 미만인 경우
            continue; // 무시

        //Create MapPoint.
        Eigen::Vector3f worldPos;
        worldPos << mvIniP3D[i].x, mvIniP3D[i].y, mvIniP3D[i].z; // 3D world 좌표 설정
        MapPoint* pMP = new MapPoint(worldPos,pKFcur,mpAtlas->GetCurrentMap()); // world 좌표와 현재 KeyFrame, 현재 맵을 통해 MapPoint 객체 생성

        pKFini->AddMapPoint(pMP,i); // 초기화 keyFrame에 MapPoint 추가
        pKFcur->AddMapPoint(pMP,mvIniMatches[i]); // 현재 KeyFrame에 MapPoint 추가

        pMP->AddObservation(pKFini,i); // MapPoint에 초기화 KeyFrame을 추적할 수 있도록 추가
        pMP->AddObservation(pKFcur,mvIniMatches[i]); // MapPoint에 현재 KeyFrame을 추적할 수 있도록 추가

        pMP->ComputeDistinctiveDescriptors(); // 디스크립터 연산 진행 (추적하도록 추가한 keyFrame에 대해서)
        pMP->UpdateNormalAndDepth(); // Normal 벡터와 Depth 업데이트

        //Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP; // 현재 Frame의 MapPoint배열에 생성한 MapPoint 추가
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false; // 현재 Frame의 Outlier인지 플레그 설정

        //Add to Map
        mpAtlas->AddMapPoint(pMP); // Atlas에 MapPoint 추가
    }


    // Update Connections
    pKFini->UpdateConnections(); // 초기화 KeyFrame에 대해서 MapPoint와의 연결 업데이트
    pKFcur->UpdateConnections(); // 현재 KeyFrame에 대해서 MapPoint와의 연결 업데이트

    std::set<MapPoint*> sMPs;
    sMPs = pKFini->GetMapPoints(); // 초기화 KeyFrame에서 MapPoint들을 가져옴

    // Bundle Adjustment
    Verbose::PrintMess("New Map created with " + to_string(mpAtlas->MapPointsInMap()) + " points", Verbose::VERBOSITY_QUIET);
    Optimizer::GlobalBundleAdjustemnt(mpAtlas->GetCurrentMap(),20); // 현재 Map에 대해서 GlobalBA 진행

    float medianDepth = pKFini->ComputeSceneMedianDepth(2); // 초기화 KeyFrame으로 부터 중앙 Depth값 계산
    float invMedianDepth; // Depth의 중앙 값
    if(mSensor == System::IMU_MONOCULAR) // IMU Monocular인 경우
        invMedianDepth = 4.0f/medianDepth; // 4.0f
    else
        invMedianDepth = 1.0f/medianDepth;

    // Depth값이 음수이거나 현재 KeyFrame의 Tracking MapPoint들이 50 미만인 경우
    if(medianDepth<0 || pKFcur->TrackedMapPoints(1)<50) // TODO Check, originally 100 tracks
    {
        Verbose::PrintMess("Wrong initialization, reseting...", Verbose::VERBOSITY_QUIET);
        mpSystem->ResetActiveMap(); // 활성화된 Map Reset 후 함수 종료
        return;
    }

    // Scale initial baseline
    Sophus::SE3f Tc2w = pKFcur->GetPose(); // 현재 KeyFrame에서 Pose를 가져옴
    Tc2w.translation() *= invMedianDepth; // Pose의 Translation *= Depth 
    pKFcur->SetPose(Tc2w); // 현재 KeyFrame의 Pose를 가중치를 곱한 결과로 변경

    // Scale points
    vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches(); // 초기화 KeyFrame의 매칭 MapPoint를 가져옴
    for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++) // 매칭 MapPoint 배열 크기 만큼 loop
    {
        if(vpAllMapPoints[iMP]) // MapPoint가 존재하는 경우
        {
            MapPoint* pMP = vpAllMapPoints[iMP]; // MapPoint를 가져옴
            pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth); // MapPoint의 pose에서 가충치를 곱한 결과를 MapPoint pose로 설정
            pMP->UpdateNormalAndDepth(); // Normal과 Depth를 Update
        }
    }

    if (mSensor == System::IMU_MONOCULAR) // IMU를 사용하는 경우
    {
        pKFcur->mPrevKF = pKFini; // 현재 KeyFrame의 이전 KeyFrame을 초기화 KeyFrame으로 초기화
        pKFini->mNextKF = pKFcur; // 초기화 KeyFrame의 다음 KeyFrame을 현재 KeyFrame으로 초기화
        pKFcur->mpImuPreintegrated = mpImuPreintegratedFromLastKF; // 현재 KeyFrame의 Preintegration 객체를 이전 KeyFrame 기준 Preintegration객체로 초기화

        // 이전 KeyFrame 기준 Preintegration객체를 현재 KeyFrame의 Bias와 현재 KeyFrame의 IMU Calibration 정보를 통해 객체 생성
        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(pKFcur->mpImuPreintegrated->GetUpdatedBias(),pKFcur->mImuCalib); 
    }


    mpLocalMapper->InsertKeyFrame(pKFini); // LocalMapping 객체에 초기화 KeyFrame을 추가
    mpLocalMapper->InsertKeyFrame(pKFcur); // LocalMapping 객체에 현재 KeyFrame을 추가
    mpLocalMapper->mFirstTs=pKFcur->mTimeStamp; // LocalMapping 객체에 첫 Timestamp를 현재 KeyFrame의 TimeStamp로 초기화

    mCurrentFrame.SetPose(pKFcur->GetPose()); // 현재 Frame의 Pose를 현재 KeyFrame의 Pose로 설정
    mnLastKeyFrameId=mCurrentFrame.mnId; // 이전 keyFrame ID를 현재 Frame의 ID로 초기화
    mpLastKeyFrame = pKFcur; // 이전 KeyFrame을 현재 KeyFrame으로 초기화
    //mnLastRelocFrameId = mInitialFrame.mnId;

    mvpLocalKeyFrames.push_back(pKFcur); // Local KeyFrame 배열에 현재 KeyFrame을 삽입
    mvpLocalKeyFrames.push_back(pKFini); // Local KeyFrame 배열에 초기화 keyFrame을 삽입
    mvpLocalMapPoints=mpAtlas->GetAllMapPoints(); // Local MapPoint 배열을 Atlas의 MapPoint로 초기화 
    mpReferenceKF = pKFcur; // 레퍼런스 KeyFrame을 현재 KeyFrame으로 초기화
    mCurrentFrame.mpReferenceKF = pKFcur; // 현재 Frame의 레퍼런스 KeyFrame을 현재 KeyFrame으로 초기화

    // Compute here initial velocity
    vector<KeyFrame*> vKFs = mpAtlas->GetAllKeyFrames(); // Atlas에서 모든 KeyFrame을 가져옴

    Sophus::SE3f deltaT = vKFs.back()->GetPose() * vKFs.front()->GetPoseInverse(); // 마지막 KeyFrame의 Pose와 첫 KeyFrame의 Pose를 곱하여 DeltaT를 구함
    mbVelocity = false;
    Eigen::Vector3f phi = deltaT.so3().log(); // DeltaT의 회전 변환에 대한 로그 값을 구해 값 저장

    double aux = (mCurrentFrame.mTimeStamp-mLastFrame.mTimeStamp)/(mCurrentFrame.mTimeStamp-mInitialFrame.mTimeStamp); // 현재 프레임과 이전 프레임 사이의 시간 간격을 사용하여 보정 계수 연산
    phi *= aux; // 회전 변환에 대한 로그 값에 보정 계수를 곱하여 최종적인 초기 속도 값을 구함

    mLastFrame = Frame(mCurrentFrame); // 이전 Frame을 현재 Frame으로 초기화

    mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints); // Atlas의 레퍼런스 MapPoint 배열을 LocalMapPoint배열로 설정

    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose()); // 현재 KeyFrame의 Pose를 MapDrawer의 현재 CameraPose로 설정

    mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.push_back(pKFini); // Atlas의 현재 Map의 초기화 KeyFrame 배열에 추가

    mState=OK; // Tracking 상태 변경

    initID = pKFcur->mnId; // 초기화 ID를 현재 KeyFrame의 ID로 
}


void Tracking::CreateMapInAtlas()
{
    mnLastInitFrameId = mCurrentFrame.mnId; // 현재 Frame의 ID를 이전 초기화 Frame ID로 저장
    mpAtlas->CreateNewMap(); // Atlas에 새로운 맵을 생성
    if (mSensor==System::IMU_STEREO || mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_RGBD)
        mpAtlas->SetInertialSensor(); // Atlas에 센서를 사용한다고 설정
    mbSetInit=false;

    mnInitialFrameId = mCurrentFrame.mnId+1; // 초기화 Frame ID를 현재 Frame ID의 다음 번호로 저장
    mState = NO_IMAGES_YET; // 현재 상태 저장

    // Restart the variable with information about the last KF
    mbVelocity = false; // 속도가 측정되지 않았다고 설정
    //mnLastRelocFrameId = mnLastInitFrameId; // The last relocation KF_id is the current id, because it is the new starting point for new map
    Verbose::PrintMess("First frame id in map: " + to_string(mnLastInitFrameId+1), Verbose::VERBOSITY_NORMAL);
    mbVO = false; // Init value for know if there are enough MapPoints in the last KF
    if(mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR) // Monocular인 경우
    {
        mbReadyToInitializate = false; // 초기화 준비가 되지 않았다고 설정
    }
    // Monocular이고 이전 KeyFrame에 대해 PreIntegration이 진행되었다면
    if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && mpImuPreintegratedFromLastKF)
    {
        delete mpImuPreintegratedFromLastKF; // 객체 삭제
        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib); // 객체 재생성
    }

    if(mpLastKeyFrame) // 이전 KeyFrame 객체가 존재 한다면
        mpLastKeyFrame = static_cast<KeyFrame*>(NULL); // 초기화

    if(mpReferenceKF) // 레퍼런스 KeyFrame 객체가 존재 한다면
        mpReferenceKF = static_cast<KeyFrame*>(NULL); // 초기화

    mLastFrame = Frame(); // 이전 Frame과 현재 Frame 초기화
    mCurrentFrame = Frame();
    mvIniMatches.clear();

    mbCreatedMap = true;
}

void Tracking::CheckReplacedInLastFrame()
{
    for(int i =0; i<mLastFrame.N; i++) // 이전 Frame의 keyPoint 배열 크기만큼 loop
    {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i]; // 이전 Frame에서 MapPoint를 가져옴

        if(pMP) // MapPoint가 존재하는 경우
        {
            MapPoint* pRep = pMP->GetReplaced(); // MapPoint의 Replace된 MapPoint를 가져옴
            if(pRep) // Replace MapPoint가 존재하는 경우
            {
                mLastFrame.mvpMapPoints[i] = pRep; // 이전 Frame의 MapPoint를 Replace MapPoint로 변경
            }
        }
    }
}


bool Tracking::TrackReferenceKeyFrame()
{
    // Compute Bag of Words vector
    mCurrentFrame.ComputeBoW(); // 현재 Frame에 대해 Bag of Word를 계산

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.7,true); // ORB matcher 객체 생성
    vector<MapPoint*> vpMapPointMatches; // 매칭 MapPoint 배열 생성

    int nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches); // 레퍼런스 KeyFrame과 현재 Frame에 대해서 매칭을 구하고 매칭 수를 반환

    if(nmatches<15) // 매칭 점의 수가 15 미만인 경우
    {
        cout << "TRACK_REF_KF: Less than 15 matches!!\n";
        return false; // Tracking 실패
    }

    mCurrentFrame.mvpMapPoints = vpMapPointMatches; // 현재 Frame의 MapPoint 배열을 매칭 MapPoint 배열로 초기화
    mCurrentFrame.SetPose(mLastFrame.GetPose()); // 현재 Frame의 Pose를 이전 Frame의 Pose로 설정

    //mCurrentFrame.PrintPointDistribution();


    // cout << " TrackReferenceKeyFrame mLastFrame.mTcw:  " << mLastFrame.mTcw << endl;
    Optimizer::PoseOptimization(&mCurrentFrame); // Pose 최적화 진행

    // Discard outliers
    int nmatchesMap = 0; // Map 매칭 수 변수 선언
    for(int i =0; i<mCurrentFrame.N; i++) // 현재 Frame의 KeyPoint 배열 크기만큼 loop
    {
        //if(i >= mCurrentFrame.Nleft) break;
        if(mCurrentFrame.mvpMapPoints[i]) // 현재 Frame에 MapPoint가 존재하는 경우
        {
            if(mCurrentFrame.mvbOutlier[i]) // Outlier인 경우
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i]; // MapPoint를 가져옴

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL); // 현재 Frame의 MapPoint를 제거
                mCurrentFrame.mvbOutlier[i]=false; // Outlier가 아니라고 변경
                if(i < mCurrentFrame.Nleft){ // MapPoint가 왼쪽 카메라에 해당하는 경우
                    pMP->mbTrackInView = false; // MapPoint의 TrackInView Flag를 false로 설정
                }
                else{
                    pMP->mbTrackInViewR = false; // MapPoint의 TrackInViewR Flag를 false로 설정
                }
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId; // MapPoint의 관찰 이전 Frame을 현재 Frame의 ID로 초기화
                nmatches--; // 매칭 수 감소
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0) // 현재 Frame의 MapPoint의 추적 대상의 수가 0 보다 큰 경우
                nmatchesMap++; // Map 매칭 수 증가
        }
    }

    if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        return true; // IMU를 사용하는 경우 Tracking 선공
    else
        return nmatchesMap>=10; // 맵 매칭 수가 10 이상인 경우 Tracking 성공
}

void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    KeyFrame* pRef = mLastFrame.mpReferenceKF; // 이전 Frame의 레퍼런스 KeyFrame을 가져옴
    Sophus::SE3f Tlr = mlRelativeFramePoses.back(); // Relative Frame pose 배열에서 최근 값을 가져옴
    mLastFrame.SetPose(Tlr * pRef->GetPose()); // pose 배열의 값과 레퍼런스 KeyFrame의 pose를 곱해 이전 Frame의 Pose로 설정

    // 이전 KeyFrame의 ID가 이전 Frame의 ID와 같거나 Monocular 이거나 LocalMapping을 사용하는 경우
    if(mnLastKeyFrameId==mLastFrame.mnId || mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR || !mbOnlyTracking)
        return; // 함수 종료

    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    vector<pair<float,int> > vDepthIdx;
    const int Nfeat = mLastFrame.Nleft == -1? mLastFrame.N : mLastFrame.Nleft;
    vDepthIdx.reserve(Nfeat);
    for(int i=0; i<Nfeat;i++)
    {
        float z = mLastFrame.mvDepth[i];
        if(z>0)
        {
            vDepthIdx.push_back(make_pair(z,i));
        }
    }

    if(vDepthIdx.empty())
        return;

    sort(vDepthIdx.begin(),vDepthIdx.end());

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    int nPoints = 0;
    for(size_t j=0; j<vDepthIdx.size();j++)
    {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(!pMP)
            bCreateNew = true;
        else if(pMP->Observations()<1)
            bCreateNew = true;

        if(bCreateNew)
        {
            Eigen::Vector3f x3D;

            if(mLastFrame.Nleft == -1){
                mLastFrame.UnprojectStereo(i, x3D);
            }
            else{
                x3D = mLastFrame.UnprojectStereoFishEye(i);
            }

            MapPoint* pNewMP = new MapPoint(x3D,mpAtlas->GetCurrentMap(),&mLastFrame,i);
            mLastFrame.mvpMapPoints[i]=pNewMP;

            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else
        {
            nPoints++;
        }

        if(vDepthIdx[j].first>mThDepth && nPoints>100)
            break;

    }
}

bool Tracking::TrackWithMotionModel()
{
    ORBmatcher matcher(0.9,true); // ORB matcher 객체 생성

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    UpdateLastFrame(); // 레퍼런스 KeyFrame 기준으로 이전 Frame Update

    // Atlas에서 IMU가 초기화 되었고 (현재 Frame의 ID가 이전 Relocalization Frame의 ID + ResetIMU 수 보다 큰 경우)
    if (mpAtlas->isImuInitialized() && (mCurrentFrame.mnId>mnLastRelocFrameId+mnFramesToResetIMU))
    {
        // Predict state with IMU if it is initialized and it doesnt need reset
        PredictStateIMU(); // IMU로 상태 예측 진행
        return true; // Tracking 성공
    }
    else
    {
        mCurrentFrame.SetPose(mVelocity * mLastFrame.GetPose()); // 현재 Frame의 pose를 속도 * 이전 Frame의 pose로 설정
    }




    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL)); // 현재 Frame의 MapPoint 배열을 NULL로 초기화

    // Project points seen in previous frame
    int th; // threshold 값 변수 선언

    if(mSensor==System::STEREO)
        th=7;
    else
        th=15; // threshold 값 저장

    // 현재 Frame과 이전 프레임, threshold 값을 통해 Projection 연산 진행. matching 수 반환
    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR);

    // If few matches, uses a wider window search
    if(nmatches<20) // 매칭 수가 20 미만인 경우
    {
        Verbose::PrintMess("Not enough matches, wider window search!!", Verbose::VERBOSITY_NORMAL);
        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL)); // 현재 Frame의 MapPoint 배열을 NULL로 초기화

        nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR); // threshold 값을 2배로 올려 다시 Projection 연산 진행
        Verbose::PrintMess("Matches with wider search: " + to_string(nmatches), Verbose::VERBOSITY_NORMAL);

    }

    if(nmatches<20) // 다시 연산을 진행하여도 매칭 수가 20 미만인 경우
    {
        Verbose::PrintMess("Not enough matches!!", Verbose::VERBOSITY_NORMAL);
        if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) // IMU를 사용하는 경우
            return true; // Tracking 성공
        else
            return false; 
    }

    // Optimize frame pose with all matches
    Optimizer::PoseOptimization(&mCurrentFrame); // 현재 Frame에 대해 pose 최적화 진행

    // Discard outliers
    int nmatchesMap = 0; // Map 매칭 수 변수 선언
    for(int i =0; i<mCurrentFrame.N; i++) // 현재 Frame의 KeyPoint 배열 크기만큼 loop
    {
        if(mCurrentFrame.mvpMapPoints[i]) // 현재 Frame에 MapPoint가 존재하는 경우
        {
            if(mCurrentFrame.mvbOutlier[i]) // Outlier인 경우
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i]; // MapPoint를 가져옴

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL); // 현재 Frame의 MapPoint를 제거
                mCurrentFrame.mvbOutlier[i]=false; // Outlier가 아니라고 변경
                if(i < mCurrentFrame.Nleft){ // MapPoint가 왼쪽 카메라에 해당하는 경우
                    pMP->mbTrackInView = false; // MapPoint의 TrackInView Flag를 false로 설정
                }
                else{
                    pMP->mbTrackInViewR = false; // MapPoint의 TrackInViewR Flag를 false로 설정
                }
                pMP->mnLastFrameSeen = mCurrentFrame.mnId; // MapPoint의 관찰 이전 Frame을 현재 Frame의 ID로 초기화
                nmatches--; // 매칭 수 감소
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0) // 현재 Frame의 MapPoint의 추적 대상의 수가 0 보다 큰 경우
                nmatchesMap++; // 맵 매칭 수 증가
        }
    }

    if(mbOnlyTracking) // LocalMapping을 사용하지 않는 경우
    {
        mbVO = nmatchesMap<10; // Map 매칭 수가 10 미만인 경우 충분한 MapPoint가 모이지 않았다고 true로 설정
        return nmatches>20; // 매칭 점 수가 20 보다 큰 경우 Tracking 성공
    }

    if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) // IMU를 사용하는 경우
        return true; // Tracking 성공
    else
        return nmatchesMap>=10;
}

bool Tracking::TrackLocalMap()
{

    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.
    mTrackedFr++; // Tracking하고 있는 Frame 수 증가

    UpdateLocalMap(); // Local KeyFrame과 Local MapPoint를 찾아 배열 Update
    SearchLocalPoints(); // LocalMap과 일치하는 Point 검색

    // TOO check outliers before PO
    int aux1 = 0, aux2=0;
    for(int i=0; i<mCurrentFrame.N; i++)
        if( mCurrentFrame.mvpMapPoints[i]) // 현재 KeyFrame의 MapPoint가 존재하는 경우
        {
            aux1++; // 가중치1 증가
            if(mCurrentFrame.mvbOutlier[i]) // MapPoint가 아웃라이너가 아닌 경우
                aux2++; // 가중치2 증가
        }

    int inliers;
    if (!mpAtlas->isImuInitialized()) // Atlas에 IMU가 초기화 되지 않았을 경우
        Optimizer::PoseOptimization(&mCurrentFrame); // 현재 Frame에 대해 Pose 최적화 진행
    else
    {
        if(mCurrentFrame.mnId<=mnLastRelocFrameId+mnFramesToResetIMU) // 현재 Frame의 ID가 이전 Relocalization Frame ID보다 작거나 같을 경우
        {
            Verbose::PrintMess("TLM: PoseOptimization ", Verbose::VERBOSITY_DEBUG);
            Optimizer::PoseOptimization(&mCurrentFrame); // 현재 Frame에 대해 Pose 최적화 진행
        }
        else
        {
            // if(!mbMapUpdated && mState == OK) //  && (mnMatchesInliers>30))
            // Map이 갱신되지 않았을 경우
            if(!mbMapUpdated) //  && (mnMatchesInliers>30))
            {
                Verbose::PrintMess("TLM: PoseInertialOptimizationLastFrame ", Verbose::VERBOSITY_DEBUG);
                // 현재 Frame에 대해 Pose 최적화 진행. 이때 이전 Frame을 기준으로 진행
                inliers = Optimizer::PoseInertialOptimizationLastFrame(&mCurrentFrame); // , !mpLastKeyFrame->GetMap()->GetIniertialBA1());
            }
            else
            {
                Verbose::PrintMess("TLM: PoseInertialOptimizationLastKeyFrame ", Verbose::VERBOSITY_DEBUG);
                // 현재 Frame에 대해 Pose 최적화 진행. 이때 이전 keyFrame을 기준으로 진행
                inliers = Optimizer::PoseInertialOptimizationLastKeyFrame(&mCurrentFrame); // , !mpLastKeyFrame->GetMap()->GetIniertialBA1());
            }
        }
    }

    aux1 = 0, aux2 = 0;
    for(int i=0; i<mCurrentFrame.N; i++)
        if( mCurrentFrame.mvpMapPoints[i]) // 현재 KeyFrame의 MapPoint가 존재하는 경우
        {
            aux1++; // 가중치1 증가
            if(mCurrentFrame.mvbOutlier[i]) // MapPoint가 아웃라이너가 아닌 경우
                aux2++; // 가중치2 증가
        }

    mnMatchesInliers = 0; // 매칭 인라이너 수 초기화

    // Update MapPoints Statistics
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i]) // 현재 Frame의 MapPoint가 존재하는 경우
        {
            if(!mCurrentFrame.mvbOutlier[i]) // MapPoint가 아웃라이너가 아닌 경우
            {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound(); // MapPoint에 대해 Found 수 증가
                if(!mbOnlyTracking) // LocalMapping을 사용하는 경우
                {
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0) // 현재 Frame의 MapPoint에 대해 추적 대상이 0보다 크면
                        mnMatchesInliers++; // 매칭 인라이너 수 증가
                }
                else
                    mnMatchesInliers++; // 매칭 인라이너 수 증가
            }
            else if(mSensor==System::STEREO)
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
        }
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    mpLocalMapper->mnMatchesInliers=mnMatchesInliers; // LocalMapping 스레드의 매칭 인라이너 수를 변경
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<50) // 현재 Frame의 ID가 이전 Relocalization Frame의 ID보다 작고 매칭 인라이너 수가 50보다 작은 경우
        return false; // Tracking 실패

    if((mnMatchesInliers>10)&&(mState==RECENTLY_LOST)) // 매칭 인라이너 수가 10보다 크고 Tracking 상태가 RECENTLY_LOST인 경우
        return true; // Tracking 성공


    if (mSensor == System::IMU_MONOCULAR) // IMU를 사용하는 경우
    {
        // 인라이너 수가 15보다 작고 Atlas에서 IMU가 초기화 되었거나
        // 인라이너 수가 50보다 작고 Atlas에서 IMU가 초기화 되지 않았을 경우
        if((mnMatchesInliers<15 && mpAtlas->isImuInitialized())||(mnMatchesInliers<50 && !mpAtlas->isImuInitialized()))
        {
            return false; // Tracking 실패
        }
        else
            return true; // Tracking 성공
    }
    else if (mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
    {
        if(mnMatchesInliers<15)
        {
            return false;
        }
        else
            return true;
    }
    else
    {
        if(mnMatchesInliers<30) // 인라이너수가 30보다 작은 경우
            return false;
        else
            return true;
    }
}

bool Tracking::NeedNewKeyFrame()
{
    // IMU를 사용하고 Atlas의 현재 Map에 IMU가 초기화 되지 않았을 경우
    if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && !mpAtlas->GetCurrentMap()->isImuInitialized())
    {
        if (mSensor == System::IMU_MONOCULAR && (mCurrentFrame.mTimeStamp-mpLastKeyFrame->mTimeStamp)>=0.25) // Monocular이고 현재 Frame과 이전 keyFrame과의 시간 차이가 0.25 이상이면
            return true; // keyFrame 생성
        else if ((mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && (mCurrentFrame.mTimeStamp-mpLastKeyFrame->mTimeStamp)>=0.25)
            return true;
        else
            return false;
    }

    if(mbOnlyTracking) // LocalMapping을 사용하지 않는 경우
        return false; // keyFrame을 생성하지 않음

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested()) { // localMapping 스레드가 멈춰있는 경우
        /*if(mSensor == System::MONOCULAR)
        {
            std::cout << "NeedNewKeyFrame: localmap stopped" << std::endl;
        }*/
        return false; // keyFrame을 생성하지 않음
    }

    const int nKFs = mpAtlas->KeyFramesInMap(); // Atlas에서 Map안에 들어있는 KeyFrame 수를 가져옴

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames) // 현재 Frame의 ID보다 이전 Relocalization Frame ID가 작고 keyFrame 수가 최대 Frame 수보다 크면
    {
        return false; // keyFrame을 생성하지 않음
    }

    // Tracked MapPoints in the reference keyframe
    int nMinObs = 3; // 최소 추적 수 변수 선언
    if(nKFs<=2) // keyFrame 수가 2이하이면
        nMinObs=2; // 최소 추적 수를 2로 변경
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs); // 레퍼런스 keyFrame에 대해 MapPoint를 구하고, 레퍼런스 매칭 point 개수를 반환 받음

    // Local Mapping accept keyframes?
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames(); // localMapping 스레드에 keyFrame이 필요한지 물어봄 (localMapping 스레드가 sleep 상태가 아니면 true)

    // Check how many "close" points are being tracked and how many could be potentially created.
    int nNonTrackedClose = 0;
    int nTrackedClose= 0;

    if(mSensor!=System::MONOCULAR && mSensor!=System::IMU_MONOCULAR)
    {
        int N = (mCurrentFrame.Nleft == -1) ? mCurrentFrame.N : mCurrentFrame.Nleft;
        for(int i =0; i<N; i++)
        {
            if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth)
            {
                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                    nTrackedClose++;
                else
                    nNonTrackedClose++;

            }
        }
        //Verbose::PrintMess("[NEEDNEWKF]-> closed points: " + to_string(nTrackedClose) + "; non tracked closed points: " + to_string(nNonTrackedClose), Verbose::VERBOSITY_NORMAL);// Verbose::VERBOSITY_DEBUG);
    }

    bool bNeedToInsertClose;
    bNeedToInsertClose = (nTrackedClose<100) && (nNonTrackedClose>70); // Monocular인 경우 항상 false임 (nNonTrackedClose가 0이기 때문)

    // Thresholds
    float thRefRatio = 0.75f; // 레퍼런스 Ratio 변수 선언
    if(nKFs<2) // keyFrame 수가 2 미만인 경우
        thRefRatio = 0.4f; // Ratio 변경

    /*int nClosedPoints = nTrackedClose + nNonTrackedClose;
    const int thStereoClosedPoints = 15;
    if(nClosedPoints < thStereoClosedPoints && (mSensor==System::STEREO || mSensor==System::IMU_STEREO))
    {
        //Pseudo-monocular, there are not enough close points to be confident about the stereo observations.
        thRefRatio = 0.9f;
    }*/

    if(mSensor==System::MONOCULAR)
        thRefRatio = 0.9f;

    if(mpCamera2) thRefRatio = 0.75f;

    if(mSensor==System::IMU_MONOCULAR) // IMU Monocular인 경우
    {
        // 매칭 인라이너 수가 350보다 큰 경우
        if(mnMatchesInliers>350) // Points tracked from the local map
            thRefRatio = 0.75f; // Ratio 변경
        else
            thRefRatio = 0.90f; // Ratio 변경
    }

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames; // 현재 Frame의 ID가 (이전 keyFrame ID + 최대 Frame 수)보다 크거나 같은 경우 true

    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    // 현재 Frame의 ID가 (이전 keyFrame ID + 최소 Frame 수)보다 크거나 같고 LocalMapping 스레드가 keyFrame이 필요한 경우
    const bool c1b = ((mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames) && bLocalMappingIdle); //mpLocalMapper->KeyframesInQueue() < 2);

    //Condition 1c: tracking is weak
    // Moncular가 아니고 IMU Monocular가 아니고 매칭 안리이너 수가 레퍼런스 매칭 수보다 작은 경우 (항상 false인 조건)
    const bool c1c = mSensor!=System::MONOCULAR && mSensor!=System::IMU_MONOCULAR && mSensor!=System::IMU_STEREO && mSensor!=System::IMU_RGBD && (mnMatchesInliers<nRefMatches*0.25 || bNeedToInsertClose) ;
   
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    // 매칭 인라이너수가 레퍼런스 매칭수에 Ratio를 곱한 값보다 작고 매칭 인라이너수가 15보다 크면 (bNeedToInsertClose는 false임)
    const bool c2 = (((mnMatchesInliers<nRefMatches*thRefRatio || bNeedToInsertClose)) && mnMatchesInliers>15);

    //std::cout << "NeedNewKF: c1a=" << c1a << "; c1b=" << c1b << "; c1c=" << c1c << "; c2=" << c2 << std::endl;
    // Temporal condition for Inertial cases
    bool c3 = false;
    if(mpLastKeyFrame) // 이전 keyFrame이 존재하는 경우
    {
        if (mSensor==System::IMU_MONOCULAR) // IMU Monocular인 경우
        {
            if ((mCurrentFrame.mTimeStamp-mpLastKeyFrame->mTimeStamp)>=0.5) // 현재 Frame과 이전 KeyFrame과의 시간 차이가 0.5 이상이면
                c3 = true; // c3 조건 변경
        }
        else if (mSensor==System::IMU_STEREO || mSensor == System::IMU_RGBD)
        {
            if ((mCurrentFrame.mTimeStamp-mpLastKeyFrame->mTimeStamp)>=0.5)
                c3 = true;
        }
    }

    bool c4 = false;
    // 매칭 인라이너 수가 75보다 작고 15보다 크거나 Tracking 상태가 RECENTLY_LOST이고 IMU Monocular인 경우
    if ((((mnMatchesInliers<75) && (mnMatchesInliers>15)) || mState==RECENTLY_LOST) && (mSensor == System::IMU_MONOCULAR)) // MODIFICATION_2, originally ((((mnMatchesInliers<75) && (mnMatchesInliers>15)) || mState==RECENTLY_LOST) && ((mSensor == System::IMU_MONOCULAR)))
        c4=true; // c4 조건을 true로 변경
    else
        c4=false;

    if(((c1a||c1b||c1c) && c2)||c3 ||c4) // 위 조건에 해당한다면
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if(bLocalMappingIdle || mpLocalMapper->IsInitializing()) // LocalMapping 스레드가 sleep 상태가 아니거나 LocalMapping 스레드가 초기화 되었다면
        {
            return true; // keyFrame을 생성
        }
        else 
        {
            mpLocalMapper->InterruptBA(); // LocalMapping 스레드에 BA를 Interrupt
            if(mSensor!=System::MONOCULAR  && mSensor!=System::IMU_MONOCULAR)
            {
                if(mpLocalMapper->KeyframesInQueue()<3)
                    return true;
                else
                    return false;
            }
            else
            {
                //std::cout << "NeedNewKeyFrame: localmap is busy" << std::endl;
                return false; // keyFrame을 생성하지 않음
            }
        }
    }
    else
        return false; // keyFrame을 생성하지 않음
}

void Tracking::CreateNewKeyFrame()
{
    // LocalMapping 스레드가 초기화 되었고 Atlas에 IMU가 초기화되지 않았다면
    if(mpLocalMapper->IsInitializing() && !mpAtlas->isImuInitialized())
        return; // 함수 종료

    if(!mpLocalMapper->SetNotStop(true)) // LocalMapping 스레드가 꺠워져 있다면
        return; // 함수 종료

    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpAtlas->GetCurrentMap(),mpKeyFrameDB); // 현재 Frame으로 부터 keyFrame 생성

    // Atlas에 IMU가 초기화 되었다면
    if(mpAtlas->isImuInitialized()) //  || mpLocalMapper->IsInitializing())
        pKF->bImu = true; // keyFrame의 IMU flag를 설정

    pKF->SetNewBias(mCurrentFrame.mImuBias); // keyFrame에 현재 Frame의 bias를 저장
    mpReferenceKF = pKF; // 레퍼런스 keyFrame을 생성한 keyFrame으로 변경
    mCurrentFrame.mpReferenceKF = pKF; // 현재 Frame의 레퍼런스 keyFrame을 생성한 keyFrame으로 변경

    if(mpLastKeyFrame) // 이전 keyFrame이 존재하는 경우
    {
        pKF->mPrevKF = mpLastKeyFrame; // keyFrame의 이전 keyFrame을 이전 keyFrame으로 설정
        mpLastKeyFrame->mNextKF = pKF; // 이전 keyFrame의 다음 keyFrame을 생성한 keyFrame으로 설정
    }
    else
        Verbose::PrintMess("No last KF in KF creation!!", Verbose::VERBOSITY_NORMAL);

    // Reset preintegration from last KF (Create new object)
    if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) // IMU를 사용하는 경우
    {
        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(pKF->GetImuBias(),pKF->mImuCalib); // 이전 keyFrame기준 Preintegration 객체를 생성한 keyFrame 기준으로 재 생성
    }

    // Monocular를 사용하지 않는 경우
    if(mSensor!=System::MONOCULAR && mSensor != System::IMU_MONOCULAR) // TODO check if incluide imu_stereo
    {
        mCurrentFrame.UpdatePoseMatrices();
        // cout << "create new MPs" << endl;
        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        int maxPoint = 100;
        if(mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
            maxPoint = 100;

        vector<pair<float,int> > vDepthIdx;
        int N = (mCurrentFrame.Nleft != -1) ? mCurrentFrame.Nleft : mCurrentFrame.N;
        vDepthIdx.reserve(mCurrentFrame.N);
        for(int i=0; i<N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                vDepthIdx.push_back(make_pair(z,i));
            }
        }

        if(!vDepthIdx.empty())
        {
            sort(vDepthIdx.begin(),vDepthIdx.end());

            int nPoints = 0;
            for(size_t j=0; j<vDepthIdx.size();j++)
            {
                int i = vDepthIdx[j].second;

                bool bCreateNew = false;

                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(!pMP)
                    bCreateNew = true;
                else if(pMP->Observations()<1)
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                }

                if(bCreateNew)
                {
                    Eigen::Vector3f x3D;

                    if(mCurrentFrame.Nleft == -1){
                        mCurrentFrame.UnprojectStereo(i, x3D);
                    }
                    else{
                        x3D = mCurrentFrame.UnprojectStereoFishEye(i);
                    }

                    MapPoint* pNewMP = new MapPoint(x3D,pKF,mpAtlas->GetCurrentMap());
                    pNewMP->AddObservation(pKF,i);

                    //Check if it is a stereo observation in order to not
                    //duplicate mappoints
                    if(mCurrentFrame.Nleft != -1 && mCurrentFrame.mvLeftToRightMatch[i] >= 0){
                        mCurrentFrame.mvpMapPoints[mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]]=pNewMP;
                        pNewMP->AddObservation(pKF,mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]);
                        pKF->AddMapPoint(pNewMP,mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]);
                    }

                    pKF->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpAtlas->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    nPoints++;
                }
                else
                {
                    nPoints++;
                }

                if(vDepthIdx[j].first>mThDepth && nPoints>maxPoint)
                {
                    break;
                }
            }
            //Verbose::PrintMess("new mps for stereo KF: " + to_string(nPoints), Verbose::VERBOSITY_NORMAL);
        }
    }


    mpLocalMapper->InsertKeyFrame(pKF); // LocalMapping 스레드에 keyFrame을 추가

    mpLocalMapper->SetNotStop(false); // LocalMapping 스레드를 꺠움

    mnLastKeyFrameId = mCurrentFrame.mnId; // 이전 keyFrame ID를 현재 Frame의 ID로 변경
    mpLastKeyFrame = pKF; // 이전 keyFrame을 생성한 keyFrame으로 변경
}

void Tracking::SearchLocalPoints()
{
    // Do not search map points already matched
    // 현재 Frame의 MapPoint 배열의 크기만큼 loop
    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit; 
        if(pMP) // MapPoint가 존재하는 경우
        {
            if(pMP->isBad()) // Bad 판정인 경우
            {
                *vit = static_cast<MapPoint*>(NULL); // MapPoint 제거
            }
            else
            {
                pMP->IncreaseVisible(); // MapPoint를 보고 있는 Frame 수 증가
                pMP->mnLastFrameSeen = mCurrentFrame.mnId; // MapPoint의 관찰 이전 Frame ID를 현재 Frame ID로 변경
                pMP->mbTrackInView = false; // Tracking 시 필요한 변수 초기화
                pMP->mbTrackInViewR = false;
            }
        }
    }

    int nToMatch=0;

    // Project points in frame and check its visibility
    // LocalMapPoint 배열의 크기만큼 loop
    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit; 

        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId) // MapPoint의 이전 Frame이 현재 Frame ID와 같을 경우 무시
            continue;
        if(pMP->isBad()) // MapPoint가 Bad인 경우 무시
            continue;
        // Project (this fills MapPoint variables for matching)
        if(mCurrentFrame.isInFrustum(pMP,0.5)) // 현재 Frame에서 MapPoint에 대해 Frustum 연산을 진행하였을 때 매칭인 경우
        {
            pMP->IncreaseVisible(); // 관찰 Frame 수 증가
            nToMatch++; // 매칭 수 증가
        }
        if(pMP->mbTrackInView) // MapPoint가 Tracking 시 필요한 경우
        {
            mCurrentFrame.mmProjectPoints[pMP->mnId] = cv::Point2f(pMP->mTrackProjX, pMP->mTrackProjY); // 현재 Frame에서 투영된 Point의 좌표를 저장
        }
    }

    if(nToMatch>0) // 매칭 점이 0보다 큰 경우
    {
        ORBmatcher matcher(0.8); // ORB matcher 객체 생성
        int th = 1; // threshold 값 선언
        if(mSensor==System::RGBD || mSensor==System::IMU_RGBD)
            th=3;
        if(mpAtlas->isImuInitialized()) // Atlas에서 IMU가 초기화된 경우
        {
            if(mpAtlas->GetCurrentMap()->GetIniertialBA2()) // Atlas에서 현재 Map이 IniertialBA2가 이루어진 경우
                th=2;
            else
                th=6;
        }
        else if(!mpAtlas->isImuInitialized() && (mSensor==System::IMU_MONOCULAR || mSensor==System::IMU_STEREO || mSensor == System::IMU_RGBD)) // IMU가 초기화 되지 않았고 IMU를 사용하는 경우
        {
            th=10;
        }

        // If the camera has been relocalised recently, perform a coarser search
        if(mCurrentFrame.mnId<mnLastRelocFrameId+2) // 현재 Frame의 ID가 Relocalization Frame ID보다 작은 경우
            th=5;

        if(mState==LOST || mState==RECENTLY_LOST) // Lost for less than 1 second
            th=15; // 15

        // 현재 Frame와 LocalMapPoint에 대해 Projection을 진행 (결과는 LocalMapping 스레드에서 사용)
        int matches = matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th, mpLocalMapper->mbFarPoints, mpLocalMapper->mThFarPoints); 
    }
}

void Tracking::UpdateLocalMap()
{
    // This is for visualization
    mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints); // Atlas에서 레퍼런스 MapPoint를 검색

    // Update
    UpdateLocalKeyFrames(); // LocalKeyFrame을 찾아 update
    UpdateLocalPoints(); // LocalPoint를 찾아 배열 update
}

void Tracking::UpdateLocalPoints()
{
    mvpLocalMapPoints.clear(); // LocalMapPoint 배열을 지움

    int count_pts = 0; // point 수 저장 변수 선언

    // localKeyFrame 배열의 크기만큼 loop
    for(vector<KeyFrame*>::const_reverse_iterator itKF=mvpLocalKeyFrames.rbegin(), itEndKF=mvpLocalKeyFrames.rend(); itKF!=itEndKF; ++itKF)
    {
        KeyFrame* pKF = *itKF; // localKeyFrame을 가져옴
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches(); // keyFrame에서 매칭 MapPoint 배열을 가져옴

        // 매칭 MapPoint 배열의 크기만큼 loop
        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {

            MapPoint* pMP = *itMP; // 매칭 MapPoint를 가져옴
            if(!pMP) // 존재하지 않는 경우 무시
                continue;
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId) // MapPoint의 레퍼런스 Frame의 ID가 현재 Frame의 ID와 같을 경우 무시
                continue;
            if(!pMP->isBad()) // MapPoint가 Bad 판정이 아닌 경우
            {
                count_pts++; // Point 수 증가
                mvpLocalMapPoints.push_back(pMP); // LocalMapPoint 배열에 삽입
                pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId; // MapPoint의 레퍼런스 Frame의 ID를 현재 Frame의 ID로 변경
            } 
        }
    }
}


void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    map<KeyFrame*,int> keyframeCounter;
    if(!mpAtlas->isImuInitialized() || (mCurrentFrame.mnId<mnLastRelocFrameId+2)) // Atlas에 IMU가 초기화 되지 않거나 현재 Frame의 ID가 Relocalization Frame ID 보다 작을 경우
    {
        for(int i=0; i<mCurrentFrame.N; i++) // 현재 Frame의 KeyPoint 수만큼 loop
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i]; // MapPoint를 가져옴
            if(pMP) // MapPoint가 존재하는 경우
            {
                if(!pMP->isBad()) // Bad 판정이 아닌 경우
                {
                    const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations(); // MapPoint에서 추적 대상을 가져옴
                    for(map<KeyFrame*,tuple<int,int>>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                        keyframeCounter[it->first]++; // 추적 대상 keyFrame에 대해 counter 증가
                }
                else
                {
                    mCurrentFrame.mvpMapPoints[i]=NULL; // 현재 Frame의 MapPoint를 제거
                }
            }
        }
    }
    else
    {
        for(int i=0; i<mLastFrame.N; i++) // 이전 Frame의 keyPoint 수 만큼 loop
        {
            // Using lastframe since current frame has not matches yet
            if(mLastFrame.mvpMapPoints[i]) // 이전 Frame의 MapPoint가 존재하는 경우
            {
                MapPoint* pMP = mLastFrame.mvpMapPoints[i]; // MapPoint를 가져옴
                if(!pMP) // MapPoint가 존재하지 않으면 무시
                    continue;
                if(!pMP->isBad()) // Bad 판정이 아닌 경우
                {
                    const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations(); // MapPoint에서 추적 대상을 가져옴
                    for(map<KeyFrame*,tuple<int,int>>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                        keyframeCounter[it->first]++; // 추적 대상 keyFrame에 대해 counter 증가
                }
                else
                {
                    // MODIFICATION
                    mLastFrame.mvpMapPoints[i]=NULL; // 이전 Frame의 MapPoint를 제거
                }
            }
        }
    }


    int max=0;
    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL);

    mvpLocalKeyFrames.clear(); // localKeyFrame 배열 초기화
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size()); // localKeyFrame 배열의 크기를 keyFrame counter에 저장된 keyFrame 개수의 3배로 재할당

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    // keyFrameCounter에 저장된 keyFrame 개수만큼 loop
    for(map<KeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {
        KeyFrame* pKF = it->first; // keyFrame을 가져옴

        if(pKF->isBad()) // keyFrame이 Bad 판정인 경우 무시
            continue;

        if(it->second>max) // counter가 max 값보다 큰 경우
        {
            max=it->second; // max값을 counter에 저장된 값으로 변경
            pKFmax=pKF; // max값을 가진 keyFrame을 변경
        }

        mvpLocalKeyFrames.push_back(pKF); // LocalKeyFrame 배열에 삽입
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId; // keyFrame의 레퍼런스 Frame ID를 현재 Frame의 ID로 변경
    }

    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    // LocalKeyFrame 배열의 크기만큼 loop
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if(mvpLocalKeyFrames.size()>80) // 80
            break; // localKeyFrame의 배열 크기가 80보다 크면 반복 종료

        KeyFrame* pKF = *itKF; // Local keyFrame을 가져옴

        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10); // keyFrame에 대해 Covisbility를 구함 (keyFrame에 대해 아직 구하지 못한 이웃을 더 구함)

        // 이웃 후보로 구해진 keyFrame의 개수 만큼 loop
        for(vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad()) // Bad 상태가 아닌 경우
            {
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId) // 이웃 후보인 keyFrame의 레퍼런스 Frame ID가 현재 Frame의 ID와 다른 경우
                {
                    mvpLocalKeyFrames.push_back(pNeighKF); // LocalKeyFrame 배열에 이웃인 KeyFrame을 삽입
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId; // 이웃 keyFrame의 레퍼런스 Frame ID를 현재 Frame의 ID로 변경
                    break; // 반복 종료
                }
            }
        }

        const set<KeyFrame*> spChilds = pKF->GetChilds(); // Local keyFrame에서 자식 KeyFrame을 가져옴
        // 자식 keyFrame의 수만큼 loop
        for(set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
        {
            KeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad()) // Bad 상태가 아닌 경우
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId) // 자식 KeyFrame의 레퍼런스 Frame ID가 현재 Frame ID와 다른 경우
                {
                    mvpLocalKeyFrames.push_back(pChildKF); // LocalKeyFrame 배열에 자식 KeyFrame을 삽입
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId; // 자식 keyFrame의 레퍼런스 Frame ID를 현재 Frame의 ID로 변경
                    break; // 반복 종료
                }
            }
        }

        KeyFrame* pParent = pKF->GetParent(); // Local keyFrame에서 부모 KeyFrame을 가져옴
        if(pParent) // 부모 keyFrame이 존재하는 경우
        {
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId) // 부모 KeyFrame의 레퍼런스 Frame ID가 현재 Frame ID와 다른 경우
            {
                mvpLocalKeyFrames.push_back(pParent); // LocalKeyFrame 배열에 부모 KeyFrame을 삽입
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId; // 부모 keyFrame의 레퍼런스 Frame ID를 현재 Frame의 ID로 변경
                break; // 반복 종료
            }
        }
    }

    // Add 10 last temporal KFs (mainly for IMU)
    // IMU를 사용하고 localKeyFrame 배열의 크기가 80보다 작은 경우
    if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) &&mvpLocalKeyFrames.size()<80)
    {
        KeyFrame* tempKeyFrame = mCurrentFrame.mpLastKeyFrame; // 현재 Frame의 이전 KeyFrame을 가져옴

        const int Nd = 20;
        for(int i=0; i<Nd; i++){
            if (!tempKeyFrame) // 임시 keyFrame이 존재하지 않는 경우 반복 종료
                break;
            if(tempKeyFrame->mnTrackReferenceForFrame!=mCurrentFrame.mnId) // 임시 KeyFrame의 레퍼런스 Frame ID가 현재 Frame ID와 다른 경우
            {
                mvpLocalKeyFrames.push_back(tempKeyFrame); // LocalKeyFrame 배열에 임시 KeyFrame을 삽입
                tempKeyFrame->mnTrackReferenceForFrame=mCurrentFrame.mnId; // 임시 keyFrame의 레퍼런스 Frame ID를 현재 Frame의 ID로 변경
                tempKeyFrame=tempKeyFrame->mPrevKF; // 임시 keyFrame을 임시 KeyFrame의 이전 KeyFrame으로 변경
            }
        }
    }

    if(pKFmax) // max 값을 가진 KeyFrame이 존재하는 경우
    {
        mpReferenceKF = pKFmax; // 레퍼런스 KeyFrame을 max값 keyFrame으로 변경
        mCurrentFrame.mpReferenceKF = mpReferenceKF; // 현재 Frame의 레퍼런스 KeyFrame을 max값 keyFrame으로 변경
    }
}

bool Tracking::Relocalization()
{
    Verbose::PrintMess("Starting relocalization", Verbose::VERBOSITY_NORMAL);
    // Compute Bag of Words Vector
    mCurrentFrame.ComputeBoW(); // 현재 Frame에 대해서 Bag of Word 계산

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation

    // 현재 Frame과 현재 Map을 통해 KeyFrame Database에서 Candidates 감지
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame, mpAtlas->GetCurrentMap());

    if(vpCandidateKFs.empty()) { // candidate keyFrame 배열이 비어있는 경우
        Verbose::PrintMess("There are not candidates", Verbose::VERBOSITY_NORMAL);
        return false; // Relocalization 실패
    }

    const int nKFs = vpCandidateKFs.size(); // candidate keyFrame의 수 저장

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75,true); // ORB matcher 객체 생성

    vector<MLPnPsolver*> vpMLPnPsolvers; // PnPsolver 배열 생성
    vpMLPnPsolvers.resize(nKFs);

    vector<vector<MapPoint*> > vvpMapPointMatches; // 매칭 MapPoint 배열 생성
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded; // Discard 배열 생성
    vbDiscarded.resize(nKFs);

    int nCandidates=0; // candidate 수를 저장할 변수 선언

    for(int i=0; i<nKFs; i++) // keyFrame 수 만큼 loop
    {
        KeyFrame* pKF = vpCandidateKFs[i]; // candidate keyFrame을 가져옴
        if(pKF->isBad()) // Bad 판정인 경우
            vbDiscarded[i] = true; // Discard로 설정
        else
        {
            int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]); // keyFrame과 현재 Frame간의 Bag of Word를 통해 매칭 Point를 구함
            if(nmatches<15) // 매칭 수가 15 미만인 경우
            {
                vbDiscarded[i] = true; // Discard로 설정
                continue;
            }
            else
            {
                MLPnPsolver* pSolver = new MLPnPsolver(mCurrentFrame,vvpMapPointMatches[i]); // PnpSolver 객체 생성
                // Ransac 진행
                pSolver->SetRansacParameters(0.99,10,300,6,0.5,5.991);  //This solver needs at least 6 points
                vpMLPnPsolvers[i] = pSolver; // PnpSolver 배열에 solver를 저장
                nCandidates++; // Candidate 수 증가
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9,true); // ORB matcher 객체 생성

    while(nCandidates>0 && !bMatch) // Candidate 수가 0보다 크고 매칭이 아직 이루어지지 않으면 loop
    {
        for(int i=0; i<nKFs; i++) // keyFrame 수 만큼 loop
        {
            if(vbDiscarded[i]) // Discard인 경우 무시
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            MLPnPsolver* pSolver = vpMLPnPsolvers[i];
            Eigen::Matrix4f eigTcw;
            bool bTcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers, eigTcw); // Ransac를 통해 inlier, Tcw 추정

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore) // Ransac이 최대 만큼 반복되었다면
            {
                vbDiscarded[i]=true; // Discard로 설정
                nCandidates--; // Candidate 수 감소
            }

            // If a Camera Pose is computed, optimize
            if(bTcw) // Tcw가 존재하는 경우
            {
                Sophus::SE3f Tcw(eigTcw);
                mCurrentFrame.SetPose(Tcw); // 현재 Frame의 pose로 설정
                // Tcw.copyTo(mCurrentFrame.mTcw);

                set<MapPoint*> sFound;

                const int np = vbInliers.size();

                for(int j=0; j<np; j++) // 인라이어 수 만큼 loop
                {
                    if(vbInliers[j]) // 인라이어인 경우
                    {
                        mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j]; // 현재 Frame의 MapPoint로 설정
                        sFound.insert(vvpMapPointMatches[i][j]); // 찾은 인라이어 MapPoint로 추가
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j]=NULL; // 현재 Frame의 MapPoint를 제거
                }

                int nGood = Optimizer::PoseOptimization(&mCurrentFrame); // 현재 Frame에 대해 Pose 최적화 진행

                if(nGood<10) // 인라이어의 수가 10 미만인 경우 무시
                    continue;

                for(int io =0; io<mCurrentFrame.N; io++) // 현재 Frame의 KeyPoint수 만큼 loop
                    if(mCurrentFrame.mvbOutlier[io]) // 아웃라이너의 경우
                        mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL); // MapPoint 제거

                // If few inliers, search by projection in a coarse window and optimize again
                if(nGood<50) // 인라이어의 수가 50 미만인 경우
                {
                    int nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,100); // 현재 Frame과 Candidate KeyFrame에 대해 Projection 진행

                    if(nadditional+nGood>=50) // 매칭 수와 인라이어의 합이 50 이상인 경우
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame); // 현재 Frame에 대해 pose 최적화

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if(nGood>30 && nGood<50) // 인라이어의 수가 30보다 크고 50 보다 작으면
                        {
                            sFound.clear(); // 인라이어 배열 초기화
                            for(int ip =0; ip<mCurrentFrame.N; ip++)
                                if(mCurrentFrame.mvpMapPoints[ip]) // Map Point가 존재하는 경우
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]); // 인라이어 배열에 MapPoint 삽입
                            nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,3,64); // Projection 다시 진행 (window 크기를 줄여서 다시 진행)

                            // Final optimization
                            if(nGood+nadditional>=50) // 매칭 수와 인라이어의 합이 50 이상인 경우
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame); // pose 최적화 진행

                                for(int io =0; io<mCurrentFrame.N; io++)
                                    if(mCurrentFrame.mvbOutlier[io]) // 아웃라이너인 경우
                                        mCurrentFrame.mvpMapPoints[io]=NULL; // MapPoint에서 제거
                            }
                        }
                    }
                }


                // If the pose is supported by enough inliers stop ransacs and continue
                if(nGood>=50) // 인라이어가 50 이상인 경우
                {
                    bMatch = true; // 매칭을 찾았다고 설정
                    break; // RANSAC 종료
                }
            }
        }
    }

    if(!bMatch) // 매칭을 찾지 못했을 경우
    {
        return false; // Relocalization 실패
    }
    else
    {
        mnLastRelocFrameId = mCurrentFrame.mnId; // 이전 Relocalization Frame ID를 현재 Frame의 ID로 초기화
        cout << "Relocalized!!" << endl;
        return true; // Relocalization 성공
    }

}

void Tracking::Reset(bool bLocMap)
{
    Verbose::PrintMess("System Reseting", Verbose::VERBOSITY_NORMAL);

    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }

    // Reset Local Mapping
    if (!bLocMap)
    {
        Verbose::PrintMess("Reseting Local Mapper...", Verbose::VERBOSITY_NORMAL);
        mpLocalMapper->RequestReset();
        Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);
    }


    // Reset Loop Closing
    Verbose::PrintMess("Reseting Loop Closing...", Verbose::VERBOSITY_NORMAL);
    mpLoopClosing->RequestReset();
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear BoW Database
    Verbose::PrintMess("Reseting Database...", Verbose::VERBOSITY_NORMAL);
    mpKeyFrameDB->clear();
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear Map (this erase MapPoints and KeyFrames)
    mpAtlas->clearAtlas();
    mpAtlas->CreateNewMap();
    if (mSensor==System::IMU_STEREO || mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_RGBD)
        mpAtlas->SetInertialSensor();
    mnInitialFrameId = 0;

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    mbReadyToInitializate = false;
    mbSetInit=false;

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();
    mCurrentFrame = Frame();
    mnLastRelocFrameId = 0;
    mLastFrame = Frame();
    mpReferenceKF = static_cast<KeyFrame*>(NULL);
    mpLastKeyFrame = static_cast<KeyFrame*>(NULL);
    mvIniMatches.clear();

    if(mpViewer)
        mpViewer->Release();

    Verbose::PrintMess("   End reseting! ", Verbose::VERBOSITY_NORMAL);
}

void Tracking::ResetActiveMap(bool bLocMap)
{
    Verbose::PrintMess("Active map Reseting", Verbose::VERBOSITY_NORMAL);
    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }

    Map* pMap = mpAtlas->GetCurrentMap();

    if (!bLocMap)
    {
        Verbose::PrintMess("Reseting Local Mapper...", Verbose::VERBOSITY_VERY_VERBOSE);
        mpLocalMapper->RequestResetActiveMap(pMap);
        Verbose::PrintMess("done", Verbose::VERBOSITY_VERY_VERBOSE);
    }

    // Reset Loop Closing
    Verbose::PrintMess("Reseting Loop Closing...", Verbose::VERBOSITY_NORMAL);
    mpLoopClosing->RequestResetActiveMap(pMap);
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear BoW Database
    Verbose::PrintMess("Reseting Database", Verbose::VERBOSITY_NORMAL);
    mpKeyFrameDB->clearMap(pMap); // Only clear the active map references
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear Map (this erase MapPoints and KeyFrames)
    mpAtlas->clearMap();


    //KeyFrame::nNextId = mpAtlas->GetLastInitKFid();
    //Frame::nNextId = mnLastInitFrameId;
    mnLastInitFrameId = Frame::nNextId;
    //mnLastRelocFrameId = mnLastInitFrameId;
    mState = NO_IMAGES_YET; //NOT_INITIALIZED;

    mbReadyToInitializate = false;

    list<bool> lbLost;
    // lbLost.reserve(mlbLost.size());
    unsigned int index = mnFirstFrameId;
    cout << "mnFirstFrameId = " << mnFirstFrameId << endl;
    for(Map* pMap : mpAtlas->GetAllMaps())
    {
        if(pMap->GetAllKeyFrames().size() > 0)
        {
            if(index > pMap->GetLowerKFID())
                index = pMap->GetLowerKFID();
        }
    }

    //cout << "First Frame id: " << index << endl;
    int num_lost = 0;
    cout << "mnInitialFrameId = " << mnInitialFrameId << endl;

    for(list<bool>::iterator ilbL = mlbLost.begin(); ilbL != mlbLost.end(); ilbL++)
    {
        if(index < mnInitialFrameId)
            lbLost.push_back(*ilbL);
        else
        {
            lbLost.push_back(true);
            num_lost += 1;
        }

        index++;
    }
    cout << num_lost << " Frames set to lost" << endl;

    mlbLost = lbLost;

    mnInitialFrameId = mCurrentFrame.mnId;
    mnLastRelocFrameId = mCurrentFrame.mnId;

    mCurrentFrame = Frame();
    mLastFrame = Frame();
    mpReferenceKF = static_cast<KeyFrame*>(NULL);
    mpLastKeyFrame = static_cast<KeyFrame*>(NULL);
    mvIniMatches.clear();

    mbVelocity = false;

    if(mpViewer)
        mpViewer->Release();

    Verbose::PrintMess("   End reseting! ", Verbose::VERBOSITY_NORMAL);
}

vector<MapPoint*> Tracking::GetLocalMapMPS()
{
    return mvpLocalMapPoints;
}

void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    mK_.setIdentity();
    mK_(0,0) = fx;
    mK_(1,1) = fy;
    mK_(0,2) = cx;
    mK_(1,2) = cy;

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}

void Tracking::UpdateFrameIMU(const float s, const IMU::Bias &b, KeyFrame* pCurrentKeyFrame)
{
    Map * pMap = pCurrentKeyFrame->GetMap(); // 현재 KeyFrame의 Map을 가져옴
    unsigned int index = mnFirstFrameId; // 첫번째 Frame의 ID를 index로 초기화
    list<ORB_SLAM3::KeyFrame*>::iterator lRit = mlpReferences.begin(); // 레퍼런스 keyFrame 배열의 반복자 생성
    list<bool>::iterator lbL = mlbLost.begin(); // Tracking 상태 배열의 반복자 생성
    for(auto lit=mlRelativeFramePoses.begin(),lend=mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lbL++) // Frame pose 배열의 크기만큼 loop
    {
        if(*lbL) // Tracking 상태가 존재한다면 무시
            continue;

        KeyFrame* pKF = *lRit; // 레퍼런스 keyFrame을 가져옴

        while(pKF->isBad()) // 레퍼런스 keyFrame이 Bad 판정인 경우 loop
        {
            pKF = pKF->GetParent(); // 레퍼런스 keyFrame의 부모 keyFrame을 가져와 레퍼런스 keyFrame으로 갱신
        }

        if(pKF->GetMap() == pMap) // 레퍼런스 keyFrame의 Map이 현재 KeyFrame의 Map과 동일한 경우
        {
            (*lit).translation() *= s; // Frame pose에 scale 값을 곱함
        }
    }

    mLastBias = b; // 인자로 들어온 bais를 이전 bias로 초기화

    mpLastKeyFrame = pCurrentKeyFrame; // 이전 keyFrame을 현재 KeyFrame으로 초기화

    mLastFrame.SetNewBias(mLastBias); // 이전 Frame의 새로운 bias를 이전 bias로 설정
    mCurrentFrame.SetNewBias(mLastBias); // 현재 Frame의 새로운 bias를 이전 bias로 설정

    while(!mCurrentFrame.imuIsPreintegrated()) // 현재 Frame가 Preintegration이 이루어지지 않았을 경우 loop
    {
        usleep(500);
    }


    if(mLastFrame.mnId == mLastFrame.mpLastKeyFrame->mnFrameId) // 이전 Frame의 id가 이전 Frame의 이전 KeFrame의 ID와 같을 경우
    {
        mLastFrame.SetImuPoseVelocity(mLastFrame.mpLastKeyFrame->GetImuRotation(),
                                      mLastFrame.mpLastKeyFrame->GetImuPosition(),
                                      mLastFrame.mpLastKeyFrame->GetVelocity()); // 이전 Frame의 IMU pose 및 속도 정보를 이전 KeyFrame의 pose 및 속도로 변경
    }
    else
    {
        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE); // 중력 가속도 행렬 생성
        const Eigen::Vector3f twb1 = mLastFrame.mpLastKeyFrame->GetImuPosition(); // 이전 Frame의 이전 KeyFrame의 IMU 행렬을 가져옴
        const Eigen::Matrix3f Rwb1 = mLastFrame.mpLastKeyFrame->GetImuRotation();
        const Eigen::Vector3f Vwb1 = mLastFrame.mpLastKeyFrame->GetVelocity(); // 이전 Frame의 이전 KeyFrame의 속도 정보를 가져옴
        float t12 = mLastFrame.mpImuPreintegrated->dT; // 이전 Frame의 시간 차이의 누적 합을 가져옴

        mLastFrame.SetImuPoseVelocity(IMU::NormalizeRotation(Rwb1*mLastFrame.mpImuPreintegrated->GetUpdatedDeltaRotation()),
                                      twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*mLastFrame.mpImuPreintegrated->GetUpdatedDeltaPosition(),
                                      Vwb1 + Gz*t12 + Rwb1*mLastFrame.mpImuPreintegrated->GetUpdatedDeltaVelocity()); // 이전 Frame의 IMU pose 및 속도 정보를 이전 KeyFrame의 갱신된 pose 및 속도와 곱하여 변경
    }

    if (mCurrentFrame.mpImuPreintegrated) // 현재 Frame의 Preintegration 객체가 존재하는 경우
    {
        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE); // 중력 가속도 행렬 생성

        const Eigen::Vector3f twb1 = mCurrentFrame.mpLastKeyFrame->GetImuPosition(); // 현재 Frame의 이전 KeyFrame의 IMU 행렬을 가져옴
        const Eigen::Matrix3f Rwb1 = mCurrentFrame.mpLastKeyFrame->GetImuRotation();
        const Eigen::Vector3f Vwb1 = mCurrentFrame.mpLastKeyFrame->GetVelocity(); // 현재 Frame의 이전 KeyFrame의 속도 정보를 가져옴
        float t12 = mCurrentFrame.mpImuPreintegrated->dT; // 현재 Frame의 시간 차이의 누적 합을 가져옴

        mCurrentFrame.SetImuPoseVelocity(IMU::NormalizeRotation(Rwb1*mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaRotation()),
                                      twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaPosition(),
                                      Vwb1 + Gz*t12 + Rwb1*mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaVelocity()); // 현재 Frame의 IMU pose 및 속도 정보를 이전 KeyFrame의 갱신된 pose 및 속도와 곱하여 변경
    }

    mnFirstImuFrameId = mCurrentFrame.mnId; // 첫번째 IMU Frame ID를 현재 Frame의 ID로 초기화
}

void Tracking::NewDataset()
{
    mnNumDataset++;
}

int Tracking::GetNumberDataset()
{
    return mnNumDataset;
}

int Tracking::GetMatchesInliers()
{
    return mnMatchesInliers;
}

void Tracking::SaveSubTrajectory(string strNameFile_frames, string strNameFile_kf, string strFolder)
{
    mpSystem->SaveTrajectoryEuRoC(strFolder + strNameFile_frames);
    //mpSystem->SaveKeyFrameTrajectoryEuRoC(strFolder + strNameFile_kf);
}

void Tracking::SaveSubTrajectory(string strNameFile_frames, string strNameFile_kf, Map* pMap)
{
    mpSystem->SaveTrajectoryEuRoC(strNameFile_frames, pMap);
    if(!strNameFile_kf.empty())
        mpSystem->SaveKeyFrameTrajectoryEuRoC(strNameFile_kf, pMap);
}

float Tracking::GetImageScale()
{
    return mImageScale;
}

#ifdef REGISTER_LOOP
void Tracking::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
}

bool Tracking::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        cout << "Tracking STOP" << endl;
        return true;
    }

    return false;
}

bool Tracking::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

bool Tracking::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

void Tracking::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopped = false;
    mbStopRequested = false;
}
#endif

} //namespace ORB_SLAM
