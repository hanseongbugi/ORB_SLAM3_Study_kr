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


#include "LocalMapping.h"
#include "LoopClosing.h"
#include "ORBmatcher.h"
#include "Optimizer.h"
#include "Converter.h"
#include "GeometricTools.h"

#include<mutex>
#include<chrono>

namespace ORB_SLAM3
{

LocalMapping::LocalMapping(System* pSys, Atlas *pAtlas, const float bMonocular, bool bInertial, const string &_strSeqName):
    mpSystem(pSys), mbMonocular(bMonocular), mbInertial(bInertial), mbResetRequested(false), mbResetRequestedActiveMap(false), mbFinishRequested(false), mbFinished(true), mpAtlas(pAtlas), bInitializing(false),
    mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false), mbAcceptKeyFrames(true),
    mIdxInit(0), mScale(1.0), mInitSect(0), mbNotBA1(true), mbNotBA2(true), mIdxIteration(0), infoInertial(Eigen::MatrixXd::Zero(9,9))
{
    mnMatchesInliers = 0;

    mbBadImu = false;

    mTinit = 0.f;

    mNumLM = 0;
    mNumKFCulling=0;

#ifdef REGISTER_TIMES
    nLBA_exec = 0;
    nLBA_abort = 0;
#endif

}

void LocalMapping::SetLoopCloser(LoopClosing* pLoopCloser)
{
    mpLoopCloser = pLoopCloser;
}

void LocalMapping::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

void LocalMapping::Run()
{
    mbFinished = false;

    while(1)
    {
        // Tracking will see that Local Mapping is busy
        SetAcceptKeyFrames(false); // Tracking 스레드에게 keyFrame에 접근하지 않는다고 설정

        // Check if there are keyframes in the queue
        if(CheckNewKeyFrames() && !mbBadImu) // 새로운 keyFrame이 있는지 확인하고, IMU가 bad 상태가 아니면
        {
#ifdef REGISTER_TIMES
            double timeLBA_ms = 0;
            double timeKFCulling_ms = 0;

            std::chrono::steady_clock::time_point time_StartProcessKF = std::chrono::steady_clock::now();
#endif
            // BoW conversion and insertion in Map
            ProcessNewKeyFrame(); // 현재 KeyFrame에 대해 Bag of Word를 계산하고 MapPoint를 구함
#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndProcessKF = std::chrono::steady_clock::now();

            double timeProcessKF = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndProcessKF - time_StartProcessKF).count();
            vdKFInsert_ms.push_back(timeProcessKF);
#endif

            // Check recent MapPoints
            MapPointCulling(); // 최근 MapPoint를 검사하여 불필요한 MapPoint를 제거
#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndMPCulling = std::chrono::steady_clock::now();

            double timeMPCulling = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndMPCulling - time_EndProcessKF).count();
            vdMPCulling_ms.push_back(timeMPCulling);
#endif

            // Triangulate new MapPoints
            CreateNewMapPoints(); // Triangulate를 통해 현재 KeyFrame과 이웃 KeyFrame간의 MaPoint를 생성

            mbAbortBA = false; // BA 중단 하지 않도록 flag 설정

            if(!CheckNewKeyFrames()) // 새로운 keyFrame 배열에 keyFrame이 존재하지 않다면
            {
                // Find more matches in neighbor keyframes and fuse point duplications
                SearchInNeighbors(); // 현재 KeyFrame의 이웃 KeyFrame간에 매칭 MapPoint의 Fusion을 진행하고 Covisibility Graph를 갱신
            }

#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndMPCreation = std::chrono::steady_clock::now();

            double timeMPCreation = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndMPCreation - time_EndMPCulling).count();
            vdMPCreation_ms.push_back(timeMPCreation);
#endif

            bool b_doneLBA = false;
            int num_FixedKF_BA = 0;
            int num_OptKF_BA = 0;
            int num_MPs_BA = 0;
            int num_edges_BA = 0;

            // 새로운 KeyFrame 배열에 KeyFrame이 존재하지 않고 LocalMapping 스레드에 중단 요청이 들어오지 않았을 경우
            if(!CheckNewKeyFrames() && !stopRequested())
            {
                if(mpAtlas->KeyFramesInMap()>2) // Atlas에 KeyFrame이 2개 보다 많을 경우
                {
                    // IMU를 사용하고 현재 KeyFrame의 Map에 IMU가 초기화 되었을 경우
                    if(mbInertial && mpCurrentKeyFrame->GetMap()->isImuInitialized())
                    {
                        // 현재 KeyFrame의 이전 KeyFrame과 현재 KeyFrame의 Camera Center 좌표의 차이 값과
                        // 현재 KeyFrame의 이전 KeyFrame의 이전 KeyFrame과 현재 KeyFrame의 이전 KeyFrame의 Camera Center 좌표의 차이 값의 합을 거리 값으로 결정
                        float dist = (mpCurrentKeyFrame->mPrevKF->GetCameraCenter() - mpCurrentKeyFrame->GetCameraCenter()).norm() +
                                (mpCurrentKeyFrame->mPrevKF->mPrevKF->GetCameraCenter() - mpCurrentKeyFrame->mPrevKF->GetCameraCenter()).norm();

                        if(dist>0.05) // 거리 값이 0.05 보다 클 경우
                            mTinit += mpCurrentKeyFrame->mTimeStamp - mpCurrentKeyFrame->mPrevKF->mTimeStamp; // 현재 KeyFrame과 이전 KeyFrame과의 Timestamp 차이를 누적
                        if(!mpCurrentKeyFrame->GetMap()->GetIniertialBA2()) // 현재 KeyFrame의 Map이 InieritalBA2가 일어나지 않았을 경우
                        {
                            if((mTinit<10.f) && (dist<0.02)) // Timestamp 누적 합이 10 보다 작고 거리 값이 0.02 보다 작을 경우
                            {
                                cout << "Not enough motion for initializing. Reseting..." << endl; // 충분한 움직임이 없다고 출력
                                unique_lock<mutex> lock(mMutexReset);
                                mbResetRequestedActiveMap = true; // 활성화 Map을 초기화하는 Flag 설정
                                mpMapToReset = mpCurrentKeyFrame->GetMap(); // 리셋할 Map을 현재 KeyFrame에서 가져옴
                                mbBadImu = true; // Bad IMU flag 설정
                            }
                        }
                        // Tracking 스레드의 매칭 인라이너가 75개 보다 많고 Monocular를 사용하거나
                        // Tracking 스레드의 매칭 인라이너가 100개보다 많고 Monocular를 사용하지 않는다면 True로 설정
                        bool bLarge = ((mpTracker->GetMatchesInliers()>75)&&mbMonocular)||((mpTracker->GetMatchesInliers()>100)&&!mbMonocular);
                        // 센서 정보를 이용한 Local Bundle Adjustment를 진행
                        Optimizer::LocalInertialBA(mpCurrentKeyFrame, &mbAbortBA, mpCurrentKeyFrame->GetMap(),num_FixedKF_BA,num_OptKF_BA,num_MPs_BA,num_edges_BA, bLarge, !mpCurrentKeyFrame->GetMap()->GetIniertialBA2());
                        b_doneLBA = true; // Local BA가 이루어졌다고 Flag 설정
                    }
                    else // IMU를 사용하지 않거나 Map에 IMU가 초기화 되지 않았을 경우
                    {
                        // Local Bundle Adjustment를 진행
                        Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame,&mbAbortBA, mpCurrentKeyFrame->GetMap(),num_FixedKF_BA,num_OptKF_BA,num_MPs_BA,num_edges_BA); 
                        b_doneLBA = true; // Local BA가 이루어졌다고 Flag 설정
                    }

                }
#ifdef REGISTER_TIMES
                std::chrono::steady_clock::time_point time_EndLBA = std::chrono::steady_clock::now();

                if(b_doneLBA)
                {
                    timeLBA_ms = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndLBA - time_EndMPCreation).count();
                    vdLBA_ms.push_back(timeLBA_ms);

                    nLBA_exec += 1;
                    if(mbAbortBA)
                    {
                        nLBA_abort += 1;
                    }
                    vnLBA_edges.push_back(num_edges_BA);
                    vnLBA_KFopt.push_back(num_OptKF_BA);
                    vnLBA_KFfixed.push_back(num_FixedKF_BA);
                    vnLBA_MPs.push_back(num_MPs_BA);
                }

#endif

                // Initialize IMU here
                if(!mpCurrentKeyFrame->GetMap()->isImuInitialized() && mbInertial) // 현재 KeyFrame의 Map에 IMU가 초기화 되지 않았고 IMU를 사용하는 경우
                {
                    if (mbMonocular) // Monocular인 경우
                        InitializeIMU(1e2, 1e10, true); // IMU 초기화 진행
                    else
                        InitializeIMU(1e2, 1e5, true);
                }


                // Check redundant local Keyframes
                KeyFrameCulling(); // 중복 local KeyFrame을 검사 (많은 MapPoint가 중복될 경우 Bad KeyFrame으로 결정)

#ifdef REGISTER_TIMES
                std::chrono::steady_clock::time_point time_EndKFCulling = std::chrono::steady_clock::now();

                timeKFCulling_ms = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndKFCulling - time_EndLBA).count();
                vdKFCulling_ms.push_back(timeKFCulling_ms);
#endif

                if ((mTinit<50.0f) && mbInertial) // 시간차이의 누적 합이 50보다 작고 IMU를 사용하는 경우
                {
                    // 현재 KeyFrame의 Map에 IMU가 초기화 되었고 Tracking 상태가 OK인 경우
                    if(mpCurrentKeyFrame->GetMap()->isImuInitialized() && mpTracker->mState==Tracking::OK) // Enter here everytime local-mapping is called
                    {
                        if(!mpCurrentKeyFrame->GetMap()->GetIniertialBA1()){ // 현재 KeyFrame의 Map에 IniertialBA1이 이루어지지 않았다면
                            if (mTinit>5.0f) // 시간차이의 누적 합이 5보다 큰 경우
                            {
                                cout << "start VIBA 1" << endl;
                                mpCurrentKeyFrame->GetMap()->SetIniertialBA1(); // 현재 KeyFrame의 Map에 IniertialBA1이 이루어졌다고 설정
                                if (mbMonocular) // Monocular인 경우
                                    InitializeIMU(1.f, 1e5, true); // IMU 초기화 진행
                                else
                                    InitializeIMU(1.f, 1e5, true);

                                cout << "end VIBA 1" << endl;
                            }
                        }
                        else if(!mpCurrentKeyFrame->GetMap()->GetIniertialBA2()){ // 현재 KeyFrame의 Map에 IniertialBA2가 이루어지지 않았다면
                            if (mTinit>15.0f){ // 시간차이의 누적 합이 15보다 큰 경우
                                cout << "start VIBA 2" << endl;
                                mpCurrentKeyFrame->GetMap()->SetIniertialBA2(); // 현재 KeyFrame의 Map에 IniertialBA2가 이루어졌다고 설정
                                if (mbMonocular) // Monocular인 경우
                                    InitializeIMU(0.f, 0.f, true); // IMU 초기화 진행
                                else
                                    InitializeIMU(0.f, 0.f, true);

                                cout << "end VIBA 2" << endl;
                            }
                        }

                        // scale refinement
                        // Atlas의 Map에 KeyFrame이 200개 이하 존재하고 시간 차이의 누적 합이 25 혹은 35, 45, 55, 65, 75 사이인 경우
                        if (((mpAtlas->KeyFramesInMap())<=200) &&
                                ((mTinit>25.0f && mTinit<25.5f)||
                                (mTinit>35.0f && mTinit<35.5f)||
                                (mTinit>45.0f && mTinit<45.5f)||
                                (mTinit>55.0f && mTinit<55.5f)||
                                (mTinit>65.0f && mTinit<65.5f)||
                                (mTinit>75.0f && mTinit<75.5f))){
                            if (mbMonocular) // Monocular인 경우
                                ScaleRefinement(); // Atlas에 있는 Map의 Scale을 미세 조정
                        }
                    }
                }
            }

#ifdef REGISTER_TIMES
            vdLBASync_ms.push_back(timeKFCulling_ms);
            vdKFCullingSync_ms.push_back(timeKFCulling_ms);
#endif

            mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame); // LoopCloser 스레드에 현재 KeyFrame을 삽입함

#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndLocalMap = std::chrono::steady_clock::now();

            double timeLocalMap = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndLocalMap - time_StartProcessKF).count();
            vdLMTotal_ms.push_back(timeLocalMap);
#endif
        }
        else if(Stop() && !mbBadImu)
        {
            // Safe area to stop
            while(isStopped() && !CheckFinish())
            {
                usleep(3000);
            }
            if(CheckFinish())
                break;
        }

        ResetIfRequested();

        // Tracking will see that Local Mapping is busy
        SetAcceptKeyFrames(true);

        if(CheckFinish())
            break;

        usleep(3000);
    }

    SetFinish();
}

void LocalMapping::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexNewKFs);
    mlNewKeyFrames.push_back(pKF);
    mbAbortBA=true;
}


bool LocalMapping::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexNewKFs);
    return(!mlNewKeyFrames.empty()); // 새로운 KeyFrame 배열에 요소가 존재하면 true
}

void LocalMapping::ProcessNewKeyFrame()
{
    {
        unique_lock<mutex> lock(mMutexNewKFs);
        mpCurrentKeyFrame = mlNewKeyFrames.front(); // 현재 keyFrame을 갱신
        mlNewKeyFrames.pop_front();
    }

    // Compute Bags of Words structures
    mpCurrentKeyFrame->ComputeBoW(); // 현재 키프레임의 Bag of World를 계산

    // Associate MapPoints to the new keyframe and update normal and descriptor
    const vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches(); // 현재 keyFrame에서 매칭된 Map Point를 얻음

    for(size_t i=0; i<vpMapPointMatches.size(); i++)
    {
        MapPoint* pMP = vpMapPointMatches[i];
        if(pMP) // Map Point가 존재하고
        {
            if(!pMP->isBad()) // Bad 상태가 아니라면
            {
                if(!pMP->IsInKeyFrame(mpCurrentKeyFrame)) // Map Point가 현재 keyFrame 안에 존재한다면
                {
                    pMP->AddObservation(mpCurrentKeyFrame, i); // Map Point를 추적 대상에 추가
                    pMP->UpdateNormalAndDepth(); // 노멀 벡터와 Depth를 갱신
                    pMP->ComputeDistinctiveDescriptors(); // Map Point에서 Distinctive Descriptors를 계산
                }
                else // this can only happen for new stereo points inserted by the Tracking
                {
                    mlpRecentAddedMapPoints.push_back(pMP); // 최근 추가된 MapPoint 배열에 추가
                }
            }
        }
    }

    // Update links in the Covisibility Graph
    mpCurrentKeyFrame->UpdateConnections(); // 현재 Key Frame의 Covisibility Graph를 갱신

    // Insert Keyframe in Map
    mpAtlas->AddKeyFrame(mpCurrentKeyFrame); // Map에 KeyFrame을 삽입
}

void LocalMapping::EmptyQueue()
{
    while(CheckNewKeyFrames()) // 새로운 KeyFrame 배열에 keyFrame이 존재하면 loop
        ProcessNewKeyFrame(); // 현재 KeyFrame에 대해 Bag of Word를 계산하고 MapPoint를 구함
}

void LocalMapping::MapPointCulling()
{
    // Check Recent Added MapPoints
    list<MapPoint*>::iterator lit = mlpRecentAddedMapPoints.begin();
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId; // 현재 keyFrame에 대해 ID를 가져옴

    int nThObs; // 추적 대상 Threshold 값
    if(mbMonocular) // Monocular인 경우
        nThObs = 2;
    else
        nThObs = 3;
    const int cnThObs = nThObs; // Threshold 값을 변화하지 않도록 const로 저장

    int borrar = mlpRecentAddedMapPoints.size(); // 최근 추가된 MapPoint 배열 크기 저장

    while(lit!=mlpRecentAddedMapPoints.end()) // 최근 추가된 MapPoint 배열 loop
    {
        MapPoint* pMP = *lit;

        if(pMP->isBad()) // MapPoint가 Bad 상태이면
            lit = mlpRecentAddedMapPoints.erase(lit); // 배열에서 지움
        else if(pMP->GetFoundRatio()<0.25f) // MapPoint의 Ratio가 0.25 미만이면 (Ratio = Found/Visible)
        {
            pMP->SetBadFlag(); // MapPoint를 Bad 상태로 변경
            lit = mlpRecentAddedMapPoints.erase(lit); // 배열에서 지움
        }
        // 현재 KeyFrame의 ID와 MapPoint의 첫 KeyFrame의 ID의 차이가 2 이상이고 MapPoint의 추적 대상이 Threshold 값 이하인 경우
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=2 && pMP->Observations()<=cnThObs) 
        {
            pMP->SetBadFlag(); // MapPoint를 Bad 상태로 변경
            lit = mlpRecentAddedMapPoints.erase(lit); // 배열에서 지움
        }
        // 현재 KeyFrame의 ID와 MapPoint의 첫 KeyFrame의 ID의 차이가 3 이상인 경우
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=3)
            lit = mlpRecentAddedMapPoints.erase(lit); // 배열에서 지움
        else
        {
            lit++; // 다음 MapPoint로 이동
            borrar--; // MapPoint 배열 크기 감소
        }
    }
}


void LocalMapping::CreateNewMapPoints()
{
    // Retrieve neighbor keyframes in covisibility graph
    int nn = 10; // Covisibility graph를 통해 구할 이웃 KeyFrame 수 (이웃 keyFrame Threshold 값)
    // For stereo inertial case
    if(mbMonocular) // Monocular인 경우 값 변경
        nn=30;
    vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn); // Covisibility graph를 통해 이웃 KeyFrame 검색

    if (mbInertial) // IMU를 사용할 경우
    {
        KeyFrame* pKF = mpCurrentKeyFrame; // KeyFrame을 가리키는 포인터 생성
        int count=0;
        // 이웃 KeyFrame 배열의 크기가 Threshold 값 이하이고 KeyFrame의 이전 keyFrame이 존재하고 Threshold 값 미만으로 반복 하였다면 loop
        while((vpNeighKFs.size()<=nn)&&(pKF->mPrevKF)&&(count++<nn)) 
        {
            vector<KeyFrame*>::iterator it = std::find(vpNeighKFs.begin(), vpNeighKFs.end(), pKF->mPrevKF); 
            if(it==vpNeighKFs.end()) // 이웃 KeyFrame 배열에서 KeyFrame의 이전 KeyFrame과 동일한 KeyFrame을 찾았다면
                vpNeighKFs.push_back(pKF->mPrevKF); // 이웃 KeyFrame 배열에 KeyFrame의 이전 keyFrame을 삽입
            pKF = pKF->mPrevKF; // KeyFrame을 가리키는 대상을 keyFrame의 이전 KeyFrame으로 변경
        }
    }

    float th = 0.6f; // ORB Threshold 값

    ORBmatcher matcher(th,false); // ORB 매칭 객체 생성

    Sophus::SE3<float> sophTcw1 = mpCurrentKeyFrame->GetPose(); // 현재 KeyFrame의 pose를 가져옴
    Eigen::Matrix<float,3,4> eigTcw1 = sophTcw1.matrix3x4(); // pose를 Matrix 객체로 변환
    Eigen::Matrix<float,3,3> Rcw1 = eigTcw1.block<3,3>(0,0); // Rcw 부분을 가져옴
    Eigen::Matrix<float,3,3> Rwc1 = Rcw1.transpose(); // Rcw를 Transpose 함
    Eigen::Vector3f tcw1 = sophTcw1.translation(); // Tcw 부분을 가져옴
    Eigen::Vector3f Ow1 = mpCurrentKeyFrame->GetCameraCenter(); // 현재 KeyFrame에서 CameaCenter를 가져옴 (Twc의 Transpose)

    const float &fx1 = mpCurrentKeyFrame->fx; // 현재 KeyFrame의 카메라 계수를 가져옴
    const float &fy1 = mpCurrentKeyFrame->fy;
    const float &cx1 = mpCurrentKeyFrame->cx;
    const float &cy1 = mpCurrentKeyFrame->cy;
    const float &invfx1 = mpCurrentKeyFrame->invfx;
    const float &invfy1 = mpCurrentKeyFrame->invfy;

    const float ratioFactor = 1.5f*mpCurrentKeyFrame->mfScaleFactor; // 현재 KeyFrame의 ScaleFactor를 가져옴
    int countStereo = 0;
    int countStereoGoodProj = 0;
    int countStereoAttempt = 0;
    int totalStereoPts = 0;
    // Search matches with epipolar restriction and triangulate
    for(size_t i=0; i<vpNeighKFs.size(); i++) // 이웃 KeyFrame 배열 loop
    {
        if(i>0 && CheckNewKeyFrames()) // 두번째 loop 부터 새로운 KeyFrame 배열에 KeyFrame이 존재한다면
            return; // 함수 종료

        KeyFrame* pKF2 = vpNeighKFs[i]; // 이웃 KeyFrame을 가져옴

        GeometricCamera* pCamera1 = mpCurrentKeyFrame->mpCamera, *pCamera2 = pKF2->mpCamera; // 현재 KeyFrame의 카메라 포인터와 이웃 KeyFrame의 카메라 포인터를 가져옴

        // Check first that baseline is not too short
        Eigen::Vector3f Ow2 = pKF2->GetCameraCenter(); // 이웃 KeyFrame의 CameaCenter를 가져옴 
        Eigen::Vector3f vBaseline = Ow2-Ow1; // 이웃 KeyFrame의 CameraCenter와 현재 KeyFrame의 CameraCenter의 차이를 Baseline으로
        const float baseline = vBaseline.norm(); // Baseline의 길이를 구함

        if(!mbMonocular)
        {
            if(baseline<pKF2->mb)
                continue;
        }
        else // Monocular인 경우
        {
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2); // 이웃 KeyFrame의 중앙 depth 값을 가져옴
            const float ratioBaselineDepth = baseline/medianDepthKF2; // baseline을 depth 값으로 나눔

            if(ratioBaselineDepth<0.01) // 나눈 값이 0.01 이하인 경우 무시
                continue;
        }

        // Search matches that fullfil epipolar constraint
        vector<pair<size_t,size_t> > vMatchedIndices;
        // 센서를 사용하고 Tracking 상태가 RECENTLY_LOST이고 현재 KeyFrame의 Map이 IniertialBA2인 경우
        bool bCoarse = mbInertial && mpTracker->mState==Tracking::RECENTLY_LOST && mpCurrentKeyFrame->GetMap()->GetIniertialBA2();

        matcher.SearchForTriangulation(mpCurrentKeyFrame,pKF2,vMatchedIndices,false,bCoarse); // 현재 KeyFrame과 이웃 KeyFrame간에 Triangulation 진행

        Sophus::SE3<float> sophTcw2 = pKF2->GetPose(); // 이웃 KeyFrame의 pose를 가져옴
        Eigen::Matrix<float,3,4> eigTcw2 = sophTcw2.matrix3x4(); // pose를 Matrix 객체로 변환
        Eigen::Matrix<float,3,3> Rcw2 = eigTcw2.block<3,3>(0,0); // Rcw 부분을 가져옴
        Eigen::Matrix<float,3,3> Rwc2 = Rcw2.transpose(); // Rcw를 Transpose 함
        Eigen::Vector3f tcw2 = sophTcw2.translation(); // Tcw 부분을 가져옴

        const float &fx2 = pKF2->fx; // 이웃 KeyFrame의 카메라 계수를 가져옴
        const float &fy2 = pKF2->fy;
        const float &cx2 = pKF2->cx;
        const float &cy2 = pKF2->cy;
        const float &invfx2 = pKF2->invfx;
        const float &invfy2 = pKF2->invfy;

        // Triangulate each match
        const int nmatches = vMatchedIndices.size(); 
        for(int ikp=0; ikp<nmatches; ikp++) // 매칭 Point의 Index 배열의 크기만큼 loop
        {
            const int &idx1 = vMatchedIndices[ikp].first; // 매칭 Point의 첫번째와 두번째 index를 가져옴
            const int &idx2 = vMatchedIndices[ikp].second;

            // 현재 KeyFrame의 왼쪽에서 관찰된 KeyPoint의 수가 -1인 경우 첫번째 index를 통해 왜곡되지 않은 KeyPoint를 가져옴
            // -1이 아닌 경우 첫번째 index가 현재 KeyFrame의 왼쪽에서 관찰된 KeyPoint의 수보다 큰 경우 첫번째 index를 통해 KeyPoint를 가져옴 
            // 모든 경우가 아니라면 현재 keyFrame의 오른쪽에서 관찰된 KeyPoint를 가져옴
            const cv::KeyPoint &kp1 = (mpCurrentKeyFrame -> NLeft == -1) ? mpCurrentKeyFrame->mvKeysUn[idx1]
                                                                         : (idx1 < mpCurrentKeyFrame -> NLeft) ? mpCurrentKeyFrame -> mvKeys[idx1]
                                                                                                               : mpCurrentKeyFrame -> mvKeysRight[idx1 - mpCurrentKeyFrame -> NLeft];
            const float kp1_ur=mpCurrentKeyFrame->mvuRight[idx1]; // 현재 KeyFrame의 stereo인 경우 scale 점수
            bool bStereo1 = (!mpCurrentKeyFrame->mpCamera2 && kp1_ur>=0); // 현재 KeyFrame의 camera가 존재하지 않고 scale 점수가 0 이상인 경우 (Monocular인 경우 false)
            // 현재 KeyFrame의 왼쪽에서 관찰된 keyPoint수가 -1이거나 첫번째 index가 현재 KeyFrame의 왼쪽에서 관찰된 KeyPoint의 수보다 큰 경우 false
            const bool bRight1 = (mpCurrentKeyFrame -> NLeft == -1 || idx1 < mpCurrentKeyFrame -> NLeft) ? false
                                                                                                         : true;

            // 이웃 KeyFrame의 왼쪽에서 관찰된 KeyPoint의 수가 -1인 경우 두번째 index를 통해 왜곡되지 않은 KeyPoint를 가져옴
            // -1이 아닌 경우 두번째 index가 이웃 KeyFrame의 왼쪽에서 관찰된 KeyPoint의 수보다 큰 경우 두번째 index를 통해 KeyPoint를 가져옴 
            // 모든 경우가 아니라면 이웃 keyFrame의 오른쪽에서 관찰된 KeyPoint를 가져옴
            const cv::KeyPoint &kp2 = (pKF2 -> NLeft == -1) ? pKF2->mvKeysUn[idx2]
                                                            : (idx2 < pKF2 -> NLeft) ? pKF2 -> mvKeys[idx2]
                                                                                     : pKF2 -> mvKeysRight[idx2 - pKF2 -> NLeft];

            const float kp2_ur = pKF2->mvuRight[idx2]; // 이웃 KeyFrame의 stereo인 경우 scale 점수
            bool bStereo2 = (!pKF2->mpCamera2 && kp2_ur>=0); // 이웃 KeyFrame의 camera가 존재하지 않고 scale 점수가 0 이상인 경우 (Monocular인 경우 false)
            // 이웃 KeyFrame의 왼쪽에서 관찰된 keyPoint수가 -1이거나 두번째 index가 이웃 KeyFrame의 왼쪽에서 관찰된 KeyPoint의 수보다 큰 경우 false
            const bool bRight2 = (pKF2 -> NLeft == -1 || idx2 < pKF2 -> NLeft) ? false
                                                                               : true;

            if(mpCurrentKeyFrame->mpCamera2 && pKF2->mpCamera2){ // 현재 KeyFrame과 이웃 KeyFrame에 두 번째 Camera가 존재하는 경우 (Stereo인 경우)
                if(bRight1 && bRight2){
                    sophTcw1 = mpCurrentKeyFrame->GetRightPose();
                    Ow1 = mpCurrentKeyFrame->GetRightCameraCenter();

                    sophTcw2 = pKF2->GetRightPose();
                    Ow2 = pKF2->GetRightCameraCenter();

                    pCamera1 = mpCurrentKeyFrame->mpCamera2;
                    pCamera2 = pKF2->mpCamera2;
                }
                else if(bRight1 && !bRight2){
                    sophTcw1 = mpCurrentKeyFrame->GetRightPose();
                    Ow1 = mpCurrentKeyFrame->GetRightCameraCenter();

                    sophTcw2 = pKF2->GetPose();
                    Ow2 = pKF2->GetCameraCenter();

                    pCamera1 = mpCurrentKeyFrame->mpCamera2;
                    pCamera2 = pKF2->mpCamera;
                }
                else if(!bRight1 && bRight2){
                    sophTcw1 = mpCurrentKeyFrame->GetPose();
                    Ow1 = mpCurrentKeyFrame->GetCameraCenter();

                    sophTcw2 = pKF2->GetRightPose();
                    Ow2 = pKF2->GetRightCameraCenter();

                    pCamera1 = mpCurrentKeyFrame->mpCamera;
                    pCamera2 = pKF2->mpCamera2;
                }
                else{
                    sophTcw1 = mpCurrentKeyFrame->GetPose();
                    Ow1 = mpCurrentKeyFrame->GetCameraCenter();

                    sophTcw2 = pKF2->GetPose();
                    Ow2 = pKF2->GetCameraCenter();

                    pCamera1 = mpCurrentKeyFrame->mpCamera;
                    pCamera2 = pKF2->mpCamera;
                }
                eigTcw1 = sophTcw1.matrix3x4();
                Rcw1 = eigTcw1.block<3,3>(0,0);
                Rwc1 = Rcw1.transpose();
                tcw1 = sophTcw1.translation();

                eigTcw2 = sophTcw2.matrix3x4();
                Rcw2 = eigTcw2.block<3,3>(0,0);
                Rwc2 = Rcw2.transpose();
                tcw2 = sophTcw2.translation();
            }

            // Check parallax between rays
            Eigen::Vector3f xn1 = pCamera1->unprojectEig(kp1.pt); // 현재 KeyFrame의 카메라에서 첫번째 keyPoint를 역투영헤 좌표를 구함
            Eigen::Vector3f xn2 = pCamera2->unprojectEig(kp2.pt); // 이웃 KeyFrame의 카메라에서 두번째 keyPoint를 역투영해 좌표를 구함

            Eigen::Vector3f ray1 = Rwc1 * xn1; // Rwc와 keyPoint의 좌표를 곱해 월드 좌표로 변환
            Eigen::Vector3f ray2 = Rwc2 * xn2;
            const float cosParallaxRays = ray1.dot(ray2)/(ray1.norm() * ray2.norm()); // 두 좌표간의 코사인 parallax를 계산

            float cosParallaxStereo = cosParallaxRays+1;
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;

            if(bStereo1) // 스테레오인 경우
                cosParallaxStereo1 = cos(2*atan2(mpCurrentKeyFrame->mb/2,mpCurrentKeyFrame->mvDepth[idx1]));
            else if(bStereo2) // 스테레오인 경우
                cosParallaxStereo2 = cos(2*atan2(pKF2->mb/2,pKF2->mvDepth[idx2]));

            // 스테레오인 경우
            if (bStereo1 || bStereo2) totalStereoPts++;
            
            cosParallaxStereo = min(cosParallaxStereo1,cosParallaxStereo2); // cosParallaxStereo 값을 최소 값으로 초기화

            Eigen::Vector3f x3D;

            bool goodProj = false; // 좋은 투영인지 검사하는 flag
            bool bPointStereo = false;

            // cosParallax값이 더 크고 (Monocular인 경우 항상 참) keyPoint간의 parallax가 0보다 크고
            // 스테레오를 사용하거나 cosParallax값이 0.9996 보다 작고 IMU를 사용하거나
            // cosParallax값이 0.9996 보다 작고 IMU를 사용하거나 IMU를 사용하지 않는다면
            if(cosParallaxRays<cosParallaxStereo && cosParallaxRays>0 && (bStereo1 || bStereo2 ||
                                                                          (cosParallaxRays<0.9996 && mbInertial) || (cosParallaxRays<0.9998 && !mbInertial)))
            {
                goodProj = GeometricTools::Triangulate(xn1, xn2, eigTcw1, eigTcw2, x3D); // 두 KeyPoint 간의 Triangulation을 연산
                if(!goodProj) // 투영 결과가 별로면 무시
                    continue;
            }
            else if(bStereo1 && cosParallaxStereo1<cosParallaxStereo2) // 스테레오인 경우
            {
                countStereoAttempt++;
                bPointStereo = true;
                goodProj = mpCurrentKeyFrame->UnprojectStereo(idx1, x3D);
            }
            else if(bStereo2 && cosParallaxStereo2<cosParallaxStereo1) // 스테레오인 경우
            {
                countStereoAttempt++;
                bPointStereo = true;
                goodProj = pKF2->UnprojectStereo(idx2, x3D);
            }
            else // Monocular이고 Parallax가 낮으면
            {
                continue; //No stereo and very low parallax
            }

            if(goodProj && bPointStereo) // 투영 결과가 좋고 Stereo인 경우
                countStereoGoodProj++;

            if(!goodProj) // 투영 결과가 좋지 않으면 무시
                continue;

            //Check triangulation in front of cameras
            float z1 = Rcw1.row(2).dot(x3D) + tcw1(2); // Rcw의 Z 값과 Triangulation한 Point 간의 내적을 구하고 tcw의 Z 값과 더함
            if(z1<=0) // Triangulation한 Point가 현재 KeyFrame의 카메라 뒷쪽에 있다면 무시
                continue;

            float z2 = Rcw2.row(2).dot(x3D) + tcw2(2);
            if(z2<=0) // Triangulation한 Point가 이웃 KeyFrame의 카메라 뒷쪽에 있다면 무시
                continue;

            //Check reprojection error in first keyframe
            const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave]; // 현재 KeyFrame에서 첫번째 KeyPoint의 옥타브에 대한 Sigma 값을 얻음
            const float x1 = Rcw1.row(0).dot(x3D)+tcw1(0); // Rcw의 X 값과 Triangulation한 Point 간의 내적을 구하고 tcw의 X 값과 더함
            const float y1 = Rcw1.row(1).dot(x3D)+tcw1(1); // Rcw의 Y 값과 Triangulation한 Point 간의 내적을 구하고 tcw의 Y 값과 더함
            const float invz1 = 1.0/z1; // Z 값의 역수를 구함

            if(!bStereo1) // Monocular인 경우
            {
                cv::Point2f uv1 = pCamera1->project(cv::Point3f(x1,y1,z1)); // 내적을 통해 구한 X, Y, Z값을 통해 현재 KeyFrame의 카메라와 Projection 진행
                float errX1 = uv1.x - kp1.pt.x; // Projection한 Point와 KeyPoint간의 차이를 구함
                float errY1 = uv1.y - kp1.pt.y;

                if((errX1*errX1+errY1*errY1)>5.991*sigmaSquare1) // 차이 값 제곱의 합이 5.991 * Sigma 값 보다 큰 경우 무시
                    continue;

            }
            else
            {
                float u1 = fx1*x1*invz1+cx1;
                float u1_r = u1 - mpCurrentKeyFrame->mbf*invz1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                float errX1_r = u1_r - kp1_ur;
                if((errX1*errX1+errY1*errY1+errX1_r*errX1_r)>7.8*sigmaSquare1)
                    continue;
            }

            //Check reprojection error in second keyframe
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave]; // 이웃 KeyFrame에서 두번째 KeyPoint의 옥타브에 대한 Sigma 값을 얻음
            const float x2 = Rcw2.row(0).dot(x3D)+tcw2(0); // Rcw의 X 값과 Triangulation한 Point 간의 내적을 구하고 tcw의 X 값과 더함
            const float y2 = Rcw2.row(1).dot(x3D)+tcw2(1); // Rcw의 Y 값과 Triangulation한 Point 간의 내적을 구하고 tcw의 Y 값과 더함
            const float invz2 = 1.0/z2; // Z 값의 역수를 구함
            if(!bStereo2) // Monocular인 경우
            {
                cv::Point2f uv2 = pCamera2->project(cv::Point3f(x2,y2,z2)); // 내적을 통해 구한 X, Y, Z값을 통해 이웃 KeyFrame의 카메라와 Projection 진행
                float errX2 = uv2.x - kp2.pt.x; // Projection한 Point와 KeyPoint간의 차이를 구함
                float errY2 = uv2.y - kp2.pt.y;
                if((errX2*errX2+errY2*errY2)>5.991*sigmaSquare2) // 차이 값 제곱의 합이 5.991 * Sigma 값 보다 큰 경우 무시
                    continue;
            }
            else
            {
                float u2 = fx2*x2*invz2+cx2;
                float u2_r = u2 - mpCurrentKeyFrame->mbf*invz2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                float errX2_r = u2_r - kp2_ur;
                if((errX2*errX2+errY2*errY2+errX2_r*errX2_r)>7.8*sigmaSquare2)
                    continue;
            }

            //Check scale consistency
            Eigen::Vector3f normal1 = x3D - Ow1; // Triangulation한 Point와 현재 KeyFrame에서 CameaCenter 간의 차이를 구함
            float dist1 = normal1.norm(); 

            Eigen::Vector3f normal2 = x3D - Ow2; // Triangulation한 Point와 이웃 KeyFrame에서 CameaCenter 간의 차이를 구함
            float dist2 = normal2.norm();

            if(dist1==0 || dist2==0) // 거리가 0 이면 무시
                continue;

            // yaml 파일에 있는 thFarPoints(무시할 거리 값) 값이 0이 아니고 
            // 차이 값이 threshold 값보다 큰 경우 무시
            if(mbFarPoints && (dist1>=mThFarPoints||dist2>=mThFarPoints)) // MODIFICATION
                continue;

            const float ratioDist = dist2/dist1; // 거리 값의 ratio를 구함 
            const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave]/pKF2->mvScaleFactors[kp2.octave]; // keyPoint의 옥타브에 따른 Scale 값의 ratio를 구함

            // 거리 값 ratio와 ScaleFactor 값의 곱이 옥타브에 따른 Scale 값 보다 작거나 
            // 거리 값 ratio가 ScaleFactor 값과 옥타브에 따른 Scale 값의 곱보다 큰 경우 무시
            if(ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor) 
                continue;

            // Triangulation is succesfull
            MapPoint* pMP = new MapPoint(x3D, mpCurrentKeyFrame, mpAtlas->GetCurrentMap()); // Triangulation한 Point에 대해 MapPoint를 생성
            if (bPointStereo) // Stereo인 경우
                countStereo++;
            
            pMP->AddObservation(mpCurrentKeyFrame,idx1); // MapPoint에 현재 KeyFrame과 이웃 KeyFrame을 추적 대상으로 추가
            pMP->AddObservation(pKF2,idx2);

            mpCurrentKeyFrame->AddMapPoint(pMP,idx1); // 현재 KeyFrame에 MapPoint 추가
            pKF2->AddMapPoint(pMP,idx2); // 이웃 KeyFrame에 MapPoint 추가

            pMP->ComputeDistinctiveDescriptors(); // MapPoint에 대해 디스크립터 연산 진행

            pMP->UpdateNormalAndDepth(); // 노멀 벡터와 depth 값 업데이트

            mpAtlas->AddMapPoint(pMP); // Atlas에 MapPoint 추가
            mlpRecentAddedMapPoints.push_back(pMP); // 최근 MapPoint 배열에 MapPoint 삽입
        }
    }    
}

void LocalMapping::SearchInNeighbors()
{
    // Retrieve neighbor keyframes
    int nn = 10; // Covisibility graph를 통해 구할 이웃 KeyFrame 수 (이웃 keyFrame Threshold 값)
    if(mbMonocular) // Monocular인 경우 값 변경
        nn=30;
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn); // Covisibility graph를 통해 이웃 KeyFrame 검색
    vector<KeyFrame*> vpTargetKFs;
    for(vector<KeyFrame*>::const_iterator vit=vpNeighKFs.begin(), vend=vpNeighKFs.end(); vit!=vend; vit++) // 이웃 KeyFrame 배열 크기 만큼 loop
    {
        KeyFrame* pKFi = *vit;
        if(pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId) // 이웃 KeyFrame이 Bad 판정 이거나 이전에 이웃 KeyFrame이 현재 KeyFrame의 Fusion Target으로 정해 졌다면 무시
            continue;
        vpTargetKFs.push_back(pKFi); // Target KeyFrame 배열에 이웃 KeyFrame을 삽입
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId; // 이웃 KeyFrame의 Fusion Target을 현재 KeyFrame으로 변경
    }

    // Add some covisible of covisible
    // Extend to some second neighbors if abort is not requested
    for(int i=0, imax=vpTargetKFs.size(); i<imax; i++) // Target KeyFrame 배열의 크기만큼 loop
    {
        const vector<KeyFrame*> vpSecondNeighKFs = vpTargetKFs[i]->GetBestCovisibilityKeyFrames(20); // Target KeyFrame의 이웃 keyFrame을 검색
        for(vector<KeyFrame*>::const_iterator vit2=vpSecondNeighKFs.begin(), vend2=vpSecondNeighKFs.end(); vit2!=vend2; vit2++) // 이웃 KeyFrame 배열 크기 만큼 loop
        {
            KeyFrame* pKFi2 = *vit2;
            // 이웃 KeyFrame이 Bad 판정 이거나 이전에 이웃 KeyFrame이 현재 KeyFrame의 Fusion Target으로 정해 졌거나 이웃 KeyFrame의 id가 현재 KeyFrame과 같다면 무시
            if(pKFi2->isBad() || pKFi2->mnFuseTargetForKF==mpCurrentKeyFrame->mnId || pKFi2->mnId==mpCurrentKeyFrame->mnId)
                continue;
            vpTargetKFs.push_back(pKFi2); // Target KeyFrame 배열에 이웃 KeyFrame을 삽입
            pKFi2->mnFuseTargetForKF=mpCurrentKeyFrame->mnId; // 이웃 KeyFrame의 Fusion Target을 현재 KeyFrame으로 변경
        }
        if (mbAbortBA) // BA가 중단 되었다면 반복 종료
            break;
    }

    // Extend to temporal neighbors
    if(mbInertial) // IMU를 사용하는 경우
    {
        KeyFrame* pKFi = mpCurrentKeyFrame->mPrevKF; // 현재 KeyFrame의 이전 KeyFrame을 가져옴
        while(vpTargetKFs.size()<20 && pKFi) // Target KeyFrame의 수가 20 미만이고 이전 KeyFrame이 존재하는 경우
        {
            if(pKFi->isBad() || pKFi->mnFuseTargetForKF==mpCurrentKeyFrame->mnId) // 이전 KeyFrame이 Bad 판정 이거나 이전 KeyFrame이 현재 KeyFrame의 Target이였던 경우 
            {
                pKFi = pKFi->mPrevKF; // 이전 KeyFrame을 이전 KeyFrame의 이전 KeyFrame으로 변경
                continue; // 무시
            }
            vpTargetKFs.push_back(pKFi); // Target KeyFrame 배열에 이전 KeyFrame을 삽입
            pKFi->mnFuseTargetForKF=mpCurrentKeyFrame->mnId; // 이전 KeyFrame의 Fusion Target을 현재 KeyFrame으로 변경
            pKFi = pKFi->mPrevKF; // 이전 KeyFrame을 이전 KeyFrame의 이전 KeyFrame으로 변경
        }
    }

    // Search matches by projection from current KF in target KFs
    ORBmatcher matcher; // ORB Matcher 객체 생성
    vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches(); // 현재 KeyFrame의 매칭 MapPoint를 가져옴
    for(vector<KeyFrame*>::iterator vit=vpTargetKFs.begin(), vend=vpTargetKFs.end(); vit!=vend; vit++) // Target KeyFrame 배열의 크기만큼 loop
    {
        KeyFrame* pKFi = *vit;

        matcher.Fuse(pKFi,vpMapPointMatches); // Target KeyFrame과 현재 KeyFrame의 매칭 MapPoint간의 Fusion을 진행
        if(pKFi->NLeft != -1) matcher.Fuse(pKFi,vpMapPointMatches,true); // Target KeyFrame의 왼쪽에서 관찰된 KeyPoint의 수가 -1인 경우 (Stereo인 경우)
    }


    if (mbAbortBA) // BA가 중단 되었다면 함수 종료
        return;

    // Search matches by projection from target KFs in current KF
    vector<MapPoint*> vpFuseCandidates; // Candiate 배열 생성
    vpFuseCandidates.reserve(vpTargetKFs.size()*vpMapPointMatches.size()); // Target KeyFrame 배열 크기와 매칭 MapPoint 배열 크기의 곱만큼 배열 크기 할당

    for(vector<KeyFrame*>::iterator vitKF=vpTargetKFs.begin(), vendKF=vpTargetKFs.end(); vitKF!=vendKF; vitKF++) // Target KeyFrame 배열 크기만큼 loop
    {
        KeyFrame* pKFi = *vitKF; // Target KeyFrame을 가져옴

        vector<MapPoint*> vpMapPointsKFi = pKFi->GetMapPointMatches(); // Target KeyFrame의 매칭 MapPoint를 가져옴

        for(vector<MapPoint*>::iterator vitMP=vpMapPointsKFi.begin(), vendMP=vpMapPointsKFi.end(); vitMP!=vendMP; vitMP++) // 매칭 MapPoint 배열 크기만큼 loop
        {
            MapPoint* pMP = *vitMP;
            if(!pMP) // MapPoint가 존재하지 않다면 무시
                continue;
            if(pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId) // MapPoint가 Bad 판정이고 MapPoint의 Fusion Target이 현재 KeyFrame이면 무시
                continue;
            pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId; // MapPoint의 Fustion Target을 현재 KeyFrame으로 변경
            vpFuseCandidates.push_back(pMP); // Candiate 배열에 MapPoint를 삽입
        }
    }

    matcher.Fuse(mpCurrentKeyFrame,vpFuseCandidates); // 현재 KeyFrame과 Candiate MapPoint 간의 Fusion을 진행
    if(mpCurrentKeyFrame->NLeft != -1) matcher.Fuse(mpCurrentKeyFrame,vpFuseCandidates,true); // 현재 KeyFrame의 왼쪽에서 관찰된 KeyPoint의 수가 -1인 경우 (Stereo인 경우)


    // Update points
    vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches(); // 현재 KeyFrame에서 매칭 MapPoint를 가져옴
    for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++) // 매칭 MapPoint 배열 크기만큼 loop
    {
        MapPoint* pMP=vpMapPointMatches[i];
        if(pMP) // MapPoint가 존재하는 경우
        {
            if(!pMP->isBad()) // MapPoint가 Bad 판정이 아닌 경우
            {
                pMP->ComputeDistinctiveDescriptors(); // MapPoint의 디스크립터를 연산
                pMP->UpdateNormalAndDepth(); // 노멀 벡터와 Depth를 Update
            }
        }
    }

    // Update connections in covisibility graph
    mpCurrentKeyFrame->UpdateConnections(); // 현재 KeyFrame의 Covisibility Graph를 Update
}

void LocalMapping::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
    unique_lock<mutex> lock2(mMutexNewKFs);
    mbAbortBA = true;
}

bool LocalMapping::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        cout << "Local Mapping STOP" << endl;
        return true;
    }

    return false;
}

bool LocalMapping::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool LocalMapping::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

void LocalMapping::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);
    if(mbFinished)
        return;
    mbStopped = false;
    mbStopRequested = false;
    for(list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
        delete *lit;
    mlNewKeyFrames.clear();

    cout << "Local Mapping RELEASE" << endl;
}

bool LocalMapping::AcceptKeyFrames()
{
    unique_lock<mutex> lock(mMutexAccept);
    return mbAcceptKeyFrames;
}

void LocalMapping::SetAcceptKeyFrames(bool flag)
{
    unique_lock<mutex> lock(mMutexAccept);
    mbAcceptKeyFrames=flag;
}

bool LocalMapping::SetNotStop(bool flag)
{
    unique_lock<mutex> lock(mMutexStop);

    if(flag && mbStopped)
        return false;

    mbNotStop = flag;

    return true;
}

void LocalMapping::InterruptBA()
{
    mbAbortBA = true;
}

void LocalMapping::KeyFrameCulling()
{
    // Check redundant keyframes (only local keyframes)
    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    // We only consider close stereo points
    const int Nd = 21; // KeyFrame 기준 수 선언
    mpCurrentKeyFrame->UpdateBestCovisibles(); // 현재 KeyFrame의 Covisiblity Graph를 Update
    vector<KeyFrame*> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames(); // 현재 KeyFrame의 Covisiblity Graph에 연결된 KeyFrame을 가져옴

    float redundant_th; // threshold 값 선언
    if(!mbInertial)
        redundant_th = 0.9;
    else if (mbMonocular) // Monocular인 경우
        redundant_th = 0.9;
    else
        redundant_th = 0.5;

    const bool bInitImu = mpAtlas->isImuInitialized(); // Atlas에 IMU가 초기화된 경우 true
    int count=0;

    // Compoute last KF from optimizable window:
    unsigned int last_ID;
    if (mbInertial) // IMU를 사용하는 경우
    {
        int count = 0;
        KeyFrame* aux_KF = mpCurrentKeyFrame; // 현재 KeyFrame을 가리키는 포인터 선언
        while(count<Nd && aux_KF->mPrevKF) // 카운터가 KeyFrame 기준 미만이고 이전 KeyFrame이 존재하는 경우 loop
        {
            aux_KF = aux_KF->mPrevKF; // 이전 KeyFrame으로 포인터 변경
            count++; // 카운터 증가
        }
        last_ID = aux_KF->mnId; // 이전 ID를 포인터가 가리키는 KeyFrame으로 변경
    }



    for(vector<KeyFrame*>::iterator vit=vpLocalKeyFrames.begin(), vend=vpLocalKeyFrames.end(); vit!=vend; vit++) // 연결된 keyFrame 배열의 크기만큼 loop
    {
        count++; // 카운터 증가
        KeyFrame* pKF = *vit;

        if((pKF->mnId==pKF->GetMap()->GetInitKFid()) || pKF->isBad()) // KeyFrame의 ID가 KeyFrame의 Map의 초기화 KeyFrame ID와 같거나 KeyFrame이 Bad 판정인 경우 무시
            continue;
        const vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches(); // KeyFrame의 매칭 MapPoint를 가져옴

        int nObs = 3; // 추적대상 수 선언
        const int thObs=nObs; // 추적 대상 threshold 값 선언
        int nRedundantObservations=0; // 중복된 추적대상 수 선언
        int nMPs=0; // MapPoint 수 선언
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++) // 매칭 MapPoint 배열의 크기만큼 loop
        {
            MapPoint* pMP = vpMapPoints[i];
            if(pMP) // MapPoint가 존재하는 경우
            {
                if(!pMP->isBad()) // MapPoint가 Bad 상태가 아닌 경우
                {
                    if(!mbMonocular) // Monocular가 아닌 경우
                    {
                        if(pKF->mvDepth[i]>pKF->mThDepth || pKF->mvDepth[i]<0)
                            continue;
                    }

                    nMPs++; // MapPoint 수 증가
                    if(pMP->Observations()>thObs) // MapPoint의 추적 대상이 threshold 값보다 큰 경우
                    {
                        // KeyFrame의 왼쪽에서 관찰된 KeyPoint 수가 -1 인 경우 KeyFrame의 왜곡되지 않은 KeyPoint의 옥타브를 scale로 설정
                        // 아닌 경우 KeyFrame의 왼쪽에서 관찰된 KeyPoint 수가 MapPoint ID보다 큰 경우 KeyPoint의 옥타브를 scale로 설정
                        // 그것도 아닌 경우 KeyFrame의 오른쪽에서 관찰된 KeyPoint의 옥타브를 scale로 설정
                        const int &scaleLevel = (pKF -> NLeft == -1) ? pKF->mvKeysUn[i].octave
                                                                     : (i < pKF -> NLeft) ? pKF -> mvKeys[i].octave
                                                                                          : pKF -> mvKeysRight[i].octave;
                        const map<KeyFrame*, tuple<int,int>> observations = pMP->GetObservations(); // MapPoint의 추적 대상을 가져옴
                        int nObs=0; // 추적대상 카운터 선언
                        for(map<KeyFrame*, tuple<int,int>>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++) // 추적 대상 배열의 크기만큼 loop
                        {
                            KeyFrame* pKFi = mit->first;
                            if(pKFi==pKF) // 추적 대상 keyFrame이 KeyFrame과 같을 경우 무시
                                continue;
                            tuple<int,int> indexes = mit->second; // 추적 대상의 index를 가져옴
                            int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes); // 왼쪽 index와 오른쪽 index를 저장
                            int scaleLeveli = -1;
                            if(pKFi -> NLeft == -1) // 추적 대상 KeyFrame의 왼쪽에서 관찰된 KeyPoint수가 -1인 경우
                                scaleLeveli = pKFi->mvKeysUn[leftIndex].octave; // Scale을 KeyFrame의 왜곡되지 않은 KeyPoint의 옥타브를 scale로 설정
                            else {
                                if (leftIndex != -1) { 
                                    scaleLeveli = pKFi->mvKeys[leftIndex].octave; // 추적 대상 KeyFrame의 왼쪽에서 관찰된 KeyPoint수가 -1이 아닌 경우 keyPoint의 옥타브를 scale로 설정
                                }
                                // 추적 대상 KeyFrame의 오른쪽에서 관찰된 KeyPoint수가 -1이 아닌 경우
                                if (rightIndex != -1) {
                                    int rightLevel = pKFi->mvKeysRight[rightIndex - pKFi->NLeft].octave;  // keyFrame의 오른쪽 keyPoint의 옥타브를 level로 설정
                                    scaleLeveli = (scaleLeveli == -1 || scaleLeveli > rightLevel) ? rightLevel
                                                                                                  : scaleLeveli; // scaleLevel이 -1이거나 scaleLevel이 level보다 큰 경우 level을 scale로 사용
                                }
                            }

                            if(scaleLeveli<=scaleLevel+1) // 추적 대상 scale이 scale보다 작거나 같으면
                            {
                                nObs++; // 추적 대상 수 추가
                                if(nObs>thObs) // 대상 수가 threshold 값보다 커지면
                                    break; // 반복 종료
                            }
                        }
                        if(nObs>thObs) // 추적 대상 수가 threshold 값보다 커지면
                        {
                            nRedundantObservations++; // 중복 추적 대상 수 증가
                        }
                    }
                }
            }
        }

        if(nRedundantObservations>redundant_th*nMPs) // 중복 추적 대상 수가 threshold 값의 MapPoint 수를 곱한 값보다 크면
        {
            if (mbInertial) // IMU를 사용하는 경우
            {
                if (mpAtlas->KeyFramesInMap()<=Nd) // Atlas에서 Map안에 있는 KeyFrame의 수가 KeyFrame 기준보다 작거나 같으면 무시
                    continue;

                if(pKF->mnId>(mpCurrentKeyFrame->mnId-2)) // KeyFrame의 ID가 현재 KeyFrame의 ID보다 큰 경우 무시
                    continue;

                if(pKF->mPrevKF && pKF->mNextKF) // KeyFrame의 이전 KeyFrame이 존재하고 다음 KeyFrame도 존재한 경우
                {
                    const float t = pKF->mNextKF->mTimeStamp-pKF->mPrevKF->mTimeStamp; // 다음 keyFrame과 이전 keyFrame과의 timestamp 차이를 저장

                    if((bInitImu && (pKF->mnId<last_ID) && t<3.) || (t<0.5)) // IMU가 초기화 되었고 KeyFrame의 ID가 이전 ID보다 작고 시간 차이가 3보다 작거나 시간 차이가 5보다 작으면
                    {
                        pKF->mNextKF->mpImuPreintegrated->MergePrevious(pKF->mpImuPreintegrated); // KeyFrame의 다음 KeyFrame의 Preintegration 객체에 bias 및 가속도, 자이로 값을 KeyFrame의 Bias 및 가속도, 자이로 값을 통해 Integration 진행
                        pKF->mNextKF->mPrevKF = pKF->mPrevKF; // KeyFrame의 다음 KeyFrame의 이전 KeyFrame을 KeyFrame의 이전 KeyFrame으로 변경
                        pKF->mPrevKF->mNextKF = pKF->mNextKF; // KeyFrame의 이전 KeyFrame의 다음 KeyFrame을 KeyFrame의 다음 KeyFrame으로 변경
                        pKF->mNextKF = NULL; // KeyFrame의 다음 keyFrame 제거
                        pKF->mPrevKF = NULL; // KeyFrame의 이전 KeyFrame 제거
                        pKF->SetBadFlag(); // KeyFrame을 Bad 상태로 변경
                    }
                    // 현재 KeyFrame의 Map이 IntertialBA2가 일어나지 않았고 KeyFrame의 IMU pose와 이전 KeyFrame의 IMU pose 차이가 0.02 보다 작고 시간 차이가 3보다 작으면
                    else if(!mpCurrentKeyFrame->GetMap()->GetIniertialBA2() && ((pKF->GetImuPosition()-pKF->mPrevKF->GetImuPosition()).norm()<0.02) && (t<3)) 
                    {
                        pKF->mNextKF->mpImuPreintegrated->MergePrevious(pKF->mpImuPreintegrated); // KeyFrame의 다음 KeyFrame의 Preintegration 객체에 bias 및 가속도, 자이로 값을 KeyFrame의 Bias 및 가속도, 자이로 값을 통해 Integration 진행
                        pKF->mNextKF->mPrevKF = pKF->mPrevKF; // KeyFrame의 다음 KeyFrame의 이전 KeyFrame을 KeyFrame의 이전 KeyFrame으로 변경
                        pKF->mPrevKF->mNextKF = pKF->mNextKF; // KeyFrame의 이전 KeyFrame의 다음 KeyFrame을 KeyFrame의 다음 KeyFrame으로 변경
                        pKF->mNextKF = NULL; // KeyFrame의 다음 keyFrame 제거
                        pKF->mPrevKF = NULL; // KeyFrame의 이전 KeyFrame 제거
                        pKF->SetBadFlag(); // KeyFrame을 Bad 상태로 변경
                    }
                }
            }
            else
            {
                pKF->SetBadFlag(); // KeyFrame을 Bad 상태로 변경
            }
        }
        if((count > 20 && mbAbortBA) || count>100) // 카운터가 20 보다 크고 BA가 중단 되었거나 카운터가 100보다 크면
        {
            break; // 반복 종료
        }
    }
}

void LocalMapping::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        cout << "LM: Map reset recieved" << endl;
        mbResetRequested = true;
    }
    cout << "LM: Map reset, waiting..." << endl;

    while(1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if(!mbResetRequested)
                break;
        }
        usleep(3000);
    }
    cout << "LM: Map reset, Done!!!" << endl;
}

void LocalMapping::RequestResetActiveMap(Map* pMap)
{
    {
        unique_lock<mutex> lock(mMutexReset);
        cout << "LM: Active map reset recieved" << endl;
        mbResetRequestedActiveMap = true;
        mpMapToReset = pMap;
    }
    cout << "LM: Active map reset, waiting..." << endl;

    while(1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if(!mbResetRequestedActiveMap)
                break;
        }
        usleep(3000);
    }
    cout << "LM: Active map reset, Done!!!" << endl;
}

void LocalMapping::ResetIfRequested()
{
    bool executed_reset = false;
    {
        unique_lock<mutex> lock(mMutexReset);
        if(mbResetRequested)
        {
            executed_reset = true;

            cout << "LM: Reseting Atlas in Local Mapping..." << endl;
            mlNewKeyFrames.clear();
            mlpRecentAddedMapPoints.clear();
            mbResetRequested = false;
            mbResetRequestedActiveMap = false;

            // Inertial parameters
            mTinit = 0.f;
            mbNotBA2 = true;
            mbNotBA1 = true;
            mbBadImu=false;

            mIdxInit=0;

            cout << "LM: End reseting Local Mapping..." << endl;
        }

        if(mbResetRequestedActiveMap) {
            executed_reset = true;
            cout << "LM: Reseting current map in Local Mapping..." << endl;
            mlNewKeyFrames.clear();
            mlpRecentAddedMapPoints.clear();

            // Inertial parameters
            mTinit = 0.f;
            mbNotBA2 = true;
            mbNotBA1 = true;
            mbBadImu=false;

            mbResetRequested = false;
            mbResetRequestedActiveMap = false;
            cout << "LM: End reseting Local Mapping..." << endl;
        }
    }
    if(executed_reset)
        cout << "LM: Reset free the mutex" << endl;

}

void LocalMapping::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LocalMapping::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LocalMapping::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;    
    unique_lock<mutex> lock2(mMutexStop);
    mbStopped = true;
}

bool LocalMapping::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

void LocalMapping::InitializeIMU(float priorG, float priorA, bool bFIBA)
{
    if (mbResetRequested) // 리셋 Flag 설정 시 함수 종료
        return;

    float minTime;
    int nMinKF;
    if (mbMonocular) // Monocular인 경우
    {
        minTime = 2.0; // 최소 시간 2로 설정
        nMinKF = 10; // 최소 KeyFrame을 10으로 설정
    }
    else
    {
        minTime = 1.0;
        nMinKF = 10;
    }


    if(mpAtlas->KeyFramesInMap()<nMinKF) // Atlas에 있는 KeyFrame의 수가 10 보다 작은 경우 함수 종료
        return;

    // Retrieve all keyframe in temporal order
    list<KeyFrame*> lpKF;
    KeyFrame* pKF = mpCurrentKeyFrame; // KeyFrame을 가리키는 포인터 선언
    while(pKF->mPrevKF) // KeyFrame의 이전 KeyFrame이 존재 한다면 loop
    {
        lpKF.push_front(pKF); // keyFrame 배열에 KeyFrame 삽입
        pKF = pKF->mPrevKF; // 이전 keyFrame을 변경
    }
    lpKF.push_front(pKF); // keyFrame 배열에 KeyFrame을 삽입
    vector<KeyFrame*> vpKF(lpKF.begin(),lpKF.end()); // keyFrame 배열의 요소를 복사

    if(vpKF.size()<nMinKF) // keyFrame 배열의 크기가 최소 KeyFrame(10) 수보다 작으면 함수 종료
        return;

    mFirstTs=vpKF.front()->mTimeStamp; // keyFrame 배열의 첫 요소 (가장 마지막에 삽입된 KeyFrame)의 Timestamp를 첫 Timestamp로 초기화
    if(mpCurrentKeyFrame->mTimeStamp-mFirstTs<minTime) // 현재 KeyFrame의 Timestamp와 첫 Timestamp와의 차이가 최소 시간(2)보다 작은 경우 함수 종료
        return;

    bInitializing = true; // 초기화가 이루어졌다고 Flag 설정

    while(CheckNewKeyFrames()) // 새로운 KeyFrame 배열에 keyFrame이 존재하면 loop
    {
        ProcessNewKeyFrame(); // 현재 KeyFrame에 대해 Bag of Word를 계산하고 MapPoint를 구함
        vpKF.push_back(mpCurrentKeyFrame); // 2개의 배열에 현재 KeyFrame을 삽입
        lpKF.push_back(mpCurrentKeyFrame);
    }

    const int N = vpKF.size();
    IMU::Bias b(0,0,0,0,0,0); // IMU의 bias 객체 생성

    // Compute and KF velocities mRwg estimation
    if (!mpCurrentKeyFrame->GetMap()->isImuInitialized()) // 현재 KeyFrame의 Map에 IMU가 초기화 되지 않았다면
    {
        Eigen::Matrix3f Rwg; // 회전 행렬 생성
        Eigen::Vector3f dirG; // 방향 벡터 생성
        dirG.setZero(); // 벡터를 0으로 초기화
        for(vector<KeyFrame*>::iterator itKF = vpKF.begin(); itKF!=vpKF.end(); itKF++) // KeyFrame 배열 크기만큼 loop
        {
            if (!(*itKF)->mpImuPreintegrated) // KeyFrame의 IMU Preintegration 객체가 존재하지 않다면 무시
                continue;
            if (!(*itKF)->mPrevKF) // keyFrame의 이전 KeyFrame이 존재하지 않다면 무시
                continue;

            dirG -= (*itKF)->mPrevKF->GetImuRotation() * (*itKF)->mpImuPreintegrated->GetUpdatedDeltaVelocity(); // 이전 KeyFrame의 IMU Rotation 행렬과 Preintegration한 Velocity의 곱을 누적
            Eigen::Vector3f _vel = ((*itKF)->GetImuPosition() - (*itKF)->mPrevKF->GetImuPosition())/(*itKF)->mpImuPreintegrated->dT; // KeyFrame의 IMU Pose와 이전 KeyFrame의 Pose의 차를 시간 차이의 합과 나눠 속도 값으로 결정
            (*itKF)->SetVelocity(_vel); // KeyFrame의 속도 값을 결정
            (*itKF)->mPrevKF->SetVelocity(_vel); // 이전 KeyFrame의 속도 값을 결정
        }

        dirG = dirG/dirG.norm(); // 법선 벡터를 방향 벡터로 결정
        Eigen::Vector3f gI(0.0f, 0.0f, -1.0f); // 아래를 향하는 단위 벡터를 선언
        Eigen::Vector3f v = gI.cross(dirG); // 방향 벡터와 수평인 벡터를 구함
        const float nv = v.norm(); // 회전 축의 크기를 결정
        const float cosg = gI.dot(dirG); // 방향 벡터와 아래를 향하는 단위 벡터간의 내적을 구함
        const float ang = acos(cosg); // 내적값의 아크 코사인 값을 각도로 결정
        Eigen::Vector3f vzg = v*ang/nv; // 회전 값 결정
        Rwg = Sophus::SO3f::exp(vzg).matrix(); // 회전 행렬 생성
        mRwg = Rwg.cast<double>(); // 회전 행렬 저장
        mTinit = mpCurrentKeyFrame->mTimeStamp-mFirstTs; // 초기 Timestamp와 현재 KeyFrame의 Timestamp 간의 차이 값 저장
    }
    else
    {
        mRwg = Eigen::Matrix3d::Identity(); // 단위 행렬 생성
        mbg = mpCurrentKeyFrame->GetGyroBias().cast<double>(); // 현재 KeyFrame에서 Gyro bias를 가져옴
        mba = mpCurrentKeyFrame->GetAccBias().cast<double>(); // 현재 KeyFrame에서 가속도 bias를 가져옴
    }

    mScale=1.0; // scale 값 초기화

    mInitTime = mpTracker->mLastFrame.mTimeStamp-vpKF.front()->mTimeStamp; // Tracking 객체의 이전 Frame의 Timestamp와 keyFrame 배열에 가장 마지막에 삽입된 KeyFrame의 Timestamp의 차이를 초기 시간으로 초기화

    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    Optimizer::InertialOptimization(mpAtlas->GetCurrentMap(), mRwg, mScale, mbg, mba, mbMonocular, infoInertial, false, false, priorG, priorA); // 센서 최적화 진행

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    if (mScale<1e-1) // 최적화 후 scale이 0.1 보다 작은 경우
    {
        cout << "scale too small" << endl;
        bInitializing=false; // 초기화가 이루어지지 않았다고 flag 설정
        return; // 함수 종료
    }

    // Before this line we are not changing the map
    {
        unique_lock<mutex> lock(mpAtlas->GetCurrentMap()->mMutexMapUpdate);
        // scale에서 1을 뺀 결과의 절대 값이 0.00001 보다 크거나 Monocular를 사용하지 않으면
        if ((fabs(mScale - 1.f) > 0.00001) || !mbMonocular) {
            Sophus::SE3f Twg(mRwg.cast<float>().transpose(), Eigen::Vector3f::Zero()); // Imu position 행렬 객체 생성
            mpAtlas->GetCurrentMap()->ApplyScaledRotation(Twg, mScale, true); // Map에 있는 KeyFrame과 MapPoint의 pose를 재설정
            mpTracker->UpdateFrameIMU(mScale, vpKF[0]->GetImuBias(), mpCurrentKeyFrame); // Tracking 객체에 있는 Frame의 pose 정보를 갱신
        }

        // Check if initialization OK
        if (!mpAtlas->isImuInitialized()) // Atlas의 IMU가 초기화되지 않았을 경우
            for (int i = 0; i < N; i++) {
                KeyFrame *pKF2 = vpKF[i]; // KeyFrame의 배열에서 keyFrame을 가져옴
                pKF2->bImu = true; // KeyFrame에 Map의 IMU가 초기화 되었다고 Flag 설정
            }
    }

    mpTracker->UpdateFrameIMU(1.0,vpKF[0]->GetImuBias(),mpCurrentKeyFrame); // Tracking 객체에 있는 Frame의 pose 정보를 갱신
    if (!mpAtlas->isImuInitialized()) // Atlas의 IMU가 초기화되지 않았을 경우
    {
        mpAtlas->SetImuInitialized(); // Atlas의 Map에 IMU가 초기화 되었다고 설정
        mpTracker->t0IMU = mpTracker->mCurrentFrame.mTimeStamp; // Tracking 객체의 초기화 IMU timestamp를 Tracking 객체의 현재 Frame Timestamp로 초기화
        mpCurrentKeyFrame->bImu = true; // 현재 KeyFrame에 Map의 IMU가 초기화 되었다고 Flag 설정
    }

    std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
    if (bFIBA) // Full Inertial BA를 수행해야 하는 경우 (true 값 인자)
    {
        if (priorA!=0.f) // 최적화 기준이 되는 가속도 값이 0이 아닌 경우
            Optimizer::FullInertialBA(mpAtlas->GetCurrentMap(), 100, false, mpCurrentKeyFrame->mnId, NULL, true, priorG, priorA);
        else
            Optimizer::FullInertialBA(mpAtlas->GetCurrentMap(), 100, false, mpCurrentKeyFrame->mnId, NULL, false);
    }

    std::chrono::steady_clock::time_point t5 = std::chrono::steady_clock::now();

    Verbose::PrintMess("Global Bundle Adjustment finished\nUpdating map ...", Verbose::VERBOSITY_NORMAL);

    // Get Map Mutex
    unique_lock<mutex> lock(mpAtlas->GetCurrentMap()->mMutexMapUpdate);

    unsigned long GBAid = mpCurrentKeyFrame->mnId; // Global BA KeyFrame ID를 선언

    // Process keyframes in the queue
    while(CheckNewKeyFrames()) // 새로운 KeyFrame배열에 KeyFrame이 존재하면 loop
    {
        ProcessNewKeyFrame(); // 현재 KeyFrame에 대해 Bag of Word를 계산하고 MapPoint를 구함
        vpKF.push_back(mpCurrentKeyFrame); // keyFrame 배열에 현재 KeyFrame을 삽입
        lpKF.push_back(mpCurrentKeyFrame);
    }

    // Correct keyframes starting at map first keyframe
    list<KeyFrame*> lpKFtoCheck(mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.begin(),mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.end()); // Atlas의 현재 Map에서 KeyFrame을 가져옴

    while(!lpKFtoCheck.empty()) // Map에 있는 KeyFrame 배열이 비어있지 않다면 loop
    {
        KeyFrame* pKF = lpKFtoCheck.front(); // 배열에 가장 앞에있는 KeyFrame을 가져옴
        const set<KeyFrame*> sChilds = pKF->GetChilds(); // KeyFrame의 자식 KeyFrame을 가져옴
        Sophus::SE3f Twc = pKF->GetPoseInverse(); // keyFrame의 pose를 가져옴
        for(set<KeyFrame*>::const_iterator sit=sChilds.begin();sit!=sChilds.end();sit++) // 자식 KeyFrame 배열의 크기만큼 loop
        {
            KeyFrame* pChild = *sit;
            if(!pChild || pChild->isBad()) // 자식 KeyFrame이 존재하지 않거나 Bad 상태인 경우 무시
                continue;

            if(pChild->mnBAGlobalForKF!=GBAid) // 자식 KeyFrame의 Global BA KeyFrame ID가 Global BA KeyFrame ID와 다를 경우
            {
                Sophus::SE3f Tchildc = pChild->GetPose() * Twc; // 자식 KeyFrame의 pose와 부모 KeyFrame의 pose를 곱함
                pChild->mTcwGBA = Tchildc * pKF->mTcwGBA; // 자식 KeyFrame의 Global BA pose를 곱한 pose와 곱해서 저장

                Sophus::SO3f Rcor = pChild->mTcwGBA.so3().inverse() * pChild->GetPose().so3();
                if(pChild->isVelocitySet()){ // 자식 keyFrame에 속도 정보가 있는 경우
                    pChild->mVwbGBA = Rcor * pChild->GetVelocity(); // 자식 keyFrame의 속도를 통해 BA 속도로 결정
                }
                else {
                    Verbose::PrintMess("Child velocity empty!! ", Verbose::VERBOSITY_NORMAL);
                }

                pChild->mBiasGBA = pChild->GetImuBias(); // 자식 KeyFrame의 BA Bias를 초기화
                pChild->mnBAGlobalForKF = GBAid; // 자식 KeyFrame의 BA KeyFrame ID를 초기화

            }
            lpKFtoCheck.push_back(pChild); // Map에 있는 KeyFrame 배열에 자식 keyFrame을 삽입
        }

        pKF->mTcwBefGBA = pKF->GetPose(); // KeyFrame의 pose를 가져와 BA 이전 pose로 설정
        pKF->SetPose(pKF->mTcwGBA); // keyFrame의 pose를 Global BA pose로 설정

        if(pKF->bImu) // KeyFrame에 Map의 IMU가 초기화 되었다면
        {
            pKF->mVwbBefGBA = pKF->GetVelocity(); // KeyFrame의 속도를 가져와 BA 이전 속도로 설정
            pKF->SetVelocity(pKF->mVwbGBA); // 속도 및 bias를 설정
            pKF->SetNewBias(pKF->mBiasGBA);
        } else {
            cout << "KF " << pKF->mnId << " not set to inertial!! \n";
        }

        lpKFtoCheck.pop_front(); // 배열에 가장 앞에 있는 KeyFrame을 버림
    }

    // Correct MapPoints
    const vector<MapPoint*> vpMPs = mpAtlas->GetCurrentMap()->GetAllMapPoints(); // 현재 Map에 있는 모든 MapPoint를 가져옴

    for(size_t i=0; i<vpMPs.size(); i++) // Map에 있는 모든 MapPoint 배열의 크기만큼 loop
    {
        MapPoint* pMP = vpMPs[i];

        if(pMP->isBad()) // MapPoint가 Bad 판정인 경우 무시
            continue;

        if(pMP->mnBAGlobalForKF==GBAid) // MapPoint의 BA KeyFrame ID가 Global BA KeyFrame ID와 같을 경우
        {
            // If optimized by Global BA, just update
            pMP->SetWorldPos(pMP->mPosGBA); // MapPoint의 pose를 MapPoint의 Global BA pose로 설정
        }
        else
        {
            // Update according to the correction of its reference keyframe
            KeyFrame* pRefKF = pMP->GetReferenceKeyFrame(); // MapPoint의 레퍼런스 KeyFrame을 가져옴

            if(pRefKF->mnBAGlobalForKF!=GBAid) // 레퍼런스 KeyFrame의 BA KeyFrame ID가 Global BA KeyFrame ID와 다를 경우 무시
                continue;

            // Map to non-corrected camera
            Eigen::Vector3f Xc = pRefKF->mTcwBefGBA * pMP->GetWorldPos(); // 레퍼런스 KeyFrame의 BA 이전 pose와 MapPoint의 pose와 곱함

            // Backproject using corrected camera
            pMP->SetWorldPos(pRefKF->GetPoseInverse() * Xc); // 레퍼런스 KeyFrame의 pose와 곱해 MapPoint의 pose로 설정
        }
    }

    Verbose::PrintMess("Map updated!", Verbose::VERBOSITY_NORMAL);

    mnKFs=vpKF.size(); // KeyFrame 수 변경
    mIdxInit++; // 초기화 index 증가

    for(list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++) // 새로운 KeyFrame 배열의 크기만큼 loop
    {
        (*lit)->SetBadFlag(); // keyFrame을 Bad 판정으로 변경
        delete *lit; // 배열에서 삭제
    }
    mlNewKeyFrames.clear(); // 새로운 KeyFrame 배열 초기화

    mpTracker->mState=Tracking::OK; // Tracking 상태 변경
    bInitializing = false; // 초기화 Flag 변경

    mpCurrentKeyFrame->GetMap()->IncreaseChangeIndex(); // 현재 KeyFrame의 Map이 변경되었다고 index 변경

    return;
}

void LocalMapping::ScaleRefinement()
{
    // Minimum number of keyframes to compute a solution
    // Minimum time (seconds) between first and last keyframe to compute a solution. Make the difference between monocular and stereo
    // unique_lock<mutex> lock0(mMutexImuInit);
    if (mbResetRequested)
        return;

    // Retrieve all keyframes in temporal order
    list<KeyFrame*> lpKF;
    KeyFrame* pKF = mpCurrentKeyFrame;
    while(pKF->mPrevKF)
    {
        lpKF.push_front(pKF);
        pKF = pKF->mPrevKF;
    }
    lpKF.push_front(pKF);
    vector<KeyFrame*> vpKF(lpKF.begin(),lpKF.end());

    while(CheckNewKeyFrames()) // 새로운 keyFrame 배열에 keyFrame이 존재하면 loop
    {
        ProcessNewKeyFrame(); // 현재 KeyFrame에 대해 Bag of Word를 계산하고 MapPoint를 구함
        vpKF.push_back(mpCurrentKeyFrame);
        lpKF.push_back(mpCurrentKeyFrame);
    }

    const int N = vpKF.size();

    mRwg = Eigen::Matrix3d::Identity();
    mScale=1.0;

    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    Optimizer::InertialOptimization(mpAtlas->GetCurrentMap(), mRwg, mScale);
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    if (mScale<1e-1) // 1e-1
    {
        cout << "scale too small" << endl;
        bInitializing=false;
        return;
    }
    
    Sophus::SO3d so3wg(mRwg);
    // Before this line we are not changing the map
    unique_lock<mutex> lock(mpAtlas->GetCurrentMap()->mMutexMapUpdate);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    if ((fabs(mScale-1.f)>0.002)||!mbMonocular)
    {
        Sophus::SE3f Tgw(mRwg.cast<float>().transpose(),Eigen::Vector3f::Zero());
        mpAtlas->GetCurrentMap()->ApplyScaledRotation(Tgw,mScale,true);
        mpTracker->UpdateFrameIMU(mScale,mpCurrentKeyFrame->GetImuBias(),mpCurrentKeyFrame);
    }
    std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();

    for(list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
    {
        (*lit)->SetBadFlag();
        delete *lit;
    }
    mlNewKeyFrames.clear();

    double t_inertial_only = std::chrono::duration_cast<std::chrono::duration<double> >(t1 - t0).count();

    // To perform pose-inertial opt w.r.t. last keyframe
    mpCurrentKeyFrame->GetMap()->IncreaseChangeIndex();

    return;
}



bool LocalMapping::IsInitializing()
{
    return bInitializing;
}


double LocalMapping::GetCurrKFTime()
{

    if (mpCurrentKeyFrame)
    {
        return mpCurrentKeyFrame->mTimeStamp;
    }
    else
        return 0.0;
}

KeyFrame* LocalMapping::GetCurrKF()
{
    return mpCurrentKeyFrame;
}

} //namespace ORB_SLAM
