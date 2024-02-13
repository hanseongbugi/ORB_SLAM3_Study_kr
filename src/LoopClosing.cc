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


#include "LoopClosing.h"

#include "Sim3Solver.h"
#include "Converter.h"
#include "Optimizer.h"
#include "ORBmatcher.h"
#include "G2oTypes.h"

#include<mutex>
#include<thread>


namespace ORB_SLAM3
{

LoopClosing::LoopClosing(Atlas *pAtlas, KeyFrameDatabase *pDB, ORBVocabulary *pVoc, const bool bFixScale, const bool bActiveLC):
    mbResetRequested(false), mbResetActiveMapRequested(false), mbFinishRequested(false), mbFinished(true), mpAtlas(pAtlas),
    mpKeyFrameDB(pDB), mpORBVocabulary(pVoc), mpMatchedKF(NULL), mLastLoopKFid(0), mbRunningGBA(false), mbFinishedGBA(true),
    mbStopGBA(false), mpThreadGBA(NULL), mbFixScale(bFixScale), mnFullBAIdx(0), mnLoopNumCoincidences(0), mnMergeNumCoincidences(0),
    mbLoopDetected(false), mbMergeDetected(false), mnLoopNumNotFound(0), mnMergeNumNotFound(0), mbActiveLC(bActiveLC)
{
    mnCovisibilityConsistencyTh = 3;
    mpLastCurrentKF = static_cast<KeyFrame*>(NULL);

#ifdef REGISTER_TIMES

    vdDataQuery_ms.clear();
    vdEstSim3_ms.clear();
    vdPRTotal_ms.clear();

    vdMergeMaps_ms.clear();
    vdWeldingBA_ms.clear();
    vdMergeOptEss_ms.clear();
    vdMergeTotal_ms.clear();
    vnMergeKFs.clear();
    vnMergeMPs.clear();
    nMerges = 0;

    vdLoopFusion_ms.clear();
    vdLoopOptEss_ms.clear();
    vdLoopTotal_ms.clear();
    vnLoopKFs.clear();
    nLoop = 0;

    vdGBA_ms.clear();
    vdUpdateMap_ms.clear();
    vdFGBATotal_ms.clear();
    vnGBAKFs.clear();
    vnGBAMPs.clear();
    nFGBA_exec = 0;
    nFGBA_abort = 0;

#endif

    mstrFolderSubTraj = "SubTrajectories/";
    mnNumCorrection = 0;
    mnCorrectionGBA = 0;
}

void LoopClosing::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

void LoopClosing::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}


void LoopClosing::Run()
{
    mbFinished =false; // LoopClosing 스레드의 작업이 종료되었다고 나타낼 flag 변수 초기화

    while(1)
    {

        //NEW LOOP AND MERGE DETECTION ALGORITHM
        //----------------------------


        if(CheckNewKeyFrames()) // loop KeyFrame 배열이 비어있지 않다면
        {
            if(mpLastCurrentKF) // 이전 KeyFrame이 존재하면다면
            {
                mpLastCurrentKF->mvpLoopCandKFs.clear(); // 이전 KeyFrame의 loop Candidate KeyFrame 배열을 초기화
                mpLastCurrentKF->mvpMergeCandKFs.clear(); // 이전 keyFrame의 Merge Candidate KeyFrame 배열을 초기화
            }
#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_StartPR = std::chrono::steady_clock::now();
#endif

            bool bFindedRegion = NewDetectCommonRegions(); // 현재 KeyFrame과 매칭 KeyFrame간 공통된 부분이 있는지 감지. 존재한다면 true

#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndPR = std::chrono::steady_clock::now();

            double timePRTotal = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndPR - time_StartPR).count();
            vdPRTotal_ms.push_back(timePRTotal);
#endif
            if(bFindedRegion) // loop나 Merge가 감지 되었다면
            {
                if(mbMergeDetected) // Merge가 감지 되었다면
                {
                    if ((mpTracker->mSensor==System::IMU_MONOCULAR || mpTracker->mSensor==System::IMU_STEREO || mpTracker->mSensor==System::IMU_RGBD) &&
                        (!mpCurrentKF->GetMap()->isImuInitialized())) // IMU를 사용한 Tracking이고 현재 keyFrame의 Map에 IMU가 초기화 되지 않았을 경우
                    {
                        cout << "IMU is not initilized, merge is aborted" << endl;
                    }
                    else
                    {
                        Sophus::SE3d mTmw = mpMergeMatchedKF->GetPose().cast<double>(); // Merge 매칭 keyFrame의 pose를 가져옴
                        g2o::Sim3 gSmw2(mTmw.unit_quaternion(), mTmw.translation(), 1.0); // pose의 쿼터니언과 translation을 통해 공간 변환 생성
                        Sophus::SE3d mTcw = mpCurrentKF->GetPose().cast<double>(); // 현재 keyFrame의 pose를 가져옴
                        g2o::Sim3 gScw1(mTcw.unit_quaternion(), mTcw.translation(), 1.0); // pose의 쿼턴니언과 translation을 통해 공간 변환 생성
                        g2o::Sim3 gSw2c = mg2oMergeSlw.inverse(); // Merge 공간 변환 행렬의 역을 복사
                        g2o::Sim3 gSw1m = mg2oMergeSlw; // Merge 공간 변환 행렬 복사

                        mSold_new = (gSw2c * gScw1); // Merge 공간 변환과 현재 keyFrame의 공간 행렬을 곱해 새로운 공간 행렬을 생성


                        if(mpCurrentKF->GetMap()->IsInertial() && mpMergeMatchedKF->GetMap()->IsInertial()) // 현재 keyFrame의 map에 IMU 정보가 있고 Merge 매칭 keyFrame의 map에 IMU 정보가 있다면
                        {
                            cout << "Merge check transformation with IMU" << endl;
                            if(mSold_new.scale()<0.90||mSold_new.scale()>1.1){ // 공간의 scale이 0.9에서 1.1 사이인 경우
                                mpMergeLastCurrentKF->SetErase(); // 이전 keyFrame을 Bad 판정으로 변경
                                mpMergeMatchedKF->SetErase(); // Merge 매칭 keyFrame을 Bad 판정으로 변경
                                mnMergeNumCoincidences = 0; // Merge 감지 수 초기화
                                mvpMergeMatchedMPs.clear(); // Merge 매칭 MapPoint 배열 초기화
                                mvpMergeMPs.clear(); // MapPoint 배열 초기화
                                mnMergeNumNotFound = 0; // Merge가 감지되지 않았을 경우 count 초기화
                                mbMergeDetected = false; // Merge가 감지되지 않았다고 flag 설정
                                Verbose::PrintMess("scale bad estimated. Abort merging", Verbose::VERBOSITY_NORMAL);
                                continue; // 아래 무시
                            }
                            // If inertial, force only yaw
                            if ((mpTracker->mSensor==System::IMU_MONOCULAR || mpTracker->mSensor==System::IMU_STEREO || mpTracker->mSensor==System::IMU_RGBD) &&
                                   mpCurrentKF->GetMap()->GetIniertialBA1()) // IMU를 사용한 Tracking 중이고 현재 KeyFrame의 Map에 IniertialBA1이 일어났을 경우
                            {
                                Eigen::Vector3d phi = LogSO3(mSold_new.rotation().toRotationMatrix()); // 새로운 공간 행렬의 회전 부분에 대한 로그 매개변수를 계산
                                phi(0)=0; // 로그 매개 변수의 Roll 및 Pitch에 해당하는 첫 번째와 두 번째 성분을 0으로 설정하여 Yaw만 유지
                                phi(1)=0;
                                mSold_new = g2o::Sim3(ExpSO3(phi),mSold_new.translation(),1.0); // 로그 매개변수를 사용하여 공간 행렬의 회전 부분을 업데이트
                            }
                        }

                        mg2oMergeSmw = gSmw2 * gSw2c * gScw1; // Merge된 KeyFrame의 공간 변환과 Merge 공간 변환 행렬의 역행렬, 현재 keyFrame의 공간 행렬을 곱해 world 좌표 변환 행렬 생성

                        mg2oMergeScw = mg2oMergeSlw; // Merge 공간 변환 행렬 복사

                        //mpTracker->SetStepByStep(true);

                        Verbose::PrintMess("*Merge detected", Verbose::VERBOSITY_QUIET);

#ifdef REGISTER_TIMES
                        std::chrono::steady_clock::time_point time_StartMerge = std::chrono::steady_clock::now();

                        nMerges += 1; // merge 수 증가
#endif
                        // TODO UNCOMMENT
                        // IMU를 사용한 Tracking 중 인 경우
                        if (mpTracker->mSensor==System::IMU_MONOCULAR ||mpTracker->mSensor==System::IMU_STEREO || mpTracker->mSensor==System::IMU_RGBD) 
                            MergeLocal2(); // Merge를 진행 (Graph 최적화)
                        else
                            MergeLocal();

#ifdef REGISTER_TIMES
                        std::chrono::steady_clock::time_point time_EndMerge = std::chrono::steady_clock::now();

                        double timeMergeTotal = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndMerge - time_StartMerge).count();
                        vdMergeTotal_ms.push_back(timeMergeTotal);
#endif

                        Verbose::PrintMess("Merge finished!", Verbose::VERBOSITY_QUIET);
                    }

                    vdPR_CurrentTime.push_back(mpCurrentKF->mTimeStamp); // 현재 KeyFrame의 Timestamp를 배열에 삽입
                    vdPR_MatchedTime.push_back(mpMergeMatchedKF->mTimeStamp); // Merge 매칭 KeyFrame의 Timestamp를 배열에 삽입
                    vnPR_TypeRecogn.push_back(1); // 인식 형태 배열에 Type 삽입

                    // Reset all variables
                    mpMergeLastCurrentKF->SetErase(); // 이전 KeyFrame을 Bad 상태로 변경
                    mpMergeMatchedKF->SetErase(); // Merge 매칭 KeyFrame을 Bad 상태로 변경
                    mnMergeNumCoincidences = 0; // Merge 감지 수 초기화
                    mvpMergeMatchedMPs.clear(); // 매칭 MapPoint 배열 초기화되지 않았다고 f
                    mvpMergeMPs.clear(); // MapPoint 배열 초기화
                    mnMergeNumNotFound = 0; // Merge가 감지되지 않았을 경우 count 초기화
                    mbMergeDetected = false; // Merge가 감지 flag 초기화

                    if(mbLoopDetected) // Loop가 감지 되었다면
                    {
                        // Reset Loop variables
                        mpLoopLastCurrentKF->SetErase(); // 이전 KeyFrame을 Bad 상태로 변경
                        mpLoopMatchedKF->SetErase(); // Loop 매칭 KeyFrame을 Bad 상태로 변경
                        mnLoopNumCoincidences = 0; // Loop 감지 수 초기화
                        mvpLoopMatchedMPs.clear(); // 매칭 MapPoint 배열 초기화
                        mvpLoopMPs.clear(); // MapPoint 배열 초기화
                        mnLoopNumNotFound = 0; // Loop가 감지되지 않았을 경우 count 초기화
                        mbLoopDetected = false; // Loop가 감지 flag 초기화
                    }

                }

                if(mbLoopDetected) // Loop가 감지되었을 경우
                {
                    bool bGoodLoop = true; // Good Loop임을 나타내는 Flag 초기화
                    vdPR_CurrentTime.push_back(mpCurrentKF->mTimeStamp); // 현재 KeyFrame의 Timestamp를 배열에 삽입
                    vdPR_MatchedTime.push_back(mpLoopMatchedKF->mTimeStamp); // 매칭 keyFrame의 Timestamp를 배열에 삽입
                    vnPR_TypeRecogn.push_back(0); // 인식 형태 배열에 Type 삽입

                    Verbose::PrintMess("*Loop detected", Verbose::VERBOSITY_QUIET);

                    mg2oLoopScw = mg2oLoopSlw; // loop에 대한 변환을 초기화 //*mvg2oSim3LoopTcw[nCurrentIndex];
                    if(mpCurrentKF->GetMap()->IsInertial()) // 현재 KeyFrame의 Map에 IMU가 초기화 된 경우
                    {
                        Sophus::SE3d Twc = mpCurrentKF->GetPoseInverse().cast<double>(); // 현재 KeyFrame의 카메라 위치를 가져옴
                        g2o::Sim3 g2oTwc(Twc.unit_quaternion(),Twc.translation(),1.0); // 공간 변환을 생성
                        g2o::Sim3 g2oSww_new = g2oTwc*mg2oLoopScw; // loop변환과 공간 변환을 곱해 새로운 행렬을 생성

                        Eigen::Vector3d phi = LogSO3(g2oSww_new.rotation().toRotationMatrix()); // 회전 부분에 대한 로그 매핑을 수행하여 회전 벡터를 얻는다.
                        cout << "phi = " << phi.transpose() << endl; 
                        if (fabs(phi(0))<0.008f && fabs(phi(1))<0.008f && fabs(phi(2))<0.349f) // raw, pitch, yaw 값이 임계치를 넘지 않는다면
                        {
                            if(mpCurrentKF->GetMap()->IsInertial()) // 현재 KeyFrame의 Map에 IMU가 초기화 된 경우
                            {
                                // If inertial, force only yaw
                                if ((mpTracker->mSensor==System::IMU_MONOCULAR ||mpTracker->mSensor==System::IMU_STEREO || mpTracker->mSensor==System::IMU_RGBD) &&
                                        mpCurrentKF->GetMap()->GetIniertialBA2()) // IMU을 사용한 추적 중이고 현재 KeyFrame의 Map에 IniertialBA2가 이루어 졌다면
                                {
                                    phi(0)=0; // raw와 pitch를 0으로 고정
                                    phi(1)=0;
                                    g2oSww_new = g2o::Sim3(ExpSO3(phi),g2oSww_new.translation(),1.0); // 로그 매개변수를 사용하여 행렬의 회전 부분을 업데이트
                                    mg2oLoopScw = g2oTwc.inverse()*g2oSww_new; // loop에 대한 변환을 갱신
                                }
                            }

                        }
                        else
                        {
                            cout << "BAD LOOP!!!" << endl;
                            bGoodLoop = false; // Bad loop라고 flag 설정
                        }

                    }

                    if (bGoodLoop) { // Good Loop인 경우

                        mvpLoopMapPoints = mvpLoopMPs; // Loop MapPoint 배열을 복사

#ifdef REGISTER_TIMES
                        std::chrono::steady_clock::time_point time_StartLoop = std::chrono::steady_clock::now();

                        nLoop += 1; // loop 수 증가

#endif
                        CorrectLoop(); // loop를 보정하고 최적화
#ifdef REGISTER_TIMES
                        std::chrono::steady_clock::time_point time_EndLoop = std::chrono::steady_clock::now();

                        double timeLoopTotal = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndLoop - time_StartLoop).count();
                        vdLoopTotal_ms.push_back(timeLoopTotal);
#endif

                        mnNumCorrection += 1; // Loop 수집 수 증가
                    }

                    // Reset all variables
                    mpLoopLastCurrentKF->SetErase(); // 이전 KeyFrame을 Bad 상태로 변경
                    mpLoopMatchedKF->SetErase(); // 매칭 KeyFrame을 Bad 상태로 변경
                    mnLoopNumCoincidences = 0; // loop 감지 수 초기화
                    mvpLoopMatchedMPs.clear(); // 매칭 MapPoint 배열 초기화
                    mvpLoopMPs.clear(); // MapPoint 배열 초기화
                    mnLoopNumNotFound = 0; // Loop가 아닌 수 초기화
                    mbLoopDetected = false; // Loop가 감지 flag 초기화
                }

            }
            mpLastCurrentKF = mpCurrentKF; // 이전 KeyFrame을 현재 keyFrame으로 변경
        }

        ResetIfRequested(); // Reset 요청이 있다면 reset 진행

        if(CheckFinish()){
            break;
        }

        usleep(5000);
    }

    SetFinish(); // Loop Closing 스레드 종료
}

void LoopClosing::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    if(pKF->mnId!=0)
        mlpLoopKeyFrameQueue.push_back(pKF);
}

bool LoopClosing::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    return(!mlpLoopKeyFrameQueue.empty()); // Loop KeyFrame 배열이 비어있지 않다면 true 반환
}

bool LoopClosing::NewDetectCommonRegions()
{
    // To deactivate placerecognition. No loopclosing nor merging will be performed
    if(!mbActiveLC) // Loop Closing 스레드가 동작중이지 않은 경우
        return false; // Detect 실패

    {
        unique_lock<mutex> lock(mMutexLoopQueue);
        mpCurrentKF = mlpLoopKeyFrameQueue.front(); // 가장 먼저 삽입된 loop KeyFrame을 현재 KeyFrame으로 초기화
        mlpLoopKeyFrameQueue.pop_front(); // 현재 KeyFrame으로 초기화한 loop KeyFrame을 배열에서 제거
        // Avoid that a keyframe can be erased while it is being process by this thread
        mpCurrentKF->SetNotErase(); // 현재 KeyFrame이 제거되지 않도록 flag 설정
        mpCurrentKF->mbCurrentPlaceRecognition = true; // 현재 KeyFrame이 현재 Place Recognition 중이라고 flag 설정

        mpLastMap = mpCurrentKF->GetMap(); // 현재 KeyFrame의 Map을 가져와 이전 Map으로 초기화
    }

    if(mpLastMap->IsInertial() && !mpLastMap->GetIniertialBA2()) // 이전 Map에 IMU가 초기화 되었고, 이전 Map에 InertialBA2가 이루어지지 않았을 경우
    {
        mpKeyFrameDB->add(mpCurrentKF); // KeyFrame Database에 현재 KeyFrame을 삽입
        mpCurrentKF->SetErase(); // 현재 KeyFrame을 지움 (Bad 판정으로 바꿈)
        return false; // Detect 실패
    }

    // STEREO를 사용해서 Tracking 중이고 이전 Map에 있는 KeyFrame의 수가 5보다 작은 경우
    if(mpTracker->mSensor == System::STEREO && mpLastMap->GetAllKeyFrames().size() < 5) //12
    {
        // cout << "LoopClousure: Stereo KF inserted without check: " << mpCurrentKF->mnId << endl;
        mpKeyFrameDB->add(mpCurrentKF); // KeyFrame Database에 현재 KeyFrame을 삽입
        mpCurrentKF->SetErase(); // 현재 KeyFrame을 지움 (Bad 판정으로 바꿈)
        return false; // Detect 실패
    }

    if(mpLastMap->GetAllKeyFrames().size() < 12) // 이전 Map에 있는 KeyFrame의 수가 12보다 작은 경우
    {
        // cout << "LoopClousure: Stereo KF inserted without check, map is small: " << mpCurrentKF->mnId << endl;
        mpKeyFrameDB->add(mpCurrentKF); // KeyFrame Database에 현재 KeyFrame을 삽입
        mpCurrentKF->SetErase(); // 현재 KeyFrame을 지움 (Bad 판정으로 바꿈)
        return false; // Detect 실패
    }

    //cout << "LoopClousure: Checking KF: " << mpCurrentKF->mnId << endl;

    //Check the last candidates with geometric validation
    // Loop candidates
    bool bLoopDetectedInKF = false; // Loop Detected Flag 초기화
    bool bCheckSpatial = false; // 공간 감지 Flag 초기화

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartEstSim3_1 = std::chrono::steady_clock::now();
#endif
    if(mnLoopNumCoincidences > 0) // Loop 감지 수가 0보다 큰 경우
    {
        bCheckSpatial = true; // 공간이 감지 되었다고 설정
        // Find from the last KF candidates
        Sophus::SE3d mTcl = (mpCurrentKF->GetPose() * mpLoopLastCurrentKF->GetPoseInverse()).cast<double>(); // 현재 KeyFrame의 pose와 이전 KeyFrame의 pose를 곱해 pose 변환 행렬을 구함
        g2o::Sim3 gScl(mTcl.unit_quaternion(),mTcl.translation(),1.0); // pose 변환 행렬의 쿼터니언과 translation 행렬을 통해 공간 변환 행렬 생성
        g2o::Sim3 gScw = gScl * mg2oLoopSlw; // 공간 변환 행렬과 loop일 때 공간 변환 행렬을 곱해 변환된 pose를 구함
        int numProjMatches = 0; // 매칭 수 초기화
        vector<MapPoint*> vpMatchedMPs;
        bool bCommonRegion = DetectAndReffineSim3FromLastKF(mpCurrentKF, mpLoopMatchedKF, gScw, numProjMatches, mvpLoopMPs, vpMatchedMPs); // 현재 KeyFrame과 매칭 KeyFrame간의 매칭 MapPoint와 pose를 구함. 매칭에 성공하면 true 반환
        if(bCommonRegion) // 매칭에 성공한 경우
        {

            bLoopDetectedInKF = true; // KeyFrame 속에서 loop가 감지 되었다고 flag 설정

            mnLoopNumCoincidences++; // loop 감지 수 증가
            mpLoopLastCurrentKF->SetErase(); // 이전 keyFrame을 지움 (Bad 판정으로 변경)
            mpLoopLastCurrentKF = mpCurrentKF; // 이전 keyFrame을 현재 KeyFrame으로 초기화
            mg2oLoopSlw = gScw; // loop 공간 행렬 변경
            mvpLoopMatchedMPs = vpMatchedMPs; // 매칭 MapPoint 배열을 loop 매칭 MapPoint 배열로 복사


            mbLoopDetected = mnLoopNumCoincidences >= 3; // loop 감지수가 3 이상인 경우 loop 감지로 판정
            mnLoopNumNotFound = 0; // loop 감지 되지 않은 수를 0으로 초기화

            if(!mbLoopDetected) // loop가 감지되지 않은 경우
            {
                cout << "PR: Loop detected with Reffine Sim3" << endl;
            }
        }
        else
        {
            bLoopDetectedInKF = false; // keyFrame 속 loop가 감지 되지 않았다고 flag 설정

            mnLoopNumNotFound++; // loop 감지 되지 않은 수 증가
            if(mnLoopNumNotFound >= 2) // loop 감지 되지 않은 수가 2 이상인 경우
            {
                mpLoopLastCurrentKF->SetErase(); // 이전 keyFrame을 지움 (Bad 판정으로 변경)
                mpLoopMatchedKF->SetErase(); // 매칭 KeyFrame을 지움 (Bad 판정으로 변경)
                mnLoopNumCoincidences = 0; // loop 감지 수 초기화
                mvpLoopMatchedMPs.clear(); // 매칭 MapPoint 배열 초기화
                mvpLoopMPs.clear(); // MapPoint 배열 초기화
                mnLoopNumNotFound = 0; // loop 감지 되지 않은 수 초기화
            }

        }
    }

    //Merge candidates
    bool bMergeDetectedInKF = false; 
    if(mnMergeNumCoincidences > 0) // Merge 감지 수가 0보다 큰 경우
    {
        // Find from the last KF candidates
        Sophus::SE3d mTcl = (mpCurrentKF->GetPose() * mpMergeLastCurrentKF->GetPoseInverse()).cast<double>(); // 현재 KeyFrame과 이전 KeyFrame과의 pose를 곱해 pose 변환 행렬을 구함

        g2o::Sim3 gScl(mTcl.unit_quaternion(), mTcl.translation(), 1.0); // pose 변환 행렬의 쿼터니언과 translation 행렬을 통해 공간 변환 행렬 생성
        g2o::Sim3 gScw = gScl * mg2oMergeSlw; // 공간 변환 행렬과 Merge일 때 공간 변환 행렬을 곱해 변환된 pose를 구함
        int numProjMatches = 0; // 매칭 수 초기화
        vector<MapPoint*> vpMatchedMPs;
        bool bCommonRegion = DetectAndReffineSim3FromLastKF(mpCurrentKF, mpMergeMatchedKF, gScw, numProjMatches, mvpMergeMPs, vpMatchedMPs); // 현재 KeyFrame과 매칭 KeyFrame간의 매칭 MapPoint와 pose를 구함. 매칭에 성공하면 true 반환
        if(bCommonRegion) // 매칭에 성공한 경우
        {
            bMergeDetectedInKF = true; // KeyFrame 속에서 Merge가 감지 되었다고 flag 설정

            mnMergeNumCoincidences++; // Merge 감지 수 증가
            mpMergeLastCurrentKF->SetErase(); // 이전 KeyFrame을 지움 (Bad 판정으로 변경)
            mpMergeLastCurrentKF = mpCurrentKF; // 현재 KeyFrame을 이전 KeyFrame으로 변경
            mg2oMergeSlw = gScw; // Merge 공간 행렬 변경
            mvpMergeMatchedMPs = vpMatchedMPs; // 매칭 MapPoint 배열 복사

            mbMergeDetected = mnMergeNumCoincidences >= 3; // Merge 감지 수가 3 이상인 경우 Merge가 감지 되었다고 flag 설정
        }
        else
        {
            mbMergeDetected = false; // Merge가 감지되지 않았다고 flag 설정
            bMergeDetectedInKF = false; // KeyFrame 속에서 Merge가 감지 되지 않았다고 flag 설정

            mnMergeNumNotFound++; // Merge 감지 되지 않은 수 증가
            if(mnMergeNumNotFound >= 2) // Merge 감지 되지 않은 수가 2 이상인 경우
            {
                mpMergeLastCurrentKF->SetErase(); // 이전 KeyFrame을 지움 (Bad 판정으로 변경)
                mpMergeMatchedKF->SetErase(); // 매칭 KeyFrame을 지움 (Bad 판정으로 변경)
                mnMergeNumCoincidences = 0; // Merge 감지 수 초기화
                mvpMergeMatchedMPs.clear(); // 매칭 MapPoint 배열 초기화
                mvpMergeMPs.clear(); // MapPoint 배열 초기화
                mnMergeNumNotFound = 0; // Merge 감지 되지 않은 수 초기화
            }


        }
    }  
#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_EndEstSim3_1 = std::chrono::steady_clock::now();

        double timeEstSim3 = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndEstSim3_1 - time_StartEstSim3_1).count();
#endif

    if(mbMergeDetected || mbLoopDetected) // Merge나 Loop가 감지된 경우
    {
#ifdef REGISTER_TIMES
        vdEstSim3_ms.push_back(timeEstSim3);
#endif
        mpKeyFrameDB->add(mpCurrentKF); // KerFrame Database에 현재 KeyFrame 삽입
        return true; // Detect 성공
    }

    //TODO: This is only necessary if we use a minimun score for pick the best candidates
    const vector<KeyFrame*> vpConnectedKeyFrames = mpCurrentKF->GetVectorCovisibleKeyFrames(); // 현재 KeyFrame의 Covisiblity Garph를 통해 이웃 KeyFrame을 가져옴

    // Extract candidates from the bag of words
    vector<KeyFrame*> vpMergeBowCand, vpLoopBowCand;
    if(!bMergeDetectedInKF || !bLoopDetectedInKF) // KeyFrame 속 Merge나 Loop가 감지되지 않았을 경우
    {
        // Search in BoW
#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_StartQuery = std::chrono::steady_clock::now();
#endif
        mpKeyFrameDB->DetectNBestCandidates(mpCurrentKF, vpLoopBowCand, vpMergeBowCand,3); // 현재 KeyFrame을 통해 KeyFrame Database에서 Candidates를 구함
#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_EndQuery = std::chrono::steady_clock::now();

        double timeDataQuery = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndQuery - time_StartQuery).count();
        vdDataQuery_ms.push_back(timeDataQuery);
#endif
    }

#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_StartEstSim3_2 = std::chrono::steady_clock::now();
#endif
    // Check the BoW candidates if the geometric candidate list is empty
    //Loop candidates
    if(!bLoopDetectedInKF && !vpLoopBowCand.empty()) // KeyFrame 속 Loop가 감지되지 않았고 Candidates Bow 배열이 비어있는 경우
    {
        mbLoopDetected = DetectCommonRegionsFromBoW(vpLoopBowCand, mpLoopMatchedKF, mpLoopLastCurrentKF, mg2oLoopSlw, mnLoopNumCoincidences, mvpLoopMPs, mvpLoopMatchedMPs); // Bow를 통해 공통된 부분이 있는지 감지
    }
    // Merge candidates
    if(!bMergeDetectedInKF && !vpMergeBowCand.empty()) // KeyFrame 속 Merge가 감지되지 않았고 Candidates Bow 배열이 비어있는 경우
    {
        mbMergeDetected = DetectCommonRegionsFromBoW(vpMergeBowCand, mpMergeMatchedKF, mpMergeLastCurrentKF, mg2oMergeSlw, mnMergeNumCoincidences, mvpMergeMPs, mvpMergeMatchedMPs); // Bow를 통해 공통된 부분이 있는지 감지
    }

#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_EndEstSim3_2 = std::chrono::steady_clock::now();

        timeEstSim3 += std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndEstSim3_2 - time_StartEstSim3_2).count();
        vdEstSim3_ms.push_back(timeEstSim3);
#endif

    mpKeyFrameDB->add(mpCurrentKF); // KeyFrame Database에 현재 KeyFrame 추가

    if(mbMergeDetected || mbLoopDetected) // Merge나 Loop가 감지된 경우
    {
        return true; // Detect 성공
    }

    mpCurrentKF->SetErase(); // 현재 KeyFrame을 지움 (Bad 상태로 변경)
    mpCurrentKF->mbCurrentPlaceRecognition = false; // 현재 KeyFrame이 현재 Place Recognition 중이 아니라고 flag 설정

    return false; // Detect 실패
}

bool LoopClosing::DetectAndReffineSim3FromLastKF(KeyFrame* pCurrentKF, KeyFrame* pMatchedKF, g2o::Sim3 &gScw, int &nNumProjMatches,
                                                 std::vector<MapPoint*> &vpMPs, std::vector<MapPoint*> &vpMatchedMPs)
{
    set<MapPoint*> spAlreadyMatchedMPs;
    nNumProjMatches = FindMatchesByProjection(pCurrentKF, pMatchedKF, gScw, spAlreadyMatchedMPs, vpMPs, vpMatchedMPs); // 현재 KeyFrame과 매칭 keyFrame간의 keyPoint를 투영해 매칭 수를 구함

    int nProjMatches = 30;
    int nProjOptMatches = 50;
    int nProjMatchesRep = 100;

    if(nNumProjMatches >= nProjMatches) // 매칭 점 수가 30 이상인 경우
    {
        //Verbose::PrintMess("Sim3 reffine: There are " + to_string(nNumProjMatches) + " initial matches ", Verbose::VERBOSITY_DEBUG);
        Sophus::SE3d mTwm = pMatchedKF->GetPoseInverse().cast<double>(); // 매칭 KeyFrame의 pose를 가져옴
        g2o::Sim3 gSwm(mTwm.unit_quaternion(),mTwm.translation(),1.0); // 매칭 KeyFrame의 상대 변환을 생성
        g2o::Sim3 gScm = gScw * gSwm; // 카메라 pose에 상대적인 변환을 곱하여 최적화를 위한 초기 추정값 생성
        Eigen::Matrix<double, 7, 7> mHessian7x7;

        bool bFixedScale = mbFixScale; // scale을 고정할 것인지 flag 설정      // TODO CHECK; Solo para el monocular inertial
        if(mpTracker->mSensor==System::IMU_MONOCULAR && !pCurrentKF->GetMap()->GetIniertialBA2()) // Monocular를 통한 추적을 진행 중이고 현재 KeyFrame의 Map에 InertialBA2가 이루어지지 않았을 경우
            bFixedScale=false; // scale 고정하지 않는다고 flag 설정
        int numOptMatches = Optimizer::OptimizeSim3(mpCurrentKF, pMatchedKF, vpMatchedMPs, gScm, 10, bFixedScale, mHessian7x7, true); // 현재 KeyFrame과 매칭 KeyFrame간에 3D 공간에서의 상대적 pose를 최적화하고 최적화된 MapPoint 수를 반환

        //Verbose::PrintMess("Sim3 reffine: There are " + to_string(numOptMatches) + " matches after of the optimization ", Verbose::VERBOSITY_DEBUG);

        if(numOptMatches > nProjOptMatches) // 최적화된 Point의 수가 50 보다 큰 경우
        {
            g2o::Sim3 gScw_estimation(gScw.rotation(), gScw.translation(),1.0); // 카메라 pose를 통해 추정 pose 객체를 생성

            vector<MapPoint*> vpMatchedMP;
            vpMatchedMP.resize(mpCurrentKF->GetMapPointMatches().size(), static_cast<MapPoint*>(NULL)); // 매칭 MapPoint 배열 크기 할당

            nNumProjMatches = FindMatchesByProjection(pCurrentKF, pMatchedKF, gScw_estimation, spAlreadyMatchedMPs, vpMPs, vpMatchedMPs); // 현재 KeyFrame과 매칭 keyFrame간의 keyPoint를 투영해 매칭 수를 구함
            if(nNumProjMatches >= nProjMatchesRep) // 매칭 수가 100 이상인 경우
            {
                gScw = gScw_estimation; // 추정 값으로 현재 pose 값 갱신
                return true; // Detect 성공
            }
        }
    }
    return false; // Detect 실패
}

bool LoopClosing::DetectCommonRegionsFromBoW(std::vector<KeyFrame*> &vpBowCand, KeyFrame* &pMatchedKF2, KeyFrame* &pLastCurrentKF, g2o::Sim3 &g2oScw,
                                             int &nNumCoincidences, std::vector<MapPoint*> &vpMPs, std::vector<MapPoint*> &vpMatchedMPs)
{
    int nBoWMatches = 20;
    int nBoWInliers = 15;
    int nSim3Inliers = 20;
    int nProjMatches = 50;
    int nProjOptMatches = 80;

    set<KeyFrame*> spConnectedKeyFrames = mpCurrentKF->GetConnectedKeyFrames(); // 현재 KeyFrame에서 연결된 KeyFrame을 가져옴

    int nNumCovisibles = 10; // Covisiblity Graph를 통해 가져올 keyFrame 수 초기화

    ORBmatcher matcherBoW(0.9, true); // ORB matcher 객체 생성
    ORBmatcher matcher(0.75, true); // ORB matcher 객체 생성

    // Varibles to select the best numbe
    KeyFrame* pBestMatchedKF;
    int nBestMatchesReproj = 0;
    int nBestNumCoindicendes = 0;
    g2o::Sim3 g2oBestScw;
    std::vector<MapPoint*> vpBestMapPoints;
    std::vector<MapPoint*> vpBestMatchedMapPoints;

    int numCandidates = vpBowCand.size(); // Candidate 배열의 크기를 통해 Candidates KeyFrame 수 저장
    vector<int> vnStage(numCandidates, 0);
    vector<int> vnMatchesStage(numCandidates, 0);

    int index = 0;
    //Verbose::PrintMess("BoW candidates: There are " + to_string(vpBowCand.size()) + " possible candidates ", Verbose::VERBOSITY_DEBUG);
    for(KeyFrame* pKFi : vpBowCand) // Candidates 배열 순회
    {
        if(!pKFi || pKFi->isBad()) // Candidates KeyFrame이 존재하지 않거나 Bad 상태인 경우 무시
            continue;

        // std::cout << "KF candidate: " << pKFi->mnId << std::endl;
        // Current KF against KF with covisibles version
        std::vector<KeyFrame*> vpCovKFi = pKFi->GetBestCovisibilityKeyFrames(nNumCovisibles); // KeyFrame의 Covisbility Graph를 통해 이웃 KeyFrame을 찾음
        if(vpCovKFi.empty()) // 이웃 KeyFrame이 존재하지 않는 경우
        {
            std::cout << "Covisible list empty" << std::endl;
            vpCovKFi.push_back(pKFi); // Candidates KeyFrame을 이웃 KeyFrame 배열에 삽입
        }
        else
        {
            vpCovKFi.push_back(vpCovKFi[0]); // 이웃 KeyFrame 배열에 첫번째 KeyFrame을 삽입
            vpCovKFi[0] = pKFi; // 첫번째 KeyFrame을 Candidates KeyFrame으로 변경
        }


        bool bAbortByNearKF = false; // 같은 KeyFrame이 있는지 찾는 flag 초기화
        for(int j=0; j<vpCovKFi.size(); ++j) // 이웃 KeyFrame 배열의 크기만큼 loop
        {
            if(spConnectedKeyFrames.find(vpCovKFi[j]) != spConnectedKeyFrames.end()) // 이웃 KeyFrame이 현재 KeyFrame에서 연결된 KeyFrame에 존재하는 경우
            {
                bAbortByNearKF = true; // 같은 KeyFrame이 있다고 flag 설정
                break; // 반복 종료
            }
        }
        if(bAbortByNearKF) // 같은 KeyFrame이 있는 경우 아래 과정 무시 (Bow를 통한 매칭 무시)
        {
            //std::cout << "Check BoW aborted because is close to the matched one " << std::endl;
            continue;
        }
        //std::cout << "Check BoW continue because is far to the matched one " << std::endl;


        std::vector<std::vector<MapPoint*> > vvpMatchedMPs;
        vvpMatchedMPs.resize(vpCovKFi.size());
        std::set<MapPoint*> spMatchedMPi;
        int numBoWMatches = 0;

        KeyFrame* pMostBoWMatchesKF = pKFi; // Candidates KeyFrame을 Bow 매칭 KeyFrame으로 초기화
        int nMostBoWNumMatches = 0; // Bow 매칭 수 초기화

        std::vector<MapPoint*> vpMatchedPoints = std::vector<MapPoint*>(mpCurrentKF->GetMapPointMatches().size(), static_cast<MapPoint*>(NULL)); // 매칭 MapPoint 배열 초기화
        std::vector<KeyFrame*> vpKeyFrameMatchedMP = std::vector<KeyFrame*>(mpCurrentKF->GetMapPointMatches().size(), static_cast<KeyFrame*>(NULL)); // 매칭 Point를 가진 KeyFrame 배열 초기화

        int nIndexMostBoWMatchesKF=0; // 매칭 KeyFrame의 index 초기화
        for(int j=0; j<vpCovKFi.size(); ++j) // 이웃 KeyFrame 배열의 크기만큼 loop
        {
            if(!vpCovKFi[j] || vpCovKFi[j]->isBad()) // 이웃 KeyFrame이 존재하지 않거나 Bad 상태인 경우 무시
                continue;

            int num = matcherBoW.SearchByBoW(mpCurrentKF, vpCovKFi[j], vvpMatchedMPs[j]); // 현재 KeyFrame과 이웃 KeyFrame간에 Bow 매칭을 진행
            if (num > nMostBoWNumMatches) // 매칭 수가 이전 매칭 보다 큰 경우
            {
                nMostBoWNumMatches = num; // 이전 매칭 수를 현재 매칭 수로 변경
                nIndexMostBoWMatchesKF = j; // 매칭 KeyFrame index 변경
            }
        }

        for(int j=0; j<vpCovKFi.size(); ++j) // 이웃 KeyFrame 배열의 크기만큼 loop
        {
            for(int k=0; k < vvpMatchedMPs[j].size(); ++k) // 매칭 MapPoint 수만큼 loop
            {
                MapPoint* pMPi_j = vvpMatchedMPs[j][k];
                if(!pMPi_j || pMPi_j->isBad()) // MapPoint가 존재하지 않거나 Bad 상태인 경우 무시
                    continue;

                if(spMatchedMPi.find(pMPi_j) == spMatchedMPi.end()) // 매칭 MapPoint가 같은 것이 존재하지 않는다면
                {
                    spMatchedMPi.insert(pMPi_j); // 중복 방지를 위해 MapPoint 삽입
                    numBoWMatches++; // 매칭 수 증가

                    vpMatchedPoints[k]= pMPi_j; // 매칭 MapPoint 배열에 저장
                    vpKeyFrameMatchedMP[k] = vpCovKFi[j]; // 매칭 KeyFrame 배열에 저장
                }
            }
        }

        //pMostBoWMatchesKF = vpCovKFi[pMostBoWMatchesKF];
        if(numBoWMatches >= nBoWMatches)  // 매칭 수가 20 이상인 경우 // TODO pick a good threshold
        {
            // Geometric validation
            bool bFixedScale = mbFixScale; // scale을 고정할 것인지 flag 설정 
            if(mpTracker->mSensor==System::IMU_MONOCULAR && !mpCurrentKF->GetMap()->GetIniertialBA2()) // Monocular를 통한 Tracking을 진행중이고 현재 KeyFrame의 Map에 InertialBA2가 이루어지지 않았을 경우
                bFixedScale=false; // Scale을 고정하지 않는다고 설정

            Sim3Solver solver = Sim3Solver(mpCurrentKF, pMostBoWMatchesKF, vpMatchedPoints, bFixedScale, vpKeyFrameMatchedMP); // Sim3 Solver 객체 생성
            solver.SetRansacParameters(0.99, nBoWInliers, 300); // at least 15 inliers // RANSAC을 위한 파라미터 설정

            bool bNoMore = false;
            vector<bool> vbInliers;
            int nInliers;
            bool bConverge = false;
            Eigen::Matrix4f mTcm;
            while(!bConverge && !bNoMore) // 매칭 MapPoint의 수가 15개 보다 많거나 RANSAC을 통해 현재 KeyFrame과 매칭 KeyFrame간 inlier를 15개 보다 많이 구하였다면 loop 종료
            {
                mTcm = solver.iterate(20,bNoMore, vbInliers, nInliers, bConverge);
                //Verbose::PrintMess("BoW guess: Solver achieve " + to_string(nInliers) + " geometrical inliers among " + to_string(nBoWInliers) + " BoW matches", Verbose::VERBOSITY_DEBUG);
            }

            if(bConverge) // RANSAC을 통해 inlier를 15개 보다 많이 구하였다면
            {
                //std::cout << "Check BoW: SolverSim3 converged" << std::endl;

                //Verbose::PrintMess("BoW guess: Convergende with " + to_string(nInliers) + " geometrical inliers among " + to_string(nBoWInliers) + " BoW matches", Verbose::VERBOSITY_DEBUG);
                // Match by reprojection
                vpCovKFi.clear(); // 이웃 keyFrame 배열 제거
                vpCovKFi = pMostBoWMatchesKF->GetBestCovisibilityKeyFrames(nNumCovisibles); // 매칭 KeyFrame의 covisibility graph를 통해 이웃 keyFrame을 찾음
                vpCovKFi.push_back(pMostBoWMatchesKF); // 이웃 keyFrame 배열에 매칭 keyFrame 삽입
                set<KeyFrame*> spCheckKFs(vpCovKFi.begin(), vpCovKFi.end()); // keyFrame check 배열에 이웃 keyFrame 배열에 있는 keyFrame을 삽입

                //std::cout << "There are " << vpCovKFi.size() <<" near KFs" << std::endl;

                set<MapPoint*> spMapPoints;
                vector<MapPoint*> vpMapPoints;
                vector<KeyFrame*> vpKeyFrames;
                for(KeyFrame* pCovKFi : vpCovKFi) // 이웃 keyFrame 배열 loop
                {
                    for(MapPoint* pCovMPij : pCovKFi->GetMapPointMatches()) // 이웃 keyFrame의 MapPoint 배열을 가져와 loop
                    {
                        if(!pCovMPij || pCovMPij->isBad()) // MapPoint가 존재하지 않거나 Bad 상태인 경우 무시
                            continue;

                        if(spMapPoints.find(pCovMPij) == spMapPoints.end()) // 이전에 배열에 삽입되지 않은 MapPoint인 경우
                        {
                            spMapPoints.insert(pCovMPij); // 중복 방지를 위해 MapPoint 삽입
                            vpMapPoints.push_back(pCovMPij); // MapPoint 배열에 MapPoint 삽입
                            vpKeyFrames.push_back(pCovKFi); // keyFrame 배열에 keyFrame 삽입
                        }
                    }
                }

                //std::cout << "There are " << vpKeyFrames.size() <<" KFs which view all the mappoints" << std::endl;

                g2o::Sim3 gScm(solver.GetEstimatedRotation().cast<double>(),solver.GetEstimatedTranslation().cast<double>(), (double) solver.GetEstimatedScale()); // RANSAC을 통해 추정한 Rotation 행렬과 Translation 행렬, Scale 값을 통해 변환 행렬 생성
                g2o::Sim3 gSmw(pMostBoWMatchesKF->GetRotation().cast<double>(),pMostBoWMatchesKF->GetTranslation().cast<double>(),1.0); // 매칭 KeyFrame의 Rotation, Translation을 통해 변환 행렬 생성
                g2o::Sim3 gScw = gScm*gSmw; // world 변환 행렬 생성 // Similarity matrix of current from the world position
                Sophus::Sim3f mScw = Converter::toSophus(gScw);

                vector<MapPoint*> vpMatchedMP;
                vpMatchedMP.resize(mpCurrentKF->GetMapPointMatches().size(), static_cast<MapPoint*>(NULL));
                vector<KeyFrame*> vpMatchedKF;
                vpMatchedKF.resize(mpCurrentKF->GetMapPointMatches().size(), static_cast<KeyFrame*>(NULL));
                int numProjMatches = matcher.SearchByProjection(mpCurrentKF, mScw, vpMapPoints, vpKeyFrames, vpMatchedMP, vpMatchedKF, 8, 1.5); // 현재 KeyFrame과 매칭 MapPoint간에 Projection을 진행하여 매칭 수를 반환 받음 
                //cout <<"BoW: " << numProjMatches << " matches between " << vpMapPoints.size() << " points with coarse Sim3" << endl;

                if(numProjMatches >= nProjMatches) // 매칭 수가 50 이상인 경우
                {
                    // Optimize Sim3 transformation with every matches
                    Eigen::Matrix<double, 7, 7> mHessian7x7;

                    bool bFixedScale = mbFixScale; // scale을 고정할 것인지 flag 설정 
                    if(mpTracker->mSensor==System::IMU_MONOCULAR && !mpCurrentKF->GetMap()->GetIniertialBA2()) // Monocular를 통한 Tracking을 진행중이고 현재 KeyFrame의 Map에 InertialBA2가 이루어지지 않았을 경우
                        bFixedScale=false; // scale을 고정하지 않는다고 flag 설정

                    int numOptMatches = Optimizer::OptimizeSim3(mpCurrentKF, pKFi, vpMatchedMP, gScm, 10, mbFixScale, mHessian7x7, true); // 현재 keyFrame과 Candidates KeyFrame간에 3D 공간에서의 상대적 pose를 최적화하고 최적화된 MapPoint 수를 반환

                    if(numOptMatches >= nSim3Inliers) // 매칭 수가 20 이상인 경우
                    {
                        g2o::Sim3 gSmw(pMostBoWMatchesKF->GetRotation().cast<double>(),pMostBoWMatchesKF->GetTranslation().cast<double>(),1.0); // 매칭 keyFrame의 Rotation과 Translation을 통해 변환 행렬 생성
                        g2o::Sim3 gScw = gScm*gSmw; // world 변환 행렬 생성 // Similarity matrix of current from the world position
                        Sophus::Sim3f mScw = Converter::toSophus(gScw);

                        vector<MapPoint*> vpMatchedMP;
                        vpMatchedMP.resize(mpCurrentKF->GetMapPointMatches().size(), static_cast<MapPoint*>(NULL));
                        int numProjOptMatches = matcher.SearchByProjection(mpCurrentKF, mScw, vpMapPoints, vpMatchedMP, 5, 1.0); // 현재 keyFrame과 매칭 MapPoint간의 projection 진행하여 매칭 수 반환 받음

                        if(numProjOptMatches >= nProjOptMatches) // 매칭 수가 80 이상인 경우
                        {
                            int max_x = -1, min_x = 1000000; // 최소, 최대 수 초기화
                            int max_y = -1, min_y = 1000000;
                            for(MapPoint* pMPi : vpMatchedMP) // 매칭 MapPoint 배열 loop
                            {
                                if(!pMPi || pMPi->isBad()) // MapPoint가 존재하지 않거나 Bad 판정인 경우 무시
                                {
                                    continue;
                                }

                                tuple<size_t,size_t> indexes = pMPi->GetIndexInKeyFrame(pKFi); // Candidates KeyFrame에서 MapPoint의 index를 가져옴
                                int index = get<0>(indexes);
                                if(index >= 0) // index가 0 이상인 경우
                                {
                                    int coord_x = pKFi->mvKeysUn[index].pt.x; // KeyFrame의 KeyPoint 배열에서 index를 통해 검색 후 x 좌표를 가져옴
                                    if(coord_x < min_x) // x 좌표가 최소 값보다 작은 경우
                                    {
                                        min_x = coord_x; // 최소 값 갱신
                                    }
                                    if(coord_x > max_x) // x 좌표가 최대 값보다 큰 경우
                                    {
                                        max_x = coord_x; // 최대 값 갱신
                                    }
                                    int coord_y = pKFi->mvKeysUn[index].pt.y; // KeyFrame의 KeyPoint 배열에서 index를 통해 검색 후 y 좌표를 가져옴
                                    if(coord_y < min_y) // y 좌표가 최소 값보다 작은 경우
                                    {
                                        min_y = coord_y; // 최소 값 갱신
                                    }
                                    if(coord_y > max_y) // y 좌표가 최대 값보다 큰 경우
                                    {
                                        max_y = coord_y; // 최대 값 갱신
                                    }
                                }
                            }

                            int nNumKFs = 0; // keyFrame 수 초기화
                            //vpMatchedMPs = vpMatchedMP;
                            //vpMPs = vpMapPoints;
                            // Check the Sim3 transformation with the current KeyFrame covisibles
                            vector<KeyFrame*> vpCurrentCovKFs = mpCurrentKF->GetBestCovisibilityKeyFrames(nNumCovisibles); // 현재 keyFrame의 covisibility graph를 통해 이웃 keyFrame을 찾음

                            int j = 0;
                            while(nNumKFs < 3 && j<vpCurrentCovKFs.size()) // 찾은 keyFrame 수가 3보다 작고 이웃 keyFrame 배열의 크기만큼 loop가 이루어지지 않았다면 loop
                            {
                                KeyFrame* pKFj = vpCurrentCovKFs[j]; 
                                Sophus::SE3d mTjc = (pKFj->GetPose() * mpCurrentKF->GetPoseInverse()).cast<double>(); // 이웃 keyFrame의 pose와 현재 keyFrame의 pose를 곱해 pose 변환 행렬 생성
                                g2o::Sim3 gSjc(mTjc.unit_quaternion(),mTjc.translation(),1.0); // 쿼터니언과 translation 부분을 통해 상대 변환을 생성
                                g2o::Sim3 gSjw = gSjc * gScw; // 변환 pose 생성
                                int numProjMatches_j = 0;
                                vector<MapPoint*> vpMatchedMPs_j;
                                bool bValid = DetectCommonRegionsFromLastKF(pKFj,pMostBoWMatchesKF, gSjw,numProjMatches_j, vpMapPoints, vpMatchedMPs_j); // projection을 통해 이웃 keyFrame과 매칭 keyFrame간에 공통된 지점이 있는지 확인

                                if(bValid) // Detect에 성공한 경우
                                {
                                    Sophus::SE3f Tc_w = mpCurrentKF->GetPose(); // 현재 keyFrame에서 pose를 가져옴
                                    Sophus::SE3f Tw_cj = pKFj->GetPoseInverse(); // candidates keyFrame에서 pose를 가져옴
                                    Sophus::SE3f Tc_cj = Tc_w * Tw_cj; // 현재 keyFrame과 candidates keyFrame간의 pose 변환을 구함
                                    Eigen::Vector3f vector_dist = Tc_cj.translation(); // pose 변환 행렬의 translation 부분을 가져옴
                                    nNumKFs++; // keyFrame 수 증가
                                }
                                j++; // index 증가
                            }

                            if(nNumKFs < 3) // Detect한 keyFrame 수가 3 미만인 경우
                            {
                                vnStage[index] = 8; // state 값 저장
                                vnMatchesStage[index] = nNumKFs; // 매칭 state keyFrame 저장
                            }

                            if(nBestMatchesReproj < numProjOptMatches) // 현재 keyFrame과 매칭 MapPoint간의 projection 진행한 결과 매칭 수가 이전 진행된 최대 projection 결과 보다 큰 경우
                            {
                                nBestMatchesReproj = numProjOptMatches; // 이전 진행된 최대 projection 결과를 갱신
                                nBestNumCoindicendes = nNumKFs; // Best Candidates 수 갱신
                                pBestMatchedKF = pMostBoWMatchesKF; // 매칭 keyFrame 갱신
                                g2oBestScw = gScw; // world 변환 갱신
                                vpBestMapPoints = vpMapPoints; // best MapPoint 갱신
                                vpBestMatchedMapPoints = vpMatchedMP; // best 매칭 MapPoint 갱신
                            }
                        }
                    }
                }
            }
            /*else
            {
                Verbose::PrintMess("BoW candidate: it don't match with the current one", Verbose::VERBOSITY_DEBUG);
            }*/
        }
        index++; // index 증가
    }

    if(nBestMatchesReproj > 0) // 최대 projection 결과가 0보다 큰 경우
    {
        pLastCurrentKF = mpCurrentKF; // 이전 keyFrame을 현재 keyFrame으로 변경
        nNumCoincidences = nBestNumCoindicendes; // loop 감지수를 Best Candidates수로 변경
        pMatchedKF2 = pBestMatchedKF; // 매칭 keyFrame을 best 매칭 keyFrame으로 변경
        pMatchedKF2->SetNotErase(); // 매칭 keyFrame을 지우지 못하게 설정
        g2oScw = g2oBestScw; // world 변환 갱신
        vpMPs = vpBestMapPoints; // MapPoint 배열을 best MapPoint 배열로 변경
        vpMatchedMPs = vpBestMatchedMapPoints; // 매칭 MapPoint를 best 매칭 MapPoint로 변경

        return nNumCoincidences >= 3; // loop 감지수가 3 이상이면 감지 성공
    }
    else
    {
        int maxStage = -1;
        int maxMatched;
        for(int i=0; i<vnStage.size(); ++i)
        {
            if(vnStage[i] > maxStage)
            {
                maxStage = vnStage[i];
                maxMatched = vnMatchesStage[i];
            }
        }
    }
    return false; // 감지 실패
}

bool LoopClosing::DetectCommonRegionsFromLastKF(KeyFrame* pCurrentKF, KeyFrame* pMatchedKF, g2o::Sim3 &gScw, int &nNumProjMatches,
                                                std::vector<MapPoint*> &vpMPs, std::vector<MapPoint*> &vpMatchedMPs)
{
    set<MapPoint*> spAlreadyMatchedMPs(vpMatchedMPs.begin(), vpMatchedMPs.end());
    nNumProjMatches = FindMatchesByProjection(pCurrentKF, pMatchedKF, gScw, spAlreadyMatchedMPs, vpMPs, vpMatchedMPs); // 현재 keyFrame과 매칭 keyFrame간 keyPoint를 투영해 매칭 수를 구함

    int nProjMatches = 30;
    if(nNumProjMatches >= nProjMatches) // 매칭 수가 30 이상인 경우
    {
        return true; // Detect 성공
    }

    return false; // Detect 실패
}

int LoopClosing::FindMatchesByProjection(KeyFrame* pCurrentKF, KeyFrame* pMatchedKFw, g2o::Sim3 &g2oScw,
                                         set<MapPoint*> &spMatchedMPinOrigin, vector<MapPoint*> &vpMapPoints,
                                         vector<MapPoint*> &vpMatchedMapPoints)
{
    int nNumCovisibles = 10; // Covisibility Graph를 통해 구할 이웃 KeyFrame 수 초기화
    vector<KeyFrame*> vpCovKFm = pMatchedKFw->GetBestCovisibilityKeyFrames(nNumCovisibles); // 매칭 keyFrame의 Covisiblity Graph를 통해 이웃 KeyFrame 검색
    int nInitialCov = vpCovKFm.size(); // 이웃 KeyFrame의 수를 초기 이웃 수로 초기화
    vpCovKFm.push_back(pMatchedKFw); // 이웃 KeyFrame 배열에 매칭 KeyFrame을 삽입
    set<KeyFrame*> spCheckKFs(vpCovKFm.begin(), vpCovKFm.end()); // keyFrame check 배열에 이웃 keyFrame 배열에 있는 keyFrame을 삽입
    set<KeyFrame*> spCurrentCovisbles = pCurrentKF->GetConnectedKeyFrames(); // 현재 KeyFrame에 연결된 KeyFrame을 가져옴
    if(nInitialCov < nNumCovisibles) // 이웃 KeyFrame의 수가 Covisibility Graph를 통해 구할 이웃 수보다 작은 경우
    {
        for(int i=0; i<nInitialCov; ++i) // 이웃 KeyFrame 수 만큼 loop
        {
            vector<KeyFrame*> vpKFs = vpCovKFm[i]->GetBestCovisibilityKeyFrames(nNumCovisibles); // 이웃 keyFrame의 Covisiblity Graph를 통해 이웃 KeyFrame 검색
            int nInserted = 0; // keyFrame 삽입 횟수 초기화
            int j = 0; // 이웃 KeyFrame을 순회할 index 초기화
            while(j < vpKFs.size() && nInserted < nNumCovisibles) // 반복 횟수가 이웃 KeyFrame의 이웃 KeyFrame의 배열 크기 보다 작고 keyFrame 삽입 횟수가 Covisibility Graph를 통해 구할 이웃 수보다 작은 경우 loop
            {
                // keyFrame check 배열에 이웃 keyFrame을 통해 구한 이웃이 존재하지 않고, 현재 KeyFrame에 연결된 keyFrame과 이웃 keyFrame을 통해 구한 이웃이 같지 않은 경우
                if(spCheckKFs.find(vpKFs[j]) == spCheckKFs.end() && spCurrentCovisbles.find(vpKFs[j]) == spCurrentCovisbles.end()) 
                {
                    spCheckKFs.insert(vpKFs[j]); // keyFrame check 배열에 keyFrame 삽입
                    ++nInserted; // 삽입 횟수 증가
                }
                ++j; // index 증가
            }
            vpCovKFm.insert(vpCovKFm.end(), vpKFs.begin(), vpKFs.end()); // 이웃 keyFrame 배열에 이웃 keyFrame의 Covisiblity Graph를 통해 구한 이웃 keyFrame을 삽입
        }
    }
    set<MapPoint*> spMapPoints;
    vpMapPoints.clear(); // MapPoint 배열 초기화
    vpMatchedMapPoints.clear(); // 매칭 MapPoint 배열 초기화
    for(KeyFrame* pKFi : vpCovKFm) // 이웃 keyFrame 배열 loop
    {
        for(MapPoint* pMPij : pKFi->GetMapPointMatches()) // 이웃 keyFrame의 매칭 MapPoint 배열을 가져오고 매칭 MapPoint 배열을 loop
        {
            if(!pMPij || pMPij->isBad()) // MapPoint가 존재하지 않거나 Bad 상태인 경우 무시
                continue;

            if(spMapPoints.find(pMPij) == spMapPoints.end()) // MapPoint배열에 MapPoint가 없는 경우.
            {
                spMapPoints.insert(pMPij); // 중복 방지를 위해 삽입
                vpMapPoints.push_back(pMPij); // MapPoint 배열에 MapPoint 삽입
            }
        }
    }

    Sophus::Sim3f mScw = Converter::toSophus(g2oScw);
    ORBmatcher matcher(0.9, true); // ORB Matcher 객체 생성

    vpMatchedMapPoints.resize(pCurrentKF->GetMapPointMatches().size(), static_cast<MapPoint*>(NULL)); // 매칭 MapPoint 배열 크기를 현재 KeyFrame의 매칭 MapPoint 배열 크기만큼 할당
    int num_matches = matcher.SearchByProjection(pCurrentKF, mScw, vpMapPoints, vpMatchedMapPoints, 3, 1.5); // 현재 KeyFrame과 변환된 pose를 통해 Projection을 진행, 매칭 Point의 수를 반환 받음

    return num_matches; // 매칭 MapPoint 수 반환
}

void LoopClosing::CorrectLoop()
{
    //cout << "Loop detected!" << endl;

    // Send a stop signal to Local Mapping
    // Avoid new keyframes are inserted while correcting the loop
    mpLocalMapper->RequestStop(); // LocalMapping 스레드에 중단 요청
    mpLocalMapper->EmptyQueue(); // LocalMapping 스레드의 KeyFrame 배열을 비움 // Proccess keyframes in the queue

    // If a Global Bundle Adjustment is running, abort it
    if(isRunningGBA()) // GlobalBA 스레드가 실행중인 경우
    {
        cout << "Stoping Global Bundle Adjustment...";
        unique_lock<mutex> lock(mMutexGBA);
        mbStopGBA = true; // Global BA의 stop flag를 설정

        mnFullBAIdx++; // Global BA의 인덱스를 증가

        if(mpThreadGBA) // GlobalBA 스레드 객체가 존재하는 경우
        {
            mpThreadGBA->detach(); // Global BA 스레드를 중지하고 할당된 메모리 해제
            delete mpThreadGBA; // 스레드 객체 제거
        }
        cout << "  Done!!" << endl;
    }

    // Wait until Local Mapping has effectively stopped
    while(!mpLocalMapper->isStopped()) // LocalMapping 스레드가 중지될때 까지 대기
    {
        usleep(1000);
    }

    // Ensure current keyframe is updated
    //cout << "Start updating connections" << endl;
    //assert(mpCurrentKF->GetMap()->CheckEssentialGraph());
    mpCurrentKF->UpdateConnections(); // 현재 KeyFrame의 Connection을 갱신
    //assert(mpCurrentKF->GetMap()->CheckEssentialGraph());

    // Retrive keyframes connected to the current keyframe and compute corrected Sim3 pose by propagation
    mvpCurrentConnectedKFs = mpCurrentKF->GetVectorCovisibleKeyFrames(); // 현재 KeyFrame의 Covisilblity Graph를 통해 이웃 keyFrame을 가져옴
    mvpCurrentConnectedKFs.push_back(mpCurrentKF); // 이웃 KeyFrame 배열에 현재 keyFrame을 추가

    //std::cout << "Loop: number of connected KFs -> " + to_string(mvpCurrentConnectedKFs.size()) << std::endl;

    KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;
    CorrectedSim3[mpCurrentKF]=mg2oLoopScw; // 현재 KeyFrame과 loop에 대한 변환을 삽입
    Sophus::SE3f Twc = mpCurrentKF->GetPoseInverse(); // 현재 KeyFrame의 pose의 역행렬을 가져옴
    Sophus::SE3f Tcw = mpCurrentKF->GetPose(); // 현재 KeyFrame의 pose를 가져옴
    g2o::Sim3 g2oScw(Tcw.unit_quaternion().cast<double>(),Tcw.translation().cast<double>(),1.0); // pose를 통해 공간 변환을 생성
    NonCorrectedSim3[mpCurrentKF]=g2oScw; // 현재 KeyFrame과 공간 변환을 삽입

    // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
    Sophus::SE3d correctedTcw(mg2oLoopScw.rotation(),mg2oLoopScw.translation() / mg2oLoopScw.scale()); // 회전 및 위치 정보를 통해 보정된 pose 객체 생성
    mpCurrentKF->SetPose(correctedTcw.cast<float>()); // 현재 KeyFrame의 pose로 설정

    Map* pLoopMap = mpCurrentKF->GetMap(); // 현재 KeyFrame의 Map을 가져옴

#ifdef REGISTER_TIMES
    /*KeyFrame* pKF = mpCurrentKF;
    int numKFinLoop = 0;
    while(pKF && pKF->mnId > mpLoopMatchedKF->mnId)
    {
        pKF = pKF->GetParent();
        numKFinLoop += 1;
    }
    vnLoopKFs.push_back(numKFinLoop);*/

    std::chrono::steady_clock::time_point time_StartFusion = std::chrono::steady_clock::now();
#endif

    {
        // Get Map Mutex
        unique_lock<mutex> lock(pLoopMap->mMutexMapUpdate);

        const bool bImuInit = pLoopMap->isImuInitialized(); // Map에 IMU가 초기화된 경우 true 저장

        for(vector<KeyFrame*>::iterator vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++) // 이웃 KeyFrame에 대해 loop
        {
            KeyFrame* pKFi = *vit;

            if(pKFi!=mpCurrentKF) // 이웃 KeyFrame이 현재 KeyFrame과 다르다면
            {
                Sophus::SE3f Tiw = pKFi->GetPose(); // keyFrame의 pose를 가져옴
                Sophus::SE3d Tic = (Tiw * Twc).cast<double>(); // 이웃의 위치를 현재 KeyFrame의 위치로 변환
                g2o::Sim3 g2oSic(Tic.unit_quaternion(),Tic.translation(),1.0); // 변환된 위치를 사용하여 공간 변환을 생성
                g2o::Sim3 g2oCorrectedSiw = g2oSic*mg2oLoopScw; // 이웃 KeyFrame의 위치를 보정
                //Pose corrected with the Sim3 of the loop closure
                CorrectedSim3[pKFi]=g2oCorrectedSiw; // 이웃 KeyFrame과 공간 변환을 삽입

                // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
                Sophus::SE3d correctedTiw(g2oCorrectedSiw.rotation(),g2oCorrectedSiw.translation() / g2oCorrectedSiw.scale()); // 회전 및 스케일된 이동 정보를 통해 이웃 KeyFrame의 pose 생성
                pKFi->SetPose(correctedTiw.cast<float>()); // 이웃 KeyFrame의 pose 설정

                //Pose without correction
                g2o::Sim3 g2oSiw(Tiw.unit_quaternion().cast<double>(),Tiw.translation().cast<double>(),1.0); // 보정되지 않은 공간 변환을 생성
                NonCorrectedSim3[pKFi]=g2oSiw; // 이웃 KeyFrame과 보정되지 않은 공간 변환을 삽입
            }  
        }

        // Correct all MapPoints obsrved by current keyframe and neighbors, so that they align with the other side of the loop
        for(KeyFrameAndPose::iterator mit=CorrectedSim3.begin(), mend=CorrectedSim3.end(); mit!=mend; mit++) // 이웃 KeyFrame과 공간 변환 객체에 대해 loop
        {
            KeyFrame* pKFi = mit->first; // 이웃 KeyFrame
            g2o::Sim3 g2oCorrectedSiw = mit->second; // 이웃 KeyFrame에 대한 공간 변환
            g2o::Sim3 g2oCorrectedSwi = g2oCorrectedSiw.inverse(); // 역행렬 생성

            g2o::Sim3 g2oSiw =NonCorrectedSim3[pKFi]; // 해당 KeyFrame에 대한 보정되지 않은 공간 변환을 얻는다.

            // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
            /*Sophus::SE3d correctedTiw(g2oCorrectedSiw.rotation(),g2oCorrectedSiw.translation() / g2oCorrectedSiw.scale());
            pKFi->SetPose(correctedTiw.cast<float>());*/

            vector<MapPoint*> vpMPsi = pKFi->GetMapPointMatches(); // KeyFrame에서 매칭 MapPoint를 얻음
            for(size_t iMP=0, endMPi = vpMPsi.size(); iMP<endMPi; iMP++) // 매칭 MapPoint 배열 loop
            {
                MapPoint* pMPi = vpMPsi[iMP];
                if(!pMPi) // MapPoint가 존재하지 않으면 무시
                    continue;
                if(pMPi->isBad()) // Bad 상태이면 무시
                    continue;
                if(pMPi->mnCorrectedByKF==mpCurrentKF->mnId) // MapPoint의 Corrected KeyFrame ID가 현재 KeyFrame의 ID와 같으면 무시
                    continue;

                // Project with non-corrected pose and project back with corrected pose
                Eigen::Vector3d P3Dw = pMPi->GetWorldPos().cast<double>(); // MapPoint world pose를 가져옵
                Eigen::Vector3d eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oSiw.map(P3Dw)); // 보정된 공간 변환을 사용하여 MapPoint pose 변경

                pMPi->SetWorldPos(eigCorrectedP3Dw.cast<float>()); // 변경된 pose를 MapPoint의 world pose로 갱신
                pMPi->mnCorrectedByKF = mpCurrentKF->mnId; // MapPoint의 Corrected KeyFrame ID를 현재 KeyFrame의 ID로 초기화
                pMPi->mnCorrectedReference = pKFi->mnId; // MapPoint의 레퍼런스 Corrected KeyFrame ID를 이웃 KeyFrame의 ID로 초기화
                pMPi->UpdateNormalAndDepth(); // MapPoint의 Normal과 Depth를 갱신
            }

            // Correct velocity according to orientation correction
            if(bImuInit) // IMU가 초기화된 경우
            {
                Eigen::Quaternionf Rcor = (g2oCorrectedSiw.rotation().inverse()*g2oSiw.rotation()).cast<float>(); // 회전 정보를 얻음
                pKFi->SetVelocity(Rcor*pKFi->GetVelocity()); // 보정된 회전에 따라 이웃 KeyFrame의 속도를 갱신
            }

            // Make sure connections are updated
            pKFi->UpdateConnections(); // 이웃 keyFrame의 Connection을 갱신
        }
        // TODO Check this index increasement
        mpAtlas->GetCurrentMap()->IncreaseChangeIndex(); // Atlas의 현재 Map에 index를 증가


        // Start Loop Fusion
        // Update matched map points and replace if duplicated
        for(size_t i=0; i<mvpLoopMatchedMPs.size(); i++) // Loop 매칭 MapPoint에 대해 loop
        {
            if(mvpLoopMatchedMPs[i]) // MapPoint가 존재하는 경우
            {
                MapPoint* pLoopMP = mvpLoopMatchedMPs[i]; // 매칭 MapPoint
                MapPoint* pCurMP = mpCurrentKF->GetMapPoint(i); // 현재 KeyFrame에서 ID가 같은 MapPoint를 얻음
                if(pCurMP) // 현재 KeyFrame의 MapPoint가 존재하는 경우
                    pCurMP->Replace(pLoopMP); // 현재 KeyFrame의 MapPoint에 대해 매칭 MapPoint로 replace 진행
                else
                {
                    mpCurrentKF->AddMapPoint(pLoopMP,i); // 현재 KeyFrame에 매칭 MapPoint를 추가
                    pLoopMP->AddObservation(mpCurrentKF,i); // 매칭 MapPoint에 현재 KeyFrame을 추적 대상으로 추가
                    pLoopMP->ComputeDistinctiveDescriptors(); // 매칭 MapPoint에 디스크립터 연산 진행
                }
            }
        }
        //cout << "LC: end replacing duplicated" << endl;
    }

    // Project MapPoints observed in the neighborhood of the loop keyframe
    // into the current keyframe and neighbors using corrected poses.
    // Fuse duplications.
    SearchAndFuse(CorrectedSim3, mvpLoopMapPoints); // KeyFrame, 공간변환, MapPoint를 통해 MapPoint를 Fusion하고 Replace

    // After the MapPoint fusion, new links in the covisibility graph will appear attaching both sides of the loop
    map<KeyFrame*, set<KeyFrame*> > LoopConnections;

    for(vector<KeyFrame*>::iterator vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++) // 이웃 KeyFrame 배열 loop
    {
        KeyFrame* pKFi = *vit;
        vector<KeyFrame*> vpPreviousNeighbors = pKFi->GetVectorCovisibleKeyFrames(); // KeyFrame의 Covisiblity Graph를 통해 이웃 KeyFrame을 얻음

        // Update connections. Detect new links.
        pKFi->UpdateConnections(); // KeyFrame의 Connection을 갱신
        LoopConnections[pKFi]=pKFi->GetConnectedKeyFrames(); // KeyFrame의 Connection KeyFrame을 저장
        for(vector<KeyFrame*>::iterator vit_prev=vpPreviousNeighbors.begin(), vend_prev=vpPreviousNeighbors.end(); vit_prev!=vend_prev; vit_prev++)
        {
            LoopConnections[pKFi].erase(*vit_prev); // Connection keyFrame에 이웃의 이웃 KeyFrame이 존재하는 경우 제거
        }
        for(vector<KeyFrame*>::iterator vit2=mvpCurrentConnectedKFs.begin(), vend2=mvpCurrentConnectedKFs.end(); vit2!=vend2; vit2++)
        {
            LoopConnections[pKFi].erase(*vit2); // Connection keyFrame에 이웃 KeyFrame이 존재하는 경우 제거
        }
    }

    // Optimize graph
    bool bFixedScale = mbFixScale; // scale 고정 flag 초기화
    // TODO CHECK; Solo para el monocular inertial
    if(mpTracker->mSensor==System::IMU_MONOCULAR && !mpCurrentKF->GetMap()->GetIniertialBA2()) // Monocular이거나 IniertialBA2가 이루어진 경우
        bFixedScale=false; // scale 고정하지 않는다고 설정

#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_EndFusion = std::chrono::steady_clock::now();

        double timeFusion = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndFusion - time_StartFusion).count();
        vdLoopFusion_ms.push_back(timeFusion);
#endif
    //cout << "Optimize essential graph" << endl;
    if(pLoopMap->IsInertial() && pLoopMap->isImuInitialized()) // 현재 KeyFrame의 Map이 IMU를 사용하고 IMU가 초기화된 경우
    {
        Optimizer::OptimizeEssentialGraph4DoF(pLoopMap, mpLoopMatchedKF, mpCurrentKF, NonCorrectedSim3, CorrectedSim3, LoopConnections); // Essential Graph를 최적화 (이때 4DOF를 통해 최적화 진행)
    }
    else
    {
        //cout << "Loop -> Scale correction: " << mg2oLoopScw.scale() << endl;
        Optimizer::OptimizeEssentialGraph(pLoopMap, mpLoopMatchedKF, mpCurrentKF, NonCorrectedSim3, CorrectedSim3, LoopConnections, bFixedScale); // essential graph 최적화
    }
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndOpt = std::chrono::steady_clock::now();

    double timeOptEss = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndOpt - time_EndFusion).count();
    vdLoopOptEss_ms.push_back(timeOptEss);
#endif

    mpAtlas->InformNewBigChange(); // Atlas의 Map에 큰 변화가 생겼다고 알림

    // Add loop edge
    mpLoopMatchedKF->AddLoopEdge(mpCurrentKF); // Loop 매칭 KeyFrame에 loop의 Edge를 현재 KeyFrame으로 초기화
    mpCurrentKF->AddLoopEdge(mpLoopMatchedKF); // 현재 KeyFrame에 loop의 Edge를 Loop 매칭 KeyFrame으로 초기화

    // Launch a new thread to perform Global Bundle Adjustment (Only if few keyframes, if not it would take too much time)
    // 현재 KeyFrame의 Map에 IMU가 초기화 되지 않았거나 Map에 KeyFrame이 200개 보다 적고 Atlas에 Map이 1개 있는 경우
    if(!pLoopMap->isImuInitialized() || (pLoopMap->KeyFramesInMap()<200 && mpAtlas->CountMaps()==1)) 
    {
        mbRunningGBA = true; // Global BA가 동작 중이라고 flag 설정
        mbFinishedGBA = false; // Global BA가 끝나지 않았다고 flag 설정
        mbStopGBA = false; // Global BA가 중지 되지 않았다고 flag 설정
        mnCorrectionGBA = mnNumCorrection; // Loop 수집 수를 Global BA 횟수로 초기화

        mpThreadGBA = new thread(&LoopClosing::RunGlobalBundleAdjustment, this, pLoopMap, mpCurrentKF->mnId); // Global BA 스레드 생성
    }

    // Loop closed. Release Local Mapping.
    mpLocalMapper->Release(); // Local Mapping 스레드를 꺠움 

    mLastLoopKFid = mpCurrentKF->mnId; // 이전 Loop KeyFrame ID를 현재 KeyFrame ID로 초기화 //TODO old varible, it is not use in the new algorithm
}

void LoopClosing::MergeLocal()
{
    int numTemporalKFs = 25; //Temporal KFs in the local window if the map is inertial.

    //Relationship to rebuild the essential graph, it is used two times, first in the local window and later in the rest of the map
    KeyFrame* pNewChild;
    KeyFrame* pNewParent;

    vector<KeyFrame*> vpLocalCurrentWindowKFs;
    vector<KeyFrame*> vpMergeConnectedKFs;

    // Flag that is true only when we stopped a running BA, in this case we need relaunch at the end of the merge
    bool bRelaunchBA = false;

    //Verbose::PrintMess("MERGE-VISUAL: Check Full Bundle Adjustment", Verbose::VERBOSITY_DEBUG);
    // If a Global Bundle Adjustment is running, abort it
    if(isRunningGBA())
    {
        unique_lock<mutex> lock(mMutexGBA);
        mbStopGBA = true;

        mnFullBAIdx++;

        if(mpThreadGBA)
        {
            mpThreadGBA->detach();
            delete mpThreadGBA;
        }
        bRelaunchBA = true;
    }

    //Verbose::PrintMess("MERGE-VISUAL: Request Stop Local Mapping", Verbose::VERBOSITY_DEBUG);
    //cout << "Request Stop Local Mapping" << endl;
    mpLocalMapper->RequestStop();
    // Wait until Local Mapping has effectively stopped
    while(!mpLocalMapper->isStopped())
    {
        usleep(1000);
    }
    //cout << "Local Map stopped" << endl;

    mpLocalMapper->EmptyQueue();

    // Merge map will become in the new active map with the local window of KFs and MPs from the current map.
    // Later, the elements of the current map will be transform to the new active map reference, in order to keep real time tracking
    Map* pCurrentMap = mpCurrentKF->GetMap();
    Map* pMergeMap = mpMergeMatchedKF->GetMap();

    //std::cout << "Merge local, Active map: " << pCurrentMap->GetId() << std::endl;
    //std::cout << "Merge local, Non-Active map: " << pMergeMap->GetId() << std::endl;

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartMerge = std::chrono::steady_clock::now();
#endif

    // Ensure current keyframe is updated
    mpCurrentKF->UpdateConnections();

    //Get the current KF and its neighbors(visual->covisibles; inertial->temporal+covisibles)
    set<KeyFrame*> spLocalWindowKFs;
    //Get MPs in the welding area from the current map
    set<MapPoint*> spLocalWindowMPs;
    if(pCurrentMap->IsInertial() && pMergeMap->IsInertial()) //TODO Check the correct initialization
    {
        KeyFrame* pKFi = mpCurrentKF;
        int nInserted = 0;
        while(pKFi && nInserted < numTemporalKFs)
        {
            spLocalWindowKFs.insert(pKFi);
            pKFi = mpCurrentKF->mPrevKF;
            nInserted++;

            set<MapPoint*> spMPi = pKFi->GetMapPoints();
            spLocalWindowMPs.insert(spMPi.begin(), spMPi.end());
        }

        pKFi = mpCurrentKF->mNextKF;
        while(pKFi)
        {
            spLocalWindowKFs.insert(pKFi);

            set<MapPoint*> spMPi = pKFi->GetMapPoints();
            spLocalWindowMPs.insert(spMPi.begin(), spMPi.end());

            pKFi = mpCurrentKF->mNextKF;
        }
    }
    else
    {
        spLocalWindowKFs.insert(mpCurrentKF);
    }

    vector<KeyFrame*> vpCovisibleKFs = mpCurrentKF->GetBestCovisibilityKeyFrames(numTemporalKFs);
    spLocalWindowKFs.insert(vpCovisibleKFs.begin(), vpCovisibleKFs.end());
    spLocalWindowKFs.insert(mpCurrentKF);
    const int nMaxTries = 5;
    int nNumTries = 0;
    while(spLocalWindowKFs.size() < numTemporalKFs && nNumTries < nMaxTries)
    {
        vector<KeyFrame*> vpNewCovKFs;
        vpNewCovKFs.empty();
        for(KeyFrame* pKFi : spLocalWindowKFs)
        {
            vector<KeyFrame*> vpKFiCov = pKFi->GetBestCovisibilityKeyFrames(numTemporalKFs/2);
            for(KeyFrame* pKFcov : vpKFiCov)
            {
                if(pKFcov && !pKFcov->isBad() && spLocalWindowKFs.find(pKFcov) == spLocalWindowKFs.end())
                {
                    vpNewCovKFs.push_back(pKFcov);
                }

            }
        }

        spLocalWindowKFs.insert(vpNewCovKFs.begin(), vpNewCovKFs.end());
        nNumTries++;
    }

    for(KeyFrame* pKFi : spLocalWindowKFs)
    {
        if(!pKFi || pKFi->isBad())
            continue;

        set<MapPoint*> spMPs = pKFi->GetMapPoints();
        spLocalWindowMPs.insert(spMPs.begin(), spMPs.end());
    }

    //std::cout << "[Merge]: Ma = " << to_string(pCurrentMap->GetId()) << "; #KFs = " << to_string(spLocalWindowKFs.size()) << "; #MPs = " << to_string(spLocalWindowMPs.size()) << std::endl;

    set<KeyFrame*> spMergeConnectedKFs;
    if(pCurrentMap->IsInertial() && pMergeMap->IsInertial()) //TODO Check the correct initialization
    {
        KeyFrame* pKFi = mpMergeMatchedKF;
        int nInserted = 0;
        while(pKFi && nInserted < numTemporalKFs/2)
        {
            spMergeConnectedKFs.insert(pKFi);
            pKFi = mpCurrentKF->mPrevKF;
            nInserted++;
        }

        pKFi = mpMergeMatchedKF->mNextKF;
        while(pKFi && nInserted < numTemporalKFs)
        {
            spMergeConnectedKFs.insert(pKFi);
            pKFi = mpCurrentKF->mNextKF;
        }
    }
    else
    {
        spMergeConnectedKFs.insert(mpMergeMatchedKF);
    }
    vpCovisibleKFs = mpMergeMatchedKF->GetBestCovisibilityKeyFrames(numTemporalKFs);
    spMergeConnectedKFs.insert(vpCovisibleKFs.begin(), vpCovisibleKFs.end());
    spMergeConnectedKFs.insert(mpMergeMatchedKF);
    nNumTries = 0;
    while(spMergeConnectedKFs.size() < numTemporalKFs && nNumTries < nMaxTries)
    {
        vector<KeyFrame*> vpNewCovKFs;
        for(KeyFrame* pKFi : spMergeConnectedKFs)
        {
            vector<KeyFrame*> vpKFiCov = pKFi->GetBestCovisibilityKeyFrames(numTemporalKFs/2);
            for(KeyFrame* pKFcov : vpKFiCov)
            {
                if(pKFcov && !pKFcov->isBad() && spMergeConnectedKFs.find(pKFcov) == spMergeConnectedKFs.end())
                {
                    vpNewCovKFs.push_back(pKFcov);
                }

            }
        }

        spMergeConnectedKFs.insert(vpNewCovKFs.begin(), vpNewCovKFs.end());
        nNumTries++;
    }

    set<MapPoint*> spMapPointMerge;
    for(KeyFrame* pKFi : spMergeConnectedKFs)
    {
        set<MapPoint*> vpMPs = pKFi->GetMapPoints();
        spMapPointMerge.insert(vpMPs.begin(),vpMPs.end());
    }

    vector<MapPoint*> vpCheckFuseMapPoint;
    vpCheckFuseMapPoint.reserve(spMapPointMerge.size());
    std::copy(spMapPointMerge.begin(), spMapPointMerge.end(), std::back_inserter(vpCheckFuseMapPoint));

    //std::cout << "[Merge]: Mm = " << to_string(pMergeMap->GetId()) << "; #KFs = " << to_string(spMergeConnectedKFs.size()) << "; #MPs = " << to_string(spMapPointMerge.size()) << std::endl;


    //
    Sophus::SE3d Twc = mpCurrentKF->GetPoseInverse().cast<double>();
    g2o::Sim3 g2oNonCorrectedSwc(Twc.unit_quaternion(),Twc.translation(),1.0);
    g2o::Sim3 g2oNonCorrectedScw = g2oNonCorrectedSwc.inverse();
    g2o::Sim3 g2oCorrectedScw = mg2oMergeScw; //TODO Check the transformation

    KeyFrameAndPose vCorrectedSim3, vNonCorrectedSim3;
    vCorrectedSim3[mpCurrentKF]=g2oCorrectedScw;
    vNonCorrectedSim3[mpCurrentKF]=g2oNonCorrectedScw;


#ifdef REGISTER_TIMES
    vnMergeKFs.push_back(spLocalWindowKFs.size() + spMergeConnectedKFs.size());
    vnMergeMPs.push_back(spLocalWindowMPs.size() + spMapPointMerge.size());
#endif
    for(KeyFrame* pKFi : spLocalWindowKFs)
    {
        if(!pKFi || pKFi->isBad())
        {
            Verbose::PrintMess("Bad KF in correction", Verbose::VERBOSITY_DEBUG);
            continue;
        }

        if(pKFi->GetMap() != pCurrentMap)
            Verbose::PrintMess("Other map KF, this should't happen", Verbose::VERBOSITY_DEBUG);

        g2o::Sim3 g2oCorrectedSiw;

        if(pKFi!=mpCurrentKF)
        {
            Sophus::SE3d Tiw = (pKFi->GetPose()).cast<double>();
            g2o::Sim3 g2oSiw(Tiw.unit_quaternion(),Tiw.translation(),1.0);
            //Pose without correction
            vNonCorrectedSim3[pKFi]=g2oSiw;

            Sophus::SE3d Tic = Tiw*Twc;
            g2o::Sim3 g2oSic(Tic.unit_quaternion(),Tic.translation(),1.0);
            g2oCorrectedSiw = g2oSic*mg2oMergeScw;
            vCorrectedSim3[pKFi]=g2oCorrectedSiw;
        }
        else
        {
            g2oCorrectedSiw = g2oCorrectedScw;
        }
        pKFi->mTcwMerge  = pKFi->GetPose();

        // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
        double s = g2oCorrectedSiw.scale();
        pKFi->mfScale = s;
        Sophus::SE3d correctedTiw(g2oCorrectedSiw.rotation(), g2oCorrectedSiw.translation() / s);

        pKFi->mTcwMerge = correctedTiw.cast<float>();

        if(pCurrentMap->isImuInitialized())
        {
            Eigen::Quaternionf Rcor = (g2oCorrectedSiw.rotation().inverse() * vNonCorrectedSim3[pKFi].rotation()).cast<float>();
            pKFi->mVwbMerge = Rcor * pKFi->GetVelocity();
        }

        //TODO DEBUG to know which are the KFs that had been moved to the other map
    }

    int numPointsWithCorrection = 0;

    //for(MapPoint* pMPi : spLocalWindowMPs)
    set<MapPoint*>::iterator itMP = spLocalWindowMPs.begin();
    while(itMP != spLocalWindowMPs.end())
    {
        MapPoint* pMPi = *itMP;
        if(!pMPi || pMPi->isBad())
        {
            itMP = spLocalWindowMPs.erase(itMP);
            continue;
        }

        KeyFrame* pKFref = pMPi->GetReferenceKeyFrame();
        if(vCorrectedSim3.find(pKFref) == vCorrectedSim3.end())
        {
            itMP = spLocalWindowMPs.erase(itMP);
            numPointsWithCorrection++;
            continue;
        }
        g2o::Sim3 g2oCorrectedSwi = vCorrectedSim3[pKFref].inverse();
        g2o::Sim3 g2oNonCorrectedSiw = vNonCorrectedSim3[pKFref];

        // Project with non-corrected pose and project back with corrected pose
        Eigen::Vector3d P3Dw = pMPi->GetWorldPos().cast<double>();
        Eigen::Vector3d eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oNonCorrectedSiw.map(P3Dw));
        Eigen::Quaterniond Rcor = g2oCorrectedSwi.rotation() * g2oNonCorrectedSiw.rotation();

        pMPi->mPosMerge = eigCorrectedP3Dw.cast<float>();
        pMPi->mNormalVectorMerge = Rcor.cast<float>() * pMPi->GetNormal();

        itMP++;
    }
    /*if(numPointsWithCorrection>0)
    {
        std::cout << "[Merge]: " << std::to_string(numPointsWithCorrection) << " points removed from Ma due to its reference KF is not in welding area" << std::endl;
        std::cout << "[Merge]: Ma has " << std::to_string(spLocalWindowMPs.size()) << " points" << std::endl;
    }*/

    {
        unique_lock<mutex> currentLock(pCurrentMap->mMutexMapUpdate); // We update the current map with the Merge information
        unique_lock<mutex> mergeLock(pMergeMap->mMutexMapUpdate); // We remove the Kfs and MPs in the merged area from the old map

        //std::cout << "Merge local window: " << spLocalWindowKFs.size() << std::endl;
        //std::cout << "[Merge]: init merging maps " << std::endl;
        for(KeyFrame* pKFi : spLocalWindowKFs)
        {
            if(!pKFi || pKFi->isBad())
            {
                //std::cout << "Bad KF in correction" << std::endl;
                continue;
            }

            //std::cout << "KF id: " << pKFi->mnId << std::endl;

            pKFi->mTcwBefMerge = pKFi->GetPose();
            pKFi->mTwcBefMerge = pKFi->GetPoseInverse();
            pKFi->SetPose(pKFi->mTcwMerge);

            // Make sure connections are updated
            pKFi->UpdateMap(pMergeMap);
            pKFi->mnMergeCorrectedForKF = mpCurrentKF->mnId;
            pMergeMap->AddKeyFrame(pKFi);
            pCurrentMap->EraseKeyFrame(pKFi);

            if(pCurrentMap->isImuInitialized())
            {
                pKFi->SetVelocity(pKFi->mVwbMerge);
            }
        }

        for(MapPoint* pMPi : spLocalWindowMPs)
        {
            if(!pMPi || pMPi->isBad())
                continue;

            pMPi->SetWorldPos(pMPi->mPosMerge);
            pMPi->SetNormalVector(pMPi->mNormalVectorMerge);
            pMPi->UpdateMap(pMergeMap);
            pMergeMap->AddMapPoint(pMPi);
            pCurrentMap->EraseMapPoint(pMPi);
        }

        mpAtlas->ChangeMap(pMergeMap);
        mpAtlas->SetMapBad(pCurrentMap);
        pMergeMap->IncreaseChangeIndex();
        //TODO for debug
        pMergeMap->ChangeId(pCurrentMap->GetId());

        //std::cout << "[Merge]: merging maps finished" << std::endl;
    }

    //Rebuild the essential graph in the local window
    pCurrentMap->GetOriginKF()->SetFirstConnection(false);
    pNewChild = mpCurrentKF->GetParent(); // Old parent, it will be the new child of this KF
    pNewParent = mpCurrentKF; // Old child, now it will be the parent of its own parent(we need eliminate this KF from children list in its old parent)
    mpCurrentKF->ChangeParent(mpMergeMatchedKF);
    while(pNewChild)
    {
        pNewChild->EraseChild(pNewParent); // We remove the relation between the old parent and the new for avoid loop
        KeyFrame * pOldParent = pNewChild->GetParent();

        pNewChild->ChangeParent(pNewParent);

        pNewParent = pNewChild;
        pNewChild = pOldParent;

    }

    //Update the connections between the local window
    mpMergeMatchedKF->UpdateConnections();

    vpMergeConnectedKFs = mpMergeMatchedKF->GetVectorCovisibleKeyFrames();
    vpMergeConnectedKFs.push_back(mpMergeMatchedKF);
    //vpCheckFuseMapPoint.reserve(spMapPointMerge.size());
    //std::copy(spMapPointMerge.begin(), spMapPointMerge.end(), std::back_inserter(vpCheckFuseMapPoint));

    // Project MapPoints observed in the neighborhood of the merge keyframe
    // into the current keyframe and neighbors using corrected poses.
    // Fuse duplications.
    //std::cout << "[Merge]: start fuse points" << std::endl;
    SearchAndFuse(vCorrectedSim3, vpCheckFuseMapPoint);
    //std::cout << "[Merge]: fuse points finished" << std::endl;

    // Update connectivity
    for(KeyFrame* pKFi : spLocalWindowKFs)
    {
        if(!pKFi || pKFi->isBad())
            continue;

        pKFi->UpdateConnections();
    }
    for(KeyFrame* pKFi : spMergeConnectedKFs)
    {
        if(!pKFi || pKFi->isBad())
            continue;

        pKFi->UpdateConnections();
    }

    //std::cout << "[Merge]: Start welding bundle adjustment" << std::endl;

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartWeldingBA = std::chrono::steady_clock::now();

    double timeMergeMaps = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_StartWeldingBA - time_StartMerge).count();
    vdMergeMaps_ms.push_back(timeMergeMaps);
#endif

    bool bStop = false;
    vpLocalCurrentWindowKFs.clear();
    vpMergeConnectedKFs.clear();
    std::copy(spLocalWindowKFs.begin(), spLocalWindowKFs.end(), std::back_inserter(vpLocalCurrentWindowKFs));
    std::copy(spMergeConnectedKFs.begin(), spMergeConnectedKFs.end(), std::back_inserter(vpMergeConnectedKFs));
    if (mpTracker->mSensor==System::IMU_MONOCULAR || mpTracker->mSensor==System::IMU_STEREO || mpTracker->mSensor==System::IMU_RGBD)
    {
        Optimizer::MergeInertialBA(mpCurrentKF,mpMergeMatchedKF,&bStop, pCurrentMap,vCorrectedSim3);
    }
    else
    {
        Optimizer::LocalBundleAdjustment(mpCurrentKF, vpLocalCurrentWindowKFs, vpMergeConnectedKFs,&bStop);
    }

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndWeldingBA = std::chrono::steady_clock::now();

    double timeWeldingBA = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndWeldingBA - time_StartWeldingBA).count();
    vdWeldingBA_ms.push_back(timeWeldingBA);
#endif
    //std::cout << "[Merge]: Welding bundle adjustment finished" << std::endl;

    // Loop closed. Release Local Mapping.
    mpLocalMapper->Release();

    //Update the non critical area from the current map to the merged map
    vector<KeyFrame*> vpCurrentMapKFs = pCurrentMap->GetAllKeyFrames();
    vector<MapPoint*> vpCurrentMapMPs = pCurrentMap->GetAllMapPoints();

    if(vpCurrentMapKFs.size() == 0){}
    else {
        if(mpTracker->mSensor == System::MONOCULAR)
        {
            unique_lock<mutex> currentLock(pCurrentMap->mMutexMapUpdate); // We update the current map with the Merge information

            for(KeyFrame* pKFi : vpCurrentMapKFs)
            {
                if(!pKFi || pKFi->isBad() || pKFi->GetMap() != pCurrentMap)
                {
                    continue;
                }

                g2o::Sim3 g2oCorrectedSiw;

                Sophus::SE3d Tiw = (pKFi->GetPose()).cast<double>();
                g2o::Sim3 g2oSiw(Tiw.unit_quaternion(),Tiw.translation(),1.0);
                //Pose without correction
                vNonCorrectedSim3[pKFi]=g2oSiw;

                Sophus::SE3d Tic = Tiw*Twc;
                g2o::Sim3 g2oSim(Tic.unit_quaternion(),Tic.translation(),1.0);
                g2oCorrectedSiw = g2oSim*mg2oMergeScw;
                vCorrectedSim3[pKFi]=g2oCorrectedSiw;

                // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
                double s = g2oCorrectedSiw.scale();

                pKFi->mfScale = s;

                Sophus::SE3d correctedTiw(g2oCorrectedSiw.rotation(),g2oCorrectedSiw.translation() / s);

                pKFi->mTcwBefMerge = pKFi->GetPose();
                pKFi->mTwcBefMerge = pKFi->GetPoseInverse();

                pKFi->SetPose(correctedTiw.cast<float>());

                if(pCurrentMap->isImuInitialized())
                {
                    Eigen::Quaternionf Rcor = (g2oCorrectedSiw.rotation().inverse() * vNonCorrectedSim3[pKFi].rotation()).cast<float>();
                    pKFi->SetVelocity(Rcor * pKFi->GetVelocity()); // TODO: should add here scale s
                }

            }
            for(MapPoint* pMPi : vpCurrentMapMPs)
            {
                if(!pMPi || pMPi->isBad()|| pMPi->GetMap() != pCurrentMap)
                    continue;

                KeyFrame* pKFref = pMPi->GetReferenceKeyFrame();
                g2o::Sim3 g2oCorrectedSwi = vCorrectedSim3[pKFref].inverse();
                g2o::Sim3 g2oNonCorrectedSiw = vNonCorrectedSim3[pKFref];

                // Project with non-corrected pose and project back with corrected pose
                Eigen::Vector3d P3Dw = pMPi->GetWorldPos().cast<double>();
                Eigen::Vector3d eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oNonCorrectedSiw.map(P3Dw));
                pMPi->SetWorldPos(eigCorrectedP3Dw.cast<float>());

                pMPi->UpdateNormalAndDepth();
            }
        }

        mpLocalMapper->RequestStop();
        // Wait until Local Mapping has effectively stopped
        while(!mpLocalMapper->isStopped())
        {
            usleep(1000);
        }

        // Optimize graph (and update the loop position for each element form the begining to the end)
        if(mpTracker->mSensor != System::MONOCULAR)
        {
            Optimizer::OptimizeEssentialGraph(mpCurrentKF, vpMergeConnectedKFs, vpLocalCurrentWindowKFs, vpCurrentMapKFs, vpCurrentMapMPs);
        }


        {
            // Get Merge Map Mutex
            unique_lock<mutex> currentLock(pCurrentMap->mMutexMapUpdate); // We update the current map with the Merge information
            unique_lock<mutex> mergeLock(pMergeMap->mMutexMapUpdate); // We remove the Kfs and MPs in the merged area from the old map

            //std::cout << "Merge outside KFs: " << vpCurrentMapKFs.size() << std::endl;
            for(KeyFrame* pKFi : vpCurrentMapKFs)
            {
                if(!pKFi || pKFi->isBad() || pKFi->GetMap() != pCurrentMap)
                {
                    continue;
                }
                //std::cout << "KF id: " << pKFi->mnId << std::endl;

                // Make sure connections are updated
                pKFi->UpdateMap(pMergeMap);
                pMergeMap->AddKeyFrame(pKFi);
                pCurrentMap->EraseKeyFrame(pKFi);
            }

            for(MapPoint* pMPi : vpCurrentMapMPs)
            {
                if(!pMPi || pMPi->isBad())
                    continue;

                pMPi->UpdateMap(pMergeMap);
                pMergeMap->AddMapPoint(pMPi);
                pCurrentMap->EraseMapPoint(pMPi);
            }
        }
    }

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndOptEss = std::chrono::steady_clock::now();

    double timeOptEss = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndOptEss - time_EndWeldingBA).count();
    vdMergeOptEss_ms.push_back(timeOptEss);
#endif


    mpLocalMapper->Release();

    if(bRelaunchBA && (!pCurrentMap->isImuInitialized() || (pCurrentMap->KeyFramesInMap()<200 && mpAtlas->CountMaps()==1)))
    {
        // Launch a new thread to perform Global Bundle Adjustment
        mbRunningGBA = true;
        mbFinishedGBA = false;
        mbStopGBA = false;
        mpThreadGBA = new thread(&LoopClosing::RunGlobalBundleAdjustment,this, pMergeMap, mpCurrentKF->mnId);
    }

    mpMergeMatchedKF->AddMergeEdge(mpCurrentKF);
    mpCurrentKF->AddMergeEdge(mpMergeMatchedKF);

    pCurrentMap->IncreaseChangeIndex();
    pMergeMap->IncreaseChangeIndex();

    mpAtlas->RemoveBadMaps();

}


void LoopClosing::MergeLocal2()
{
    //cout << "Merge detected!!!!" << endl;

    // 임시 키프레임의 수 (Map이 IMU 정보를 사용하는 경우)
    int numTemporalKFs = 11; //TODO (set by parameter): Temporal KFs in the local window if the map is inertial.

    //Relationship to rebuild the essential graph, it is used two times, first in the local window and later in the rest of the map
    KeyFrame* pNewChild;
    KeyFrame* pNewParent;

    vector<KeyFrame*> vpLocalCurrentWindowKFs;
    vector<KeyFrame*> vpMergeConnectedKFs;

    KeyFrameAndPose CorrectedSim3, NonCorrectedSim3; // keyFrame과 수정된 pose을 저장하는 Map 객체 생성
    // NonCorrectedSim3[mpCurrentKF]=mg2oLoopScw;

    // Flag that is true only when we stopped a running BA, in this case we need relaunch at the end of the merge
    bool bRelaunchBA = false; // Global BA를 나중에 다시 시작해야 하는지 여부를 나타내는 플래그

    //cout << "Check Full Bundle Adjustment" << endl;
    // If a Global Bundle Adjustment is running, abort it
    if(isRunningGBA()) // 현재 Global BA가 실행 중인 경우
    {
        unique_lock<mutex> lock(mMutexGBA);
        mbStopGBA = true; // Global BA 중지 flag 설정

        mnFullBAIdx++; // Global BA의 인덱스를 증가

        if(mpThreadGBA) // Global BA 스레드가 존재한다면
        {
            mpThreadGBA->detach(); // Global BA 스레드를 중지하고 할당된 메모리 해제
            delete mpThreadGBA; // Global BA 스레드 객체 제거
        }
        bRelaunchBA = true; // Global BA를 나중에 다시 시작해야된다고 flag 설정
    }


    //cout << "Request Stop Local Mapping" << endl;
    mpLocalMapper->RequestStop(); // LocalMapping 스레드에 중단 요청을 보냄
    // Wait until Local Mapping has effectively stopped
    while(!mpLocalMapper->isStopped()) // LocalMapping 스레드가 중단될 때 까지 대기
    {
        usleep(1000);
    }
    //cout << "Local Map stopped" << endl;

    Map* pCurrentMap = mpCurrentKF->GetMap(); // 현재 KeyFrame에서 Map을 현재 Map으로 초기화
    Map* pMergeMap = mpMergeMatchedKF->GetMap(); // Merge 매칭 keyFrame에서 Map을 Merge Map으로 초기화

    {
        float s_on = mSold_new.scale(); // 공간 행렬의 scale을 가져와 저장
        Sophus::SE3f T_on(mSold_new.rotation().cast<float>(), mSold_new.translation().cast<float>()); // 공간 행렬의 변환 정보 생성

        unique_lock<mutex> lock(mpAtlas->GetCurrentMap()->mMutexMapUpdate);

        //cout << "KFs before empty: " << mpAtlas->GetCurrentMap()->KeyFramesInMap() << endl;
        mpLocalMapper->EmptyQueue(); // LocalMapping 스레드의 KeyFrame 배열을 비움
        //cout << "KFs after empty: " << mpAtlas->GetCurrentMap()->KeyFramesInMap() << endl;

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        //cout << "updating active map to merge reference" << endl;
        //cout << "curr merge KF id: " << mpCurrentKF->mnId << endl;
        //cout << "curr tracking KF id: " << mpTracker->GetLastKeyFrame()->mnId << endl;
        bool bScaleVel=false;
        if(s_on!=1) // scale이 1이 아닌 경우
            bScaleVel=true; // Map에서 속도 계산 시 scale 정보를 사용한다고 설정
        mpAtlas->GetCurrentMap()->ApplyScaledRotation(T_on,s_on,bScaleVel); // Atlas의 현재 Map에 스케일과 회전 변환을 적용
        mpTracker->UpdateFrameIMU(s_on,mpCurrentKF->GetImuBias(),mpTracker->GetLastKeyFrame()); // scale 값, 현재 KeyFrame의 Bias, Tracking의 이전 KeyFrame을 통해 IMU pose 및 속도 정보 갱신

        std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
    }

    const int numKFnew=pCurrentMap->KeyFramesInMap(); // 현재 Map에 KeyFrame이 몇개 있는지 저장
    
    // IMU를 사용한 추적 중이고 현재 Map에 IniertialBA2가 일어나지 않았다면
    if((mpTracker->mSensor==System::IMU_MONOCULAR || mpTracker->mSensor==System::IMU_STEREO || mpTracker->mSensor==System::IMU_RGBD)
       && !pCurrentMap->GetIniertialBA2()){
        // Map is not completly initialized
        Eigen::Vector3d bg, ba;
        bg << 0., 0., 0.; // 자이로 bias 초기화
        ba << 0., 0., 0.; // 가속도 bias 초기화
        Optimizer::InertialOptimization(pCurrentMap,bg,ba); // 현재 Map의 IMU 데이터 최적화
        IMU::Bias b (ba[0],ba[1],ba[2],bg[0],bg[1],bg[2]); // 최적화된 bias 정보를 통해 객체 생성
        unique_lock<mutex> lock(mpAtlas->GetCurrentMap()->mMutexMapUpdate);
        mpTracker->UpdateFrameIMU(1.0f,b,mpTracker->GetLastKeyFrame()); // 최적화된 bias와 Tracking의 이전 KeyFrame을 통해 IMU pose 및 속도 정보 갱신

        // Set map initialized
        pCurrentMap->SetIniertialBA2(); // 현재 Map에 IniertialBA2가 일어났다고 설정
        pCurrentMap->SetIniertialBA1(); // 현재 Map에 IniertialBA1이 일어났다고 설정
        pCurrentMap->SetImuInitialized(); // 현재 Map에 IMU가 초기화 되었다고 설정

    }


    //cout << "MergeMap init ID: " << pMergeMap->GetInitKFid() << "       CurrMap init ID: " << pCurrentMap->GetInitKFid() << endl;

    // Load KFs and MPs from merge map
    //cout << "updating current map" << endl;
    {
        // Get Merge Map Mutex (This section stops tracking!!)
        unique_lock<mutex> currentLock(pCurrentMap->mMutexMapUpdate); // We update the current map with the Merge information
        unique_lock<mutex> mergeLock(pMergeMap->mMutexMapUpdate); // We remove the Kfs and MPs in the merged area from the old map


        vector<KeyFrame*> vpMergeMapKFs = pMergeMap->GetAllKeyFrames(); // Merge Map에 있는 모든 KeyFrame을 가져옴
        vector<MapPoint*> vpMergeMapMPs = pMergeMap->GetAllMapPoints(); // Merge Map에 있는 MapPoint를 가져옴


        for(KeyFrame* pKFi : vpMergeMapKFs) // Merge Map에 있는 KeyFrame들에 대해 loop
        {
            if(!pKFi || pKFi->isBad() || pKFi->GetMap() != pMergeMap) // keyFrame이 존재하지 않거나 Bad 상태이거나 저장된 Map이 Merge Map과 다르면 무시
            {
                continue;
            }

            // Make sure connections are updated
            pKFi->UpdateMap(pCurrentMap); // keyFrame의 Map을 현재 Map으로 갱신
            pCurrentMap->AddKeyFrame(pKFi); // 현재 Map에 KeyFrame을 추가
            pMergeMap->EraseKeyFrame(pKFi); // Merge Map에 KeyFrame을 지움
        }

        for(MapPoint* pMPi : vpMergeMapMPs) // Merge Map에 있는 MapPoint들에 대해 loop
        {
            if(!pMPi || pMPi->isBad() || pMPi->GetMap() != pMergeMap) // MapPoint가 존재하지 않거나 Bad 상태이거나 저장된 Map이 Merge Map과 다르면 무시
                continue;

            pMPi->UpdateMap(pCurrentMap); // MapPoint의 Map을 현재 Map으로 갱신
            pCurrentMap->AddMapPoint(pMPi); // 현재 Map에 MapPoint를 추가
            pMergeMap->EraseMapPoint(pMPi); // Merge Map에 MapPoint를 지움
        }

        // Save non corrected poses (already merged maps)
        vector<KeyFrame*> vpKFs = pCurrentMap->GetAllKeyFrames(); // 현재 Map에 있는 모든 KeyFrame을 가져옴
        for(KeyFrame* pKFi : vpKFs) // 현재 Map에 있는 모든 KeyFrame에 대해 loop
        {
            Sophus::SE3d Tiw = (pKFi->GetPose()).cast<double>();
            g2o::Sim3 g2oSiw(Tiw.unit_quaternion(),Tiw.translation(),1.0);
            NonCorrectedSim3[pKFi]=g2oSiw; // KeyFrame들의 pose 정보를 저장
        }
    }

    //cout << "MergeMap init ID: " << pMergeMap->GetInitKFid() << "       CurrMap init ID: " << pCurrentMap->GetInitKFid() << endl;

    //cout << "end updating current map" << endl;

    // Critical zone
    //bool good = pCurrentMap->CheckEssentialGraph();
    /*if(!good)
        cout << "BAD ESSENTIAL GRAPH!!" << endl;*/

    //cout << "Update essential graph" << endl;
    // mpCurrentKF->UpdateConnections(); // to put at false mbFirstConnection
    pMergeMap->GetOriginKF()->SetFirstConnection(false); // MergeMap의 초기화 KeyFrame을 가져와 First Connection flag를 flase로 설정
    pNewChild = mpMergeMatchedKF->GetParent(); // Merge 매칭 KeyFrame의 부모 KeyFrame을 새로운 자식 KeyFrame으로 초기화 // Old parent, it will be the new child of this KF
    pNewParent = mpMergeMatchedKF; // Merge 매칭 KeyFrame을 새로운 부모 KeyFrame으로 초기화 // Old child, now it will be the parent of its own parent(we need eliminate this KF from children list in its old parent)
    mpMergeMatchedKF->ChangeParent(mpCurrentKF); // Merge 매칭 KeyFrame의 부모 keyFrame을 현재 KeyFrame으로 변경
    while(pNewChild) // 자식 keyFrame이 존재 한다면 loop
    {
        pNewChild->EraseChild(pNewParent); // 자식 KeyFrame에 새로운 부모 KeyFrame이 자식으로 존재 한다면 지움 // We remove the relation between the old parent and the new for avoid loop
        KeyFrame * pOldParent = pNewChild->GetParent(); // 자식 KeyFrame에서 부모 keyFrame을 가져와 이전 부모 KeyFrame으로 초기화
        pNewChild->ChangeParent(pNewParent); // 자식 KeyFrame의 부모를 새로운 부모 KeyFrame으로 변경
        pNewParent = pNewChild; // 새로운 부모 KeyFrame을 새로운 자식 KeyFrame으로 초기화
        pNewChild = pOldParent; // 새로운 자식 KeyFrame을 이전 부모 KeyFrame으로 초기화

    }


    //cout << "MergeMap init ID: " << pMergeMap->GetInitKFid() << "       CurrMap init ID: " << pCurrentMap->GetInitKFid() << endl;

    //cout << "end update essential graph" << endl;

    /*good = pCurrentMap->CheckEssentialGraph();
    if(!good)
        cout << "BAD ESSENTIAL GRAPH 1!!" << endl;*/

    //cout << "Update relationship between KFs" << endl;
    vector<MapPoint*> vpCheckFuseMapPoint; // MapPoint vector from current map to allow to fuse duplicated points with the old map (merge)
    vector<KeyFrame*> vpCurrentConnectedKFs;

    mvpMergeConnectedKFs.push_back(mpMergeMatchedKF); // Merge 매칭 KeyFrame을 Merge Connected KeyFrame 배열에 삽입
    vector<KeyFrame*> aux = mpMergeMatchedKF->GetVectorCovisibleKeyFrames(); // 매칭 KeyFrame의 Covisiblity Graph를 통해 이웃 KeyFrame을 가져옴
    mvpMergeConnectedKFs.insert(mvpMergeConnectedKFs.end(), aux.begin(), aux.end()); // Merge Connected KeyFrame배열에 이웃 KeyFrame을 삽입
    if (mvpMergeConnectedKFs.size()>6) // 배열의 크기가 6보다 큰 경우
        mvpMergeConnectedKFs.erase(mvpMergeConnectedKFs.begin()+6,mvpMergeConnectedKFs.end()); // 6번째 이후 KeyFrame은 제거
    /*mvpMergeConnectedKFs = mpMergeMatchedKF->GetVectorCovisibleKeyFrames();
    mvpMergeConnectedKFs.push_back(mpMergeMatchedKF);*/

    mpCurrentKF->UpdateConnections(); // 현재 KeyFrame의 Connection을 갱신
    vpCurrentConnectedKFs.push_back(mpCurrentKF); // 현재 Connection KeyFrame 배열에 현재 KeyFrame을 삽입
    /*vpCurrentConnectedKFs = mpCurrentKF->GetVectorCovisibleKeyFrames();
    vpCurrentConnectedKFs.push_back(mpCurrentKF);*/
    aux = mpCurrentKF->GetVectorCovisibleKeyFrames(); // 현재 KeyFrame의 Covisiblity Graph를 통해 이웃 KeyFrame을 가져옴
    vpCurrentConnectedKFs.insert(vpCurrentConnectedKFs.end(), aux.begin(), aux.end()); // 현재 Connected KeyFrame배열에 이웃 KeyFrame을 삽입
    if (vpCurrentConnectedKFs.size()>6) // 배열의 크기가 6보다 큰 경우
        vpCurrentConnectedKFs.erase(vpCurrentConnectedKFs.begin()+6,vpCurrentConnectedKFs.end()); // 6번째 이후 KeyFrame은 제거

    set<MapPoint*> spMapPointMerge;
    for(KeyFrame* pKFi : mvpMergeConnectedKFs) // Merge Connected KeyFrame배열을 loop
    {
        set<MapPoint*> vpMPs = pKFi->GetMapPoints(); // KeyFrame의 모든 MapPoint를 가져옴
        spMapPointMerge.insert(vpMPs.begin(),vpMPs.end()); // Merge MapPoint 배열에 삽입
        if(spMapPointMerge.size()>1000) // 배열의 크기가 1000 보다 크면 반복 종료
            break;
    }

    /*cout << "vpCurrentConnectedKFs.size() " << vpCurrentConnectedKFs.size() << endl;
    cout << "mvpMergeConnectedKFs.size() " << mvpMergeConnectedKFs.size() << endl;
    cout << "spMapPointMerge.size() " << spMapPointMerge.size() << endl;*/


    vpCheckFuseMapPoint.reserve(spMapPointMerge.size());
    std::copy(spMapPointMerge.begin(), spMapPointMerge.end(), std::back_inserter(vpCheckFuseMapPoint)); // 모은 MapPoint들을 Fusion Check 배열에 복사
    //cout << "Finished to update relationship between KFs" << endl;

    //cout << "MergeMap init ID: " << pMergeMap->GetInitKFid() << "       CurrMap init ID: " << pCurrentMap->GetInitKFid() << endl;

    /*good = pCurrentMap->CheckEssentialGraph();
    if(!good)
        cout << "BAD ESSENTIAL GRAPH 2!!" << endl;*/

    //cout << "start SearchAndFuse" << endl;
    SearchAndFuse(vpCurrentConnectedKFs, vpCheckFuseMapPoint); // 현재 연결된 KeyFrame과 MapPoint에 대해 pose를 사용하여 MapPoint를 Fusion하고 Replace
    //cout << "end SearchAndFuse" << endl;

    //cout << "MergeMap init ID: " << pMergeMap->GetInitKFid() << "       CurrMap init ID: " << pCurrentMap->GetInitKFid() << endl;

    /*good = pCurrentMap->CheckEssentialGraph();
    if(!good)
        cout << "BAD ESSENTIAL GRAPH 3!!" << endl;

    cout << "Init to update connections" << endl;*/


    for(KeyFrame* pKFi : vpCurrentConnectedKFs) // 현재 연결된 KeyFrame에 대해서 loop
    {
        if(!pKFi || pKFi->isBad()) // KeyFrame이 존재하지 않거나 Bad 상태인 경우 무시
            continue;

        pKFi->UpdateConnections(); // KeyFrame의 Connection을 갱신
    }
    for(KeyFrame* pKFi : mvpMergeConnectedKFs) // Merge Connection KeyFrame에 대해서 loop
    {
        if(!pKFi || pKFi->isBad()) // KeyFrame이 존재하지 않거나 Bad 상태인 경우 무시
            continue;

        pKFi->UpdateConnections(); // KeyFrame의 Connection을 갱신
    }
    //cout << "end update connections" << endl;

    //cout << "MergeMap init ID: " << pMergeMap->GetInitKFid() << "       CurrMap init ID: " << pCurrentMap->GetInitKFid() << endl;

    /*good = pCurrentMap->CheckEssentialGraph();
    if(!good)
        cout << "BAD ESSENTIAL GRAPH 4!!" << endl;*/

    // TODO Check: If new map is too small, we suppose that not informaiton can be propagated from new to old map
    if (numKFnew<10){ // 현재 Map에 KeyFrame이 10개 보다 적게 존재한다면
        mpLocalMapper->Release(); // LocalMapping 스레드를 깨우고 함수 종료
        return;
    }

    /*good = pCurrentMap->CheckEssentialGraph();
    if(!good)
        cout << "BAD ESSENTIAL GRAPH 5!!" << endl;*/

    // Perform BA
    bool bStopFlag=false;
    KeyFrame* pCurrKF = mpTracker->GetLastKeyFrame(); // Tracking 스레드에서 이전 KeyFrame을 가져옴
    //cout << "start MergeInertialBA" << endl;
    Optimizer::MergeInertialBA(pCurrKF, mpMergeMatchedKF, &bStopFlag, pCurrentMap,CorrectedSim3); // 현재 KeyFrame과 Merge KeyFrame간의 inertial BA를 수행 (그래프 최적화를 수행)
    //cout << "end MergeInertialBA" << endl;

    /*good = pCurrentMap->CheckEssentialGraph();
    if(!good)
        cout << "BAD ESSENTIAL GRAPH 6!!" << endl;*/

    // Release Local Mapping.
    mpLocalMapper->Release(); // LocalMapping 스레드를 깨움


    return;
}

void LoopClosing::CheckObservations(set<KeyFrame*> &spKFsMap1, set<KeyFrame*> &spKFsMap2)
{
    cout << "----------------------" << endl;
    for(KeyFrame* pKFi1 : spKFsMap1)
    {
        map<KeyFrame*, int> mMatchedMP;
        set<MapPoint*> spMPs = pKFi1->GetMapPoints();

        for(MapPoint* pMPij : spMPs)
        {
            if(!pMPij || pMPij->isBad())
            {
                continue;
            }

            map<KeyFrame*, tuple<int,int>> mMPijObs = pMPij->GetObservations();
            for(KeyFrame* pKFi2 : spKFsMap2)
            {
                if(mMPijObs.find(pKFi2) != mMPijObs.end())
                {
                    if(mMatchedMP.find(pKFi2) != mMatchedMP.end())
                    {
                        mMatchedMP[pKFi2] = mMatchedMP[pKFi2] + 1;
                    }
                    else
                    {
                        mMatchedMP[pKFi2] = 1;
                    }
                }
            }

        }

        if(mMatchedMP.size() == 0)
        {
            cout << "CHECK-OBS: KF " << pKFi1->mnId << " has not any matched MP with the other map" << endl;
        }
        else
        {
            cout << "CHECK-OBS: KF " << pKFi1->mnId << " has matched MP with " << mMatchedMP.size() << " KF from the other map" << endl;
            for(pair<KeyFrame*, int> matchedKF : mMatchedMP)
            {
                cout << "   -KF: " << matchedKF.first->mnId << ", Number of matches: " << matchedKF.second << endl;
            }
        }
    }
    cout << "----------------------" << endl;
}


void LoopClosing::SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap, vector<MapPoint*> &vpMapPoints)
{
    ORBmatcher matcher(0.8); // ORB Matcher 객체 생성

    int total_replaces = 0; // MapPoint replace의 총 합을 초기화

    //cout << "[FUSE]: Initially there are " << vpMapPoints.size() << " MPs" << endl;
    //cout << "FUSE: IntiallmTcmy there are " << CorrectedPosesMap.size() << " KFs" << endl;
    for(KeyFrameAndPose::const_iterator mit=CorrectedPosesMap.begin(), mend=CorrectedPosesMap.end(); mit!=mend;mit++) // keyFrame과 수정된 pose 행렬을 저장한 배열을 loop
    {
        int num_replaces = 0; // replace한 수를 저장할 변수 초기화
        KeyFrame* pKFi = mit->first; // KeyFrame을 가져옴
        Map* pMap = pKFi->GetMap(); // KeyFrame의 Map을 가져옴

        g2o::Sim3 g2oScw = mit->second; // 수정된 pose를 가져옴
        Sophus::Sim3f Scw = Converter::toSophus(g2oScw);

        vector<MapPoint*> vpReplacePoints(vpMapPoints.size(),static_cast<MapPoint*>(NULL)); // MapPoint 배열의 크기만큼 배열 생성
        int numFused = matcher.Fuse(pKFi,Scw,vpMapPoints,4,vpReplacePoints); // 키프레임과 수정된 pose를 기반으로 맵 포인트를 퓨전, 퓨전 수를 반환

        // Get Map Mutex
        unique_lock<mutex> lock(pMap->mMutexMapUpdate);
        const int nLP = vpMapPoints.size(); // fusion된 MapPoint의 개수를 저장
        for(int i=0; i<nLP;i++)
        {
            MapPoint* pRep = vpReplacePoints[i]; 
            if(pRep) // 퓨전 MapPoint가 존재한다면
            {


                num_replaces += 1; // MapPoint replace수 증가
                pRep->Replace(vpMapPoints[i]); // 퓨전 MapPoint에 대해 기존의 MapPoint를 replace

            }
        }

        total_replaces += num_replaces; // MapPoint replace의 총 합을 누적
    }
    //cout << "[FUSE]: " << total_replaces << " MPs had been fused" << endl;
}


void LoopClosing::SearchAndFuse(const vector<KeyFrame*> &vConectedKFs, vector<MapPoint*> &vpMapPoints)
{
    ORBmatcher matcher(0.8); // ORB Matcher 객체 생성

    int total_replaces = 0; // MapPoint replace의 총 합을 초기화

    //cout << "FUSE-POSE: Initially there are " << vpMapPoints.size() << " MPs" << endl;
    //cout << "FUSE-POSE: Intially there are " << vConectedKFs.size() << " KFs" << endl;
    for(auto mit=vConectedKFs.begin(), mend=vConectedKFs.end(); mit!=mend;mit++) // connection KeyFrame 배열을 loop
    {
        int num_replaces = 0; // replace한 수를 저장할 변수 초기화
        KeyFrame* pKF = (*mit);
        Map* pMap = pKF->GetMap(); // KeyFrame의 Map을 가져옴
        Sophus::SE3f Tcw = pKF->GetPose(); // KeyFrame의 pose를 가져옴
        Sophus::Sim3f Scw(Tcw.unit_quaternion(),Tcw.translation()); // pose를 통해 공간 변환 생성
        Scw.setScale(1.f); // 변환의 scale을 1로 설정
        /*std::cout << "These should be zeros: " <<
            Scw.rotationMatrix() - Tcw.rotationMatrix() << std::endl <<
            Scw.translation() - Tcw.translation() << std::endl <<
            Scw.scale() - 1.f << std::endl;*/
        vector<MapPoint*> vpReplacePoints(vpMapPoints.size(),static_cast<MapPoint*>(NULL)); // MapPoint 배열의 크기만큼 배열 생성
        matcher.Fuse(pKF,Scw,vpMapPoints,4,vpReplacePoints); // 키프레임과 수정된 pose를 기반으로 맵 포인트를 퓨전, 퓨전 수를 반환

        // Get Map Mutex
        unique_lock<mutex> lock(pMap->mMutexMapUpdate);
        const int nLP = vpMapPoints.size(); // fusion된 MapPoint의 개수를 저장
        for(int i=0; i<nLP;i++)
        {
            MapPoint* pRep = vpReplacePoints[i];
            if(pRep) // 퓨전 MapPoint가 존재한다면
            {
                num_replaces += 1; // MapPoint replace수 증가
                pRep->Replace(vpMapPoints[i]); // 퓨전 MapPoint에 대해 기존의 MapPoint를 replace
            }
        }
        /*cout << "FUSE-POSE: KF " << pKF->mnId << " ->" << num_replaces << " MPs fused" << endl;
        total_replaces += num_replaces;*/
    }
    //cout << "FUSE-POSE: " << total_replaces << " MPs had been fused" << endl;
}



void LoopClosing::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }

    while(1)
    {
        {
        unique_lock<mutex> lock2(mMutexReset);
        if(!mbResetRequested)
            break;
        }
        usleep(5000);
    }
}

void LoopClosing::RequestResetActiveMap(Map *pMap)
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetActiveMapRequested = true;
        mpMapToReset = pMap;
    }

    while(1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if(!mbResetActiveMapRequested)
                break;
        }
        usleep(3000);
    }
}

void LoopClosing::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if(mbResetRequested)
    {
        cout << "Loop closer reset requested..." << endl;
        mlpLoopKeyFrameQueue.clear();
        mLastLoopKFid=0;  //TODO old variable, it is not use in the new algorithm
        mbResetRequested=false;
        mbResetActiveMapRequested = false;
    }
    else if(mbResetActiveMapRequested)
    {

        for (list<KeyFrame*>::const_iterator it=mlpLoopKeyFrameQueue.begin(); it != mlpLoopKeyFrameQueue.end();)
        {
            KeyFrame* pKFi = *it;
            if(pKFi->GetMap() == mpMapToReset)
            {
                it = mlpLoopKeyFrameQueue.erase(it);
            }
            else
                ++it;
        }

        mLastLoopKFid=mpAtlas->GetLastInitKFid(); //TODO old variable, it is not use in the new algorithm
        mbResetActiveMapRequested=false;

    }
}

void LoopClosing::RunGlobalBundleAdjustment(Map* pActiveMap, unsigned long nLoopKF)
{  
    Verbose::PrintMess("Starting Global Bundle Adjustment", Verbose::VERBOSITY_NORMAL);

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartFGBA = std::chrono::steady_clock::now();

    nFGBA_exec += 1; // Global BA 실행 횟수 증가

    vnGBAKFs.push_back(pActiveMap->GetAllKeyFrames().size()); // 활성화 Map에 있는 모든 KeyFrame의 개수를 배열에 삽입
    vnGBAMPs.push_back(pActiveMap->GetAllMapPoints().size()); // 활성화 Map에 있는 모든 MapPoint의 개수를 배열에 삽입
#endif

    const bool bImuInit = pActiveMap->isImuInitialized(); 

    if(!bImuInit) // 활성화 Map에 IMU가 초기화 되지 않았다면
        Optimizer::GlobalBundleAdjustemnt(pActiveMap,10,&mbStopGBA,nLoopKF,false); // 활성화 Map을 통해 Global BA를 수행
    else
        Optimizer::FullInertialBA(pActiveMap,7,false,nLoopKF,&mbStopGBA); // 활성화 Map을 통해 Inertial BA 수행

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndGBA = std::chrono::steady_clock::now();

    double timeGBA = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndGBA - time_StartFGBA).count();
    vdGBA_ms.push_back(timeGBA);

    if(mbStopGBA)
    {
        nFGBA_abort += 1;
    }
#endif

    int idx =  mnFullBAIdx; // Global BA Index를 복사
    // Optimizer::GlobalBundleAdjustemnt(mpMap,10,&mbStopGBA,nLoopKF,false);

    // Update all MapPoints and KeyFrames
    // Local Mapping was active during BA, that means that there might be new keyframes
    // not included in the Global BA and they are not consistent with the updated map.
    // We need to propagate the correction through the spanning tree
    {
        unique_lock<mutex> lock(mMutexGBA);
        if(idx!=mnFullBAIdx) // 복사한 index와 Global BA Index가 다른경우 종료
            return;

        if(!bImuInit && pActiveMap->isImuInitialized()) // 활성화 Map에 IMU가 초기화되지 않았을 경우 종료
            return;

        if(!mbStopGBA) // Global BA 중지 명령이 없다면
        {
            Verbose::PrintMess("Global Bundle Adjustment finished", Verbose::VERBOSITY_NORMAL);
            Verbose::PrintMess("Updating map ...", Verbose::VERBOSITY_NORMAL);

            mpLocalMapper->RequestStop(); // LocalMapping 스레드에 중지 요청
            // Wait until Local Mapping has effectively stopped

            while(!mpLocalMapper->isStopped() && !mpLocalMapper->isFinished()) // Local Mapping 스레드가 멈출때 까지 대기
            {
                usleep(1000);
            }

            // Get Map Mutex
            unique_lock<mutex> lock(pActiveMap->mMutexMapUpdate);
            // cout << "LC: Update Map Mutex adquired" << endl;

            //pActiveMap->PrintEssentialGraph();
            // Correct keyframes starting at map first keyframe
            list<KeyFrame*> lpKFtoCheck(pActiveMap->mvpKeyFrameOrigins.begin(),pActiveMap->mvpKeyFrameOrigins.end()); // 활성화 Map에서 KeyFrame을 가져옴

            while(!lpKFtoCheck.empty()) // 배열이 비어있지 않는다면 loop
            {
                KeyFrame* pKF = lpKFtoCheck.front(); 
                const set<KeyFrame*> sChilds = pKF->GetChilds(); // KeyFrame의 자식 KeyFrame을 가져옴
                //cout << "---Updating KF " << pKF->mnId << " with " << sChilds.size() << " childs" << endl;
                //cout << " KF mnBAGlobalForKF: " << pKF->mnBAGlobalForKF << endl;
                Sophus::SE3f Twc = pKF->GetPoseInverse(); // KeyFrame의 pose를 저장
                //cout << "Twc: " << Twc << endl;
                //cout << "GBA: Correct KeyFrames" << endl;
                for(set<KeyFrame*>::const_iterator sit=sChilds.begin();sit!=sChilds.end();sit++) // 자식 keyFrame에 대해 loop
                {
                    KeyFrame* pChild = *sit;
                    if(!pChild || pChild->isBad()) // 자식 KeyFrame이 존재하지 않거나 Bad 상태인 경우 무시
                        continue;

                    if(pChild->mnBAGlobalForKF!=nLoopKF) // 자식 keyFrame의 Global BA KeyFrame의 수가 현재 KeyFrame의 ID와 다르다면
                    {
                        //cout << "++++New child with flag " << pChild->mnBAGlobalForKF << "; LoopKF: " << nLoopKF << endl;
                        //cout << " child id: " << pChild->mnId << endl;
                        Sophus::SE3f Tchildc = pChild->GetPose() * Twc; // 자식 KeyFrame의 pose와 부모 KeyFrame의 pose의 역을 곱해 자식 KeyFrame의 pose를 조정
                        //cout << "Child pose: " << Tchildc << endl;
                        //cout << "pKF->mTcwGBA: " << pKF->mTcwGBA << endl;
                        pChild->mTcwGBA = Tchildc * pKF->mTcwGBA; // 자식 KeyFrame의 Global BA pose를 조정한 pose와 곱해 누적 //*Tcorc*pKF->mTcwGBA;

                        Sophus::SO3f Rcor = pChild->mTcwGBA.so3().inverse() * pChild->GetPose().so3(); // 자식 KeyFrame의 Global BA pose와 pose를 곱함
                        if(pChild->isVelocitySet()){ // 자식 KeyFrame의 속도 정보가 초기화 되어 있다면
                            pChild->mVwbGBA = Rcor * pChild->GetVelocity(); // 자식 keyFrame의 Global BA 추정 속도를 초기화
                        }
                        else
                            Verbose::PrintMess("Child velocity empty!! ", Verbose::VERBOSITY_NORMAL);


                        //cout << "Child bias: " << pChild->GetImuBias() << endl;
                        pChild->mBiasGBA = pChild->GetImuBias(); // 자식 KeyFrame의 Global BA Bias를 초기화


                        pChild->mnBAGlobalForKF = nLoopKF; // 자식 KeyFrame의 Global BA KeyFrame ID를 초기화

                    }
                    lpKFtoCheck.push_back(pChild); // 배열에 자식 KeyFrame을 삽입
                }

                //cout << "-------Update pose" << endl;
                pKF->mTcwBefGBA = pKF->GetPose(); // KeyFrame의 Global BA이전 pose를 초기화
                //cout << "pKF->mTcwBefGBA: " << pKF->mTcwBefGBA << endl;
                pKF->SetPose(pKF->mTcwGBA); // KeyFrame의 pose를 추정 pose로 변경
                /*cv::Mat Tco_cn = pKF->mTcwBefGBA * pKF->mTcwGBA.inv();
                cv::Vec3d trasl = Tco_cn.rowRange(0,3).col(3);
                double dist = cv::norm(trasl);
                cout << "GBA: KF " << pKF->mnId << " had been moved " << dist << " meters" << endl;
                double desvX = 0;
                double desvY = 0;
                double desvZ = 0;
                if(pKF->mbHasHessian)
                {
                    cv::Mat hessianInv = pKF->mHessianPose.inv();

                    double covX = hessianInv.at<double>(3,3);
                    desvX = std::sqrt(covX);
                    double covY = hessianInv.at<double>(4,4);
                    desvY = std::sqrt(covY);
                    double covZ = hessianInv.at<double>(5,5);
                    desvZ = std::sqrt(covZ);
                    pKF->mbHasHessian = false;
                }
                if(dist > 1)
                {
                    cout << "--To much distance correction: It has " << pKF->GetConnectedKeyFrames().size() << " connected KFs" << endl;
                    cout << "--It has " << pKF->GetCovisiblesByWeight(80).size() << " connected KF with 80 common matches or more" << endl;
                    cout << "--It has " << pKF->GetCovisiblesByWeight(50).size() << " connected KF with 50 common matches or more" << endl;
                    cout << "--It has " << pKF->GetCovisiblesByWeight(20).size() << " connected KF with 20 common matches or more" << endl;

                    cout << "--STD in meters(x, y, z): " << desvX << ", " << desvY << ", " << desvZ << endl;


                    string strNameFile = pKF->mNameFile;
                    cv::Mat imLeft = cv::imread(strNameFile, CV_LOAD_IMAGE_UNCHANGED);

                    cv::cvtColor(imLeft, imLeft, CV_GRAY2BGR);

                    vector<MapPoint*> vpMapPointsKF = pKF->GetMapPointMatches();
                    int num_MPs = 0;
                    for(int i=0; i<vpMapPointsKF.size(); ++i)
                    {
                        if(!vpMapPointsKF[i] || vpMapPointsKF[i]->isBad())
                        {
                            continue;
                        }
                        num_MPs += 1;
                        string strNumOBs = to_string(vpMapPointsKF[i]->Observations());
                        cv::circle(imLeft, pKF->mvKeys[i].pt, 2, cv::Scalar(0, 255, 0));
                        cv::putText(imLeft, strNumOBs, pKF->mvKeys[i].pt, CV_FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 0, 0));
                    }
                    cout << "--It has " << num_MPs << " MPs matched in the map" << endl;

                    string namefile = "./test_GBA/GBA_" + to_string(nLoopKF) + "_KF" + to_string(pKF->mnId) +"_D" + to_string(dist) +".png";
                    cv::imwrite(namefile, imLeft);
                }*/


                if(pKF->bImu) // KeyFrame에 IMU 정보가 있다면
                {
                    //cout << "-------Update inertial values" << endl;
                    pKF->mVwbBefGBA = pKF->GetVelocity(); // KeyFrame의 Global BA이전 속도를 초기화
                    //if (pKF->mVwbGBA.empty())
                    //    Verbose::PrintMess("pKF->mVwbGBA is empty", Verbose::VERBOSITY_NORMAL);

                    //assert(!pKF->mVwbGBA.empty());
                    pKF->SetVelocity(pKF->mVwbGBA); // 추정된 속도 값으로 변경
                    pKF->SetNewBias(pKF->mBiasGBA); // 추정된 bias 값으로 변경                   
                }

                lpKFtoCheck.pop_front(); // 배열에서 keyFrame 제거
            }

            //cout << "GBA: Correct MapPoints" << endl;
            // Correct MapPoints
            const vector<MapPoint*> vpMPs = pActiveMap->GetAllMapPoints(); // 활성화 Map에서 모든 MapPoint를 가져옴

            for(size_t i=0; i<vpMPs.size(); i++) // MapPoint 배열 loop
            {
                MapPoint* pMP = vpMPs[i];

                if(pMP->isBad()) // MapPoint가 Bad 상태인 경우 무시
                    continue;

                if(pMP->mnBAGlobalForKF==nLoopKF) // MapPoint의 Global BA KeyFrame ID가 현재 KeyFrame의 ID와 같으면
                {
                    // If optimized by Global BA, just update
                    pMP->SetWorldPos(pMP->mPosGBA); // MapPoint의 world pose를 갱신
                }
                else
                {
                    // Update according to the correction of its reference keyframe
                    KeyFrame* pRefKF = pMP->GetReferenceKeyFrame(); // MapPoint의 레퍼런스 KeyFrame을 가져옴

                    if(pRefKF->mnBAGlobalForKF!=nLoopKF) // 레퍼런스 keyFrame의 Global BA KeyFrame ID가 현재 KeyFrame의 ID와 같으면 무시
                        continue;

                    /*if(pRefKF->mTcwBefGBA.empty())
                        continue;*/

                    // Map to non-corrected camera
                    // cv::Mat Rcw = pRefKF->mTcwBefGBA.rowRange(0,3).colRange(0,3);
                    // cv::Mat tcw = pRefKF->mTcwBefGBA.rowRange(0,3).col(3);
                    Eigen::Vector3f Xc = pRefKF->mTcwBefGBA * pMP->GetWorldPos(); // 레퍼런스 keyFrame의 Global BA이전 pose와 MapPoint의 world pose를 곱해 pose 추정

                    // Backproject using corrected camera
                    pMP->SetWorldPos(pRefKF->GetPoseInverse() * Xc); // MapPoint의 world pose 조정
                }
            }

            pActiveMap->InformNewBigChange(); // 활성화 Map에 큰 변화가 있었다고 알려줌
            pActiveMap->IncreaseChangeIndex(); // 활성화 Map에 index를 증가

            // TODO Check this update
            // mpTracker->UpdateFrameIMU(1.0f, mpTracker->GetLastKeyFrame()->GetImuBias(), mpTracker->GetLastKeyFrame());

            mpLocalMapper->Release(); // LocalMapping 스레드를 깨움

#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndUpdateMap = std::chrono::steady_clock::now();

            double timeUpdateMap = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndUpdateMap - time_EndGBA).count();
            vdUpdateMap_ms.push_back(timeUpdateMap);

            double timeFGBA = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndUpdateMap - time_StartFGBA).count();
            vdFGBATotal_ms.push_back(timeFGBA);
#endif
            Verbose::PrintMess("Map updated!", Verbose::VERBOSITY_NORMAL);
        }

        mbFinishedGBA = true; // Global BA가 끝났다고 flag 설정
        mbRunningGBA = false; // Global BA가 실행중이 아니라고 flag 설정
    }
}

void LoopClosing::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    // cout << "LC: Finish requested" << endl;
    mbFinishRequested = true;
}

bool LoopClosing::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LoopClosing::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

bool LoopClosing::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}


} //namespace ORB_SLAM
