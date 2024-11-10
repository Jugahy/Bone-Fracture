# Bone-Fracture Detection
전주 효사랑 가족 병원 방사선실에서 프로젝트 인턴으로 뼈 골절 감지 딥러닝 모델 개발<br/>
Developed a deep learning model for bone fracture detection as a project intern in the radiation room of Jeonju Hyosarang Family Hospital


### Table of contents 

1. [Overview](#1️⃣-overview)
2. [Role](#2️⃣-role)
3. [Skills & Process](#3️⃣-skills)
4. [References](#4️⃣-process)
5. [Structure](#5️⃣-structure)


## 1️⃣ Overview

### 1-1. 개발 배경
WE-Meet을 통해 전주 효사랑 가족 병원 방사선실에서 프로젝트 인턴으로 함께했습니다. 주어진 문제는 전라북도 노인의 수 급증에 따른 노인 골절 환자 수가 증가하는 상황에서 X-ray를 통해 골절을 감지할 때  노인 뼈의 특성상 아래와 같은 감지에 어려움을 주는 문제가 있습니다.

- 골밀도 감소 : 나이가 들면서 뼈의 골밀도가 낮아져 X-ray, CT 이미지에서 뼈가 더 희미하게 보일 수 있습니다.
- 관절염 및 퇴행성 질환 : 노인들에게서 흔히 볼 수 있는 관절염, 퇴행성 질환은 뼈와 관절 주변의 변형을 초래하여 뼈 구조의 식별을 더 어렵게 만듭니다.

이런 문제들로 인해 골절을 발견하지 못하게 되면 통증과 불편함이 지속되고 2차 부상의 위험이 있습니다.

이를 해결하기 위해 여러 딥러닝 기술을 활용하여 **Bone Fracture Detection** 프로젝트를 진행하게 되었습니다.
<br/>
<br/>
>### WE-Meet
>WE(Work Experience)-Meet 프로젝트는 산업계에서 문제해결 및 프로젝트 주제를 제시하고 대학생이 직접 프로젝트를 수행합니다. 기업과 대학은 학생에게 일경험 기회를 제공하여 우수인재를 발굴·검증하는 사회적 역할을 이행하고 대학생은 기업에 대한 이해와 적응력을 향상할 수 있습니다.



## 2️⃣ Role
|<img src="https://github.com/user-attachments/assets/bef1a11a-d69d-440a-9ed5-7c8f39548c5a" width="150" height="150"/>|
|:-:|
|Jeong GangHyeon<br/>[@JUGAHY](https://github.com/JUGAHY)|

### Jeong GangHyeon
* AI Technical Papers Review and Implementation
* Data from multiple hospitals, collecting published fracture data
* FracAtlas data preprocessing, visualization



## 3️⃣ Skills 

### Project skills 

__1. Language & Tool__ 

- Python 3.8 
- Jupyter notebook

__2. DeepLearning Model__

- VGG-16
- Fast R-CNN
- YOLOv7
- Mask R-CNN
- U-net

__3. Application deployment__

- Streamlit



## 4️⃣ Process

### 4-1. Data Selection ([0.데이터 선정.ipynb](https://github.com/Jugahy/Bone-Fracture/blob/main/0.%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EC%84%A0%EC%A0%95.ipynb))
* [FracAtlas : A Dataset for Fracture Classification, Localization and Segmentation of Musculoskeletal Radiographs](https://github.com/Jugahy/AI-Paper/blob/main/Bone_Fracture/FracAtlas%20%3A%20A%20Dataset%20for%20Fracture%20Classification%2C%20Localization%20and%20Segmentation%20of%20Musculoskeletal%20Radiographs/(Review)%20FracAtlas%20%3A%20A%20Dataset%20for%20Fracture%20Classification%2C%20Localization%20and%20Segmentation%20of%20Musculoskeletal%20Radiographs.ipynb) 논문 참고하여 FracAtlas 데이터 선정하게 되었습니다.
* 여러 부위의 골절 데이터와 Object Detection을 위한 라벨링이 되어있기 때문에 FracAtlas 데이터를 사용하기 가장 적합합니다.
![image](https://github.com/user-attachments/assets/31886bbf-c11e-4bef-a879-935b705356d6)

### 4-2. Data Introduction [(1.데이터 소개.ipynb)](https://github.com/Jugahy/Bone-Fracture/blob/main/1.%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EC%86%8C%EA%B0%9C.ipynb)

* FracAlas 데이터 폴더 안에 dataset이라는 csv가 있습니다. dataset은 아래와 같이 구성되어 있습니다.
    * 골절 부위
      * hand : (ex : IMG0000019.jpg)
      * leg : (ex : IMG0000092.jpg)
      * hip : (ex : IMG0002180.jpg)
      * shoulder : (ex : IMG0002302.jpg)
      * mixed : 여러 부위 골절
    * 특징
      * hardware : 뼈 고정 금속 여부
      * multiscan : X-ray, CT 스캔 등 여러 가지 스캔 방법을 사용하여 얻은 데이터
      * fracture : 골절 여부
      * fracture_count : 골절의 개수
    * 촬영방법
      * frontal : 정면 이미지
      * lateral : 측면 이미지
      * oblique : 사선 이미지

  * 각 부위 별 정상, 골절 데이터 수 시각화 / 골절 수 분포
<p align="center">
  <img src="https://github.com/user-attachments/assets/0e1f5fae-88ff-4805-991d-cf670758469f" alt="각 부위 별 정상, 골절 데이터 수 시각화" width="45%" />
  <img src="https://github.com/user-attachments/assets/e161ae5a-824d-45a7-867f-4059dca9e96f" alt="골절 수 분포" width="45%" />
</p>
<br/>
<br/>

* Label 데이터를 불러와 골절 데이터에 overlab 해보았습니다. (box, polygon 두 형태로 제공)
  
![image](https://github.com/user-attachments/assets/7aa7b47d-1e7d-4f41-80e3-62d9904ed74a)
![image](https://github.com/user-attachments/assets/2f990146-d178-4e2e-bb37-8524f326c131)
![image](https://github.com/user-attachments/assets/270d2c3a-981f-4778-b697-50fb10add7d2)

