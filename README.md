# MOKPO AI 상권 분석 (교육용)

인공지능 기초 수행평가용 웹앱입니다. **모든 학습 데이터는 코드로 생성한 가상 데이터**이며 실제 목포 상권을 설명하지 않습니다.

## 기능
- 지역/업종/월 선택
- K-means로 8개 가상 지역의 상권 특성 군집화
- Random Forest 회귀로 가상 월매출 예측
- 학습/테스트 분리, MAE/R² 확인
- 규칙 기반 경영전략 제안

## iPad에서 배포
1. GitHub에서 Public 저장소 생성
2. `app.py`, `requirements.txt`, `README.md`를 저장소 최상위에 업로드 (GitHub 웹사이트의 Add file > Upload files)
3. https://share.streamlit.io 에서 GitHub 연결
4. 저장소 선택, Branch `main`, Main file path `app.py` 지정 후 Deploy
5. 오류 발생 시 앱 로그를 확인하고 메시지를 ChatGPT에 전달

## 실제 데이터로 바꾸려면
가상 데이터 생성 함수 `make_demo_data()`를 실제 공개 데이터의 적절한 로딩/전처리 함수로 교체하고, 변수 단위와 예측 대상(예: 음식점 월매출)을 정의해야 합니다. 지역 간 비교 가능성, 결측값, 데이터 이용 조건, 시간 기반 검증, 개인정보 보호를 확인하세요.
