# DeepLearning-SentimentAnalysis
PKNU 딥러닝AI개론(Introduction to Deep Learning AI) 과제

딥러닝AI개론의 기말 대체과제를 위한 레포지토리입니다. 아래 사항들을 주의해 진행합니다.
1. 발표시간은 학생당 10분 이상으로 한다. (시간 초과는 상관없으나, 주어진 시간보다 부족할 경우 감점이 있을 수 있음)
2. 발표자는 발표 후, 간단한 질의응답 시간 (3~5분)을 갖는다.
3. 발표자 및 팀원이 해당 날짜에 결석할 경우, 기말고사는 0점 처리된다.
4. 데이터마이닝개론 수업을 중복해서 듣는 학생이 동일한 내용/주제로 발표할 시, 0점처리 된다.
5. 발표자는 발표 전날 자정까지 발표 자료(ppt)를 교수에게 메일로 보내야 된다.
6. 데이터분석기법은 수업시간에 다룬 기법을 활용한다.
7. 데이터분석에 활용한 코드 반드시 첨부 (코드 따로 제출)

# Init. Project
최근에 나온 영화 Venom the last dance의 리뷰를 긍정 또는 부정으로 예측하는 모델을 만들고 싶었음.
1. 외국의 유명한 영화 리뷰 웹사이트인 [RottenTomato](https://www.rottentomatoes.com/m/venom_the_last_dance)에서의 평가를 웹 크롤링 할 것이며,
2. 감정분석은 자연어 처리의 영역으로, sequence의 길이가 일정하지 않은 것의 처리가 필요함. 따라서, RNN 혹은 LSTM과 같은 Sequence 모델을 사용할 것입니다.
3. 훈련데이터셋은 Kaggle의 [IMDB리뷰데이터셋](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)을 기반으로 하여 학습할 것이며, **리뷰**라는 같은 도메인이라고 생각하여 적절하다고 생각합니다.

# Tasks
- [x] RottenTomato Review Crawler
- [ ] Training Sequence DeepLearning Model at `PyTorch` 

# Workspace Structure
아래와 같이 데이터베이스 디렉토리를 생성해 주세요.
```bash
./
+---+--- database ---+--- IMDB Dataset.csv
                     +--- *_review.json
```