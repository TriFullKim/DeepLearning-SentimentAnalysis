# DeepLearning-SentimentAnalysis
PKNU 딥러닝AI개론(Introduction to Deep Learning AI) 과제

# Init. Project
최근에 나온 영화 Venom the last dance의 리뷰를 긍정 또는 부정으로 예측하는 모델을 만들고 싶었음.
1. 외국의 유명한 영화 리뷰 웹사이트인 [RottenTomato](https://www.rottentomatoes.com/m/venom_the_last_dance)에서의 평가를 웹 크롤링 할 것이며,
2. 감정분석은 자연어 처리의 영역으로, sequence의 길이가 일정하지 않은 것의 처리가 필요함. 따라서, RNN 혹은 LSTM과 같은 Sequence 모델을 사용할 것입니다.
3. 훈련데이터셋은 Kaggle의 [IMDB리뷰데이터셋](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)을 기반으로 하여 학습할 것이며, **리뷰**라는 같은 도메인이라고 생각하여 적절하다고 생각합니다.

# Workspace Structure
아래와 같이 데이터베이스안에 데이터를 추가 해 주세요.
```bash
./database/IMDB Dataset.csv
```

# Run Crawling
```bash
python crawling.py
```