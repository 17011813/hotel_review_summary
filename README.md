# hotel_review_summary

데이터셋은 [호텔 리뷰](https://www.kaggle.com/datafiniti/hotel-reviews) 에서 다운받았습니다.

데이터 셋에서 Text랑 Summary 부분만 필요하기 때문에 코드에서 처리해 놨습니다.

훈련은 LSTM 인코더와 Attention 디코더를 이용해서 총 26번 epoch를 돌아 학습합니다.

Loss는 다음과 같이 나타납니다.
<img src="https://user-images.githubusercontent.com/48427281/90232231-50f55d80-de57-11ea-9350-bdb692e78181.JPG" width="90%"></img>

최종 결과로는 다음과 같이 나타납니다.

원문 :  stayed days nice hotel one offered breakfast plus room th floor big queen size beds tv 
실제 요약문 : great location 
예측 요약문 :  great location


원문 :  average place stay budget traveler dont mind save hotel spend places shopping nearby downtown san francisco walking distance union square attractions restaurants night city offer wifi rooms 
실제 요약문 : center inn hotel san francisco 
예측 요약문 :  good value for the price


원문 :  quality inn average compared quality inns clean convenient reason review higher room enough towels appeared maid never finished hallway long time machine third floor work front day person seemed 
실제 요약문 : convenient location 
예측 요약문 :  good stay


원문 :  nothing best staff rooms everything amazing thanks much making christmas trip best ever cannot wait stay even maids stopped greet us walked hall 
실제 요약문 : best hotel ever 
예측 요약문 :  great stay
