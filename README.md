# National-Science-Council 科技部大專生研究計畫
## 以LSTM探討美國股市指數變動對我國股市加權指數開盤之影響

## 摘要
各國間相互流通的貿易關係與各大經濟體之整合，讓國際金融市場的關係日漸密切，而被視為景氣領先指標之一的股價指數，容易受到國內外經濟、社會以及政治情勢的影響，金融危機的發生，更提升各國股市間的相關性。其中，美國作為國際金融市場的龍頭，對各國股市具有單向的外溢效果，與台股之相關性更達到八成，因此本研究旨在透過LSTM長短期記憶網路模型結合台股加權指數與美股指數進行深度學習，利用人工智慧挖掘台灣股市市場隱藏趨勢，分析人力不易處理的股市動態影響因素與相互因果關係，深度剖析美股理想股價指數標的——標準普爾500指數(S&P 500)以及高科技產業指標——納斯達克指數(NASDAQ)對台灣發行量加權股價指數之遞延效果及影響程度，結果顯示，納斯達克綜合指數與台灣加權指數間的相關性與溢出效應較標準普爾500指數明顯，且台灣加權指數中具有納斯達克綜合指數的隱藏趨勢，可透過LSTM模型將其挖掘出，並提升模型對隔日台灣加權指數開盤價走勢的預測能力，而利用標準普爾500指數進行訓練的模型，則推測為受到指數組成成分不同的影響，導致其指數增長趨勢對台灣加權指數的影響力不如預期。除此之外，美中貿易戰與疫情等外部因素也對本研究之預測模型造成影響，即使以人工智慧進行價格預測，投資者也須承擔一定的風險，由於突發性的事件會對股市造成衝擊性的影響且無法被模型預測，投資者應持續關注世界局勢脈動與金融市場變化，未來也將持續嘗試改良現有的預測模型，更全面的解析股票市場的結構，協助投資者進行風險分散與股票交易，提升其投資績效與獲利。

關鍵詞：人工智慧、加權指數、長短期記憶、深度學習

## 結論
本次研究共建立了7種預測模型，分別採用了差分整合移動平均自我迴歸模型（ARIMA）、線性迴歸模型（Linear Regression）、人工神經網路模型（ANN）以及四種長短期記憶模（LSTM），除了比較各種模型的預測表現外，並根據實驗結果發現，在平均表現上，Linear Regression模型具有最低的均方根誤差，其次為滾動式預測的ARIMA模型，第三則為單變量的LSTM模型，而ANN模型表現最差且相差最大，展現出ARIMA、Linear Regression與LSTM面對時間序列數據時的處理能力。同時，滾動式預測的方法也讓原先代表傳統的ARIMA模型獲得了與現代創新技術並肩的實力，不過LSTM模型的最優表現（RMSE=141.59，R2_Score=0.9911）則是超越了Linear Regression模型（RMSE=142.02，R2_Score=0.9911）與ARIMA模型（RMSE=142.21，R2_Score=0.9911），意味著LSTM模型具備的預測能力上限依舊超越了ARIMA模型。
	至於美股與台股間的遞延關係與影響效果，根據LSTM多變量模型的預測結果推論，納斯達克綜合指數與台灣加權指數間的相關性與溢出效應較標準普爾500指數明顯，且台灣加權指數中具有納斯達克綜合指數的隱藏趨勢，可透過LSTM模型將其挖掘出，並提升模型對隔日台灣加權指數開盤價走勢的預測能力，而利用標準普爾500指數進行訓練的模型則不如預期，推測受到兩種指數組成成分不同的影響，導致其指數增長趨勢對台灣加權指數的影響力不同，納斯達克綜合指數以高科技產業為主，與台灣加權指數組成成分較為相似，標準普爾500指數則是選擇美國市場上500間企業作為其指數成分，代表著美國整體的經濟大環境，而除了成分因素外，受到美中貿易戰與疫情等外部因素的影響，台灣股市走向與美國股市走向略有不同，也對本研究之預測模型產生影響。
	股票作為一種投資工具，受到各式各樣因素的影響，即使以人工智慧進行價格預測，投資者也須承擔一定的風險，雖本次研究之模型預測能力十分亮眼，但由於突發性的事件會對股市造成衝擊性的影響且無法被模型預測，投資者應持續關注世界局勢脈動與金融市場變化，而未來也將持續嘗試改良現有的預測模型，加入更多的影響因素進行訓練，並結合不同的特徵擷取方法，協助人工智慧模型更全面的解析股票市場的結構。

## 參考文獻
1.	姜淑美、陳明麗、蔡佩珊(2005)。國際股價指數現貨與期貨報酬外溢性及不對稱性效果之研究.經營管理論叢，1卷2期，23-39。
2.	王冠閔、吳書慧(2006)，台灣股、匯市與美國股市傳導機制之實證分析，運籌研究集刊，10期，1-15。
3.	A. A. Ariyo, A. O. Adewumi and C. K. Ayo, "Stock Price Prediction Using the ARIMA Model," 2014 UKSim-AMSS 16th International Conference on Computer Modelling and Simulation, 2014, pp. 106-112
4.	Abhishek Dutta, Jaydip Sen, &Sidra Mehtab(2020),Robust Predictive Models For The Indian IT Sector Using Machine Learning And Deep Learning 
5.	Adam, K., Smagulova, K., Krestinskaya, O., & James, A. P. (2018, December). Wafer quality inspection using memristive LSTM, ANN, DNN and HTM. In 2018 IEEE Electrical Design of Advanced Packaging and Systems Symposium (EDAPS) (pp. 1-3). IEEE.
6.	Adil Moghar,&Mhamed Hamiche(2020),Stock Market Prediction Using LSTM Recurrent Neural Network,Procedia Computer Science, 170, 1168-1173.
7.	Benvenuto, D., Giovanetti, M., Vassallo, L., Angeletti, S., & Ciccozzi, M. (2020). Application of the ARIMA model on the COVID-2019 epidemic dataset. Data in brief, 29, 105340.
8.	Cheol S. Eun,Sangdal Shim(1989), International Transmission of Stock Market Movements, The Journal of Financial and Quantitative Analysis,24, 241-256.
9.	Donald Lien, Geul Lee, Li Yang, &Yuyin Zhang(2018),Volatility spillovers among the U.S. and Asian stock markets: A comparison between the periods of Asian currency crisis and subprime credit crisis,The North American Journal of Economics and Finance, 46,Pages 187-201.
10.	E.L. de Faria, Marcelo P. Albuquerque, J.L. Gonzalez, J.T.P. Cavalcante, Marcio P. Albuquerque, (2009). Predicting the Brazilian stock market through neural networks and adaptive exponential smoothing methods.
11.	Göçken, M., Özçalıcı, M., Boru, A., & Dosdoğru, A. T. (2016). Integrating metaheuristics and artificial neural networks for improved stock price prediction. Expert Systems with Applications, 44, 320-331.
12.	Huang, S. C., Yang, C. B., & Chen, H. H. (2018). Trading Decision of Taiwan Stocks with the Help of United States Stock Market. Procedia Computer Science, 126, 87-96.
13.	J. Contreras, R. Espinola, F. J. Nogales and A. J. Conejo, "ARIMA models to predict next-day electricity prices," in IEEE Transactions on Power Systems, vol. 18, no. 3, pp. 1014-1020, Aug. 2003, doi: 10.1109/TPWRS.2002.804943.
14.	Lee, B. S., Rui, O. M., & Wang, S. S. (2004). Information transmission between the NASDAQ and Asian second board markets. Journal of Banking & Finance, 28(7), 1637-1670.
15.	Liao, S.H., Chou,S.Y(2013). Data mining investigation of co-movements on the Taiwan and China stock markets for future investment portfolio, Expert Systems with Applications, 40,1542–1554.
16.	Mehtab, S., & Sen, J. (2020, November). Stock price prediction using CNN and LSTM-based deep learning models. In 2020 International Conference on Decision Aid Sciences and Application (DASA) (pp. 447-453). IEEE.
17.	Moghaddam, A. H., Moghaddam, M. H., & Esfandyari, M. (2016). Stock market index prediction using artiﬁcial neural network. Journal of Economics, Finance750and Administrative Science,21, 89–93.
18.	Niu, H., Xu, K., & Wang, W. (2020). A hybrid stock price index forecasting model based on variational mode decomposition and LSTM network. Applied Intelligence, 50(12), 4296-4309.
19.	S. Selvin, R. Vinayakumar, E. A. Gopalakrishnan, V. K. Menon,& K. P. Soman(2017), Stock price prediction using LSTM, RNN and CNN-sliding window model, International Conference on Advances in Computing, Communications and Informatics (ICACCI), 1643-1647.
20.	S. Tiwari, A. Bharadwaj and S. Gupta, "Stock price prediction using data analytics," 2017 International Conference on Advances in Computing, Communication and Control (ICAC3), 2017, pp. 1-5, doi: 10.1109/ICAC3.2017.8318783.
21.	Sima Siami-Namini, Akbar Siami Namin (2018). Forecasting economics and financial time series: ARIMA vs. LSTM.
22.	Uma Gurav,Nandini Sidnal(2018), Predict Stock Market Behavior: Role of Machine Learning Algorithms, Intelligent Computing and Information and Communication,383-394.
23.	Vinay Kumar Reddy Chimmula, &Lei Zhang(2020),Time series forecasting of COVID-19 transmission in Canada using LSTM networks,Chaos, Solitons & Fractals,135,109864.
24.	Wang, GJ., Xie, C. & Stanley(2018), H.E. Correlation Structure and Evolution of World Stock Markets: Evidence from Pearson and Partial Correlation-Based Networks. Comput Econ,51, 607–635. 
25.	Wanjawa, B. W., & Muchemi, L. (2014). Ann model to predict stock prices at stock exchange markets. arXiv preprint arXiv:1502.06434.
26.	Zhuge, Q., Xu, L., & Zhang, G. (2017). LSTM Neural Network with Emotional Analysis for prediction of stock price. Engineering letters, 25(2).




