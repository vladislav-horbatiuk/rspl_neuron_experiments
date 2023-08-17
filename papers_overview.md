# Обзор работ

### Hierarchical Forecasting, Rob J Hyndman, 2017 Bejing Workshop on Forecasting

 * [File](papers/3-Hierarchical.pdf).
 * Tries to set up proper problem definition and basic mathematical framework
for the case of forecasting multiple time series that may be grouped/organized
into hierarchy in multiple ways. May be worse looking for sample datasets.

### A Comparative Simulation Study of ARIMA and Computational Intelligent Techniques for Forecasting Time Series Data, H. Fawzy et al., 2022

 * [File](papers/7z4d6176219g6d.pdf).
 * Looks like a basic hybrid ARIMA+ANN approach where ANN is used to model ARIMA residuals.
Artificially simulated data is used to compare approaches.

### Time series forecasting in the artificial intelligence milieu, I. Botha, 2022

 * [File](papers/836-8959-1-PB.pdf).
 * Bullshit, 2 pages about nothing.

### Economic Forecasting using Artificial Intelligence, A. Csapai et al.

 * [File](papers/1102csapai.pdf)
 * Simply compares existing approaches - linear model, regression tree, bagging,
random forest and boosting. Nothing new.

### Grouped functional time series forecasting: An application to age-specific mortality rates, R. Hyndman et al., 2016

 * [File](papers/1609.04222.pdf)
 * Another paper dedicating to different methods for predicting grouped time
series in this case - Japanese age-specific mortality rates for 47 prefectures.
Data should be available at http://www.ipss.go.jp/p-toukei/JMD/index-en.html.

### Deep Time Series Forecasting with Shape and Temporal Criteria, V. Le Guen et al., 2022

 * [File](papers/2104.04610.pdf)
 * Looks like a pretty interesting paper that considers a task of a multi-step
trajectory prediction (i.e. we are given a beginning x[1:T] of some trajectory
and want to predict how it will behave in the future x[T+1:T+K]). It introduces
a new DILATE loss function (instead of something like MSE) that is used to
compare predicted and ground truth future trajectories and since it is also
differentiable - this loss is used for optimizing parameters of corresponding
prediction model. It also has a links to some interesting time series datasets,
in particular https://www.cs.ucr.edu/~eamonn/time_series_data_2018/.

### Stock Price Forecasting: Hybrid Model of Artificial Intelligent Methods, Chong Wu et al., 2015

 * [File](papers/3836-Article Text-31087-1-10-20150223.pdf)
 * Looks like a pretty interesting paper that considers a task of a multi-step
trajectory prediction (i.e. we are given a beginning x[1:T] of some trajectory
and want to predict how it will behave in the future x[T+1:T+K]). It introduces
a new DILATE loss function (instead of something like MSE) that is used to
compare predicted and ground truth future trajectories and since it is also
differentiable - this loss is used for optimizing parameters of corresponding
prediction model. It also has a links to some interesting time series datasets,
in particular https://www.cs.ucr.edu/~eamonn/time_series_data_2018/.


