# Apple-Stock-Move-Prediction

## Goal
To provide a daily buy/sell recommendation for Apple, using both traditional
and alternative data as features in the ML model and utilizing various
classification and ensemble learning techniques.

## Challenges
Some of the biggest challenges we have run into in this study include:
- Time Series split: since market data are mostly time series data, common practice of random splitting is not practical. We had to use expanding window through time to split the data for cross validation.
- Future-bias: it is extremely common to accidentally use future data, which introduces peek-ahead bias in the result. For example, standardization for both train and test needs to be done with mean and standard deviation from training dataset only. Also, when we convert buy/sell signal into trading, we should only look at testing set because we trained the model with all the training data and it would be classified as using future information if we trade throughout training time period. In addition, when we used LDA to do feature selection, we have also accidentally used future data which increased our accuracy artificially, as we fit and transformed the entire dataset to its most discriminative directions instead of just using training data.
- Data Frequency: For macro economical data and quarterly earnings’ report, we struggled to fit into the daily prediction scheme. To extrapolate quarterly data into daily data, it turned out to introduce more noise than useful information.
- Feature Selection: as shown in the previous section, multiple features are highly correlated and multiple feature selection algorithms have been tried out. We originally used LDA to convert all features into one single input, but soon we figured out that we were using the entire dataset (including test dataset), introducing future-bias into the algorithm.
- Model Selection: Although XGBoost resulted in the highest test accuracy, it did not do very well on the cross validation dataset. Thus, we wanted a more stable model that did well both on cross-validation as well as test.

## Summary
1. 50+ raw features are first collected with web-scraping, then cleaned by removing noises and normalizing the data, later engineered to have consistent frequencies and range. Out of these raw features, a small group of features are selected by leveraging Linear Discriminant Analysis, Recursive Feature Elimination, and PCA.
2. Trained, tuned, and tested various classification models including Logistic regression, SVM, Random forest, LDA, XGBOOST, and NeuralNets.
3. Based on the selected evaluation metrics including F1 score and AUC, the voting ensemble model, composed of Decision Trees, NeuralNets, and XGBOOST with RFE feature reduction, was proven to have the best out of sample model performance.
