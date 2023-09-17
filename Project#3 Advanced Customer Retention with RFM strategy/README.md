# Advanced Customer Analytics: Segmentation, Customer Churn, and Predictive Marketing Model

## Introduction
Enhancing customer engagement and retention in the competitive retail industry is crucial for businesses to drive revenue growth and bolster customer loyalty. By leveraging advanced customer analytics, including segmentation, churn prediction, and predictive marketing models, retail companies can pinpoint high-risk customers, comprehend their specific needs and preferences, and devise targeted retention strategies to mitigate churn. Through predictive modeling, businesses can accurately anticipate customer responses to future marketing campaigns, thereby enhancing overall customer engagement and loyalty.

## Research Questions
- How can we segment customers based on their demographic, socioeconomic, and purchasing behavior attributes?
- Which customers are at high and low risk of churning?
- How can we predict the response of future marketing campaigns?

## Dependencies
* pandas
* NumPy
* matplotlib
* seaborn
* sklearn
* imbalanced-learn
* xgboost

## Data Exploration

### Exploratory Data Analysis
- Key determinants of customer spending include income, education level, marital status, and segmentation based on demographics and behavior.
- The average spending range was $0-200 per product.

## Models and Results

### Clustering
- Utilized K-means clustering to segment data based on demographics and behavioral values.
- Constructed two K-means Models: one using "Income" and "Total Amount Spent" features, and another using "Age", "Income", and "Total Amount Spent".

### Segmentation and Customer Churn
- RFM analysis was employed to identify customer segments based on Recency, Frequency, and Monetary value, resulting in 9 distinct parts.
- Logistic regression was used for churn prediction, identifying high-risk segments like "at_Risk" with a predicted churn rate of 60%.
- By integrating RFM analysis and logistic regression, targeted retention strategies can be designed for high-risk segments, enhancing customer retention and reducing churn.

### Predictive Marketing
- Various machine learning models, including logistic regression, SMOTE, XGBoost, and Random Forest, were applied to predict customers' responses to marketing campaigns based on past campaign response rates.
- Logistic regression exhibited the highest accuracy and weighted F1 score, while random forest showcased the highest precision for the positive class.

## Conclusion
By conducting exploratory data analysis and segmenting customers using K-means clustering and RFM analysis, businesses can identify valuable customers and prevent churn. Predictive models such as logistic regression, SMOTE, XGBoost, and Random Forest were trained to enhance campaign response rates. These insights can aid businesses in optimizing marketing and retention strategies, elevating customer satisfaction, and boosting profits.

## Acknowledgments
- Special thanks to our instructor, Chris Asakeiwicz, for his guidance and support throughout the project.
