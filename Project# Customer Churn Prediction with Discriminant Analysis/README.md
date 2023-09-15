# Customer Churn Prediction Model Using Discriminant Analysis

## Project Summary:

### 1. Problem Overview and Motivation:
Banks need to predict customer churn to maintain healthy business and customer relationships. This model focuses on predicting whether a customer will leave or stay with a bank based on their data. Key motivations include reducing negative brand feedback, retaining existing loyal customers, and saving on the higher cost of acquiring new ones.

### 2. The Optimization Model:
Using discriminant analysis, we've built a model to predict customer retention or churn. The model parameters include objective function, decision variables, constraints, and inputs like Credit score, country, gender, age, tenure, balance, number of products, credit card status, active membership, and estimated salary.

### 3. Model Input Parameter Estimation and Data Requirement:
The model leverages data from 10,000 bank customers. It focuses on parameters such as data source, variables, and instances. Some key attributes include credit score, gender, age, tenure, balance, product number, and estimated salary.

### 4. The Optimization Solution Method:
We implemented the Evolutionary solver method for optimization, which uses genetic algorithms. This approach can provide near-optimal solutions for challenging problems.

### 5. Optimal Solution Structure and Insights:
The obtained solution suggests that there are multiple weight and cutoff values that can achieve approximately 79.71% correct classification rate. The classification heavily relies on the balance attribute, suggesting that individuals with higher balances are often classified as loyal customers.

### 6. Conclusions, Model Limitations, and Extensions:
The data, which comprised of 10,000 customers, indicated that post optimization, nearly 80% of the data was classified correctly. There were, however, some discrepancies as with any predictive model.

## Skills Highlighted:

- **Data Analysis and Interpretation:** Extracted meaningful insights from raw data of 10,000 bank customers.
- **Statistical Analysis:** Employed discriminant analysis as the primary method for prediction.
- **Optimization:** Implemented the Evolutionary solver method using genetic algorithms.
- **Data Visualization:** Used charts and matrices to better understand and represent data patterns and correlations.
- **Collaboration:** Worked effectively as a team, dividing tasks and consolidating research for a cohesive project report.

## Conclusion:

The model provides a near-accurate prediction of customer churn, proving its potential utility for the banking sector. However, like all models, it has limitations. While it classifies about 80% of the data correctly, discrepancies remain. The model heavily relies on a customer's balance, indicating room for improvement in its predictive power across other attributes. Future iterations might consider integrating more sophisticated algorithms or integrating more data sources for improved accuracy.

### References:
- [Bank Marketing Campaign](https://www.neuraldesigner.com/learning/examples/bank-marketing-campaign)
- [Credit Risk Management](https://www.neuraldesigner.com/learning/examples/credit-risk-management)
- [Churn Prevention](https://www.neuraldesigner.com/solutions/churn-prevention)
- [Customer Segmentation](https://www.neuraldesigner.com/solutions/customer-segmentation)
