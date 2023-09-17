# Airbnb Price Prediction using Spark

## Overview
This project aims to predict Airbnb prices using Spark. Airbnb offers a unique platform for individuals to either provide or seek short-term lodging in private residences. Recently, Rio de Janeiro has become a popular destination for Airbnb listings. Accurately predicting the appropriate price for these listings is essential for hosts to optimize their earnings and for Airbnb to ensure competitive pricing, thereby driving more bookings. This project leverages machine learning regression models to predict Airbnb room prices in Rio de Janeiro, Brazil, utilizing a dataset spanning from August 2018 to May 2020. To handle the large dataset and evaluate the scalability and computing capabilities, we leveraged the AWS infrastructure.

## Dependencies
- Apache Spark
- Databricks
- Jupyter Notebook
- AWS Services:
  - Amazon EMR (Elastic MapReduce)
  - Amazon S3
  - Amazon EC2
- Python Libraries:
  - PySpark
  - Matplotlib
  - Seaborn
  - Pandas

## Dataset
This dataset comprises a total of 740K rows, representing unique data for 65,920 listings, averaging 11 data points per listing. Initially containing 108 variables, our preprocessing steps streamlined the dataset to 400K rows and 9 columns, after rigorous cleaning and refinement.
It includes various features, such as:
- `id`: Listing ID
- `name`: Name of the listing
- `host_id`: Host ID
- `host_name`: Name of the host
- `neighbourhood_group`: Location
- `neighbourhood`: Area
- `latitude`: Latitude of the listing
- `longitude`: Longitude of the listing
- `room_type`: Type of room
- `price`: Price of the listing
- `minimum_nights`: Minimum nights of stay
- `number_of_reviews`: Number of reviews
- `last_review`: Last review date
- `reviews_per_month`: Reviews per month
- `calculated_host_listings_count`: Number of listings by the host
- `availability_365`: Number of days the listing is available in a year

## AWS Infrastructure and Pipeline
To manage the big dataset and ensure efficient processing, we utilized the AWS infrastructure. The pipeline was set up as follows:

1. **Amazon S3**: Used as the primary storage for our dataset. Raw data was ingested into S3 buckets, and processed data was stored back into S3.
2. **Amazon EMR**: Leveraged for distributed data processing using Spark. EMR clusters were set up to read data from S3, process it, and write the results back to S3.
3. **Amazon EC2**: Used for hosting and running auxiliary services and applications related to the project.

This pipeline ensured that the data processing was scalable, fault-tolerant, and efficient.

## Methodology
1. **Data Cleaning**: The dataset was cleaned to remove any null values and outliers.
2. **Exploratory Data Analysis (EDA)**: EDA was performed to understand the distribution of data and the relationship between different features.
3. **Feature Engineering**: New features were created based on the existing data to improve the model's performance.
4. **Model Building**: Various machine learning models were built and evaluated based on their performance.
5. **Infrastructure Performance**: Amazon EMR consistently outperformed local processing, showcasing the benefits of parallel processing.

## Conclusions
- The Gradient Boosting Trees Regression Model stood out in terms of prediction accuracy with 0.58 score.
- However, when considering execution time, the Decision Tree Regression model is preferable with least exe time of 6.95 seconds.
- The most important features for predicting Airbnb prices are `room_type`, `neighbourhood_group`, and `availability_365`.
- The model can be further improved by incorporating more features and using advanced machine learning techniques.

## Future Work
- Incorporate more features to improve the model's accuracy.
- Explore advanced machine learning techniques for better performance.
- Optimize the AWS infrastructure for cost and performance.

## Acknowledgements
This project was completed as part of an academic assignment and utilized AWS services for scalable data processing.
