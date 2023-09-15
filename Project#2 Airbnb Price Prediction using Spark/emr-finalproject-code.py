#!/usr/bin/env python
# coding: utf-8

# In[9]:


from pyspark.sql import SparkSession
from pyspark.sql.utils import AnalysisException
import pyspark.sql.functions as F
import pyspark.sql.types as T
from IPython.display import display
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from pyspark.ml.feature import ChiSqSelector
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler, OneHotEncoder, ChiSqSelector, VectorIndexer
from pyspark.ml import Pipeline
from pyspark.ml.linalg import  Vectors
from pyspark.sql import DataFrame
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
import pandas as pd
from xgboost.spark import SparkXGBRegressor
from pyspark.ml.regression import DecisionTreeRegressor
import time
import os


# In[11]:


S3_DATA_SOURCE_PATH = 's3://bia678-finalproject/data-source/Finaldataairbnb2.csv'


# In[12]:


S3_DATA_OUTPUT_PATH = 's3://bia678-finalproject/data-output/resultsdb.csv'


# In[13]:


spark = SparkSession.builder.getOrCreate()
airbnbdata = spark.read.csv(S3_DATA_SOURCE_PATH,
                sep=",",
                inferSchema=True,
                header=True
)
 
print(airbnbdata.count(), len(airbnbdata.columns))


# In[ ]:


for x in airbnbdata.columns:
    airbnbdata.select(x).summary().show()
    break


# In[ ]:


airbnbdata.printSchema()


# In[ ]:


airbnbdata[['host_response_time', 'bed_type', 'room_type','price','host_response_rate']].show()


# In[ ]:


# convert PySpark DataFrame to Pandas DataFrame
pandas_df = airbnbdata.toPandas()

# create a pairplot using seaborn
sns.pairplot(pandas_df)


# In[ ]:


plt.figure(figsize=(10,10))
sns.scatterplot(x="room_type", y="price", data=airbnbdata.toPandas())
plt.title("Room Type vs Price",size=15, weight='bold')


# In[ ]:


plt.figure(figsize=(10,10))
sns.scatterplot(x="bed_type", y="price", data=airbnbdata.toPandas())
plt.title("Bed Type vs Price",size=15, weight='bold')


# In[ ]:


plt.figure(figsize=(10,10))
sns.scatterplot(x="host_has_profile_pic", y="price", data=airbnbdata[['host_has_profile_pic','price']].toPandas())
plt.title("host has profile pic vs price",size=15, weight='bold')


# In[ ]:


plt.figure(figsize=(10,10))
sns.distplot(airbnbdata.select("price").toPandas())
plt.title("Price Distribution Plot",size=15, weight='bold')


# In[ ]:


airbnbdata = airbnbdata.withColumn("log_price", F.log("price"))
airbnbdata[['log_price']].show()


# In[ ]:


plt.figure(figsize=(12,10))
sns.distplot(airbnbdata.select("log_price").toPandas()+1)
plt.title("Log-Price Distribution Plot",size=15, weight='bold')


# In[ ]:


plt.figure(figsize=(7,7))
log_price = airbnbdata.select("log_price").toPandas()["log_price"]
stats.probplot(log_price, plot=sns.mpl.pyplot)
plt.show()


# In[ ]:


plt.figure(figsize=(15,12))
# Create a correlation matrix using the Pearson method
corr = airbnbdata.select(airbnbdata.columns).toPandas().corr(method="pearson")

# Create a heatmap using the correlation matrix
sns.heatmap(corr, annot=True)
plt.title("Correlation Matrix",size=15, weight='bold')


# In[ ]:


airbnbmodel= airbnbdata.drop("host_listings_count")
#airbnb_model_x = airbnbmodel.select(*(F.col(airbnbmodel.columns[i]).alias(airbnbmodel.columns[i]) for i in range(len(airbnbmodel.columns)-1)))
#airbnb_model_y = airbnbmodel.select(F.col(airbnbmodel.columns[-1]).alias(airbnbmodel.columns[-1]))

# Print the DataFrames
#airbnb_model_x.show()
#airbnb_model_y.count()


# In[ ]:


airbnbmodel = airbnbmodel.na.drop()


# In[ ]:


airbnbmodel.dtypes


# In[ ]:


catCols=[x for (x, dataType) in airbnbmodel.dtypes if dataType=="string"]
numCols=[x for (x, dataType) in airbnbmodel.dtypes if ((dataType=="double")&(x != "price")&(x != "log_price"))]
print(airbnbmodel[numCols])


# In[ ]:


coef_var=['host_response_rate','host_total_listings_count','accommodates','bathrooms','bedrooms','beds','guests_included']
vecort_assem=VectorAssembler(inputCols=numCols,outputCol="features")
output=vecort_assem.transform(airbnbmodel)
output.select("features").show(10, truncate=False)
output.count()
final_df=output.select('features','log_price')


# In[ ]:


scaler =StandardScaler(inputCol='features', outputCol='scaled_feat',withStd=True,withMean=False)
scaled_model=scaler.fit(final_df)
cluster_df=scaled_model.transform(final_df)


# In[ ]:


cluster_df.show()


# In[ ]:


#train_data, test_data = final_df.randomSplit([0.7,0.3])
train_data, test_data = cluster_df.randomSplit([0.7,0.3])

# Split the data into four subsets with sizes 20%, 40%, 60%, and 80%
train_data_split = train_data.randomSplit([0.2, 0.4, 0.6, 0.8])
    
train_data_split_0 =train_data_split[0]   
train_data_split_1 =train_data_split[1]   
train_data_split_2 =train_data_split[2]   
train_data_split_3 =train_data_split[3]   

test_data_split = test_data.randomSplit([0.2, 0.4, 0.6, 0.8])

test_data_split_0 =test_data_split[0]   
test_data_split_1 =test_data_split[1]   
test_data_split_2 =test_data_split[2]   
test_data_split_3 =test_data_split[3] 


# In[ ]:


#linear Regression
def linear_regression(train_data, test_data):
    startlr = time.perf_counter()
    lr = LinearRegression(featuresCol = 'scaled_feat', labelCol='log_price', maxIter=10, regParam=0.01, elasticNetParam=0.01) 
    lr_model = lr.fit(train_data)

    trainingSummary = lr_model.summary


    lr_predictions = lr_model.transform(test_data)
    #lr_predictions.select("prediction","log_price","scaled_feat").show(5)

    lr_evaluator = RegressionEvaluator(predictionCol="prediction",                      labelCol="log_price",metricName="rmse")
    #print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
    #print("MAE: %f" % trainingSummary.meanAbsoluteError)
    rmseLinear=lr_evaluator.evaluate(lr_predictions)
    print("RMSE   on val data = %g" % rmseLinear)
   
    endlr = time.perf_counter()
    exectimelr=endlr - startlr 
    print(f"Execution time: {exectimelr} seconds")
    return (exectimelr, rmseLinear)


# In[ ]:


# random forest
def random_forestregressor(train_data, test_data):
    startrf = time.perf_counter()
    rf = RandomForestRegressor(featuresCol = 'scaled_feat', labelCol='log_price', 
                           maxDepth=13, 
                           minInstancesPerNode=1,
                           bootstrap=True
                          )
    rf_model = rf.fit(train_data)
    rf_predictions = rf_model.transform(test_data)
   # rf_predictions.select("prediction","log_price","scaled_feat").show(5)



    rf_evaluator = RegressionEvaluator(predictionCol="prediction",                  labelCol="log_price",metricName="rmse")
    rf_rmse=rf_evaluator.evaluate(rf_predictions)
    print("RMSE Squared (R2) on val data = %g" % rf_rmse)
    endrf=time.perf_counter()
    exectimerf=endrf - startrf 
    print(f"Execution time: {exectimerf} seconds")
    return (exectimerf, rf_rmse)


# In[ ]:


# Create a Decision Tree Regressor
def decisiontree_regressor(train_data, test_data):
    startdt = time.perf_counter()
    dt_regressor = DecisionTreeRegressor(featuresCol="scaled_feat", labelCol="log_price", maxDepth=15)

    # Fit the model on the training data
    dt_regressor_model = dt_regressor.fit(train_data)

    # Make predictions on the test data
    dt_regressor_predictions = dt_regressor_model.transform(test_data)
    #dt_regressor_predictions.select("prediction","log_price","scaled_feat").show(5)
    # Evaluate the model's performance on the test data
    dt_regressor_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="log_price", metricName="rmse")
    rmse_dt = dt_regressor_evaluator.evaluate(dt_regressor_predictions)
    print("rmse:", rmse_dt)
    enddt=time.perf_counter()
    exectimedt=enddt - startdt
    return (exectimedt, rmse_dt)


# In[ ]:


#xgboost
def xgboost_regressor(train_data, test_data):
    startxgb = time.perf_counter()
    xgb_regressor = SparkXGBRegressor(num_workers=3, label_col="log_price",  features_col="scaled_feat", max_depth=15)
    xgb_regressor_model = xgb_regressor.fit(train_data)
    xgb_predictions  = xgb_regressor_model.transform(test_data)
    #xgb_predictions.select("prediction","log_price","scaled_feat").show(5)
    xgb_evaluator = RegressionEvaluator(predictionCol="prediction",                      labelCol="log_price",metricName="rmse")
    rmse_xgb= xgb_evaluator.evaluate(xgb_predictions)
    
    # Evaluate the model on the test data
    print("RMSE on test data = %g" % rmse_xgb)
    endxgb=time.perf_counter()
    exectimexgb=endxgb-startxgb
    return (exectimexgb, rmse_xgb)


# In[ ]:


# 20% data
result =[]
lrexetime_2, lrrmse_2= linear_regression(train_data_split_0,test_data_split_0)
result.append({ "execution_time": lrexetime_2, "rmse": lrrmse_2, "Model": "lr", "data": 0.2 })
rfexetime_2, rfrmse_2=random_forestregressor(train_data_split_0,test_data_split_0)
result.append({ "execution_time": rfexetime_2, "rmse": rfrmse_2, "Model": "rf", "data": 0.2 })
dtexetime_2, dtrmse_2=decisiontree_regressor(train_data_split_0,test_data_split_0)
result.append({ "execution_time": dtexetime_2, "rmse": dtrmse_2, "Model": "dt", "data": 0.2 })
xgbexetime_2, xgbrmse_2=xgboost_regressor(train_data_split_0,test_data_split_0)
result.append({ "execution_time": xgbexetime_2, "rmse": xgbrmse_2, "Model": "xgb", "data": 0.2 })


# In[ ]:


# 40% data
lrexetime_4, lrrmse_4= linear_regression(train_data_split_1,test_data_split_1)
result.append({ "execution_time": lrexetime_4, "rmse": lrrmse_4, "Model": "lr", "data": 0.4 })

rfexetime_4, rfrmse_4=random_forestregressor(train_data_split_1,test_data_split_1)
result.append({ "execution_time": rfexetime_4, "rmse": rfrmse_4, "Model": "rf", "data": 0.4 })

dtexetime_4, dtrmse_4=decisiontree_regressor(train_data_split_1,test_data_split_1)
result.append({ "execution_time": dtexetime_4, "rmse": dtrmse_4, "Model": "dt", "data": 0.4 })

xgbexetime_4, xgbrmse_4=xgboost_regressor(train_data_split_1,test_data_split_1)
result.append({ "execution_time": xgbexetime_4, "rmse": xgbrmse_4, "Model": "xgb", "data": 0.4 })


# In[ ]:


# 60% data

lrexetime_6, lrrmse_6 =linear_regression(train_data_split_2,test_data_split_2)
result.append({ "execution_time": lrexetime_6, "rmse": lrrmse_6, "Model": "lr", "data": 0.6 })

rfexetime_6, rfrmse_6=random_forestregressor(train_data_split_2,test_data_split_2)
result.append({ "execution_time": rfexetime_6, "rmse": rfrmse_6, "Model": "rf", "data": 0.6 })

dtexetime_6, dtrmse_6=decisiontree_regressor(train_data_split_2,test_data_split_2)
result.append({ "execution_time": dtexetime_6, "rmse": dtrmse_6, "Model": "dt", "data": 0.6 })

xgbexetime_6, xgbrmse_6 = xgboost_regressor(train_data_split_2,test_data_split_2)
result.append({ "execution_time": xgbexetime_6, "rmse": xgbrmse_6, "Model": "xgb", "data": 0.6 })


# In[ ]:


# 80% data
lrexetime_8, lrrmse_8 =linear_regression(train_data_split_3,test_data_split_3)
result.append({ "execution_time": lrexetime_8, "rmse": lrrmse_8, "Model": "lr", "data": 0.8 })

rfexetime_8, rfrmse_8=random_forestregressor(train_data_split_3,test_data_split_3)
result.append({ "execution_time": rfexetime_8, "rmse": rfrmse_8, "Model": "rf", "data": 0.8 })

dtexetime_8, dtrmse_8=decisiontree_regressor(train_data_split_3,test_data_split_3)
result.append({ "execution_time": dtexetime_8, "rmse": dtrmse_8, "Model": "dt", "data": 0.8 })

xgbexetime_8, xgbrmse_8 = xgboost_regressor(train_data_split_3,test_data_split_3)
result.append({ "execution_time": xgbexetime_8, "rmse": xgbrmse_8, "Model": "xgb", "data": 0.8 })


# In[ ]:


# 100% data
lrexetime, lrrmse =linear_regression(train_data,test_data)
result.append({ "execution_time": lrexetime, "rmse": lrrmse, "Model": "lr", "data": 1.0 })

rfexetime, rfrmse=random_forestregressor(train_data,test_data)
result.append({ "execution_time": rfexetime, "rmse": rfrmse, "Model": "rf", "data": 1.0 })

dtexetime, dtrmse=decisiontree_regressor(train_data,test_data)
result.append({ "execution_time": dtexetime, "rmse": dtrmse, "Model": "dt", "data": 1.0 })

xgbexetime, xgbrmse = xgboost_regressor(train_data,test_data)
result.append({ "execution_time": xgbexetime, "rmse": xgbrmse, "Model": "xgb", "data": 1.0 })


# In[ ]:


resultstab = spark.createDataFrame(result)


# In[ ]:


resultstab.show()
resultstab.write.mode('overwrite').csv("S3_DATA_OUTPUT_PATH")

