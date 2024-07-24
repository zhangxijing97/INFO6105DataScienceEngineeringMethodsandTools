import pyspark.sql as spl
from pyspark.ml.classification import LinearSVC
from pyspark.ml.feature import VectorAssembler


# Spark Session
spark = spl.SparkSession.builder.appName('SparkLecturer').getOrCreate()

#Input Dateset
org_df = spark.read.csv("high_income.csv", header=True, inferSchema=True)

# Converting features into a single column
data_df = VectorAssembler(inputCols=['Experience', 'Hours', 'Age'], outputCol="features").transform(org_df)

# Split Train 80% and Test 20%
train_df, test_df = data_df.randomSplit(weights=[0.8,0.2], seed=200)

#SVM model
svm = LinearSVC()
model = svm.fit(train_df)

#Accuracy
accuracy = model.evaluate(test_df).accuracy
print("Test accuracy:  ", accuracy)
