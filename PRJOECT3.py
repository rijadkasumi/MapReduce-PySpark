from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline
from pyspark.mllib.evaluation import MulticlassMetrics

# Start a spark session
spark = SparkSession.builder.appName('cancer11_diagnosis').getOrCreate()

# Load data 
df = spark.read.csv('/Users/rijadkasumi/Desktop/project3/project3_data.csv', header = True, inferSchema = True)

# Convert the diagnosis column from categorical to numerical because of B Bening and M Malignant
indexer = StringIndexer(inputCol="diagnosis", outputCol="label")
df = indexer.fit(df).transform(df)

# Select the features
# Transformer -  It transforms the data by collecting multiple columns of your DataFrame and putting them into a single vector column. 
# This is necessary because each machine learning algorithm in Spark MLlib requires the input data to be in this format.
assembler = VectorAssembler(
    inputCols=['Radius_mean', 'Texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
               'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
               'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
               'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se',
               'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
               'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
               'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'],
    outputCol="features")

# Split the data into training, validation and testing sets
# Fist split 60% train, then rest 40% temp and then split the temp into Validation 20 and test 20
#train_data, temp_data = df.randomSplit([0.6, 0.4], seed = 2023): The randomSplit function is used to randomly split the data into two subsets. The arguments [0.6, 0.4] denote that 60% of the data should go into the first subset (train_data) and 40% should go into the second subset (temp_data).
# The seed parameter ensures that the random split is reproducible. In other words, if you run the code multiple times, the same split will be generated because the randomness is initiated with the same starting point (i.e., seed).
# This line of code takes the temp_data subset from the previous split and splits it further into two equal subsets: validation_data and test_data. Each of these subsets now constitutes 50% of the temp_data or effectively, 20% of the original data.
# the seed is important thing is to use the same seed value if you want to reproduce the same splits.
#A lso, it is often good practice to report the seed value you used when you are publishing results that involved random processes, so that others can reproduce your results exactly. 
# It adds transparency to your work.
train_data, temp_data = df.randomSplit([0.6, 0.4], seed = 2023)
validation_data, test_data = temp_data.randomSplit([0.5, 0.5], seed = 2023)


# LogisticRegression and RandomForestClassifier are Estimators. 

# Define the logistic regression model using MLlib
# Limitin 10 iterations
lr = LogisticRegression(maxIter=10)

# Define the random forest model using MLlib
rf = RandomForestClassifier()

# ParamGrid for Cross Validation
# Logistic Regression Parameter Grid
# Regularization parameters to prevent overliffting by adding penalty to the loss function.
# that the grid will include two possible values for the regParam parameter: 0.1 and 0.01.
# fitIntercept Specifies whether to include an intercept (a.k.a. bias) term in the logistic regression model.
#  If False, the model will not have an intercept term. If True, it will
#  In the provided code, 0.1 and 0.01 are chosen as possible values, 
#  which means the model will try both a small amount and a moderate amount of regularization to see which works best.
paramGrid_lr = ParamGridBuilder()\
    .addGrid(lr.regParam, [0.1, 0.01]) \
    .addGrid(lr.fitIntercept, [False, True])\
    .build()
# Random Forest Parameter Grid 
# numTrees 10,30 specifies the number of trees in the RF model- possible values of 10 and 30
# it's common to start with a lower number of trees and gradually increase it to see how performance improves
# maxdepth the depth of each decision tree
# Typical starting values can range between 5 and 30, depending on the complexity of the problem and the amount of data available
paramGrid_rf = ParamGridBuilder() \
    .addGrid(rf.numTrees, [10, 30]) \
    .addGrid(rf.maxDepth, [5, 10]) \
    .build()


# Define the pipeline
# In Spark MLlib, a Pipeline is a sequence of data processing stages
#The VectorAssembler is a Transformer. It transforms the data by collecting multiple columns of your DataFrame and putting them into a single vector column.
# This is necessary because each machine learning algorithm in Spark MLlib requires the input data to be in this format.
# LogisticRegression and RandomForestClassifier are Estimators.
# An Estimator abstracts the concept of a learning algorithm or any algorithm that fits or trains on data. 
# They implement a method fit(), which accepts a DataFrame and produces a Model, which is a Transformer.
#assembler and lr (or rf) are bundled together into a Pipeline for logistic regression and random forest respectively. When the pipeline's fit() method is called for training, it:
#Executes the VectorAssembler's transform() method to create a new DataFrame with the features vector column,
#Then calls the fit() method of LogisticRegression or RandomForestClassifier (depending on the pipeline) on this new DataFrame.
#This results in a model that can be used for making predictions. The model itself is a PipelineModel, which is a Transformer. When its transform() method is called to make predictions, it:
#Executes the VectorAssembler's transform() method on the input DataFrame to create a new DataFrame with the features vector column,
#Then calls the transform() method of the logistic regression or random forest model on this new DataFrame.
#The use of a pipeline simplifies the code and makes it easier to understand the sequence of data processing and machine learning tasks. It also makes it easier to experiment with different sequences and combinations of data processing and machine learning stages.

pipeline_lr = Pipeline(stages=[assembler, lr])
pipeline_rf = Pipeline(stages=[assembler, rf])

# Cross validation

# estimator: This is the machine learning model you want to train. This can be a standalone estimator or a pipeline with various stages.
# Here we're using the pipeline defined for the Logistic Regression model (pipeline_lr) and the Random Forest model (pipeline_rf).
#The CrossValidator is a method used for model selection and hyperparameter tuning. It partitions the input dataset into a set of "folds". For each of these folds, it trains the model on data from the remaining folds, and then evaluates the model's performance on the held-out fold.
# It repeats this process for each fold, so that every data point in the dataset is held out exactly once
# estimatorParamMaps: This is a list of hyperparameters and their respective values that you want to tune for the estimator.
#  For each combination of hyperparameters, CrossValidator will train the estimator and evaluate its performance. The best performing model (as determined by the evaluator) is selected.
#  We're passing the hyperparameter grid that was defined earlier for each model (paramGrid_lr and paramGrid_rf).
# evaluator: This is the metric used to measure the performance of the model on the validation set.
# We're using BinaryClassificationEvaluator(), which defaults to measuring the area under the ROC curve.
# numFolds: This is the number of folds used in the cross-validation process, as explained above.
# numFolds=5), so the dataset will be split into 5 parts, and the process will be repeated 5 times.
crossval_lr = CrossValidator(estimator=pipeline_lr,
                             estimatorParamMaps=paramGrid_lr,
                             evaluator=BinaryClassificationEvaluator(),
                             numFolds=5)

crossval_rf = CrossValidator(estimator=pipeline_rf,
                             estimatorParamMaps=paramGrid_rf,
                             evaluator=BinaryClassificationEvaluator(),
                             numFolds=5)

# Fit the models
# cvModel_lr = crossval_lr.fit(train_data): This line of code is fitting the Logistic Regression model on the training data using the cross-validation process defined earlier.
# The fit() function is used to train the model. crossval_lr is the cross-validator object that was defined in the previous step, and train_data is the training dataset. 
# The output of this function, cvModel_lr, is the best Logistic Regression model found by cross-validation.
cvModel_lr = crossval_lr.fit(train_data)
cvModel_rf = crossval_rf.fit(train_data)

# Predict on validation data
# prediction_lr_val = cvModel_lr.transform(validation_data): Here, the trained Logistic Regression model (cvModel_lr) is being used to make predictions on the validation dataset.
# The transform() function takes the input data, applies the model to it, and outputs a dataframe that includes a column of predicted labels.
prediction_lr_val = cvModel_lr.transform(validation_data)
prediction_rf_val = cvModel_rf.transform(validation_data)

# So, after these lines of code are run, you have two dataframes: prediction_lr_val and prediction_rf_val.
# Each dataframe includes the original validation data, along with a column of predicted labels from the Logistic Regression model and Random Forest model respectively. 


# Evaluate the models on validation data
# evaluator = BinaryClassificationEvaluator(): This line of code creates an evaluator object that can be used to evaluate binary classification models.
# By default, the metric used is area under the ROC curve, which is a common metric for evaluating the performance of binary classifiers.
evaluator = BinaryClassificationEvaluator()
print("Validation Area Under ROC Logistic Regression: " + str(evaluator.evaluate(prediction_lr_val, {evaluator.metricName: "areaUnderROC"})))
print("Validation Area Under ROC Random Forest: " + str(evaluator.evaluate(prediction_rf_val, {evaluator.metricName: "areaUnderROC"})))

# Predict on test data
#When its transform() method is called to make predictions, it:
#Executes the VectorAssembler's transform() method on the input DataFrame to create a new DataFrame with the features vector column,
# Then calls the transform() method of the logistic regression or random forest model on this new DataFrame.
prediction_lr_test = cvModel_lr.transform(test_data)
prediction_rf_test = cvModel_rf.transform(test_data)

# Evaluate the models on test data
print("Test Area Under ROC Logistic Regression: " + str(evaluator.evaluate(prediction_lr_test, {evaluator.metricName: "areaUnderROC"})))
print("Test Area Under ROC Random Forest: " + str(evaluator.evaluate(prediction_rf_test, {evaluator.metricName: "areaUnderROC"})))

# Calculate precision, recall and F1 score
def print_metrics(predictions):
    predictions_and_labels = predictions.select(['prediction','label']).rdd
    metrics = MulticlassMetrics(predictions_and_labels)
    labels = predictions_and_labels.map(lambda lp: lp.label).distinct().collect()
    for label in sorted(labels):
        original_label = 'Malignant' if label else 'Benign'
        print("Class %s (numeric label: %s) precision = %s" % (original_label, label, metrics.precision(label)))
        print("Class %s (numeric label: %s) recall = %s" % (original_label, label, metrics.recall(label)))
        print("Class %s (numeric label: %s) F1 Measure = %s" % (original_label, label, metrics.fMeasure(label, beta=1.0)))


print_metrics(prediction_lr_test)
print_metrics(prediction_rf_test)


#Logistic Regression:
#Logistic Regression is a common algorithm for binary classification problems. 
#The output of a logistic regression model is a probability that the given input point belongs to a certain class. 
#This is particularly suitable for problems like yours, where the task is to predict one of two classes: benign or malignant. 
#The simplicity of the Logistic Regression model, its interpretability, and its fast training times make it a good first choice for a binary classification problem.
#Random Forest:
#The Random Forest algorithm is a type of ensemble method that combines multiple decision tree models to improve predictive performance.
#Random Forests can capture complex patterns in the data by building a large number of decision trees and averaging their predictions. 
#They can handle both numerical and categorical data, and they are less prone to overfitting than individual decision trees.
#Furthermore, they offer feature importance metrics which could be beneficial for understanding what features are most influential in predicting the class of a tumor.