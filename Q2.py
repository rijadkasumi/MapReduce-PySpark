# Import the necessary libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Starting the spark session HetioNet
spark = SparkSession.builder.appName("HetioNet").getOrCreate()

# Reading the tsv files
edges = spark.read.format("csv").option("delimiter", "\t").option("header", "true").load("/Users/rijadkasumi/Desktop/data/edges.tsv")
nodes = spark.read.format("csv").option("delimiter", "\t").option("header", "true").load("/Users/rijadkasumi/Desktop/data/nodes.tsv")

### Question 2 : 
# Compute the number of diseases associated with 1, 2, 3, ..., n drugs. 
# Output results with the top 5 number of diseases in a descending order.

# Filtering edges for Compound-Disease relationships
compound_disease = edges.filter(edges.source.startswith('Compound') & edges.target.startswith('Disease'))

# Converting DataFrame to RDD and map to create key-value pairs of the form (drug, disease)
compound_disease_pairs = compound_disease.rdd.map(lambda x: (x.source, x.target))

# Grouping by drug to collect all associated diseases into a list
# Groups all diseaeses associated with each compound into a list
# list return the list of elements in its argumnet
grouped_comp_diseases = compound_disease_pairs.groupByKey().mapValues(list)

# Counting the number of diseases for each drug and convert the result to a DataFrame Diseases column contains the number of diseases assosiated with each compound
comp_disease_count = grouped_comp_diseases.map(lambda x: (len(x[1]),)).toDF(['Diseases'])


# Grouping by the number of diseases to count the number of drugs for each count
# agg to count the number of rows for each in group(number of drugs that are associated with each number of diseases)
# *  count the number of compounds with the same number of associated diseases
# and then rename the column Drugs and Diseases

disease_comp_count = comp_disease_count.groupBy('Diseases').agg({"*": "count"}).withColumnRenamed("count(1)", "Drugs")

# Converting the results to integers and sort by drug count in descending order
disease_comp_count = disease_comp_count.withColumn('Drugs', disease_comp_count['Drugs'].cast("int")) \
    .withColumn('Diseases', disease_comp_count['Diseases'].cast("int")) \
    .sort('Diseases', ascending=True)

# Renaming the columns
disease_comp_count = disease_comp_count.select(col("Diseases").alias("Drugs"), col("Drugs").alias("Diseases"))

# Outputting the results
disease_comp_count.show(5)

