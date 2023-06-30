# Import the necessary libraries
from pyspark.sql import SparkSession
# Starting the spark session HetioNet
spark = SparkSession.builder.appName("HetioNet").getOrCreate()
# Reading the tsv files
edges = spark.read.format("csv").option("delimiter", "\t").option("header", "true").load("/Users/rijadkasumi/Desktop/data/edges.tsv")
nodes = spark.read.format("csv").option("delimiter", "\t").option("header", "true").load("/Users/rijadkasumi/Desktop/data/nodes.tsv")

# Testing if the tsv files are properly read
# edges.show(3)
# nodes.show(3)

### Question 1 :
# For each drug, compute the number of genes and the number of diseases associated with the drug.
# Output results with top 5 number of genes in a descending order.

# Filter edges for Compound-Gene relationships
# filter the edges to only include rows of the source column which starts with the Compound and target with start with Gene
compounds_genes = edges.filter(edges.source.startswith('Compound') & edges.target.startswith('Gene'))

# filter target column will keep only the the rows where edge connects a Compound node to a Gene node
# So if a source start with Compounds and the target a disease combine.
compounds_diseases = edges.filter(edges.source.startswith('Compound')& edges.target.startswith('Disease'))

# Converting the DataFrames to RDDs
compounds_genes_rdd = compounds_genes.rdd
compounds_diseases_rdd = compounds_diseases.rdd

# Join pattern 
# Combining the two RDDs using the union transformation, union to concatenate the compounds_genes_rdd and compounds_diseases_rdd to return only one RDD wil the elements of the two.
combined_genes_diseases= compounds_genes_rdd.union(compounds_diseases_rdd)

# Map the combined RDD to create key-value pairs of the form (compound, (num_genes, num_diseases))

#The resulting RDD is called count_compounds.
count_compounds = combined_genes_diseases.map(lambda x: (x.source, (1 if x.target.startswith('Gene') else 0, 1 if x.target.startswith('Disease') else 0)))

# Reduce by key to compute the number of genes and diseases associated with each compound

#The resulting RDD is called compound_gene_disease_count
compound_gene_disease_count = count_compounds.reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))

# Take the top 5 results
#  The key parameter is used to specify the sorting criteria, which is the number of genes associated with each compound in descending order
# (-x[1][0] means the second element of the tuple, which is the tuple of counts, and the first element of that tuple, which is the count of genes.
# The top 5 compounds with the most genes are obtained using the takeOrdered transformation on the compound_gene_disease_count RDD.
results = compound_gene_disease_count.takeOrdered(5, key=lambda x: -x[1][0])


# Print out the top results:
print("\nTop 5 number of genes in a descending order")
print("Drugs:\t\t\t#genes:\t#diseases:")
for drug, counts in results:
    count_genes = counts[0]
    count_disease = counts[1]
    print(f"{drug}\t{count_genes}\t {count_disease}")

