# Import the necessary libraries
from pyspark.sql import SparkSession
# Starting the spark session HetioNet
spark = SparkSession.builder.appName("HetioNet").getOrCreate()
# Reading the tsv files
edges = spark.read.format("csv").option("delimiter", "\t").option("header", "true").load("/Users/rijadkasumi/Desktop/data/edges.tsv")
nodes = spark.read.format("csv").option("delimiter", "\t").option("header", "true").load("/Users/rijadkasumi/Desktop/data/nodes.tsv")

### Question 3 :
# Get the name of drugs that have the top 5 number of genes. Out put the results.disease_compounds.show(2)

# Filter edges for Compound-Gene relationships
compound_genes = edges.filter(edges.source.startswith('Compound') & edges.target.startswith('Gene'))

# Convert the DataFrame to RDD and map to create key-value pairs of the form (compound, [gene1, gene2, ...])
compound_gene_pairs = compound_genes.rdd.map(lambda x: (x.source, [x.target]))

# Reduce by key to combine the genes associated with each 
# a,b: a+b combine the genes assosiated with the disease
combined_compound_genes = compound_gene_pairs.reduceByKey(lambda a, b: a + b)

# Map to create key-value pairs of the form (compound, count of genes associated with the compound
compound_gene_counts = combined_compound_genes.map(lambda x: (x[0], len(x[1])))

# Get the top 5 compounds with the highest gene counts
# -x1 takes each par in compounds_gene_coun and returns the negative of the count of genes
top_compounds = compound_gene_counts.takeOrdered(5, key=lambda x: -x[1])

# Join with the nodes DataFrame to get the names of the compounds
# Convert top Compunds into RDD and parallelize to use Spark operation
# Convert id and count back to DataFrame to join the rdd top_compounds with the nodes.
results = spark.sparkContext.parallelize(top_compounds) \
    .toDF(["id", "count"]) \
    .join(nodes.filter(nodes.kind == "Compound"), "id") \
    .select("name", "count")

# Output the results
print("\nTop 5 drugs with the highest gene counts:")
print("Drugs:\t\tGenes:")
for row in results.collect():
   print("{}\t{}".format(row.name, row["count"]))
    
