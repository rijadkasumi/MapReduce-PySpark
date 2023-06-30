This project is build using PySpark with Python programming language and tested in macOS

Requiremnts :
PySpark 3.11 or newer
Python 3.6 or newer
Anaconda

Installation:
https://docs.anaconda.com/anaconda/install/index.html
conda install -c conda-forge pyspark
sudo apt-get install python

Running the Pyspark Files:
Open the terminal
Activate the Anaconda environment : conda activate /Users/....
Then type - > pyspark  

A way of executing Pyspark files is starting and creating a SparkSession in the terminal and feeding the commands as you go, here is an example how that would look:
# Import the necessary libraries
from pyspark.sql import SparkSession
# Starting the spark session HetioNet
spark = SparkSession.builder.appName("HetioNet").getOrCreate()
...
But in this project the files are as python scripts that can be run from a command line, so no need to feed the commands as you go.

*
The files in this project are saved as a python script that can be run from the command line just navigating to the right directory and using the commands:
python Q1.py
python Q2.py
python Q3.py

Once the command is run it the will automatically read the code and start a SparkSession and execute the code to produce the desired output.
Be aware to also make sure that the .tsv files are in the correct directory, this is an example how to set the right directory.
Set the correct path to the .tsv file to properly read the files and perform the tasks, for example:
edges = spark.read.format("csv").option("delimiter", "\t").option("header", "true").load("/Users/JohnDoe/Desktop/HetioNet/edges.tsv)


