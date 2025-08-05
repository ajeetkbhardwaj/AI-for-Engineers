PySpark

What is Apache Spark ?

> Apache Spark is an open-source, distributed computing system designed for processing large-scale data quickly and efficiently. It provides an intuitive programming model for working with structured and unstructured data, enabling users to perform transformations and actions on datasets using high-level APIs.

What are the key features that Spark has ?

> 1. Speed: Spark processes data in memory, making it significantly faster than traditional disk-based frameworks like Hadoop MapReduce.
> 2. Scalability: It can scale seamlessly from a single machine to thousands of nodes in a cluster.
> 3. Versatility: Spark supports multiple workloads, including batch processing, interactive querying, streaming, machine learning, and graph processing.

Note : At the core of Spark lies the Resilient Distributed Dataset (RDD), a fault-tolerant and immutable data abstraction that enables distributed data processing. Transformations on RDDs are executed lazily, building a lineage graph that allows efficient fault recovery.

Now, we know the Apache Spark's core features and neded to get hands on how to perform big data processing using its Python API — PySpark i.e hands-on experience with RDDs, Spark optimizations, monitoring via Spark UI, and performing data analytics tasks.

## Setup References

1. How to install the openjdk in conda environment and python : https://stackoverflow.com/questions/27003920/how-to-install-java-locally-no-root-on-linux-if-possible
2. Install the pyspark and findspark because we working in the jupyter notebook

# References

1. ML with PySpark : https://www.kaggle.com/code/fatmakursun/pyspark-ml-tutorial-for-beginners
2. ML with PySpark and Tracking Experiments : https://mlflow.org/docs/latest/ml/traditional-ml/sparkml/
3. https://mlflow.org/docs/latest/ml/traditional-ml/sparkml/guide/
