from pyspark.sql import SparkSession
from sentence_transformers import SentenceTransformer

spark = (
    SparkSession.builder
    .appName("AeroDocs-RAG-Ingestion")
    .master("local[*]")
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    .getOrCreate()
)

# load chunks
chunked_df = spark.read.parquet("output/chunks")
chunked_df.show()
print(chunked_df.count())

# initialize embedding model
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# convert spark df to pandas batches


