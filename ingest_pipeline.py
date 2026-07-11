from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, StringType
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pyspark.sql.functions import pandas_udf
import pyspark.sql.functions as F
import pandas as pd

spark = (
    SparkSession.builder
    .appName("AeroDocs-RAG-Ingestion")
    .master("local[*]")
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)

# pandas udf to split texts
@pandas_udf(ArrayType(StringType()))
def split_into_chunks(text_series: pd.Series)->pd.Series:
    """
    pandas udf to split text into overlapping chunks
    :param text_series:
    :return:
    """
    return text_series.apply(
        lambda text: (
            text_splitter.split_text(text)
            if text and isinstance(text, str)
            else []
        )
    )

# TODO: read all md files instead of just one
# df = spark.read.text("./aircraft_amm_mock/ATA_*.md")
df = spark.read.text("./aircraft_amm_mock/ATA_21_AirConditioning_Pack_Removal.md")
print(df.count())
print(df.columns)
df.show()

# apply pandas df for chunking
chunked_df = df.withColumn(
    "chunks",
    split_into_chunks(F.col("value"))
).withColumn(
    "chunk_text",
    F.explode(F.col("chunks"))
)

chunked_df.show()
print(chunked_df.count())

# save output
chunked_df.write.mode("overwrite").parquet("output/chunks")
