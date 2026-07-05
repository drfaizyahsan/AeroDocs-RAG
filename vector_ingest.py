import hashlib
from pyspark.sql import SparkSession
from sentence_transformers import SentenceTransformer

spark = (
    SparkSession.builder
    .appName("AeroDocs-RAG-Ingestion")
    .master("local[*]")
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    .getOrCreate()
)

def generate_chunk_id(chunk_index: int)->str:
    """
    create chunk id for each chunk in order to have
    idempotency keys
    :param chunk_index:
    :return:
    """
    k = f"{chunk_index}:"
    return hashlib.sha256(k.encode()).hexdigest()

# load chunks
chunked_df = spark.read.parquet("output/chunks")
chunked_df.show()
print(chunked_df.count())

# initialize embedding model
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# convert spark df to pandas batches
batch_size = 128

chunked_pandas = chunked_df.toPandas()

for start in range(0, len(chunked_pandas), batch_size):

    batch = chunked_pandas.iloc[start: start + batch_size]
    documents = batch["chunk_text"].tolist()

    # generate embeddings
    embeddings = embedding_model.encode(
        documents,
        show_progress_bar=False
    ).tolist()

    print(len(embeddings), len(embeddings[0]))

    # generate ids for idempotency keys
    ids = [generate_chunk_id(i) for i in range(start, start+len(batch))]

    print(ids)

    break





