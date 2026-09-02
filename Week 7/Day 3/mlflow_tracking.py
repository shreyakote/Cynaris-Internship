import mlflow


mlflow.set_experiment("LlamaIndex Document Indexing and RAG")


with mlflow.start_run(run_name="LlamaIndex Basic"):

    mlflow.log_param("documents", 5)
    mlflow.log_param("queries", 10)
    mlflow.log_param("embedding_model", "nomic-embed-text:latest")
    mlflow.log_param("llm_model", "llama3.2:3b")
    mlflow.log_param("vector_store", "LlamaIndex InMemory")

    print("MLflow run created successfully.")


with mlflow.start_run(run_name="LlamaIndex ChromaDB"):

    mlflow.log_param("documents", 5)
    mlflow.log_param("queries", 10)
    mlflow.log_param("embedding_model", "nomic-embed-text:latest")
    mlflow.log_param("llm_model", "llama3.2:3b")
    mlflow.log_param("vector_store", "ChromaDB")

    print("ChromaDB MLflow run created successfully.")


print("\nMLflow tracking completed.")