from vector_store import VectorStore

def main():
    vector_store = VectorStore()
    folder_id = "1O0qLphDwEE35xHYeuq6CZn07rqv-MZcB"
    chunks = vector_store.load_from_gdrive("temp", folder_id)
    vector_store.create_vector_store(chunks)

if __name__ == "__main__":
    main()