from vector_store import VectorStore

def main():
    vector_store = VectorStore()
    src_folder_id = "1PzQmrHD9yiOP9lH1pm_WRVp1MxBUr_jK"
    dest_folder_id = "1O0qLphDwEE35xHYeuq6CZn07rqv-MZcB"
    chunks = vector_store.load_from_gdrive("temp", src_folder_id, dest_folder_id)
    choice = input("Enter (1) to create vector store or (2) to add documents to existing vector store: ")
    if choice == "1":
        vector_store.create_vector_store(chunks)
    else:
        vector_store.add_documents(chunks)

if __name__ == "__main__":
    main()