from pathlib import Path
from vector_store import VectorStore
from langchain.docstore.document import Document

def main():
    vector_store = VectorStore()
    while True:
        print("Vector Store Operations")
        print("1. Add Document via Google Drive")
        print("2. Add Document via Local File")
        choice = input("Enter your choice: ")
        if choice.lower() == "q":
            break
        if choice == "1":
            gdrive_new_folder = "1xHvM3Vx_DaCuk1reGbPPi_c2gKi_Gdx5"
            gdrive_processed_folder = "1YyL5rBE2gyTDXDmsB79631ZUprzHeCUl"
            vector_store.load_from_gdrive("./temp", gdrive_new_folder, gdrive_processed_folder)
            print("Document added from Google Drive.")
        elif choice == "2":
            base_path = Path("./data")
            split_dir = base_path / "split"
            no_split_dir = base_path / "no_split"
            if split_dir.exists() and split_dir.is_dir():
                print("Found 'split' directory → using pre-chunked files.")
                for f in sorted(split_dir.iterdir()):
                    if f.is_file():
                        with f.open('r', encoding='utf-8') as fh:
                            content = fh.read()
                        chunks = vector_store.chunking(content, source_path=str(f))
                        vector_store.update_vector_store(chunks)
                print("Documents added from data/split (chunked).")
            elif no_split_dir.exists() and no_split_dir.is_dir():
                print("Found 'no_split' directory → loading without chunking.")
                for f in sorted(no_split_dir.iterdir()):
                    if f.is_file():
                        with f.open('r', encoding='utf-8') as fh:
                            content = fh.read()
                            documents = Document(page_content=content, metadata={"source": str(f)})
                            vector_store.update_vector_store([documents])
                print("Documents added from data/no_split (un-chunked).")
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()