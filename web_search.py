import requests
from bs4 import BeautifulSoup
import re
from typing import List
from langchain_core.documents import Document
from vector_store import VectorStore

# URLs from web_search.py
urls = [
    "https://www.ctu.edu.vn/gioithieu.html",
    "https://www.ctu.edu.vn/gioithieu/tong-quan.html",
    "https://www.ctu.edu.vn/gioithieu/dang-uy.html",
    "https://www.ctu.edu.vn/gioithieu/hoi-dong-truong.html",
    "https://www.ctu.edu.vn/gioithieu/ban-giam-hieu.html",
    "https://www.ctu.edu.vn/gioithieu/hoi-dong-kh-dt.html",
    "https://www.ctu.edu.vn/gioithieu/hoi-dong-quan-ly-chat-luong.html",
    "https://www.ctu.edu.vn/gioithieu/doan-the.html",
    "https://www.ctu.edu.vn/gioithieu/lanh-dao-don-vi.html",
    "https://www.ctu.edu.vn/gioithieu/cac-khu.html",
]

def clean_text(text: str) -> str:
    """Clean the text by removing HTML tags and extra whitespace"""
    clean = re.sub(r'<[^>]+>', '', text)
    clean = re.sub(r'\s+', ' ', clean)
    return clean.strip()

def extract_content_with_headers(soup: BeautifulSoup) -> str:
    """Extract content while preserving header structure"""
    content = []
    
    # Get main content area
    main_content = soup.find(['article', 'main', 'div'], class_=['item-page', 'content', 'main-content'])
    if not main_content:
        main_content = soup
    
    # Process headers and content
    for tag in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'ul', 'ol']):
        if tag.name.startswith('h'):
            # Add appropriate markdown headers
            level = int(tag.name[1])
            header_text = clean_text(tag.get_text())
            if header_text:
                content.append(f"{'#' * level} {header_text}")
        elif tag.name == 'p':
            text = clean_text(tag.get_text())
            if text:
                content.append(text)
        elif tag.name in ['ul', 'ol']:
            for li in tag.find_all('li'):
                text = clean_text(li.get_text())
                if text:
                    content.append(f"- {text}")
    
    return '\n\n'.join(content)

def fetch_and_process_urls() -> List[str]:
    """Fetch content from URLs and return list of markdown-formatted texts"""
    texts = []
    
    for url in urls:
        try:
            print(f"📥 Fetching content from: {url}")
            page = requests.get(url, timeout=10)
            page.raise_for_status()
            
            soup = BeautifulSoup(page.content, "html.parser")
            
            # Extract title and add as H1
            title = soup.title.string if soup.title else url.split('/')[-1]
            content = [f"# {clean_text(title)}"]
            
            # Extract and structure content
            content.append(extract_content_with_headers(soup))
            
            # Add to texts list
            texts.append('\n\n'.join(content))
            print(f"Processed: {url}")
            
        except Exception as e:
            print(f"Error processing {url}: {e}")
    
    return texts

def main():
    # Initialize vector store
    vector_store = VectorStore()
    
    # Fetch and process URLs
    print("Starting content extraction...")
    texts = fetch_and_process_urls()
    
    # Create chunks and add to vector store
    print("Processing documents into chunks...")
    chunks = []
    for i, text in enumerate(texts):
        source_url = urls[i]
        doc_chunks = vector_store.chunking(text, source_url)
        chunks.extend(doc_chunks)
    
    # Save the index
    print("Saving vector store index...")
    vector_store.add_documents(chunks)
    print("Process completed successfully!")

if __name__ == "__main__":
    main()