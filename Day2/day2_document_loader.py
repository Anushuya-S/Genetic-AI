from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
import os

# Load PDFs
loader1 = PyPDFLoader("pdfs/book.pdf")
docs1 = loader1.load()

loader2 = PyPDFLoader("pdfs/notes.pdf")
docs2 = loader2.load()

documents = docs1 + docs2

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

print("Loaded documents:", len(documents))


# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

print("Total chunks:", len(chunks))


# Add metadata
for chunk in chunks:
    source = chunk.metadata.get("source", "")

    chunk.metadata["filename"] = os.path.basename(source)
    chunk.metadata["page_number"] = chunk.metadata.get("page", 0)
    chunk.metadata["upload_date"] = str(datetime.now().date())

    if "book" in source:
        chunk.metadata["source_type"] = "textbook"
    else:
        chunk.metadata["source_type"] = "notes"


# Filter function
def filter_chunks(chunks, **filters):
    result = []

    for chunk in chunks:
        match = True
        for k, v in filters.items():
            if chunk.metadata.get(k) != v:
                match = False
                break
        if match:
            result.append(chunk)

    return result


# Test
if __name__ == "__main__":
    print("\n--- TESTING ---")

    print("Book chunks:", len(filter_chunks(chunks, filename="book.pdf")))
    print("Notes chunks:", len(filter_chunks(chunks, filename="notes.pdf")))

    print("\nSample metadata:")
    print(chunks[0].metadata)