# Annotated Search Engine

## Project Overview

**Language:** Python  
**Key Technology:** TF-IDF Algorithm  
**Features:** Document indexing, relevance ranking, and search functionality

## About

This project implements a search engine that uses TF-IDF scoring to intelligently rank documents based on query relevance. The algorithm considers both how frequently terms appear in individual documents and how rare they are across the entire corpus, providing more accurate search results than simple keyword matching.

## Running the Program

### 1. Save the File
Save the code as:
```
annotated_search.py
```

### 2. Run the Tests
Use pytest to run all included unit tests:
```bash
pytest annotated_search.py -v
```

### 3. Run the Program Manually
To execute the search engine demo directly:
```bash
python annotated_search.py
```

## Requirements

- Python 3.x
- pytest (for running tests)

Install pytest if needed:
```bash
pip install pytest
```

[Add license information if applicable]
