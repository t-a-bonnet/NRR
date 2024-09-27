# NRR: Neural Retrieval Refinement for Entity Matching

## Overview

**NRR (Neural Retrieval Refinement)** is an entity matching library designed to address the specific needs of the cultural heritage domain, with a focus on matching structured data (e.g. Linked Art) and unstructured full-text content (e.g. exhibition catalogues, newspaper texts, and blog posts). NRR combines retrieval and classification approaches to produce accurate entity matches using the PyTerrier information retrieval framework and a multi-layer perceptron (MLP) neural network classifier.

## Features

1. **LDG Retrieval Model**: 
   NRR uses the LDG retrieval model to generate candidate matches for a given query.

2. **Similarity Measures**: 
   After retrieval, NRR computes similarity measures for each query-text pair:
   - **Fuzzy Matching Score** (Levenshtein Distance)
   - **Smith-Waterman Algorithm**
   - **Jaro-Winkler Distance**
   - **Longest Common Subsequence**
   
   Features derived from these similarity measures are used as inputs for the classifier.

3. **MLP Neural Network Classifier**: 
   NRR uses a neural network classifier, trained on approximately 8,000 annotated query-text pairs, to predict the final matches. The classifier learned to assess if a query-text pair is a match based on the derived similarity features.

4. **Preprocessing Pipelines**: 
   - **Query Generation**: Automatically generate queries from Linked Art using URIs.
   - **Text Extraction**: Extract text from various file types including PDFs, JPGs, JPEGs, and PNGs, preparing them for the retrieval and matching stages.

5. **Postprocessing and Filtering**: 
   Remove matches where the query consists of only one word with a high term frequency, which is useful for large datasets containing ambiguous or non-descriptive terms such as "fragment" or "panel."

## Installation

To install the NRR library, clone the repository and install the dependencies:

```bash
git clone https://github.com/t-a-bonnet/NRR.git
cd nrr
pip install -r requirements.txt
```

## Example Usage

### Import and Initialize

```python
from nrr import NRR
nrr = NRR(index_path='./pd_index')
```

### Entity Matching

```python
results = nrr.match(query_df, text_df)
```

where query_df is a dataframe of queries in a column called 'query' and numerical query ids in a column called 'qids', and where text_df is a dataframe of texts in a column called 'text' and numerical document numbers in a column called 'docnos'.

### Preprocessing

#### Linked Art to Query

```python
query_df = nrr.structured_data_to_query(structured_data_df)
```

where structured_data_df is a dataframe of Linked Art URIs in a column called 'linked_art_uri'.

#### Extract Machine Readable Text from PDF

```python
text_df = nrr.extract('directory/to/files')
```

#### Extract Text from Image Files or PDF Using OCR

```python
text_df = nrr.ocr('directory/to/files')
```

### Postprocessing

```python
results = nrr.postprocess(results)
```

where results is a dataframe with a column called 'query'.

## Google Colab Tutorial

For a comprehensive demonstration of NRR and a step-by-step guide on its usage, please refer to the [Google Colab tutorial](https://colab.research.google.com/drive/1pwWTMatqy-sxB5etYUMTkXN5uqCd5pyg#scrollTo=D8_nd5tyNEcq).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.