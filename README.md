# NRR: Neural Retrieval Refinement for Entity Matching

## Overview

**NRR (Neural Retrieval Refinement)** is an entity matching module designed as part of the [Enriching Exhibition Scholarship project](https://www.sps.ed.ac.uk/research/research-project/enriching-exhibition-scholarship#:~:text=Description,by%20the%20Linked%20Art%20collaboration.) to address the specific needs of the cultural heritage domain, with a focus on matching structured data (e.g. Linked Art) and unstructured full-text content (e.g. exhibition catalogues, newspaper texts, and blog posts). NRR combines retrieval and classification approaches to produce accurate entity matches using the PyTerrier information retrieval framework and a multi-layer perceptron (MLP) neural network classifier.

## Features

### 1. **Preprocessing**: 
   - **Query Generation**: Automatically generate queries from Linked Art using URIs.
   - **Text Extraction**: Extract text from various file types including PDFs, JPGs, JPEGs, and PNGs, preparing them for entity matching.

### 2. **Entity Matching**

   - **LGD Retrieval Model**: 
   NRR uses the Log-Logistic (LGD) retrieval model (Clinchant and Gaussier, 2009) to generate candidate matches for a given query.

   - **Similarity Measures**: 
   After retrieval, NRR computes similarity measures for each query-text pair in the LGD search results:
	•	Fuzzy Matching Score (Levenshtein Distance)
	•	Smith-Waterman Algorithm
	•	Jaro-Winkler Distance
	•	Longest Common Subsequence
   
        Features derived from these similarity measures are used as inputs for the classifier.

   - **MLP Neural Network Classifier**: 
   NRR uses a neural network classifier, trained on approximately 8,000 annotated query-text pairs, to predict the final matches. The classifier learned to assess if a query-text pair is a match based on the derived similarity features.

### 4. **Postprocessing**: 
   Remove matches where the query consists of only one word with a high term frequency, which is useful for large datasets containing ambiguous or non-descriptive terms such as "fragment" or "panel."

## Installation

To install the NRR module, clone the repository and install the dependencies:

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

### Preprocessing

#### Linked Art to Query

```python
query_df = nrr.structured_data_to_query(structured_data_df)
```

In this function:

   - 'structured_data_df' is a pandas DataFrame that contains:
       - A column named 'linked_art_uri', which holds the Linked Art URIs.

The nrr.structured_data_to_query() function retrieves the Linked Art JSON-LD records for the specified URIs, extracts the object titles and creator names, and constructs queries by concatenating them. The resulting queries are organized into a DataFrame with a column called 'query'.

#### Extract Machine Readable Text from PDF

```python
text_df = nrr.extract('directory/to/files')
```

In this function:

   - 'directory/to/files' is the path to the folder containing the PDF files from which text will be extracted.

The nrr.extract() function scans the specified directory for PDF files and extracts the text from each page of these files. The extracted text is organized into a DataFrame, where each entry includes the file path and the corresponding text content.

#### Extract Text from Image Files or PDF Using OCR

```python
text_df = nrr.ocr('directory/to/files')
```

In this function:

   - 'directory/to/files' is the path to the folder containing image and PDF files from which text will be extracted.

The nrr.ocr() function processes all files in the specified directory, applying Optical Character Recognition (OCR) to extract text from supported image formats (JPG, JPEG, PNG) and PDF files. The extracted text is compiled into a DataFrame, with each entry containing the file path and the associated text content.

### Entity Matching

```python
results = nrr.match(query_df, text_df)
```

In this function:

   - 'query_df' is a pandas DataFrame that contains:
       - A column named 'query', which holds the query text.
       - A column named 'qid', which contains numerical query IDs.
   - 'text_df' is a pandas DataFrame that contains:
       - A column named 'text', which holds the text documents.
       - A column named 'docno', which contains numerical document IDs.
   - Optional Arguments:
       - num_results (int): Specifies the maximum number of results returned by the retrieval model for each query. The default value is 50.
       - include_file_names (bool): If set to True, the function will include file names in the output results. The default value is False.
       - file_name_column (str): The name of the column containing file names, required only if include_file_names is set to True.

The nrr.match() function will match entities between these two DataFrames based on the provided queries and texts.

### Postprocessing

```python
results = nrr.postprocess(results)
```

In this function:

   - 'results' is a pandas DataFrame that contains:
       - A column named ‘query’, which holds the original query text.
       - Other columns containing the matching results and their corresponding details.

The nrr.postprocess() function refines the results by filtering out matches to one-word queries consisting of common words based on term frequency.

## Google Colab Tutorial

For a comprehensive demonstration of NRR and a step-by-step guide on its usage, please refer to the [Google Colab tutorial](https://colab.research.google.com/drive/1pwWTMatqy-sxB5etYUMTkXN5uqCd5pyg#scrollTo=D8_nd5tyNEcq).

## Enriching Exhibition Scholarship Project Participants

**[Aruna Devi Bhaugeerutty](https://www.linkedin.com/in/arunab/) (Co-Investigator)**
Head of Digital Collections, The Ashmolean Museum

**[Tyler Bonnet (Post-Doctoral Researcher)](https://www.linkedin.com/in/tylerbonnet/)**
Research Fellow, Neuropolitics Research Lab, University of Edinburgh

**[Kelly Davis (Data Engineer)](https://www.linkedin.com/in/daviskellyk/)**
Cultural Heritage Data Engineer, Yale University

**[Emmanuelle Delmas-Glass (Co-Investigator)](https://www.linkedin.com/in/emmanuelle-delmas-glass-3929343/)**
Head Collections Information and Access, Yale Center for British Art

**[Clare Llewellyn (AHRC Principal Investigator)](https://www.sps.ed.ac.uk/staff/clare-llewellyn)**
Lecturer in Governance, Technology, and Data, University of Edinburgh

**[Kevin Page (Co-Investigator)](https://eng.ox.ac.uk/people/kevin-page/)**
Associate Faculty and Senior Researcher, Oxford e-Research Centre, Department of Engineering Science, University of Oxford

**[Robert Sanderson (Chair of the Advisory Board)](https://www.linkedin.com/in/robert-sanderson/)**
Director for Cultural Heritage Metadata projects, in the office of the Vice Provost.

**[Andrew Shapland (Co-Investigator)](https://www.ashmolean.org/people/andrew-shapland)**
Sir Arthur Evans Curator of Bronze Age and Classical Greece, The Ashmolean Museum

**[Kayla Shipp (NEH Principal Investigator)](https://kaylashipp.com/)**
Program Manager for the Yale Digital Humanities Lab

## Funding details
This work was supported by the Arts and Humanities Research Council [AH/Y006011/1] and the National Endowment for the Humanities [HND-284978-22].

## References

- Clinchant, Stéphane and Gaussier, Eric. (2009) ‘Bridging language modeling and divergence from randomness models: A log-logistic model for IR’ in *Conference on the Theory of Information Retrieval*, pp. 54-65.

## License

The NRR code is licensed under the MIT License.

However, this project also depends on PyTerrier, which is licensed under the Mozilla Public License 2.0 (MPL 2.0). When distributing this project, you must comply with the terms of both licenses.

- MIT License applies to the NRR code.
- MPL 2.0 applies to the PyTerrier code.

For more information, refer to:
- [MIT License](./LICENSE)
- [Mozilla Public License Version 2.0](http://mozilla.org/MPL/2.0/)