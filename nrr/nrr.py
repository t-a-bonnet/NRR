import concurrent.futures
import glob
import nltk
import numpy as np
import pandas as pd
import pyterrier as pt
import textdistance
import torch
import torch.nn as nn
import os
import pdfplumber
import pytesseract
import shutil
import re
import requests
import string as st
import urllib.request

from collections import Counter
from fuzzywuzzy import fuzz
from nltk import ngrams
from nltk.corpus import stopwords
from pdf2image import convert_from_path
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from unidecode import unidecode

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Get English stopwords from NLTK
stop_words = set(stopwords.words('english'))

# Define the MLP model architecture
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


def calculate_fuzzy_matching_score(query, text):
    return fuzz.token_set_ratio(query, text)

def calculate_jaro_winkler_distance(query, text):
    return textdistance.jaro_winkler(query, text)

def calculate_smith_waterman_similarity(query, text):
    matrix = [[0] * (len(str(text)) + 1) for _ in range(len(str(query)) + 1)]
    max_score = 0
    max_i = 0
    max_j = 0
    for i in range(1, len(str(query)) + 1):
        for j in range(1, len(str(text)) + 1):
            match = 2 if str(query)[i - 1] == str(text)[j - 1] else -1
            score = max(
                matrix[i - 1][j - 1] + match,
                matrix[i - 1][j] - 1,
                matrix[i][j - 1] - 1,
                0
            )
            matrix[i][j] = score
            if score > max_score:
                max_score = score
                max_i = i
                max_j = j
    distance = max_score
    while max_i > 0 and max_j > 0 and matrix[max_i][max_j] != 0:
        if matrix[max_i][max_j] == matrix[max_i - 1][max_j - 1] + (2 if str(query)[max_i - 1] == str(text)[max_j - 1] else -1):
            max_i -= 1
            max_j -= 1
        elif matrix[max_i][max_j] == matrix[max_i - 1][max_j] - 1:
            max_i -= 1
        else:
            max_j -= 1
    return distance

def calculate_lcs(query, text):
    m = len(str(query))
    n = len(str(text))
    lcs_matrix = np.zeros((m+1, n+1))
    for i in range(1, m+1):
        for j in range(1, n+1):
            if str(query)[i-1] == str(text)[j-1]:
                lcs_matrix[i][j] = lcs_matrix[i-1][j-1] + 1
            else:
                lcs_matrix[i][j] = max(lcs_matrix[i-1][j], lcs_matrix[i][j-1])
    return lcs_matrix[m][n]

def preprocess_text(text):
    if not text or pd.isna(text):
        return None
    text = unidecode(text)
    text = ''.join([x if x.isalpha() or x.isspace() else ' ' for x in text])
    special_characters = "¶¬©£ª√®"
    characters_to_remove = st.punctuation + special_characters
    text = ''.join([x for x in text if x not in characters_to_remove])
    text = text.lower()
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def get_files_list(path):
    return [file for file in glob.glob(path, recursive=True) if os.path.isfile(file) and '__MACOSX' not in file and not file.startswith('._')]

# Helper function to fetch JSON-LD data from a given URI
def fetch_json_ld(uri):
    headers = {'Accept': 'application/ld+json'}
    try:
        response = requests.get(uri, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to fetch JSON-LD from {uri}, status code: {response.status_code}")
            return None
    except requests.RequestException as e:
        print(f"An error occurred while fetching data from {uri}: {e}")
        return None

# Function to extract object_title from linked art
def extract_object_title(json_ld):
    try:
        identified_by = json_ld.get('identified_by', [])
        for item in identified_by:
            if item.get('type') == 'Name':
                return item.get('content')
        print("Object title not found.")
        return None
    except KeyError as e:
        print(f"Error extracting object title: {e}")
        return None

# Function to extract creator name from linked art
def extract_creator_name(json_ld):
    try:
        produced_by = json_ld.get('produced_by', {})
        if isinstance(produced_by, dict):
            carried_out_by = produced_by.get('carried_out_by', {})
            if isinstance(carried_out_by, dict):
                identified_by = carried_out_by.get('identified_by', [])
                for item in identified_by:
                    if item.get('type') == 'Name':
                        return item.get('content')
        print("Creator name not found.")
        return None
    except KeyError as e:
        print(f"Error extracting creator name: {e}")
        return None

class NRR:
    def __init__(self, index_path, mlp_model_path='nrr_mlp.pt', model_url='https://github.com/t-a-bonnet/NRR/blob/main/nrr/nrr_mlp.pt?raw=true'):
        # Initialize PyTerrier
        pt.init()

        # Check if the MLP model file exists
        if not os.path.exists(mlp_model_path):
            print(f"{mlp_model_path} not found. Downloading from {model_url}...")
            urllib.request.urlretrieve(model_url, mlp_model_path)
            print(f"Downloaded model to {mlp_model_path}.")

        # Define and load the MLP model
        self.input_size = 4 
        self.hidden_size = 64
        self.model = MLP(input_size=self.input_size, hidden_size=self.hidden_size, output_size=1)
        self.model.load_state_dict(torch.load(mlp_model_path, map_location=torch.device('cpu')))
        self.model.eval()

    def search(self, query_df, text_df, include_file_names=False, file_name_column=None):
        # Check if 'qid' column exists in query_df
        if 'qid' not in query_df.columns:
            raise ValueError("Error: 'qid' column is missing in query_df. Please add the column and try again.")
        
        # Check if 'docno' column exists in text_df
        if 'docno' not in text_df.columns:
            raise ValueError("Error: 'docno' column is missing in text_df. Please add the column and try again.")
        
        # If file names are to be included, check for the file name column
        if include_file_names:
            if file_name_column is None:
                return "Error: File name column not provided. Please specify the file name column."
            if file_name_column not in text_df.columns:
                return f"Error: The specified file name column '{file_name_column}' does not exist in text_df."
        
        # Ensure 'qid' and 'docno' are strings
        query_df['qid'] = query_df['qid'].astype(str)
        text_df['docno'] = text_df['docno'].astype(str)

        # Ensure queries are preprocessed
        query_df['query'] = query_df['query'].apply(preprocess_text)
        query_df.dropna(subset=['query'], inplace=True)
        query_df.reset_index(drop=True, inplace=True)

        # Ensure texts are preprocessed
        text_df['text'] = text_df['text'].apply(preprocess_text)
        text_df.dropna(subset=['text'], inplace=True)
        text_df.reset_index(drop=True, inplace=True)

        # Remove existing index directory
        index_path = './pd_index'
        if os.path.exists(index_path):
            shutil.rmtree(index_path)

        # Indexing texts
        pd_indexer = pt.DFIndexer(index_path)
        index = pd_indexer.index(text_df['text'], text_df['docno'])

        # Set up the retrieval model
        br = pt.BatchRetrieve(index, controls={'wmodel': 'LGD', 'max_results': 100})
        br.setControl('wmodel', 'LGD')

        # Similarity score means
        fuzzy_matching_mean = 91.57859531772576
        jaro_winkler_mean = 0.5251933067625418
        smith_waterman_mean = 45.26086956521739
        lcs_mean = 43.03010033444816

        def calculate_similarities(row, query):
            # Calculate all similarities for a given row
            fuzzy_matching = calculate_fuzzy_matching_score(query, row['text'])
            jaro_winkler = calculate_jaro_winkler_distance(query, row['text'])
            smith_waterman = calculate_smith_waterman_similarity(query, row['text'])
            lcs = calculate_lcs(query, row['text'])
            return fuzzy_matching, jaro_winkler, smith_waterman, lcs

        def search_and_classify(query_df, num_results, text_df):
            results_dict = {}
            
            for _, query_row in query_df.iterrows():
                qid = query_row['qid']
                query = query_row['query']

                # Perform the retrieval using PyTerrier
                ranks = br.search(query)
                ranks = ranks[:num_results]
                ranks['qid'] = qid
                ranks['query'] = ''
                ranks['text'] = ''
                if include_file_names:
                    ranks[file_name_column] = ''

                ranks = ranks.sort_values(by=['score'], ascending=False)
                ranks.rename(columns={'score': 'ret_score'}, inplace=True)
                ranks.drop(columns={'docid', 'rank'}, inplace=True)

                # Fill in the query, text, and optionally file name columns
                for index, row in ranks.iterrows():
                    ranks.at[index, 'query'] = query
                    docno = row['docno']
                    text_match = text_df[text_df['docno'] == docno]
                    if not text_match.empty:
                        ranks.at[index, 'text'] = text_match.iloc[0]['text']
                        if include_file_names:
                            ranks.at[index, file_name_column] = text_match.iloc[0][file_name_column]

                # Parallelize similarity calculations
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = {
                        executor.submit(calculate_similarities, row, query): index for index, row in ranks.iterrows()
                    }

                    for future in concurrent.futures.as_completed(futures):
                        index = futures[future]
                        try:
                            fuzzy_matching, jaro_winkler, smith_waterman, lcs = future.result()
                            ranks.at[index, 'fuzzy_matching'] = fuzzy_matching
                            ranks.at[index, 'jaro_winkler'] = jaro_winkler
                            ranks.at[index, 'smith_waterman'] = smith_waterman
                            ranks.at[index, 'lcs'] = lcs
                        except Exception as e:
                            print(f"Error calculating similarities for row {index}: {e}")

                # Create features by subtracting the means
                ranks['fuzzy_matching_feature'] = ranks['fuzzy_matching'] - fuzzy_matching_mean
                ranks['jaro_winkler_feature'] = ranks['jaro_winkler'] - jaro_winkler_mean
                ranks['smith_waterman_feature'] = ranks['smith_waterman'] - smith_waterman_mean
                ranks['lcs_feature'] = ranks['lcs'] - lcs_mean

                # Prepare input features for the model
                features = ranks[['fuzzy_matching_feature', 'jaro_winkler_feature', 'smith_waterman_feature', 'lcs_feature']].values
                features_tensor = torch.tensor(features, dtype=torch.float32)

                # Perform classification
                with torch.no_grad():
                    outputs = self.model(features_tensor)
                    preds = torch.round(torch.sigmoid(outputs)).squeeze().cpu().numpy()

                # Add predictions to results
                ranks['prediction'] = preds

                # Reset index and store in dictionary
                ranks.reset_index(drop=True, inplace=True)
                results_dict[qid] = ranks

            return results_dict

        # Call the search function and return the results
        return search_and_classify(query_df, num_results=10, text_df=text_df)

    def ocr(self, directory):
        rows = []
        supported_image_formats = ('.jpg', '.jpeg', '.png')
        supported_pdf_format = '.pdf'

        # Get all files in the directory
        files_list = get_files_list(os.path.join(directory, '**', '*'))

        for file in files_list:
            # Handle image files (JPG, JPEG, PNG)
            if file.lower().endswith(supported_image_formats):
                try:
                    img = Image.open(file)
                    text = pytesseract.image_to_string(img).replace('\n', ' ')
                    rows.append({'file': file, 'text': text})
                except Exception as e:
                    print(f"Error processing image {file}: {e}")

            # Handle PDF files
            elif file.lower().endswith(supported_pdf_format):
                try:
                    # Convert PDF to images
                    images = convert_from_path(file)
                    
                    for page, img in enumerate(images, start=1):
                        text = pytesseract.image_to_string(img).replace('\n', ' ')
                        rows.append({'file': f"{file}/page_{page}", 'text': text})
                except Exception as e:
                    print(f"Error processing PDF {file}: {e}")

        # Convert results to DataFrame
        return pd.DataFrame(rows)

    def extract(self, directory):
        rows = []

        # Get all PDF files in the directory using the get_files_list function
        files_list = get_files_list(os.path.join(directory, '**', '*.pdf'))

        for file in files_list:
            try:
                with pdfplumber.open(file) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            text = text.replace('\n', ' ')
                            rows.append({'file': f"{file}/page_{page}", 'text': text})
            except Exception as e:
                print(f"Error processing {file}: {e}")

        return pd.DataFrame(rows)

    def postprocess(self, df):
        # Function to preprocess queries by removing stopwords
        def preprocess_query(query):
            words = query.split()
            filtered_words = [word for word in words if word.lower() not in stop_words]
            return ' '.join(filtered_words)

        df['query'] = df['query'].apply(preprocess_query)
        word_freq = Counter(' '.join(df['query']).split())
        average_freq = sum(word_freq.values()) / len(word_freq)

        def filter_common_words(query):
            words = query.split()
            if len(words) == 1 and word_freq[words[0]] > average_freq:
                return False
            return True

        df = df[df['query'].apply(filter_common_words)]
        df.reset_index(drop=True, inplace=True)

        return df
    
    # Function to process structured data to form queries
    def structured_data_to_query(self, df):
        # Add new columns to store the extracted object title and creator name
        df['object_title'] = None
        df['creator_name'] = None
        
        # Loop over each row in the DataFrame
        for idx, row in df.iterrows():
            linked_art_uri = row['linked_art_uri']
            
            # Fetch the JSON-LD data from the URI
            json_ld_data = fetch_json_ld(linked_art_uri)
            
            if json_ld_data:
                # Extract object title and creator name
                object_title = extract_object_title(json_ld_data)
                creator_name = extract_creator_name(json_ld_data)
                
                # Store the results in the respective columns
                df.at[idx, 'object_title'] = object_title
                df.at[idx, 'creator_name'] = creator_name

        # Create 'query' column by concatenating 'object_title' and 'creator_name'
        df['query'] = df.apply(lambda row: f"{row['object_title']} {row['creator_name']}".strip() if row['object_title'] and row['creator_name'] else None, axis=1)
        
        # Drop 'object_title' and 'creator_name' columns
        df.drop(columns=['object_title', 'creator_name'], inplace=True)
        
        return df