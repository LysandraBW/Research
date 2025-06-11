import os
import time
import pandas as pd
from pyalex import Works
from itertools import chain
from transformers import pipeline
from beautifultable import BeautifulTable

# Constants
TERMINAL_SIZE = os.get_terminal_size()
HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKCYAN = '\033[96m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

# Stack of Pages
pages = []

# To-Be Labeled DF
labeled_df = None
labeled_df_filepath = None

# Unlabeled DF
unlabeled_df = None
unlabeled_df_filepath = None

# Revert Abstract Function
def revert_abstract(inverted_abstract):
    if not inverted_abstract:
        return ""
    abstract = ""
    i = 0
    while True:
        index_found = False
        for k, v in inverted_abstract.items():
            # print(k, v)
            if i in v:
                if abstract and k not in [".", ",", "?", "!"]:
                    abstract += " "
                abstract += k
                i += 1
                index_found = True
        if not index_found:
            break
    return abstract

# On Topic Function
classifier = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
def on_topic(text, verbose=False):
    # Topic and Threshold
    # I'm using a different model which seems to be more confident,
    # so I'm going to increase the threshold.
    topics = [("ecology", 0.9), ("interaction", 0.9)]
            
    for topic, threshold in topics:
        result = classifier(text, [topic])
        if verbose:
            print(result["scores"][0])
        if result["scores"][0] < threshold:
            return False
    return True

# CMD UI Functions
def clear_screen():
    os.system('cls')

def print_border():
    print('-' * TERMINAL_SIZE.columns)
    
def print_header():
    print(f"{OKBLUE}{BOLD}Label Helper™{ENDC}")
    print("Use this program to quickly create labeled datasets.")
    print_border()

def page_menu():
    global pages
    global labeled_df
    global labeled_df_filepath
    
    clear_screen()
    print_header()
    print(f"{BOLD}Create or Load Labeled Dataset{ENDC}")
    print("Start by entering the filepath of the labeled dataset to create or add onto.")
    print("Ensure that the filepath entered is for a .csv file.")
    print_border()
    while True:
        labeled_df_filepath = input("Enter Filename of Dataset: ")
    
        # Existing File
        if os.path.isfile(labeled_df_filepath):
            print(f"Loading File: '{labeled_df_filepath}'")
            try:
                labeled_df = pd.read_csv(labeled_df_filepath)
            except Exception:
                print(f"{FAIL}Error Occurred. Try Again.{ENDC}\n")
                continue
        # Creating New File
        else:
            if labeled_df_filepath[-4:] != ".csv":
                print(f"{FAIL}Must Be CSV File (.csv). Try Again.{ENDC}\n")
                continue
            try:
                print(f"Creating File: '{labeled_df_filepath}'")
                labeled_df = pd.DataFrame(columns=['Title','Abstract','DOI','Score'])
                labeled_df.to_csv(labeled_df_filepath, index=False)
            except Exception:
                print(f"{FAIL}Error Occurred. Try Again.{ENDC}\n")
                continue

        # If this statement is reached,
        # it's likely that everything is fine.
        break

    # Next Page
    pages.append("Method")

def page_method():
    global pages
    
    clear_screen()
    print_header()
    print(f"{BOLD}Find Papers to Score{ENDC}")
    print("Now, we need to find unscored papers. Enter [1] to view a list of datasets that contain unscored papers or [2] to look for papers on OpenAlex (requires an Internet connection).")
    print_border()
    
    # Action
    print("Actions:")
    print("[A] Go Back")
    print("[B] Exit Program")
    print("[1] Load Unscored Dataset")
    print("[2] Search for Unscored Papers")
    print_border()

    # Receive Input
    while True:
        try:
            option = input("Enter Input: ")
            if option in ['1', '2']:
                option = int(option)
                break
            if option in ['A', 'B']:
                break
            raise ValueError('Invalid Input')
        except ValueError:
            print(f"{FAIL}Must Enter 1 or 2. Try Again.{ENDC}\n")
            continue
        break

    # Next Page
    if option == 1: 
        pages.append("Method Load")
    elif option == 2:
        pages.append("Method Search")
    elif option == 'A':
        pages.pop()
    elif option == 'B':
        exit()

def page_method_load():
    global pages
    global unlabeled_df
    global unlabeled_df_filepath
    
    clear_screen()
    print_header()
    print(f"{BOLD}Load Unscored Dataset{ENDC}")
    print("Enter the number of the dataset to score or a letter that corresponds to an action (i.e. 'Go Back' and 'Exit Program').")
    print_border()

    # Action
    print("Actions:")
    print("[A] Go Back")
    print("[B] Exit Program")
    print_border()

    # Datasets
    # The unscored datasets should be DataFrames stored as CSV files
    # in a directory './UnscoredDatasets'.
    print("Datasets:")
    
    directory = "./UnscoredDatasets"
    files = [f for f in os.listdir(directory) if os.path.isfile(f"{directory}/{f}") and f[-4:] == '.csv']
    number_files = len(files)

    # Print Files
    for i, f in enumerate(files):
        print(f"[{i+1}] {f}")
    
    print_border()

    # Receive Input
    while True:
        try:
            option = input("Enter Input: ")
            if option == 'A':
                pages.pop()
                return
            if option == 'B':
                exit()
            
            option = int(option)
            if option < 0 or option > number_files:
                raise ValueError(f'Number Must Be Between 1 and {number_files}')
        except ValueError:
            print(f"{FAIL}Must Enter Number Between 1 and {number_files}. Try Again.{ENDC}\n")
            continue

        unlabeled_df_filepath = f"{directory}/{files[option - 1]}"
        print("Loading File:",unlabeled_df_filepath)
        unlabeled_df = pd.read_csv(unlabeled_df_filepath)
        break

    # Next Page
    pages.append("Preview")

def page_method_search():
    global pages
    global unlabeled_df
    global unlabeled_df_filepath
    
    clear_screen()
    print_header()
    print(f"{BOLD}Search Unscored Papers{ENDC}")
    print("Enter a search query (i.e. 'intraguild predation') to load papers from OpenAlex or a letter that corresponds to an action (i.e. 'Go Back' and 'Exit Program'). If you want to search 'A', you should instead input 'a', or else you will end up going back a page; the same goes for 'B' and 'b'.")
    print_border()

    # Action
    print("Actions:")
    print("[A] Go Back")
    print("[B] Exit Program")
    print_border()

    while True:
        query = input("Enter Input: ")

        # Go Back
        if query == 'A':
            pages.pop()
            return

        # Exit
        if query == 'B':
            exit()

        # Searching OpenAlex
        num_pages = 0
        max_pages = 1000
        works = Works().search(query)
        pager = works.paginate(per_page=200)

        # Empty DF
        unlabeled_df = pd.DataFrame(columns=['Title','Abstract','DOI','Score'])

        # Number Results
        num_results = min(works.count(), max_pages)
        if not num_results:
            print("No Results. Try Again\n")
            continue
        print(f"Number Results: {num_results}")
        print("Searching...")
        
        # Populating DF
        for page in pager:
            for work in page:
                if num_pages >= max_pages:
                    break
                abstract = revert_abstract(work["abstract_inverted_index"])
                try:
                    if not abstract or not on_topic(abstract):
                        continue
                except Exception:
                    print("Out of Memory. Ending Search")
                    max_pages = 0
                paper = pd.DataFrame([[work["title"], abstract, work["doi"], 0]],  columns=['Title','Abstract','DOI','Score'])
                unlabeled_df = pd.concat([unlabeled_df, paper])
                num_pages += 1
            if num_pages >= max_pages:
                break

        # Save DF
        unlabeled_df.reset_index(drop=True, inplace=True)
        unlabeled_df_filepath = f"./UnscoredDatasets/Search-{query.replace(' ', '-')}.csv"
        unlabeled_df.to_csv(unlabeled_df_filepath, index=False)
        break

    # Next Page
    pages.append("Preview")

def page_preview():
    global pages
    global unlabeled_df
    global unlabeled_df_filepath

    NUM_PAPERS_PER_TABLE = 5
    ABSTRACT_MAX_LENGTH = 150
    
    # Tables
    table_i = 0
    tables = []
    for i, row in unlabeled_df.iterrows():
        table = None if len(tables) == 0 else tables[-1]
        if i % NUM_PAPERS_PER_TABLE == 0:
            tables.append(BeautifulTable())
            table = tables[-1]
            table.column_headers = ["Title", "Abstract", "DOI", "Score"]

        shortened_abstract = row['Abstract'][:ABSTRACT_MAX_LENGTH] + '...' if len(row['Abstract']) > ABSTRACT_MAX_LENGTH else ''
        table.append_row([row['Title'], shortened_abstract,  row['DOI'],  row['Score']])
    num_tables = len(tables)
    
    while True:
        clear_screen()
        print_header()
        print(f"{BOLD}Preview Dataset{ENDC}")
        print("Before scoring, you can check whether your chosen dataset looks correct. Enter the [1] to continue; [2] to go to the previous page of papers; [3] to go to the next page of papers; or a letter that corresponds to an action (i.e. [A] for 'Go Back' and [B] for 'Exit Program').")
        print_border()
        
        # Actions
        print("Actions:")
        print("[A] Go Back")
        print("[B] Exit Program")
        print("[1] Continue")
        print("[2] Previous Page")
        print("[3] Next Page")
        print_border()

        # Showing Table Index
        print(f"{table_i+1}/{num_tables}")
        print_border()
        
        # Showing Table
        print(f"{tables[table_i]}")
        print_border()

        # Receive Input
        try:
            option = input("Enter Input: ")

            # Go Back
            if option == 'A':
                pages.pop()
                return
    
            # Exit
            if option == 'B':
                exit()

            # Integer Entered
            option = int(option)
            
            if option == 1:
                break
            elif option == 2:
                table_i = max(0, table_i - 1)
                continue
            elif option == 3:
                table_i = min(len(tables) - 1, table_i + 1)
                continue
            else:
                print(f"{FAIL}Must Enter 1, 2, 3, 'A', or 'B'. Try Again.{ENDC}\n")
                continue
        except ValueError:
            print(f"{FAIL}Must Enter 1, 2, 3, 'A', or 'B'. Try Again.{ENDC}\n")
            continue

    # Next Page
    pages.append("Score")

def page_score():
    global pages
    global unlabeled_df
    global unlabeled_df_filepath
    global labeled_df
    global labeled_df_filepath
    
    paper_i = 0
    num_papers = unlabeled_df.shape[0]

    while paper_i < num_papers:
        clear_screen()
        print_header()
        print(f"{BOLD}Score Papers{ENDC}")
        print("For each paper, enter an integer between 0 and 3. 0 indicates a lower quality and 3 indicates a higher quality. If the paper has already been scored (you should see a 'Current Score') or if you would just like to skip a paper, press the enter key without any input. Should you want to exit or go back (i.e. [A] for 'Go Back' and [B] for 'Exit Program'), your scores will automatically be saved.")
        print_border()

        # Actions
        print("Actions:")
        print("[A] Go Back")
        print("[B] Exit Program")
        print("[C] Exit Scoring")
        print_border()

        print(f"{paper_i+1}/{num_papers}")
        print_border()

        # Paper Information
        title = unlabeled_df.iloc[paper_i]['Title']
        abstract = unlabeled_df.iloc[paper_i]['Abstract']
        doi = unlabeled_df.iloc[paper_i]['DOI']
        print(f"{BOLD}Title:{ENDC} {title}")
        print(f"{BOLD}Abstract:{ENDC} {abstract}")
        print(f"{BOLD}DOI:{ENDC} {doi}")

        # Current Score (if Any)
        matching_labeled_row = labeled_df.loc[(labeled_df['Title'] == title) & (labeled_df['DOI'] == doi)]
        not_found = matching_labeled_row.empty
        if not not_found:
            print(f"{WARNING}{BOLD}*Current Score: {matching_labeled_row.iloc[0]['Score']}{ENDC}{ENDC}")
        print_border()

        exit_loop = False
        while True:
            score = input("Enter Input: ")
            try:
                # Go Back
                if score == 'A':
                    pages.pop()
                    return
                # Exit Program
                elif score == 'B':
                    exit()
                # Exit Scoring
                elif score == 'C':
                    exit_loop = True
                    break
                elif not_found:
                    labeled_paper = pd.DataFrame([[title, abstract, doi, int(score)]],  columns=['Title','Abstract','DOI','Score'])
                    labeled_df = pd.concat([labeled_df, labeled_paper])
                else:
                    labeled_df.loc[(labeled_df['Title'] == title) & (labeled_df['DOI'] == doi), 'Score'] = int(score)
                
                paper_i += 1
                labeled_df.reset_index(drop=True, inplace=True)
                labeled_df.to_csv(labeled_df_filepath, index=False)
                
                break
            except Exception:
                # No Input Entered (Skip)
                if not score:
                    paper_i += 1
                    break
                print(f"{FAIL}Must Enter 0, 1, 2, 3, or Nothing. Try Again.{ENDC}\n")
                continue

        if exit_loop:
            break

    print(f"{OKGREEN}Scored Papers{ENDC}")
    time.sleep(1)
    pages = ["Method"]

while True:
    if len(pages) == 0:
        page_menu()
    else:
        last_page = pages[-1]
        if last_page == "Method":
            page_method()
        elif last_page == "Method Load":
            page_method_load()
        elif last_page == "Method Search":
            page_method_search()
        elif last_page == "Preview":
            page_preview()
        elif last_page == "Score":
            page_score()