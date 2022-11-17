# # Record Linkage Parallelized
# This notebook is an example of a parallelized implementation of recording linkage using fuzzy matching in Python.
# * This notebook uses [fuzzywuzzy](https://github.com/seatgeek/fuzzywuzzy) for fuzzy matching which is an efficient implementation of Levensteihn string matching from Seatgeek
# * [Dask](https://dask.org/) allows us to scale the computationally expensive task of fuzzy string matching through parallelism

# This notebook takes 2 DataFrames that have no primary key to match on and applies fuzzy matching logic to return a new DataFrame that contains all of the info from the first Dataframe along with info from the second DataFrame for matched rows.

# ### Install Python Libraries
#!pip install fuzzywuzzy[speedup]
from IPython import get_ipython
import pandas as pd
from fuzzywuzzy import fuzz, process, string_processing, utils
import dask.multiprocessing
import dask.threaded
import dask.dataframe as dd
import math
import time
import numpy as np

# ### Import Data
# This example performs an approximate string match on company names from [NASDAQ, S&P500, NYSE exchanges](https://datahub.io/collections/stock-market-data) with companies names from the [SEC Edgar Company Database](https://www.sec.gov/edgar/searchedgar/accessing-edgar-data.htm).

nasdaq = pd.read_csv('nasdaq.csv')
sp500 = pd.read_csv('s_and_p_500.csv')
nyse = pd.read_csv('nyse.csv')
other = pd.read_csv('other.csv')
sec = pd.read_csv('sec_edgar_company_info.csv')

nasdaq['Stock Exchange'] = 'Nasdaq'
sp500['Stock Exchange'] = 'S&P 500'
nyse['Stock Exchange'] = 'NYSE'
other['Stock Exchange'] = 'Other'

stocks = nasdaq.append(sp500).append(nyse).append(other)
stocks = stocks.drop_duplicates(subset = 'Symbol')

# ### Pre-Processing
# Pre-processing strings in both datasets by replacing non-alphanumeric characters with whitespace, making strings lowercase, and stripping whitespace. This is done in a parallelized manner using dask.

def pre_process(text):
    processed = string_processing.StringProcessor.replace_non_letters_non_numbers_with_whitespace(text)
    processed = string_processing.StringProcessor.to_lower_case(processed)
    processed = string_processing.StringProcessor.strip(processed)
    return [processed, text]

def pre_process_parallelized(df, col):
    dmaster = dd.from_pandas(df, npartitions = dask.multiprocessing.multiprocessing.cpu_count())
    processed = dmaster[col].apply(lambda x: pre_process(x), meta = ('x','f8'))
    processed = processed.compute(scheduler = 'processes')
    return processed

def pre_processed_parallelized_df(df, col):
    clean = []
    orig = []
    
    processed = pre_process_parallelized(df, col)
    for i in processed:
        clean.append(i[0])
        orig.append(i[1])
        
    df_processed = pd.DataFrame(list(zip(clean, orig)), 
                      columns = [col + ' Clean', 'Orig'])
    
    df_processed = df.merge(df_processed, how = 'left', left_on = col, right_on = 'Orig')\
                     .drop(['Orig'], axis = 1)\
                     .drop_duplicates()
    return df_processed

stocks = pre_processed_parallelized_df(stocks, 'Company Name')
sec = pre_processed_parallelized_df(sec, 'Company Name')


# ### Full Match
# Once the pre-processing is complete, we want to first perform a full match on the cleaned company names to reduce the complexity of the fuzzy matching on remaining names.

stocks_full_match = stocks.merge(sec, how = 'inner', on = 'Company Name Clean', suffixes = (' Set A', ' Set B'))
print(len(stocks_full_match), 'full matches out of', len(stocks),
      '({:.1%})'.format(len(stocks_full_match)/len(stocks)))

# Separate unmatched stocks and SEC companies to use for fuzzy matching
stocks_not_matched = stocks[~stocks['Company Name Clean'].isin(stocks_full_match['Company Name Clean'])]
sec_not_matched = sec[~sec['Company Name Clean'].isin(stocks_full_match['Company Name Clean'])]

print(len(stocks_not_matched), 'stocks not matched out of', len(stocks),
      '({:.1%})'.format(len(stocks_not_matched)/len(stocks)))
print(len(sec_not_matched), 'SEC companies not matched out of', len(stocks),
      '({:.1%})'.format(len(sec_not_matched)/len(sec)))


# ### Fuzzy Match
# Extract the best match for each stock against the 660k SEC company names and use a score cutoff of 90% match to join the 2 datasets on approximate company name

# function to extract the best match using a score cut-off
def fuzzy_match(set_a, set_b, scorer, score_cutoff):
    return process.extractOne(set_a, set_b, scorer = scorer, score_cutoff = score_cutoff)

# function to parallelize fuzzywuzzy's extractOne function
# splits dataframes into dask dataframes equal to the number of cores of your CPU, parallelizes compute on the CPU cores
def fuzzy_match_parallelized(set_a, col_a, set_b, col_b, scorer, score_cutoff):
    dmaster = dd.from_pandas(set_a, npartitions = dask.multiprocessing.multiprocessing.cpu_count()) 
    match = dmaster[col_a].apply(lambda x: fuzzy_match(x, set_b[col_b], scorer, score_cutoff), meta = ('x','f8'))
    match = match.compute(scheduler = 'processes')
    return match

# use the results of fuzzy matching to join both datasets together
def fuzzy_merge(set_a, col_a, set_b, col_b, scorer, score_cutoff):
    matches = fuzzy_match_parallelized(set_a, col_a, set_b, col_b, scorer, score_cutoff)

    set_a_idx = []
    set_b_idx = []
    match_ratio = []

    for idx, i in enumerate(matches):
        if i is not None:
            set_a_idx.append(idx)
            set_b_idx.append(i[2])
            match_ratio.append(i[1])
        else:
            set_a_idx.append(idx)
            set_b_idx.append(np.nan)
            match_ratio.append(np.nan)

    match_df = pd.DataFrame(list(zip(set_a_idx, set_b_idx, match_ratio)), 
                          columns = ['set_a_idx', 'set_b_idx', 'Match Ratio'])

    df = set_a.merge(match_df, how = 'left', left_index = True, right_on = 'set_a_idx').merge(set_b, how = 'left', left_on = 'set_b_idx', right_index = True, suffixes = (' Set A', ' Set B'))

    df = df[[col_a + ' Set A', col_b + ' Set B', 'Match Ratio'] + [i for i in list(df.columns) if i not in [col_a + ' Set A', col_b + ' Set B', 'Match Ratio']]].drop(['set_a_idx', 'set_b_idx'], axis = 1)
    
    return df


# run the fuzzy match logic on the SEC and company data
scorer = fuzz.ratio # fuzzy-match scorer - Levensteihn distance
score_cutoff = 80
set_a = stocks_not_matched # pandas df for first set of entities
col_a = 'Company Name Clean' # column from set a to match on
set_b = sec_not_matched  # pandas df for second set of entities
col_b = 'Company Name Clean' # column from set b to match on

df = fuzzy_merge(set_a, col_a, set_b, col_b, scorer, score_cutoff)


df = df.append(stocks_full_match, sort = 'True')
df = df.drop_duplicates(subset = 'Symbol')
df = df.reset_index(drop = True)


print(len(df.dropna(subset = ['Company CIK Key'])), 'stocks matched out of', len(stocks),
      '({:.1%})'.format(len(df.dropna(subset = ['Company CIK Key']))/len(stocks)))