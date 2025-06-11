# Inventory
This file explains what each dataset contains, meaning what types of papers they
contain. Each dataset contains the columns: "Title", "DOI", "Abstract", and "Score".
The "Score" column stores the score for each paper. What the score is and what it 
means is determined by the method.

For the datasets that were small enough, I created another dataset wherein it was filtered.
This should be done for all datasets, but my GPU doesn't have enough memory and I have no
access to HiPerGator. A star by a dataset name means that there is also a filtered version.

## A: 
This dataset consists of papers found with the 1st search level 
("trait" and "phenotype"). As the 1st search level is broad, there are a 
lot of papers that are not related to ecology. Keep in mind that 30k papers 
were found, but I was limited at 10k.

## SubA*:
This dataset contains a sample of A. A sample was used so that the filter
could be applied because A is too large and my GPU doesn't have enough 
memory. I need to try HiPerGator but I can't access it for some reason.

## B:
This dataset consists of papers found with the 1st and 2nd search levels. 
This is also a large dataset, but it's not as large as A. The filter still
cannot be used on this dataset.

## SubB*:
This dataset contains a sample of B.

## C*:
This dataset consists of papers found with the 1st, 2nd, and 3rd search levels.
This is a small dataset.

## D*:
This dataset consists of papers found with the 1st, 2nd, and 3rd search levels.
However, to produce more papers, the 3rd search level was broken up into 
individual words instead of phrases.

## Examples:
This dataset consists of papers that I found in the research proposal.

## First:
This dataset consists of the papers that I first turned in. This dataset
has already been filtered at a previous time. However, it used the older
zero-shot classification model.

## Baseline-1:
This dataset uses the first 30 rows of First. These 30
rows are also scored and is thus used as a baseline to evaluate the
various methods.