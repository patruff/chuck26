# Film ranking

The committed list ranks films released in 1990 through 2015 inclusive by number of ratings in the fixed MovieLens 32M snapshot. Ties use the numeric MovieLens ID. The CSV preserves the original MovieLens title and year. The generation title puts English articles first, removes parenthetical aliases and punctuation-only dashes, and speaks ampersands as “and”.

Run `python pipeline/film_lists/rank_movielens.py pipeline/film_lists/top200_1990_2015.csv` to reproduce the ranking. The script verifies the release archive before counting its 32,000,204 ratings. [GroupLens release](https://grouplens.org/datasets/movielens/32m/) · [Data format, citation and usage license](https://files.grouplens.org/datasets/movielens/ml-32m-README.html)

The film-list data is distributed under the MovieLens 32M usage license, including its attribution and research-use conditions. See the linked usage license for the complete terms. The University of Minnesota and GroupLens do not endorse this project. F. Maxwell Harper and Joseph A. Konstan (2015), *The MovieLens Datasets: History and Context*. [DOI](https://doi.org/10.1145/2827872)
