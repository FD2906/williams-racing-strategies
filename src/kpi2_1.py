import pandas as pd
import numpy as np

"""
Q2 / KPI 2 - 

Question 2: “To what extent did Williams benefit from safety car 
            or virtual safety car periods in pit strategy or position gains 
            during the 2015-2019 F1 seasons, and how can these opportunities be better exploited?”

KPI 2: 
- 2.1 Pit Stop Efficiency Score: Average pit stop duration and variance, 
    benchmarked against midfield teams (e.g. % sec compared to the mean and std. duration of others) 
    (can be done using the Kaggle dataset)

- 2.2 Safety Car Opportunity Capture: Positions gained/lost during safety car or VSC periods.
    (can only be done with Fast-F1)

Hypothesis 2: Williams gained fewer positions during safety car or VSC periods than midfield rivals between 2015-2019, 
            due in part to slower pit stops and reduced ability to defend or consolidate gained positions after race restarts.

Steps:
(KPI 2.1 - with dataset)

Goal: quantify how Williams and rivals performed in pit stops compared to *fastest* and *most consistent* teams
in any filtered scenario (e.g. season, GP, chaotic/non-chaotic/all race sessions)

1. Define a function with the optional filters implemented as function parameters
2. Group and aggregate the filtered data by constructor_ref and compute mean, std. dev, and pit stop count.
3. Benchmark by identifying and quantifying fastest team and most consistent team - smallest of mean and std
4. Compute % over benchmark by adding new columns.

(KPI 2.2 - with Fast-F1)


Next steps, in stage 4 - visualisation, with filters, and hypothesis testing using ttest_ind() and similar methods.
"""

df = pd.read_csv('processed_data/constructor-pit-stops-validated.csv')

def get_pit_stats(df: pd.DataFrame,
                    gp_year: int | list[int], # mandatory filter - must specify which year
                    gp_name: str | list[str] = None, # optional filter - by GP name
                    long_stop_flag: bool = None, # optional filter - by long stops or non-long stops only
                    chaotic_race_flag: bool = None, # optional filter - by chaotic or non-chaotic sessions only
                    verbose: bool = True): 
    """
    Steps:
    1. Filter by mandatory and optional filters from function parameters
    2. Group and aggregate filtered data to compute mean, median, std, MAD, pit count
    3. Return constructors by their mean, and std pit efficiency - ordered by mean desc. 

    Arguments:
    df -- DataFrame containing constructor pit stop data
    gp_year -- Single year or list of years to filter (**not optional**)
    gp_name -- Single GP name or list of GP names to filter (optional)
    long_stop_flag -- Filter to include or exclude long stops (optional)
    chaotic_race_flag -- Filter for chaotic vs clean sessions (optional)
    verbose -- If True, print filtering information (default: True)

    Return:
    A DataFrame with the mean and standard deviation of pit stops (in both ms and mm:ss:ms format), 
    along with lap counts, considering applied filters.
    """
     
    # 1. ---------- filter the data if parameters are provided ---------- 
    if gp_year is not None: 
        df = df[df['gp_year'].isin([gp_year] if isinstance(gp_year, int) else gp_year)]
        if verbose: 
            print(f"Filtering data for year(s): {gp_year}")
    if gp_name is not None: 
        df = df[df['gp_name'].isin([gp_name] if isinstance(gp_name, str) else gp_name)]
        if verbose: 
            print(f"Filtering data for GP name(s): {gp_name}")
    if long_stop_flag is not None: 
        df = df[df['long_stop_flag'].isin([long_stop_flag] if isinstance(long_stop_flag, bool) else long_stop_flag)]
        if verbose:
            print('Filtering data for long pit stops.' if long_stop_flag else 'Filtering data to exclude long pit stops.')
    if chaotic_race_flag is not None: 
        df = df[df['chaotic_race_flag'].isin([chaotic_race_flag] if isinstance(chaotic_race_flag, bool) else chaotic_race_flag)]
        if verbose:
            print('Filtering data for chaotic race sessions.' if chaotic_race_flag else 'Filtering data to exclude chaotic race sessions.')
    
    # 2. ---------- group and calculate statistical metrics ----------
    
    """
    as outliers have been left in, we are using median and MAD along with mean and std
    benchmarking in the next function will be applied on median and MAD found here
    """
    pit_stats = df.groupby('constructor_ref').agg(
        median_s = ('pit_duration_ms', lambda x: round(x.median() / 1000, 3)),
        mean_s = ('pit_duration_ms', lambda x: round(x.mean() / 1000, 3)),
        mad_s = ('pit_duration_ms', lambda x: round(np.median(np.abs(x - np.median(x))) / 1000, 3)),
        std_s = ('pit_duration_ms', lambda x: round(x.std() / 1000, 3)),
        n_pitstops = ('pit_duration_ms', 'count') 
    )

    return pit_stats.sort_values(by = ['median_s', 'mad_s', 'n_pitstops']) # return dataframe


# 3 & 4. Benchmarking function


def benchmark_against_best(df_agg: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Steps:
    1. Filter by mandatory and optional filters from function parameters
    2. Group and aggregate filtered data to compute mean, median, std, MAD, pit count
    3. Return constructors by their mean, and std pit efficiency - ordered by mean desc. 

    Arguments:
    df.agg -- Aggregated dataframe containing pit stop statistics
    verbose -- boolean value controlling if additional results are printed, e.g. winning teams

    Return:
    A DataFrame identifying the baseline constructor in both median and MAD pit stop times
    """

    # 1. -------- Benchmark teams against the fastest and most consistent values --------
    df_agg = df_agg.reset_index()
    benchmark_stats = df_agg[['constructor_ref', 'median_s', 'mad_s', 'n_pitstops']].copy() 

    fastest = benchmark_stats['median_s'].min() # find fastest 
    most_consistent = benchmark_stats['mad_s'].min() # find most consistent

    benchmark_stats['slower_by_s'] = (
        benchmark_stats['median_s'] - fastest
    )
    
    benchmark_stats['percent_slower'] = (
        round(((benchmark_stats['median_s'] - fastest) / fastest) * 100, 2)
    )

    benchmark_stats['percent_less_consistent'] = (
        round(((benchmark_stats['mad_s'] - most_consistent) / most_consistent) * 100, 2)
    )

    # reorder columns
    benchmark_stats = benchmark_stats[['constructor_ref', 'median_s', 'slower_by_s', 'percent_slower', 'mad_s', 'percent_less_consistent', 'n_pitstops']] 

    # 2. ---------- Find fastest and most consistent teams ----------

    # reference column values
    fastest_team_ref = benchmark_stats.loc[benchmark_stats['percent_slower'] == 0, 'constructor_ref'].values[0]
    most_consistent_team_ref = benchmark_stats.loc[benchmark_stats['percent_less_consistent'] == 0, 'constructor_ref'].values[0]

    # find the actual names, based on the global dataframe - if verbose True
    if verbose:
        fastest_team = df.loc[df['constructor_ref'] == fastest_team_ref, 'constructor'].unique()[0] 
        most_consistent_team = df.loc[df['constructor_ref'] == most_consistent_team_ref, 'constructor'].unique()[0]

        print(f"\nFastest team (median): {fastest_team}")
        print(f"Most consistent team (MAD): {most_consistent_team}")

    # 3. Return dataframe

    return benchmark_stats.sort_values(by = ['percent_slower', 'percent_less_consistent']).set_index('constructor_ref')


# --------- Testing ---------


print("\n ----- Test 1: Filtered pit stop data - 2015 to 2019, exclude long stops and chaotic races. ----- \n")
test1 = get_pit_stats(df, [2015, 2016, 2017, 2018, 2019], long_stop_flag = False, chaotic_race_flag = False, verbose = False) 
print(test1)
test1_benchmarking = benchmark_against_best(test1)
print("\n")
print(test1_benchmarking)
print("\n")


print("\n ---------------------- Test 2: Non-filtered pit stop data - 2015 to 2019. ----------------------\n")
test2 = get_pit_stats(df, [2015, 2016, 2017, 2018, 2019], verbose = False) # all results 
print(test2)
test2_benchmarking = benchmark_against_best(test2)
print("\n")
print(test2_benchmarking)
print("\n")
