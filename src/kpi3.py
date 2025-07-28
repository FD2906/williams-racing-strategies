import pandas as pd

"""
Q3 / KPI 3 - Driver Lap Time Consistency Index

Question 3: "How does the standard deviation of lap times for each Williams driver during a race 
            compare to their teammate across a season, and what interventions can improve consistency?"

KPI 3: Driver Lap Time Consistency Index - Lap time standard deviation per driver, per race.

Hypothesis 3: Rookie or less experienced Williams drivers had significantly higher lap time variance than 
            their teammates during races in the 2015-2019 seasons, suggesting lower in-race consistency 
            due to inexperience or adaptability challenges.

Steps:
1. Write a general function grouping drivers by df['rookie_or_experienced'], and the std dev of df['lap_time_ms']
2. Add a column to each converting lap_time_ms to mm:ss:ms
3. Compare the two dataframes with each other.

Next steps, in stage 4 - visualisation, with filters, and hypothesis testing using ttest_ind() and similar methods.
"""

df = pd.read_csv('processed_data/driver-lap-times-validated.csv') # load the data

# -------------------------------------------------------------------------------------------------------- # 
# 1. Aggregation function

def get_laptime_consistency(
        df: pd.DataFrame, 
        experience_level: str = None,
        year: int | list[int] = None, 
        gp_name: str | list[str] = None, 
        verbose: bool = True) -> pd.DataFrame:
    """
    Steps:
    1. Apply optional filters for experience level, year, and GP name.
    2. Drop missing or invalid lap times.
    3. Group by experience level and calculate the mean and standard deviation of lap times in milliseconds.
    4. Count the number of laps for each experience level.
    5. Convert mean and standard deviation lap times from milliseconds to mm:ss:ms format.
    6. Merge results into a summary DataFrame and rename columns for clarity.

    Arguments:
    df -- DataFrame containing lap time data
    experience_level -- 'rookie' or 'experienced' to filter by experience level (optional)
    year -- Single year or list of years to filter (optional)
    gp_name -- Single GP name or list of GP names to filter (optional)
    verbose -- If True, print filtering information (default: True)

    Return:
    A DataFrame with the mean and standard deviation of lap times (in both ms and mm:ss:ms format), 
    along with lap counts for each experience level, considering optional filters.
    """

    # 1. ---------- filter the data if parameters are provided ---------- 
    if experience_level is not None: 
        df = df[df['rookie_or_experienced'] == experience_level]
        if verbose: 
            print(f"Filtering data for experience level: {experience_level}")
    if year is not None: 
        df = df[df['gp_year'].isin([year] if isinstance(year, int) else year)]
        if verbose: 
            print(f"Filtering data for year(s): {year}")
    if gp_name is not None: 
        df = df[df['gp_name'].isin([gp_name] if isinstance(gp_name, str) else gp_name)]
        if verbose: 
            print(f"Filtering data for GP name(s): {gp_name}")

    # 2. ---------- drop missing or invalid times, just in case ----------
    df = df[df['lap_time_ms'] > 0]

    # ---------- 3. group and calculate statistical metrics ---------- 
    times_in_ms  = df[['rookie_or_experienced', 'lap_time_ms']] # select the relevant columns from df

    # group by experience level and calculate mean and standard deviation lap time
    grouped_by_experience = times_in_ms.groupby('rookie_or_experienced').agg(
        mean_lap_time_ms=('lap_time_ms', 'mean'),
        std_dev_lap_time_ms=('lap_time_ms', 'std')
    ).reset_index()

    # count number of laps for each experience level for statistical testing
    n_laps = times_in_ms.groupby('rookie_or_experienced').size().reset_index(name='n_laps')

    # merge counts in the main summary dataframe
    grouped_by_experience = pd.merge(grouped_by_experience, n_laps, on='rookie_or_experienced')

    # ---------- 4. convert ms to mm:ss:ms ----------
    grouped_by_experience['mean_lap_time'] = grouped_by_experience.apply(
        lambda time: f"{int(time['mean_lap_time_ms'] // 60000):02}:{int((time['mean_lap_time_ms'] % 60000) // 1000):02}.{int(time['mean_lap_time_ms'] % 1000):03}",
        axis=1 
    )
    grouped_by_experience['std_dev_lap_time'] = grouped_by_experience.apply(
        lambda time: f"{int(time['std_dev_lap_time_ms'] // 60000):02}:{int((time['std_dev_lap_time_ms'] % 60000) // 1000):02}.{int(time['std_dev_lap_time_ms'] % 1000):03}",
        axis=1 
    )

    # ---------- 5. rename columns and return result ----------
    grouped_by_experience = grouped_by_experience.rename(columns={ # rename columns for clarity
        'rookie_or_experienced': 'experience_level',
        'mean_lap_time_ms': 'mean_ms',
        'mean_lap_time': 'mean_formatted',
        'std_dev_lap_time_ms': 'std_dev_ms',
        'std_dev_lap_time': 'std_dev_formatted'
    })
    return grouped_by_experience[['experience_level', 'mean_ms', 'mean_formatted', 'std_dev_ms', 'std_dev_formatted', 'n_laps']]

print("\n")
print(get_laptime_consistency(df, 
                              year = [2017, 2018, 2019], 
                              gp_name = ['Monaco Grand Prix', 'Hungarian Grand Prix', 'Singapore Grand Prix']
                              ))  # Example usage of the function
print("\n")