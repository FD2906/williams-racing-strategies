import pandas as pd

"""
Q4 / KPI 4 - Driver Lap Time Consistency Index

Question 4: "How does the standard deviation of lap times for each Williams driver during a race 
            compare to their teammate across a season, and what interventions can improve consistency?"

KPI 4: Driver Lap Time Consistency Index - Lap time standard deviation per driver, per race.

Hypothesis 4: Rookie or less experienced Williams drivers had significantly higher lap time variance than 
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

def get_laptime_consistency(df: pd.DataFrame, experience: str, year:int) -> pd.DataFrame:
    """
    Steps required:
    1. Filter the DataFrame for the specified experience level and year.
    2. Group by experience level and year and calulate the standard deviation of lap times.
    3. Convert lap times from milliseconds to mm:ss:ms format.
    4. Store results in a new DataFrame and return

    Arguments:
    df -- DataFrame containing lap time data
    experience -- 'rookie' or 'experienced' to filter the DataFrame
    year -- int representing the year to filter the DataFrame

    Return:
    A DataFrame with the standard deviation of lap times for the specified experience level and year.
    """

    # Filter the DataFrame for the specified experience level and year
    filtered_df = df[(df['rookie_or_experienced'] == experience) & (df['year'] == year)]

    # Group by driver and calculate the standard deviation of lap times
    grouped_df = filtered_df.groupby('driver')['lap_time_ms'].std().reset_index()

    # Convert lap times from milliseconds to mm:ss:ms format
    grouped_df['lap_time_std'] = grouped_df['lap_time_ms'].apply(
        lambda x: f"{int(x // 60000):02}:{int((x % 60000) // 1000):02}:{int(x % 1000):03}"
        # :02 and :03 ensure that the minutes and seconds are always two or three digits respectively.
    )

    return grouped_df[['driver', 'lap_time_std']]

# test the function
rookie_lap_time_std = get_laptime_consistency(df, 'rookie', 2015)
experienced_lap_time_std = get_laptime_consistency(df, 'experienced', 2015)

print("Rookie Lap Time Standard Deviation (2015):")
print(rookie_lap_time_std)

print("\nExperienced Lap Time Standard Deviation (2015):")
print(experienced_lap_time_std)