import pandas as pd

"""
Q1 / KPI 1 - Grid-to-Finish Delta 

Question 1: "Which circuits saw Williams lose the most positions 
    from race start to finish during the 2015-2019 F1 seasons, 
    and what track characteristics explain these losses to inform targeted setup and strategy adjustments?”

KPI 1: Grid-to-Finish Position Delta - Average positions gained/lost from the start to end of race.

Hypothesis 1: Williams' grid-to-finish position delta was significantly worse at high-downforce, technical circuits between 2015-2019 
    compared to midfield rivals, likely due to cornering limitations in car performance that reduced overtaking and defending capabilities.

Steps:
Calculate the delta between the grid position and the finish position for each driver.
Can be scaled up to apply to constructors by aggregating the deltas of all drivers involved with a constructor.

1. Retrieve driver-level delta first, found in df_results['grid_delta']. 
2. Aggregate the deltas to get the constructor-level delta - df['is_williams'] = True for just Williams drivers, 
    or group by df['constructor_name'] for all drivers in a constructor.
3. Calculate the average delta for Williams drivers and rival constructors on all tracks.

Rest of hypothesis 1: dealing with high-downforce tracks, *Monaco, Singapore and Hungoraring*, three of the selected ten circuits.

4. Apply constructor-level deltas, but filter for high-downforce tracks by checking if gp_name falls into a predefined list, 
    ['Monaco Grand Prix', 'Singapore Grand Prix', 'Hungarian Grand Prix'].
5. Calculate the average delta for Williams and rival constructors on these tracks.

Next steps, in stage 4 - hypothesis testing using ttest_ind() and similar methods.
"""

df = pd.read_csv('processed_data/grid-to-finish-validated.csv') # load the data

# -------------------------------------------------------------------------------------------------------- # 

# step 1 - retrieve driver-level delta
def get_driver_level_delta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retrieve the grid-to-finish delta for each driver.
    Add a column 'gained_or_lost' to indicate if the driver gained or lost positions.
    Add a column 'num_places' to indicate placed gained/lost - this is the absolute value of the delta 

    Arguments:
    df (pd.DataFrame): The dataframe containing the grid-to-finish data.

    Returns:
    pd.DataFrame: A dataframe containing the driver name, GP year, GP name, 
    grid delta, gained or lost status, and number of places.
    """
    df['gained_or_lost'] = df['grid_delta'].apply(lambda x: 'lost' if x < 0 else 'gained')
    df['num_places'] = df['grid_delta'].abs()

    return df[['driver_name', 'gp_year', 'gp_name', 'grid_delta', 'gained_or_lost', 'num_places']]


# print(get_driver_level_delta(df)) # test code

# -------------------------------------------------------------------------------------------------------- # 

# step 2 - aggregate the driver-level deltas to get the constructor-level delta
# create a function with a 'constructor_ref' parameter. default argument is 'williams', otherwise it will be the constructor name passed in.

# get all unique constructor reference names from the dataframe
# print(df['constructor_ref'].unique().tolist())
# results are ['williams' 'renault' 'haas' 'force_india' 'racing_point'] - williams, and its midfield rivals

def get_constructor_level_delta(df: pd.DataFrame, constructor_ref: str = 'williams') -> pd.DataFrame:
    """
    Group by constructor_ref, and calculate the mean of grid_delta for each constructor.
    
    Arguments:
    df (pd.DataFrame): The dataframe containing the grid-to-finish data.
    constructor_ref (str): The constructor reference name to filter by. Default is 'williams'.

    Returns:
    pd.DataFrame: A dataframe containing the average grid-to-finish delta for the specified constructor.
    columns are 'constructor_ref', 'gp_year', 'gp_name', 'avg_grid_delta'.
    """

    # filter the dataframe for the specified constructor
    df_constructor = df[df['constructor_ref'] == constructor_ref]

    # group by gp_year and gp_name, and calculate the mean of grid_delta
    df_constructor_grouped = df_constructor.groupby(['gp_year', 'gp_name']).agg(
        avg_grid_delta=('grid_delta', 'mean')
    ).reset_index()

    # add a column for the constructor reference name
    df_constructor_grouped['constructor_ref'] = constructor_ref

    df_constructor_grouped['gained_or_lost'] = df_constructor_grouped['avg_grid_delta'].apply(lambda x: 'lost' if x < 0 else 'gained')
    df_constructor_grouped['num_places'] = df_constructor_grouped['avg_grid_delta'].abs()

    return df_constructor_grouped[['constructor_ref', 'gp_year', 'gp_name', 'avg_grid_delta', 'gained_or_lost', 'num_places']]

# print(get_constructor_level_delta(df).head()) # test on williams, it being the default constructor for the function

for constructor in df['constructor_ref'].unique().tolist():
    print(f"Constructor: {constructor}")
    print(get_constructor_level_delta(df, constructor_ref=constructor).head(5)) # test on all constructors, preview head of each dataframe
    print("\n")  # add a newline for better readability

# -------------------------------------------------------------------------------------------------------- #

# step 3 - calculate the average delta for Williams drivers and rival constructors on all tracks

def get_average_delta_all_tracks(df: pd.DataFrame) -> pd.DataFrame: 
    """
    Using the function get_constructor_level_delta, which calculates the average grid-to-finish delta for a constructor and returns a dataframe,
    this function will calculate the average grid-to-finish delta for all constructors on all tracks.

    We need to loop through a list of midfield constructors, and apply the get_constructor_level_delta function to each constructor.
    Then we will extract and concatenate the results into a single dataframe.

    Arguments:
    df (pd.DataFrame): The dataframe containing the grid-to-finish data.

    Returns:
    pd.DataFrame: A dataframe containing the average grid-to-finish delta for all constructors on all tracks.
    Columns are 'constructor_ref', 'avg_grid_delta_year'
    (avg_grid_delta_year is the average of avg_grid_delta for each constructor across all tracks. 
    This is negative, indicating lost positions, or positive, indicating gained ones)
    """

    # initalise a new dataframe to store the results and to return
    df_all_constructors = pd.DataFrame(columns=['constructor_ref', 'avg_grid_delta_year'])

    # get all constructor reference names from the dataframe
    constructor_refs = df['constructor_ref'].unique().tolist()

    for constructor in constructor_refs:
        df_constructor = get_constructor_level_delta(df, constructor) # gets constructor level deltas by race
        avg_grid_delta_year = df_constructor['avg_grid_delta'].mean() # get year-long average of avg_grid_delta for each constructor across all tracks
        df_all_constructors = pd.concat(
            [df_all_constructors, pd.DataFrame({'constructor_ref': [constructor], 'avg_grid_delta_year': [avg_grid_delta_year]})], ignore_index=True
            ) # add the results to the dataframe
        
    return df_all_constructors


#print(get_average_delta_all_tracks(df)) # test the function


# the issue with the function above is that it gets the average across all tracks, all 5 years between 2015-2019. 
# we need to first build in a filter to the function to filter by year - in the form of an argument
# stakeholders want to see the average delta for each constructor by year, not across all years
# this will later help with data visualisation, as well as hypothesis testing


def get_average_constructor_delta_by_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    An improved version of the above function, now providing a breakdown of average grid delta by year.

    Arguments:
    df (pd.DataFrame): The dataframe containing the grid-to-finish data.
    year (int): The year to view average constructor deltas.

    Returns:
    pd.DataFrame: A dataframe containing the average grid-to-finish delta for all constructors on all tracks.
    Columns are 'constructor_ref', 'year', 'avg_grid_delta_year'
    (avg_grid_delta_year is the average of avg_grid_delta for each constructor across all tracks. 
    This is negative, indicating lost positions, or positive, indicating gained ones)
    """

    results = [] # list to store each row's dict - passed into pd.DataFrame on function return

    # get a list of constructor reference names from the main dataframe
    constructor_refs = df['constructor_ref'].unique().tolist()

    for constructor in constructor_refs:
        df_constructor = get_constructor_level_delta(df, constructor)  # gets constructor level deltas by race
        df_constructor_year = df_constructor[df_constructor["gp_year"] == year]  # filter for race deltas of the year specified
        avg_grid_delta_year = df_constructor_year["avg_grid_delta"].mean() # get the mean delta throughout the entire year

        if not pd.isna(avg_grid_delta_year): # prevent NaN rows - as at least one constructor did not exist in each given year 2015-2019
            results.append({
                "constructor_ref": constructor,
                "year": year,
                "avg_grid_delta_year": avg_grid_delta_year
            })

    return pd.DataFrame(results).sort_values(by = 'avg_grid_delta_year', ascending=False).reset_index(drop=True) # sort by avg delta, and reset index


# view average constructor deltas between 2015-2019
for year in range(2015, 2020):
    print(f"Year: {year}")
    print(get_average_constructor_delta_by_year(df, year))
    print("\n") # new line for better legibility


# -------------------------------------------------------------------------------------------------------- # 

# step 4 and 5
#4. Apply constructor-level deltas, but filter for high-downforce tracks by checking if gp_name falls into a predefined list, 
    #['Monaco Grand Prix', 'Singapore Grand Prix', 'Hungarian Grand Prix'].
#5. Calculate the average delta for Williams and rival constructors on these tracks.

df_high_downforce = df[df['gp_name'].isin(['Monaco Grand Prix', 'Singapore Grand Prix', 'Hungarian Grand Prix'])].reset_index(drop=True)

df_high_downforce.to_csv('processed_data/delta-high-downforce.csv') # for observation

# print(df_high_downforce)

# quick sanity checks before analysis
# 1. checking team-level row counts:
#print(df_high_downforce.groupby("constructor")["grid_delta"].count().sort_values(ascending=False))

# 2. per-circuit distribution - checking for over-representation
#rint(df_high_downforce['gp_name'].value_counts())

# Focusing on the three high-downforce, technical tracks, run get_average_constructor_delta_by_year function
# on df_high_downforce

print("Constructor-level grid-to-finish position delta, by year, on high-downforce & technical tracks.")
print("2015-2019. Monaco, Singapore, Hungarian GPs. Williams, Renault, Haas, Racing Point/Force India")
print("Note: 'avg_grid_delta_year'")
print("'+' means a constructor, on average, gained positions in-race compared to their starting position.")
print("'-' means a constructor typically lost positions compared to their starting position.\n")
for year in range(2015, 2020):
    print(f"Year: {year}")
    print(get_average_constructor_delta_by_year(df_high_downforce, year))
    print("\n") # new line for better legibility

