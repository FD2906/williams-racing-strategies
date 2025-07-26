import fastf1
import pandas as pd

"""
KPI 3 - Relative Racecraft Performance

Strategic Question 3: “In qualifying sessions between 2015 and 2019, where did Williams lose the most time relative to midfield competitors — 
                        and were these performance gaps more pronounced in technical versus power-focused circuits or sectors?”

KPI 3 - Qualifying Performance Delta: 
Average time lost by Williams to the best-performing midfield team per lap (or sector, where available), 
grouped by circuit and sector type (technical, balanced, power).

Hypothesis 3: Across 2015-2019, Williams lost more time relative to midfield rivals in low-speed, technical sectors 
                than in high-speed, straight sectors, highlighting a consistent weakness in cornering performance.
"""


circuit_type_manual = {
    "Italian Grand Prix": "power",
    "Monaco Grand Prix": "technical",
    "British Grand Prix": "balanced",
    "Belgian Grand Prix": "power",
    "Spanish Grand Prix": "balanced",
    "Singapore Grand Prix": "technical",
    "Brazilian Grand Prix": "power",
    "Hungarian Grand Prix": "technical",
    "Austrian Grand Prix": "power",
    "Japanese Grand Prix": "balanced"
}

sector_type_manual = {
    ("Italian Grand Prix", 1): "power", 
    ("Italian Grand Prix", 2): "power", 
    ("Italian Grand Prix", 3): "power", 

    ("Monaco Grand Prix", 1): "technical", 
    ("Monaco Grand Prix", 2): "technical", 
    ("Monaco Grand Prix", 3): "technical", 

    ("British Grand Prix", 1): "balanced", 
    ("British Grand Prix", 2): "balanced", 
    ("British Grand Prix", 3): "balanced", 

    ("Belgian Grand Prix", 1): "power", 
    ("Belgian Grand Prix", 2): "balanced", 
    ("Belgian Grand Prix", 3): "power", 

    ("Spanish Grand Prix", 1): "power", 
    ("Spanish Grand Prix", 2): "balanced", 
    ("Spanish Grand Prix", 3): "balanced", 
    
    ("Singapore Grand Prix", 1): "technical", 
    ("Singapore Grand Prix", 2): "technical", 
    ("Singapore Grand Prix", 3): "technical", 

    ("Brazilian Grand Prix", 1): "power", 
    ("Brazilian Grand Prix", 2): "balanced", 
    ("Brazilian Grand Prix", 3): "power", 

    ("Hungarian Grand Prix", 1): "balanced", 
    ("Hungarian Grand Prix", 2): "technical", 
    ("Hungarian Grand Prix", 3): "technical", 

    ("Austrian Grand Prix", 1): "power", 
    ("Austrian Grand Prix", 2): "power", 
    ("Austrian Grand Prix", 3): "power", 

    ("Japanese Grand Prix", 1): "technical", 
    ("Japanese Grand Prix", 2): "balanced", 
    ("Japanese Grand Prix", 3): "balanced"
}

# test code, to see if we can load multiple sessions and concatenate them into a single DataFrame.
# below code already executed and stored in processed_data/all_laps_df.csv
"""
races = list(circuit_type_manual.keys()) # the 10 circuits
years = [2018, 2019] # fast-f1 telemetry data only available 2018 onwards
session_type = 'Q'  # qualifying sessions only

all_sessions = []

for year in years:
    for race in races:
        session = fastf1.get_session(year, race, session_type)
        session.load(laps=True, telemetry=False)
        laps = session.laps
        laps['Year'] = year
        laps['Race'] = race
        all_sessions.append(laps)


all_laps_df = pd.concat(all_sessions, ignore_index=True)
all_laps_df.to_csv("processed_data/all_laps.csv") # store loaded data to a new csv
print(f"Loaded {len(all_sessions)} sessions successfully.") # print success message
"""

# -------------------- DATA FORMATTING AND VALIDATION --------------------

# Set up a MultiIndex DataFrame for sector_type_manual 
index = pd.MultiIndex.from_tuples(list(sector_type_manual.keys()), names = ['race', 'sector'])
df_sector_type_manual = pd.Series(list(sector_type_manual.values()), index = index, name = 'sector_type')

#print(df_sector_type_manual) # works

# Remove all '0 days' tags from all time columns in all_laps_df.csv

df = pd.read_csv("processed_data/all_laps.csv") # load the csv

time_columns = "Time,LapTime,PitOutTime,PitInTime,Sector1Time,Sector2Time,Sector3Time,Sector1SessionTime,Sector2SessionTime,Sector3SessionTime,LapStartTime".split(",")
# print(time_columns) # works

df.to_csv("processed_data/all_laps_cleaned.csv", index = False)

# -------------------- FURTHER STEPS REQUIRED --------------------

"""
1. Load processed data "all_laps_df_cleaned.csv" as DataFrame
    Ensure sector and lap time columns are clean and parseable (e.g. convert str to time data)
2. Filter by teams, valid laps, and each driver's fastest lap.
    Remember: filter fastest laps only - avoids external noise. 
    
        Example code: df_fastest = df.loc[df.groupby(['Year', 'Race', 'Driver'])['LapTime'].idxmin()]
        Then, use dropna(subset = [...]) or filter out == 0.0 after parsing times.

3. Feature Engineering: 
    3.1. Generate comparison baseline - for each session, find the best/lowest sector times for rivals
        Forms session-specific baseline for each sector.
    3.2 Calculate Williams' delta to the fastest midfield team. 
        E.g. delta = Williams sector time - fastest midfield sector time. Repeat for S1, S2, S3. Store as s1_delta, s2_delta, s3_delta in seconds.

            Example code: pd.to_timedelta(df['Sector1Time']).dt.total_seconds()

    3.3 Attach circuit & sector type - use mapping to add a circuit_type column and label each circuit/sector as power, technical or balanced
4. Aggregation & Grouping: 
    4.1 Melt data to long or MultiIndex format - each row should represent one sector per lap e.g. index1 = lap, index2 = sector (3 index2s for 1 index1)
        Could store as 'Year', 'Race', 'Driver', 'Sector', 'Sector_Delta', 'Sector_Type'
        Helpful for groupby(), visualisation, hypothesis tests.
    4.2. Aggregate by type - calculate average (mean/median) and spread (std/MAD/IQR) with sector delta for Williams through circuit_type/sector_type.
    4.3. Aggregate for other midfield rivals for broader comparison/sanity checks.
5. Identify and treat any outliers, if present. Label carefully if needed, or consider removing them.
6. Prepare for visualisation and hypothesis testing. 
    6.1 Export to CSV if needed. e.g each row: `Year`, `Race`, `Driver`, `Sector`, `Sector_Delta`, `Sector_Type`, `Circuit_Type`.
    6.2 For hypothesis testing: group sector deltas by sector type or use t-tests for statistical significance.
    6.3 For visualisation: use boxplots, heatmaps, or barplots.
"""

# ------------------- STEPS 1 & 2 - LOAD AND FURTHER PROCESS DATA -------------------
df = df[[
    'Unnamed: 0', 'Year', 'Race', 'Driver', 'DriverNumber', 'Team',
    'LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time', 
    'IsPersonalBest', 'Deleted', 'IsAccurate'
]] # keep relevant columns only

df.rename(columns = {"Unnamed: 0" : "Id"}, inplace = True) # rename first column to Id

# filter for just Williams, Racing Point, Force India, Haas, and Renault
df_midfield = df[df['Team'].isin(['Williams', 'Racing Point', 'Force India', 'Haas F1 Team', 'Renault'])] 

# ensures best and accurate laps (accurate in Fast-F1 means non-deleted).
df_best_midfield = df_midfield[(df_midfield['IsPersonalBest']) & (df_midfield['IsAccurate'] == True)] 

# reset index - df is now complete for further analysis
df_best_midfield = df_best_midfield.set_index('Id')

# time values - ensure these are of dtype Timedelta
print(df_best_midfield['LapTime'].dtype) # object - we need to convert

time_columns_2 = "LapTime,Sector1Time,Sector2Time,Sector3Time".split(",") # all time cols used
for col in time_columns_2:
        df_best_midfield[col] = pd.to_timedelta(df_best_midfield[col]) # convert time string to pd.timedelta dtype

print(df_best_midfield.info()) # four time values are all timedelta64s - above code works
print(df_best_midfield.head(10)) # peek first ten rows

# export to a csv
df_best_midfield.to_csv("processed_data/all_laps_best_midfield.csv")

# This approach provides all personal best times, which is ok, because
# more lap/sector time data can be used and aggregated for comparing and benchmarking averages

# ------------------- STEP 3 -------------------

"""
3. Feature Engineering: 
    3.1. Generate comparison baseline - for each session, find the best/lowest sector times for rivals
        Forms session-specific baseline for each sector.
    3.2 Calculate Williams' delta to the fastest midfield team. 
        E.g. delta = Williams sector time - fastest midfield sector time. Repeat for S1, S2, S3. Store as s1_delta, s2_delta, s3_delta in seconds.

            Example code: pd.to_timedelta(df['Sector1Time']).dt.total_seconds()
"""

# retrieve sessions, and go 
for year in [2018, 2019]:
    for race in circuit_type_manual.keys(): 
          pass