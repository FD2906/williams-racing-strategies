import fastf1
import pandas as pd

"""
KPI 2 - Relative Racecraft Performance

Strategic Question 2: "“In 2018 and 2019 qualifying sessions, where did Williams lose the most time relative to the fastest midfield performances, 
						and were these performance gaps more pronounced in technical versus power-focused sectors?”"

KPI 2 - Qualifying Sector Performance Delta: 
		Time deficit of Williams' fastest qualifying performance to the fastest midfield performance per sector, grouped by sector type.

Hypothesis 2: Williams' fastest qualifying sector times showed significantly larger deficits to midfield rivals 
			in technical sectors compared to power sectors, indicating fundamental aerodynamic and chassis setup limitations 
			rather than engine-related deficiencies.
"""


circuit_type = {
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

sector_type = {
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
races = list(circuit_type.keys()) # the 10 circuits
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
all_laps_df.to_csv("processed_data/all-laps.csv") # store loaded data to a new csv
print(f"Loaded {len(all_sessions)} sessions successfully.") # print success message
"""

# -------------------- DATA FORMATTING AND VALIDATION --------------------

# Set up a MultiIndex DataFrame for sector_type 
index = pd.MultiIndex.from_tuples(list(sector_type.keys()), names = ['race', 'sector'])
df_sector_type = pd.Series(list(sector_type.values()), index = index, name = 'sector_type')

#print(df_sector_type) # works

# Remove all '0 days' tags from all time columns in all_laps_df.csv

df = pd.read_csv("processed_data/all-laps.csv") # load the csv

time_columns = "Time,LapTime,PitOutTime,PitInTime,Sector1Time,Sector2Time,Sector3Time,Sector1SessionTime,Sector2SessionTime,Sector3SessionTime,LapStartTime".split(",")
# print(time_columns) # works

df.to_csv("processed_data/all-laps-cleaned.csv", index = False)

# -------------------- FURTHER STEPS REQUIRED --------------------

"""
1. Load processed data "all_laps_df_cleaned.csv" as DataFrame
	Ensure sector and lap time columns are clean and parseable (e.g. convert str to time data)
2. Filter by teams, valid laps, and each driver's fastest lap.
	Remember: filter accurate, fastest laps only - avoids external noise. 
3. Feature Engineering: 
	3.1. Generate comparison baseline - for each session, find the best/lowest sector times.
		Forms session-specific baseline for each sector.
	3.2 Calculate Williams' delta to the fastest midfield team, if Williams' isn't the fastest.
		E.g. delta = Williams sector time - fastest midfield sector time. Repeat for S1, S2, S3. 
			Store as s1_delta, s2_delta, s3_delta in seconds.

			Code needed for comparison: pd.to_timedelta(df['Sector1Time']).dt.total_seconds()

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
#print(df_best_midfield['LapTime'].dtype) # object - we need to convert

time_columns_2 = "LapTime,Sector1Time,Sector2Time,Sector3Time".split(",") # all time cols used
for col in time_columns_2:
		df_best_midfield[col] = pd.to_timedelta(df_best_midfield[col]) # convert time string to pd.timedelta dtype

#print(df_best_midfield.info()) # four time values are all timedelta64s - above code works
#print(df_best_midfield.head(10)) # peek first ten rows

# export to a csv
df_best_midfield.to_csv("processed_data/all-laps-best-midfield.csv")

# This approach provides all personal best times, which is ok, because
# more lap/sector time data can be used and aggregated for comparing and benchmarking averages

# ------------------- STEP 3: FEATURE ENGINEERING -------------------

"""
3. Feature Engineering: 
	3.1. Generate comparison baseline - for each session, find the best/lowest sector times.
		Forms session-specific baseline for each sector.

	3.2 Calculate Williams' delta to the fastest midfield team, if Williams' isn't the fastest.
		E.g. delta = Williams sector time - fastest midfield sector time. Repeat for S1, S2, S3. 
			Store as s1_delta, s2_delta, s3_delta in seconds.

			Code needed for comparison: pd.to_timedelta(df['Sector1Time']).dt.total_seconds()

	3.3 Attach circuit & sector type - use mapping to add a circuit_type column and label each circuit/sector as power, technical or balanced
"""

# 3.1 - generate comparison baseline.

# retrieve sectors and FL, going by year then by the 10 GPs. 
fastest_s1s = {}
fastest_s2s = {}
fastest_s3s = {}
fastest_laps = {} # a secondary stat to have - not essential

for year in [2018, 2019]:
	for race in circuit_type.keys(): # list of 10 chosen GPs

		# filter the df in each case for given year and race
		filtered_df = df_best_midfield[(df_best_midfield['Year'] == year) & (df_best_midfield['Race'] == race)]

		# Find fastest S1 row
		idx_fastest_s1 = filtered_df['Sector1Time'].idxmin()
		fastest_s1_row = filtered_df.loc[idx_fastest_s1]
		
		# Similarly for sectors 2 & 3:
		idx_fastest_s2 = filtered_df['Sector2Time'].idxmin()
		fastest_s2_row = filtered_df.loc[idx_fastest_s2]
		
		idx_fastest_s3 = filtered_df['Sector3Time'].idxmin()
		fastest_s3_row = filtered_df.loc[idx_fastest_s3]

		# find fastest lap row - a nice to have.
		idx_fastest_lap = filtered_df['LapTime'].idxmin()
		fastest_lap_row = filtered_df.loc[idx_fastest_lap]

		# use tuple (year, race) as a key, and a list driver, team, fastest lap/sector time as a value. 
		fastest_s1s[(year, race)] = [fastest_s1_row['Driver'], fastest_s1_row['Team'], fastest_s1_row['Sector1Time']]
		fastest_s2s[(year, race)] = [fastest_s2_row['Driver'], fastest_s2_row['Team'], fastest_s2_row['Sector2Time']]
		fastest_s3s[(year, race)] = [fastest_s3_row['Driver'], fastest_s3_row['Team'], fastest_s3_row['Sector3Time']]
		fastest_laps[(year, race)] = [fastest_lap_row['Driver'], fastest_lap_row['Team'], fastest_lap_row['LapTime']]



# print out results - midfield fastest S1s
"""
print("\nFastest midfield qualifying S1 times across 10 selected circuits, 2018 and 2019.\n")
for race, details in fastest_s1s.items():
	print(f"{race[0]} {race[1]}: DRIVER: {details[0]}. TEAM: {details[1]}. FASTEST S1 TIME: {details[2]}.")

# midfield fastest S2s
print("\nFastest midfield qualifying S2 times across 10 selected circuits, 2018 and 2019.\n")
for race, details in fastest_s2s.items():
	print(f"{race[0]} {race[1]}: DRIVER: {details[0]}. TEAM: {details[1]}. FASTEST S2 TIME: {details[2]}.")

# midfield fastest S3s
print("\nFastest midfield qualifying S3 times across 10 selected circuits, 2018 and 2019.\n")
for race, details in fastest_s3s.items():
	print(f"{race[0]} {race[1]}: DRIVER: {details[0]}. TEAM: {details[1]}. FASTEST S3 TIME: {details[2]}.")

# midfield FLs
print("\nFastest midfield qualifying lap times across 10 selected circuits, 2018 and 2019.\n")
for race, details in fastest_laps.items():
	print(f"{race[0]} {race[1]}: DRIVER: {details[0]}. TEAM: {details[1]}. FASTEST LAP TIME: {details[2]}.")
"""


# 3.2 - calculate williams' delta to the fastest midfield team

"""
NB: In our sample of fastest sector (and lap) times, Williams never achieved the fastest qualifying sector or lap
in any of the 10 selected circuits across 2018-2019.

This mean there's no need to account for edge cases where Williams 
sets the baseline for sector/lap delta calculations in the Fast-F1 derived dataset.
All team deltas represent a deficit to one of Haas, Renault, Force India, or Racing Point.

For my retrospective project, Williams will always be benchmarked against 
the session's best-performing rival, never themselves.

In all cases, delta = Williams' sector time - fastest midfield sector time.
"""

# filter Williams' fastest laps from df_best_midfield dataframe.
df_best_williams = df_best_midfield[df_best_midfield["Team"] == "Williams"]

df_best_williams.to_csv("processed_data/williams-best-laps.csv")

williams_fastest_s1s = {}
williams_fastest_s2s = {}
williams_fastest_s3s = {}
williams_fastest_laps = {}

for year in [2018, 2019]: # get years
	for race in circuit_type.keys(): # get all circuits

		# filter for just williams' times
		filtered_df_williams = df_best_williams[(df_best_williams['Year'] == year) & (df_best_williams['Race'] == race)]
		if filtered_df_williams.empty:
			print(f"\nNo valid and accurate Williams lap data for {year} {race} is available. Skipping.\n")
			continue

		# Find fastest S1 row
		idx_fastest_s1 = filtered_df_williams['Sector1Time'].idxmin()
		fastest_s1_row = filtered_df_williams.loc[idx_fastest_s1]

		# Similarly for sectors 2 & 3:
		idx_fastest_s2 = filtered_df_williams['Sector2Time'].idxmin()
		fastest_s2_row = filtered_df_williams.loc[idx_fastest_s2]
		
		idx_fastest_s3 = filtered_df_williams['Sector3Time'].idxmin()
		fastest_s3_row = filtered_df_williams.loc[idx_fastest_s3]

		# find fastest lap row - a nice to have.
		idx_fastest_lap = filtered_df_williams['LapTime'].idxmin()
		fastest_lap_row = filtered_df_williams.loc[idx_fastest_lap]

		# use tuple (year, race) as a key, and a list driver, team, fastest lap/sector time as a value. 
		williams_fastest_s1s[(year, race)] = [fastest_s1_row['Driver'], fastest_s1_row['Team'], fastest_s1_row['Sector1Time']]
		williams_fastest_s2s[(year, race)] = [fastest_s2_row['Driver'], fastest_s2_row['Team'], fastest_s2_row['Sector2Time']]
		williams_fastest_s3s[(year, race)] = [fastest_s3_row['Driver'], fastest_s3_row['Team'], fastest_s3_row['Sector3Time']]
		williams_fastest_laps[(year, race)] = [fastest_lap_row['Driver'], fastest_lap_row['Team'], fastest_lap_row['LapTime']]


# print out results - midfield fastest S1s
"""
print("\nWilliams' fastest midfield qualifying S1 times across 10 selected circuits, 2018 and 2019.\n")
for race, details in williams_fastest_s1s.items():
	print(f"{race[0]} {race[1]}: DRIVER: {details[0]}. TEAM: {details[1]}. FASTEST S1 TIME: {details[2]}.")

# midfield fastest S2s
print("\nWilliams' fastest midfield qualifying S2 times across 10 selected circuits, 2018 and 2019.\n")
for race, details in williams_fastest_s2s.items():
	print(f"{race[0]} {race[1]}: DRIVER: {details[0]}. TEAM: {details[1]}. FASTEST S2 TIME: {details[2]}.")

# midfield fastest S3s
print("\nWilliams' fastest midfield qualifying S3 times across 10 selected circuits, 2018 and 2019.\n")
for race, details in williams_fastest_s3s.items():
	print(f"{race[0]} {race[1]}: DRIVER: {details[0]}. TEAM: {details[1]}. FASTEST S3 TIME: {details[2]}.")

# midfield FLs
print("\nWilliams' fastest midfield qualifying lap times across 10 selected circuits, 2018 and 2019.\n")
for race, details in williams_fastest_laps.items():
	print(f"{race[0]} {race[1]}: DRIVER: {details[0]}. TEAM: {details[1]}. FASTEST LAP TIME: {details[2]}.")

# maybe refine the above section to make it into a function? applied twice already. 
"""

"""
CODE CAME ACROSS AN ISSUE: 
Stroll and Sirtokin's quali hotlaps from Spain & Hungary GP 2018 were filtered out by 'is_accurate == False'.
This may be due to changing weather conditions, exceeding track limits, penalised laps - so we are leaving out these sessions. 
"""


deltas = {} # intialise a dictionary for storing sector and lap deltas
fastest_team = {}
pct_slower = {}

# extract each (year, race) key from fastest_s1s
for (year, race) in williams_fastest_laps.keys(): # any williams dict works here, as it will have 18 observations instead of the 20 in fastest_laps
	# year, race are repeated across all dicts - you can use them on all. 
	overall_fastest_s1 = fastest_s1s[(year, race)][2].total_seconds() # retrieve time, and convert to seconds
	overall_fastest_s2 = fastest_s2s[(year, race)][2].total_seconds()
	overall_fastest_s3 = fastest_s3s[(year, race)][2].total_seconds()
	overall_fastest_lap = fastest_laps[(year, race)][2].total_seconds()

	s1_fastest = fastest_s1s[(year, race)][1] # accesses fastest_team
	s2_fastest = fastest_s2s[(year, race)][1] # accesses fastest_team
	s3_fastest = fastest_s3s[(year, race)][1] # accesses fastest_team
	s4_fastest = fastest_laps[(year, race)][1] # accesses fastest_team

	williams_fastest_s1 = williams_fastest_s1s[(year, race)][2].total_seconds() # do same for williams
	williams_fastest_s2 = williams_fastest_s2s[(year, race)][2].total_seconds() # do same for williams
	williams_fastest_s3 = williams_fastest_s3s[(year, race)][2].total_seconds() # do same for williams
	williams_fastest_lap = williams_fastest_laps[(year, race)][2].total_seconds() # do same for williams

	s1_delta = round(williams_fastest_s1 - overall_fastest_s1, 3) # calculate deltas
	s2_delta = round(williams_fastest_s2 - overall_fastest_s2, 3)
	s3_delta = round(williams_fastest_s3 - overall_fastest_s3, 3)
	lap_delta = round(williams_fastest_lap - overall_fastest_lap, 3)

	# percent slower = (delta / fastest_time) * 100
	s1_pct_slower = round((s1_delta / overall_fastest_s1) * 100, 3)
	s2_pct_slower = round((s2_delta / overall_fastest_s2) * 100, 3)
	s3_pct_slower = round((s3_delta / overall_fastest_s3) * 100, 3)
	lap_pct_slower = round((lap_delta / overall_fastest_lap) * 100, 3)
	
	# print(f"Williams' fastest S2 for {year} {race}: {williams_fastest_s2}; Overall fastest S2: {overall_fastest_s2}") # test line

	deltas[(year, race)] = [s1_delta, s2_delta, s3_delta, lap_delta] # append result to deltas dict
	fastest_team[(year, race)] = [s1_fastest, s2_fastest, s3_fastest, s4_fastest]
	pct_slower[(year, race)] = [s1_pct_slower, s2_pct_slower, s3_pct_slower, lap_pct_slower]


# display deltas and fastest teams
for key in deltas.keys():
    year, race = key
    s1_delta, s2_delta, s3_delta, lap_delta = deltas[key]
    team1, team2, team3, team4 = fastest_team[key]

    print(f"{year} {race}")
    print(f"S1 Delta: {s1_delta}; S2 Delta: {s2_delta}; S3 Delta: {s3_delta}; Lap Delta: {lap_delta}")
    print(f"S1 Fastest: {team1}; S2 Fastest: {team2}; S3 Fastest: {team3}; Lap Fastest: {team4}\n")

"""
In every analysed event, Williams set the slowest, or near-slowest times in qualifying.
The general trend is that Williams was consistently off the pace in all types of sectors.
Sessions with unreliable, unrepresentative, or weather-impacted data were excluded to ensure analysis only compares standard dry-qualifying performance.
(e.g. 2018 Spanish GP, 2018 Hungarian GP)
"""


# 3.3 - form a dataframe with sectors, deltas and their labelled sector types

records = [] # form a new list, to be made into a dataframe, to store data on deltas, sectors, sector types, fastest team information.

for (year, race), sector_deltas in deltas.items():
	# form a list to get [S1 team, S2 team, S3 team, S4 team]
	fastest_teams_for_session = fastest_team.get((year, race), [None, None, None, None]) 
	pct_slower_values = pct_slower.get((year, race), [None, None, None, None])

    # for each of the three sectors
	for sector_no, sector_delta in enumerate(sector_deltas[:3], start=1):
		fastest_team_for_sector = fastest_teams_for_session[sector_no - 1] # due to 0-based indexing
		pct_slower_val = pct_slower_values[sector_no - 1] # 0-based indeig

		record = {
			'year': year,
			'race': race,
			'sector': sector_no,
			'sector_delta': sector_delta,
			'pct_slower': pct_slower_val,
			'fastest_team': fastest_team_for_sector, 
			'sector_type': sector_type.get((race, sector_no)), # maps sector type
			'circuit_type': circuit_type.get(race) # maps circuit type
		}
		records.append(record)

# Optionally, convert to DataFrame for easier manipulation
df_labelled_sectors = pd.DataFrame(records)

print(df_labelled_sectors.info())
print(df_labelled_sectors)

# data now stored in df_labelled_sectors

# -------------------- 4. AGGREGATION AND GROUPING -----------------

"""
4. Aggregation & Grouping: 
	4.1 Melt data to long or MultiIndex format - each row should represent one sector per lap e.g. index1 = lap, index2 = sector (3 index2s for 1 index1)
		Could store as 'Year', 'Race', 'Driver', 'Sector', 'Sector_Delta', 'Sector_Type'
		Helpful for groupby(), visualisation, hypothesis tests.
	4.2. Aggregate by type - calculate average (mean/median) and spread (std/MAD/IQR) with sector delta for Williams through circuit_type/sector_type.
	4.3. Aggregate for other midfield rivals for broader comparison/sanity checks.
"""

# 4.1, converting to a long format - is done already. 

# 4.2 calculate mean and std dev of absolute time deltas by sector type
mean_std_deltas = (
    df_labelled_sectors
    .groupby('sector_type')['sector_delta']
    .agg(mean_delta='mean', std_delta='std')
    .reset_index()
    .round(3)
)

# calculate mean and std dev of percentage deltas by sector type
mean_std_pcts = (
    df_labelled_sectors
    .groupby('sector_type')['pct_slower']
    .agg(mean_pct_slower='mean', std_pct_slower='std')
    .reset_index()
    .round(3)
)

# print results clearly
print("\nBenchmarking Williams' Sector Performance (Time Delta in Seconds):\n")
print(mean_std_deltas.to_string(index=False))

print("\nBenchmarking Williams' Sector Performance (Percent Slower vs. Fastest Team):\n")
print(mean_std_pcts.to_string(index=False))

# 4.3 - repeating process for other midfield teams might take too long
# instead sanity check by counting fastest_team by sector/circuit type
# create a multiindex table, sort by n_fastest within each sector type, descending.

team_counts = df_labelled_sectors.groupby(['sector_type', 'fastest_team']).size().reset_index(name='n_fastest') # group by sector_type and fastest team

team_counts_multi = team_counts.set_index(['sector_type', 'fastest_team'])
team_counts_multi_sorted = team_counts_multi.groupby(level='sector_type', group_keys=False).apply(
    lambda x: x.sort_values('n_fastest', ascending=False)
)

print("\nWhich of the midfield teams were the fastest in 2018 and 2019 in which sector types?\n")
print(team_counts_multi_sorted)


# ---------------- 5 & 6. FINAL STEPS ---------------

"""
5. Identify and treat any outliers, if present. Label carefully if needed, or consider removing them.
6. Prepare for visualisation and hypothesis testing. 
	6.1 Export to CSV if needed. e.g each row: `Year`, `Race`, `Driver`, `Sector`, `Sector_Delta`, `Sector_Type`, `Circuit_Type`.
	6.2 For hypothesis testing: group sector deltas by sector type or use t-tests for statistical significance.
	6.3 For visualisation: use boxplots, heatmaps, or barplots.
"""

# 5. z-score and visual (boxplot) checks for outliers

import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.stats import zscore 

# z-score check (over +3 or under -3)
df_labelled_sectors['zscore'] = zscore(df_labelled_sectors['sector_delta'])

print("\nZ-score check for outliers - results:\n")
print(df_labelled_sectors[df_labelled_sectors['zscore'].abs() > 3])

# create a boxplot to spot outliers visually
sns.boxplot(
	x = 'sector_type', 
	y = 'sector_delta', 
	data = df_labelled_sectors
)
plt.title("Williams Sector Delta by Sector Type")
plt.show()

"""
Empty DataFrame from z-score check confirms no outlier values (using |zscore| > 3 test)
'is_accurate == True', non-deleted laps and manual session curation yielded a dataset free of extreme values.
No evidence of values that could distort summary statistics or bias hypothesis testing. 
Results reflect genuine and representative Williams versus midfield gaps.
"""

# 6. Export final labelled sectors dataframe to CSV.

df_labelled_sectors.to_csv("processed_data/williams-deltas-by-sector-type.csv") 