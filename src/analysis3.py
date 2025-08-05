# load final dataset used in 3.2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, mannwhitneyu

df = pd.read_csv("/Users/frankdong/Documents/Analytics Local/williams-racing-strategies/processed_data/laptimes_std.csv")

print(df.head())
print(df.info())

"""
EDA - let the data speak for itself through visualisations. 
Get to know the data both visually and statistically, lay the groundwork for analysis and hypothesis testing.

1. Start with core, rough visualisations. 
Histograms, boxplots, violin plots, scatterplots, barplots.  

2. Save all interesting "candidate" plots, output and save by exporting as PNG for review.
Plots may reveal surprising outliers, odd clusters, or clear trends

3. Annotate and document observations in Markdown
Write brief markdown notes next to each PNG. Note patterns, anomalies, large/small group sizes, etc.

4. Identify outliers, check sample sizes, assess normality. 
Key for hypothesis testing - boxplots and histograms help spot outliers and skew.
Check sample sizes with .groupby() or .value_counts() - are all groups (teams, circuits, years) large enough for statistical tests?

5. Iterate and refine - select the most informative charts for polish, annotation, and inclusion in Tableau

"""

# --- Steps 1, 2, 3. Rough Viz, save interesting plots, annotate observations ---

print(df)

# 'laptime consistency index' KPI now stored as laptime_std_ms. 

print(df['rookie_or_experienced'].value_counts())
print("\n")
# N(experienced) = 89. N(rookie) = 50. n > 30, which is good. 

# -------------------- 1: Delta boxplot of grid delta by sector ------------------

# set up new figure and size - taller boxplot helps compare heights 
plt.figure(figsize=(9, 9))

# create boxplot
consistency_by_experience = sns.boxplot(
    x = 'rookie_or_experienced', 
    y = 'laptime_std_ms', 
    data = df, 
    hue = 'rookie_or_experienced', 
    palette = 'Set2', # unbiased colour set for boxplot visualisation
    order = ['experienced', 'rookie'] # order the boxes this way
)

# set title, xlabel and ylable
consistency_by_experience.set_title("Lap Time Consistency (Standard Deviation) by Experience Level")
consistency_by_experience.set_xlabel('Experience Level')
consistency_by_experience.set_ylabel('Standard Deviation (ms)')

plt.grid(linewidth = 0.25)
plt.show()

"""
This boxplot breaks down Williams' lap time consistency, the standard deviation of all lap times
recorded in a race session by an experienced or rookie driver. 

The box captures the middle 50% of lap time standard deviations (the interquartile range or IQR), 
while the median line indicates the typical lap time consistency for each driver skill level. 

Experienced drivers demonstrate a slightly superior consistency compared to rookie drivers, with 
a median laptime standard deviation of 3600 ms or 3.6 s, compared to the 4050 ms or 4.05 s of rookies.

However, experienced drivers experience a more variable range of results, in both its box and whiskers.
It's box is larger compared to that of the rookies - 50% of values fall between 2.37 - 5.92s, compared to 
2.71 - 5.24 s of rookies. 
The 'experienced' boxplot also has a larger whisker range between 0.25 - 11.00 s and two outliers between 12 and 14 s.
These outliers need to be excluded in a final t-test.
"""

# ------------------- 2: Histplots for normality --------------------

# plot a grid of histograms, with each of the three representing the sector type
grid = sns.FacetGrid(
    df, 
    col = 'rookie_or_experienced', 
    col_order = ['experienced', 'rookie'],
    sharex = True, sharey = True, 
    height = 4, aspect = 1
)
grid.map(
    sns.histplot, 
    'laptime_std_ms', 
    kde=True, 
    stat='count', 
    bins=15, 
    color='royalblue'
)

# annotate counts to each plot on the grid
for ax, experience in zip(grid.axes.flat, ['experienced', 'rookie']):
    n = df[df['rookie_or_experienced'] == experience].shape[0]
    ax.text(0.95, 0.95, f'n = {n}', ha='right', va='top', transform=ax.transAxes,
            fontsize=12, bbox=dict(boxstyle='round', alpha=0.2))
    ax.set_xlabel('Standard Deviation (ms)')
    ax.set_ylabel('Count')
    ax.set_title(f'{experience.capitalize()} Drivers')

plt.suptitle("Williams' Laptime Consistency Distributions by Driver Experience Level", y=1.08, fontsize=16)
plt.tight_layout()
plt.show()

"""
Rookie drivers histogram showcases a pattern which mirrors a normal distribution curve - 
making observations fairly reliable and suitable for a t-test. 

However, due to the two large outliers, the experienced drivers histogram features a long right tail, 
and is heavily skewed left. We will now label and remove these outliers before carrying out tests.
"""


# ---------------- 3: Experienced Driver Histogram, no outliers ----------------

# 1. filter for experienced williams drivers
df_no_outliers = df.copy()

# 2. identify the two largest outliers in lap time standard deviation per race/driver
largest_two_outliers = df_no_outliers[df_no_outliers['rookie_or_experienced'] == 'experienced'].nlargest(2, 'laptime_std_ms')

# 3. (optional) display the details of these outliers for reporting
print("Removed outliers (for annotation):")
print(largest_two_outliers[['gp_year', 'gp_name', 'driver_name', 'laptime_std_ms']])

# 4. drop the two largest outliers from your experienced drivers DataFrame
df_no_outliers = df_no_outliers.drop(largest_two_outliers.index)

# 5. replot the histogram with KDE for the cleaned data
plt.figure(figsize=(12, 8)) # set up a new figure

no_outliers_histplot = sns.histplot(
    df_no_outliers['laptime_std_ms'], 
    kde=True, 
    bins=15, 
    color='royalblue'
)

no_outliers_histplot.set_title("Williams' Lap Time Consistency Distribution (Experienced Drivers) - Without Top 2 Outliers")
no_outliers_histplot.set_xlabel('Standard Deviation (ms)')
no_outliers_histplot.set_ylabel('Count')

plt.grid(linewidth=0.25)
plt.show()

print("\n")
print(df_no_outliers) # a dataframe, storing no outliers, is now available here.

"""
Compared to the previous histplot, this one is slightly less left skewed, 
with the exclusion of the two largest outliers specified by the boxplot. 

However, ~10 values between 8,000 and 11,000 ms result in a stronger right tail. 

Remember, most parametric tests, e.g. t-test for mean, assume the data has no major outliers, 
and is roughly normal. 
A big negative outlier can inflate the calculated standard error, making it harder to achieve
statistical significance.

However, as we meet n > 30 for both samples, and roughly normal distributions - let's carry out
both a parametric and a non-parametric test -> Welch's t-test and Mann Whitney U Test
"""

# ------------------ Step 4. Perform hypothesis testing ------------------

# Count the samples involved: 
sample_counts = df_no_outliers['rookie_or_experienced'].value_counts()
print("\nSample sizes: ", sample_counts)

"""
Sample sizes:  rookie_or_experienced
experienced    87
rookie         50
Name: count, dtype: int64

For both samples, n > 30. 
Rookie is roughly normal is shape, but experienced is slighly skewed left. 

Perform a standard Welch's t-test followed by a Mann-Whitney U Test 
Make observations, note differences, compare results at a 95% confidence level.
"""

"""
Hypothesis Recap: 

Rookie drivers had higher lap time variance then their teammates during the 2015-2019 seasons.

Groups: 
- Experienced: n = 87
- Rookie: n = 50
- Two different populations - experienced drivers and rookies.
- Testing if rookies have greater standard deviation, not just different. Direction matters.

Two-sample, independent, one-tailed Welch's t-test. 

H0 (Null): μ_Rookie ≤ μ_Experienced
H1 (Alt): μ_Rookie > μ_Experienced
"""

# 1. extract relevant data for the test
experienced_data = df_no_outliers[df_no_outliers['rookie_or_experienced'] == 'experienced']['laptime_std_ms']
rookie_data = df_no_outliers[df_no_outliers['rookie_or_experienced'] == 'rookie']['laptime_std_ms']

# 2. run the Welch's one-tailed t-test
t_stat, p_value = ttest_ind(
    rookie_data, 
    experienced_data, 
    equal_var = False,
    alternative = 'greater' # defines alternative hypothesis
)

print("\nWelch's t-test for rookies vs. experienced drivers consistency 95% confidence level.\n")
print(f"t-statistic: {t_stat:.3f}")
print(f"One-tailed p-value: {p_value:.4f}")

alpha = 0.05  # 95% confidence level
if p_value < alpha:
    print("\nReject the null hypothesis (H1): Rookies have significantly higher lap time variance than experienced drivers during 2015-2019")
else: # p_value_one_tailed >= alpha
    print("\nFail to reject the null (H0): No significant evidence that rookies have greater lap time variance compared to experienced drivers between 2015-2019.")

"""
Welch's t-test for rookies vs. experienced drivers consistency 95% confidence interval.

t-statistic: -0.214
One-tailed p-value: 0.5844

Fail to reject the null (H0): No significant evidence that rookies have graeter lap time variance compared to experienced drivers between 2015-2019.

A H0 result here, but let's check the non-parametric test first before we jump to conclusions.
"""

# 3. Run the Mann-Whitney U Test
m_stat, p_value_2 = mannwhitneyu(rookie_data, experienced_data, alternative="greater")

print("\nMann-Whitney U Test for rookies vs. experienced drivers consistency at 95% confidence level.")
print(f"Mann-Whitney U statistic: {m_stat:.3f}")
print(f"One-tailed p-value: {p_value_2:.4f}")

if p_value_2 < alpha:
    print("\nReject the null hypothesis (H1): Rookies have significantly higher lap time variance than experienced drivers during 2015-2019")
else: # p_value_one_tailed >= alpha
    print("\nFail to reject the null (H0): No significant evidence that rookies have greater lap time variance compared to experienced drivers between 2015-2019.")

"""
Mann-Whitney U Test for rookies vs. experienced drivers consistency at 95% confidence level.
Mann-Whitney U statistic: 2256.000
One-tailed p-value: 0.3595

Fail to reject the null (H0): No significant evidence that rookies have greater lap time variance compared to experienced drivers between 2015-2019.
"""

"""
Both p-values are well above the 95% confidence level, 0.05 threshold
This means there's insufficient evidence to reject the null hypothesis that rookies do not have greater lap time variance.
The data does not support the claim that rookies have significantly less consistency than their experienced teammates
over those seasons. 

Interventions aimed solely at rookies for consistency improvement might require re-evaluation, or further factors 
should be carefully investigated. 
"""