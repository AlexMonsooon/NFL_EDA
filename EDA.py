import pandas as pd 
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import chi2_contingency

engine = create_engine('mysql+pymysql://root:password@127.0.0.1:3306/sys')    
print("Connection successful.")

################################################################################
def fetch_existing_data(
    engine, 
    table_name, 
    cols='*', 
    joins=None, 
    groupby=None, 
    orderby=None
):
    """
    Fetch data from a database with optional JOINs, GROUP BY, and ORDER BY clauses.

    Parameters:
    - engine: SQLAlchemy engine for database connection.
    - table_name: Main table name to fetch data from.
    - cols: Columns to select (default is '*').
    - joins: List of tuples for JOINs [(join_table, join_condition)].
    - groupby: Columns for GROUP BY clause.
    - orderby: Columns for ORDER BY clause.

    Returns:
    - DataFrame: Query result as a pandas DataFrame.
    """
    
    query = f"SELECT {cols} FROM {table_name}"

    if joins:
        for join_table, join_condition in joins:
            query += f" JOIN {join_table} ON {join_condition}"

    if groupby:
        groupby_clause = ", ".join(groupby)
        query += f" GROUP BY {groupby_clause}"

    if orderby:
        orderby_clause = ", ".join(orderby)
        query += f" ORDER BY {orderby_clause}"
        
    return pd.read_sql(query, engine)

################################################################################
# Convert 'Record' column to a percentage, -1 for first week
def record_to_percentage(record):
    try:
        wins, games = map(int, record.split('/'))
        return (wins / games) * 100 if games > 0 else 0
    except:
        return -1  # Handle invalid or missing data gracefully

###############################################################################
###############################################################################
###############################################################################
games_df = fetch_existing_data(engine, 'games', orderby=['Game_Date'])

## can drop games_id and FullTeam cols, have Tm to use as our main
games_df.drop(columns=['games_id', 'Tm'], inplace=True)

day_mapping = {'Sun': 0, 'Mon': 1, 'Tue': 2, 'Wed': 3, 'Thu': 4, 'Fri': 5, 'Sat': 6}
games_df['Game_Day'] = games_df.loc[:, 'Game_Day'].map(day_mapping)
    
# Change x/x to % won and shift records by 1
games_df['Record'] = games_df['Record'].apply(record_to_percentage)

################################################################################
cats_cols = ['FullTeam', 'Opp', 'Coach', 'Stadium', 'Surface', 'Roof']

discrete_cols = ['Season', 'Game_Week', 'Game_Day', 'Beat_Spread', 'Result', 'Turnovers', 'Rest', 'Penalties', 'HA']

continous_cols = ['Duration', 'Attendance', 'Spread', 'Over_Under', 'PF', 'PA',
       'First_Downs', 'Total_Yards', 'Third_Down_Conv_', 'Fourth_Down_Conv_',
       'Time_of_Possession', 'Penalty_Yards', 'Record']

##############################################################################
# VISUALATIONS FOR EDA 
for col in games_df.columns:
    print(games_df[col].value_counts())
    print()


print(games_df.isna().sum())


# Univariate analysis for discrete, continous, and categorical columns
for col in games_df.columns:
    if col in cats_cols:
        plt.figure(figsize=(10, 6))
        games_df[col].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
        plt.title(f'Bar Chart for {col} (Categorical Data)')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.show()
        
    elif col in discrete_cols:
        # Discrete data visualization (Bar Chart)
        plt.figure(figsize=(8, 4))
        games_df[col].value_counts().sort_index().plot(kind='bar', color='skyblue')
        plt.title(f'Bar Chart for {col} (Discrete Data)')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()
        
    
    # elif col in continous_cols:
    #     if games_df[col].dtype in ['float64', 'int64']:
        
            # Histogram Plot
            # how data is distributed in absolute terms.
            # plt.figure(figsize=(8, 4))
            # plt.hist(games_df[col], bins=30, alpha=0.7, edgecolor='black')
            # plt.title(f'Histogram for {col}')
            # plt.xlabel(col)
            # plt.ylabel('Frequency')
            # plt.show()

            # box plots
            # summarize the data's central tendency, variability, and potential outliers.
            # plt.figure(figsize=(8, 4))
            # sns.boxplot(x=games_df[col])
            # plt.title(f'Box Plot for {col}')
            # plt.xlabel(col)
            # plt.show()
            
            # Violin Plot 
            # balance of density estimation and statistical summary.
            # plt.figure(figsize=(8, 4))
            # sns.violinplot(x=games_df[col], inner='quartile')
            # plt.title(f'Violin Plot for {col}')
            # plt.xlabel(col)
            # plt.show()
    
            # Probability Density Function Plot
            # 
            # plt.figure(figsize=(8, 4))
            # sns.kdeplot(games_df[col], fill=True, alpha=0.6, linewidth=2)
            # plt.title(f'Probability Density Function for {col}')
            # plt.xlabel(col)
            # plt.ylabel('Density')
            # plt.show()
            
            
            # Cumulative Distribution Function Plot
            # interested in percentiles or comparing distributions.
            # plt.figure(figsize=(8, 4))
            # sns.ecdfplot(games_df[col])
            # plt.title(f'Cumulative Distribution Function for {col}')
            # plt.xlabel(col)
            # plt.ylabel('Cumulative Probability')
            # plt.show()
            

##############################################################################
# Bivariate analysis for discrete, continous, and categorical columns

# pearson measures strength of linear relationships, assumes normality, sensitive to outliers
# spearman measures strength of monotonic relationships, no distribution assumptions, robust to outliers
# kendall measures monotonic relationships by probability of agreement between ranks, no distribution assumptions, robust to outliers

def correlation(df):
  correlation_matrix = df[continous_cols + discrete_cols].corr(method='spearman') # method = pearson, kendall, spearman

  correlation_threshold = 0.7
  correlated_columns = {}

  for i, column in enumerate(correlation_matrix.columns):
      correlated_columns[column] = []
      for j, correlation_value in enumerate(correlation_matrix.iloc[:, i]):
          if abs(correlation_value) >= correlation_threshold and i != j:
              correlated_columns[column].append((correlation_matrix.columns[j], correlation_value))

  print(correlated_columns)
  
  plt.figure(figsize=(16, 12))
  sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
  plt.title("Correlation Heatmap")
  plt.show()

correlation(games_df)

###############################################################################
###############################################################################
def cramers_v(df):
    """
    Calculate Cramér's V for pairs of categorical columns.

    Parameters:
        df (DataFrame): The DataFrame containing the categorical variables.
        cats_cols (list): List of categorical column names.

    Returns:
        cramers_v_matrix (DataFrame): A DataFrame of Cramér's V values.
    """
    # Initialize a matrix to store Cramér's V values
    cramers_v_matrix = pd.DataFrame(index=cats_cols, columns=cats_cols)

    for col1 in cats_cols:
        for col2 in cats_cols:
            if col1 == col2:
                cramers_v_matrix.loc[col1, col2] = 1.0  # Same column comparison
            else:
                # Create contingency table
                contingency_table = pd.crosstab(df[col1], df[col2])
                chi2, _, _, _ = chi2_contingency(contingency_table)
                n = contingency_table.sum().sum()
                r, k = contingency_table.shape
                cramers_v_value = np.sqrt(chi2 / (n * (min(r - 1, k - 1))))
                cramers_v_matrix.loc[col1, col2] = cramers_v_value

    cramers_v_matrix = cramers_v_matrix.astype(float)

    print("\nCramér's V Matrix (Categorical Variables):")
    print(cramers_v_matrix)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cramers_v_matrix, annot=True, fmt=".3f", cmap="coolwarm", cbar=True, square=True)
    plt.title("Cramér's V Heatmap (Categorical Variables)")
    plt.xlabel("Categorical Columns")
    plt.ylabel("Categorical Columns")
    plt.show()

    return cramers_v_matrix

cramers_v(games_df)
###############################################################################
################################################################################
# scatter plots of all the numerical variable combinations
sns.pairplot(games_df[continous_cols + discrete_cols], hue='Result')
plt.show()

## Easier to see when 5 graphs are plotted at a time
filt_df = games_df[continous_cols + discrete_cols]
hue = 'Result'
vars_per_line = 5
all_vars = list(filt_df.columns.symmetric_difference([hue]))

for var in all_vars:
    rest_vars = list(all_vars)
    rest_vars.remove(var)
    while rest_vars:
        line_vars = rest_vars[:vars_per_line]
        del rest_vars[:vars_per_line]
        line_var_names = ", ".join(line_vars)
        sns.pairplot(games_df, x_vars=line_vars, y_vars=[var], hue=hue, palette='bright')
        plt.show()
        plt.close()

################################################################################
# show numerical with categorical
sns.violinplot(data=games_df, x='Coach', y='Spread', palette='pastel', inner='quartile')
plt.title('Violin Plot of Numerical_Col by Categorical_Col')
plt.show()

###############################################################################
## shows categorical with result plot
# need to add FullTeam to Surface, Roof, Stadium

def cat_against_target(df):
    for col in cats_cols:
        sns.countplot(data=games_df, x=col, hue='Result', palette='pastel')
        plt.show()

cat_against_target(games_df)

###############################################################################
# shows numerical with result plot
def num_against_target(df):
    num_cols = continous_cols + discrete_cols
    for col in num_cols:
        plt.figure(figsize=(14, 8))
        sns.violinplot(data=df, x='Result', y=col, hue='Result', palette='pastel')
        plt.title(f'Violin Plot for {col}')
        plt.show()
        
num_against_target(games_df)

###############################################################################

