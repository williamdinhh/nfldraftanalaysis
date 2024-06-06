"""

"""

# import statements
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning and model accuracy testing
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def load_data(combine_data, player_stats):
    """Load and merge NFL datasets.

    Args:
        combine_data (str): Path to the combine data CSV file.
        player_stats (str): Path to the player stats CSV file.

    Returns:
        pd.DataFrame: Merged DataFrame of all datasets

    Note: Games_y is total games played, Games_x is games per season 
    """
    # Load the datasets
    combine_stats = pd.read_csv(combine_data)
    player_data = pd.read_csv(player_stats)

    # Merge datasets for easier data analysis
    # use inner merge so same values across all 3 datasets are consistent.
    final_combined_data = pd.merge(combine_stats,
                                   player_data,
                                   left_on='Player',
                                   right_on='Player',
                                   how='inner')

    # Remove rows with NaN values
    final_combined_data.dropna()

    # Group by player name and sum up the total games and touchdowns
    total_stats_by_player = final_combined_data.groupby('Player').agg({
        'Games': 'sum',
        'RecTD': 'sum',
        'RshTD': 'sum'
    }).reset_index()

    # Calculate total touchdowns
    total_stats_by_player['TotalTouchdowns'] = total_stats_by_player['RecTD'] + total_stats_by_player['RshTD']

    # Merge total games and total touchdowns back into final_combined_data
    final_combined_data = pd.merge(final_combined_data, total_stats_by_player[['Player', 'Games', 'TotalTouchdowns']],
                                   on='Player', how='left')

    # Rename columns
    final_combined_data.rename(columns={'Games': 'TotalGames',
                                        'TotalTouchdowns': 'TotalTDs'}, inplace=True)

    return final_combined_data


def perform_eda(data):
    """exploratory data analysis (EDA).

    Args:
        data (pd.DataFrame): Merged DataFrame of NFL datasets.
    """
    # Summary statistics
    print(data.describe())

    # Distribution of draft picks
    plt.figure(figsize=(10, 6))
    sns.histplot(data['Pick'], bins=30, kde=True)
    plt.title('Distribution of Draft Picks')
    plt.xlabel('Draft Pick')
    plt.ylabel('Frequency')
    plt.savefig("pngs//Distribution of Draft Picks.png")

    correlation_draft_career = data['Pick'].corr(data['Games_y'])
    print(f'Correlation between draft position and career length: {correlation_draft_career:.2f}')
    plt.figure(figsize=(10, 6))
    plt.scatter(data['Pick'], data['Games_y'], alpha=0.5)
    plt.title('Draft Position vs. Career Length')
    plt.xlabel('Draft Position')
    plt.ylabel('Career Length (Games) Played')
    plt.savefig("pngs//Draft Position vs. Career Longevity and Performance.png")


def analyze_combine_stats(data):
    """Analyze NFL combine stats vs. career success.

    Args:
        data (pd.DataFrame): Merged DataFrame of NFL datasets.
    """
    combine_columns = ['Forty', 'Vertical', 'BenchReps',
                       'BroadJump', 'Cone', 'Shuttle']
    performance_columns = ['Games_y', 'RshTD', 'RecTD', 'TotalTDs']

    correlations = data[combine_columns + performance_columns].corr()

    plt.figure(figsize=(16, 12))
    sns.heatmap(correlations, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Combine Stats and Career Performance Correlation Matrix')
    plt.savefig("pngs//Combine Stats and Career Performance Correlation Matrix.png")
    # plt.show()


def machine_learning_model_career_success(data):
    """Build a predictive model to predict career success based
    on combine stats and draft position.

    Args:
        data (pd.DataFrame): Merged DataFrame of NFL datasets.
    """
    features = ['Forty', 'Vertical', 'BenchReps',
                'BroadJump', 'Cone', 'Shuttle', 'Pick']
    target = 'Games_y'

    # Drop rows with NaN values in the target variable since it was getting
    # angry bc not all of the data is actually present
    data = data.dropna(subset=[target])

    X_train, X_test, y_train, y_test = train_test_split(data[features],
                                                        data[target],
                                                        test_size=0.2)

    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse:.2f}')

    feature_importance = pd.DataFrame({'feature': features,
                                       'importance': model.feature_importances_})

    feature_importance = feature_importance.sort_values(by='importance',
                                                        ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature',
                data=feature_importance)
    plt.title('Drill vs Pick Importance')
    plt.savefig("pngs//DrillPickImportance.png")
    # plt.show()


def main():
    """
    Load files to be merged into load_in_data,
    then allows us to run every function
    """
    nfl_data = load_data(
        'combine_data_since_2000_PROCESSED_2018-04-26.csv',
        'NFL Player Stats(92 - 22).csv'
    )

    # check if everything looks good here
    nfl_data.to_csv('merged_nfl_data.csv', index=False)

    perform_eda(nfl_data)
    analyze_combine_stats(nfl_data)
    machine_learning_model_career_success(nfl_data)


if __name__ == '__main__':
    main()
