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
    combine_stats = pd.read_csv(combine_data)
    player_data = pd.read_csv(player_stats)

    # use inner merge so same values across all 3 datasets are consistent.
    final_combined_data = pd.merge(combine_stats,
                                   player_data,
                                   left_on='Player',
                                   right_on='Player',
                                   how='inner')

    # remove qbs since we're not going to be looking at passing metrics
    mask = (final_combined_data['Pos_x'] != 'QB') & (final_combined_data['Pos_y'] != 'QB')
    final_combined_data = final_combined_data.loc[mask]

    # remove rows with NaN values
    final_combined_data.dropna()

    # group by player name and sum up the total games and touchdowns
    total_stats_by_player = final_combined_data.groupby('Player').agg({
        'Games': 'sum',
        'RecTD': 'sum',
        'RshTD': 'sum'
    }).reset_index()

    # calculate total touchdowns
    total_stats_by_player['TotalTouchdowns'] = total_stats_by_player['RecTD'] + total_stats_by_player['RshTD']

    # merge total games and total touchdowns back into final_combined_data
    final_combined_data = pd.merge(final_combined_data, total_stats_by_player[['Player', 'Games', 'TotalTouchdowns']],
                                   on='Player', how='left')

    return final_combined_data


def scatter_plot(data):
    """scatter plot of draft position vs career length by total games played

    Args:
        data (pd.DataFrame): Merged DataFrame of NFL datasets.
    """
    correlation_draft_career = data['Pick'].corr(data['Games_y'])
    print(f'Correlation between draft position and career length: {correlation_draft_career:.2f}')
    plt.figure(figsize=(10, 6))
    plt.scatter(data['Pick'], data['Games_y'], alpha=0.5)
    plt.title('Draft Position vs. Career Length')
    plt.xlabel('Draft Position')
    plt.ylabel('Career Length (Games) Played')
    plt.savefig("pngs//Draft Position vs. Career Longevity and Performance.png")
    # plt.show()


def scatter_plot_av(data):
    """
    Created by PFR founder Doug Drinen, the Approximate Value (AV)
    method is an attempt to put a single number on the seasonal value
    of a player at any position from any year (since 1960

    Args:
        data (pd.DataFrame): Merged DataFrame of NFL datasets.
    """
    correlation_draft_career = data['Pick'].corr(data['AV'])
    print(f'Correlation between draft position and AV: {correlation_draft_career:.2f}')
    plt.figure(figsize=(10, 6))
    plt.scatter(data['Pick'], data['AV'], alpha=0.5)
    plt.title('Draft Position vs. AV')
    plt.xlabel('Draft Position')
    plt.ylabel('AV')
    plt.savefig("pngs//Draft Position vs. AV.png")
    # plt.show()


def draft_performance_by_team_av(data):
    """analysis of draft performance among teams by AV

    Created by PFR founder Doug Drinen, the Approximate Value (AV)
    method is an attempt to put a single number on the seasonal value
    of a player at any position from any year (since 1960).

    Args:
        data (pd.DataFrame): Merged DataFrame of NFL datasets.
    """
    team_av = data.groupby('Team')['AV'].mean().sort_values(ascending=False)
    plt.figure(figsize=(18, 8))
    sns.barplot(x=team_av.values, y=team_av.index, legend=False)
    plt.title('Average Approximate Value (AV) by Team')
    plt.xlabel('Average AV')
    plt.ylabel('Team')
    plt.savefig("pngs//Average Approximate Value (AV) by Team")
    # plt.show()


def analyze_combine_stats(data):
    """Analyze NFL combine stats vs. career success.

    Args:
        data (pd.DataFrame): Merged DataFrame of NFL datasets.
    """
    combine_columns = ['Forty', 'Vertical', 'BenchReps',
                       'BroadJump', 'Cone', 'Shuttle']
    performance_columns = ['Games_y', 'RshTD', 'RecTD', 'TotalTouchdowns']

    correlations = data[combine_columns + performance_columns].corr()

    plt.figure(figsize=(16, 12))
    sns.heatmap(correlations, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Combine Stats and Career Performance Correlation Matrix')
    plt.savefig("pngs//Combine Stats and Career Performance Correlation Matrix.png")
    # plt.show()


def analyze_combine_stats_av(data):
    """_summary_

    Args:
        data (_type_): _description_
    """
    combine_columns = ['Forty', 'Vertical', 'BenchReps',
                       'BroadJump', 'Cone', 'Shuttle']
    performance_columns = ['AV']

    correlations = data[combine_columns + performance_columns].corr()

    plt.figure(figsize=(16, 12))
    sns.heatmap(correlations, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Combine Stats and AV Correlation Matrix')
    plt.savefig("pngs//Combine Stats and AV Correlation Matrix.png")
    # plt.show()


def machine_learning_model_career_success_total_games(data):
    """Build a predictive model to predict career success based
    on combine stats and draft position.

    Args:
        data (pd.DataFrame): Merged DataFrame of NFL datasets.
    """
    features = ['Forty', 'Vertical', 'BenchReps',
                'BroadJump', 'Cone', 'Shuttle']
    target = 'Pick'

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
    plt.title('Combine Drill vs Pick Importance')
    plt.savefig("pngs//DrillPickImportance.png")
    # plt.show()


def machine_learning_model_career_success_AV(data):
    """Build a predictive model to predict career success based
    on combine stats and draft position.

    Args:
        data (pd.DataFrame): Merged DataFrame of NFL datasets.
    """
    features = ['Forty', 'Vertical', 'BenchReps',
                'BroadJump', 'Cone', 'Shuttle']
    target = 'AV'

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
    plt.title('Combine Drill vs AV Importance')
    plt.savefig("pngs//AVPickImportance.png")
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

    scatter_plot(nfl_data)
    scatter_plot_av(nfl_data)
    draft_performance_by_team_av(nfl_data)
    analyze_combine_stats(nfl_data)
    analyze_combine_stats_av(nfl_data)
    machine_learning_model_career_success_total_games(nfl_data)
    machine_learning_model_career_success_AV(nfl_data)


if __name__ == '__main__':
    main()
