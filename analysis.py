"""
Ian Kraemer and William Dinh
CSE 163 Final Project
"""

# import statements
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from data_loader import load_data

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def scatter_plot(data):
    """scatter plot of draft position vs career length by total games played

    Args:
        data (pd.DataFrame): Merged DataFrame of NFL datasets.
    """
    draft_car = data['Pick'].corr(data['Games_y'])
    print(
        f'Correlation between draft position & career length: {draft_car:.2f}')
    plt.figure(figsize=(10, 6))
    plt.scatter(data['Pick'], data['Games_y'], alpha=0.5)
    plt.title('Draft Position vs. Career Length')
    plt.xlabel('Draft Position')
    plt.ylabel('Career Length (Games) Played')
    plt.savefig(
        "pngs//Draft Position vs. Career Longevity and Performance.png")
    # plt.show()


def scatter_plot_av(data):
    """
    Created by PFR founder Doug Drinen, the Approximate Value (AV)
    method is an attempt to put a single number on the seasonal value
    of a player at any position from any year (since 1960

    Args:
        data (pd.DataFrame): Merged DataFrame of NFL datasets.
    """
    draft_av = data['Pick'].corr(data['AV'])
    print(
        f'Correlation between draft position and AV: {draft_av:.2f}')
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
    combine_columns = ['Forty', 'Vertical', 'Ht', 'Wt',
                       'BroadJump', 'Cone', 'Shuttle']
    performance_columns = ['Games_y', 'RshTD', 'RecTD',
                           'TotalTouchdowns', 'AV']

    correlations = data[combine_columns + performance_columns].corr()

    plt.figure(figsize=(16, 12))
    sns.heatmap(correlations, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Combine Stats and Career Performance Correlation Matrix')
    plt.savefig(
        "pngs//Combine Stats and Career Performance Correlation Matrix.png")
    # plt.show()


def machine_learning_model_career_success_total_games(data):
    """Build a predictive model to predict career success based
    on combine stats and draft position using Random Forest Regressor.

    Args:
        data (pd.DataFrame): Merged DataFrame of NFL datasets.
    """
    features = ['Forty', 'Vertical', 'BenchReps', 'Ht', 'Wt',
                'BroadJump', 'Cone', 'Shuttle']
    target = 'Games_y'

    # drop rows with NaN values in the target variable
    data = data.dropna(subset=[target])

    X_train, X_test, y_train, y_test = train_test_split(data[features],
                                                        data[target],
                                                        test_size=0.2
                                                        )

    model = RandomForestRegressor(n_estimators=200)

    # train the model + predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # calculate Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    print(
        f'Mean Squared Error of target, Total Games Played: {mse:.2f}')

    # calculate variance of the target variable 'total games played'
    variance = np.var(data[target])
    print(
        f'Variance of target, Total Games Played: {variance:.2f}')

    # calculate r squared value
    r2 = r2_score(y_test, y_pred)
    print(f'R^2 Score: {r2:.2f}')

    feature_importance = pd.DataFrame(
        {'feature': features,
         'importance': model.feature_importances_})

    feature_importance = feature_importance.sort_values(by='importance',
                                                        ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature',
                data=feature_importance)
    plt.title('Combine Drill vs Total Career Games Played Importance')
    plt.xlabel("Total Career Games Played Importance")
    plt.ylabel("Drill")
    plt.savefig("pngs//DrillPickImportance.png")
    # plt.show()


def machine_learning_model_career_success_AV(data):
    """Build a predictive model to predict career success based
    on combine stats and draft position.

    Args:
        data (pd.DataFrame): Merged DataFrame of NFL datasets.

    Returns:
        mse (float): Mean squared error of the model predictions.
        variance_AV (float): Variance of the target variable 'AV'.
    """
    features = ['Forty', 'Vertical', 'BenchReps', 'Ht', 'Wt',
                'BroadJump', 'Cone', 'Shuttle']
    target = 'AV'

    # drop rows with NaN values in the target variable
    data = data.dropna(subset=[target])

    X_train, X_test, y_train, y_test = train_test_split(data[features],
                                                        data[target],
                                                        test_size=0.2)

    model = RandomForestRegressor(n_estimators=200)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error of target, AV: {mse:.2f}')

    # calculate variance of the target variable 'AV'
    variance_AV = np.var(data[target])
    print(f'Variance of Target AV: {variance_AV:.2f}')

    # calculate r squared value
    r2 = r2_score(y_test, y_pred)
    print(f'R^2 Score: {r2:.2f}')

    feature_importance = pd.DataFrame(
        {'feature': features,
         'importance': model.feature_importances_})

    feature_importance = feature_importance.sort_values(by='importance',
                                                        ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature',
                data=feature_importance)
    plt.title('Combine Drill vs AV Importance')
    plt.xlabel("AV Importance")
    plt.ylabel("Drill")
    plt.savefig("pngs//AVPickImportance.png")


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
    machine_learning_model_career_success_total_games(nfl_data)
    machine_learning_model_career_success_AV(nfl_data)


if __name__ == '__main__':
    main()
