"""
William Dinh and Ian Kraemer
Loading Data module for final project
"""
import pandas as pd


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
    mask = (final_combined_data['Pos_x'] != 'QB') & (
        final_combined_data['Pos_y'] != 'QB')
    final_combined_data = final_combined_data.loc[mask]

    # group by player name and sum up the total games and touchdowns
    total_stats_by_player = final_combined_data.groupby('Player').agg({
        'Games': 'sum',
        'RecTD': 'sum',
        'RshTD': 'sum'
    }).reset_index()

    # calculate total touchdowns
    total_stats_by_player[
        'TotalTouchdowns'] = total_stats_by_player[
            'RecTD'] + total_stats_by_player['RshTD']

    # merge total games and total touchdowns back into final_combined_data
    final_combined_data = pd.merge(
        final_combined_data,
        total_stats_by_player[['Player', 'Games', 'TotalTouchdowns']],
        on='Player',
        how='left')

    return final_combined_data
