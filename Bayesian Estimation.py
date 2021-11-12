import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
%matplotlib inline

from scipy.stats import beta


df_stats = pd.read_csv('Shots_data.csv')
def set_plot_params(size):
    plt.rcParams["figure.figsize"] = [16,9]
    SIZE = size
    plt.rc('font', size=14)  
    plt.rc('axes', titlesize=14)  
    plt.rc('axes', labelsize=14)  
    plt.rc('xtick', labelsize=14)  
    plt.rc('ytick', labelsize=14)
    plt.rc('legend', fontsize=12)  
    plt.rc('figure', titlesize=SIZE)
def remove_duplicate_players(data_frame):
    """Removes duplicate rows of players which played for multiple teams during the season"""
    # Input should be just one season data
    
    player_occurrences = {}
    for i in range(len(data_frame)):
        player_name = data_frame.iloc[i]['Player']
        player_team = data_frame.iloc[i]['Tm']
        index_row = data_frame.index[i]
        if player_name not in player_occurrences:
            player_occurrences[player_name] = []
        player_occurrences[player_name].append((player_team, index_row))

    for key in player_occurrences:
        curr_list = player_occurrences[key]
        if len(curr_list) == 1:
            continue
        for team, index in curr_list:
            if team != "TOT":
                data_frame = data_frame.drop(index)
    return data_frame

def organize_data(season):
    df = pd.read_csv('Data/Seasons_Stats.csv') 
    df = df[df['Year'] == season]
    df = remove_duplicate_players(df)
    df = df[['Player', '3P', '3PA', '3P%']]
    df = df[df['3PA'] > 20]
    a = beta.fit(list(df['3P%']),floc=0, fscale=1)[0]
    b =  beta.fit(list(df['3P%']),floc=0, fscale=1)[1]
    df['3PEstimate'] = (df['3P'] + a) / (df['3PA'] + a + b)
    df['a'] = df['3P'] + a
    df['b'] = df['3PA'] - df['3P'] + b
    print('alpha: {:.2}'.format(a))
    print('beta: {:.2}'.format(b))
    return (df, a, b)

def make_plots(df, a, b):
    plt.rcParams["figure.figsize"] = [10,6]
    
    plt.figure()
    plt.hist(df['3P%'], bins=30)
    plt.xlabel('3PT %')
    plt.ylabel('Number of Players')
    plt.title('Distribution of 3PT%')
    plt.savefig('plots/3PT%.png')
    
    plt.figure()
    x = np.linspace(0.01, 0.99, 100)
    y = beta.pdf(x, a, b)
    plt.hist(df['3P%'], bins=30, normed=True, label='Emperical')
    plt.plot(x, y, 'k-', lw=2, label='Beta')
    plt.xlabel('3PT %')
    plt.ylabel('Number of Players')
    plt.title('Beta Approximation of 3PT%')
    plt.legend()
    plt.xlim(0.1, 0.7)
    plt.savefig('plots/betaapprox.png')

    plt.figure()
    plt.hist(df['3P%'], bins=30, normed=True, label='Emperical', alpha=0.6)
    plt.hist(df['3PEstimate'], bins =12, alpha=0.6, label='Estimate', normed=True)
    plt.xlabel('3PT %')
    plt.ylabel('Number of Players')
    plt.title('Bayesian Estimation of 3PT%')
    plt.legend()
    plt.savefig('plots/estimation.png')
    
    plt.figure()
    y1 = beta.pdf(x, a, b)
    plt.plot(x, y1, 'r--', lw=1, alpha=0.6, label='League\nDistribution')

#set_plot_params(36)
df, a, b = organize_data(2016)
make_plots(df, a, b)

