import numpy as np
import pandas as pd
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import CategoricalColorMapper, HoverTool, Slider, Select
from bokeh.layouts import row, column, widgetbox
from bokeh.io import curdoc

# Read in the Datasets
df_players = pd.read_csv('../Data/Players.csv')
df_stats = pd.read_csv('../Data/Seasons_Stats.csv')

df_stats = df_stats[['Year', 'Player', 'Pos', 'Age', 'Tm', 'G', 'TS%', 'FG', 'FGA', 'FG%', '3P', '3PA', 
        '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'TRB', 'AST', 'STL', 'BLK', 
        'TOV', 'PTS']]
        
df_stats = df_stats.dropna()
df_stats.Year = df_stats.Year.astype(int)
df_stats['RB'] = df_stats['TRB']
del df_stats['TRB']


def fan_pts(pts, reb, ast, blk, st, to):
    "Returns a one statistic summary of a players performance"
    return (1*pts + 1.2*reb + 1.5*ast + 3*blk + 3*st - 1*to)
    
    
pd.options.display.float_format = '{:,.2f}'.format

df_stats['PTS_avg'] = df_stats['PTS'] / df_stats['G']
df_stats['AST_avg'] = df_stats['AST'] / df_stats['G']
df_stats['STL_avg'] = df_stats['STL'] / df_stats['G']
df_stats['RB_avg'] = df_stats['RB'] / df_stats['G']
df_stats['BLK_avg'] = df_stats['BLK'] / df_stats['G']
df_stats['TOV_avg'] = df_stats['TOV'] / df_stats['G']
df_stats['Fan_PTS'] = fan_pts(df_stats['PTS'], df_stats['RB'], df_stats['AST']
                              , df_stats['BLK'], df_stats['STL'], df_stats['TOV'])
df_stats['Fan_PTS_avg'] = df_stats['Fan_PTS'] / df_stats['G']
df_stats["3Pfract"] = df_stats["3PA"]/df_stats.FGA

palette = ['aliceblue','antiquewhite','aqua','aquamarine','azure','beige','bisque','black',
           'blanchedalmond','blue','blueviolet','brown','burlywood','cadetblue','chartreuse',
           'chocolate','coral','cornflowerblue','cornsilk','crimson','cyan','darkblue','darkcyan',
           'darkgoldenrod','darkgray','darkgreen','darkkhaki','darkmagenta','darkolivegreen',
           'darkorange','darkorchid','darkred','darksalmon','darkseagreen','darkslateblue',
           'darkslategray','darkturquoise','darkviolet','red']

color_mapper = CategoricalColorMapper(factors=df_stats['Tm'].unique().tolist(),
                                      palette=palette)

p1 = figure(x_axis_label='3 Points Attempted', y_axis_label='3 Points Made', tools='box_select')
p2 = figure(x_axis_label='2 Points Attempted', y_axis_label='2 Points Made', tools='box_select')

slider = Slider(title='Year', start=1980, end=2017, step=1, value=2006)
menu = Select(options=df_stats['Tm'].unique().tolist(), value='GSW', title='Team')

source = ColumnDataSource(data={'x_3p': df_stats['3PA'], 'y_3p': df_stats['3P'],
                                'Tm': df_stats['Tm'], 'x_2p': df_stats['2PA'],
                                'y_2p': df_stats['2P'], 'Year': df_stats['Year'],
                                'Player': df_stats['Player']})
                                
def callback(attr, old, new):
    new_x_3p = df_stats[(df_stats['Year'] == slider.value) &
                               (df_stats['Tm'] == menu.value)]['3PA']

    new_y_3p = df_stats[(df_stats['Year'] == slider.value) &
                               (df_stats['Tm'] == menu.value)]['3P']

    new_tm = df_stats[(df_stats['Year'] == slider.value) &
                             (df_stats['Tm'] == menu.value)]['Tm']

    new_x_2p = df_stats[(df_stats['Year'] == slider.value) &
                               (df_stats['Tm'] == menu.value)]['2PA']

    new_y_2p = df_stats[(df_stats['Year'] == slider.value) &
                               (df_stats['Tm'] == menu.value)]['2P']

    new_year = df_stats[(df_stats['Year'] == slider.value) &
                               (df_stats['Tm'] == menu.value)]['Year']

    new_player = df_stats[(df_stats['Year'] == slider.value) &
                                 (df_stats['Tm'] == menu.value)]['Player']

    source.data = {'x_3p': new_x_3p, 'y_3p': new_y_3p, 'Tm': new_tm, 'x_2p': new_x_2p,
                   'y_2p': new_y_2p, 'Year': new_year, 'Player': new_player}


slider.on_change('value', callback)
menu.on_change('value', callback)

p1.circle('x_3p', 'y_3p', source=source, alpha=0.8, nonselection_alpha=0.1,
          color=dict(field='Tm', transform=color_mapper), legend='Tm')

p2.circle('x_2p', 'y_2p', source=source, alpha=0.8, nonselection_alpha=0.1,
          color=dict(field='Tm', transform=color_mapper), legend='Tm')

p1.legend.location = 'bottom_right'
p2.legend.location = 'bottom_right'

hover1 = HoverTool(tooltips=[('Player', '@Player')])
p1.add_tools(hover1)
hover2 = HoverTool(tooltips=[('Player', '@Player')])
p2.add_tools(hover2)

column1 = column(widgetbox(menu), widgetbox(slider))
layout = row(column1, p1, p2)

curdoc().add_root(layout)