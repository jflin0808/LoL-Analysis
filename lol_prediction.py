import requests
import json
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def getAPI():
    f = open('api_key.txt', 'r')
    return f.read()

    
def checkDuplicates(list):
    if len(list) == len(set(list)):
        return False
    else:
        return True


def requestSummonerData(region, ign, api_key):
    URL = "https://" + region + ".api.riotgames.com/lol/summoner/v4/summoners/by-name/" + ign + '?api_key=' + api_key
    response = requests.get(URL)
    return response.json()


def requestChallengerLeague(region, queue, api_key):
    URL = 'https://' + region + '.api.riotgames.com/lol/league/v4/challengerleagues/by-queue/' + queue + '?api_key=' + api_key
    response = requests.get(URL)
    return response.json()


def requestMatchHistory(region, puuid, start, end, api_key):
    URL = 'https://' + region + '.api.riotgames.com/lol/match/v5/matches/by-puuid/' + puuid + '/ids?start=' + start + '&count=' + end + '&api_key=' + api_key
    response = requests.get(URL)
    return response.json()


def requestMatchData(region, match_id, api_key):
    URL = 'https://' + region + '.api.riotgames.com/lol/match/v5/matches/' + match_id + '?api_key=' + api_key 
    response = requests.get(URL)
    return response.json()


def cleanData(matches, continent, api):
    data = {'team': [], 'baron': [], 'champion': [], 'dragon': [], 'inhibitor': [], 'riftHerald': [], 'tower': [], 'win': []}
    objectives = ['baron', 'champion', 'dragon', 'inhibitor', 'riftHerald', 'tower']

    for match in matches:
        match_data = requestMatchData(continent, match, api)
        try:
            for team in range(2):
                data['team'].append(match_data['info']['teams'][team]['teamId'])
                data['win'].append(match_data['info']['teams'][team]['win'])
                for objective in objectives:
                    data[objective].append(match_data['info']['teams'][team]['objectives'][objective]['kills'])
        except KeyError:
            pass
    return data


def labelEncode(df):
    df.loc[df['win'] == True, 'win'] = 1
    df.loc[df['win'] == False, 'win'] = 0
    df.loc[df['team'] == 100, 'team'] = 0
    df.loc[df['team'] == 200, 'team'] = 1


def saveData():
    region, continent, api = 'NA1', 'americas', getAPI()
    matches = []
    
    challenger_json = requestChallengerLeague(region, 'RANKED_SOLO_5x5', api)
    challenger_list = challenger_json['entries']
    sorted_challengers = sorted(challenger_list, key=lambda x: x['leaguePoints'], reverse=True)

    for challenger in sorted_challengers:
        try:
            challenger_data = requestSummonerData(region, challenger['summonerName'], api)
            match_history = requestMatchHistory(continent, challenger_data['puuid'], '0', '5', api)
            for match in match_history:
                if match not in matches:
                    matches.append(match)
        except KeyError:
            pass
    
    print(matches)
    data = cleanData(matches, continent, api)
    df = pd.DataFrame.from_dict(data)
    labelEncode(df)
    df.to_csv('challenger_match_data.csv')


def EDA(df):
    sns.set(style='darkgrid')

    #Relationship between features and label plot
    fig, ax = plt.subplots(3, 2, figsize=(10,15))
    ax[0,0].set_title("Baron kills Relation to Wins")
    ax[0,1].set_title("Champion kills Relation to Wins")
    ax[1,0].set_title("Dragon kills Relation to Wins")
    ax[1,1].set_title("Inhibitor kills Relation to Wins")
    ax[2,0].set_title("Rift Herald kills Relation to Wins")
    ax[2,1].set_title("Tower kills Relation to Wins")
    sns.boxplot(ax = ax[0,0], data=df, x='win', y='baron')
    sns.boxplot(ax = ax[0,1], data=df, x='win', y='champion')
    sns.boxplot(ax = ax[1,0], data=df, x='win', y='dragon')
    sns.boxplot(ax = ax[1,1], data=df, x='win', y='inhibitor')
    sns.boxplot(ax = ax[2,0], data=df, x='win', y='riftHerald')
    sns.boxplot(ax = ax[2,1], data=df, x='win', y='tower')
    fig.subplots_adjust(hspace=0.5)
    print(plt.show())

    # Correlation Heatmap between all features
    corr = df.drop(['team', 'win'], axis=1)
    corr = corr.corr()
    plt.figure(figsize=(15,10))
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
    print(plt.show())

    # Relationship between champion kills and tower kills plot
    champion_to_towers = df.groupby(['champion']).agg({'tower':'mean'}).reset_index()
    fig = px.scatter(champion_to_towers, x='champion', y='tower', size='tower', color='champion',
               size_max=60, title='Relationship between champion kills and tower kills')
    print(fig.show())


def separateData(df):
    X = df.drop('win', axis=1)
    y = df['win'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=100)
    return X_train, X_test, y_train, y_test


def scaleData(X_train, X_test):
    X_scaler = StandardScaler()
    scaled_X_train = X_scaler.fit_transform(X_train)
    scaled_X_test = X_scaler.transform(X_test)
    return scaled_X_train, scaled_X_test


def dashDisplay(df):    
    features = ['baron', 'champion', 'dragon', 'inhibitor', 'riftHerald', 'tower']
    champion_to_towers = df.groupby(['champion']).agg({'tower':'mean'}).reset_index()

    app = dash.Dash(__name__)
    app.layout = html.Div(children=[
        html.H1(
            children='LoL Challenger Match Data Analysis',
            style={
                'textAlign': 'center',
            }
        ),

        html.Div(
            children='Match History Analysis from the top 300 LoL players in NA',
            style={
                'textAlign': 'center',
            }
        ),

        html.Div(
            dcc.Dropdown(id='feature-choice', 
                         options=[{'label':x, 'value':x}
                                   for x in sorted(df[features].columns)],
                         value='Features'
            )
        ),

        html.Div(
            dcc.Graph(id='feature-label-relationship', figure={}
            )
        ),

        html.Div(
            dcc.Graph(id='champion-tower-relationship', figure=px.scatter(champion_to_towers, x='champion', y='tower', size='tower', color='champion', 
                      size_max=50, title='Relationship between champion kills and tower kills')
            )
        ),
    ])

    @app.callback(
        Output(component_id='feature-label-relationship', component_property='figure'),
        Input(component_id='feature-choice', component_property='value')
    )
    def interactive_graph(value_feature):
        fig = px.box(df, x='win', y=value_feature, title='{feature} kills relationship to win'.format(feature=value_feature))
        return fig

    app.run_server(debug=True)


def main():
    df = pd.read_csv('challenger_match_data.csv')
    # EDA(df)
    # X_train, X_test, y_train, y_test = separateData(df)
    # scaled_X_train, scaled_X_test= scaleData(X_train, X_test)
    # lr = LogisticRegression().fit(scaled_X_train, y_train)
    # y_predict = lr.predict(scaled_X_test)
    # print("Accuracy Score: " + str(accuracy_score(y_test, y_predict)*100) + "%")
    dashDisplay(df)
    

if __name__ == '__main__':
    main()
    