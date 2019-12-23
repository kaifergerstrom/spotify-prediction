import spotipy, pickle, os, warnings
import pandas as pd

import matplotlib.pyplot as plt

from spotipy.oauth2 import SpotifyClientCredentials

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import graphviz
import pydotplus
import io
from scipy import misc
import imageio

warnings.filterwarnings('ignore')

client_credentials_manager = SpotifyClientCredentials(os.environ["SPOTIFY_ID"], os.environ["SPOTIFY_SECRET"])  # Initialize spotify api
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)  # Create spotify object


def get_song_info(track_name, artist_name):  # Gets the info of the searched song to pass into the dataset
    tracks = sp.search(track_name, limit=5, type='track')['tracks']['items']
    for track in tracks:
        for artist in track['artists']:
            if artist['name'] == artist_name and track['name'] == track_name:
                features = sp.audio_features(track['id'])[0]  # Get features of the song
                df = pd.DataFrame()
                new_row = {  # New row data for pandas dataframe
                    'danceability': features['danceability'], 
                    'energy': features['energy'], 
                    'loudness': features['loudness'], 
                    'speechiness': features['speechiness'], 
                    'acousticness': features['acousticness'], 
                    'instrumentalness': features['instrumentalness'], 
                    'valence': features['valence'], 
                    'duration_ms': features['duration_ms'],
                    'key': features['key'],
                }
                df = df.append(new_row, ignore_index=True)  # Append new row to dataframe
                return df
    raise ValueError("Unable to find artist or song!")  


def get_playlist_info(playlist_id, verdict):
    try:
        tracks = sp.user_playlist(user=None, playlist_id=playlist_id)['tracks']
        
        df = pd.DataFrame(columns=['verdict', 'id', 'artist', 'title', 'popularity', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'key', 'mode', 'time_signature'])  # Create empty dataframe with features for dataset

        for item in tracks['items']:  # Loop through all the tracks
            track = item['track']  # Get track info from item
            artist = track['artists'][0]['name']  # Get artist name
            title = track['name']  # Get song name
            features = sp.audio_features(track['id'])[0]  # Get features of the song\
            
            new_row = {  # New row data for pandas dataframe
                'verdict': verdict, 
                'id': track['id'], 
                'artist': artist, 
                'title': title, 
                'popularity': track['popularity'],
                'danceability': features['danceability'], 
                'energy': features['energy'], 
                'loudness': features['loudness'], 
                'speechiness': features['speechiness'], 
                'acousticness': features['acousticness'], 
                'instrumentalness': features['instrumentalness'], 
                'liveness': features['liveness'], 
                'valence': features['valence'], 
                'tempo': features['tempo'],
                'duration_ms': features['duration_ms'],
                'key': features['key'],
                'mode': features['mode'],
                'time_signature': features['time_signature']
            }
            df = df.append(new_row, ignore_index=True)  # Append new row to dataframe
        return df
    except:
        pass


def create_df(positives, negatives, limit, output_file):  # Creates balanced dataset of songs I like/don't like
    
    p_df = []  # Define empty list for all the positive data samples
    for playlist_id in positives:  # Loop through all the positives
        p_df.append(get_playlist_info(playlist_id, 1))  # Extract the song info for each
    p_df = pd.concat(p_df)  # Concat all the positive songs

    n_df = []  # Define empty list for all the negative data samples
    for playlist_id in negatives:  # Loop through all the negatives
        n_df.append(get_playlist_info(playlist_id, 0))  # Extract the song info for each
    n_df = pd.concat(n_df)  # Concat all the negative songs

    dataset = pd.concat([p_df, n_df], axis=0)  # Combine the two data sets
    dataset['verdict'] = dataset['verdict'].astype('int')
    dataset.to_pickle(output_file)  # Serialize the data frame


def display_histogram(df, features):
    figure = plt.figure(figsize=(15, 15))

    for index, feature in enumerate(features, start=1):
        
        hist_id = int("33{}".format(index))
        pos = df[df['verdict'] == 1][feature]
        neg = df[df['verdict'] == 0][feature]
        axis = figure.add_subplot(hist_id)
        axis.set_xlabel(feature)
        axis.set_ylabel("Count")
        axis.set_title("Song {} vs Like Distribution".format(feature.capitalize()))
        pos.hist(alpha=0.5, bins=30)
        axis2 = figure.add_subplot(hist_id)
        neg.hist(alpha=0.5, bins=30)

    plt.show()


def display_tree(tree, features, path):  # Display tree design (found online)
    f = io.StringIO()
    export_graphviz(tree, out_file=f, feature_names=features)
    pydotplus.graph_from_dot_data(f.getvalue()).write_png(path)
    img = imageio.imread(path)
    plt.rcParams["figure.figsize"] = (20,20)
    plt.imshow(img)


df = pickle.load(open("data1", 'rb'))

train, test = train_test_split(df, test_size=0.2)

#display_histogram(df, ["danceability", "duration_ms", "loudness", "speechiness", "valence", "energy", "key", "acousticness", "instrumentalness"])

c = DecisionTreeClassifier(min_samples_split=100)  # Define amount of branches of splits
features = ["danceability", "duration_ms", "loudness", "speechiness", "valence", "energy", "key", "acousticness", "instrumentalness"]  # Features to train for tree

# Create training and testing data
X_train = train[features]
y_train = train["verdict"]
X_test = test[features]
y_test = test["verdict"]

dt = c.fit(X_train, y_train)

display_tree(dt, features, "dt.png")


song_test = get_song_info("Lyin' Eyes - 2013 Remaster", "Eagles") 
y_pred = c.predict(song_test)

print(y_pred)