import spotipy, pickle, os, graphviz, pydotplus, io, imageio, argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from spotipy.oauth2 import SpotifyClientCredentials
from scipy import misc

client_credentials_manager = SpotifyClientCredentials("4aff136e10844aeaa521cc7b31dd7d0b", "c765b590f728472b915749f71e2786d2")  # Initialize spotify api
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)  # Create spotify object


def get_song_info(track_name, artist_name, features_list):  # Gets the info of the searched song to pass into the dataset
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
                    'liveness': features['liveness'], 
                    'valence': features['valence'], 
                    'tempo': features['tempo'],
                    'duration_ms': features['duration_ms'],
                    'key': features['key'],
                    'mode': features['mode'],
                    'time_signature': features['time_signature']
                }
                
                df = df.append(new_row, ignore_index=True)  # Append new row to dataframe
                df = df[features_list]
                return df
    raise ValueError("Unable to find artist or song! {} by {}".format(track_name, artist_name))  


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
        pass  # If there is an error in the song lookup, just don't add it


def create_df(positives, negatives, output_file):  # Creates dataset of songs I like/don't like (Serializes two different files for balancing later)
        p_df = []  # Define empty list for all the positive data samples
        for playlist_id in positives:  # Loop through all the positives
            info = get_playlist_info(playlist_id, 1)
            if info is None:
                print("Too Large!")
                continue
            p_df.append(info)  # Extract the song info for each
            print(info.head(1))
        p_df = pd.concat(p_df)  # Concat all the positive songs

        p_df['verdict'] = p_df['verdict'].astype('int')
        p_df.to_pickle("{}_positive".format(output_file))  # Serialize the data frame

        n_df = []  # Define empty list for all the negative data samples
        for playlist_id in negatives:  # Loop through all the negatives
            info = get_playlist_info(playlist_id, 0)
            if info is None:
                print("Too Large!")
                continue
            n_df.append(info)  # Extract the song info for each
            print(info.head(1))
                
        n_df = pd.concat(n_df)  # Concat all the negative songs

        n_df['verdict'] = n_df['verdict'].astype('int')
        n_df.to_pickle("{}_negative".format(output_file))  # Serialize the data frame

    


def display_histogram(df, features):  # Input a array of features and automatically display all graph data
    figure = plt.figure(figsize=(15, 15))  # Define 15x15 grid for histograms

    for index, feature in enumerate(features, start=1):  # Loop through all the features to graph
        hist_id = int("33{}".format(index))  # Create auto incrementing id for each histogram

        pos = df[df['verdict'] == 1][feature]  # Define positive range
        neg = df[df['verdict'] == 0][feature]  # Define negative range
        axis = figure.add_subplot(hist_id)  # Add the current histogram to the figure

        axis.set_xlabel(feature)  # Define x axis
        axis.set_ylabel("Count")  # Define y axis
        axis.set_title("Song {} Comparison".format(feature.capitalize()))  # Auto generate title for graph
        pos.hist(alpha=0.5, bins=30)  # Make positive graph semi-tranparent to see stacking
        axis2 = figure.add_subplot(hist_id)  # Add the second graph to the same sub plot
        neg.hist(alpha=0.5, bins=30)  # Make negative graph semi-tranparent to see stacking

    plt.show()  # Display the graph


def display_tree(tree, features, path):  # Display tree design (Found online)
    f = io.StringIO()
    export_graphviz(tree, out_file=f, feature_names=features)
    pydotplus.graph_from_dot_data(f.getvalue()).write_png(path)
    img = imageio.imread(path)
    plt.rcParams["figure.figsize"] = (20,20)
    plt.imshow(img)


def train_model(train, features):  # Input df and features to create tree, return trained model
    c = DecisionTreeClassifier(min_samples_split=100)  # Define amount of branches of splits
    
    # Create training data
    X_train = train[features]
    y_train = train["verdict"]

    dt = c.fit(X_train, y_train)  # Train model
    return dt  # Return trained model


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--title", type=str, required=True, help="title of song to test")
    ap.add_argument("-a", "--artist", type=str, required=True, help="artist of song")
    ap.add_argument("-d", "--display", type=int, default=-1, help="Whether or not graph should be displayed")
    args = vars(ap.parse_args())

    outputFile = "data/SpotifyData"  # Define prefix for serialized data file
    songTitle = args['title'].strip()
    songArtist = args['artist'].strip()

    # Id's of playlist with songs I like/don't like
    positive_songs = [
        "3Ho3iO0iJykgEQNbjB2sic", 
        "37i9dQZF1EjfkFhCYC8kMJ", 
        "37i9dQZF1E9WCZtRi5ziiy", 
        "37i9dQZF1EtdCxZ90JCh5B", 
        "37i9dQZF1DZ06evO4keHBe", 
        "37i9dQZF1DZ06evO0uJWHS", 
        "37i9dQZF1DWWGFQLoP9qlv", 
        "37i9dQZF1DWWiDhnQ2IIru", 
        "6R7sBHnTajW2RSSOBxkwox", 
        "5IQuYVRlf2UNyFZ5irJidP", 
        "37i9dQZF1DXbSbnqxMTGx9", 
        "37i9dQZF1DZ06evO2aNAaI", 
        "37i9dQZF1DZ06evO3ZBu5a", 
        "37i9dQZF1DZ06evO2Ael1n", 
        "37i9dQZF1DWZMCPjHG57gq", 
        "37i9dQZF1DX9wC1KY45plY", 
        "1EUCUV5dwPenYxq6BXiF4J",
        "0oZwhjUzewJXV0fx7ebjKA"
    ]
    negative_songs = [
        "37i9dQZF1DXcz8eC5kMSWZ", 
        "3LXm8Fx6qJTp2Jrue5nV5y", 
        "37i9dQZF1DWV50ly9XYOua", 
        "37i9dQZF1DX0XUsuxWHRQd", 
        "43mnKzy2LaOV1q6LhsJERY", 
        "6NcZ2oRnWrNImoTfj2SFi6",
        "37i9dQZF1DX4JAvHpjipBk",
        "37i9dQZEVXbdTUVEnH2rcC",
        "4bHIl5A3vizYVGsFJMkq5k",
        "06S64oWn7uGnfGrwjh5IWb",
        "7KD6Sxlel6a6sP5EJYE5tW",
        "696luHNN8urjfK9XP1Bn2m",
        "4pvXfBhqpoCckQuQ7YZBgn",
        "5EposIonErtCNfY4p2OpMw",
        "37i9dQZF1DXe9hay4VT07f",
        "37i9dQZF1DX0BcQWzuB7ZO"
    ]

    if not os.path.isfile("{}_positive".format(outputFile)) and not os.path.isfile("{}_negative".format(outputFile)):  # If the file does not exist
        print("Creating Dataset!")
        create_df(positive_songs, negative_songs, "SpotifyData")  # Create serialized data set
        print("Two Files Serialized in {}".format(outputFile))
    
    features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'key', 'mode']  # Define the features to train off

    df_pos = pickle.load(open("{}_positive".format(outputFile), 'rb'))
    df_neg = pickle.load(open("{}_negative".format(outputFile), 'rb'))
    
    df = pd.concat([df_pos, df_neg])  # Load the serialized dataframe  # Combine the data sets into one large one

    if args["display"] > 0: display_histogram(df, features[:9])  # Display histogram data

    train, test = train_test_split(df, test_size=0.2)  # Split the test and training data x% for testing

    X_test = test[features]  # Define the testing data
    y_test = test["verdict"]  # Define the testing answers

    model = train_model(train, features)  # Train the model

    #display_tree(model, features, "data/tree.png")  # Save image of decision tree

    song = get_song_info(songTitle, songArtist, features)

    y_pred = model.predict(song)  # Predict the inputed values
    
    if y_pred == 1:
        print("You will like {} by {}".format(songTitle, songArtist))
    else:
        print("You will not like {} by {}".format(songTitle, songArtist))
