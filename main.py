import pandas as pd
import spotipy


def get_song_info(track_name, artist):
    try:
        info = pd.read_csv('SpotifyFeatures.csv')
        track_search = info.loc[(info['track_name'] == track_name) & (info['artist_name'] == artist)]
        track_search.sort_values(by=['popularity'], inplace=True)
        track = track_search.iloc[-1]
        return track
    except:
        raise IndexError('Invalid song title or name!')

get_song_info("Wish You Were Here", "Pink Floyd")