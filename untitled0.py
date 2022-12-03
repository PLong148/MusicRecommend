import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv(r"C:\Users\admin\Desktop\Project DS\data.csv")

#df.info()

feature_cols=['acousticness', 'danceability', 'duration_ms', 'energy',
              'instrumentalness', 'key', 'liveness', 'loudness', 'mode',
              'speechiness', 'tempo', 'time_signature', 'valence',]

scaler = MinMaxScaler()
normalized_df =scaler.fit_transform(df[feature_cols])

#pd.plotting.scatter_matrix(df[:500], figsize=( 20,12 ))

indices = pd.Series(df.index, index=df['song_title']).drop_duplicates()

cosine = cosine_similarity(normalized_df)

def generate_recommendation(song_title, model_type):

    score=list(enumerate(model_type[indices[song_title]]))
      
    similarity_score = sorted(score,key = lambda x:x[1],reverse = True)
    
    similarity_score = similarity_score[1:11]
    top_songs_index = [i[0] for i in similarity_score]
    top_songs=df['song_title'].iloc[top_songs_index]

    return top_songs

song = input("Enter your song: ")
print("Recommended Songs:")
print(generate_recommendation(song,cosine).values)


