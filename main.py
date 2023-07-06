import warnings
from ast import literal_eval
import numpy as np
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
import requests

warnings.simplefilter('ignore')
warnings.simplefilter('once')


def supreme(Title):
    def sendData(l):
        Temp = []
        List = requests.get(
            "https://api.themoviedb.org/3/genre/movie/list?api_key=533c31a769099d0e37e2c33c02f3afa4&language=en-US").json()[
            'genres']
        genreList = {}
        for i in List:
            genreList[i['id']] = i['name']

        for film in l:
            d = {}
            x = requests.get(
                "https://api.themoviedb.org/3/search/movie?api_key=533c31a769099d0e37e2c33c02f3afa4&query=" + str(
                    film)).json()
            url = "http://image.tmdb.org/t/p/w500" + str(x.get('results')[0].get('poster_path'))
            Data = x.get('results')[0]

            # -----------------
            d['title'] = film
            d['url'] = url
            d['popularity'] = Data['popularity']
            d['release_date'] = Data['release_date']
            d['overview'] = Data['overview']
            d['genres'] = [genreList[i] for i in Data['genre_ids']]
            d['vote_average'] = Data['vote_average']
            # ------------------
            Temp.append(d)
        return Temp

    task = 1
    s = ["Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", "Drama", "Family", "Foreign", "Fantasy",
         "History", "Horror", "Music", "Mystery", "Romance", "Science Fiction", "Thriller", "War", "Western"]

    if Title.lower() in [i.lower() for i in s]:
        task = 2
    final = []
    for i in Title.split():
        temp = []
        X = 0
        for j in i:
            if X == 0:
                temp.append(j.upper())
                X += 1
            else:
                temp.append(j)
                X += 1
        final.append(''.join(temp))
    Title = ' '.join(final)

    md = pd.read_csv('data/movies_metadata.csv')

    md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(
        lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

    vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = md[md['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(0.95)

    md['year'] = pd.to_datetime(md['release_date'], errors='coerce').apply(
        lambda x: str(x).split('-')[0] if x != np.nan else np.nan)

    qualified = md[(md['vote_count'] >= m) & (md['vote_count'].notnull()) & (md['vote_average'].notnull())][
        ['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')

    def weighted_rating(x):
        v = x['vote_count']
        R = x['vote_average']
        return (v / (v + m) * R) + (m / (m + v) * C)

    qualified['wr'] = qualified.apply(weighted_rating, axis=1)

    s = md.apply(lambda x: pd.Series(x['genres']), axis=1).stack().reset_index(level=1, drop=True)

    s.name = 'genre'
    gen_md = md.drop('genres', axis=1).join(s)

    def build_chart(genre, percentile=0.85):
        DF = gen_md[gen_md['genre'] == genre]
        Vote_counts = DF[DF['vote_count'].notnull()]['vote_count'].astype('int')
        Vote_averages = DF[DF['vote_average'].notnull()]['vote_average'].astype('int')
        CC = Vote_averages.mean()
        M = Vote_counts.quantile(percentile)

        Qualified = DF[(DF['vote_count'] >= M) & (DF['vote_count'].notnull()) & (DF['vote_average'].notnull())][
            ['title', 'year', 'vote_count', 'vote_average', 'popularity']]
        Qualified['vote_count'] = Qualified['vote_count'].astype('int')
        Qualified['vote_average'] = Qualified['vote_average'].astype('int')

        Qualified['wr'] = Qualified.apply(
            lambda x: (x['vote_count'] / (x['vote_count'] + M) * x['vote_average']) + (M / (M + x['vote_count']) * CC),
            axis=1)
        Qualified = Qualified.sort_values('wr', ascending=False).head(250)

        return Qualified

    if task == 2:
        df = build_chart(Title).head(10)
        return task, Title, sendData(list(df['title']))

    links_small = pd.read_csv('data/links_small.csv')
    links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')

    md = md.drop([19730, 29503, 35587])

    md['id'] = md['id'].astype('int')
    smd = md[md['id'].isin(links_small)]

    smd['tagline'] = smd['tagline'].fillna('')
    smd['description'] = smd['overview'] + smd['tagline']
    smd['description'] = smd['description'].fillna('')

    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform(smd['description'])

    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    smd = smd.reset_index()
    indices = pd.Series(smd.index, index=smd['title'])

    Credits = pd.read_csv('data/credits.csv')
    for _ in Credits:
        Credits['id'] = Credits['id'].astype('int')

    keywords = pd.read_csv('data/keywords.csv')
    keywords['id'] = keywords['id'].astype('int')

    md['id'] = md['id'].astype('int')
    md = md.merge(Credits, on='id')
    md = md.merge(keywords, on='id')

    smd = md[md['id'].isin(links_small)]

    smd['cast'] = smd['cast'].apply(literal_eval)
    smd['crew'] = smd['crew'].apply(literal_eval)
    smd['keywords'] = smd['keywords'].apply(literal_eval)
    smd['cast_size'] = smd['cast'].apply(lambda x: len(x))
    smd['crew_size'] = smd['crew'].apply(lambda x: len(x))

    def get_director(x):
        for i in x:
            if i['job'] == 'Director':
                return i['name']
        return np.nan

    smd['director'] = smd['crew'].apply(get_director)
    smd['cast'] = smd['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    smd['cast'] = smd['cast'].apply(lambda x: x[:3] if len(x) >= 3 else x)
    smd['keywords'] = smd['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    smd['cast'] = smd['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
    smd['director'] = smd['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
    smd['director'] = smd['director'].apply(lambda x: [x, x])

    s = smd.apply(lambda x: pd.Series(x['keywords']), axis=1).stack().reset_index(level=1, drop=True)
    s.name = 'keyword'

    s = s.value_counts()
    s = s[s > 1]
    stemmer = SnowballStemmer('english')
    stemmer.stem('dogs')

    def filter_keywords(x):
        words = []
        for i in x:
            if i in s:
                words.append(i)
        return words

    smd['keywords'] = smd['keywords'].apply(filter_keywords)
    smd['keywords'] = smd['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
    smd['keywords'] = smd['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
    smd['soup'] = smd['keywords'] + smd['cast'] + smd['director'] + smd['genres']
    smd['soup'] = smd['soup'].apply(lambda x: ' '.join(x))

    count = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
    count_matrix = count.fit_transform(smd['soup'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    smd = smd.reset_index()
    indices = pd.Series(smd.index, index=smd['title'])

    def improved_recommendations(title):
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:26]
        movie_indices = [i[0] for i in sim_scores]

        movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year']]
        Vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
        Vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
        Vote_averages.mean()
        M = Vote_counts.quantile(0.60)
        Qualified = movies[
            (movies['vote_count'] >= M) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
        Qualified['vote_count'] = Qualified['vote_count'].astype('int')
        Qualified['vote_average'] = Qualified['vote_average'].astype('int')
        Qualified['wr'] = Qualified.apply(weighted_rating, axis=1)
        Qualified = Qualified.sort_values('wr', ascending=False).head(10)
        return Qualified

    if task == 1:
        return task, Title, sendData(list(improved_recommendations(Title)['title']))

    reader = Reader()

    ratings = pd.read_csv('data/ratings_small.csv')
    ratings.head()

    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

    svd = SVD()
    cross_validate(svd, data, measures=['RMSE', 'MAE'])
    Trainset = data.build_full_trainset()
    svd.fit(Trainset)

    svd.predict(1, 302, 3)

    def convert_int(x):
        try:
            return int(x)
        except:
            return np.nan


    id_map = pd.read_csv('data/links_small.csv')[['movieId', 'tmdbId']]
    id_map['tmdbId'] = id_map['tmdbId'].apply(convert_int)
    id_map.columns = ['movieId', 'id']
    id_map = id_map.merge(smd[['title', 'id']], on='id').set_index('title')

    indices_map = id_map.set_index('id')

    def hybrid(userId, title):
        idx = indices[title]
        # var = id_map.loc[title]['id']
        # var = id_map.loc[title]['movieId']

        sim_scores = list(enumerate(cosine_sim[int(idx)]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:26]
        movie_indices = [i[0] for i in sim_scores]

        movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year', 'id']]
        movies['est'] = movies['id'].apply(lambda x: svd.predict(userId, indices_map.loc[x]['movieId']).est)
        movies = movies.sort_values('est', ascending=False)
        return movies.head(10)

    if task == 1:
        df = hybrid(1, Title)
        return list(df['title'])
