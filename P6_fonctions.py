import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import time

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.manifold import TSNE 
from sklearn import cluster, metrics

from matplotlib import offsetbox
from tensorflow.keras.utils import load_img, img_to_array
from matplotlib.image import imread
from skimage.transform import resize

''' Fonctions générales '''


def missing_cells(df):
    '''Calcule le nombre de cellules manquantes sur le data set total.
    Keyword arguments:
    df -- le dataframe

    return : le nombre de cellules manquantes de df
    '''
    return df.isna().sum().sum()


def missing_cells_perc(df):
    '''Calcule le pourcentage de cellules manquantes sur le data set total.
    Keyword arguments:
    df -- le dataframe

    return : le pourcentage de cellules manquantes de df
    '''
    return df.isna().sum().sum()/(df.size)


def missing_general(df):
    '''Donne un aperçu général du nombre de données manquantes dans le data frame.
    Keyword arguments:
    df -- le dataframe
    '''
    print('Nombre total de cellules manquantes :', missing_cells(df))
    print('Nombre de cellules manquantes en % : {:.2%}'
          .format(missing_cells_perc(df)))


def valeurs_manquantes(df):
    '''Prend un data frame en entrée et créer en sortie un dataframe contenant
    le nombre de valeurs manquantes et leur pourcentage pour chaque variables.
    Keyword arguments:
    df -- le dataframe

    return : dataframe contenant le nombre de valeurs manquantes et
    leur pourcentage pour chaque variable
    '''
    tab_missing = pd.DataFrame(columns=['Variable',
                                        'Missing values',
                                        'Missing (%)'])
    tab_missing['Variable'] = df.columns
    missing_val = list()
    missing_perc = list()

    for var in df.columns:
        nb_miss = missing_cells(df[var])
        missing_val.append(nb_miss)
        perc_miss = missing_cells_perc(df[var])
        missing_perc.append(perc_miss)

    tab_missing['Missing values'] = list(missing_val)
    tab_missing['Missing (%)'] = list(missing_perc)
    return tab_missing


def bar_missing(df):
    '''Affiche le barplot présentant le nombre de données présentes par variable.
    Keyword arguments:
    df -- le dataframe
    '''
    msno.bar(df)
    plt.title('Nombre de données présentes par variable', size=15)
    plt.show()


def barplot_missing(df):
    '''Affiche le barplot présentant le pourcentage de
    données manquantes par variable.
    Keyword arguments:
    df -- le dataframe
    '''
    proportion_nan = df.isna().sum()\
        .divide(df.shape[0]/100).sort_values(ascending=False)

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 30))
    sns.barplot(y=proportion_nan.index, x=proportion_nan.values)
    plt.title('Pourcentage de données manquantes par variable', size=15)
    plt.show()

def bar_plot(df, colonnes, long, larg):
    ''' Affiche les bar plots pour chaque variable renseignée.
    Keyword arguments:
    df -- le dataframe
    colonnes -- variables à afficher
    long -- nombre de figure en longueur
    larg -- nombre de figure en largeur
    '''
    fig = plt.figure(figsize=(40, 40))
    for i, col in enumerate(colonnes, 1):
        ax = fig.add_subplot(long, larg, i)
        count = df[col].value_counts()
        count.plot(kind="bar", ax=ax)
        plt.xticks(rotation=90, ha='right', fontsize=20)
        ax.set_title(col, fontsize=20)
    plt.tight_layout(pad=2)
    plt.show()


def pie_plot(df, colonnes):
    '''Affiche un pie plot présentant la répartition de la variable renseignée.
    Keyword arguments:
    df -- le dataframe
    colonnes -- variables à afficher
    '''
    for col in colonnes:
        labels = list(df[col].value_counts().sort_index().index.astype(str))
        count = df[col].value_counts().sort_index()

        plt.figure(figsize=(10, 10))
        plt.pie(count, autopct='%1.2f%%')
        plt.title('Répartition de {}'.format(col), size=20)
        plt.legend(labels)
        plt.show()

def ARI_fct_tsne(features, liste_cat, label_true) :
    '''Calcul Tsne, détermination des clusters et calcul ARI entre les vraies catégories et n° de clusters.'''
    time1 = time.time()
    nb_clusters = len(liste_cat)
    
    tsne = TSNE(n_components=2, perplexity=30, n_iter=2000, 
                init='random', learning_rate=200, random_state=0)
    X_tsne = tsne.fit_transform(features)
    
    # Détermination des clusters à partir des données après Tsne 
    cls = KMeans(n_clusters=nb_clusters, n_init=100, random_state=0)
    cls.fit(X_tsne)
    ARI = np.round(adjusted_rand_score(label_true, cls.labels_),4)
    time2 = np.round(time.time() - time1,0)
    print("ARI : ", ARI, "time : ", time2)
    
    return ARI, X_tsne, cls.labels_

def TSNE_visu_fct(X_tsne, liste_cat, label_true, labels, ARI) :
    '''Visualisation du Tsne selon les vraies catégories et selon les clusters.'''
    fig = plt.figure(figsize=(15,6))
    
    ax = fig.add_subplot(121)
    scatter = ax.scatter(X_tsne[:,0],X_tsne[:,1], c=label_true, cmap='Set1')
    ax.legend(handles=scatter.legend_elements()[0], labels=liste_cat, loc="best", title="Categorie")
    plt.title('Représentation par catégories réelles')
    
    ax = fig.add_subplot(122)
    scatter = ax.scatter(X_tsne[:,0],X_tsne[:,1], c=labels, cmap='Set1')
    ax.legend(handles=scatter.legend_elements()[0], labels=set(labels), loc="best", title="Clusters")
    plt.title('Représentation par clusters')
    
    plt.show()
    print("ARI : ", ARI)
       
def eboulis(pca):
    '''Réalise un éboulis de valeurs propres'''
    scree = pca.explained_variance_ratio_*100
    scree_cum = scree.cumsum()
    
    plt.figure(figsize=(10,5))
    plt.bar(np.arange(len(scree)) + 1, scree)
    plt.plot(np.arange(len(scree)) + 1, scree_cum,c="red",marker='o')
    plt.xlabel("Rang de l'axe d'inertie")
    plt.ylabel("Pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    plt.show(block=False)

def plot_TSNE_images(X, df, path):
    '''Affiche le graphe TSNE en 2 dimensions avec les images à la place des points.'''
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure(figsize=(15, 15))
    ax = plt.subplot(111)

    if hasattr(offsetbox, 'AnnotationBbox'):
        shown_images = np.array([[1., 1.]])
        for i in range(df.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 5e-4:
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            props={'boxstyle':'round', 'edgecolor':'white'}
            
            image = imread(path + df['image'][i])
            image = resize(image, (230, 230)) 

            imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(image,zoom=0.1),
                                                X[i], bboxprops=props)
            ax.add_artist(imagebox)

''' Fonctions traitement de texte '''

def tokenizer_fct(sentence) :
    ''' Tokenisation du texte en argument.'''
    word_tokens = word_tokenize(sentence)
    return word_tokens

def stop_word_filter_fct(list_words, stop_w) :
    ''' Filtrage de la liste de mots avec les stopwords renseignés en argument.'''
    filtered_w = [w for w in list_words if not w in stop_w]
    filtered_w2 = [w for w in filtered_w if len(w) > 2]
    return filtered_w2

def lower_start_fct(list_words) :
    '''Transformation de la liste de mots en minuscule.'''
    lw = [w.lower() for w in list_words]
    return lw

def lemma_fct(list_words) :
    '''Lemmatizer'''
    lemmatizer = WordNetLemmatizer()
    lem_w = [lemmatizer.lemmatize(w) for w in list_words]
    return lem_w

def stem_fct(list_words) :
    '''Lemmatizer'''
    stemmer = PorterStemmer()
    stem_w = [stemmer.stem(w) for w in list_words]
    return stem_w

def transform_bow_fct(text, stop_w) :
    '''Fonction de préparation du texte pour le bag of words (Countvectorizer et Tf_idf, Word2Vec)'''
    word_tokens = tokenizer_fct(text)
    sw = stop_word_filter_fct(word_tokens, stop_w)
    lw = lower_start_fct(sw)
    transf_desc_text = ' '.join(lw)
    return transf_desc_text

def transform_bow_lem_fct(text, stop_w) :
    '''Fonction de préparation du texte pour le bag of words avec lemmatization.'''
    word_tokens = tokenizer_fct(text)
    sw = stop_word_filter_fct(word_tokens, stop_w)
    lw = lower_start_fct(sw)
    lem_w = lemma_fct(lw)    
    transf_desc_text = ' '.join(lem_w)
    return transf_desc_text

def transform_bow_stem_fct(text, stop_w) :
    '''Fonction de préparation du texte pour le bag of words avec stemming.'''
    word_tokens = tokenizer_fct(text)
    sw = stop_word_filter_fct(word_tokens, stop_w)
    lw = lower_start_fct(sw)
    stem_w = stem_fct(lw)    
    transf_desc_text = ' '.join(stem_w)
    return transf_desc_text

def transform_dl_fct(text) :
    '''Fonction de préparation du texte pour le Deep learning (USE et BERT).'''
    word_tokens = tokenizer_fct(text)
    lw = lower_start_fct(word_tokens)
    transf_desc_text = ' '.join(lw)
    return transf_desc_text

def display_topics(model, feature_names, no_top_words):
    '''Affiche les topics trouvés.'''
    for topic_idx, topic in enumerate(model.components_):
        print("Topic {}:".format(topic_idx))
        print(" ".join([feature_names[i] 
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

''' Fonctions traitement d'images '''

def conf_mat_transform(y_true, y_pred, corresp):
    '''Créer une matrice de confusion en fonction de la correspondance donnée en paramètres.'''
    conf_mat = metrics.confusion_matrix(y_true,y_pred)
    
    if corresp == 'argmax':
        corresp = np.argmax(conf_mat, axis=0)
    
    print ("Correspondance des clusters : ", corresp)
    labels = pd.Series(y_true, name="y_true").to_frame()
    labels['y_pred'] = y_pred
    labels['y_pred_transform'] = labels['y_pred'].apply(lambda x : corresp[x]) 
    
    return labels['y_pred_transform']