# KModes_using_python

    from urllib.request import urlopen
    import json
    import pandas as pd

    #====================================open data=======================================
    url = "https://gist.githubusercontent.com/jorinvo/7f19ce95a9a842956358/raw/e319340c2f6691f9cc8d8cc57ed532b5093e3619/data.json"
    response = urlopen(url)
    data_json = json.loads(response.read())
    #print(data_json)
    
    df = pd.DataFrame.from_dict(data_json, orient='columns')

    #drop missing value
    cleansing=df.dropna()

    #check double data --> beda nama harusnya beda credit card
    duplicateCC = cleansing[cleansing.duplicated(['creditcard'])]
    #duplicateCC
    #===========================KModes Clustering=================================
    #import warnings
    warnings.filterwarnings('ignore')

    # Importing all required packages
    import numpy as np
    import pandas as pd
    from kmodes.kmodes import KModes

    # Data viz lib
    import matplotlib.pyplot as plt
    import seaborn as sns
    %matplotlib inline
    from matplotlib.pyplot import xticks

    #========================indentify K optimum====================================
    cost = []
    K = range(1,5)
    for num_clusters in list(K):
    kmode = KModes(n_clusters=num_clusters, init = "random", n_init = 5, verbose=2)
    kmode.fit_predict(cleansing)
    cost.append(kmode.cost_)
    
    plt.plot(K, cost, 'bx-')
    plt.xlabel('No. of clusters')
    plt.ylabel('Cost')
    plt.title('Elbow Method For Optimal k')
    plt.show()

    #==============================Clustering K=3==================================
    kmode = KModes(n_clusters=3, init = "random", n_init = 5, verbose=1)
    clusters = kmode.fit_predict(cleansing)
    clusters

    #cetak tabel cluster
    cleansing.insert(0, "Cluster", clusters, True)
    cleansing

    #save hasil clustering
    cleansing.to_csv('C:/Users/Bappenas/OneDrive/Pribadi/Narasi TV/clust.csv', index=False) #folder path bisa disesuaikan
