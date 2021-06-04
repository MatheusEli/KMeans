from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    # ler dados
    dados = pd.read_csv("r15.csv")

    # instanciar o kMeans
    # 1 parametro = n de grupos

    km = KMeans(n_clusters=15, init='k-means++', n_init=10, max_iter=300, tol=1e-4)

    # executar o kMeans
    agrupamento = km.fit(dados)

    # instanciar visualização
    plt.scatter(dados.iloc[:, 0], dados.iloc[:, 1],
                c=agrupamento.labels_, label='True Position')

    # apresentar o gráfico
    plt.show()
