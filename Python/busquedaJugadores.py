import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

df = pd.read_csv("./oficialNormalizado.csv")
df.shape

df2 = df[["Jugador","Equipo","Liga","PosicionPrimaria","PosicionSecundaria","PosicionTerciaria"
    ,"Edad","VM","FechaFinContrato","PaisNacimiento","Altura","Peso","Pie"]]

df = df[:-1]
df = df[(df.PosicionPrimaria).str.contains("LWB|LB")]
df = df[(df.MinutosJugados >= 500)]
df = df.rename(columns={"%DuelosGanados": "PorcDuelosGanados", "AccionesDefensivasRealizadas/90": "AccionesDefensivasRealizadas90"
    ,"%DuelosDefensivosGanados": "PorcDuelosDefensivosGanados", "%ExitoCentrosDer": "ExitoCentrosDer"})

def filtradoDataframe(dataframe):
    df = dataframe[(dataframe.MinutosJugados >= 900)]
    X, y = df.iloc[:,19:117].values, df.iloc[:,0].values
    sc = StandardScaler()
    X_std = sc.fit_transform(X)
    
    pca = PCA(n_components = 32)
    pca.fit(X_std)
    X_pca = pca.transform(X_std)
    
    exp1 = pca.explained_variance_ratio_
    print('Varianza 4 componentes',sum(exp1[0:4]))
    print('Varianza 8 componentes',sum(exp1[0:8]))
    print('Varianza 12 componentes',sum(exp1[0:12]))
    print('Varianza 16 componentes',sum(exp1[0:16]))
    print('Varianza 20 componentes',sum(exp1[0:20]))
    print('Varianza 24 componentes',sum(exp1[0:24]))
    print('Varianza 28 componentes',sum(exp1[0:28]))
    print('Varianza 32 componentes',sum(exp1[0:32]))
    print(df.shape)
    
    df_pca_resultado = pd.DataFrame(data = X_pca, columns = ["PC1","PC2","PC3","PC4","PC5","PC6","PC7","PC8","PC9","PC10","PC11","PC12"
            ,"PC13","PC14","PC15","PC16","PC17","PC18","PC19","PC20","PC21","PC22","PC23","PC24","PC25"
            ,"PC26","PC27","PC28","PC29","PC30","PC31","PC32"], index=y)
    df_pca_resultado = df_pca_resultado.rename_axis('Jugador')
    return df_pca_resultado

df_pca_resultado = filtradoDataframe(df)

df_pca_resultado.head()

#Definición del método pearson para buscar la similitud entre jugadores
def metodoPearson(df, jugador):
    df2 = df.T.corr(method='pearson')
    df2 = df2.filter(like=jugador, axis=0)
    df2 = df2.T
    df2 = df2*100
    df2 = df2.sort_values(by=[jugador],ascending=False)
    return df2 

#Probamos con un jugador cualquiera
data = metodoPearson(df_pca_resultado,'Renan Lodi')

dataframe = pd.merge(data, df2, on='Jugador')
dataframe = dataframe.drop_duplicates(subset=['Jugador'])

dataframe = dataframe[dataframe["VM"] <= 500000].reset_index(drop=True)
dataframe = dataframe[dataframe["Liga"] == "España2B"].reset_index(drop=True)
dataframe.head(10)


