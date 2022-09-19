import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

df = pd.read_csv("./oficial.csv")

df2 = df

df2['Goles'] = np.where(df2['Goles']>=0, (df2['Goles']*df2['ELO'])/79,0)
df2['xG'] = np.where(df2['xG']>=0, (df2['xG']*df2['ELO'])/79,0)
df2['Asistencias'] = np.where(df2['Asistencias']>=0, (df2['Asistencias']*df2['ELO'])/79,0)
df2['xA'] = np.where(df2['xA']>=0, (df2['xA']*df2['ELO'])/79,0)
df2['%DuelosGanados'] = np.where(df2['%DuelosGanados']>=0, (df2['%DuelosGanados']*df2['ELO'])/79,0)
df2['%DuelosDefensivosGanados'] = np.where(df2['%DuelosDefensivosGanados']>=0, (df2['%DuelosDefensivosGanados']*df2['ELO'])/79,0)
df2['%DuelosAereosGanados'] = np.where(df2['%DuelosAereosGanados']>=0, (df2['%DuelosAereosGanados']*df2['ELO'])/79,0)
df2['RecuperacionPosesionMedianteEntrada/90'] = np.where(df2['RecuperacionPosesionMedianteEntrada/90']>=0, (df2['RecuperacionPosesionMedianteEntrada/90']*df2['ELO'])/79,0)
df2['Interceptaciones/90'] = np.where(df2['Interceptaciones/90']>=0, (df2['Interceptaciones/90']*df2['ELO'])/79,0)
df2['RecuperacionPosesionMedianteIntercepcion/90'] = np.where(df2['RecuperacionPosesionMedianteIntercepcion/90']>=0, (df2['RecuperacionPosesionMedianteIntercepcion/90']*df2['ELO'])/79,0)
df2['AccionesAtaqueExitosas/90'] = np.where(df2['AccionesAtaqueExitosas/90']>=0, (df2['AccionesAtaqueExitosas/90']*df2['ELO'])/79,0)
df2['Goles/90'] = np.where(df2['Goles/90']>=0, (df2['Goles/90']*df2['ELO'])/79,0)
df2['GolesSinPenaltis'] = np.where(df2['GolesSinPenaltis']>=0, (df2['GolesSinPenaltis']*df2['ELO'])/79,0)
df2['GolesSinPenaltis/90'] = np.where(df2['GolesSinPenaltis/90']>=0, (df2['GolesSinPenaltis/90']*df2['ELO'])/79,0)
df2['xG/90'] = np.where(df2['xG/90']>=0, (df2['xG/90']*df2['ELO'])/79,0)
df2['GolesCabeza'] = np.where(df2['GolesCabeza']>=0, (df2['GolesCabeza']*df2['ELO'])/79,0)
df2['GolesCabeza/90'] = np.where(df2['GolesCabeza/90']>=0, (df2['GolesCabeza/90']*df2['ELO'])/79,0)
df2['%TirosAGol'] = np.where(df2['%TirosAGol']>=0, (df2['%TirosAGol']*df2['ELO'])/79,0)
df2['Asistencias/90'] = np.where(df2['Asistencias/90']>=0, (df2['Asistencias/90']*df2['ELO'])/79,0)
df2['%ExitoRegates'] = np.where(df2['%ExitoRegates']>=0, (df2['%ExitoRegates']*df2['ELO'])/79,0)
df2['%DuelosAtacantesGanados'] = np.where(df2['%DuelosAtacantesGanados']>=0, (df2['%DuelosAtacantesGanados']*df2['ELO'])/79,0)
df2['ToquesAreaPenalti/90'] = np.where(df2['ToquesAreaPenalti/90']>=0, (df2['ToquesAreaPenalti/90']*df2['ELO'])/79,0)
df2['xA/90'] = np.where(df2['xA/90']>=0, (df2['xA/90']*df2['ELO'])/79,0)
df2['Asistencias/90'] = np.where(df2['Asistencias/90']>=0, (df2['Asistencias/90']*df2['ELO'])/79,0)
df2['SegundasAsistencias/90'] = np.where(df2['SegundasAsistencias/90']>=0, (df2['SegundasAsistencias/90']*df2['ELO'])/79,0)
df2['TercerasAsistencias/90'] = np.where(df2['TercerasAsistencias/90']>=0, (df2['TercerasAsistencias/90']*df2['ELO'])/79,0)
df2['%PrecisionDesmarques'] = np.where(df2['%PrecisionDesmarques']>=0, (df2['%PrecisionDesmarques']*df2['ELO'])/79,0)
df2['JugadasClaves/90'] = np.where(df2['JugadasClaves/90']>=0, (df2['JugadasClaves/90']*df2['ELO'])/79,0)
df2['%PrecisionPasesUltimoTercio'] = np.where(df2['%PrecisionPasesUltimoTercio']>=0, (df2['%PrecisionPasesUltimoTercio']*df2['ELO'])/79,0)
df2['%PrecisionPasesAreaPequeña'] = np.where(df2['%PrecisionPasesAreaPequeña']>=0, (df2['%PrecisionPasesAreaPequeña']*df2['ELO'])/79,0)
df2['%PrecisionPasesProfundidad'] = np.where(df2['%PrecisionPasesProfundidad']>=0, (df2['%PrecisionPasesProfundidad']*df2['ELO'])/79,0)
df2['%PrecisionPasesProgresivos'] = np.where(df2['%PrecisionPasesProgresivos']>=0, (df2['%PrecisionPasesProgresivos']*df2['ELO'])/79,0)

df2.to_csv('oficialNormalizado.csv',encoding='utf-8-sig')


