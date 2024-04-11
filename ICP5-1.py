import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#principal component analysis
#apply PCA on CC dataset
ccData = pd.read_csv('C:\\Users\\anori\\OneDrive\\Desktop\\Programming\\CS4710 Intro to Machine Learning\\ICP5\\datasets (1) (3)\\datasets\\CC GENERAL.csv')
ccData.dropna(inplace=True)

xCC = ccData.drop(columns = ['CUST_ID'])

scalerCC = StandardScaler()
xCCScaled = scalerCC.fit_transform(xCC)

pcaCC = PCA(n_components=2)
xPCA = pcaCC.fit_transform(xCCScaled)

#apply Kmeans
kmeansCC = KMeans(n_clusters=3)
kmeansCC.fit(xPCA)

yClusterMeansCC= kmeansCC.predict(xPCA)
#calculate and print silhouette score
silhouetteScoreCC = silhouette_score(xPCA, yClusterMeansCC)
print("SilhouetteScoreCC: ", silhouetteScoreCC)


#use pd_speech_features.csv
speechDataSet = pd.read_csv('C:\\Users\\anori\\OneDrive\\Desktop\\Programming\\CS4710 Intro to Machine Learning\\ICP5\\datasets (1) (3)\\datasets\\pd_speech_features.csv')

xSpeech = speechDataSet.drop(columns = ['class'])
ySpeech = speechDataSet['class']

scalerSpeech = StandardScaler()
xSpeechScaled = scalerSpeech.fit_transform(xSpeech)

#apply PCA (k=3)
pcaSpeech = PCA(n_components=3)
xSpeechPCA = pcaSpeech.fit_transform(xSpeechScaled)

#use svm to report performance
xTrain, xTest, yTrain, yTest = train_test_split(xSpeechPCA, ySpeech, test_size=0.2, random_state=42)
svmClassifier = SVC()
svmClassifier.fit(xTrain, yTrain)

svmScore = svmClassifier.score(xTest, yTest)
print('SVM score: ', svmScore)

#apply LDA on iris
irisData = pd.read_csv('C:\\Users\\anori\\OneDrive\\Desktop\\Programming\\CS4710 Intro to Machine Learning\\ICP5\\datasets (1) (3)\\datasets\\Iris.csv')
xIris = irisData.drop(columns=['Species'])
yIris = irisData['Species']

ldaIris = LinearDiscriminantAnalysis(n_components=2)
xIrisLda = ldaIris.fit_transform(xIris, yIris)


#identify the difference between pca and lda
#pca works to maximize variance and is a unsupervised technique
#lda works to maximize class separability and is a supervised technique