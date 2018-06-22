import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


label_train = train['label']
datas = train.drop(['label'],axis= 1)

Knn = KNeighborsClassifier(n_neighbors= 20)
Knn.fit(datas,label_train)
result = Knn.predict(test)

obj = pd.DataFrame({"ImageID":list(range(1,len(result)+1)),"Label":result})
obj.to_csv('result.csv',index= False,header= True)

