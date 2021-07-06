import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import KFold,RepeatedKFold
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

df = pd.read_excel('../raw.xlsx')
names = df.columns.to_list()
df_norm = (df[names]-np.mean(df[names]))/np.std(df[names])
y = np.array(df_norm['Eads']).reshape(-1,1)

for features in [['WF']]:
# for features in [['WF','WFIE','RAM'],['WF','WFIE','RAM','Ee'],['WF','WFIE','RAM','Ee','WEN']]:
    x = np.array(df_norm[features]).reshape(-1,len(features))
    gamma,C,R2,R2_train,R2_test,RMSE_train,MAE_train,RMSE_test,MAE_test=[],[],[],[],[],[],[],[],[]
    columns = ['gamma','C','R2','R2_train','R2_test','RMSE_train','MAE_train','RMSE_test','MAE_test']
    for c in [1e-3,1e-2,1e-1,1,1e1,1e2,1e3,1e4]:
        for g in [1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8]:
            model = SVR(kernel='rbf', C=c, gamma=g)
            gamma.append(g)
            C.append(c)
            folds,repeats = 10,50
            kf = RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=10)
            r2 = r2_score(y,model.fit(x,y.ravel()).predict(x))
            R2.append(r2)

            r2_train,r2_test,rmse_train,mae_train,rmse_test,mae_test=[],[],[],[],[],[]
            for train, test in kf.split(x,y):
                x_train,y_train = x[train],y[train]
                x_test,y_test = x[test],y[test]

                y_train_pred = model.fit(x_train,y_train.ravel()).predict(x_train)
                y_test_pred = model.fit(x_train,y_train.ravel()).predict(x_test)

                std = np.std(df['Eads']); mean = np.mean(df['Eads'])
                r2_train.append(r2_score(y_train*std+mean,y_train_pred*std+mean))
                r2_test.append(r2_score(y_test*std+mean,y_test_pred*std+mean))
                mae_test.append(mean_absolute_error(y_test*std+mean,y_test_pred*std+mean))
                mae_train.append(mean_absolute_error(y_train*std+mean,y_train_pred*std+mean))
                rmse_test.append(mean_squared_error(y_test*std+mean,y_test_pred*std+mean)**0.5)
                rmse_train.append(mean_squared_error(y_train*std+mean,y_train_pred*std+mean)**0.5)

            R2_train.append(np.array(r2_train).mean())
            R2_test.append(np.array(r2_test).mean())
            RMSE_train.append(np.array(rmse_train).mean())
            MAE_train.append(np.array(mae_train).mean())
            RMSE_test.append(np.array(rmse_test).mean())
            MAE_test.append(np.array(mae_test).mean())
            print(c,g)

    results = pd.DataFrame(data = list(zip(gamma,C,R2,R2_train,R2_test,RMSE_train,MAE_train,RMSE_test,MAE_test)),columns=columns)
    results = results.sort_values(by=['R2_test'],ascending=False)
    results.to_csv(f'demension_{len(features)}.csv',index=False)
