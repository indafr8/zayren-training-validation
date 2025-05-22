from processing import ProcessingData

from sklearn.svm import SVR
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import pandas as pd
import joblib
import os

import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)

class TrainModels():
    def __init__(self, broker_id):
        self.broker_id = broker_id
        self._create_folders()
        processing = ProcessingData(broker_id)
        self.data = processing.get_data()
        self.df_distances = self._avg_distances(self.data)
        self.train_data = self._normalize_data(self.data)
    
    def _create_folders(self):
        print('Creating folders...')
        #main folder
        if not os.path.exists(f'output_data/{self.broker_id}/'):
            os.makedirs(f'output_data/{self.broker_id}/')
        
        # models route folders
        if not os.path.exists(f'output_data/{self.broker_id}/models_s2s/'):
            os.makedirs(f'output_data/{self.broker_id}/models_s2s/')
        
        if not os.path.exists(f'output_data/{self.broker_id}/models_s2s/lr_models/'):
            os.makedirs(f'output_data/{self.broker_id}/models_s2s/lr_models/')
        
        if not os.path.exists(f'output_data/{self.broker_id}/models_s2s/svr_models/'):
            os.makedirs(f'output_data/{self.broker_id}/models_s2s/svr_models/')
        
        if not os.path.exists(f'output_data/{self.broker_id}/models_s2s/rnn_models/'):
            os.makedirs(f'output_data/{self.broker_id}/models_s2s/rnn_models/')
        
        if not os.path.exists(f'output_data/{self.broker_id}/models_s2s/rnn_models/models/'):
            os.makedirs(f'output_data/{self.broker_id}/models_s2s/rnn_models/models/')
        
        if not os.path.exists(f'output_data/{self.broker_id}/models_s2s/rnn_models/scalers/'):
            os.makedirs(f'output_data/{self.broker_id}/models_s2s/rnn_models/scalers/')
        
        if not os.path.exists(f'output_data/{self.broker_id}/models_s2s/new_models/'):
            os.makedirs(f'output_data/{self.broker_id}/models_s2s/new_models/')
        
        #distance models
        if not os.path.exists(f'output_data/{self.broker_id}/models_distance/lr_models/'):
            os.makedirs(f'output_data/{self.broker_id}/models_distance/lr_models/')
        
        if not os.path.exists(f'output_data/{self.broker_id}/models_distance/svr_models/'):
            os.makedirs(f'output_data/{self.broker_id}/models_distance/svr_models/')
        
        if not os.path.exists(f'output_data/{self.broker_id}/models_distance/rnn_models/'):
            os.makedirs(f'output_data/{self.broker_id}/models_distance/rnn_models/')
        
        if not os.path.exists(f'output_data/{self.broker_id}/models_distance/rnn_models/models/'):
            os.makedirs(f'output_data/{self.broker_id}/models_distance/rnn_models/models/')
        
        if not os.path.exists(f'output_data/{self.broker_id}/models_distance/rnn_models/scalers/'):
            os.makedirs(f'output_data/{self.broker_id}/models_distance/rnn_models/scalers/')
            
    
    def _avg_distances(self, data):
        unique_routes = data['c2c_route'].unique()

        s2s_id = []
        s2s_route = []
        c2c_route = []
        distance = []

        for route in unique_routes:
            if str(route) != 'nan':
                df_temp = data[data['c2c_route'] == route]
                s2s_id.append(df_temp['s2s_id'].values[0])
                s2s_route.append(df_temp['s2s_route'].values[0])
                c2c_route.append(df_temp['c2c_route'].values[0])
                distance.append(int(df_temp['distance'].mean()))
        
        df_result = pd.DataFrame({
            's2s_id': s2s_id,
            's2s_route': s2s_route,
            'c2c_route': c2c_route,
            'distance': distance
        })

        return df_result
    
    def _normalize_data(self, data):
        df = data.copy()
        scaler = StandardScaler()
        df[['distance']] = scaler.fit_transform(df[['distance']])

        df = df[['distance','month','adjust_price_usd']]
        return
    
    def _new_model(self,X,Y):
        x = X.copy()[['distance','month']]
        y = Y.copy()
        lm = RANSACRegressor()     # Linear regression model
        lm.fit(x, y)

        return lm
        
    def _svr_train(self, X, Y):
        x = X.copy()[['distance', 'month']]
        y = Y.copy()
        
        param_grid = {
            'C': [0.1, 1, 10, 15, 20, 50],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10],
            'epsilon': [0.01, 0.1, 0.2, 0.5]
        }
        
        svr = SVR(kernel="rbf")
        grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(x, y)
        
        best_model = grid_search.best_estimator_
        print(f"Mejores hiperparámetros: {grid_search.best_params_}")
        
        return best_model
    
    def _lr_train(self,X,Y):
        x = X.copy()[['distance','month']]
        y = Y.copy()
        lm = LinearRegression()     # Linear regression model
        lm.fit(x, y)

        return lm

    def _rnn_train(self,X,Y):
        x = X.copy()[['distance','month','s2s_id']]
        y = Y.copy()

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(x)
        y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.1, random_state=42)
        # Convertir las entradas a un formato adecuado para la RNN
        X_train = np.array(X_train).reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test = np.array(X_test).reshape((X_test.shape[0], 1, X_test.shape[1]))

        early_stopping = EarlyStopping(monitor='val_loss', 
                                        min_delta=0.0001, 
                                        patience=15, 
                                        verbose=1, 
                                        mode='min', 
                                        baseline=0.003, 
                                        restore_best_weights=True)

        model = self._architecture(X_train)
        # Entrenamiento del modelo con early stopping
        model.fit(X_train, y_train, 
                            epochs=200, 
                            batch_size=32, 
                            validation_data=(X_test, y_test), 
                            verbose=1, 
                            callbacks=[early_stopping])
        
        output = (model, scaler_X, scaler_y)
        return output
        
    def _architecture(self,m):
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(m.shape[1], m.shape[2]), return_sequences=True))
        # Batch normalization para estabilización y acortar tiempos de entrenamiento
        model.add(BatchNormalization()) 
        # Segunda capa LSTM con Batch Normalization
        model.add(LSTM(50, activation='relu', return_sequences=False))
        model.add(BatchNormalization())
        # Añadimos dropout para evitar sobreajustes
        model.add(Dropout(0.2))
        # Capa totalmente conectada
        model.add(Dense(1))
        # Compilamos el modelo
        model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
        return model

    def _train_route_models(self):
        routes = self.data['s2s_id'].unique()
        self.routes_id = routes
        self.routes = [self.data.loc[self.data['s2s_id'] == route, 's2s_route'].iloc[0] for route in self.routes_id]

        rnn_models = {}
        svr_models = {}
        lr_models = {}
        new_models = {}
        
        for route in routes:
            print('-----------------------------------\n')
            print(f'Training route: {route}')
            print('\n-----------------------------------')
            df_route = self.data[self.data['s2s_id'] == route]
            X = df_route[['distance','month','s2s_id']]
            Y = df_route['adjust_price_usd']

            svr_models[route] = self._svr_train(X,Y)
            lr_models[route] = self._lr_train(X,Y)
            rnn_models[route] = self._rnn_train(X,Y)
            new_models[route] = self._new_model(X,Y)
        
        return svr_models, lr_models, rnn_models, new_models
    
    def _svr_distance_train(self, X, Y):
        x = X.copy()[['distance','month']]
        y = Y.copy()
        sigma = 31.23318
        gamma_value = 1 / (2 * sigma ** 2)
        svr = SVR(kernel="rbf", C=15, gamma=gamma_value,epsilon=0.1)          # SVM model
        svr.fit(x, y)

        return svr
    
    def _lr_distance_train(self,X,Y):
        x = X.copy()[['distance','month']]
        y = Y.copy()
        lm = RANSACRegressor()     # Linear regression model
        lm.fit(x, y)

        return lm
    
    def _rnn_distance_train(self,X,Y):
        x = X.copy()[['distance','month']]
        y = Y.copy()

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(x)
        y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.1, random_state=42)
        # Convertir las entradas a un formato adecuado para la RNN
        X_train = np.array(X_train).reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test = np.array(X_test).reshape((X_test.shape[0], 1, X_test.shape[1]))

        early_stopping = EarlyStopping(monitor='val_loss', 
                                        min_delta=0.0001, 
                                        patience=15, 
                                        verbose=1, 
                                        mode='min', 
                                        baseline=0.003, 
                                        restore_best_weights=True)

        model = self._architecture(X_train)
        # Entrenamiento del modelo con early stopping
        model.fit(X_train, y_train, 
                            epochs=200, 
                            batch_size=32, 
                            validation_data=(X_test, y_test), 
                            verbose=1, 
                            callbacks=[early_stopping])
        
        output = (model, scaler_X, scaler_y)
        return output
    
    def _train_distance_models(self):
        distance_thres = 60

        data1 = self.data[self.data['distance'] <= distance_thres]
        data2 = self.data[self.data['distance'] > distance_thres]

        svr_models = {}
        lr_models = {}
        rnn_models = {}

        data = [data1, data2]
        labels = ['short', 'long']

        for i in range(2):
            print('-----------------------------------\n')
            print(f'Training distance model: {labels[i]}')
            print('\n-----------------------------------')
            df_route = data[i]
            X = df_route[['distance','month']]
            Y = df_route['adjust_price_usd']

            svr_models[labels[i]] = self._svr_distance_train(X,Y)
            lr_models[labels[i]] = self._lr_distance_train(X,Y)
            rnn_models[labels[i]] = self._rnn_distance_train(X,Y)

        return svr_models, lr_models, rnn_models
        #return lr_models
    
    def get_distance_models(self):
        svr_models, lr_models, rnn_models = self._train_distance_models()
        svr_files = []
        lr_files = []
        rnn_files = []
        scalerx_files = []
        scalery_files = []

        for route, model in lr_models.items():
            name =f"lr_model_{route}.pkl"
            joblib.dump(model, f'output_data/{self.broker_id}/models_distance/lr_models/{name}')
            lr_files.append(name)
        
        for route, model in svr_models.items():
            name = f"svr_model_{route}.pkl"
            joblib.dump(model, f'output_data/{self.broker_id}/models_distance/svr_models/{name}')
            svr_files.append(name)
        
        for route, model in rnn_models.items():
            name1 = f"rnn_model_{route}.h5"
            name2 = f"scaler_X_{route}.pkl"
            name3 = f"scaler_y_{route}.pkl"
            model[0].save(f'output_data/{self.broker_id}/models_distance/rnn_models/models/{name1}')
            joblib.dump(model[1], f'output_data/{self.broker_id}/models_distance/rnn_models/scalers/{name2}')
            joblib.dump(model[2], f'output_data/{self.broker_id}/models_distance/rnn_models/scalers/{name3}')
            rnn_files.append(name1)
            scalerx_files.append(name2)
            scalery_files.append(name3)

        paths = {
            'models': ['short', 'long'],
            'lr_models': lr_files,
            'svr_models': svr_files,
            'rnn_models': rnn_files,
            'scaler_X': scalerx_files,
            'scaler_y': scalery_files
        }

        df = pd.DataFrame(paths)
        df.to_csv('output_data/0/paths_distance_models.csv', index=False)

    def get_models(self):
        svr_models, lr_models, rnn_models, new_models = self._train_route_models()
        svr_files = []
        lr_files = []
        rnn_files = []
        new_files = []
        scalerx_files = []
        scalery_files = []

        for route, model in lr_models.items():
            name = f"lr_model_{route}.pkl"
            joblib.dump(model, f'output_data/{self.broker_id}/models_s2s/lr_models/{name}')
            lr_files.append(name)
        
        for route, model in svr_models.items():
            name = f"svr_model_{route}.pkl"
            joblib.dump(model, f'output_data/{self.broker_id}/models_s2s/svr_models/{name}')
            svr_files.append(name)
        
        for route, model in rnn_models.items():
            name1 = f"rnn_model_{route}.h5"
            name2 = f"scaler_X_{route}.pkl"
            name3 = f"scaler_y_{route}.pkl"
            model[0].save(f'output_data/{self.broker_id}/models_s2s/rnn_models/models/{name1}')
            joblib.dump(model[1], f'output_data/{self.broker_id}/models_s2s/rnn_models/scalers/{name2}')
            joblib.dump(model[2], f'output_data/{self.broker_id}/models_s2s/rnn_models/scalers/{name3}')
            rnn_files.append(name1)
            scalerx_files.append(name2)
            scalery_files.append(name3)

        for route, model in new_models.items():
            name = f"new_model_{route}.pkl"
            joblib.dump(model, f'output_data/{self.broker_id}/models_s2s/new_models/{name}')
            new_files.append(name)

        paths = {
            's2s_id': self.routes_id,
            's2s_route': self.routes,
            'lr_models': lr_files,
            'svr_models': svr_files,
            'rnn_models': rnn_files,
            'scaler_X': scalerx_files,
            'scaler_y': scalery_files,
            'new_models': new_files
        }

        df = pd.DataFrame(paths)
        df.to_csv('output_data/0/paths_routes_models.csv', index=False)
        
        
if __name__ == '__main__':
    broker_id = 0
    tm = TrainModels(broker_id)
    distances = tm.df_distances
    distances.to_csv(f'output_data/{broker_id}/distances.csv', index=False)
    #tm.get_models()
    #tm.get_distance_models()