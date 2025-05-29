import joblib
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np

def names_models_routes(df_names_models, route_s2s):
    #select model
    lr_model_name = df_names_models.loc[df_names_models['s2s_route'] == route_s2s, 'lr_models'].values[0]
    svr_model_name = df_names_models.loc[df_names_models['s2s_route'] == route_s2s, 'svr_models'].values[0]
    rnn_model_name = df_names_models.loc[df_names_models['s2s_route'] == route_s2s, 'rnn_models'].values[0]
    scalerx_name = df_names_models.loc[df_names_models['s2s_route'] == route_s2s, 'rnn_scaler_x'].values[0]
    scalery_name = df_names_models.loc[df_names_models['s2s_route'] == route_s2s, 'rnn_scaler_y'].values[0]
    new_model = df_names_models.loc[df_names_models['s2s_route'] == route_s2s, 'new_models'].values[0]
    s2s_id = df_names_models.loc[df_names_models['s2s_route'] == route_s2s, 's2s_id'].values[0]
    return lr_model_name, svr_model_name, rnn_model_name, scalerx_name, scalery_name, new_model

def names_models_distance(df_names_models, type_model):
    #select model
    lr_model_name = df_names_models.loc[df_names_models['models'] == type_model, 'lr_models'].values[0]
    svr_model_name = df_names_models.loc[df_names_models['models'] == type_model, 'svr_models'].values[0]
    rnn_model_name = df_names_models.loc[df_names_models['models'] == type_model, 'rnn_models'].values[0]
    scalerx_name = df_names_models.loc[df_names_models['models'] == type_model, 'rnn_scaler_x'].values[0]
    scalery_name = df_names_models.loc[df_names_models['models'] == type_model, 'rnn_scaler_y'].values[0]

    return lr_model_name, svr_model_name, rnn_model_name, scalerx_name, scalery_name

def load_models(folder_path, lr_model_name, svr_model_name, rnn_model_name, scalerx_name, scalery_name, new_model_name=None):
    #load models
    lr_model = joblib.load(folder_path + 'lr_models/' + lr_model_name)
    svr_model = joblib.load(folder_path + 'svr_models/' + svr_model_name)
    rnn_model = load_model(folder_path + 'rnn_models/models/' + rnn_model_name)
    scaler_X = joblib.load(folder_path + 'rnn_models/scalers/' + scalerx_name)
    scaler_y = joblib.load(folder_path + 'rnn_models/scalers/' + scalery_name)

    if new_model_name:
        new_model = joblib.load(folder_path + 'new_models/' + new_model_name)
        return lr_model, svr_model, rnn_model, scaler_X, scaler_y, new_model
    
    return lr_model, svr_model, rnn_model, scaler_X, scaler_y

def ensemble(SVR_prediction,LR_prediction,RNN_prediction):
    pred_matrix = np.array([SVR_prediction, LR_prediction, RNN_prediction])
    diferencias = np.abs(pred_matrix - np.mean(pred_matrix))
    pesos_inversos = 1 / (diferencias + 1e-5)
    pesos_normalizados = pesos_inversos / np.sum(pesos_inversos)
    ensemble = round(np.dot(pesos_normalizados,pred_matrix),2)
    return ensemble

def ml_prediction(model, distance, month):
    X_pred = np.array([[distance, month]])
    prediction = model.predict(X_pred)
    return prediction[0]

def ml_distance_prediction(model, distance, month):
    X_pred = np.array([[distance, month]])
    prediction = model.predict(X_pred)
    return prediction[0]

def rnn_prediction(model, scaler_X, scaler_y, distance, month, id):
    X_pred = np.array([[distance, month, id]])
    X_pred = scaler_X.transform(X_pred)
    X_pred = X_pred.reshape((X_pred.shape[0], 1, X_pred.shape[1]))
    prediction = model.predict(X_pred)
    prediction = scaler_y.inverse_transform(prediction)
    return prediction[0][0]

def rnn_distance_prediction(model, scaler_X, scaler_y, distance, month):
    X_pred = np.array([[distance, month]])
    X_pred = scaler_X.transform(X_pred)
    X_pred = X_pred.reshape((X_pred.shape[0], 1, X_pred.shape[1]))
    prediction = model.predict(X_pred)
    prediction = scaler_y.inverse_transform(prediction)
    return prediction[0][0]

def price_pred(broker_id, route_s2s, query, distance, month):
    if query == 'city2city' or query == 'state2state':
        folder_path = "api_local/models_s2s/"
        df_names_models = pd.read_csv('api_local/paths_routes_models.csv')
        lr_model_name, svr_model_name, rnn_model_name, scalerx_name, scalery_name, new_model_name = names_models_routes(df_names_models, route_s2s)
        lr_model, svr_model, rnn_model, scaler_X, scaler_y, new_model = load_models(folder_path, lr_model_name, svr_model_name, rnn_model_name, scalerx_name, scalery_name, new_model_name)
        lr_pred = ml_prediction(lr_model, distance, month)
        svr_pred = ml_prediction(svr_model, distance, month)
        new_pred = ml_prediction(new_model, distance, month)
        rnn_pred = rnn_prediction(rnn_model, scaler_X, scaler_y, distance, month, int(df_names_models.loc[df_names_models['s2s_route'] == route_s2s, 's2s_id'].values[0]))
        price = ensemble(svr_pred, lr_pred, rnn_pred)

        return lr_pred, svr_pred, rnn_pred, price, new_pred

    else:
        folder_path = "api_local/models_distance/"
        df_names_models = pd.read_csv("api_local/paths_distance_models.csv")

        type_model = 'short' if distance < 60 else 'long'

        lr_model_name, svr_model_name, rnn_model_name, scalerx_name, scalery_name = names_models_distance(df_names_models, type_model)
        lr_model, svr_model, rnn_model, scaler_X, scaler_y = load_models(folder_path, lr_model_name, svr_model_name, rnn_model_name, scalerx_name, scalery_name)
        lr_pred = ml_distance_prediction(lr_model, distance, month)
        svr_pred = ml_distance_prediction(svr_model, distance, month)
        rnn_pred = rnn_distance_prediction(rnn_model, scaler_X, scaler_y, distance, month)

        price = ensemble(svr_pred, lr_pred, rnn_pred)
        
        return lr_pred, svr_pred, rnn_pred, price, None

    
    