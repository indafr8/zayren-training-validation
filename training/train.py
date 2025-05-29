from typing import Dict, List, Tuple, Union, Any, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import os
import warnings
import logging
from dataclasses import dataclass
from sklearn.svm import SVR
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from processing import ProcessingData

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

warnings.simplefilter(action='ignore', category=RuntimeWarning)

@dataclass
class ModelConfig:
    """Configuración para los modelos de entrenamiento."""
    batch_size: int = 32
    epochs: int = 200
    early_stopping_patience: int = 15
    early_stopping_delta: float = 0.0001
    early_stopping_baseline: float = 0.003
    test_size: float = 0.1
    random_state: int = 42
    distance_threshold: int = 60
    learning_rate: float = 0.01
    lstm_units: int = 50
    dropout_rate: float = 0.2
    min_samples_per_route: int = 10
    validation_split: float = 0.2
    max_models_per_route: int = 4

class TrainModels:
    """
    Clase para entrenar diferentes modelos de machine learning para predicción de precios.
    
    Esta clase maneja el entrenamiento de:
    - Modelos por ruta (SVR, Linear Regression, RNN)
    - Modelos por distancia (corta/larga)
    - Preprocesamiento y normalización de datos
    """
    
    def __init__(self, broker_id: str, config: Optional[ModelConfig] = None, new_data_folder: Optional[str] = None):
        """
        Inicializa la clase TrainModels.

        Args:
            broker_id: ID del broker
            config: Configuración para los modelos
        """
        self.broker_id = str(broker_id)
        self.config = config or ModelConfig()
        self.output_dir = Path(f'output_data/{self.broker_id}')
        self.models_dir = self.output_dir / 'models'
        self.checkpoints_dir = self.output_dir / 'checkpoints'
        
        try:
            self._create_folders()
            processing = ProcessingData(broker_id, train_data_folder=new_data_folder)
            self.dirty_data = processing.dirty_data
            self.dirty_data.to_csv(f'past_data/base_data_{self.broker_id}.csv', index=False)
            self.data = processing.clean_data
            
            if self.data.empty:
                raise ValueError("No hay datos disponibles para entrenar")
                
            self.df_distances = self._avg_distances(self.data)
            self.train_data = self._normalize_data(self.data)
            
            # Validar datos
            self._validate_data()
            
        except Exception as e:
            logger.error(f"Error al inicializar TrainModels: {str(e)}")
            raise

    def _validate_data(self) -> None:
        """Valida los datos de entrada."""
        required_columns = ['distance', 'month', 'adjust_price_usd', 's2s_id', 's2s_route']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        
        if missing_columns:
            raise ValueError(f"Faltan columnas requeridas: {missing_columns}")
            
        if self.data['distance'].min() <= 0:
            raise ValueError("Hay distancias menores o iguales a cero")
            
        if self.data['adjust_price_usd'].min() <= 0:
            raise ValueError("Hay precios menores o iguales a cero")

    def _create_folders(self) -> None:
        """Crea la estructura de directorios necesaria para los modelos."""
        folders = [
            self.output_dir,
            self.models_dir,
            self.checkpoints_dir,
            self.models_dir / 'models_s2s',
            self.models_dir / 'models_s2s/lr_models',
            self.models_dir / 'models_s2s/svr_models',
            self.models_dir / 'models_s2s/rnn_models',
            self.models_dir / 'models_s2s/rnn_models/models',
            self.models_dir / 'models_s2s/rnn_models/scalers',
            self.models_dir / 'models_s2s/new_models',
            self.models_dir / 'models_distance/lr_models',
            self.models_dir / 'models_distance/svr_models',
            self.models_dir / 'models_distance/rnn_models',
            self.models_dir / 'models_distance/rnn_models/models',
            self.models_dir / 'models_distance/rnn_models/scalers'
        ]
        
        for folder in folders:
            folder.mkdir(parents=True, exist_ok=True)
    
    def _avg_distances(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula las distancias promedio por ruta.

        Args:
            data: DataFrame con los datos originales

        Returns:
            pd.DataFrame: DataFrame con las distancias promedio
        """
        routes = data['c2c_route'].dropna().unique()
        
        result_data = {
            's2s_id': [],
            's2s_route': [],
            'c2c_route': [],
            'distance': []
        }
        
        for route in routes:
            df_temp = data[data['c2c_route'] == route]
            if len(df_temp) >= self.config.min_samples_per_route:
                result_data['s2s_id'].append(df_temp['s2s_id'].iloc[0])
                result_data['s2s_route'].append(df_temp['s2s_route'].iloc[0])
                result_data['c2c_route'].append(route)
                result_data['distance'].append(int(df_temp['distance'].mean()))
        
        return pd.DataFrame(result_data)
    
    def _normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normaliza los datos de entrenamiento.

        Args:
            data: DataFrame con los datos a normalizar

        Returns:
            pd.DataFrame: DataFrame con los datos normalizados
        """
        df = data.copy()
        scaler = StandardScaler()
        df[['distance']] = scaler.fit_transform(df[['distance']])
        return df[['distance', 'month', 'adjust_price_usd']]
    
    def _create_lstm_model(self, input_shape: Tuple[int, int]) -> Model:
        """
        Crea la arquitectura del modelo LSTM.

        Args:
            input_shape: Forma de los datos de entrada

        Returns:
            Model: Modelo LSTM compilado
        """
        model = Sequential([
            LSTM(self.config.lstm_units, activation='relu', 
                 input_shape=input_shape, return_sequences=True),
            BatchNormalization(),
            LSTM(self.config.lstm_units, activation='relu', return_sequences=False),
            BatchNormalization(),
            Dropout(self.config.dropout_rate),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _train_rnn(self, X: pd.DataFrame, y: pd.Series, route_id: str) -> Tuple[Model, MinMaxScaler, MinMaxScaler]:
        """
        Entrena un modelo RNN.

        Args:
            X: Features de entrenamiento
            y: Target de entrenamiento
            route_id: ID de la ruta para guardar checkpoints

        Returns:
            Tuple[Model, MinMaxScaler, MinMaxScaler]: Modelo entrenado y scalers
        """
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled,
            test_size=self.config.test_size,
            random_state=self.config.random_state
        )
        
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        
        checkpoint_path = self.checkpoints_dir / f'rnn_checkpoint_{route_id}.h5'
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                min_delta=self.config.early_stopping_delta,
                patience=self.config.early_stopping_patience,
                verbose=1,
                mode='min',
                baseline=self.config.early_stopping_baseline,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        model = self._create_lstm_model((X_train.shape[1], X_train.shape[2]))
        
        try:
            model.fit(
                X_train, y_train,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                validation_data=(X_test, y_test),
                verbose=1,
                callbacks=callbacks
            )
        except Exception as e:
            logger.error(f"Error entrenando modelo RNN para ruta {route_id}: {str(e)}")
            if checkpoint_path.exists():
                model = load_model(str(checkpoint_path))
            else:
                raise
        
        return model, scaler_X, scaler_y
    
    def _train_svr(self, X: pd.DataFrame, y: pd.Series) -> SVR:
        """
        Entrena un modelo SVR con búsqueda de hiperparámetros.

        Args:
            X: Features de entrenamiento
            y: Target de entrenamiento

        Returns:
            SVR: Modelo SVR entrenado
        """
        param_grid = {
            'C': [0.1, 1, 10, 15, 20, 50],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10],
            'epsilon': [0.01, 0.1, 0.2, 0.5]
        }
        
        svr = SVR(kernel="rbf")
        grid_search = GridSearchCV(
            svr, param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        try:
            grid_search.fit(X[['distance', 'month']], y)
            logger.info(f"Mejores hiperparámetros: {grid_search.best_params_}")
            return grid_search.best_estimator_
        except Exception as e:
            logger.error(f"Error en GridSearchCV: {str(e)}")
            # Fallback a parámetros por defecto
            svr = SVR(kernel="rbf", C=1.0, gamma='scale', epsilon=0.1)
            svr.fit(X[['distance', 'month']], y)
            return svr
    
    def _train_route_models(self) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        Entrena modelos para cada ruta.

        Returns:
            Tuple[Dict, Dict, Dict, Dict]: Modelos entrenados (svr, lr, rnn, new)
        """
        routes = self.data['s2s_id'].unique()
        self.routes_id = routes
        self.routes = [
            self.data.loc[self.data['s2s_id'] == route, 's2s_route'].iloc[0]
            for route in routes
        ]
        
        models = {
            'svr': {},
            'lr': {},
            'rnn': {},
            'new': {}
        }
        
        for route in routes:
            try:
                logger.info(f'\nTraining route: {route}')
                df_route = self.data[self.data['s2s_id'] == route]
                
                if len(df_route) < self.config.min_samples_per_route:
                    logger.warning(f"Ruta {route} tiene muy pocos registros, saltando...")
                    continue
                    
                X = df_route[['distance', 'month', 's2s_id']]
                y = df_route['adjust_price_usd']
                
                models['svr'][route] = self._train_svr(X, y)
                models['lr'][route] = LinearRegression().fit(X[['distance', 'month']], y)
                models['rnn'][route] = self._train_rnn(X, y, str(route))
                models['new'][route] = RANSACRegressor().fit(X[['distance', 'month']], y)
                
            except Exception as e:
                logger.error(f"Error entrenando modelos para ruta {route}: {str(e)}")
                continue
        
        return models['svr'], models['lr'], models['rnn'], models['new']
    
    def _train_distance_models(self) -> Tuple[Dict, Dict, Dict]:
        """
        Entrena modelos basados en distancia.

        Returns:
            Tuple[Dict, Dict, Dict]: Modelos entrenados (svr, lr, rnn)
        """
        data_short = self.data[self.data['distance'] <= self.config.distance_threshold]
        data_long = self.data[self.data['distance'] > self.config.distance_threshold]
        
        models = {
            'svr': {},
            'lr': {},
            'rnn': {}
        }
        
        for label, data in [('short', data_short), ('long', data_long)]:
            try:
                logger.info(f'\nTraining distance model: {label}')
                if len(data) < self.config.min_samples_per_route:
                    logger.warning(f"Modelo {label} tiene muy pocos registros, saltando...")
                    continue
                    
                X = data[['distance', 'month']]
                y = data['adjust_price_usd']
                
                models['svr'][label] = self._train_svr(X, y)
                models['lr'][label] = RANSACRegressor().fit(X, y)
                models['rnn'][label] = self._train_rnn(X, y, f"distance_{label}")
                
            except Exception as e:
                logger.error(f"Error entrenando modelo de distancia {label}: {str(e)}")
                continue
        
        return models['svr'], models['lr'], models['rnn']
    
    def _save_models(self, models: Dict, model_type: str, is_distance: bool = False) -> List[str]:
        """
        Guarda los modelos entrenados.

        Args:
            models: Diccionario de modelos a guardar
            model_type: Tipo de modelo ('svr', 'lr', 'rnn', 'new')
            is_distance: Si los modelos son basados en distancia

        Returns:
            List[str]: Rutas de los archivos guardados
        """
        base_path = self.models_dir / ('models_distance' if is_distance else 'models_s2s')
        saved_files = []
        scalers_x = []
        scalers_y = []
        
        for key, model in models.items():
            try:
                if model_type == 'rnn':
                    model_path = base_path / f'rnn_models/models/rnn_model_{key}.h5'
                    scaler_x_path = base_path / f'rnn_models/scalers/scaler_X_{key}.pkl'
                    scaler_y_path = base_path / f'rnn_models/scalers/scaler_y_{key}.pkl'
                    
                    model[0].save(str(model_path))
                    joblib.dump(model[1], str(scaler_x_path))
                    joblib.dump(model[2], str(scaler_y_path))
                    
                    saved_files.append(f'rnn_model_{key}.h5')
                    scalers_x.append(f'scaler_X_{key}.pkl')
                    scalers_y.append(f'scaler_y_{key}.pkl')
                else:
                    model_path = base_path / f'{model_type}_models/{model_type}_model_{key}.pkl'
                    joblib.dump(model, str(model_path))
                    saved_files.append(f'{model_type}_model_{key}.pkl')
                    
            except Exception as e:
                logger.error(f"Error guardando modelo {model_type} para {key}: {str(e)}")
                continue
        
        if model_type == 'rnn':
            return saved_files, scalers_x, scalers_y
        return saved_files
    
    def get_distance_models(self) -> None:
        """Entrena y guarda los modelos basados en distancia."""
        try:
            svr_models, lr_models, rnn_models = self._train_distance_models()
            
            # Obtener las etiquetas que tienen todos los modelos entrenados
            common_labels = set(svr_models.keys()) & set(lr_models.keys()) & set(rnn_models.keys())
            
            # Filtrar y guardar solo los modelos de las etiquetas comunes
            rnn_files, scalers_x, scalers_y = self._save_models(
                {k: rnn_models[k] for k in common_labels}, 
                'rnn', 
                True
            )
            
            saved_files = {
                'models': list(common_labels),
                'lr_models': self._save_models({k: lr_models[k] for k in common_labels}, 'lr', True),
                'svr_models': self._save_models({k: svr_models[k] for k in common_labels}, 'svr', True),
                'rnn_models': rnn_files,
                'rnn_scaler_x': scalers_x,
                'rnn_scaler_y': scalers_y
            }
            
            # Verificar que todos los arrays tienen la misma longitud
            lengths = {k: len(v) for k, v in saved_files.items()}
            if len(set(lengths.values())) > 1:
                logger.warning(f"Longitudes diferentes en saved_files de distancia: {lengths}")
                # Encontrar la longitud mínima
                min_length = min(lengths.values())
                # Truncar todos los arrays a la longitud mínima
                saved_files = {k: v[:min_length] for k, v in saved_files.items()}
            
            pd.DataFrame(saved_files).to_csv(
                self.output_dir / 'paths_distance_models.csv',
                index=False
            )
            
        except Exception as e:
            logger.error(f"Error en get_distance_models: {str(e)}")
            raise
    
    def get_models(self) -> None:
        """Entrena y guarda los modelos por ruta."""
        try:
            svr_models, lr_models, rnn_models, new_models = self._train_route_models()
            
            # Obtener las rutas que tienen todos los modelos entrenados
            common_routes = set(svr_models.keys()) & set(lr_models.keys()) & set(rnn_models.keys()) & set(new_models.keys())
            
            # Filtrar las rutas y sus nombres
            filtered_routes_id = []
            filtered_routes = []
            for route_id in common_routes:
                route_name = self.data[self.data['s2s_id'] == route_id]['s2s_route'].iloc[0]
                filtered_routes_id.append(route_id)
                filtered_routes.append(route_name)
            
            # Filtrar y guardar solo los modelos de las rutas comunes
            rnn_files, scalers_x, scalers_y = self._save_models({k: rnn_models[k] for k in common_routes}, 'rnn')
            
            saved_files = {
                's2s_id': filtered_routes_id,
                's2s_route': filtered_routes,
                'lr_models': self._save_models({k: lr_models[k] for k in common_routes}, 'lr'),
                'svr_models': self._save_models({k: svr_models[k] for k in common_routes}, 'svr'),
                'rnn_models': rnn_files,
                'rnn_scaler_x': scalers_x,
                'rnn_scaler_y': scalers_y,
                'new_models': self._save_models({k: new_models[k] for k in common_routes}, 'new')
            }
            
            # Verificar que todos los arrays tienen la misma longitud
            lengths = {k: len(v) for k, v in saved_files.items()}
            if len(set(lengths.values())) > 1:
                logger.warning(f"Longitudes diferentes en saved_files: {lengths}")
                # Encontrar la longitud mínima
                min_length = min(lengths.values())
                # Truncar todos los arrays a la longitud mínima
                saved_files = {k: v[:min_length] for k, v in saved_files.items()}
            
            pd.DataFrame(saved_files).to_csv(
                self.output_dir / 'paths_routes_models.csv',
                index=False
            )
            
        except Exception as e:
            logger.error(f"Error en get_models: {str(e)}")
            raise

if __name__ == '__main__':
    try:
        broker_id = '0'
        config = ModelConfig()
        #new_data_folder = Path('../data/0/DB_0004__2025-01-27').resolve()
        new_data_folder = Path('../data/0/DB_0005__2025-03-07').resolve()
        print(f"Usando carpeta de datos: {new_data_folder}")
        
        if not new_data_folder.exists():
            raise FileNotFoundError(f"La carpeta de datos no existe: {new_data_folder}")
            
        tm = TrainModels(broker_id, config, new_data_folder)
        
        # Guardar distancias
        tm.df_distances.to_csv(
            tm.output_dir / 'distances.csv',
            index=False
        )
        
        # Entrenar modelos
        #tm.get_models()
        #tm.get_distance_models()
        
    except Exception as e:
        logger.error(f"Error en el programa principal: {str(e)}")
        raise