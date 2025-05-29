import pandas as pd
import numpy as np
from api_local.ai_service import price_pred
from api_local.stats_service import city2city_info, state2state_info
from api_local.utils_service import load_data, query_type
import warnings
from typing import Dict, List, Tuple
from datetime import datetime
import json
from pathlib import Path

warnings.filterwarnings("ignore")

class Validation:
    def __init__(self, broker_id: str = '0'):
        """
        Inicializa el sistema de validación.
        
        Args:
            broker_id: ID del broker a validar
        """
        self.broker_id = broker_id
        self.output_dir = Path(f'output_data/{self.broker_id}/validation')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Cargar datos
        self.df_old = pd.read_csv(f'base_data_{self.broker_id}.csv')
        self.df_new = pd.read_csv(f'new_data_{self.broker_id}.csv')

        # Convertir a datetime
        self.df_old['inserted_at'] = pd.to_datetime(self.df_old['inserted_at'])
        self.df_new['inserted_at'] = pd.to_datetime(self.df_new['inserted_at'])

        # Procesar datos
        self.df_new = self.__new_instances(self.df_old, self.df_new)
        self.results = self.__predict_price(self.df_new)
        
        # Calcular métricas
        self.metrics = self.__calculate_metrics()
        
        # Guardar resultados
        self.__save_results()

    def __new_instances(self, df_old: pd.DataFrame, df_new: pd.DataFrame) -> pd.DataFrame:
        """
        Filtra las nuevas instancias y muestra estadísticas.
        
        Args:
            df_old: DataFrame con datos históricos
            df_new: DataFrame con datos nuevos
            
        Returns:
            pd.DataFrame: DataFrame con solo las nuevas instancias
        """
        df_new = df_new[df_new['inserted_at'] > df_old['inserted_at'].max()]
        
        print("\n=== Estadísticas de Datos ===")
        print(f"Última actualización en dataset histórico: {df_old['inserted_at'].max()}")
        print(f"Última actualización en dataset nuevo: {df_new['inserted_at'].max()}")
        print(f"Primera actualización en dataset nuevo: {df_new['inserted_at'].min()}")
        print(f"Número de nuevas instancias: {df_new.shape[0]:,}")
        print(f"Rutas únicas en nuevo dataset: {len(df_new['s2s_route'].unique()):,}")
        
        return df_new

    def __calculate_error_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calcula métricas de error para cada modelo.
        
        Args:
            data: DataFrame con los resultados
            
        Returns:
            Dict[str, float]: Diccionario con las métricas calculadas
        """
        metrics = {}
        models = ['lr', 'svr', 'rnn', 'ensamble', 'new']
        
        for model in models:
            # Calcular RMSE
            metrics[f'{model}_rmse'] = np.sqrt(np.mean((data[f'{model}_price'] - data['price_usd'])**2))
            
            # Calcular MAE
            metrics[f'{model}_mae'] = np.mean(np.abs(data[f'{model}_price'] - data['price_usd']))
            
            # Calcular MAPE
            mape = np.mean(np.abs((data[f'{model}_price'] - data['price_usd']) / data['price_usd'])) * 100
            metrics[f'{model}_mape'] = mape
            
            # Calcular R²
            ss_res = np.sum((data['price_usd'] - data[f'{model}_price'])**2)
            ss_tot = np.sum((data['price_usd'] - np.mean(data['price_usd']))**2)
            metrics[f'{model}_r2'] = 1 - (ss_res / ss_tot)
        
        return metrics

    def __calculate_threshold_metrics(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Calcula métricas de porcentaje de predicciones dentro de diferentes umbrales.
        
        Args:
            data: DataFrame con los resultados
            
        Returns:
            Dict[str, Dict[str, float]]: Diccionario con las métricas por umbral
        """
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
        models = ['lr', 'svr', 'rnn', 'ensamble', 'new']
        metrics = {}
        
        for model in models:
            metrics[model] = {}
            for threshold in thresholds:
                within_threshold = np.abs(data[f'{model}_diff']) <= threshold
                metrics[model][f'within_{int(threshold*100)}%'] = np.mean(within_threshold) * 100
        
        return metrics

    def __calculate_metrics(self) -> Dict:
        """
        Calcula todas las métricas de evaluación.
        
        Returns:
            Dict: Diccionario con todas las métricas
        """
        return {
            'error_metrics': self.__calculate_error_metrics(self.results),
            'threshold_metrics': self.__calculate_threshold_metrics(self.results),
            'timestamp': datetime.now().isoformat(),
            'total_samples': len(self.results),
            'unique_routes': len(self.results['s2s_route'].unique())
        }

    def __save_results(self) -> None:
        """Guarda los resultados y métricas en archivos."""
        # Guardar métricas
        metrics_file = self.output_dir / f'validation_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=4)
    
        
        print(f"\nResultados guardados en:")
        print(f"- Métricas: {metrics_file}")

    def print_metrics(self) -> None:
        """Imprime un resumen de las métricas de evaluación."""
        print("\n=== Métricas de Error ===")
        for model in ['lr', 'svr', 'rnn', 'ensamble', 'new']:
            print(f"\n{model.upper()}:")
            print(f"RMSE: ${self.metrics['error_metrics'][f'{model}_rmse']:,.2f}")
            print(f"MAE: ${self.metrics['error_metrics'][f'{model}_mae']:,.2f}")
            print(f"MAPE: {self.metrics['error_metrics'][f'{model}_mape']:.2f}%")
            print(f"R²: {self.metrics['error_metrics'][f'{model}_r2']:.4f}")
        
        print("\n=== Porcentaje de Predicciones Dentro del Umbral ===")
        for model in ['lr', 'svr', 'rnn', 'ensamble', 'new']:
            print(f"\n{model.upper()}:")
            for threshold in [10, 20, 30, 40, 50]:
                metric = self.metrics['threshold_metrics'][model][f'within_{threshold}%']
                print(f"Dentro de ±{threshold}%: {metric:.2f}%")

    def __predict_price(self, df_new: pd.DataFrame) -> pd.DataFrame:
        """
        Realiza predicciones de precios para cada ruta.
        
        Args:
            df_new: DataFrame con los datos a predecir
            
        Returns:
            pd.DataFrame: DataFrame con las predicciones
        """
        def predict_row(row):
            route_s2s = row['s2s_route']
            route_c2c = row['c2c_route']
            print(f"Predicting price for route {route_s2s}")

            segment_data, carrier_data = load_data(flag=0, broker_id=self.broker_id)
            query = query_type(route_c2c, route_s2s, segment_data)

            lr_price, svr_price, rnn_price, ensamble_price, new_price = price_pred(
                broker_id=self.broker_id, 
                route_s2s=route_s2s, 
                query=query, 
                distance=row['distance'], 
                month=row['month']
            )

            if query == 'city2city':
                c2c_stats = city2city_info(route_c2c, row['distance'], segment_data)
                s2s_stats = state2state_info(route_s2s, row['distance'], segment_data)
            else:
                s2s_stats = state2state_info(route_s2s, row['distance'], segment_data)
                c2c_stats = {
                    'mean_price': 0, 'range_min': 0, 'range_max': 0, 
                    'count_segments': 0, 'price_mille_median': 0, 
                    'last_trip': 0, 'last_price': 0, 
                    'mean_price_ten': 0, 'price_mille_ten': 0
                }

            return pd.Series([
                lr_price, svr_price, rnn_price, ensamble_price, new_price,
                c2c_stats['mean_price'], c2c_stats['range_min'], c2c_stats['range_max'],
                c2c_stats['count_segments'], c2c_stats['price_mille_median'],
                c2c_stats['last_trip'], c2c_stats['last_price'],
                c2c_stats['mean_price_ten'], c2c_stats['price_mille_ten'],
                s2s_stats['mean_price'], s2s_stats['range_min'], s2s_stats['range_max'],
                s2s_stats['count_segments'], s2s_stats['price_mille_median'],
                s2s_stats['last_trip'], s2s_stats['last_price'],
                s2s_stats['mean_price_ten'], s2s_stats['price_mille_ten']
            ])

        # Aplicar predicciones
        df_new[[
            'lr_price', 'svr_price', 'rnn_price', 'ensamble_price', 'new_price',
            'c2c_mean_price', 'c2c_range_min', 'c2c_range_max', 'c2c_count_segments',
            'c2c_price_mille_median', 'c2c_last_trip', 'c2c_last_price',
            'c2c_mean_price_ten', 'c2c_price_mille_ten',
            's2s_mean_price', 's2s_range_min', 's2s_range_max', 's2s_count_segments',
            's2s_price_mille_median', 's2s_last_trip', 's2s_last_price',
            's2s_mean_price_ten', 's2s_price_mille_ten'
        ]] = df_new.apply(predict_row, axis=1)

        # Calcular diferencias porcentuales
        for model in ['lr', 'svr', 'rnn', 'ensamble', 'new']:
            df_new[f'{model}_diff'] = np.where(
                df_new['price_usd'] == 0, 0,
                (df_new[f'{model}_price'] - df_new['price_usd']) / df_new['price_usd']
            )

        df_new['s2s_price_diff'] = np.where(
            df_new['price_usd'] == 0, 0,
            (df_new['s2s_mean_price'] - df_new['price_usd']) / df_new['price_usd']
        )

        return df_new

if __name__ == '__main__':
    # Ejecutar validación
    val = Validation()
    
    # Imprimir métricas
    val.print_metrics()