from typing import List, Optional, Union
import pandas as pd
import numpy as np
from pathlib import Path
from update import UpdateData

class ProcessingData:
    """
    Clase para procesar y limpiar datos de transporte.
    
    Esta clase realiza el procesamiento de datos incluyendo:
    - Cálculo de precio por milla
    - Filtrado de outliers usando IQR
    - Winsorización de datos
    """
    
    def __init__(self, broker_id: str):
        """
        Inicializa la clase ProcessingData.

        Args:
            broker_id: ID del broker para procesar sus datos
        """
        self.broker_id = broker_id
        self.output_dir = Path(f'output_data/{self.broker_id}')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            update = UpdateData(broker_id)
            self.data = update.get_update_data()
            self.data = self.__price_mile(self.data)
            self.unique_routes = self.data['s2s_route'].unique()
            self.data = self.__preprocessing()
            self.data.to_csv(self.output_dir / 'segments_data.csv', index=False)
        except Exception as e:
            print(f"Error al procesar datos: {str(e)}")
            raise
    
    def get_data(self) -> pd.DataFrame:
        """
        Obtiene los datos procesados.

        Returns:
            pd.DataFrame: DataFrame con los datos procesados
        """
        return self.data

    def __preprocessing(self) -> pd.DataFrame:
        """
        Realiza el preprocesamiento de datos para cada ruta única.

        Returns:
            pd.DataFrame: DataFrame con todos los datos procesados
        """
        processed_data: List[pd.DataFrame] = []

        for route in self.unique_routes:
            try:
                df = self.data[self.data['s2s_route'] == route].copy()
                df = self.__iqr_filter(df)
                df = self.__winsoration(df)         
                #df = self.__smooth_winsoration(df)
                if not df.empty:
                    processed_data.append(df)
            except Exception as e:
                print(f"Error procesando ruta {route}: {str(e)}")
                continue

        if not processed_data:
            return pd.DataFrame()
            
        return pd.concat(processed_data, ignore_index=True)

    def __price_mile(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula el precio por milla para cada registro.

        Args:
            data: DataFrame con los datos originales

        Returns:
            pd.DataFrame: DataFrame con la columna price_mile agregada
        """
        df = data.copy()
        df['price_mile'] = df['adjust_price_usd'] / df['distance']
        return df
    
    def __iqr_filter(self, data: pd.DataFrame, min_records: int = 10) -> pd.DataFrame:
        """
        Filtra outliers usando el método IQR.

        Args:
            data: DataFrame a filtrar
            min_records: Número mínimo de registros requeridos después del filtrado

        Returns:
            pd.DataFrame: DataFrame filtrado o el original si no hay suficientes registros
        """
        df = data.copy()
        
        # Calcular cuartiles y límites
        Q1 = df['price_mile'].quantile(0.25)
        Q3 = df['price_mile'].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filtrar outliers
        df_filtered = df[
            (df['price_mile'] >= lower_bound) & 
            (df['price_mile'] <= upper_bound)
        ]

        return df_filtered if len(df_filtered) >= min_records else df
    
    def __winsoration(
        self, 
        data: pd.DataFrame,
        lower_percentile: float = 0.05,
        upper_percentile: float = 0.95
    ) -> pd.DataFrame:
        """
        Aplica winsorización a los datos para limitar valores extremos.

        Args:
            data: DataFrame a procesar
            lower_percentile: Percentil inferior para la winsorización
            upper_percentile: Percentil superior para la winsorización

        Returns:
            pd.DataFrame: DataFrame con los datos winsorizados
        """
        df = data.copy()
        
        # Calcular límites
        q95 = df['price_mile'].quantile(upper_percentile)
        q05 = df['price_mile'].quantile(lower_percentile)
        
        # Aplicar winsorización
        df['price_mile'] = df['price_mile'].clip(lower=q05, upper=q95)
        
        return df.reset_index(drop=True)
    
    def __smooth_winsoration(
        self,
        data: pd.DataFrame,
        column: str = 'price_mile',
        lower_percentile: float = 0.05,
        upper_percentile: float = 0.95,
        smoothing_factor: float = 0.25
    ) -> pd.DataFrame:
        """
        Aplica winsorización suavizada a los datos.

        Args:
            data: DataFrame a procesar
            column: Columna a winsorizar
            lower_percentile: Percentil inferior
            upper_percentile: Percentil superior
            smoothing_factor: Factor de suavizado (0-1)

        Returns:
            pd.DataFrame: DataFrame con los datos winsorizados suavemente
        """
        df = data.copy()
        
        # Calcular límites
        lower_limit = df[column].quantile(lower_percentile)
        upper_limit = df[column].quantile(upper_percentile)

        # Función de suavizado
        def smooth(value: float) -> float:
            if value < lower_limit:
                return lower_limit + (value - lower_limit) * smoothing_factor
            elif value > upper_limit:
                return upper_limit + (value - upper_limit) * smoothing_factor
            return value

        # Aplicar suavizado
        df[column] = df[column].apply(smooth)
        
        return df