from update import UpdateData
import pandas as pd
import numpy as np

class ProcessingData():
    def __init__(self,broker_id):
        self.broker_id = broker_id
        Update = UpdateData(broker_id)
        self.data = Update.get_update_data()
        self.data = self._price_mile(self.data)
        self.unique_routes = self.data['s2s_route'].unique()
        self.data = self._preprocessing()
        self.data.to_csv(f'output_data/{self.broker_id}/segments_data.csv', index=False)
    
    def get_data(self):
        #imprimir numero de renglones
        return self.data

    def _preprocessing(self):
        processed_data = []  # Lista en lugar de DataFrame vacío

        for route in self.unique_routes:
            df = self.data[self.data['s2s_route'] == route]
            df = self._iqr_filter(df)
            df = self._winsoration(df)
            
            if df is not None and not df.empty:  # Verifica que df no sea None ni vacío
                processed_data.append(df)

        return pd.concat(processed_data, axis=0).reset_index(drop=True) if processed_data else pd.DataFrame()


    def _price_mile(self, data):
        df = data.copy()
        df['price_mile'] = df['adjust_price_usd'] / df['distance']
        return df
    
    def _iqr_filter(self, data):
        df = data.copy()
        Q1 = df['price_mile'].quantile(0.25)
        Q3 = df['price_mile'].quantile(0.75)
        IQR = Q3 - Q1

        # Definir límites
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filtrar el DataFrame
        df_filtered = df[(df['price_mile'] >= lower_bound) & (df['price_mile'] <= upper_bound)]

        if len(df_filtered) >=10:
            df_filtered = df_filtered.reset_index(drop=True)
            return df_filtered
        else:
            return df
        
    def _winsoration(self, data): #upper limit: 0.95, down limit: 0.05
        df = data.copy()
        q95 = df['price_mile'].quantile(0.95)
        q05 = df['price_mile'].quantile(0.05)
        df['price_mile'] = np.where(df['price_mile'] > q95, q95, df['price_mile'])
        df['price_mile'] = np.where(df['price_mile'] < q05, q05, df['price_mile'])
        df = df.reset_index(drop=True)
        return df
    
    # def _winsoration(self, datax):
    #     column, new_column_name, lower_percentile, upper_percentile = 'price_mile', 'price_mile', 0.05, 0.95
    #     # Calcular los límites inferior y superior basados en los percentiles
    #     lower_limit = datax[column].quantile(lower_percentile)
    #     upper_limit = datax[column].quantile(upper_percentile)

    #     # Aplicar winsorización suavizada
    #     def smooth(value):
    #         if value < lower_limit:
    #             return lower_limit + (value - lower_limit) * 0.25  # Suavizar por interpolación
    #         elif value > upper_limit:
    #             return upper_limit + (value - upper_limit) * 0.25  # Suavizar por interpolación
    #         return value

    #     # Crear la nueva columna con los valores ajustados
    #     datax[new_column_name] = datax[column].apply(smooth)
        
    #     return datax