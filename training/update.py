import pandas as pd
import numpy as np
import unicodedata
import re
import sqlite3
from typing import Dict, List, Optional, Union
from pathlib import Path

# Constantes para las regiones
REGIONES_MX: Dict[str, List[str]] = {
    "mx1": ["baja california", "baja california sur", "sonora", "sinaloa"],
    "mx2": ["chihuahua", "coahuila", "nuevo leon", "tamaulipas", "durango", "coahuila de zaragoza"],
    "mx3": ["nayarit", "jalisco", "colima", "michoacan", "guanajuato", "queretaro", "aguascalientes", "zacatecas", "san luis potosi"],
    "mx4": ["estado de mexico", "hidalgo", "morelos", "puebla", "tlaxcala", "ciudad de mexico", "mexico"],
    "mx5": ["guerrero", "oaxaca", "chiapas", "veracruz", "tabasco", "campeche", "yucatan", "quintana roo"]
}

REGIONES_US: Dict[str, List[str]] = {
    "us1": ["maine", "new hampshire", "vermont", "massachusetts", "rhode island", "connecticut", "new york", "new jersey", "pennsylvania"],
    "us2": ["ohio", "michigan", "indiana", "illinois", "wisconsin", "minnesota", "iowa", "missouri", "north dakota", "south dakota", "nebraska", "kansas"],
    "us3": ["delaware", "maryland", "virginia", "west virginia", "north carolina", "south carolina", "georgia", "florida", "kentucky", "tennessee", "alabama", "mississippi", "arkansas", "louisiana", "oklahoma", "texas"],
    "us4": ["montana", "idaho", "wyoming", "colorado", "new mexico", "arizona", "utah", "nevada", "washington", "oregon", "california", "alaska", "hawaii"]
}

class UpdateData:
    """Clase para actualizar y procesar datos de transporte."""
    
    def __init__(self, broker_id: str, train_data_folder: Union[str, Path]):
        """
        Inicializa la clase UpdateData.

        Args:
            broker_id: ID del broker
            train_data_folder: Ruta a la carpeta de datos de entrenamiento
        """
        self.broker_id = broker_id
        self.train_data_folder = Path(train_data_folder)
        self.folder = self.train_data_folder

        try:
            self.base_df = pd.read_csv(f'past_data/{self.broker_id}.csv')
            self.__update_data()
        except FileNotFoundError:
            print(f"El archivo past_data/{self.broker_id}.csv no existe")
            self._new_broker_data()
        except Exception as e:
            print(f"Error al inicializar UpdateData: {str(e)}")
            raise

    def __update_data(self):
        """
        Actualiza los datos cuando existe un archivo histórico.
        """
        try:
            df = self.__sql_query()
            df = self.__normalize_text(df)
            df = self.__zero_filter(df)
            df = self.__convert_currency(df)
            df = self.__inflation_adjustment(df)
            df = self.__add_s2s_column(df)
            df = self.__select_routes(df)
            df = self.__add_s2s_id_update(df)
            df = self.__normalize_cities(df)
            self.data = pd.concat([self.base_df, df], ignore_index=True)
        except Exception as e:
            print(f"Error actualizando datos: {str(e)}")
            raise

    def get_update_data(self) -> pd.DataFrame:
        """
        Obtiene los datos procesados.

        Returns:
            pd.DataFrame: DataFrame con los datos procesados
        """
        return self.data

    def __normalize_cities(self, df_new: pd.DataFrame) -> pd.DataFrame:
        """
        Normaliza las ciudades en las columnas de ubicación.
        """
        # (campo_afectado, campo_referencia, valor_viejo, valor_nuevo)
        correcciones = [
            ('state', 'state', 'mexico', 'estado de mexico'),
            ('state', 'state', 'baja california', 'baja california norte'),
            ('state', 'state', 'coahuila de zaragoza', 'coahuila'),
            
            ('state', 'city', 'ciudad de mexico', 'ciudad de mexico'),
            ('state', 'city', 'iztapalapa', 'ciudad de mexico'),
            
            ('city', 'city', 'ciudad juarez', 'juarez'),
            ('city', 'city', 'nava municipality', 'nava'),
            ('city', 'city', 'silao de la victoria', 'silao'),
            ('city', 'city', 'leon de los aldama', 'leon de los aldamas'),
            ('city', 'city', 'villa de tezontepec', 'tezontepec'),
            ('city', 'city', 'atotonilco de tula', 'atotonilco tula'),
            ('city', 'city', 'tepeji del rio de ocampo, hgo, mexico', 'tepeji de ocampo'),
            ('city', 'city', 'zapote del valle', 'zapote'),
            ('city', 'city', 'san pedro tlaquepaque', 'tlaquepaque'),
            ('city', 'city', 'robbinsville twp', 'robbinsville'),
            ('city', 'city', 'robbinsville township', 'robbinsville'),
            ('city', 'city', 'monroe township', 'monroe'),
            ('city', 'city', 'south brunswick township', 'south brunswick terrace'),
            ('city', 'city', 'north brunswick township', 'north brunswick'),
            ('city', 'city', 'hamilton township', 'hamilton'),
            ('city', 'city', 'pennsauken township', 'pennsauken'),
            ('city', 'city', 'sparta township', 'sparta'),
            ('city', 'city', 'ciudad general escobedo', 'escobedo'),
            ('city', 'city', 'ciudad apodaca', 'apodaca'),
            ('city', 'city', 'parque industrial ciudad mitras', 'mitras'),
            ('city', 'city', 'ciudad de allende', 'allende'),
            ('city', 'city', 'ciudad santa catarina', 'santa catarina'),
            ('city', 'city', 'san pedro garza garcia', 'san pedro'),
            ('city', 'city', 'heroica puebla de zaragoza', 'puebla de zaragoza'),
            ('city', 'city', 'ldo', 'laredo'),
            ('city', 'city', 'tlalnepantla de baz', 'tlalnepantla'),
            ('city', 'city', 'santa maria totoltepec', 'san'),
            ('city', 'city', 'jilotepec de molina enriquez', 'jilotepec'),
            ('city', 'city', 'san cristobal nexquipayac', 'nexquipayac'),
            ('city', 'city', 'tecamac de felipe villanueva', 'tecamac'),
            ('city', 'city', 'san francisco coacalco', 'coacalco'),
            ('city', 'city', 'los reyes acaquilpan', 'los reyes'),
            ('city', 'city', 'valle de chalco solidaridad', 'chalco'),
        ]
        
        df = df_new.copy()
        for campo_mod, campo_ref, valor_viejo, valor_nuevo in correcciones:
            df.loc[df["pickup_"+campo_ref] == valor_viejo, "pickup_"+campo_mod] = valor_nuevo
            df.loc[df["dropoff_"+campo_ref] == valor_viejo, "dropoff_"+campo_mod] = valor_nuevo

        return df

    def _new_broker_data(self):
        """
        Procesa datos nuevos cuando no hay datos históricos.
        """
        try:
            df = self.__sql_query()
            df = self.__normalize_text(df)
            df = self.__zero_filter(df)
            df = self.__convert_currency(df)
            df = self.__inflation_adjustment(df)
            df = self.__add_s2s_column(df)
            df = self.__select_routes(df)
            df = self.__add_s2s_id_new(df)
            df = self.__normalize_cities(df)
            self.data = df
        except Exception as e:
            print(f"Error procesando nuevos datos: {str(e)}")
            raise

    def __region_state(self, state: str) -> str:
        """
        Determina la región a la que pertenece un estado.

        Args:
            state: Nombre del estado

        Returns:
            str: Código de la región (mx1-mx5, us1-us4, o 'unknown')
        """
        state = state.lower()
        for region, states in REGIONES_MX.items():
            if state in states:
                return region
        for region, states in REGIONES_US.items():
            if state in states:
                return region
        return "unknown"

    def __add_validation_labels(self, df_new: pd.DataFrame) -> pd.DataFrame:
        """
        Agrega etiquetas de validación basadas en regiones.

        Args:
            df_new: DataFrame con los datos a procesar

        Returns:
            pd.DataFrame: DataFrame con las etiquetas de región agregadas
        """
        df = df_new.copy()
        
        # Vectorización de la asignación de regiones
        df['region_pickup'] = df['pickup_state'].apply(self.__region_state)
        df['region_dropoff'] = df['dropoff_state'].apply(self.__region_state)
        df['region_label'] = df['region_pickup'] + ' - ' + df['region_dropoff']
        
        return df

    def __sql_query(self) -> pd.DataFrame:
        """
        Ejecuta una consulta SQL compleja para obtener datos de segmentos y envíos.

        Returns:
            pd.DataFrame: DataFrame con los resultados de la consulta
        """
        conn = None
        try:
            # Cargar datos en chunks para optimizar memoria
            chunk_size = 10000
            dfs = {}
            
            for file_name in ['segment_bill_items', 'segment_bills', 'segments', 
                            'segments_lanes', 'shipments', 'shipments_details']:
                file_path = self.folder / f"{file_name}.csv"
                print(f"Intentando leer archivo: {file_path}")
                dfs[file_name] = pd.concat(
                    pd.read_csv(file_path, chunksize=chunk_size, low_memory=False)
                )

            # Filtrar segment_bill_items antes de cargarlo en SQLite
            dfs['segment_bill_items'] = dfs['segment_bill_items'][
                dfs['segment_bill_items']['type'] == "carriers_price"
            ]

            conn = sqlite3.connect(':memory:')
            
            # Cargar datos en SQLite
            for name, df in dfs.items():
                df.to_sql(name, conn, index=False, if_exists="replace")

            # Optimizar la consulta SQL
            query = """
            WITH filtered_segments AS (
                SELECT s.*, sl.*
                FROM segments s
                INNER JOIN segments_lanes sl ON s.id = sl.segment_id
                WHERE s.status = 'completed'
            ),
            bill_info AS (
                SELECT 
                    sbi.segment_bill_id,
                    sb.segment_id,
                    sbi.total AS base_carrier_price,
                    sbi.currency_code,
                    sbi.exchange_rate
                FROM segment_bill_items sbi
                INNER JOIN segment_bills sb ON sbi.segment_bill_id = sb.id
            ),
            shipment_info AS (
                SELECT 
                    s.id AS shipment_id,
                    s.inserted_at,
                    s.total_price,
                    sd.trailer_type
                FROM shipments s
                INNER JOIN shipments_details sd ON s.id = sd.shipment_id
            )
            SELECT 
                fs.*,
                bi.base_carrier_price,
                bi.currency_code,
                bi.exchange_rate,
                si.inserted_at,
                si.total_price,
                si.trailer_type
            FROM filtered_segments fs
            INNER JOIN bill_info bi ON fs.segment_id = bi.segment_id
            INNER JOIN shipment_info si ON fs.shipment_id = si.shipment_id;
            """

            df = pd.read_sql_query(query, conn)
            df = df.loc[:, ~df.columns.duplicated()]
            
            if 'shipment_status' in df.columns:
                df.drop('shipment_status', axis=1, inplace=True)

            return df

        except Exception as e:
            print(f"Error en la consulta SQL: {str(e)}")
            raise
        finally:
            if conn is not None:
                conn.close()

    def __write_carriers(self):
        df = pd.read_csv(f"{self.folder}carriers_info.csv")

        carriers = df[['carrier_id', 'name', 'status', 'phone_number', 'email']]
        carriers.to_csv(f'output_data/{self.broker_id}/carriers_info.csv', index=False)

    def __normalize_text(self, df_new: pd.DataFrame) -> pd.DataFrame:
        """
        Normaliza el texto en las columnas de ubicación.

        Args:
            df_new: DataFrame con los datos a normalizar

        Returns:
            pd.DataFrame: DataFrame con el texto normalizado
        """
        df_text = df_new.copy()
        columns = ['pickup_country', 'pickup_state', 'pickup_city', 
                'dropoff_country', 'dropoff_state', 'dropoff_city']
        
        # Diccionario de correcciones
        corrections = {
            "state of mexico": "estado de mexico",
            "mexico city": "ciudad de mexico",
            "in": "indiana",
            "fl": "florida",
            "il": "illinois",
            "tn": "tennessee",
            "mo": "missouri",
            "tx": "texas",
            "nj": "new jersey"
        }

        def normalize_string(s: str) -> str:
            if not isinstance(s, str):
                return s
            s = unicodedata.normalize('NFKD', s).encode('ascii', errors='ignore').decode('utf-8')
            s = s.lower()
            for wrong, correct in corrections.items():
                s = re.sub(rf'\b{re.escape(wrong)}\b', correct, s)
            return s

        # Aplicar la normalización a las columnas relevantes
        for col in columns:
            df_text[col] = df_text[col].apply(normalize_string)

        return df_text
        
    def __zero_filter(self, df_new: pd.DataFrame) -> pd.DataFrame:
        """
        Filtra registros con valores cero o negativos en precios y distancias.

        Args:
            df_new: DataFrame a filtrar

        Returns:
            pd.DataFrame: DataFrame filtrado
        """
        return df_new[
            (df_new['base_carrier_price'] > 0) & 
            (df_new['distance'] > 0)
        ]
    
    def __convert_currency(self, df_new: pd.DataFrame) -> pd.DataFrame:
        """
        Convierte los precios a USD.

        Args:
            df_new: DataFrame con los precios a convertir

        Returns:
            pd.DataFrame: DataFrame con los precios en USD
        """
        df = df_new.copy()
        df['price_usd'] = np.where(
            df['currency_code'] != 'usd',
            df['base_carrier_price'] / df['exchange_rate'],
            df['base_carrier_price']
        )
        return df
    
    def __inflation_adjustment(self, df_new: pd.DataFrame) -> pd.DataFrame:
        """
        Ajusta los precios por inflación según el año.

        Args:
            df_new: DataFrame con los precios a ajustar

        Returns:
            pd.DataFrame: DataFrame con los precios ajustados
        """
        df = df_new.copy()
        df['inserted_at'] = pd.to_datetime(df['inserted_at'])
        df['year'] = df['inserted_at'].dt.year
        df['month'] = df['inserted_at'].dt.month
        
        # Factores de ajuste por año
        inflation_factors = {
            2020: 1.2547,
            2023: 1.0488
        }
        
        df['adjust_price_usd'] = df['price_usd']
        for year, factor in inflation_factors.items():
            mask = df['year'] == year
            df.loc[mask, 'adjust_price_usd'] *= factor
        
        return df

    def __add_s2s_column(self, df_new: pd.DataFrame) -> pd.DataFrame:
        """
        Agrega columnas de rutas estado a estado y ciudad a ciudad.

        Args:
            df_new: DataFrame al que agregar las rutas

        Returns:
            pd.DataFrame: DataFrame con las rutas agregadas
        """
        df = df_new.copy()
        df['s2s_route'] = df['pickup_state'] + ' - ' + df['dropoff_state']
        df['c2c_route'] = df['pickup_city'] + ' - ' + df['dropoff_city']
        return df
    
    def __select_routes(self, df_new: pd.DataFrame, min_routes: int = 20) -> pd.DataFrame:
        """
        Filtra las rutas que tienen un mínimo de registros.

        Args:
            df_new: DataFrame a filtrar
            min_routes: Número mínimo de registros por ruta

        Returns:
            pd.DataFrame: DataFrame con las rutas filtradas
        """
        return df_new.groupby('s2s_route').filter(lambda x: len(x) >= min_routes)
    
    def __add_s2s_id_new(self, df_new: pd.DataFrame) -> pd.DataFrame:
        """
        Agrega IDs únicos a las rutas estado a estado.

        Args:
            df_new: DataFrame al que agregar los IDs

        Returns:
            pd.DataFrame: DataFrame con los IDs de ruta agregados
        """
        df = df_new.copy()
        s2s_routes = df['s2s_route'].unique()
        s2s_dict = {s2s: idx for idx, s2s in enumerate(s2s_routes)}
        df['s2s_id'] = df['s2s_route'].map(s2s_dict)
        return df
    
    def __add_s2s_id_update(self, df_new: pd.DataFrame) -> pd.DataFrame:
        """
        Actualiza los IDs de ruta manteniendo los existentes y agregando nuevos.

        Args:
            df_new: DataFrame con las nuevas rutas

        Returns:
            pd.DataFrame: DataFrame con los IDs de ruta actualizados
        """
        df = self.base_df.copy()
        new_df = df_new.copy()
        new_df['s2s_id'] = np.nan

        # Crear diccionario de rutas existentes
        s2s_dict = dict(zip(
            df.drop_duplicates(subset=['s2s_route'])['s2s_route'],
            df.drop_duplicates(subset=['s2s_route'])['s2s_id']
        ))
        
        # Asignar IDs existentes
        new_df['s2s_id'] = new_df['s2s_route'].map(s2s_dict)
        
        # Asignar nuevos IDs a rutas sin ID
        max_id = df['s2s_id'].max()
        new_routes = new_df['s2s_route'][new_df['s2s_id'].isna()].unique()
        
        for i, route in enumerate(new_routes, start=1):
            new_id = max_id + i
            s2s_dict[route] = new_id
            new_df.loc[new_df['s2s_route'] == route, 's2s_id'] = new_id

        return new_df

