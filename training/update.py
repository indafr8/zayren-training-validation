import pandas as pd
import numpy as np
import unicodedata
import re
import sqlite3

class UpdateData():
    def __init__(self, broker_id):
        self.folder = "../data/0/DB_0005__2025-03-07/"
        self.broker_id = broker_id
        try:
            self.base_df = pd.read_csv(f'past_data/{self.broker_id}.csv')
            self._update_data()
        except FileNotFoundError:
            print(f"El archivo past_data/{self.broker_id}.csv no existe")
            self._new_broker_data()
    
    def get_update_data(self):
        self.new_data['s2s_id'] = self.new_data['s2s_id'].astype(int)
        return self.new_data
    
    def _update_data(self):
        data = self._sql_query()
        data = self._zero_filter(data)
        data = self._convert_currency(data)
        data = self._inflation_adjustment(data)
        data = self._normalize_text(data)
        data = self._add_s2s_column(data)
        data = self._select_routes(data)
        data = self._add_s2s_id_update(data)
        self.new_data = self._add_validation_labels(data)
        self.new_data.to_csv(f'past_data/base_data_{self.broker_id}.csv', index=False)
        
        self._write_carriers()

    def _new_broker_data(self):
        data = self._sql_query()
        data = self._zero_filter(data)
        data = self._convert_currency(data)
        data = self._inflation_adjustment(data)
        data = self._normalize_text(data)
        data = self._add_s2s_column(data)
        data = self._select_routes(data)
        data = self._add_s2s_id_new(data)
        self.new_data = self._add_validation_labels(data)
        self.new_data.to_csv(f'past_data/base_data_{self.broker_id}.csv', index=False)
        self._write_carriers()

    def _add_validation_labels(self, df_new):
        df = df_new.copy()

        # Diccionario de las 5 regiones de México
        regiones_mx = {
            "mx1": ["baja california", "baja california sur", "sonora", "sinaloa"],
            "mx2": ["chihuahua", "coahuila", "nuevo leon", "tamaulipas", "durango", "coahuila de zaragoza"],
            "mx3": ["nayarit", "jalisco", "colima", "michoacan", "guanajuato", "queretaro", "aguascalientes", "zacatecas", "san luis potosi"],
            "mx4": ["estado de mexico", "hidalgo", "morelos", "puebla", "tlaxcala", "ciudad de mexico", "mexico"],
            "mx5": ["guerrero", "oaxaca", "chiapas", "veracruz", "tabasco", "campeche", "yucatan", "quintana roo"]
        }

        # Diccionario de las 4 regiones de Estados Unidos
        regiones_us = {
            "us1": [  # Northeast (9 estados)
                "maine",
                "new hampshire",
                "vermont",
                "massachusetts",
                "rhode island",
                "connecticut",
                "new york",
                "new jersey",
                "pennsylvania"
            ],
            "us2": [  # Midwest (12 estados)
                "ohio",
                "michigan",
                "indiana",
                "illinois",
                "wisconsin",
                "minnesota",
                "iowa",
                "missouri",
                "north dakota",
                "south dakota",
                "nebraska",
                "kansas"
            ],
            "us3": [  # South (16 estados)
                "delaware",
                "maryland",
                "virginia",
                "west virginia",
                "north carolina",
                "south carolina",
                "georgia",
                "florida",
                "kentucky",
                "tennessee",
                "alabama",
                "mississippi",
                "arkansas",
                "louisiana",
                "oklahoma",
                "texas"
            ],
            "us4": [  # West (13 estados)
                "montana",
                "idaho",
                "wyoming",
                "colorado",
                "new mexico",
                "arizona",
                "utah",
                "nevada",
                "washington",
                "oregon",
                "california",
                "alaska",
                "hawaii"
            ]
        }

        # Función para asignar la región correspondiente al pickup_state en una nueva columna
        def region_state(state, region_dict_mx, region_dict_us):
            for region, states in region_dict_mx.items():
                if state in states:
                    return region
            for region, states in region_dict_us.items():
                if state in states:
                    return region
            return "unknown"
        
        # Aplicar la función a las columnas 'pickup_state' y 'dropoff_state'
        df['region_pickup'] = df['pickup_state'].apply(region_state, args=(regiones_mx, regiones_us))
        df['region_dropoff'] = df['dropoff_state'].apply(region_state, args=(regiones_mx, regiones_us))

        df['region_label'] = df['region_pickup'] + ' - ' + df['region_dropoff']
        return df
    
    def _write_carriers(self):
        df = pd.read_csv(f"{self.folder}carriers_info.csv")

        carriers = df[['carrier_id', 'name', 'status', 'phone_number', 'email']]
        carriers.to_csv(f'output_data/{self.broker_id}/carriers_info.csv', index=False)

    def _sql_query(self):
        segment_bill_items = pd.read_csv(f"{self.folder}segment_bill_items.csv")
        segment_bill_items = segment_bill_items[segment_bill_items['type'] == "carriers_price"]

        segment_bills = pd.read_csv(f"{self.folder}/segment_bills.csv")
        segments = pd.read_csv(f"{self.folder}/segments.csv")
        segments_lanes = pd.read_csv(f"{self.folder}/segments_lanes.csv")
        shipments = pd.read_csv(f"{self.folder}/shipments.csv")
        shipments_details = pd.read_csv(f"{self.folder}/shipments_details.csv")

        conn = sqlite3.connect(':memory:')
        
        segment_bill_items.to_sql("segment_bill_items", conn, index=False, if_exists="replace")
        segment_bills.to_sql("segment_bills", conn, index=False, if_exists="replace")
        segments.to_sql("segments", conn, index=False, if_exists="replace")
        segments_lanes.to_sql("segments_lanes", conn, index=False, if_exists="replace")
        shipments.to_sql("shipments", conn, index=False, if_exists="replace")
        shipments_details.to_sql("shipments_details", conn, index=False, if_exists="replace")

        query = """
        WITH seg_filtered AS (
            SELECT *
            FROM segments
            WHERE id IN (
                SELECT segment_id 
                FROM segments_lanes
            )
        ),
        seg_bills_filtered AS (
            SELECT *
            FROM segment_bills
            WHERE segment_id IN (
                SELECT id 
                FROM seg_filtered
            )
        ),
        seg_bill_items_filtered AS (
            SELECT *
            FROM segment_bill_items
            WHERE segment_bill_id IN (
                SELECT id 
                FROM seg_bills_filtered
            )
        ),
        seg_lanes_filtered AS (
            SELECT *
            FROM segments_lanes
            WHERE segment_id IN (
                SELECT id 
                FROM seg_filtered
            )
        ),
        seg_interest AS (
            SELECT
                seg_lanes_filtered.segment_id,
                seg_filtered.shipment_id,
                seg_filtered.carrier_id,
                seg_filtered.status,
                seg_filtered.segment_type,
                seg_lanes_filtered.pickup_country,
                seg_lanes_filtered.pickup_state,
                seg_lanes_filtered.pickup_city,
                seg_lanes_filtered.dropoff_country,
                seg_lanes_filtered.dropoff_state,
                seg_lanes_filtered.dropoff_city,
                seg_lanes_filtered.distance
            FROM seg_filtered
            JOIN seg_lanes_filtered 
                ON seg_filtered.id = seg_lanes_filtered.segment_id
        ),
        seg_bill_interest AS (
            SELECT
                seg_bill_items_filtered.segment_bill_id,
                seg_bills_filtered.segment_id,
                seg_bill_items_filtered.name,
                seg_bill_items_filtered.total AS base_carrier_price,
                seg_bill_items_filtered.currency_code,
                seg_bill_items_filtered.exchange_rate
            FROM seg_bill_items_filtered
            JOIN seg_bills_filtered 
                ON seg_bill_items_filtered.segment_bill_id = seg_bills_filtered.id
        ),
        shipments_combined AS (
            SELECT
                shipments.id AS shipment_id,
                shipments.status AS shipment_status,
                shipments.inserted_at,
                shipments.total_price,
                shipments_details.trailer_type
            FROM shipments
            JOIN shipments_details 
                ON shipments.id = shipments_details.shipment_id
        )
        SELECT
            seg_interest.*,
            seg_bill_interest.base_carrier_price,
            seg_bill_interest.currency_code,
            seg_bill_interest.exchange_rate,
            shipments_combined.*  -- Usamos * para traer todas las columnas de esta CTE
        FROM seg_interest
        JOIN seg_bill_interest 
            ON seg_interest.segment_id = seg_bill_interest.segment_id
        JOIN shipments_combined 
            ON seg_interest.shipment_id = shipments_combined.shipment_id
        WHERE seg_interest.status = 'completed';
        """

        df = pd.read_sql_query(query, conn)
        # quitamos las columnas duplicadas
        df = df.loc[:, ~df.columns.duplicated()]

        # quitamos la columna 'shipment_status' porque ya tenemos 'status'
        df.drop('shipment_status', axis=1, inplace=True)

        conn.close()

        return df
    
    def _normalize_text(self, df_new):
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

        def normalize_string(s):
            if isinstance(s, str):
                # Normalizar caracteres y convertir a minúsculas
                s = unicodedata.normalize('NFKD', s).encode('ascii', errors='ignore').decode('utf-8')
                s = s.lower()
                
                # Sustituir palabras completas evitando reemplazos parciales
                for wrong, correct in corrections.items():
                    s = re.sub(rf'\b{re.escape(wrong)}\b', correct, s)  # Solo reemplaza palabras completas
                return s
            return s  # Si no es string, lo deja igual

        # Aplicar la normalización a las columnas relevantes
        for col in columns:
            df_text[col] = df_text[col].apply(normalize_string)

        return df_text
        
    def _zero_filter(self, df_new):
        df_zero = df_new.copy()
        df_zero = df_zero[df_zero['base_carrier_price'] > 0]
        df_zero = df_zero[df_zero['distance'] > 0]
        return df_zero
    
    def _convert_currency(self, df_new):
        df = df_new.copy()
        df['price_usd'] = np.where(df['currency_code'] != 'usd', df['base_carrier_price'] / df['exchange_rate'], df['base_carrier_price'])
        return df
    
    def _inflation_adjustment(self, df_new):
        df = df_new.copy()
        df['inserted_at'] = pd.to_datetime(df['inserted_at'])
        df['year'] = df['inserted_at'].dt.year
        df['month'] = df['inserted_at'].dt.month
        
        df_2020 = df[df['year'] == 2020]
        df_2023 = df[df['year'] == 2023]
        df_others = df[~df['year'].isin([2020, 2023])]

        df_2020['adjust_price_usd'] = df_2020['price_usd'] * 1.2547
        df_2023['adjust_price_usd'] = df_2023['price_usd'] * 1.0488
        df_others['adjust_price_usd'] = df_others['price_usd']

        df = pd.concat([df_2020, df_2023, df_others])
        
        return df

    def _add_s2s_column(self, df_new):
        df = df_new.copy()
        df['s2s_route'] = df['pickup_state'] + ' - ' + df['dropoff_state']
        df['c2c_route'] = df['pickup_city'] + ' - ' + df['dropoff_city']
        return df
    
    def _select_routes(self, df_new):
        df = df_new.copy()
        df = df.groupby('s2s_route').filter(lambda x: len(x) >= 20)
        return df
    
    def _add_s2s_id_new(self, df_new):
        # Copiamos el DataFrame entrante para no modificarlo directamente
        df = df_new.copy()

        # obtenemos las rutas unicas 2s2_route
        s2s_routes = df['s2s_route'].unique()

        # creamos un diccionario con las rutas unicas y un id
        s2s_dict = {s2s: idx for idx, s2s in enumerate(s2s_routes)}

        # creamos una columna target con el id de la ruta
        df['s2s_id'] = df['s2s_route'].map(s2s_dict)

        return df
    
    def _add_s2s_id_update(self, df_new):
        df = self.base_df.copy()
        new_df = df_new.copy()
        new_df['s2s_id'] = np.nan  # Inicializar con NaN

        # Obtener valores únicos de 's2s_route' en el dataset base
        df_unique = df.drop_duplicates(subset=['s2s_route'])

        # Diccionario con 's2s_route' como clave y 's2s_id' como valor
        s2s_dict = dict(zip(df_unique['s2s_route'], df_unique['s2s_id']))
        #impirmir numero de nan
        
        # Asignar valores existentes
        new_df['s2s_id'] = new_df['s2s_route'].map(s2s_dict)
        
        # Asignar nuevos ID a las rutas que no tienen target
        max_id = df['s2s_id'].max()  # Obtener el ID máximo existente
        new_routes = new_df['s2s_route'][new_df['s2s_id'].isna()].unique()

        for i, route in enumerate(new_routes, start=1):
            s2s_dict[route] = max_id + i  # Asignar nuevo ID
            new_df.loc[new_df['s2s_route'] == route, 's2s_id'] = max_id + i

        return new_df  # Retornar el DataFrame actualizado con los nuevos targets