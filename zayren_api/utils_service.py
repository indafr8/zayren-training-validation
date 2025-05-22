import pandas as pd
import googlemaps
import os

def load_data(flag, broker_id):
    if flag == 0: # broker data
        df_segments = pd.read_csv('../api_local/segments_data.csv')
        df_carriers = pd.read_csv('../api_local/carriers_info.csv')
    else: #global data
        df_segments = pd.read_csv('segments_data.csv')
        df_carriers = pd.read_csv('carriers_info.csv')
    return df_segments, df_carriers

def query_type(route_c2c, route_s2s, df_segments):
    if route_s2s in df_segments['s2s_route'].values:
        #print('Existe State to State')
        if route_c2c in df_segments['c2c_route'].values:
            #print('Existe City to City')
            return 'city2city'
        else:
           # print('No existe City to City')
            return 'state2state'
    else:
        #print('No existe State to State')
        return 'pickup'

def get_distance(c2c_route, route_s2s, query_type, broker_id):
    if query_type == 'city2city':
        df_distances = pd.read_csv(f'../data/{broker_id}/distances.csv')
        distance = df_distances.loc[df_distances['c2c_route'] == c2c_route, 'distance'].values
        print(f"Distancia encontrada en el dataset: {distance[0]} millas.")
        return distance[0]
    else:
            df_distances = pd.read_csv('../data/API_distances_saved.csv')
            if c2c_route in df_distances['c2c_route'].values:
                distance = df_distances.loc[df_distances['c2c_route'] == c2c_route, 'distance'].values
                print(f"Distancia encontrada en el API_distances_saved: {distance[0]} millas.")
                return distance[0]
            else:
                print(f"La ruta '{c2c_route}' no se encontró en el dataset. Consultando API de Google Maps...")
                return get_distance_api(c2c_route, route_s2s)

def get_distance_api(route_c2c, route_s2s):
    API_KEY = 'AIzaSyBQ0cBrxzx6Gh7EYC1nBpBCQKvi4vC15dc'
    gmaps = googlemaps.Client(key=API_KEY)

    try:
        origenc, destinoc = route_c2c.split(' - ')
        origens, destinos = route_s2s.split(' - ')
        
        origen = f"{origens}, {origenc}"
        destino = f"{destinos}, {destinoc}"
        
        result = gmaps.distance_matrix(
            origins=origen,
            destinations=destino,
            mode="driving",
            units="imperial"  # Distancia en millas
        )

        distance_text = result['rows'][0]['elements'][0]['distance']['text']  # Ejemplo: "124 mi"
        distance_miles = float(distance_text.split(' ')[0].replace(',', ''))  # Convertir "124 mi" a 124.0
        print(f"Distancia obtenida de Google Maps API: {distance_miles} millas.")

        filename = '../data/API_distances_saved.csv'

        if os.path.exists(filename):
            df = pd.read_csv(filename)
        else:
            df = pd.DataFrame(columns=['c2c_route', 'distance'])

        # Agregar la nueva ruta y distancia
        new_entry = {'c2c_route': route_c2c, 'distance': distance_miles}
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)

        # Guardar el DataFrame actualizado en el archivo
        df.to_csv(filename, index=False)
        print(f"Ruta y distancia guardadas en {filename}.")

        return distance_miles
    
    except Exception as e:
        print(f"Error al obtener la distancia para la ruta '{route_c2c}' usando la API de Google Maps: {e}")
        return None

