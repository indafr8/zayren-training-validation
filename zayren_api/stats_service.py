import pandas as pd

def city2city_info(route_c2c, distance, df_segments):
    query_data = df_segments[df_segments['c2c_route'] == route_c2c]
    count_segments = query_data.shape[0]  # Número de segmentos

    # Calcular el precio por milla
    price_mille = query_data['adjust_price_usd'] / query_data['distance']
    price_mille_median = price_mille.median()
    price_mille_median = round(price_mille_median, 2)

    # Calcular precio promedio
    mean_price = price_mille_median * distance
    mean_price = round(mean_price, 2)

    # Calcular rango intercuartílico (IQR)
    q1 = price_mille.quantile(0.25)  # Percentil 25
    q3 = price_mille.quantile(0.75)  # Percentil 75
    iqr = q3 - q1

    # Ajustar rango de precios mínimo y máximo usando IQR
    price_mille_ajusted_min = q1  # Precio mínimo ajustado
    range_min = distance * price_mille_ajusted_min
    range_min = round(range_min, 2)

    price_mille_ajusted_max = q3  # Precio máximo ajustado
    range_max = distance * price_mille_ajusted_max
    range_max = round(range_max, 2)


    # Ordenar los últimos segmentos por fecha y seleccionar las columnas relevantes
    last_segments = query_data.copy()
    last_segments = last_segments.sort_values(by='inserted_at', ascending=False)
    last_segments['distance'] = last_segments['distance']
    last_segments['price_mille'] = last_segments['adjust_price_usd'] / last_segments['distance']

    last_ten_segments = last_segments.head(10)

    # Calcular métricas finales para los últimos segmentos
    price_mille_ten = last_ten_segments['adjust_price_usd'].median() / distance
    price_mille_ten = round(price_mille_ten,2)
    
    last_trip = str(last_segments['inserted_at'].iloc[0])[0:10]

    last_price = last_ten_segments['adjust_price_usd'].iloc[0]
    mean_price_ten = last_ten_segments['adjust_price_usd'].median()

    #diccionario de metricas
    stats = {"mean_price": float(mean_price), 
            "range_min": float(range_min), 
            "range_max": float(range_max), 
            "count_segments": int(count_segments), 
            "price_mille_median": float(price_mille_median), 
            "last_trip": str(last_trip), 
            "last_price": float(last_price), 
            "mean_price_ten": float(mean_price_ten),
            "price_mille_ten": float(price_mille_ten)}
    
    # Retornar métricas calculadas
    return stats

def state2state_info(route_s2s, distance, df_segments):
    query_data = df_segments[df_segments['s2s_route'] == route_s2s]
    count_segments = query_data.shape[0]  # Número de segmentos

    # Calcular el precio por milla
    price_mille = query_data['adjust_price_usd'] / query_data['distance']
    price_mille_median = price_mille.median()
    price_mille_median = round(price_mille_median, 2)

    # Calcular precio promedio
    mean_price = price_mille_median * distance
    mean_price = round(mean_price, 2)

    # Calcular rango intercuartílico (IQR)
    q1 = price_mille.quantile(0.25)  # Percentil 25
    q3 = price_mille.quantile(0.75)  # Percentil 75
    iqr = q3 - q1

    # Ajustar rango de precios mínimo y máximo usando IQR
    price_mille_ajusted_min = q1  # Precio mínimo ajustado
    range_min = distance * price_mille_ajusted_min
    range_min = round(range_min, 2)

    price_mille_ajusted_max = q3  # Precio máximo ajustado
    range_max = distance * price_mille_ajusted_max
    range_max = round(range_max, 2)

    # Ordenar los últimos segmentos por fecha y seleccionar las columnas relevantes
    last_segments = query_data.copy()
    last_segments = last_segments.sort_values(by='inserted_at', ascending=False)
    last_segments['distance'] = last_segments['distance']
    last_segments['price_mille'] = last_segments['adjust_price_usd'] / last_segments['distance']

    last_ten_segments = last_segments.head(10)

    # Calcular métricas finales para los últimos segmentos
    price_mille_ten = last_ten_segments['adjust_price_usd'].median() / distance
    price_mille_ten = round(price_mille_ten,2)
    
    last_trip = str(last_segments['inserted_at'].iloc[0])[0:10]

    last_price = last_ten_segments['adjust_price_usd'].iloc[0]
    mean_price_ten = last_ten_segments['adjust_price_usd'].median()

    stats = {"mean_price": float(mean_price), 
            "range_min": float(range_min), 
            "range_max": float(range_max), 
            "count_segments": int(count_segments), 
            "price_mille_median": float(price_mille_median), 
            "last_trip": str(last_trip), 
            "last_price": float(last_price), 
            "mean_price_ten": float(mean_price_ten),
            "price_mille_ten": float(price_mille_ten)}
    
    return stats