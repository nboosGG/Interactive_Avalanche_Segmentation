import geopandas
import numpy as np

def add_colum(ddf, column_names, column_defaults):

    for i in range(len(column_names)):
        fieldname = column_names[i]
        default_val = column_defaults[i]
        if fieldname not in ddf.columns:
            ddf[fieldname] = None #init
            #ddf[fieldname] = gdf['field_name'].astype('float64')
            ddf[fieldname] = default_val
            #print("i, ", i, "name and def value: ", fieldname, default_val)
    return ddf
    


path = "/home/boosnoel/Downloads/20220319_Dorfberg_lv95/20220319_Dorfberg_lv95.shp"

dorfberg_avs = geopandas.read_file(path)

custom_path = "/home/boosnoel/Documents/exampleS/TEMP/albulapass_avalanches.shp"

albula_avs = geopandas.read_file(custom_path)
print("Dorfberg shp from database")
print(dorfberg_avs)
print("shape: ", list(dorfberg_avs.columns.values))
print(albula_avs)

albula_avs = add_colum(albula_avs, ["datim", "datim_a", "typ", "moisture", "trg_typ", "frac_wdh"], ["2022-03-18T12:00:00+01:00", "PM_12_H", "FULL_DEPTH", "WET", "NATURAL", "0.0"])
print(albula_avs)


