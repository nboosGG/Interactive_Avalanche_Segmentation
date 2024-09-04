# -----------------------------------------------------------------------------
# Usage: python test_sendavalanche.py
# By: Gerd Weiss & Noel Boos
# -----------------------------------------------------------------------------
import requests
from requests_oauthlib import OAuth2Session
from oauthlib.oauth2 import DeviceClient

import webbrowser
from datetime import datetime, timedelta
import time

from shapely.geometry import mapping
import json
import numpy as np

import geopandas as gpd


class Export():


    def temp(poly_list):
        print("how to iterate over polygon")
        poly_list_adjusted = []
        for i in range(len(poly_list)):
            poly = poly_list[i]

            #print("representative point: ", poly.representative_point())
            #print(dir(poly))
            arr = np.array(poly.exterior.coords)
            print(np.shape(arr))
            #print("holes dir: ", type(poly.interiors), dir(poly.interiors))
            holes = []
            print(np.shape(holes))
            for interior in poly.interiors:
                hole = np.array(interior.coords)
                holes.append(hole)
            holes = np.array(holes)

            if len(poly.interiors) == 0:
                poly_list_adjusted.append(np.array([arr]))
            else:
                poly_list_adjusted.append(np.array([arr,holes])) 
        
        #return poly_list_adjusted

    def list_stripper(ll):
        ret_ll = []
        for l in ll:
            ret_ll.append(l[0])
        return ret_ll


    def combine_shp_list(pl, al):

        assert(len(pl) == len(al))

        ret_l = []
        for i in range(len(pl)):
            el = []
            pl_i = pl[i]
            al_i = al[i]
            el.append(pl_i[0])
            el = [*el, *al_i]
            el.append(pl_i[1])
            ret_l.append(el)
        return ret_l

            