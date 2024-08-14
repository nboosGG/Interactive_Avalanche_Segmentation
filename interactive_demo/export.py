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

import numpy as np

class Export():


    def convert_poly(poly_list):
        poly_list_adjusted = []
        for i in range(len(poly_list)):
            poly = poly_list[i]

            arr = np.array(poly.exterior.coords)
            holes = []
            print(np.shape(holes))
            for interior in poly.interiors:
                hole = np.array(interior.coords[:])
                holes.append(hole)
            holes = np.array(holes)

            if len(poly.interiors) == 0:
                poly_list_adjusted.append(np.array([arr]))
            else:
                poly_list_adjusted.append(np.array([arr,holes])) 
        
        return poly_list_adjusted

    #poly_list: [polygon1, polygon2, ...]
    # attribute_list: [[time, aval_type, aval_size, snow_moisture, release_type], [...], ...]
    def export_DB(poly_list, attribute_list):
        print("attribute list: ", attribute_list)

        print("poly list length: ", len(poly_list), "attribute list length: ", len(attribute_list))

        if len(poly_list) != len(attribute_list):
            print("export failed, polylist and attributelist not same length")
            return

        # -----------------------------------------------------------------------------
        # Configuration
        # -----------------------------------------------------------------------------
        issuer = 'https://sso-t.slf.ch/auth/realms/SLF'
        client_id = 'ext_easymapper'
        proaval_url = 'https://pro-d.slf.ch/aval'


        # 1) start authorization request
        # -----------------------------------------------------------------------------
        issuer_divice = issuer + '/protocol/openid-connect/auth/device'

        auth_body = {
            'grant_type': DeviceClient.grant_type,
            'client_id' : client_id,
            'scope' : 'openid'
        }

        auth_response = requests.post(issuer_divice, data=auth_body)
        auth_response_json = auth_response.json()
        divice_code = auth_response_json['device_code']
        verification_uri_complete = auth_response_json['verification_uri_complete']
        expires_in = auth_response_json['expires_in']


        # 2) User authentication
        # -----------------------------------------------------------------------------
        # Contains two steps
        # a) The user must log in via the browser and authorize access to their roles.
        # b) We poll for the token. 
        #    The request is successful after the user has logged in.
        # -----------------------------------------------------------------------------

        # 2a) open webbrowser for user authentication
        # -----------------------------------------------------------------------------
        print('open browser on ' + verification_uri_complete)
        webbrowser.open(verification_uri_complete)
        print('browser open')

        # 2b) poll for token
        # -----------------------------------------------------------------------------
        issuer_token_endpoint = issuer + '/protocol/openid-connect/token'
        token_request_body = {
        'grant_type': DeviceClient.grant_type,
        'client_id' : client_id,
        'device_code' : divice_code
        }

        expires_at = datetime.now() + timedelta(seconds=expires_in)
        token = None

        print('start token polling')
        while token == None:
            # wait always 5 seconds between token requests
            time.sleep(5)

            if expires_at < datetime.now():
                raise Exception('Login failed, device code becomes invalide')

            print('poll for token')
            token_response = requests.post(issuer_token_endpoint, data=token_request_body)

            if token_response.status_code == requests.codes.ok:
                token = token_response.json()

        print('token received!!')


        # 3) Now we have the token and we can start sending avalanches
        # -----------------------------------------------------------------------------
        # We use OAuth2Session to add the bearer token authentication header and to 
        # refresh the token if required
        # -----------------------------------------------------------------------------
        proaval_url_v4 = proaval_url + '/v4/avalanches'

        refresh_url = issuer + '/protocol/openid-connect/token'

        # This function must be defined, but is not necessary for the token update
        def token_updater(token) :
            print('token updated')

        oauth = OAuth2Session(client_id, 
                            token=token,
                            auto_refresh_url=refresh_url,
                            auto_refresh_kwargs={'client_id' : client_id},
                            client=DeviceClient(client_id),
                            token_updater= token_updater)
        
        poly_list_converted = Export.convert_poly(poly_list)
        
        for i in range(len(poly_list_converted)):
            poly = poly_list_converted[i]
            #print("polygon: ", type(poly))
            #print(poly)
            attributes = attribute_list[i]
            #print("curr attributes: ", attributes)
            #print(dir(poly))

            ar = oauth.post(proaval_url_v4, json = {
            'location': {
                'type': 'Point',
                'coordinates': [7.44162646, 46.91366545]
            },
            'triggerDateTime': attribute_list[0], #datetime.now().astimezone().isoformat(),
            'avalancheType': attributes[1],
            'avalancheSize': attributes[2],
            'avalancheMoisture': attributes[3],
            'triggerType': attributes[4],
            'zones': [
            {
                'type': 'AVAL_SHAPE',
                'geo': {
                'type': 'Polygon',
                #'coordinates': [[[7.44125503, 46.91346396],[7.44172585, 46.91371936], [7.44178468, 46.91331609], [7.44094113, 46.91225414], [7.44098034, 46.91135349], [7.43933258, 46.91134006], [7.43890101, 46.91206595], [7.43994069,46.91237513], [7.44080383, 46.91298004], [7.44125503, 46.91346396]]]
                'coordinates': poly.tolist()
                }
            }
            ]
            },
            headers= {
            'ch.slf.pro.system': 'AlpRS_drone_upload',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
            }
            )

  
            if ar.status_code == requests.codes.created:
                print ('avalanche ' + ar.json()['id'] + ' created')
            else:
                print ('avalanche creation failed with status ' + str(ar.status_code) + ': ' + ar.text)            

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
        

    
            