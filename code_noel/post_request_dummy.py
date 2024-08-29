import requests
import datetime


url = 'https://pro-d.slf.ch/aval/v4pro/avalanches'
myobj = {
  "location": {
    "type": "Point",
    "coordinates": [
      9.84738,
      46.81222
    ]
  },
  "locationCanton": "BE / VS",
  "locationMunicipality": "Sils im Engadin/Segl",
  "triggerDateTime": "2020-01-06T08:08:08+01:00",
  "triggerDateTimeAcc": "ACCURATE",
  "triggerType": "UNKNOWN",
  "triggerTypeOtherDetail": "Steinbock",
  "triggerTypePersons": [
    "SKI",
    "MOUNTAINEER"
  ],
  "triggerTypePersonOther": "Gleitschirmflieger",
  "avalancheType": "FULL_DEPTH",
  "avalancheSize": "UNKNOWN",
  "avalancheMoisture": "UNKNOWN",
  "fractureThicknessMean": 255,
  "fractureWidth": 21.4,
  "fractureWidthAcc": "PM_500_M",
  "startZoneAspect": "N",
  "startZoneSlopeAngle": "30_35",
  "startZoneTerrain": [
    "WIDE_OPEN_SLOPE",
    "AREA_ADJACENT_TO_THE_RIDGELINE"
  ],
  "startZoneElevation": "2400",
  "startZoneElevationAcc": "ACCURATE",
  "depositHeight": 3.5,
  "depositWidth": 35.5,
  "weakLayer": "IN_OLD_SNOW",
  "note": "Eine schöne Lawine",
  "confidential": "true",
  "zones": [
    {
      "version": 1,
      "type": "START_ZONE_POINT",
      "geo": {
        "type": "Point",
        "coordinates": [
          9.84738,
          46.81222
        ]
      },
      "acc": "ACCURATE"
    }
  ]
}
print("type: ", type(myobj))

dd = datetime.datetime.now()

print("type: ", type(dd))



#x = requests.post(url, json = myobj, headers=(ch.slf.pro.system=IAS_mapper))
#x = requests.post(url, json = myobj)

#print(x.text)


