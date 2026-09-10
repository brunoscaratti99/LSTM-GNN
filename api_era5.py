import cdsapi

dataset = "reanalysis-era5-pressure-levels"


for year in [1996,1995,1994,1993,1992,1991,1990,1989,1988,1987]:
    print("Downloading "+str(year)+"...")
    request = {
        "product_type": ["reanalysis"],
        "variable": [
            "u_component_of_wind",
            "v_component_of_wind"
        ],
        "year": [str(year)],
        "month": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12"
        ],
        "day": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12",
            "13", "14", "15",
            "16", "17", "18",
            "19", "20", "21",
            "22", "23", "24",
            "25", "26", "27",
            "28", "29", "30",
            "31"
        ],
        "time": [
            "00:00", "01:00", "02:00",
            "03:00", "04:00", "05:00",
            "06:00", "07:00", "08:00",
            "09:00", "10:00", "11:00",
            "12:00", "13:00", "14:00",
            "15:00", "16:00", "17:00",
            "18:00", "19:00", "20:00",
            "21:00", "22:00", "23:00"
        ],
        "pressure_level": ["500", "850"],
        "data_format": "grib",
        "download_format": "zip",
        "area": [-25, -60, -35, -45]
    }

    client = cdsapi.Client()
    client.retrieve(dataset, request).download()
