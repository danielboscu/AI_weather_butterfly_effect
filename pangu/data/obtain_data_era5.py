import os
import earthkit.data as ekd
import numpy as np

# Root for input ICs (orig_data/), run output (data/).
pangu_scratch = os.environ["pangu_scratch"]

# ERA5-G-pl
d1_pl = ekd.from_source(
    "cds",
    "reanalysis-era5-complete",
    request={
        "class": "ea",
        "date": "2021-06-26",
        "expver": "1",
        "levelist": "50/100/150/200/250/300/400/500/600/700/850/925/1000",
        "levtype": "pl",
        "param": "129.128/130.128/131/132/133.128",
        "stream": "oper",
        "time": "00:00:00",
        "type": "an",
        "grid": "0.25/0.25",
        "format": "netcdf",
    },
)

d1_pl = d1_pl.to_xarray()
atmos_vars = [ 'z','q','t','u','v']
d1_pl = d1_pl.sortby("pressure_level", ascending=False)
d1_pl = d1_pl[atmos_vars]
np.save(f"{pangu_scratch}/orig_data/era5_g/era5_g_an_pl.npy", d1_pl.to_array().values)

d1_sfc = ekd.from_source(
    "cds",
    "reanalysis-era5-complete",
    request={
        "class": "ea",
        "date": "2021-06-26",
        "expver": "1",
        "levtype": "sfc",
        "param": "151.128/165.128/166.128/167.128",
        "stream": "oper",
        "time": "00:00:00",
        "type": "an",
        "grid": "0.25/0.25",
        "format": "netcdf",
    }
)
d1_sfc = d1_sfc.to_xarray()
np.save(f"{pangu_scratch}/orig_data/era5_g/era5_g_an_sfc.npy", d1_sfc.to_array().values)

# ERA5-EDA-an-pl
d2_an_pl = ekd.from_source(
    "cds",
    "reanalysis-era5-complete",
    request={
        "class": "ea",
        "date": "2021-06-26",
        "expver": "1",
        "levelist": "50/100/150/200/250/300/400/500/600/700/850/925/1000",
        "levtype": "pl",
        "number": "0/1/2/3/4/5/6/7/8/9",
        "param": "129.128/130.128/131/132/133.128",
        "stream": "enda",
        "time": "00:00:00",
        "type": "an",
        "grid": "0.25/0.25",
        "format": "netcdf",
    },
)
d2_an_pl = d2_an_pl.to_xarray()
atmos_vars = [ 'z','q','t','u','v']
d2_an_pl = d2_an_pl.sortby("pressure_level", ascending=False)
d2_an_pl = d2_an_pl[atmos_vars]
np.save(f"{pangu_scratch}/orig_data/era5_eda/era5_eda_an_pl.npy", d2_an_pl.to_array().values)

d2_an_sfc = ekd.from_source(
    "cds",
    "reanalysis-era5-complete",
    request={
        "class": "ea",
        "date": "2021-06-26",
        "expver": "1",
        "levtype": "sfc",
        "number": "0/1/2/3/4/5/6/7/8/9",
        "param": "151.128/165.128/166.128/167.128",
        "stream": "enda",
        "time": "00:00:00",
        "type": "an",
        "grid": "0.25/0.25",
        "format": "netcdf",
    }
)
d2_an_sfc = d2_an_sfc.to_xarray()
np.save(f"{pangu_scratch}/orig_data/era5_eda/era5_eda_an_sfc.npy", d2_an_sfc.to_array().values)
