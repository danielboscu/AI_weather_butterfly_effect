import os
import earthkit.data as ekd
import numpy as np

# Root for input ICs (orig_data/), run output (data/).
pangu_scratch = os.environ["pangu_scratch"]
# Oper-G-pl
d1_pl = ekd.from_source(
    "mars",
    request={
        "class": "od",
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
d1_pl = d1_pl.sortby("level", ascending=False)
d1_pl = d1_pl[atmos_vars]
d1_pl = d1_pl.squeeze("time", drop=True)
np.save(f"{pangu_scratch}/orig_data/oper_g/oper_g_pl.npy", d1_pl.to_array().values.astype(np.float32))

d1_sfc = ekd.from_source(
    "mars",
    request={
        "class": "od",
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
d1_sfc = d1_sfc.squeeze("time", drop=True)
np.save(f"{pangu_scratch}/orig_data/oper_g/oper_g_sfc.npy", d1_sfc.to_array().values.astype(np.float32))

# Oper-EDA-an-pl
# This uses an offset of 9 hours to make the analysis time 2021-06-26 00:00 UTC, which is the same as the other datasets.
# May be physically different ERA5
d2_an_pl = ekd.from_source(
    "mars",
    request={
        "anoffset": 9,
        "class": "od",
        "date": "2021-06-26",
        "expver": 1,
        "levelist": "50/100/150/200/250/300/400/500/600/700/850/925/1000",
        "levtype": "pl",
        "number": "0/1/2/3/4/5/6/7/8/9/10/11/12/13/14/15/16/17/18/19/20/21/22/23/24/25/26/27/28/29/30/31/32/33/34/35/36/37/38/39/40/41/42/43/44/45/46/47/48/49/50",
        "param": "129.128/130.128/131/132/133.128",
        "stream": "elda",
        "time": "00:00:00",
        "type": "an",
        "grid": "0.25/0.25",
        "format": "netcdf"
    },
)
d2_an_pl = d2_an_pl.to_xarray()
atmos_vars = [ 'z','q','t','u','v']
d2_an_pl = d2_an_pl.sortby("level", ascending=False)
d2_an_pl = d2_an_pl[atmos_vars]
d2_an_pl = d2_an_pl.squeeze("time", drop=True)
np.save(f"{pangu_scratch}/orig_data/oper_eda/oper_eda_an_pl.npy", d2_an_pl.to_array().values.astype(np.float32))

d2_an_sfc = ekd.from_source(
    "mars",
    request={
        "anoffset": 9,
        "class": "od",
        "date": "2021-06-26",
        "expver": 1,
        "levtype": "sfc",
        "number": "0/1/2/3/4/5/6/7/8/9/10/11/12/13/14/15/16/17/18/19/20/21/22/23/24/25/26/27/28/29/30/31/32/33/34/35/36/37/38/39/40/41/42/43/44/45/46/47/48/49/50",
        "param": "151.128/165.128/166.128/167.128",
        "stream": "elda",
        "time": "00:00:00",
        "type": "an",
        "grid": "0.25/0.25",
        "format": "netcdf"
    }
)
d2_an_sfc = d2_an_sfc.to_xarray()
d2_an_sfc = d2_an_sfc.squeeze("time", drop=True)
np.save(f"{pangu_scratch}/orig_data/oper_eda/oper_eda_an_sfc.npy", d2_an_sfc.to_array().values.astype(np.float32))

# # Oper-EDA-v4-pl

# d2_v4 = ekd.from_source(
#     "mars",
#     request={
#         "anoffset": 12,
#         "class": "od",
#         "date": "2021-06-25",
#         "expver": 1,
#         "levelist": "50/100/150/200/250/300/400/500/600/700/850/925/1000",
#         "levtype": "pl",
#         "number": "0/1/2/3/4/5/6/7/8/9/10",
#         "param": "129.128/130.128/131/132/133.128",
#         "step": 3,
#         "stream": "elda",
#         "time": "21:00:00",
#         "type": "4v",
#         "target": "output"
#     },
# )
# d2_v4.to_target("file", f"{pangu_scratch}/orig_data/oper_eda_v4_pl.nc")


##### FOR ERA5 YOU NEED TO DO CDS

# # ERA5-G
# d3_pl = ekd.from_source(
#     "mars",
#     request={
#         "class": "ea",
#         "stream": "oper",
#         "type": "an",
#         "date": "2021-06-26",
#         "param": "129.128/130.128/131/132/133.128",
#         "levtype": "pl",
#         "levelist": "50/100/150/200/250/300/400/500/600/700/850/925/1000",
#         "time": "0000",
#         "format": "netcdf",
#     },
# )
# d3_pl.to_target("file", f"{pangu_scratch}/orig_data/era5_g_pl.nc")

# # ERA5-EDA
# d4_pl = ekd.from_source(
#     "mars",
#     request={
#         "class": "ea",
#         "date": "2021-06-26",
#         "expver": 1,
#         "levelist": "50/100/150/200/250/300/400/500/600/700/850/925/1000",
#         "levtype": "pl",
#         "number": "0/1/2/3/4/5/6/7/8/9/10",
#         "param": "129.128/130.128/131/132/133.128",
#         "stream": "enda",
#         "time": "00:00:00",
#         "type": "an"
#     },
# )
# d4_pl.to_target("file", f"{pangu_scratch}/orig_data/era5_eda_pl.nc")
