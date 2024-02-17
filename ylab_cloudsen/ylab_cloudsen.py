from collections import OrderedDict
from typing import Union

import numpy as np
import pandas as pd
import rioxarray
import xarray as xr
from rasterio.enums import Resampling


class SatObject:
    """
    Main class for satellite scene handling
    """

    def __init__(
        self,
        rad_offset=None,  # Radiometric offset (for processing baseline 04.00 and later)
        Aerosol=None,
        Blue=None,
        Green=None,
        Red=None,
        RedEdge1=None,
        RedEdge2=None,
        RedEdge3=None,
        NIR=None,
        NIR2=None,
        WaterVapor=None,
        Cirrus=None,
        SWIR1=None,
        SWIR2=None,
        TIR1=None,
        TIR2=None,
        HV=None,
        VH=None,
        HH=None,
        VV=None,
    ):
        self.rad_offset: Union(float, None) = rad_offset
        self.Aerosol: Union(str, xr.DataArray) = Aerosol
        self.Blue: Union(str, xr.DataArray) = Blue
        self.Green: Union(str, xr.DataArray) = Green
        self.Red: Union(str, xr.DataArray) = Red
        self.RedEdge1: Union(str, xr.DataArray) = RedEdge1
        self.RedEdge2: Union(str, xr.DataArray) = RedEdge2
        self.RedEdge3: Union(str, xr.DataArray) = RedEdge3
        self.NIR: Union(str, xr.DataArray) = NIR
        self.NIR2: Union(str, xr.DataArray) = NIR2
        self.WaterVapor: Union(str, xr.DataArray) = WaterVapor
        self.Cirrus: Union(str, xr.DataArray) = Cirrus
        self.SWIR1: Union(str, xr.DataArray) = SWIR1
        self.SWIR2: Union(str, xr.DataArray) = SWIR2
        self.TIR1: Union(str, xr.DataArray) = TIR1
        self.TIR2: Union(str, xr.DataArray) = TIR2
        self.HV: Union(str, xr.DataArray) = HV
        self.VH: Union(str, xr.DataArray) = VH
        self.HH: Union(str, xr.DataArray) = HH
        self.VV: Union(str, xr.DataArray) = VV
        self.ref = OrderedDict(
            {
                "Aerosol": 0,
                "Blue": 1,
                "Green": 2,
                "Red": 3,
                "RedEdge1": 4,
                "RedEdge2": 5,
                "RedEdge3": 6,
                "NIR": 7,
                "NIR2": 8,
                "WaterVapor": 9,
                "Cirrus": 10,
                "SWIR1": 11,
                "SWIR2": 12,
                "TIR1": 13,
                "TIR2": 14,
                "HV": 15,
                "VH": 16,
                "HH": 17,
                "VV": 18,
            }
        )
        self.ref_rev = OrderedDict({v: k for k, v in self.ref.items()})
        self.arr_stack = None
        self.bounds = None

    def apply_rad_offset(self, obj):
        # Set values > 0 and < 1000 to 0
        obj = obj.where(~((obj > 0) & (obj < (-self.rad_offset))), 0)
        # Subtract 1000 from all values greater than 0
        obj = obj.where(obj <= 0, obj + self.rad_offset)
        return obj

    def load_data(self, obj):
        if isinstance(obj, str):
            with rioxarray.open_rasterio(obj, "r") as src:
                raster = np.squeeze(src, axis=0)
                return raster
        if isinstance(obj, xr.DataArray):
            return obj
        elif obj is None:
            return None
        else:
            raise TypeError("object must be a string, or an xarray.")

    # If somebody know how to do it in a more elegant way, please, feel free
    # to implement it
    @property
    def Aerosol(self):
        return self._arr_stack.sel(band="Aerosol")

    @Aerosol.setter
    def Aerosol(self, value):
        self._Aerosol = self.load_data(value)

    @property
    def Blue(self):
        return self._arr_stack.sel(band="Blue")

    @Blue.setter
    def Blue(self, value):
        self._Blue = self.load_data(value)

    @property
    def Green(self):
        return self._arr_stack.sel(band="Green")

    @Green.setter
    def Green(self, value):
        self._Green = self.load_data(value)

    @property
    def Red(self):
        return self._arr_stack.sel(band="Red")

    @Red.setter
    def Red(self, value):
        self._Red = self.load_data(value)

    @property
    def RedEdge1(self):
        return self._arr_stack.sel(band="RedEdge1")

    @RedEdge1.setter
    def RedEdge1(self, value):
        self._RedEdge1 = self.load_data(value)

    @property
    def RedEdge2(self):
        return self._arr_stack.sel(band="RedEdge2")

    @RedEdge2.setter
    def RedEdge2(self, value):
        self._RedEdge2 = self.load_data(value)

    @property
    def RedEdge3(self):
        return self._arr_stack.sel(band="RedEdge3")

    @RedEdge3.setter
    def RedEdge3(self, value):
        self._RedEdge3 = self.load_data(value)

    @property
    def NIR(self):
        return self._arr_stack.sel(band="NIR")

    @NIR.setter
    def NIR(self, value):
        self._NIR = self.load_data(value)

    @property
    def NIR2(self):
        return self._arr_stack.sel(band="NIR2")

    @NIR2.setter
    def NIR2(self, value):
        self._NIR2 = self.load_data(value)

    @property
    def WaterVapor(self):
        return self._arr_stack.sel(band="WaterVapor")

    @WaterVapor.setter
    def WaterVapor(self, value):
        self._WaterVapor = self.load_data(value)

    @property
    def Cirrus(self):
        return self._arr_stack.sel(band="Cirrus")

    @Cirrus.setter
    def Cirrus(self, value):
        self._Cirrus = self.load_data(value)

    @property
    def SWIR1(self):
        return self._arr_stack.sel(band="SWIR1")

    @SWIR1.setter
    def SWIR1(self, value):
        self._SWIR1 = self.load_data(value)

    @property
    def SWIR2(self):
        return self._arr_stack.sel(band="SWIR2")

    @SWIR2.setter
    def SWIR2(self, value):
        self._SWIR2 = self.load_data(value)

    @property
    def TIR1(self):
        return self._arr_stack.sel(band="TIR1")

    @TIR1.setter
    def TIR1(self, value):
        self._TIR1 = self.load_data(value)

    @property
    def TIR2(self):
        return self._arr_stack.sel(band="TIR2")

    @TIR2.setter
    def TIR2(self, value):
        self._TIR2 = self.load_data(value)

    @property
    def HV(self):
        return self._arr_stack.sel(band="HV")

    @HV.setter
    def HV(self, value):
        self._HV = self.load_data(value)

    @property
    def VH(self):
        return self._arr_stack.sel(band="VH")

    @VH.setter
    def VH(self, value):
        self._VH = self.load_data(value)

    @property
    def HH(self):
        return self._arr_stack.sel(band="HH")

    @HH.setter
    def HH(self, value):
        self._HH = self.load_data(value)

    @property
    def VV(self):
        return self._arr_stack.sel(band="VV")

    @VV.setter
    def VV(self, value):
        self._VV = self.load_data(value)

    @property
    def arr_stack(self):
        return self._arr_stack

    @arr_stack.setter
    def arr_stack(self, value):
        out_arr = []
        band_names = []
        resolutions = []
        for _, attr in self.ref_rev.items():
            obj = eval(f"self._{attr}")
            if isinstance(obj, xr.DataArray):
                # Get the band name and resolution
                resolutions.append(obj.rio.resolution())
                band_names.append(attr)
                out_arr.append(obj)
            # Remove xr.DataArray from an attribute -- prevent data duplication
            setattr(self, f"_{attr}", None)
            setattr(self, attr, None)

        # Resample to match minimal resolution
        min_res = min(resolutions)
        reference_rst = out_arr[min(resolutions.index(min_res), len(resolutions) - 1)]
        for index, value in enumerate(out_arr):
            # Apply radiometric offset, if needed
            if self.rad_offset is not None:
                out_arr[index] = self.apply_rad_offset(value, self.rad_offset)
            # Resample to minimal resolution
            if resolutions[index] != min_res:
                out_arr[index] = value.rio.reproject(
                    dst_crs=reference_rst.rio.crs,
                    shape=reference_rst.shape,
                    transform=reference_rst.rio.transform(),
                    resampling=Resampling.nearest,
                )

        # Stack bands into a 3D array with band names as index
        out_arr = xr.concat(out_arr, pd.Index(band_names, name="band"))
        self._arr_stack = out_arr
