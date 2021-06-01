# coding: utf-8
import datetime as dt
import os
import re
import sys
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import rioxarray as riox
import xarray as xr
from tqdm import tqdm

from distributed import Client, LocalCluster


def _param(ds):
    if hasattr(ds, 'data_vars'):
        crs = ds.crs

        if 'LAI' in ds.data_vars:
            product = 'LAI'
            da = ds.LAI
        elif 'FCOVER' in ds.data_vars:
            product = 'FCOVER'
            da = ds.FCOVER
        elif 'FAPAR' in ds.data_vars:
            product = 'FAPAR'
            da = ds.FAPAR
        elif 'NDVI' in ds.data_vars:
            product = 'NDVI'
            da = ds.NDVI
        elif 'DMP' in ds.data_vars:
            product = 'DMP'
            da = ds.DMP
        elif 'GDMP' in ds.data_vars:
            product = 'GDMP'
            da = ds.GDMP
    else:
        product = ds.attrs['long_name']

        da = ds.rename({'band': 'time', 'y': 'lat', 'x': 'lon'})

        crs = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],' \
              'TOWGS84[0,0,0,0,0,0,0],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],' \
              'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9108"]],AUTHORITY["EPSG","4326"]]'

    if 'LAI' in product:
        param = {'product': 'LAI',
                 'short_name': 'leaf_area_index',
                 'long_name': 'Leaf Area Index Resampled 1 Km',
                 'grid_mapping': crs,
                 'flag_meanings': 'Missing',
                 'flag_values': '255',
                 'units': '',
                 'PHYSICAL_MIN': 0,
                 'PHYSICAL_MAX': 7,
                 'DIGITAL_MAX': 210,
                 'SCALING': 1./30,
                 'OFFSET': 0}

    elif 'FCOVER' in product:
        param = {'product': 'FCOVER',
                 'short_name': 'vegetation_area_fraction',
                 'long_name': 'Fraction of green Vegetation Cover Resampled 1 Km',
                 'grid_mapping': crs,
                 'flag_meanings': 'Missing',
                 'flag_values': '255',
                 'units': '',
                 'valid_range': '',
                 'PHYSICAL_MIN': 0,
                 'PHYSICAL_MAX': 1.,
                 'DIGITAL_MAX': 250,
                 'SCALING': 1./250,
                 'OFFSET': 0}

    elif 'FAPAR' in product:
        param = {'product': 'FAPAR',
                 'short_name': 'Fraction_of_Absorbed_Photosynthetically_Active_Radiation',
                 'long_name': 'Fraction of Absorbed Photosynthetically Active Radiation Resampled 1 KM',
                 'grid_mapping': crs,
                 'flag_meanings': 'Missing',
                 'flag_values': '255',
                 'units': '',
                 'valid_range': '',
                 'PHYSICAL_MIN': 0,
                 'PHYSICAL_MAX': 0.94,
                 'DIGITAL_MAX': 235,
                 'SCALING': 1./250,
                 'OFFSET': 0}

    elif 'NDVI' in product:
        param = {'product': 'NDVI',
                 'short_name': 'Normalized_difference_vegetation_index',
                 'long_name': 'Normalized Difference Vegetation Index Resampled 1 Km',
                 'grid_mapping': crs,
                 'flag_meanings': 'Missing cloud snow sea background',
                 'flag_values': '[251 252 253 254 255]',
                 'units': '',
                 'PHYSICAL_MIN': -0.08,
                 'PHYSICAL_MAX': 0.92,
                 'DIGITAL_MAX': 250,
                 'SCALING': 1./250,
                 'OFFSET': -0.08}

    elif 'DMP' in product:
        param = {'product': 'DMP',
                 'short_name': 'dry_matter_productivity',
                 'long_name': 'Dry matter productivity Resampled 1KM',
                 'grid_mapping': crs,
                 'flag_meanings': 'sea',
                 'flag_values': '-2',
                 'units': 'kg / ha / day',
                 'PHYSICAL_MIN': 0,
                 'PHYSICAL_MAX': 327.67,
                 'DIGITAL_MAX': 32767,
                 'SCALING': 1./100,
                 'OFFSET': 0}

    elif 'GDMP' in product:
        param = {'product': 'GDMP',
                 'short_name': 'Gross_dry_matter_productivity',
                 'long_name': 'Gross dry matter productivity Resampled 1KM',
                 'grid_mapping': crs,
                 'flag_meanings': 'sea',
                 'flag_values': '-2',
                 'units': 'kg / hectare / day',
                 'PHYSICAL_MIN': 0,
                 'PHYSICAL_MAX': 655.34,
                 'DIGITAL_MAX': 32767,
                 'SCALING': 1./50,
                 'OFFSET': 0}

    else:
        sys.exit('GLC product not found please chek')

    return da, param


def _downloader(user, psw, folder):
    url = 'https://land.copernicus.vgt.vito.be/manifest/'

    session = requests.Session()
    session.auth = (user, psw)

    manifest = session.get(url, allow_redirects=True)
    products = pd.read_html(manifest.text)[0][2:-1]['Name']
    products = products[products.str.contains('300_')].reset_index(drop=True)
    print(products)
    val = input('Please select the product from the list:')
    url = f'{url}{products[int(val)]}'

    manifest = session.get(url, allow_redirects=True)
    product = pd.read_html(manifest.text)[0][-2:-1]['Name'].values[0]
    purl = f'{url}{product}'
    r = session.get(purl, stream=True)
    rows = r.text.split('\n')
    dates = pd.DataFrame()
    for line in rows[:-1]:
        r = re.search(r"\d\d\d\d/\d\d/\d\d", line)
        dates = dates.append(pd.DataFrame([line], index=[pd.to_datetime(r[0], format="%Y/%m/%d")]))

    val = input('Please insert the date in teh format YYYY/MM/DD:')

    dates = dates.sort_index()
    i = dates.index.searchsorted(dt.datetime.strptime(val, "%Y/%m/%d"))
    link = dates.iloc[i][0]
    filename = os.path.basename(link)
    if folder != '':
        path = sys.path.join(folder, filename)
    else:
        path = filename

    r = session.get(link, stream=True)

    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    t = tqdm(total=total_size, unit='iB', unit_scale=True)

    with open(path, 'wb') as f:
        for data in r.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()
    if total_size != 0 and t.n != total_size:
        print("ERROR, something went wrong")

    return path


def _aoi(da, ds, AOI):
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def bnd_box_adj(my_ext):
        lat_1k = np.round(np.arange(80., -60., -1. / 112), 8)
        lon_1k = np.round(np.arange(-180., 180., 1. / 112), 8)

        lat_300 = ds.lat.values
        lon_300 = ds.lon.values
        ext_1k = np.zeros(4)

        # UPL Long 1K
        ext_1k[0] = find_nearest(lon_1k, my_ext[0]) - 1. / 336
        # UPL Lat 1K
        ext_1k[1] = find_nearest(lat_1k, my_ext[1]) + 1. / 336

        # LOWR Long 1K
        ext_1k[2] = find_nearest(lon_1k, my_ext[2]) + 1. / 336
        # LOWR Lat 1K
        ext_1k[3] = find_nearest(lat_1k, my_ext[3]) - 1. / 336

        # UPL
        my_ext[0] = find_nearest(lon_300, ext_1k[0])
        my_ext[1] = find_nearest(lat_300, ext_1k[1])

        # LOWR
        my_ext[2] = find_nearest(lon_300, ext_1k[2])
        my_ext[3] = find_nearest(lat_300, ext_1k[3])
        return my_ext

    if len(AOI):
        assert AOI[0] <= AOI[2], 'min Longitude is bigger than correspond Max, ' \
                                       'pls change position or check values.'
        assert AOI[1] >= AOI[3], 'min Latitude is bigger than correspond Max, ' \
                                       'pls change position or check values.'
        assert ds.lon[0] <= AOI[0] <= ds.lon[-1], 'min Longitudinal value out of original dataset Max ext.'
        assert ds.lat[-1] <= AOI[1] <= ds.lat[0], 'Max Latitudinal value out of original dataset Max ext.'

        assert ds.lon[0] <= AOI[2] <= ds.lon[-1], 'Max Longitudinal value out of original dataset Max ext.'
        assert ds.lat[-1] <= AOI[3] <= ds.lat[0], 'min Latitudinal value out of original dataset Max ext.'

        adj_ext = bnd_box_adj(AOI)
        try:
            da = da.sel(lon=slice(adj_ext[0], adj_ext[2]), lat=slice(adj_ext[1], adj_ext[3]))
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            raise sys.exit(1)
    else:
        da = da.shift(lat=1, lon=1)
    return da


def _date_extr(path):
    _, tail = os.path.split(path)
    pos = [pos for pos, char in enumerate(tail) if char == '_'][2]
    date = tail[pos + 1: pos + 9]
    date_h = pd.to_datetime(date, format='%Y%m%d')
    return date, date_h


def _resampler(path, my_ext, plot, out_folder):

    # Load the dataset
    if path.endswith('.nc'):
        ds = xr.open_dataset(path, mask_and_scale=False, chunks='auto')
        extension = '.nc'
    elif path.endswith('.tif'):
        ds = riox.open_rasterio(path, mask_and_scale=False, chunks='auto')
        extension = '.tif'

    # select parameters according to the product.
    da, param = _param(ds)
    date, date_h = _date_extr(path)

    # AOI
    da = _aoi(da, ds, my_ext)

    # Algorithm core
    try:
        # create the mask according to the fixed values
        da_msk = da.where(da <= param['DIGITAL_MAX'])

        # create the coarsen dataset
        coarsen = da_msk.coarsen(lat=3, lon=3, boundary='trim').mean(keep_attrs=False)

        # force results to integer
        coarsen_int = np.rint(coarsen)

        # mask the dataset according to the minumum required values
        vo = xr.where(da <= param['DIGITAL_MAX'], 1, 0)
        vo_cnt = vo.coarsen(lat=3, lon=3, boundary='trim').sum(keep_attrs=False)
        da_r = coarsen_int.where(vo_cnt >= 5)

        # force nan to int
        da_r = xr.where(np.isnan(da_r), 255, coarsen_int)

        # Add time dimension
        if not da_r.time:
            da_r = da_r.assign_coords({'time': date_h})
            da_r = da_r.expand_dims(dim='time', axis=0)

    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)
        raise sys.exit(1)
    # Output write
    name = param['product']

    if len(my_ext) != 0:
        file_name = f'CGLS_{name}_{date}_1KM_Resampled_AOI'
    else:
        file_name = f'CGLS_{name}_{date}_1KM_Resampled_'

    if extension == '.nc':
        try:
            da_r.name = param['product']
            da_r.attrs['short_name'] = param['short_name']
            da_r.attrs['long_name'] = param['long_name']
            da_r.attrs['_FillValue'] = int(255)
            da_r.attrs['scale_factor'] = np.float32(param['SCALING'])
            da_r.attrs['add_offset'] = np.float32(param['OFFSET'])

            dso = xr.Dataset({param['product']: da_r, 'crs': param['grid_mapping']})
            dso.attrs = ds.attrs
            remove_list = ['processing_level', 'identifier', 'institution', 'processing_mode', 'archive_facility', 'parent_identifier', 'history']
            for i in remove_list:
                dso.attrs.pop(i, '')

            dso.attrs['title'] = '10-daily Normalized Difference Vegetation Index 1Km resampled'

            prmts = dict({param['product']: {'dtype': 'i4', 'zlib': 'True', 'complevel': 4}})

            out_file = os.path.join(out_folder, f'{file_name}.nc')

            ds.to_netcdf(out_file, encoding=prmts)

        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            raise sys.exit(1)

    elif extension == '.tif':
        try:
            da_r.name = param['product']
            da_r.attrs['short_name'] = param['short_name']
            da_r.attrs['long_name'] = param['long_name']
            da_r.attrs['_FillValue'] = int(255)
            da_r.attrs['scale_factor'] = np.float32(param['SCALING'])
            da_r.attrs['add_offset'] = np.float32(param['OFFSET'])

            name = param['product']
            if len(my_ext) != 0:
                file_name = f'CGLS_{name}_{date}_1KM_Resampled_AOI'
            else:
                file_name = f'CGLS_{name}_{date}_1KM_Resampled'

            out_file = os.path.join(out_folder, f'{file_name}.tif')

            da_r.rio.write_crs("epsg:4326", inplace=True)
            da_r.rio.set_nodata(-999, inplace=True)
            da_r = da_r.rename({'time': 'band', 'lat': 'y', 'lon': 'x'})

            da_r.rio.to_raster(out_file, **{'compress': 'lzw'})

        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            raise sys.exit(1)

    print(f'{file_name} resampled')

    # Plot
    if plot:
        da_r.plot(robust=True, cmap='YlGn', figsize=(15, 10))
        plt.title(f'Copernicus Global Land\n Resampled {name} to 1K over Europe\n date: {date_h.date()}')
        plt.ylabel('latitude')
        plt.xlabel('longitude')
        plt.draw()
        plt.show()


def main(path, out_folder, user, psw, AOI, plot):

    # Create the processing environment
    cluster = LocalCluster()
    client = Client(cluster)

    # Processing
    if path == '':
        # Download and process
        assert user, 'User ID is empty'
        assert psw, 'Password is empty'

        path = _downloader(user, psw, out_folder)
        _resampler(path, AOI, plot, out_folder)
    elif os.path.isfile(path):
        # Single file process
        _resampler(path, AOI, plot, out_folder)
    elif os.path.isdir(path):
        # Multiprocessing for local files
        if not os.listdir(path):
            print("Directory is empty")
        else:
            for filename in os.listdir(path):
                if filename.endswith(".nc"):
                    path_ = os.path.join(path, filename)
                    _resampler(path_, AOI, plot, out_folder)
    else:
        assert os.path.isfile(path), print('Sorry! Path isn\'t pointing to a file')

    print('Conversion done')
    cluster.close()
    client.close()


if __name__ == '__main__':
    """ Copernics Global Land Resampler 333m to 1 Km

        The aim of this tool is to facilitate the resampling of the 333m ProbaV Copernicus Global Land Service products
        [1] (i.e. NDVI, FaPAR LAI, ... ) to the coarsen resolution of 1km.
        With the present release only the main indexes per products can be resampled. Other indexes, like the RMSE,
        can't be resampled.
        More info a about quality assessment can be found in the report create for the R version of this tool [2]

        [1] https://land.copernicus.eu/global/themes/vegetation
        [2] https://github.com/xavi-rp/ResampleTool_notebook/blob/master/Resample_Report_v2.5.pdf
    """

    '''
    Instructions:
    The tool is able to process act in different way according to the necessity

    - Single file: fill the path with the exact position and the file name 
      The tool is able to ingest NetCDF files and Tiff. Both will keep the format. NetCDF->NetCDF, Tif->Tif 
    - SemiAutomatic download (single file): Leave the path empty, a wizard will help in the selection and download.
      If the semiautomatic download is selected as option user ID and password must be defined. 
      Credential can be obtained here https://land.copernicus.vgt.vito.be/PDF/portal/Application.html#Home 
      through the Register form (on the upper right part of the page)
    '''
    print('Copernics Global Land Resampler 333m to 1 Km')

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('-i', '--input_path', help='Input file path', type=str, default='')
        parser.add_argument('-o', '--output_path', help='Output file path', type=str, default='')
        parser.add_argument('-u', '--user_name', help='User name', type=str, default='')
        parser.add_argument('-p', '--password', help='Password', type=str, default='')
        parser.add_argument('-a', '--aoi',  help='Area of interest', nargs='+', type=float, default=[])
        parser.add_argument('-plt', '--plot', help='Option to plot the result', type=bool, default=False)
        args = parser.parse_args()

        main(args.input_path,
             args.output_path,
             args.user_name,
             args.password,
             args.aoi,
             args.plot)

    except AssertionError as error:
        pass
    except KeyboardInterrupt:
        print('Process killed by user')
        raise sys.exit(1)
