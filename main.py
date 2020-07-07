# coding: utf-8
import xarray as xr
import sys
import numpy as np
import matplotlib.pyplot as plt


def GCL():
  
    path = 'D:/Data/CGL_subproject_coarse_res/04_ndvi/c_gls_NDVI300_201905210000_GLOBE_PROBAV_V1.0.1.nc'
    my_ext = [-18.58, 62.95, 51.57, 28.5]

    ds = xr.open_dataset(path, mask_and_scale=True)

    if 'LAI' in ds.data_vars:
        product = 'LAI'
        DN_max = 7.
        da = ds.LAI
        short_name = ''
        long_name = ''
    elif 'FCOVER' in ds.data_vars:
        product = 'FCOVER'
        DN_max = .94
        da = ds.FCOVER
        short_name = ''
        long_name = ''
    elif 'FAPAR' in ds.data_vars:
        product = 'FAPAR'
        DN_max = .94
        da = ds.FAPAR
        short_name = ''
        long_name = ''
    elif 'NDVI' in ds.data_vars:
        product = 'NDVI'
        DN_max = .92
        da = ds.NDVI
        short_name = 'Normalized_difference_vegetation_index'
        long_name = 'Normalized Difference Vegetation Index Resampled 1 Km'
    elif 'DMP' in ds.data_vars:
        DN_max = 327.67
        product = 'DMP'
        da = ds.DMP
        short_name = ''
        long_name = ''
    elif 'GDMP' in ds.data_vars:
        product = 'GDMP'
        DN_max = 655.34
        da = ds.GDMP
        short_name = ''
        long_name = ''
    else:
        sys.exit('GLC product not found please chek')

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def bnd_box_adj(my_est):
        lat_1k = np.arange(80., -60., -1. / 112)
        lon_1k = np.arange(-180., 180., 1. / 112)

        lat_300 = ds.lat.values
        lon_300 = ds.lon.values
        ext_1K = np.zeros(4)

        # TODO find a more pythonic way
        ext_1K[0] = find_nearest(lon_1k, my_ext[0]) - 1 / 224
        ext_1K[1] = find_nearest(lat_1k, my_ext[1]) + 1 / 224
        ext_1K[2] = find_nearest(lon_1k, my_ext[2]) - 1 / 224
        ext_1K[3] = find_nearest(lat_1k, my_ext[3]) + 1 / 224

        my_ext[0] = find_nearest(lat_300, ext_1K[0])
        my_ext[1] = find_nearest(lon_300, ext_1K[1])
        my_ext[2] = find_nearest(lat_300, ext_1K[2])
        my_ext[3] = find_nearest(lon_300, ext_1K[3])
        return my_ext

    if len(my_ext):
        assert my_ext[1] >= my_ext[3], 'min Latitude is bigger than correspond Max, ' \
                                       'pls change position or check values.'
        assert my_ext[0] <= my_ext[2], 'min Longitude is bigger than correspond Max, ' \
                                       'pls change position or check values.'
        assert ds.lat[-1] <= my_ext[3] <= ds.lat[0], 'min Latitudinal value out of original dataset Max ext.'
        assert ds.lat[-1] <= my_ext[1] <= ds.lat[0], 'Max Latitudinal value out of original dataset Max ext.'
        assert ds.lon[0] <= my_ext[0] <= ds.lon[-1], 'min Longitudinal value out of original dataset Max ext.'
        assert ds.lon[0] <= my_ext[2] <= ds.lon[-1], 'Max Longitudinal value out of original dataset Max ext.'
        adj_ext = bnd_box_adj(my_ext)

        da = da.sel(lon=slice(adj_ext[0], adj_ext[2]), lat=slice(adj_ext[1], adj_ext[3]))
    else:
        da = da.shift(lat=1, lon=1)

    # TODO differentiate according to the different products structures
    da_msk = da.where(da <= DN_max)

    coarsen = da_msk.coarsen(lat=3, lon=3, coord_func=np.mean, boundary='trim', keep_attrs=False).mean()

    vo = xr.where(da <= DN_max, 1, 0)
    vo_cnt = vo.coarsen(lat=3, lon=3, coord_func=np.mean, boundary='trim', keep_attrs=False).sum()
    da_r = coarsen.where(vo_cnt >= 5)

    da_r.name = product
    da_r.attrs['short_name'] = short_name
    da_r.attrs['long_name'] = long_name
    prmts = dict({product: {'dtype': 'f8', 'zlib': 'True', 'complevel': 4}})
    da_r.to_netcdf(f'D:/Data/CGL_subproject_coarse_res/Tests/CGL_{product}_1KM_R_Europe_20190521.nc', encoding=prmts)
    print('Done')

    da_r.plot(robust=True, cmap='YlGn', figsize=(15, 10))

    plt.title(f'Copernicus Global Land\n Resampled {product} to 1K over Europe\n date: 20190521')
    plt.ylabel('latitude')
    plt.xlabel('longitude')
    plt.draw()


if __name__ == '__main__':
    GCL()
