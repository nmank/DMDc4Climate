import os
import sys
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

from sklearn.decomposition import PCA

from cartopy.util import add_cyclic_point
import cartopy.crs as ccrs


GtC_to_GtCO2 = 3.67
Gt_to_Mt = 1e3
kg_to_Mt = 1e-9
yr_to_s = 3600 * 24 * 365
EARTH_RADIUS = 6371000.0  # m

def log_transform(X: np.array, min_val: float, eps: float = 1e-8):
    return np.log(X-min_val + eps)

def load_emissions_dataset(filepath: str)-> xr.Dataset:
    """
    Load emissions data

    Parameters
    ----------
    filepath: string
        Path to emissions data.
    Returns
    -------
        xarray dataset of emissions
    """
    inputs = xr.open_dataset(filepath).compute()
    inputs.CO2.data = inputs.CO2.data / GtC_to_GtCO2
    inputs.CO2.attrs['units'] = 'GtC/yr'
    inputs.CH4.data = inputs.CH4.data * Gt_to_Mt
    inputs.CH4.attrs['units'] = 'MtCH4/yr'
    inputs.SO2.data = inputs.SO2.data * kg_to_Mt * yr_to_s
    inputs.SO2.attrs['units'] = 'MtSO2/yr m-2'
    inputs.BC.data = inputs.BC.data * kg_to_Mt * yr_to_s
    inputs.BC.attrs['units'] = 'MtBC/yr m-2'
    return inputs

def _guess_bounds(points: np.array, bound_position: float = 0.5) -> np.array:
    """
    Guess bounds of grid cells.

    Simplified function from iris.coord.Coord.

    Parameters
    ----------
    points: numpy.array
        Array of grid points of shape (N,).
    bound_position: float, optional
        Bounds offset relative to the grid cell centre.
    Returns
    -------
    Array of shape (N, 2).
    """
    diffs = np.diff(points)
    diffs = np.insert(diffs, 0, diffs[0])
    diffs = np.append(diffs, diffs[-1])

    min_bounds = points - diffs[:-1] * bound_position
    max_bounds = points + diffs[1:] * (1 - bound_position)

    return np.array([min_bounds, max_bounds]).transpose()

def grid_cell_areas(lon1d: np.array, lat1d: np.array, radius: float = EARTH_RADIUS) -> np.array:
    """
    Calculate grid cell areas given 1D arrays of longitudes and latitudes
    for a planet with the given radius.

    Parameters
    ----------
    lon1d: numpy.array
        Array of longitude points [degrees] of shape (M,)
    lat1d: numpy.array
        Array of latitude points [degrees] of shape (M,)
    radius: float, optional
        Radius of the planet [metres] (currently assumed spherical)
    Returns
    -------
    Array of grid cell areas [metres**2] of shape (M, N).
    """
    lon_bounds_radian = np.deg2rad(_guess_bounds(lon1d))
    lat_bounds_radian = np.deg2rad(_guess_bounds(lat1d))
    area = _quadrant_area(lat_bounds_radian, lon_bounds_radian, radius)
    return area

def _quadrant_area(radian_lat_bounds: np.array, radian_lon_bounds: np.array, radius_of_earth: float) -> np.array:
    """
    Calculate spherical segment areas.
    Taken from SciTools iris library.
    Area weights are calculated for each lat/lon cell as:
        .. math::
            r^2 (lon_1 - lon_0) ( sin(lat_1) - sin(lat_0))
    The resulting array will have a shape of
    *(radian_lat_bounds.shape[0], radian_lon_bounds.shape[0])*
    The calculations are done at 64 bit precision and the returned array
    will be of type numpy.float64.
    Parameters
    ----------
    radian_lat_bounds: numpy.array
        Array of latitude bounds (radians) of shape (M, 2)
    radian_lon_bounds: numpy.array
        Array of longitude bounds (radians) of shape (N, 2)
    radius_of_earth: float
        Radius of the Earth (currently assumed spherical)
    Returns
    -------
    Array of grid cell areas of shape (M, N).
    """
    # ensure pairs of bounds
    if (
        radian_lat_bounds.shape[-1] != 2
        or radian_lon_bounds.shape[-1] != 2
        or radian_lat_bounds.ndim != 2
        or radian_lon_bounds.ndim != 2
    ):
        raise ValueError("Bounds must be [n,2] array")

    # fill in a new array of areas
    radius_sqr = radius_of_earth ** 2
    radian_lat_64 = radian_lat_bounds.astype(np.float64)
    radian_lon_64 = radian_lon_bounds.astype(np.float64)

    ylen = np.sin(radian_lat_64[:, 1]) - np.sin(radian_lat_64[:, 0])
    xlen = radian_lon_64[:, 1] - radian_lon_64[:, 0]
    areas = radius_sqr * np.outer(ylen, xlen)

    # we use abs because backwards bounds (min > max) give negative areas.
    return np.abs(areas)

def extract_forcing_agents(train_data: xr.Dataset, test_data: xr.Dataset) -> tuple:
    """
    Extract forcing agents from training and testing emissions data.

    Parameters
    ----------
    train_data: xr.Dataset
        training emissions data
    test_data: xr.Dataset
        testing emissions data
    Returns
    -------
    numpy array of times, numpy array of emissions
    """
    # Get dimensions
    t_train, nlat, nlong = train_data.BC.values.shape
    t_test = test_data.BC.values.shape[0]

    # Extract time steps array
    times = np.hstack([train_data.time.values, test_data.time.values])

    # Extract cumulative C02 emissions
    cum_CO2_emissions = np.expand_dims(np.hstack([train_data.CO2.values, test_data.CO2.values]), axis = 1)

    # Extract CH4 emissions
    CH4_emissions = np.expand_dims(np.hstack([train_data.CH4.values, test_data.CH4.values]), axis = 1)

    # Extract SO2 emissions
    SO2_train = train_data.SO2.values.reshape([t_train,nlat*nlong])
    SO2_test = test_data.SO2.values.reshape([t_test,nlat*nlong])
    SO2_data = np.vstack([SO2_train, SO2_test])
    SO2_pca = PCA(n_components = 5) # to get >95% variance for all scenarios
    SO2_emissions = SO2_pca.fit_transform(SO2_data)
    # SO2_emissions = np.expand_dims(np.mean(SO2_data, axis = 1), axis =1)

    # Extract BC emissions
    BC_train = train_data.BC.values.reshape([t_train,nlat*nlong])
    BC_test = test_data.BC.values.reshape([t_test,nlat*nlong])
    BC_data = np.vstack([BC_train, BC_test])
    BC_pca = PCA(n_components = 4)# to get >95% variance for all scenarios
    BC_emissions = BC_pca.fit_transform(BC_data)
    # BC_emissions = np.expand_dims(np.mean(BC_data, axis = 1), axis = 1)

    emissions = np.hstack([cum_CO2_emissions, CH4_emissions, SO2_emissions, BC_emissions])

    return times, emissions

def crop_data(data_array: np.array, lat_data: xr.DataArray, lon_data: xr.DataArray, region_id: str):
    """
    Extract forcing agents from training and testing emissions data.

    Parameters
    ----------
    data_array: np.array
        flattened array of dataset
    lat_data: xr.DataArray
        latitude data
    lon_data: xr.DataArray
        longitude data
    region_id: str
        the region to crop
    Returns
    -------
    a xr.DataArray of the cropped data with corresponding latitude and longitude coordinates
    """

    # use this code https://github.com/lm2612/GPRegression/blob/master/RegionLatitudes.py
    RegionLonsLats = {'Arctic':(0.,360.,66.,90.),
            'NorthAmerica':(230.,300.,10.,66.),
            'NorthPacific':(145.  ,230.,10.,66.),
            'SouthPacific':(180., 360.-80., -50.,10.),
            'SouthAmerica':(360.-80.,360.-35.,-50.,10.),
            'Antarctic':(0.,360.,-90.,-66.),
            'SouthernOcean':(0.,360.,-66.,-50.),
            'SouthAtlantic':(360.-35.,10. ,-50.,10.),
            'NorthAtlantic':(300.,340.,10.,66.),
            'NorthernAfrica':(340.,50.,10.,35.), # would prefer this to start at 5 deg?
            'SouthernAfrica':(10.,50.,-50.,10.),
            'Europe':(340.,50.,35.,66.),
            'Russia':(50.,  100. ,35.,66.),
            'SouthAsia':(50.,100.,0.,35.),
            'IndianOcean':(50.,100.  ,-50. ,0.),
            'Oceania':(100.,180. ,-50., 10.),
            'EastAsia':(100.,145.,10.,66. ),
            'Global':(0.,360.,-90.,90.),
            'NH':(0.,360.,0.,90.),
            'SH':(0.,360.,-90.,0.),
            'NHML':(0.,360.,30.,60.),
            'SHML':(0.,360.,-60.,-30.),
            'NHHL':(0.,360.,60.,90.),
            'SHHL':(0.,360.,-90.,-60.),
            'Tropics':(0.,360.,-30.,30.)
    }

    pixel_vals = np.real(data_array).reshape(96, 144)

    data = xr.DataArray(  data = pixel_vals,
                                    dims = ('lat', 'lon'),
                                    coords = {'lat': lat_data,
                                            'lon': lon_data})
    
    min_lon, max_lon, min_lat, max_lat = RegionLonsLats[region_id]

    if min_lon < max_lon:
            mask_lon = (data.lon >= min_lon) & (data.lon <= max_lon)
    else:
            mask_lon = (data.lon >= min_lon) | (data.lon <= max_lon)
            
    if min_lat < max_lat:
            mask_lat = (data.lat >= min_lat) & (data.lat <= max_lat)
    else:
            mask_lat = (data.lat >= min_lat) | (data.lat <= max_lat)


    data = data.where(mask_lon & mask_lat, drop=True)

    return data

def make_plot(the_data: np.array, lat_data: np.array, lon_data: np.array, fig, ax, 
              mode_num: int = 0, colorbar: bool = False, min_val: float = 0, max_val: float = 1):
    """
    Plot the data

    Parameters
    ----------
    the_data: np.array
        flattened array of dataset
    lat_data: xr.DataArray
        latitude data
    lon_data: xr.DataArray
        longitude data
    fig: Figure
        pyplot figure for the plot
    ax: Axes
        pyplot axis for the plot
    mode_num: int
        mode number for title of the plot
    colorbar: bool
        add a colorbar?
    min_val: float
        minimum value for map
    max_val: float
        maximum value for map
    Returns
    -------
    plotly figure and axis of the plot
    """
              
    pixel_vals = np.real(the_data).reshape(96, 144)

    field = xr.DataArray(data = pixel_vals,
                         dims = ('lat', 'lon'),
                         coords = {'lat': lat_data,
                                        'lon': lon_data})

    cmap = 'RdBu_r'

    wrap_data, wrap_lon = add_cyclic_point(field.values,
                                            coord=field.lon,
                                            axis=field.dims.index('lon'))

    levels = np.linspace(min_val, max_val, 150)

    cx = ax.contourf(wrap_lon,
                        field.lat,
                        wrap_data,
                        levels=levels,
                        cmap=cmap,
                        extend='both',
                        transform=ccrs.PlateCarree())
    ax.set_global()
    ax.coastlines()

    

    


    if mode_num > 0:
        ax.set_title(f'Mode {mode_num}')

    if colorbar:
        ax.gridlines(draw_labels=True)
        # cax = ax.inset_axes((-.04, 0, 0.02, 1))
        cax = ax.inset_axes((-.08, 0, 0.02, 1))
        cbar = fig.colorbar(cx, cax=cax)
        cbar.set_ticks(np.linspace(min_val, max_val,7))
        cax.yaxis.tick_left()
        cbar.ax.tick_params(labelsize=14)


    return fig, ax

def get_real(dmds: dict, ssp: str, mode) -> np.array: 
    """
    Get the scaled dmd mode

    Parameters
    ----------
    dmds: dict
        dmd objects from DMDc
    ssp: string
        string of SSP
    mode: int or list
        the modes to plot
    Returns
    -------
    the dmd mode as a numpy array (real valued)
    """

    if type(mode) == list:
        the_mode = np.real(dmds[ssp].modes[:, mode[0]]*dmds[ssp].amplitudes[mode[0]] + dmds[ssp].modes[:, mode[1]]*dmds[ssp].amplitudes[mode[1]])
    else:
        the_mode = np.real(dmds[ssp].modes[:, mode]*dmds[ssp].amplitudes[mode])
        
    return the_mode

def TDE_wravel(X: np.array, lag: int):
    """
    Generate a time lagged (time delay embedded) numpy array.

    Parameters
    ----------
    X: np.array
        The data to be time delay embedded. (Time x Features)
    lag: int
        The time delay
    Returns
    -------
    Time delay embedded nunmpy array
    """

    T, N = X.shape
    X_delay = np.zeros((T-lag+1, N*lag))
    for l in range(lag):
        X_delay[:,l*N:(l+1)*N] = X[l:T-(lag-1)+l]
    
    return X_delay