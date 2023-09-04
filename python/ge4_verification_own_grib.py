import cartopy.feature as cfeat
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import iris
import iris.plot as iplt
import iris.quickplot as qplt
import os
import sys
import numpy as np
import datetime
from datetime import timedelta
import math
import iris_grib
import matplotlib as mpl
import matplotlib.colors as colors
import iris.analysis.stats as stats


""" usage: python3 ge4_verification_own_grib.py suite_name cycle field pressure_level timestep domain spec_range type ver_option outpath nens
           ver_res
    cycle: time string of the form yyyymmddThhmmZ
    field: uwind, vwind, wind, temperature, geopotential_height, mslp,
           accumulated_precipitation_x (where x is one of 3h, 6h, ... timestep - in hours), total_cloud, wind_10m,
           precipitation_rate, uwind_10m, vwind_10m, temperature_2m, dew_temperature_2m, high_cloud, middle_cloud, low_cloud
    pressure: 1000.0, 925.0, 850.0, 700.0, 500.0, 400.0, 300.0, 250.0, 200.0
    timestep: 0, 3, 6, 9, ... 99 (for rainfall), 198 (other fields) (in hours)
    domain: global, regional, australia, australia_wmo, greenland
    spec_range: specify field plot range values (low-high) or type 'auto' if to be determined automatically
    type: mean, stdev, var, prob? (where ? is a threshold value e.g., 'prob4000' would be interpreted as the probability
                                   of exceeding a value of 4000 if field is 'geopotential_height'i; note that this doesn't
                                   work for the wind (vector) field)
    ver_option: ver=0 (No verification); ver=1 (Verification against mean analysis); ver=2 (Verification against ensemble analysis) 
    outpath: output file path
    nens: number of ensemble members
    ver_res: verification (and plot) resolution in degrees (generally 0.5 for plots and 1.5 for verification)
""" 
# number of contour levels
nlevs = 10

# number of thresholds for RPS calculation
nthresh = 10


def extract_domain(cube_g):
    """ Extracts subset of global cube as defined by domain size 
        and resolution
    """
    # make sure lons are in [0, 360] range
    if lonmin < 0.0:
        lonmin2 = lonmin + 360.0
    else:
        lonmin2 = lonmin

    if lonmax < 0.0:
        lonmax2 = lonmax + 360.0
    else:
        lonmax2 = lonmax

    # create new verification grid
    nlon = int(round((lonmax2 - lonmin2)/ver_res))
    nlat = int(round((latmax - latmin)/ver_res))
    sample_points = [('longitude', np.linspace(lonmin2, lonmax2, nlon)),
                     ('latitude',  np.linspace(latmin, latmax, nlat))]

    cube_intp = cube_g.interpolate(sample_points, iris.analysis.Linear())

    cube_intp.coord('longitude').guess_bounds()
    cube_intp.coord('latitude').guess_bounds()
    cube_g.coord('longitude').guess_bounds()
    cube_g.coord('latitude').guess_bounds()

    cube_regrd = cube_g.regrid(cube_intp, iris.analysis.AreaWeighted())


    return cube_regrd


def get_field(infile, field, field_name, pressure_level, surf, timestep):
    """
        Select relevant field, time, and pressure level from cube
    """
    loaded_cubes = iris_grib.load_cubes(infile)
    all_cubes = list(loaded_cubes)
    selected_cubes = []
    for cube in all_cubes:
        if cube.name() == field_name:
            if surf is False:
                level = cube.coord('pressure').points[0]
            else:
                level = 100.0*pressure_level
            if abs(level/100.0 - pressure_level) < 1.0E-2:
                time = cube.coord('forecast_period').points[0]
                # accumulated precip forecast period is at midpoint of start and end accum period
                if field_name == 'unknown':
                    time  = 2.0*time
                if time == int(timestep):
                    cube2d = cube
    # extract cube in specified domain
    cube_domain = extract_domain(cube2d)

    return cube_domain


def get_ens_field(suite_name, cycle, field, field_name, field_name_long, pressure_level, surf, timestep):

    """ Get ensemble member cubes corresponding to field, pressure level, and timestep"""

    # define lagged timestep (6 h ahead of specified timestep)
    timestep_nolag = timestep
    timestep_lag = str(int(timestep) + 6)
    timestep_lag_12h = str(int(timestep) + 12)

    # obtain data filepaths
    infiles, ens_nums = get_data_path(suite_name, cycle, timestep, field_name, surf)

    # for accumulated rain, find user-specified accumulation time and base timestamp
    if 'accumulated' in field:
        accum_time = int(field.split("_")[2].strip('h'))
        base_timestep_nolag = int(timestep_nolag) - accum_time
        base_timestep_lag = int(timestep_lag) - accum_time
        base_timestep_lag_12h = int(timestep_lag_12h) - accum_time
    # for rainfall rate, accumulated rainfall over 6 h needs to be computed 
    elif 'rate' in field:
        accum_time = 6 
        base_timestep_nolag = int(timestep_nolag) - accum_time
        base_timestep_lag = int(timestep_lag) - accum_time
        base_timestep_lag_12h = int(timestep_lag_12h) - accum_time
    else:
        base_timestep_nolag = timestep
        base_timestep_lag = timestep
        base_timestep_lag_12h = timestep

    # get ensemble of 2d cubes (not including control member) - a 3d cube
    cubes = []
    cnt = 0
    for infile in infiles:
        # work out correct timesteps (dependend on lag)
        if ens_nums[cnt] < 18:
            timestep = timestep_nolag
            base_timestep = base_timestep_nolag
        elif ens_nums[cnt] >= 18 and ens_nums[cnt] < 36:
            timestep = timestep_lag
            base_timestep = base_timestep_lag
        else:
            timestep = timestep_lag_12h
            base_timestep = base_timestep_lag_12h
        # for accumulated precipitation, first subtract base amount
        if 'accumulated' in field:
            if base_timestep > 0:
                cube_base = get_field(infile, field, field_name_long, pressure_level, surf, base_timestep)
                cube_val = get_field(infile, field, field_name_long, pressure_level, surf, timestep)
                cube_p = iris.analysis.maths.subtract(cube_val, cube_base)
            else:
                cube_p = get_field(infile, field, field_name_long, pressure_level, surf, timestep) 
        # for rainfall rate, subtract amount from previous timestep and divide by timestep (in hr);
        # if timestep = 6 h, then no previous timestep to subtract, so just divide by timestep
        elif 'rate' in field:
            cube_val = get_field(infile, field, field_name_long, pressure_level, surf, timestep)
            if base_timestep > 0:
                cube_base = get_field(infile, field, field_name_long, pressure_level, surf, base_timestep)
                cube_p = iris.analysis.maths.subtract(cube_val, cube_base)
                cube_p = iris.analysis.maths.divide(cube_p, 6.0)
            else:
                cube_p = iris.analysis.maths.divide(cube_val, 6.0)
        else:
            cube_p = get_field(infile, field, field_name_long, pressure_level, surf, timestep)
        cubes.append(cube_p)
        cnt = cnt + 1

    # reconcile metadata before merging 
    icube = 0
    for cube in cubes:
        attrs = [attr for attr in cube.cell_methods]
        if attrs == []:
            cell_method = iris.coords.CellMethod(method='sum', coords='time')
            cube.add_cell_method(cell_method)                                                     
        cube.remove_coord('realization')
        dim = iris.coords.DimCoord([icube], standard_name='realization', units='1')
        cube.add_aux_coord(dim)
        coord_names = [coord.name() for coord in cube.coords()]
        if 'time' in coord_names:
            cube.remove_coord('time')
        if 'forecast_period' in coord_names:
            cube.remove_coord('forecast_period')
        icube = icube + 1

    cube = iris.cube.CubeList(cubes).merge_cube()

    cube_units = cubes[0].units

    # special treatment required when verifying against analysis control member (ver=1)
    if (ver_flag == True and ver == '1') or (nens == 1):
        cube0 = cube.extract(iris.Constraint(realization=0))
        cube0 = iris.util.new_axis(cube0, 'realization')
        return cube0, cube_units

    return cube, cube_units


def get_field_stat(cube, type):
    """ 
        Compute ensemble statistics 
    """

    # compute relevant ensemble statistic - resulting in 2d cube
    if type == 'mean':
        cube_stat = cube.collapsed('realization', iris.analysis.MEAN)
    elif type == 'stdev':
        cube_stat = cube.collapsed('realization', iris.analysis.STD_DEV)
    elif type == 'var':
        cube_stat = cube.collapsed('realization', iris.analysis.VARIANCE)
    elif type[0:4] == 'prob':
       cube_stat = cube.collapsed('realization', iris.analysis.PROPORTION,
                                  function=lambda values: values <= thresh)
    elif type[0:15] == 'exceedance_prob':
       cube_stat = cube.collapsed('realization', iris.analysis.PROPORTION,
                                  function=lambda values: values > thresh)

    return cube_stat


def domain_params(domain):
    # define domain boundaries and other parameters
    if domain == 'global':
        lonmin = 0.0
        lonmax = 360.0
        latmin = -90.0
        latmax = 90.0
        res = 5.0
        shrink_factor = 0.8
        step = 15.0
    elif domain == 'regional':
        lonmin = 65.0
        lonmax = 184.625
        latmin = -65.0
        latmax = 17.125
        res = 2.5
        shrink_factor = 1.0
        step = 5.0
    elif domain == 'australia':
        lonmin = 95.0
        lonmax = 169.69
        latmin = -55.0
        latmax = 4.73
        res = 2.5
        shrink_factor = 1.0
        step = 5.0
    elif domain == 'australia_wmo':
        lonmin = 90.0
        lonmax = 180.0
        latmin = -55.0
        latmax = -10.0 
        res = 2.5
        shrink_factor = 1.0
        step = 5.0
    elif domain == 'australia_tropics':
        lonmin = 90.0
        lonmax = 180.0
        latmin = -15.0
        latmax = 15.0
        res = 2.5
        shrink_factor = 1.0
        step = 5.0
    elif domain == 'australia_only':
        lonmin = 112.0
        lonmax = 156.0
        latmin = -45.0
        latmax = -10.0
        res = 2.5
        shrink_factor = 1.0
        step = 5.0
    elif domain == 'greenland':
        lonmin = -70.0
        lonmax = -10.0
        latmin = 60.0
        latmax = 90.0
        res = 2.5
        shrink_factor = 0.8 
        step = 5.0
    elif domain == 'north_europe':
        lonmin = 0.0
        lonmax = 60.0
        latmin = 60.0
        latmax = 90.0
        res = 2.5
        shrink_factor = 0.8
        step = 5.0
    elif domain == 'asia_wmo':
        lonmin = 60.0
        lonmax = 145.0
        latmin = 25.0
        latmax = 65.0
        res = 2.5
        shrink_factor = 1.0
        step = 5.0
    elif domain == 'tropics':
        lonmin = 0.0
        lonmax = 360.0
        latmin = -20.0
        latmax = 20.0
        res = 2.5
        shrink_factor = 1.0
        step = 5.0
    elif domain == 'northern_hemisphere':
        lonmin = 0.0
        lonmax = 360.0
        latmin = 0.0
        latmax = 90.0
        res = 2.5
        shrink_factor = 1.0
        step = 5.0
    elif domain == 'southern_hemisphere':
        lonmin = 0.0
        lonmax = 360.0
        latmin = -90.0
        latmax = 0.0
        res = 2.5
        shrink_factor = 1.0
        step = 5.0
    elif domain == 'northern_extratropics':
        lonmin = 0.0
        lonmax = 360.0
        latmin = 20.0
        latmax = 90.0
        res = 2.5
        shrink_factor = 1.0
        step = 5.0
    elif domain == 'southern_extratropics':
        lonmin = 0.0
        lonmax = 360.0
        latmin = -90.0
        latmax = -20.0
        res = 2.5
        shrink_factor = 1.0
        step = 5.0

    return lonmin, lonmax, latmin, latmax, res, shrink_factor, step


def field_params(field):
    # relate specified field name to data-specific names and other params 
    field_name = ''
    field_name_long = ''
    surf = False
    if field == 'uwind':
        field_name = 'x_wind'
        field_name_long = 'x_wind'
        surf = False
        ntimes = 1
    if field == 'vwind':
        field_name = 'y_wind'
        field_name_long = 'y_wind'
        surf = False
        ntimes = 1
    if field == 'temperature':
        field_name = 'air_temperature'
        field_name_long = 'air_temperature'
        surf = False
        ntimes = 1
    elif field == 'geopotential_height':
        field_name = 'geopotential_height'
        field_name_long = 'geopotential_height'
        surf = False
        ntimes  = 1
    elif field == 'relative_humidity':
        field_name = 'relative_humidity'
        field_name_long = 'relative_humidity'
        surf = False
    elif field == 'uwind_10m':
        field_name = 'x_wind'
        field_name_long = 'x_wind'
        surf = True
    elif field == 'vwind_10m':
        field_name = 'y_wind'
        field_name_long = 'y_wind'
        surf = True
    elif field == 'temperature_2m':
        field_name = 'air_temperature'
        field_name_long = 'air_temperature'
        surf = True
    elif field == 'mslp':
        field_name = 'air_pressure_at_sea_level'
        field_name_long = 'air_pressure_at_sea_level'
        surf = True
    elif field == 'dew_temperature_2m':
        field_name = 'dew_point_temperature'
        field_name_long = 'dew_point_temperature'
        surf = True
    elif 'accumulated_precipitation' in field:
        field_name = 'unknown'
        field_name_long = 'unknown'
        surf = True
    elif field == 'high_cloud':
        field_name = 'high_type_cloud_area_fraction'
        field_name_long = 'high_type_cloud_area_fraction'
        surf = True
    elif field == 'middle_cloud':
        field_name = 'medium_type_cloud_area_fraction'
        field_name_long = 'medium_type_cloud_area_fraction'
        surf = True
    elif field == 'low_cloud':
        field_name = 'low_type_cloud_area_fraction'
        field_name_long = 'low_type_cloud_area_fraction'
        surf = True
    elif field == 'total_cloud':
        field_name = 'cloud_area_fraction_assuming_maximum_random_overlap'
        field_name_long = 'cloud_area_fraction_assuming_maximum_random_overlap'
        surf = True
    elif field == 'wind_10m':
        surf = True
    elif field == 'precipitation_rate':
        field_name = 'unknown'
        field_name_long = 'unknown'
        surf = True

    return field_name, field_name_long, surf


def get_data_path(suite_name, cycle, timestep, field_name, surf):

    # compute lagged cycle (-6 h):
    ta = datetime.datetime.strptime(cycle, "%Y%m%dT%H%MZ")
    tlag = ta - datetime.timedelta(6.0/24.0)
    cycle_lag = tlag.strftime("%Y%m%dT%H%MZ")

    # compute lagged cycle (-12 h):
    ta = datetime.datetime.strptime(cycle, "%Y%m%dT%H%MZ")
    tlag = ta - datetime.timedelta(12.0/24.0)
    cycle_lag_12h = tlag.strftime("%Y%m%dT%H%MZ")

    # ensemble members
    first = 0
    last = first + nens
  
    # identify suitable paths + file names and put in list
    infile = []
    ens_nums = []
    for iens in range(first, last):
        if iens < 18:
        # path to data file
            dir = "/scratch/dp9/mjz548/marsdata/" + suite_name + "/" + cycle \
                + "/" + str(iens)
            if field_name == 'unknown':
                flnm = "data_precip.grib"
            elif surf == True:
                flnm = "data_sfc.grib"
            else:
                flnm = "data.grib"
        elif iens >= 18 and iens < 36:
        # path to data file
            dir = "/scratch/dp9/mjz548/marsdata/" + suite_name + "/" + cycle_lag \
                + "/" + str(iens-18)
            if field_name == 'unknown':
                flnm = "data_precip.grib"
            elif surf == True:
                flnm = "data_sfc_lag6.grib"
            else:
                flnm = "data_lag6.grib"
        else:
        # path to data file
            dir = "/scratch/dp9/mjz548/marsdata/" + suite_name + "/" + cycle_lag_12h \
                + "/" + str(iens-36)
            if field_name == 'unknown':
                flnm = "data_precip.grib"
            elif surf == True:
                flnm = "data_sfc_lag12.grib"
            else:
                flnm = "data_lag12.grib"
        filepath = os.path.join(dir, flnm)
        if os.path.isfile(filepath) == True:
            infile.append(filepath)
            ens_nums.append(iens)

    return infile, ens_nums


def find_anal(field, cycle, timestep):
    # generally, truth taken to be at +6h but for accumulated precip may need longer lead time
    if 'accumulated_precipitation'in field:
        lead_time = int(field.split("_")[2].strip('h'))
    else:
        lead_time = 6
    # find correct analysis cycle and timestep
    t0 = datetime.datetime.strptime(cycle, "%Y%m%dT%H%MZ")
    ta = t0 + datetime.timedelta((float(timestep)-float(lead_time))/24.0)
    anal_cycle = ta.strftime("%Y%m%dT%H%MZ")
    anal_timestep = lead_time

    return anal_cycle, anal_timestep 
    

def extract_data(suite_name, cycle, field, pressure_level, surf, field_name,
                 field_name_long, timestep, lonmin, lonmax, latmin, latmax, res):
    """
        Extracts data cubes from relevant files as specified by user
    """
    global u_norm, v_norm
    # read selected field
    if field == 'wind':
        # read u wind
        uwind, u_units = get_ens_field(suite_name, cycle, field, 'x_wind', 'x_wind', pressure_level,
                                       surf, timestep)
        # read v wind
        vwind, v_units = get_ens_field(suite_name, cycle, field, 'y_wind', 'y_wind', pressure_level,
                                       surf, timestep)
        # Create a cube containing the wind speed.
        windspeed = (uwind ** 2 + vwind ** 2) ** 0.5
        windspeed.rename("windspeed")
        field_cube = windspeed
        # find units
        field_units = str(u_units)
        # interpolate to sparse grid (to draw vector arrows)
        nlon = int(round((lonmax - lonmin)/res))
        nlat = int(round((latmax - latmin)/res))
        sample_points = [('longitude', np.linspace(lonmin, lonmax, nlon)),
                         ('latitude',  np.linspace(latmin, latmax, nlat))]
        uwind_intp = uwind.interpolate(sample_points, iris.analysis.Linear())
        vwind_intp = vwind.interpolate(sample_points, iris.analysis.Linear())
        windspeed_intp = windspeed.interpolate(sample_points,
                                               iris.analysis.Linear())
        # Normalise the data for uniform arrow size
        u_norm = uwind_intp / windspeed_intp
        v_norm = vwind_intp / windspeed_intp
    elif field == 'wind_10m':
        surf = True
        # read u wind
        uwind, u_units = get_ens_field(suite_name, cycle, field, 'x_wind', 'x_wind', pressure_level,
                                       surf, timestep)
        # read v wind
        vwind, v_units = get_ens_field(suite_name, cycle, field, 'y_wind', 'y_wind', pressure_level,
                                       surf, timestep)
        # make sure u, v on same grid
        vwind_r = vwind.regrid(uwind, iris.analysis.Linear())
        # Create a cube containing the wind speed.
        windspeed = (uwind ** 2 + vwind_r ** 2) ** 0.5
        windspeed.rename("windspeed")
        field_cube = windspeed
        # find units
        field_units = str(u_units)
        # interpolate to sparse grid (to draw vector arrows)
        nlon = int(round((lonmax - lonmin)/res))
        nlat = int(round((latmax - latmin)/res))
        sample_points = [('longitude', np.linspace(lonmin, lonmax, nlon)),
                         ('latitude',  np.linspace(latmin, latmax, nlat))]
        uwind_intp = uwind.interpolate(sample_points, iris.analysis.Linear())
        vwind_intp = vwind.interpolate(sample_points, iris.analysis.Linear())
        windspeed_intp = windspeed.interpolate(sample_points,
                                               iris.analysis.Linear())
        # Normalise the data for uniform arrow size
        u_norm = uwind_intp / windspeed_intp
        v_norm = vwind_intp / windspeed_intp
    else:
        # read general field specified by stash string
        field_cube, field_units = get_ens_field(suite_name, cycle, field, field_name,
                                                field_name_long, pressure_level, surf, timestep)
        # find units
        if field == 'geopotential_height':
            field_units = 'm'
        elif 'accumulated_precipitation' in field:
            field_units = 'mm'
        elif field == 'precipitation_rate':
            field_units = 'mm/h'
        elif field == 'total_cloud':
            field_units = '%'
        else:
            field_units = str(field_units)

    return field_cube, field_units 


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def plot_data(field_cube, field_units, spec_range, shrink_factor, domain,
              lonmin, lonmax, latmin, latmax, type, pressure_level, timestep,
              suite_name, field, surf, cycle):
    """ plot gridded data and save to file
    """
    # plot contours
    if field == 'wind' or field == 'wind_10m':
        extend_opt = 'max'
    else:
        extend_opt = 'both'
    if spec_range == "auto":
        plot = iplt.contourf(field_cube, nlevs, extend=extend_opt, cmap=mpl.cm.viridis)
    else:
        minval = float(spec_range.split(':')[0])
        maxval = float(spec_range.split(':')[1])
        clevs = np.arange(minval, maxval, nlevs+1)
        clevs = np.asarray([minval + (maxval-minval)*i/nlevs for i in range(nlevs+1)])
        plot = iplt.contourf(field_cube, levels=clevs, extend=extend_opt, cmap=mpl.cm.viridis)

    # plot coastlines
    plt.gca().coastlines(color='grey')
    gl = plt.gca().gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False

    # colour bar
    cbar = plt.colorbar(plot, location='right', shrink=shrink_factor)

    # plot wind arrows if needed
    if field == 'wind' or field == 'wind_10m':
        iplt.quiver(u_norm_mean, v_norm_mean, pivot="middle", color='black')

    # truncate plot according to chosen domain
    if domain != 'global':
        plt.xlim(right=lonmax, left=lonmin)
        plt.ylim(top=latmax, bottom=latmin)

    # type string - special handling when type is 'prob'
    if type[0:4] == 'prob':
        field_str = field + " <= " + type[4:] + " " + field_units + "\n" + " ensemble prob"
    elif type[0:15] == 'exceedance_prob':
        field_str = "\n" + field + " > " + type[15:] + " " + field_units + " ensemble prob"
    else:
        field_str = field + " (" + field_units + ")" + " " + "\n" + "ensemble " + type

    # title string
    if surf == False:
        level_str = str(pressure_level).replace(".0", "") + "mb"
    else:
        level_str = ""
    if int(timestep) == 0:
        title_str = cycle + " " + level_str + " " + field_str
    else:
        title_str = cycle + " +" + timestep.lstrip("0") + "h " + level_str + " " + \
            field_str
    plt.title(title_str)

    # save figure
    if surf == False:
        level_str = str(pressure_level).replace(".0", "mb") + "_"
    else:
        level_str = ""
    outflnm = field + "." + suite_name + "." + cycle + "." + level_str + timestep + "h" + \
        "." + type
    out_path = os.path.join(outpath, outflnm)
    plt.savefig(out_path, dpi=300, format='png')

    qplt.show()


def calc_stats(model, obs):
    """ Computes spatial mean, stdev, and rmse of (model-obs)"""
    err_cube = iris.analysis.maths.subtract(model, obs)
    coords = ('longitude', 'latitude')
    mean_err = err_cube.collapsed(coords, iris.analysis.MEAN)
    stdev_err = err_cube.collapsed(coords, iris.analysis.STD_DEV)
    rmse = err_cube.collapsed(coords, iris.analysis.RMS)
    maxval = model.collapsed(coords, iris.analysis.MAX)
    minval = model.collapsed(coords, iris.analysis.MIN)
    patcor = stats.pearsonr(model, obs, coords)

    return mean_err.data, stdev_err.data, rmse.data, maxval.data, patcor.data, minval.data


def calc_av_stdev(var_cube):
    """ Computes spatial average standard deviation from ensemble variance"""
    coords = ('longitude', 'latitude')
    av_var = var_cube.collapsed(coords, iris.analysis.MEAN)
    av_stdev = math.sqrt(av_var.data)

    return av_stdev


def calc_av_stdev2(stdev_cube):
    """ Computes spatial average standard deviation from ensemble standard deviation"""
    coords = ('longitude', 'latitude')
    av_stdev = stdev_cube.collapsed(coords, iris.analysis.MEAN)

    return av_stdev.data


def calc_gen_brier(model, obs):
    """ Computes generalised Brier score """
    err_cube = iris.analysis.maths.subtract(model, obs)
    std_term = iris.analysis.maths.exponentiate(err_cube, 2)
    one_minus_obs = iris.analysis.maths.add(obs, -1.0)
    new_term = iris.analysis.maths.multiply(obs, one_minus_obs)
    comb_term = iris.analysis.maths.subtract(std_term, new_term)
    coords = ('longitude', 'latitude')
    gen_brier = comb_term.collapsed(coords, iris.analysis.MEAN)

    return gen_brier.data


def minmax(field, level, cycle, timestep):
    """Read field min max values from file for use in RPS calculation
    """
    with open('minmax.dat', "r") as file:
        lines = [line.rstrip() for line in file]
    search_string  = " ".join([field, str(level), cycle, timestep])
    indx = [lines.index(line) for line in lines if search_string in line][0]
    minval = float(lines[indx].split(" ")[4])
    maxval = float(lines[indx].split(" ")[5])

    return minval, maxval


def calc_orog_mask(level):
    # read orography (UM ancil file)
    fin="/home/548/mjz548/store/mule/qrparm.orog.nc"
    loaded_cubes = iris.fileformats.netcdf.load_cubes(fin)
    all_cubes = list(loaded_cubes)
    field_name = 'surface_altitude'
    for cube in all_cubes:
        if cube.name() == field_name:
            orog_orig = cube

    # interpolate to verification grid
    orog_new = extract_domain(orog_orig)

    # specify standard heights at a pressure level
    if abs(level-1000.0) < 0.1:
        std_ht = 111.0
    elif abs(level-925.0) < 0.1:
        std_ht = 762.0
    elif abs(level-850.0) < 0.1:
        std_ht = 1458.0
    elif abs(level-700.0) < 0.1:
        std_ht = 3013.0
    elif abs(level-500.0) < 0.1:
        std_ht = 5576.0
    elif abs(level-400.0) < 0.1:
        std_ht = 7187.0
    elif abs(level-300.0) < 0.1:
        std_ht = 9166.0
    elif abs(level-250.0) < 0.1:
        std_ht = 10366.0
    elif abs(level-200.0) < 0.1:
        std_ht = 11787.0

    orog_arr = np.array(orog_new.data)
    masked_field = np.where(orog_arr>std_ht, True, False)

    return masked_field


def main():
    # Read user selections 
    suite_name = sys.argv[1]
    cycle = sys.argv[2]
    field = sys.argv[3]
    pressure_level = float(sys.argv[4])
    timestep = sys.argv[5]
    domain = sys.argv[6]
    spec_range = sys.argv[7]
    type = sys.argv[8]
    ver_option = sys.argv[9]
    global outpath
    outpath = sys.argv[10]
    global nens
    nens = int(sys.argv[11])

    # verification grid resolution
    global ver_res
    ver_res = float(sys.argv[12]) 

    global ver
    ver = ver_option.split('=')[1]

    # flag to indicate whether cube is computed for forecast or analysis
    global ver_flag
    ver_flag = False

    # obtain domain boundaries
    global lonmin, lonmax, latmin, latmax
    lonmin, lonmax, latmin, latmax, res, shrink_factor, step = domain_params(domain)
 
    # obtain stash value and other parameters related to chosen field
    field_name, field_name_long, surf = field_params(field)

    # obtain correct path to data files
    infile, ens_nums = get_data_path(suite_name, cycle, timestep, field_name, surf)

    # get forecast ensemble members
    field_cube, field_units = extract_data(suite_name, cycle, field, pressure_level, surf, field_name,
                                           field_name_long, timestep, lonmin, lonmax, latmin, latmax, res)

    # by default compute prob threshold from user input
    global thresh
    if type[0:4] == 'prob':
        thresh = float(type[4:])
    if type[0:15] == 'exceedance_prob':
        thresh = float(type[15:])

    # calculate orographic mask
    global orog_mask
    orog_mask = calc_orog_mask(pressure_level)

    # apply mask to forecast fields
    field_cube = iris.util.mask_cube(field_cube, orog_mask)

    # compute verification stats relative to analysis if required
    # 0 - no ver; 1 - control member as truth; 2 - whole ensemble as truth 
    # (ensemble mean or prob of "subceedance" as truth)
    if ver != '0':
        # obtain correct cycle (and timestep) corresponding to analysis:
        anal_cycle, anal_timestep = find_anal(field, cycle, timestep)

        # obtain path to analysis data
        anal_infile, ens_nums = get_data_path(suite_name, anal_cycle, anal_timestep, field_name, surf)

        # get analysis ensemble members
        ver_flag = True
        anal_field_cube, field_units = \
            extract_data(suite_name, anal_cycle, field, pressure_level, surf, field_name,
                         field_name_long, anal_timestep, lonmin, lonmax, latmin, latmax, res)

        # apply orographic mask to analysis
        anal_field_cube = iris.util.mask_cube(anal_field_cube, orog_mask)

        # mean field
        mean_field_cube = get_field_stat(field_cube, 'mean')
        mean_anal_field_cube =  get_field_stat(anal_field_cube, 'mean')
        mean_err, stdev_err, rmse, maxfld, patcor, minfld = \
            calc_stats(mean_field_cube, mean_anal_field_cube)

        # var field
        var_field_cube = get_field_stat(field_cube, 'var')
        av_stdev = calc_av_stdev(var_field_cube)

        # prob field
        coords = ('longitude', 'latitude')

        # write RPS threshold values to file
        if suite_name == 'ge3':
            # determine min, max from mean field and write to file for use by ge4 run 
            minval = (mean_anal_field_cube.collapsed(coords, iris.analysis.MIN)).data
            maxval = (mean_anal_field_cube.collapsed(coords, iris.analysis.MAX)).data
            file_path = "./minmax." + cycle + "." + domain + "." + suite_name \
                + "." + field + "." + str(pressure_level).replace(".0", "") +  "." + timestep
            with open(file_path, "w") as minmax_file:
                print(field, pressure_level, cycle, timestep, minval, maxval, file=minmax_file)
        else:
            # read min, max values from ge3 file
            minval, maxval = minmax(field, pressure_level, cycle, timestep)
        # compute thresholds from min, max values
        thresholds = [minval+(maxval-minval)*(i+1)/(nthresh+1) for i in range(nthresh)]
        rps = 0.0 
        for thresh in thresholds:
            prob_field_cube = get_field_stat(field_cube, 'prob')
            prob_anal_field_cube = get_field_stat(anal_field_cube, 'prob')
            rps = rps + calc_gen_brier(prob_field_cube, prob_anal_field_cube)
        rps = rps/len(thresholds)

        # write stats to file
        with open(outpath, "w") as ver_file:
            print(mean_err, stdev_err, rmse, av_stdev, rps, maxfld, patcor, 
                  minfld, file=ver_file)
 
        # write RPS threshold values to file
        if suite_name == 'ge3':
            file_path = "./minmax." + cycle + "." + domain + "." + suite_name \
                + "." + field + "." + str(pressure_level).replace(".0", "") +  "." + timestep
            with open(file_path, "w") as minmax_file:
                print(field, pressure_level, cycle, timestep, minval, maxval, file=minmax_file)
    else:
        # obtain gridded data in accordance with user specs (ver = 0; no verification)
        agg_field_cube = get_field_stat(field_cube, type)

        # obtain mean field wind vectors if needed
        if field == 'wind' or field == 'wind_10m':
            global u_norm_mean
            global v_norm_mean
            u_norm_mean = get_field_stat(u_norm, type)
            v_norm_mean = get_field_stat(v_norm, type)

        # plot gridded data
        plot_data(agg_field_cube, field_units, spec_range, shrink_factor, domain,
                  lonmin, lonmax, latmin, latmax, type, pressure_level, timestep,
                  suite_name, field, surf, cycle)


if __name__ == "__main__":
    main()
