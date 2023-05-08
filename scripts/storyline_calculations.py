#I AM HUMAN
# I AM ROBOT
# I AM GAIA
import xarray as xr
import numpy as np
import statsmodels.api as sm
import pandas as pd
import json 
import os
from esmvaltool.diag_scripts.shared import run_diagnostic, get_cfg, group_metadata
from sklearn import linear_model
import glob
from scipy import signal
import netCDF4
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature
from cartopy.util import add_cyclic_point
import matplotlib.path as mpath
import matplotlib as mpl
import random
# To use R packages:
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import rpy2.robjects as robjects
# Convert pandas.DataFrames to R dataframes automatically.
pandas2ri.activate()
relaimpo = importr("relaimpo")
    
#Across models regression class
class storyline_calc(object):
    """ Open the senstivity coefficients from the multiple linear regression analysis
    and do the calculations that are done in a typical storyline analysis of CMIP ensembles
    
    List of functions:
        - regression_data: load regression data, regressors and target variable
        - open_regression_coef: open regression coefficients
        - open_lmg_coef: open relative importance coefficients
        - plot_regression_lmg_map: plot relative importace maps
        - plot_regression_coef_map: plot coefficient maps
        - plot_storylines: plot storyline maps
        - evaluate model spread in the target and storyline span of the uncertainty
    """
    def __init__(self):
        self.what_is_this = 'This class is used to analyize the storyline results'
    
    def regression_data(self,variable,regressors,regressor_names):
        """Define the regression target variable 
        this is here to be edited if some opperation is needed on the DataArray
        
        :param variable: DataArray
        :return: target variable for the regression  
        """
        self.target = variable
        regressor_indices = regressors
        self.regression_y = sm.add_constant(regressors.values)
        self.regressors = regressors.values
        self.rd_num = len(regressor_names)
        self.regressor_names = regressor_names

   
    def open_regression_coef(self,path,var):
        """ Open regression coefficients and pvalues to plot
        :param path: saving path
        :return maps: list of list of coefficient maps
        :return maps_pval:  list of coefficient pvalues maps
        :return R2: map of fraction of variance
        """ 
        maps = []; maps_pval = []
        coef_maps = xr.open_dataset(path+'/'+var+'/regression_coefficients.nc')
        coef_pvalues = xr.open_dataset(path+'/'+var+'/regression_coefficients_pvalues.nc')
        maps = [coef_maps[variable] for variable in self.regressor_names]
        maps_pval = [coef_pvalues[variable] for variable in self.regressor_names]
        R2 = xr.open_dataset(path+'/'+var+'/R2.nc')
        return maps, maps_pval, R2    

    def open_lmg_coef(self,path,var):
        """ Open regression coefficients and pvalues to plot
        :param path: saving path
        :return maps: list of list of coefficient maps
        :return maps_pval:  list of coefficient pvalues maps
        :return R2: map of fraction of variance
        """ 
        maps = []; maps_pval = []
        coef_maps = xr.open_dataset(path+'/'+var+'/regression_coefficients_relative_importance.nc')
        coef_pvalues = xr.open_dataset(path+'/'+var+'/regression_coefficients_pvalues.nc')
        maps = [coef_maps[variable] for variable in self.regressor_names[1:]]
        maps_pval = [coef_pvalues[variable] for variable in self.regressor_names]
        R2 = xr.open_dataset(path+'/'+var+'/R2.nc')
        return maps, maps_pval, R2    
    
    def plot_regression_lmg_map(self,path,var,output_path):
        """ Plots figure with all of 
        :param regressor_names: list with strings naming the independent variables
        :param path: saving path
        :return: none
        """
        maps, maps_pval, R2 = self.open_lmg_coef(path,var)
        cmapU850 = mpl.colors.ListedColormap(['darkblue','navy','steelblue','lightblue',
                                            'lightsteelblue','white','white','mistyrose',
                                            'lightcoral','indianred','brown','firebrick'])
        cmapU850.set_over('maroon')
        cmapU850.set_under('midnightblue')
        path_era = '/datos/ERA5/mon'
        u_ERA = xr.open_dataset(path_era+'/era5.mon.mean.nc')
        u_ERA = u_ERA.u.sel(lev=850).sel(time=slice('1979','2018'))
        u_ERA = u_ERA.groupby('time.season').mean(dim='time').sel(season='DJF')

        fig_coef = plt.figure(figsize=(20, 16),dpi=100,constrained_layout=True)
        projection_stereo = ccrs.SouthPolarStereo(central_longitude=300)
        projection_plate = ccrs.PlateCarree(180)
        data_crs = ccrs.PlateCarree()
        for k in range(self.rd_num-1):
            lat = maps[k].lat
            lon = np.linspace(0,360,len(maps[k].lon))
            var_c, lon_c = add_cyclic_point(maps[k].values,lon)
            #SoutherHemisphere Stereographic
            if var == 'ua':
                ax = plt.subplot(3,3,k+1,projection=projection_stereo)
                ax.set_extent([0,359.9, -90, 0], crs=data_crs)
                theta = np.linspace(0, 2*np.pi, 100)
                center, radius = [0.5, 0.5], 0.5
                verts = np.vstack([np.sin(theta), np.cos(theta)]).T
                circle = mpath.Path(verts * radius + center)
                ax.set_boundary(circle, transform=ax.transAxes)
            elif var == 'sst':
                ax = plt.subplot(3,3,k+1,projection=projection_plate)
            else: 
                ax = plt.subplot(3,3,k+1,projection=projection_stereo)
            clevels = np.arange(0,40,2)
            im=ax.contourf(lon_c, lat, var_c*100,clevels,transform=data_crs,cmap='OrRd',extend='both')
            cnt=ax.contour(u_ERA.lon,u_ERA.lat, u_ERA.values,levels=[8],transform=data_crs,linewidths=1.2, colors='black', linestyles='-')
            plt.clabel(cnt,inline=True,fmt='%1.0f',fontsize=8)
            if maps_pval[k+1].min() < 0.05: 
                levels = [maps_pval[k+1].min(),0.05,maps_pval[k+1].max()]
                ax.contourf(maps_pval[k+1].lon, lat, maps_pval[k+1].values,levels, transform=data_crs,levels=levels, hatches=["...", " "], alpha=0)
            elif maps_pval[k+1].min() < 0.10:
                levels = [maps_pval[k+1].min(),0.10,maps_pval[k+1].max()]
                ax.contourf(maps_pval[k+1].lon, lat, maps_pval[k+1].values,levels, transform=data_crs,levels=levels, hatches=["...", " "], alpha=0)
            else:
                print('No significant values for ',self.regressor_names[k+1]) 
            plt.title(self.regressor_names[k+1],fontsize=18)
            ax.add_feature(cartopy.feature.COASTLINE,alpha=.5)
            ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
            ax.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
            ax.set_extent([-180, 180, -90, -25], ccrs.PlateCarree())
        plt1_ax = plt.gca()
        left, bottom, width, height = plt1_ax.get_position().bounds
        if var == 'ua':
            colorbar_axes1 = fig_coef.add_axes([left+0.5, bottom, 0.01, height*2])
        elif var == 'sst':
            colorbar_axes1 = fig_coef.add_axes([left+0.3, bottom, 0.01, height*2])    
        else:
            colorbar_axes1 = fig_coef.add_axes([left+0.5, bottom, 0.01, height*2])
        cbar = fig_coef.colorbar(im, colorbar_axes1, orientation='vertical')
        cbar.set_label('relative importance',fontsize=14) #rotation = radianes
        cbar.ax.tick_params(axis='both',labelsize=14)
            
        plt.subplots_adjust(bottom=0.2, right=.95, top=0.8)
        if var == 'ua':
            plt.savefig(output_path+'/regression_coefficients_relative_importance_u850',bbox_inches='tight')
        elif var == 'sst':
            plt.savefig(output_path+'/regression_coefficients_relative_importance_sst',bbox_inches='tight')
        else:
            plt.savefig(output_path+'/regression_coefficients_relative_importance_XXX',bbox_inches='tight')   
        plt.clf

        return fig_coef


    def plot_regression_coef_map(self,path,var,output_path):
        """ Plots figure with all of 
        :param regressor_names: list with strings naming the independent variables
        :param path: saving path
        :return: none
        """
        maps, maps_pval, R2 = self.open_regression_coef(path,var)
        cmapU850 = mpl.colors.ListedColormap(['darkblue','navy','steelblue','lightblue',
                                            'lightsteelblue','white','white','mistyrose',
                                            'lightcoral','indianred','brown','firebrick'])
        cmapU850.set_over('maroon')
        cmapU850.set_under('midnightblue')
        path_era = '/datos/ERA5/mon'
        u_ERA = xr.open_dataset(path_era+'/era5.mon.mean.nc')
        u_ERA = u_ERA.u.sel(lev=850).sel(time=slice('1979','2018'))
        u_ERA = u_ERA.groupby('time.season').mean(dim='time').sel(season='DJF')

        fig_coef = plt.figure(figsize=(20, 16),dpi=100,constrained_layout=True)
        projection_stereo = ccrs.SouthPolarStereo(central_longitude=300)
        projection_plate = ccrs.PlateCarree(180)
        data_crs = ccrs.PlateCarree()
        for k in range(self.rd_num):
            lat = maps[k].lat
            lon = np.linspace(0,360,len(maps[k].lon))
            var_c, lon_c = add_cyclic_point(maps[k].values,lon)
            #SoutherHemisphere Stereographic for winds
            if var == 'ua':
                ax = plt.subplot(3,3,k+1,projection=projection_stereo)
                ax.set_extent([0,359.9, -90, 0], crs=data_crs)
                theta = np.linspace(0, 2*np.pi, 100)
                center, radius = [0.5, 0.5], 0.5
                verts = np.vstack([np.sin(theta), np.cos(theta)]).T
                circle = mpath.Path(verts * radius + center)
                ax.set_boundary(circle, transform=ax.transAxes)
            #Plate Carree map for SST
            elif var == 'sst':
                ax = plt.subplot(3,3,k+1,projection=projection_plate)
            else: 
                ax = plt.subplot(3,3,k+1,projection=projection_stereo)
            if k == 0:
                if var == 'ua':
                    clevels = np.arange(-2,2.5,0.5)
                    im0=ax.contourf(lon_c, lat, var_c,clevels,transform=data_crs,cmap=cmapU850,extend='both')
                elif var == 'sst':
                    clevels = np.arange(-1,1.25,0.25)
                    im0=ax.contourf(lon_c, lat, var_c,clevels,transform=data_crs,cmap=cmapU850,extend='both')
                else:
                    clevels = np.arange(-2,2.5,0.5)
            else:
                clevels = np.arange(-.6,.7,0.1)
                im=ax.contourf(lon_c, lat, var_c,clevels,transform=data_crs,cmap=cmapU850,extend='both')
            cnt=ax.contour(u_ERA.lon,u_ERA.lat, u_ERA.values,levels=[8],transform=data_crs,linewidths=1.2, colors='black', linestyles='-')
            plt.clabel(cnt,inline=True,fmt='%1.0f',fontsize=8)
            if maps_pval[k].min() < 0.05: 
                levels = [maps_pval[k].min(),0.05,maps_pval[k].max()]
                ax.contourf(maps_pval[k].lon, lat, maps_pval[k].values,levels, transform=data_crs,levels=levels, hatches=["...", " "], alpha=0)
            elif maps_pval[k].min() < 0.10:
                levels = [maps_pval[k].min(),0.10,maps_pval[k].max()]
                ax.contourf(maps_pval[k].lon, lat, maps_pval[k].values,levels, transform=data_crs,levels=levels, hatches=["...", " "], alpha=0)
            else:
                print('No significant values for ',self.regressor_names[k]) 
            plt.title(self.regressor_names[k],fontsize=18)
            ax.add_feature(cartopy.feature.COASTLINE,alpha=.5)
            ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
            ax.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
            if var == 'ua':
                ax.set_extent([-180, 180, -90, -25], ccrs.PlateCarree())
            elif var == 'sst':
                ax.set_extent([-60, 220, -80, 40], ccrs.PlateCarree(central_longitude=180))
            else: 
                ax.set_extent([-60, 220, -80, 40], ccrs.PlateCarree(central_longitude=180))
            
        plt1_ax = plt.gca()
        left, bottom, width, height = plt1_ax.get_position().bounds
        if var == 'ua':
            colorbar_axes1 = fig_coef.add_axes([left+0.28, bottom, 0.01, height*2])
            colorbar_axes2 = fig_coef.add_axes([left+0.36, bottom, 0.01, height*2])
        elif var == 'sst':
            colorbar_axes1 = fig_coef.add_axes([left+0.3, bottom, 0.01, height*3])
            colorbar_axes2 = fig_coef.add_axes([left+0.38, bottom, 0.01, height*3])
        cbar = fig_coef.colorbar(im0, colorbar_axes1, orientation='vertical')
        cbar2 = fig_coef.colorbar(im, colorbar_axes2, orientation='vertical')
        if var == 'ua':
            cbar.set_label('m/s/std(rd)',fontsize=14) #rotation = radianes
            cbar2.set_label('m/s/std(rd)',fontsize=14) #rotation = radianes
        elif var == 'sst':
            cbar.set_label('K/std(rd)',fontsize=14) #rotation = radianes
            cbar2.set_label('K/std(rd)',fontsize=14) #rotation = radianes
        else:
            cbar.set_label('X/std(rd)',fontsize=14) #rotation = radianes
            cbar2.set_label('X/std(rd)',fontsize=14) #rotation = radianes
        cbar.ax.tick_params(axis='both',labelsize=14)
        cbar2.ax.tick_params(axis='both',labelsize=14)
            
        plt.subplots_adjust(bottom=0.2, right=.95, top=0.8)
        if var == 'ua':
            plt.savefig(output_path+'/regression_coefficients_u850',bbox_inches='tight')
        elif  var == 'sst':
            plt.savefig(output_path+'/regression_coefficients_sst',bbox_inches='tight')
        else:
            plt.savefig(output_path+'/regression_coefficients_unknown_var',bbox_inches='tight')
        
        plt.clf

        return fig_coef

    def plot_storylines_lowGW(self,path,var,output_path):
        """ Plots figure with all of 
        :param regressor_names: list with strings naming the independent variables
        :param path: saving path
        :return: none
        """
        maps, maps_pval, R2 = self.open_regression_coef(path,var)
        storylines = [maps[0]-1.3*maps[3]+1.3*maps[1]+1.3*maps[4]+1.3*maps[5],
                      maps[0]-1.3*maps[3]+1.3*maps[1]+1.3*maps[4]-1.3*maps[5],
                      maps[0]-1.3*maps[3]+1.3*maps[1]-1.3*maps[4]+1.3*maps[5],
                      maps[0]-1.3*maps[3]+1.3*maps[1]-1.3*maps[4]-1.3*maps[5],
                      maps[0]-1.3*maps[3]-1.3*maps[1]+1.3*maps[4]+1.3*maps[5],
                      maps[0]-1.3*maps[3]-1.3*maps[1]+1.3*maps[4]-1.3*maps[5],
                      maps[0]-1.3*maps[3]-1.3*maps[1]-1.3*maps[4]+1.3*maps[5],
                      maps[0]-1.3*maps[3]-1.3*maps[1]-1.3*maps[4]-1.3*maps[5],]
        storyline_names = [' - GW + CP + TW + VB ',' - GW + CP + TW - VB ',
                           ' - GW + CP - TW + VB ',' - GW + CP - TW - VB ',
                           ' - GW - CP + TW + VB ',' - GW - CP + TW - VB ',
                           ' - GW - CP - TW + VB ',' - GW - CP - TW - VB ',]
        cmapU850 = mpl.colors.ListedColormap(['darkblue','navy','steelblue','lightblue',
                                            'lightsteelblue','white','white','mistyrose',
                                            'lightcoral','indianred','brown','firebrick'])
        cmapU850.set_over('maroon')
        cmapU850.set_under('midnightblue')
        path_era = '/datos/ERA5/mon'
        u_ERA = xr.open_dataset(path_era+'/era5.mon.mean.nc')
        u_ERA = u_ERA.u.sel(lev=850).sel(time=slice('1979','2018'))
        u_ERA = u_ERA.groupby('time.season').mean(dim='time').sel(season='DJF')

        fig_coef = plt.figure(figsize=(20, 16),dpi=100,constrained_layout=True)
        projection_stereo = ccrs.SouthPolarStereo(central_longitude=300)
        projection_plate = ccrs.PlateCarree(180)
        data_crs = ccrs.PlateCarree()
        for k in range(len(storylines)+1):
            lat = maps[0].lat
            lon = np.linspace(0,360,len(maps[0].lon))
            mem, lon_c = add_cyclic_point(maps[0].values,lon)
            lon = np.linspace(0,360,len(maps[0].lon))
            var_c, lon_c = add_cyclic_point(storylines[k-1].values,lon)
            #SoutherHemisphere Stereographic for winds
            if var == 'ua':
                ax = plt.subplot(3,3,k+1,projection=projection_stereo)
                ax.set_extent([0,359.9, -90, 0], crs=data_crs)
                theta = np.linspace(0, 2*np.pi, 100)
                center, radius = [0.5, 0.5], 0.5
                verts = np.vstack([np.sin(theta), np.cos(theta)]).T
                circle = mpath.Path(verts * radius + center)
                ax.set_boundary(circle, transform=ax.transAxes)
            #Plate Carree map for SST
            elif var == 'sst':
                ax = plt.subplot(3,3,k+1,projection=projection_plate)
            else: 
                ax = plt.subplot(3,3,k+1,projection=projection_stereo)
            if k == 0:
                if var == 'ua':
                    clevels = np.arange(-2,2.5,0.5)
                    im0=ax.contourf(lon_c, lat, mem,clevels,transform=data_crs,cmap=cmapU850,extend='both')
                elif var == 'sst':
                    clevels = np.arange(-1,1.25,0.25)
                    im0=ax.contourf(lon_c, lat, mem,clevels,transform=data_crs,cmap=cmapU850,extend='both')
                else:
                    clevels = np.arange(-2,2.5,0.5)
            else:
                if var == 'ua':
                    clevels = np.arange(-2,2.5,0.5)
                elif var == 'sst':
                    clevels = np.arange(-1,1.25,0.25)
                else:
                    clevels = np.arange(-2,2.5,0.5)
                im=ax.contourf(lon_c, lat, var_c,clevels,transform=data_crs,cmap=cmapU850,extend='both')
            cnt=ax.contour(u_ERA.lon,u_ERA.lat, u_ERA.values,levels=[8],transform=data_crs,linewidths=1.2, colors='black', linestyles='-')
            plt.clabel(cnt,inline=True,fmt='%1.0f',fontsize=8)
            if maps_pval[0].min() < 0.05: 
                levels = [maps_pval[0].min(),0.05,maps_pval[0].max()]
                ax.contourf(maps_pval[0].lon, lat, maps_pval[0].values,levels, transform=data_crs,levels=levels, hatches=["...", " "], alpha=0)
            elif maps_pval[0].min() < 0.10:
                levels = [maps_pval[0].min(),0.10,maps_pval[0].max()]
                ax.contourf(maps_pval[0].lon, lat, maps_pval[0].values,levels, transform=data_crs,levels=levels, hatches=["...", " "], alpha=0)
            else:
                print('No significant values for ',self.regressor_names[k]) 
            if k == 0:
                plt.title('MEM',fontsize=18)
            else:
                plt.title(storyline_names[k-1],fontsize=18)
            ax.add_feature(cartopy.feature.COASTLINE,alpha=.5)
            ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
            ax.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
            if var == 'ua':
                ax.set_extent([-180, 180, -90, -25], ccrs.PlateCarree())
            elif var == 'sst':
                ax.set_extent([-60, 220, -80, 40], ccrs.PlateCarree(central_longitude=180))
            else: 
                ax.set_extent([-60, 220, -80, 40], ccrs.PlateCarree(central_longitude=180))
            
        plt1_ax = plt.gca()
        left, bottom, width, height = plt1_ax.get_position().bounds
        if var == 'ua':
            colorbar_axes1 = fig_coef.add_axes([left+0.28, bottom, 0.01, height*3])
            colorbar_axes2 = fig_coef.add_axes([left+0.36, bottom, 0.01, height*3])
        elif var == 'sst':
            colorbar_axes1 = fig_coef.add_axes([left+0.3, bottom, 0.01, height*4])
            colorbar_axes2 = fig_coef.add_axes([left+0.38, bottom, 0.01, height*4])
        cbar = fig_coef.colorbar(im0, colorbar_axes1, orientation='vertical')
        cbar2 = fig_coef.colorbar(im, colorbar_axes2, orientation='vertical')
        if var == 'ua':
            cbar.set_label('m/s/std(rd)',fontsize=14) #rotation = radianes
            cbar2.set_label('m/s/std(rd)',fontsize=14) #rotation = radianes
        elif var == 'sst':
            cbar.set_label('K/std(rd)',fontsize=14) #rotation = radianes
            cbar2.set_label('K/std(rd)',fontsize=14) #rotation = radianes
        else:
            cbar.set_label('X/std(rd)',fontsize=14) #rotation = radianes
            cbar2.set_label('X/std(rd)',fontsize=14) #rotation = radianes
        cbar.ax.tick_params(axis='both',labelsize=14)
        cbar2.ax.tick_params(axis='both',labelsize=14)
            
        plt.subplots_adjust(bottom=0.2, right=.95, top=0.8)
        if var == 'ua':
            plt.savefig(output_path+'/storylines_u850_lowGW',bbox_inches='tight')
        elif  var == 'sst':
            plt.savefig(output_path+'/storylines_sst_lowGW',bbox_inches='tight')
        else:
            plt.savefig(output_path+'/storylines_unknown_var_lowGW',bbox_inches='tight')
        
        plt.clf

        return fig_coef

    def plot_storylines(self,path,var,output_path):
        """ Plots figure with all of 
        :param regressor_names: list with strings naming the independent variables
        :param path: saving path
        :return: none
        """
        maps, maps_pval, R2 = self.open_regression_coef(path,var)
        storylines = [maps[0]+1.3*maps[3]+1.3*maps[1]+1.3*maps[4]+1.3*maps[5],
                      maps[0]+1.3*maps[3]+1.3*maps[1]+1.3*maps[4]-1.3*maps[5],
                      maps[0]+1.3*maps[3]+1.3*maps[1]-1.3*maps[4]+1.3*maps[5],
                      maps[0]+1.3*maps[3]+1.3*maps[1]-1.3*maps[4]-1.3*maps[5],
                      maps[0]+1.3*maps[3]-1.3*maps[1]+1.3*maps[4]+1.3*maps[5],
                      maps[0]+1.3*maps[3]-1.3*maps[1]+1.3*maps[4]-1.3*maps[5],
                      maps[0]+1.3*maps[3]-1.3*maps[1]-1.3*maps[4]+1.3*maps[5],
                      maps[0]+1.3*maps[3]-1.3*maps[1]-1.3*maps[4]-1.3*maps[5],]
        storyline_names = [' + GW + CP + TW + VB ',' + CP + TW - VB ',
                           ' + CP - TW + VB ',' + GW + CP - TW - VB ',
                           ' + GW - CP + TW + VB ',' + GW - CP + TW - VB ',
                           ' + GW - CP - TW + VB ',' + GW - CP - TW - VB ',]
        cmapU850 = mpl.colors.ListedColormap(['darkblue','navy','steelblue','lightblue',
                                            'lightsteelblue','white','white','mistyrose',
                                            'lightcoral','indianred','brown','firebrick'])
        cmapU850.set_over('maroon')
        cmapU850.set_under('midnightblue')
        path_era = '/datos/ERA5/mon'
        u_ERA = xr.open_dataset(path_era+'/era5.mon.mean.nc')
        u_ERA = u_ERA.u.sel(lev=850).sel(time=slice('1979','2018'))
        u_ERA = u_ERA.groupby('time.season').mean(dim='time').sel(season='DJF')

        fig_coef = plt.figure(figsize=(20, 16),dpi=100,constrained_layout=True)
        projection_stereo = ccrs.SouthPolarStereo(central_longitude=300)
        projection_plate = ccrs.PlateCarree(180)
        data_crs = ccrs.PlateCarree()
        for k in range(len(storylines)+1):
            lat = maps[0].lat
            lon = np.linspace(0,360,len(maps[0].lon))
            mem, lon_c = add_cyclic_point(maps[0].values,lon)
            lon = np.linspace(0,360,len(maps[0].lon))
            var_c, lon_c = add_cyclic_point(storylines[k-1].values,lon)
            #SoutherHemisphere Stereographic for winds
            if var == 'ua':
                ax = plt.subplot(3,3,k+1,projection=projection_stereo)
                ax.set_extent([0,359.9, -90, 0], crs=data_crs)
                theta = np.linspace(0, 2*np.pi, 100)
                center, radius = [0.5, 0.5], 0.5
                verts = np.vstack([np.sin(theta), np.cos(theta)]).T
                circle = mpath.Path(verts * radius + center)
                ax.set_boundary(circle, transform=ax.transAxes)
            #Plate Carree map for SST
            elif var == 'sst':
                ax = plt.subplot(3,3,k+1,projection=projection_plate)
            else: 
                ax = plt.subplot(3,3,k+1,projection=projection_stereo)
            if k == 0:
                if var == 'ua':
                    clevels = np.arange(-2,2.5,0.5)
                    im0=ax.contourf(lon_c, lat, mem,clevels,transform=data_crs,cmap=cmapU850,extend='both')
                elif var == 'sst':
                    clevels = np.arange(-1,1.25,0.25)
                    im0=ax.contourf(lon_c, lat, mem,clevels,transform=data_crs,cmap=cmapU850,extend='both')
                else:
                    clevels = np.arange(-2,2.5,0.5)
            else:
                if var == 'ua':
                    clevels = np.arange(-2,2.5,0.5)
                elif var == 'sst':
                    clevels = np.arange(-1,1.25,0.25)
                else:
                    clevels = np.arange(-2,2.5,0.5)
                im=ax.contourf(lon_c, lat, var_c,clevels,transform=data_crs,cmap=cmapU850,extend='both')
            cnt=ax.contour(u_ERA.lon,u_ERA.lat, u_ERA.values,levels=[8],transform=data_crs,linewidths=1.2, colors='black', linestyles='-')
            plt.clabel(cnt,inline=True,fmt='%1.0f',fontsize=8)
            if maps_pval[0].min() < 0.05: 
                levels = [maps_pval[0].min(),0.05,maps_pval[0].max()]
                ax.contourf(maps_pval[0].lon, lat, maps_pval[0].values,levels, transform=data_crs,levels=levels, hatches=["...", " "], alpha=0)
            elif maps_pval[0].min() < 0.10:
                levels = [maps_pval[0].min(),0.10,maps_pval[0].max()]
                ax.contourf(maps_pval[0].lon, lat, maps_pval[0].values,levels, transform=data_crs,levels=levels, hatches=["...", " "], alpha=0)
            else:
                print('No significant values for ',self.regressor_names[k]) 
            if k == 0:
                plt.title('MEM',fontsize=18)
            else:
                plt.title(storyline_names[k-1],fontsize=18)
            ax.add_feature(cartopy.feature.COASTLINE,alpha=.5)
            ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
            ax.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
            if var == 'ua':
                ax.set_extent([-180, 180, -90, -25], ccrs.PlateCarree())
            elif var == 'sst':
                ax.set_extent([-60, 220, -80, 40], ccrs.PlateCarree(central_longitude=180))
            else: 
                ax.set_extent([-60, 220, -80, 40], ccrs.PlateCarree(central_longitude=180))
            
        plt1_ax = plt.gca()
        left, bottom, width, height = plt1_ax.get_position().bounds
        if var == 'ua':
            colorbar_axes1 = fig_coef.add_axes([left+0.28, bottom, 0.01, height*3])
            colorbar_axes2 = fig_coef.add_axes([left+0.36, bottom, 0.01, height*3])
        elif var == 'sst':
            colorbar_axes1 = fig_coef.add_axes([left+0.3, bottom, 0.01, height*4])
            colorbar_axes2 = fig_coef.add_axes([left+0.38, bottom, 0.01, height*4])
        cbar = fig_coef.colorbar(im0, colorbar_axes1, orientation='vertical')
        cbar2 = fig_coef.colorbar(im, colorbar_axes2, orientation='vertical')
        if var == 'ua':
            cbar.set_label('m/s/std(rd)',fontsize=14) #rotation = radianes
            cbar2.set_label('m/s/std(rd)',fontsize=14) #rotation = radianes
        elif var == 'sst':
            cbar.set_label('K/std(rd)',fontsize=14) #rotation = radianes
            cbar2.set_label('K/std(rd)',fontsize=14) #rotation = radianes
        else:
            cbar.set_label('X/std(rd)',fontsize=14) #rotation = radianes
            cbar2.set_label('X/std(rd)',fontsize=14) #rotation = radianes
        cbar.ax.tick_params(axis='both',labelsize=14)
        cbar2.ax.tick_params(axis='both',labelsize=14)
            
        plt.subplots_adjust(bottom=0.2, right=.95, top=0.8)
        if var == 'ua':
            plt.savefig(output_path+'/storylins_u850',bbox_inches='tight')
        elif  var == 'sst':
            plt.savefig(output_path+'/storylines_sst',bbox_inches='tight')
        else:
            plt.savefig(output_path+'/storylines_unknown_var',bbox_inches='tight')
        
        plt.clf

        return fig_coef
    
    def plot_boxplot(self,path,var,boxes,output_path):
        """ Plots figure with all of 
        :param regressor_names: list with strings naming the independent variables
        :param path: saving path
        :return: none
        """
        maps, maps_pval, R2 = self.open_regression_coef(path,var)
        storylines = [maps[0]+1.3*maps[3]+1.3*maps[1]+1.3*maps[4]+1.3*maps[5],
                      maps[0]+1.3*maps[3]+1.3*maps[1]+1.3*maps[4]-1.3*maps[5],
                      maps[0]+1.3*maps[3]+1.3*maps[1]-1.3*maps[4]+1.3*maps[5],
                      maps[0]+1.3*maps[3]+1.3*maps[1]-1.3*maps[4]-1.3*maps[5],
                      maps[0]+1.3*maps[3]-1.3*maps[1]+1.3*maps[4]+1.3*maps[5],
                      maps[0]+1.3*maps[3]-1.3*maps[1]+1.3*maps[4]-1.3*maps[5],
                      maps[0]+1.3*maps[3]-1.3*maps[1]-1.3*maps[4]+1.3*maps[5],
                      maps[0]+1.3*maps[3]-1.3*maps[1]-1.3*maps[4]-1.3*maps[5],
                      maps[0]-1.3*maps[3]+1.3*maps[1]+1.3*maps[4]+1.3*maps[5],
                      maps[0]-1.3*maps[3]+1.3*maps[1]+1.3*maps[4]-1.3*maps[5],
                      maps[0]-1.3*maps[3]+1.3*maps[1]-1.3*maps[4]+1.3*maps[5],
                      maps[0]-1.3*maps[3]+1.3*maps[1]-1.3*maps[4]-1.3*maps[5],
                      maps[0]-1.3*maps[3]-1.3*maps[1]+1.3*maps[4]+1.3*maps[5],
                      maps[0]-1.3*maps[3]-1.3*maps[1]+1.3*maps[4]-1.3*maps[5],
                      maps[0]-1.3*maps[3]-1.3*maps[1]-1.3*maps[4]+1.3*maps[5],
                      maps[0]-1.3*maps[3]-1.3*maps[1]-1.3*maps[4]-1.3*maps[5],]
        storyline_names = [' + GW + CP + TW + VB ',' + CP + TW - VB ',
                           ' + CP - TW + VB ',' + GW + CP - TW - VB ',
                           ' + GW - CP + TW + VB ',' + GW - CP + TW - VB ',
                           ' + GW - CP - TW + VB ',' + GW - CP - TW - VB ',
                           ' - GW + CP + TW + VB ',' + CP + TW - VB ',
                           ' - CP - TW + VB ',' + GW + CP - TW - VB ',
                           ' - GW - CP + TW + VB ',' + GW - CP + TW - VB ',
                           ' - GW - CP - TW + VB ',' + GW - CP - TW - VB ',]
        dic = {}; dic_sl = {}
        for i,box in enumerate(boxes):
            dic[i] = [self.target.isel(model=m).sel(lat=slice(box[1],box[0]),lon=slice(box[2],box[3])).mean(dim=('lon','lat')) for m,modeel in enumerate(self.target.model)]
            dic_sl[i] = [sl.sel(lat=slice(box[1],box[0]),lon=slice(box[2],box[3])).mean(dim=('lon','lat')) for sl in storylines]    
        
        fig = plt.figure(figsize=(10,12))
        ax = plt.axes([0.1,0.1,0.8,0.8])
        for i,box in enumerate(boxes):
            violin = ax.boxplot(dic[i], positions=[i])
            for m in range(len(storylines)):
                ax.scatter(i,dic_sl[i][m])

        #Do something to have the info of the box somewhere
        plt.ylim(-1,1.7)
        plt.grid()
        plt.ylabel('$\Delta$ SST',fontsize = 20)
        fig.savefig(output_path+'/boxplot_'+var+'_storylines.png')




def stand_detr(dato):
    anom = (dato - np.mean(dato))/np.std(dato)
    return signal.detrend(anom)

def filtro(dato):
    """Apply a rolling mean of 5 years and remov the NaNs resulting bigining and end"""
    signal = dato - dato.rolling(time=10, center=True).mean()
    signal_out = signal.dropna('time', how='all')
    return signal_out
                          
def stand(dato):
    anom = (dato - np.mean(dato))/np.std(dato)
    return anom


def main(config):
    """Run the diagnostic."""
    cfg=get_cfg(os.path.join(config["run_dir"],"settings.yml"))
    print(cfg)
    meta = group_metadata(config["input_data"].values(), "alias")
    rd_list = []
    target_wind_list = []; target_sst_list = []
    models = []
    count = 0
    for alias, alias_list in meta.items():
        #print(f"Computing index regression for {alias}\n")
        ts_dict = {m["variable_group"]: xr.open_dataset(m["filename"])[m["short_name"]].sel(time=slice('1991','2022')).mean(dim='time') - xr.open_dataset(m["filename"])[m["short_name"]].sel(time=slice('1950','1979')).mean(dim='time') for m in alias_list if (m["variable_group"] != "ua850") & (m["variable_group"] != "tos_iod_e") & (m["variable_group"] != "tos_iod_w")& (m["variable_group"] != "sst")}
        rd_list.append(ts_dict)
        target_wind = [xr.open_dataset(m["filename"])[m["short_name"]].sel(time=slice('1991','2022')).mean(dim='time') - xr.open_dataset(m["filename"])[m["short_name"]].sel(time=slice('1950','1979')).mean(dim='time') for m in alias_list if m["variable_group"] == "ua850"]
        target_sst = [xr.open_dataset(m["filename"])[m["short_name"]].sel(time=slice('1991','2022')).mean(dim='time') - xr.open_dataset(m["filename"])[m["short_name"]].sel(time=slice('1950','1979')).mean(dim='time') for m in alias_list if m["variable_group"] == "sst"]
        if len(target_wind) != 0:
            target_wind_list.append(target_wind[0])
            target_sst_list.append(target_sst[0])
            count +=1
            models.append(count)
        else:
            continue
    
    #Across model regrssion - create data array
    regressor_names = rd_list[0].keys()
    regressors = {}
    for rd in regressor_names:
        list_values = [np.mean(np.array(rd_list[m][rd])) for m,model in enumerate(rd_list)]
        print(rd,list_values)
        regressors[rd] = np.array(list_values)
    
    #Create directories to store results
    os.chdir(config["work_dir"])
    os.getcwd()
    os.makedirs("across_model_regression_historical",exist_ok=True)
    os.chdir(config["plot_dir"])
    os.getcwd()
    os.makedirs("across_model_regression_historical",exist_ok=True)
    #Evalutate coefficients and make plots
    ua850_ens = xr.concat(target_wind_list,dim="model")
    var = 'ua'
    os.chdir(config["work_dir"]+'/across_model_regression_historical')
    os.getcwd()
    os.makedirs(var,exist_ok=True)
    #SL = storyline_calc()
    #SL.regression_data(ua850_ens,pd.DataFrame(regressors).apply(stand,axis=0),pd.DataFrame(regressors).keys().insert(0,'MEM'))
    #SL.plot_storylines(config["work_dir"]+'/across_model_regression_historical',var,config["plot_dir"]+'/across_model_regression_historical')
    #SL.plot_storylines_lowGW(config["work_dir"]+'/across_model_regression_historical',var,config["plot_dir"]+'/across_model_regression_historical')                    
    sst_ens = xr.concat(target_sst_list,dim="model")
    var = 'sst'
    os.chdir(config["work_dir"]+'/across_model_regression_historical')
    os.getcwd()
    os.makedirs(var,exist_ok=True)
    SL = storyline_calc()
    SL.regression_data(sst_ens,pd.DataFrame(regressors).apply(stand,axis=0),pd.DataFrame(regressors).keys().insert(0,'MEM'))
    #SL.plot_regression_coef_map(config["work_dir"]+'/across_model_regression_historical',var,config["plot_dir"]+'/across_model_regression_historical')                    
    #SL.plot_storylines(config["work_dir"]+'/across_model_regression_historical',var,config["plot_dir"]+'/across_model_regression_historical')                    
    #SL.plot_storylines_lowGW(config["work_dir"]+'/across_model_regression_historical',var,config["plot_dir"]+'/across_model_regression_historical')                    
    boxes = [[-45,-75,180,280]]
    SL.plot_boxplot(config["work_dir"]+'/across_model_regression_historical',var,boxes,config["plot_dir"]+'/across_model_regression_historical')                    
              
 
if __name__ == "__main__":
    with run_diagnostic() as config:
        main(config)
                              
