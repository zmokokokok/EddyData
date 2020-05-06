from py_eddy_tracker.dataset.grid import RegularGridDataset
import os
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import logging
from netCDF4 import Dataset
from datetime import datetime
# img_dir = 'F:/origin_data/'
# output_dir = 'F:/filtered_SSH/'
# num = 0

# for img in os.listdir(img_dir):
#     num += 1
#     img_path = os.path.join(img_dir, img)  # F:/origin_data/20180923.nc
#     grid_name = img_dir + img
#     lon_name = 'longitude'
#     lat_name = 'latitude'
#     h = RegularGridDataset(grid_name, lon_name, lat_name)
#     h.bessel_high_filter('adt', 500, order=3) # filtered image set the code true
#
#     fig = plt.figure(figsize=(3, 2))
#     ax = fig.add_axes([.03, .03, .94, .94])
#     # ax.set_title('ADT (m)')
#     ax.set_ylim(4, 30)
#     ax.set_xlim(105.5,150)
#     plt.axis('off')
#     plt.gca().xaxis.set_major_locator(plt.NullLocator())
#     plt.gca().yaxis.set_major_locator(plt.NullLocator())
#     plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
#     plt.margins(0,0)
#     ax.set_aspect('equal')
#     # m = h.display(ax, name='adt', vmin=-1, vmax=1) # not flitered image
#     m = h.display(ax, name='adt', vmin=-.1, vmax=.1) # filtered image
#     # ax.grid(True)
#     # plt.colorbar(m, cax=fig.add_axes([.94, .51, .01, .45]))
#     out_item_name = img.split('.')[0] + '.png'
#     fig.savefig(output_dir + out_item_name, bbox_inches = 'tight')
#     if num == 5480:
#         break



# logging.getLogger().setLevel('DEBUG') # Values: ERROR, WARNING, INFO, DEBUG
img_dir = 'G:/all_data/origin_data/'
output_dir = 'G:/all_data/temporary/'
num = 0
for img in os.listdir(img_dir):
    num += 1
    img_path = os.path.join(img_dir,img) # F:/origin_data/20180923.nc
    grid_name = img_dir + img
    lon_name = 'longitude'
    lat_name = 'latitude'
    h = RegularGridDataset(grid_name, lon_name, lat_name)
    h.bessel_high_filter('adt', 500, order=3)
    date = datetime(2019, 6, 1)
    a, c = h.eddy_identification(
    'adt', 'ugos', 'vgos', # Variable to use for identification
    date, # Date of identification
    0.002, # step between two isolines of detection (m)
    pixel_limit=(5, 2000), # Min and max of pixel can be include in contour
    shape_error=55, # Error maximal of circle fitting over contour to be accepted
    bbox_surface_min_degree=.125 ** 2, # degrees surface minimal to take in account contour
    )

    fig = plt.figure(figsize=(3, 2))
    ax = fig.add_axes([.03,.03,.94,.94])
    # ax.set_title('Eddies detected -- Cyclonic(red) and Anticyclonic(blue)')
    ax.set_ylim(4,30)
    ax.set_xlim(105.5,150)

    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)

    ax.set_aspect('equal')
    a.display(ax, color='b', linewidth=.5)
    c.display(ax, color='r', linewidth=.5)
    # ax.grid()

    out_item_name = img.split('.')[0] + '.png'
    fig.savefig(output_dir + out_item_name , bbox_inches = 'tight')

    if num == 3:
        break



# grid_name = 'F:/origin_data/20180923.nc'
# lon_name = 'longitude'
# lat_name = 'latitude'
# h = RegularGridDataset(grid_name, lon_name, lat_name)
# h.bessel_high_filter('adt', 500, order=3)
# # h.write('tmp/grid.nc')
# fig = plt.figure(figsize=(3, 2))
# ax = fig.add_axes([.03,.03,.94,.94])
# # ax.set_title('ADT Filtered (m)')
# ax.set_ylim(4, 30)
# ax.set_xlim(105.5,150)
# plt.axis('off')
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
# plt.margins(0,0)
# ax.set_aspect('equal')
# m = h.display(ax, name='adt', vmin=-.1, vmax=.1)
# # ax.grid(True)
# # plt.colorbar(m, cax=fig.add_axes([.94, .02, .01, .45]))
# fig.savefig('share/png/filter.png', bbox_inches = 'tight')

# raw = RegularGridDataset(grid_name, lon_name, lat_name)
# filtered = RegularGridDataset(grid_name, lon_name, lat_name)
# filtered.bessel_low_filter('adt', 150, order=3)
#
# areas = dict(
#     sud_pacific=dict(llcrnrlon=188, urcrnrlon=280, llcrnrlat=-64, urcrnrlat=-7),
#     atlantic_nord=dict(llcrnrlon=290, urcrnrlon=340, llcrnrlat=19.5, urcrnrlat=43),
#     indien_sud=dict(llcrnrlon=35, urcrnrlon=110, llcrnrlat=-49, urcrnrlat=-26),
#     )
#
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(111)
# ax.set_title('Spectrum')
# ax.set_xlabel('km')
# for name_area, area in areas.items():
#
#     lon_spec, lat_spec = raw.spectrum_lonlat('adt', area=area)
#     mappable = ax.loglog(*lat_spec, label='lat %s raw' % name_area)[0]
#     ax.loglog(*lon_spec, label='lon %s raw' % name_area, color=mappable.get_color(), linestyle='--')
#
#     lon_spec, lat_spec = filtered.spectrum_lonlat('adt', area=area)
#     mappable = ax.loglog(*lat_spec, label='lat %s high' % name_area)[0]
#     ax.loglog(*lon_spec, label='lon %s high' % name_area, color=mappable.get_color(), linestyle='--')
#
# ax.set_xscale('log')
# ax.legend()
# ax.grid()
# #fig.savefig('share/png/spectrum.png')
#
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(111)
# ax.set_title('Spectrum ratio')
# ax.set_xlabel('km')
# for name_area, area in areas.items():
#     lon_spec, lat_spec = filtered.spectrum_lonlat('adt', area=area, ref=raw)
#     mappable = ax.plot(*lat_spec, label='lat %s high' % name_area)[0]
#     ax.plot(*lon_spec, label='lon %s high' % name_area, color=mappable.get_color(), linestyle='--')
#
# ax.set_xscale('log')
# ax.legend()
# ax.grid()
# fig.savefig('share/png/spectrum_ratio.png')

# from netCDF4 import Dataset
# with Dataset(date.strftime('share/Anticyclonic_%Y%m%d.nc'), 'w') as h:
#     a.to_netcdf(h)
# with Dataset(date.strftime('share/Cyclonic_%Y%m%d.nc'), 'w') as h:
#     c.to_netcdf(h)

