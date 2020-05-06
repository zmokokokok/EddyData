from netCDF4 import Dataset

nc_obj = Dataset('G:/all_data/trajectory/eddy_trajectory_2.0exp_19930101_20180118.nc')
# ugos_name = 'ugos'
# vgos_name = 'vgos'
#
#
# print(nc_obj.variables['adt'])
print(nc_obj.variables.keys())
print(nc_obj.variables['latitude'])
# print(nc_obj.variables.keys())
print(nc_obj.variables['speed_radius'])
print(nc_obj.variables['observed_flag'])
print(nc_obj.variables['speed_average'])
print(nc_obj.variables['amplitude'])
print(nc_obj.variables['cyclonic_type'])
# print(nc_obj.variables['time'])
# for i in nc_obj.variables.keys():
#     print(i)
# print('---------------------------------------')
# print(nc_obj.variables['ugos'].shape)
# for i in range(len(nc_obj.variables['speed_radius'])):
#     print(i)