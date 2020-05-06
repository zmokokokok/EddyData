from netCDF4 import Dataset
from matplotlib import pyplot as plt
import os

input_dir = 'F:/origin_data/'
output_dir = 'F:/geostrophic_flow/'
num = 0
for img in os.listdir(input_dir):
    num += 1
    print(num)
    img_path = os.path.join(input_dir,img)
    # print(img_path)
    # exit()
    nc_obj = Dataset(img_path)
    lon = nc_obj.variables['longitude']
    lat = nc_obj.variables['latitude']

    ugos = nc_obj.variables['ugos']

    vgos = nc_obj.variables['vgos']
    # print(ugos[0])
    # print(vgos[0])
    # exit()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_axes([.03,.03,.94,.94])
    ax.set_ylim(4,30)
    ax.set_xlim(105.5,150)
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    ax.set_aspect('equal')
    plt.quiver(lon, lat, ugos[0], vgos[0], scale=.5, scale_units='xy', headaxislength=4.5,
               headlength=4.5, headwidth=3, color='k', angles='xy', width=0.001,
               minshaft=1.5, pivot='middle')
    out_item_name = img.split('.')[0] + '.pdf'
    plt.savefig(output_dir + out_item_name, bbox_inches = 'tight')

    # if num == 1:
    #     break

