from fusion_utils.radar_GUI_util import *
import argparse
import os
import shutil

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--footage', nargs='+', type=str, default='remote', help='Recording file')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    args = parse_opt()
    #parameters
    intrinsic = [747.9932, 0., 655.5036, 0., 746.6126, 390.1168, 0., 0., 1.]
    footage = 'record/' + args.footage
    assert os.path.exists(footage)
    # get extrinsic matrix from radar to camera
    cfg_name = 'config/' + args.footage.strip('.bag') + '.yaml'
    if not os.path.exists(cfg_name):
        shutil.copyfile('config/cfg_template.yaml', cfg_name)
    r2c_ext = r2c_extrinsic(intrinsic, footage)
    c2g_ext = c2g_extrinsic(r2c_ext, footage)
    # get road mask
    road_mask_x, road_mask_y = road_mask(intrinsic, r2c_ext, c2g_ext, footage)
    cfg_write(cfg_name, r2c_ext, c2g_ext, intrinsic, road_mask_x, road_mask_y)
    print(r2c_ext, c2g_ext, intrinsic, road_mask_x, road_mask_y)

# write into cfg file

