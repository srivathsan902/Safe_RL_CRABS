def print_args_decorator(func):
    def wrapper(*args, **kwargs):
        if hasattr(func, '__self__'):
            # It's a method, so print class name and method name
            print(f"Arguments for {func.__self__.__class__.__name__}.{func.__name__}:")
        else:
            # It's a regular function
            print(f"Arguments for {func.__name__}:")
        
        if args:
            for i, arg in enumerate(args):
                if hasattr(arg, 'shape'):
                    print(f"Positional arg {i}: type={type(arg)}, shape={arg.shape}")
                else:
                    print(f"Positional arg {i}: type={type(arg)}, value={arg}")
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(value, 'shape'):
                    print(f"Keyword arg '{key}': type={type(value)}, shape={value.shape}")
                else:
                    print(f"Keyword arg '{key}': type={type(value)}, value={value}")
        
        result = func(*args, **kwargs)
        return result
    
    return wrapper

import safety_gymnasium
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def scale_angle(angle):
    '''Scales the Angles between 0 and 360, Input must be in degrees'''
    
    angle_scaled = angle%360

    if angle_scaled<0:
        angle_scaled+=360
    
    return angle_scaled

def get_x_vals(x_indices,lidar_resolution):
    '''Returns in Degrees'''

    index_to_angle_factor = 360/lidar_resolution
    x_vals = x_indices*index_to_angle_factor

    # Handling the Case where there is a jump
    # if len(x_indices)==0:
    #     print(x_indices)
    if max(x_indices)-min(x_indices)==2:
        pass

    else:
        if 1 in x_indices:
            x_vals[0:2] += 360
        else:
            x_vals[0] += 360

    return x_vals

def get_lidar_r_theta(lidar_values,lidar_resolution,max_lidar_distance):
    '''
    The lidar values are processed and location of the center of the circle is returned from agent's frame of reference
    
    Returns
        theta_lidar_rad, r: Position of the Circle Centre wrt agent
        info_lidar: Has info if the agent is within or out of limits
    '''
    # print('lidar_values', lidar_values.shape)
    x_indices = np.where(lidar_values>0)[0]
    ''' have the indices of top 3 largest values in the lidar_values array '''
    x_indices = np.argsort(lidar_values)[-3:]
    # print('x_indices', x_indices)
    # if len(x_indices)>3:
    #     raise
        # We assumed that the lidar generates only 3 non-zero values, get_x_vals() uses this assumption
    
    if len(x_indices)>0:
        y_vals = lidar_values[x_indices]
        x_vals = get_x_vals(x_indices,lidar_resolution)

        a,b,c = np.polyfit(x_vals,y_vals,2)
        theta_lidar = (-b/(2*a)) # Its in degrees
        lidar_max = c - b**2/(4*a)
        r = (1-lidar_max)*max_lidar_distance

        theta_lidar = scale_angle(theta_lidar)+180/lidar_resolution
        info_lidar = {'x_vals':x_vals,'y_vals':y_vals,'within_limits':True}

    else:
        theta_lidar = -1
        r = max_lidar_distance
        info_lidar = {'x_vals':-1,'y_vals':-1,'within_limits':False}

    theta_lidar_rad = theta_lidar*np.pi/180
    return theta_lidar_rad,r,info_lidar

def get_coords(observations,max_lidar_distance,lidar_resolution):
    '''Processes the 16-dim observation vector and return (x,y,theta) coordinates of the agent
    by using the magnetometer for orientation and lidar for distance and angle from the origin
    '''
    all_x = []
    all_y = []
    all_theta_local = []
    all_info_lidar = []

    # Processing the State
    for obs in observations:
        # print('obs', obs.shape)
        lidar_values = obs[-16:]
        mag0 = obs[9]
        mag1 = obs[10]
        theta_local = np.arctan2(mag0,mag1)
        # theta_lidar = (np.argmax(lidar_values)+1)*2*np.pi/lidar_resolution
        theta_lidar,r,info_lidar = get_lidar_r_theta(lidar_values,lidar_resolution,max_lidar_distance)
        
        if info_lidar['within_limits']==False:
            # print("Agent OUTSIDE LIMITS")
            return -1,-1,-1,info_lidar

        r = (1-max(lidar_values))*max_lidar_distance

        theta_bot = theta_lidar+theta_local+np.pi
        theta_bot_scaled = scale_angle(theta_bot*180/np.pi)*np.pi/180 #This is to bring it back to [0,2pi]

        x,y = r*np.cos(theta_bot_scaled),r*np.sin(theta_bot_scaled)
        # print("X Vals: {} | Y vals : {} | Theta Lidar: {:.2f} | Theta Local: {:.2f} | theta_bot_scaled :{:.2f} ".format(np.round(info_lidar['x_vals'],2),np.round(info_lidar['y_vals'],2),theta_lidar*180/np.pi,theta_local*180/np.pi,theta_bot_scaled*180/np.pi))
        all_x.append(x)
        all_y.append(y)
        all_theta_local.append(theta_local)
        all_info_lidar.append(info_lidar)

    all_x = torch.tensor(all_x).float()
    all_y = torch.tensor(all_y).float()
    all_theta_local = torch.tensor(all_theta_local)

    # return x,y,theta_local,info_lidar
    return all_x, all_y, all_theta_local, all_info_lidar