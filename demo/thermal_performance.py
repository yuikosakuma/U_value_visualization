import numpy as np
import pandas as pd
import math

from kalman import kalman_1R1C, kalman_2R1C, kalman_3R1C
from kalman import residual_1R1C, residual_2R1C, residual_3R1C

rng = pd.date_range('2019/1/2', '2019/1/3', freq='0.5H', tz='Asia/Tokyo')
rng_input = pd.date_range('2019/1/2', '2019/1/3', freq='1H', 
                    tz='Asia/Tokyo')

def select_data_without_change(selected_data, c_time, threshold):
    change_rate = selected_data['AC_power'].pct_change().abs()
    # select data after change
    check_threshold =  change_rate[change_rate>threshold]

    # if smaller than 1hour, select data
    if len(check_threshold) > 0:
        cng_time = check_threshold[-1:].index[0] + pd.Timedelta('5min')
        if (c_time- cng_time > pd.Timedelta('1hour')):
            new_selected_data = selected_data[cng_time : c_time]
        else:
            new_selected_data = np.nan
    else:
        new_selected_data = selected_data
    return new_selected_data

def calc_u(window, home_num, model_num, threshold, rng=rng, time_t=60, cop=3.6):
    window_str = '{}h'.format(window)
    file = './processed_data_ver_3/home_{}.csv'.format(home_num)
    df = pd.read_csv(file)
    df.index = pd.to_datetime(df['Time'])
    #df = df.tz_localize("Etc/Greenwich")
    df = df.tz_convert("Asia/Tokyo")
    simulation = df.loc[:,'t_in':'AC_power']
    #simulation = simulation.resample('1min').interpolate()
    
    u_list = []
    residual_list = []
    for i in range(len(rng)):
        p_time = rng[i] - pd.Timedelta(window_str)
        selected_data = simulation[p_time:rng[i]]
        new_selected_data = select_data_without_change(selected_data, rng[i], threshold)
        if isinstance(new_selected_data, pd.DataFrame):
            r_rate = '{}s'.format(time_t)
            new_selected_data = new_selected_data.resample(r_rate).interpolate()
            t_i = np.array(new_selected_data['t_in'])
            t_a = np.array(new_selected_data['t_out'])
            input_power = np.array(new_selected_data['AC_power']) * cop
            if model_num  == 1:
                param = kalman_1R1C(t_i, t_a, input_power, time_t)
                sigma, diff = residual_1R1C(t_i, t_a, input_power, param[1:3], time_t)
            elif model_num  == 2:
                t_h = np.array((new_selected_data['t_corridor'] +new_selected_data['t_corridor'])/2)
                param = kalman_2R1C(t_i, t_a, t_h, input_power, time_t)
                sigma, diff = residual_2R1C(t_i, t_a, t_h, input_power, param[1:4], time_t)
            elif model_num  == 3:
                t_h = np.array((new_selected_data['t_corridor'] +new_selected_data['t_corridor'])/2)
                t_s = np.array(new_selected_data['t_s'])
                param = kalman_3R1C(t_i, t_a, t_h, t_s, input_power, time_t)
                sigma, diff = residual_3R1C(t_i, t_a, t_h, t_s, input_power, param[1:5], time_t)

            c = 1/float(param[1])
            w = float(param[2]) * c
            u = w / 141
            u_list.append(abs(u))
            residual_list.append(sigma)
        else:
            u_list.append(np.nan)
            residual_list.append(np.nan)
    return rng.time, u_list, residual_list

def output_text(home_num):
    occupancy = pd.read_csv( './processed_data_ver_3/occupancy.csv')
    occupancy = occupancy.loc[:, '1':'6']
    occupancy = pd.concat([occupancy, occupancy[:1]])
    occupancy.index = rng_input
    rs_occupancy = occupancy.resample('0.5H').mean().fillna(method='ffill')

    ventilation = pd.read_csv( './processed_data_ver_3/ventilation.csv')
    ventilation = ventilation.loc[:, '1':'6']
    ventilation = pd.concat([ventilation, ventilation[:1]])
    ventilation.index = rng_input
    rs_ventilation = ventilation.resample('0.5H').mean().fillna(method='ffill')

    home_str = '{}'.format(home_num)
    vent = np.array(rs_ventilation[home_str])
    vent_list = ['auto' if v==0 else 'auto+forced' for v in vent]

    occ_list = np.array(rs_occupancy[home_str]*4).astype(int)

    text_list =\
    ['occupancy={0}, ventilation={1}'.format(occ_list[i], vent_list[i]) 
     for i in range(len(occ_list))]
    return text_list

