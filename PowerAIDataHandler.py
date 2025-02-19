# import libs
from decouple import Config, RepositoryEnv, Csv #https://github.com/HBNetwork/python-decouple/issues/116
import numpy as np
from numpy.fft import fft, ifft
import pymysql
from urllib.request import urlopen
import json
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from functools import reduce
from random import random
from prettytable import PrettyTable

#class definition
class ClassPowerAIDataHandler() :
    def __init__(self, fname_dotenv) :

        # configuration
        # minpow means consumption only by the device
        self.base_pow = 200
        self.device_list = {
            1: {'name' : 'espresso-machine', 'minpow' : 800, 'maxpow' : 1500},
            2: {'name' : 'washing-machine', 'minpow' : 0, 'maxpow': 3000},
            4: {'name' : 'dish-washer', 'minpow' : 1900, 'maxpow': 2600},
            8: {'name' : 'induction-cooker', 'minpow' : 500, 'maxpow': 3500},
            16: {'name': 'irrigation-system', 'minpow': 800, 'maxpow': 1500},
            32: {'name': 'oven', 'minpow': 900, 'maxpow': 5500},
            64: {'name': 'microwave', 'minpow': 800, 'maxpow': 1500},
            128: {'name': 'kitchen-light', 'minpow': 250, 'maxpow': 500},
            256: {'name': 'living-room-light', 'minpow': -self.base_pow, 'maxpow': 300}, # OKO for base load
            512: {'name': 'dining-room-light', 'minpow': 0, 'maxpow': 400}, #OKO data looks strange ...
            1024: {'name': 'ground-floor-light', 'minpow': 400},
            2048: {'name': 'upper-floor-light', 'minpow': 180},
        }

        #setup connection to mysql database
        self.config = Config(RepositoryEnv(fname_dotenv))

        #init member vars 
        self.data_start = self.config('mydata_start')
        

    def read_events_from_db(self) :

        self.event_list = dict([(key, []) for key in self.device_list.keys()])

        for key in self.device_list.keys() :

            if self.config('myhost') == 'localhost':
                conn = pymysql.connect(
                    host=self.config('myhost'),
                    user=self.config('myuser'),
                    password=self.config('mypassword'),
                    database=self.config('mydatabase'),
                    cursorclass=pymysql.cursors.DictCursor)
                cur = conn.cursor()

                # read from mysql db
                cur.execute("SELECT * FROM data WHERE device = " + str(key) + " AND timestamp > " + str(self.data_start))
                # get all rows where device is active
                self.data_list = cur.fetchall()
                conn.close()
            else:
                with urlopen(self.config('myhost') + '/device_data/' + str(key) + '/' + str(self.data_start)) as response :
                    self.data_list = json.loads(response.read())
                
            # store values in dict = { device_id : [ timestamp : [], value : [], device : [] ] }
            timestamp_before = 0
            event_id = -1        
            for row in self.data_list :
                
                if row['timestamp'] - timestamp_before > 100000 : #always called at first iteration #fk: muss das beim Testen berücksichtigt werden?
                    event_id = event_id + 1
                    self.event_list[key].append({'timestamp' : [], 'value' : [], 'device' : []})
                    
                self.event_list[key][event_id]['timestamp'].append(row['timestamp'])
                self.event_list[key][event_id]['value'].append(row['value'])
                self.event_list[key][event_id]['device'].append(row['device'])

                timestamp_before = row['timestamp']


    def skip_events_from_device(self, device_id, event_skip_list):
        for event_id in sorted(event_skip_list, reverse=True):
            self.event_list[device_id].pop(event_id-1) #-1, to start from 1 in Jupyter
               

    def delete_event_from_device_from_db(self, device_id, event_id, timestamp_check):
        ts_from  = self.event_list[device_id][event_id-1]['timestamp'][0]
        ts_to    = self.event_list[device_id][event_id-1]['timestamp'][-1]
        if int(ts_from) <= int(timestamp_check) <= int(ts_to):
            if self.config('myhost') == 'localhost':
                conn = pymysql.connect(
                    host=self.config('myhost'),
                    user=self.config('myuser'),
                    password=self.config('mypassword'),
                    database=self.config('mydatabase'),
                    cursorclass=pymysql.cursors.DictCursor)
                cur = conn.cursor()
                # write to local mysql db
                #OKO cur.execute(f"UPDATE data SET device = device & ~{device_id} WHERE timestamp >= {ts_from} AND timestamp <= {ts_to};")
                print(f"OKO DB execution disabled. Command requested:\nUPDATE data SET device = device & ~{device_id} WHERE timestamp >= {ts_from} AND timestamp <= {ts_to};")
                
                conn.close()
            else:
                with urlopen(f"{self.config('myhost')}/update/{device_id}/{ts_from}/{ts_to}/0") as response :
                    print(json.loads(response.read()))

    
    def filter_events_by_minpow(self) :
        for key in self.device_list :
            
            for i in range(len(self.event_list[key])) :
                
                delete_mode = False
                delete_list = np.array([])
                
                for t in range(len(self.event_list[key][i]['value'])) :  
                    if (self.event_list[key][i]['value'][t] < self.base_pow + self.device_list[key]['minpow'] or 
                        self.event_list[key][i]['value'][t] > self.base_pow + self.device_list[key]['maxpow']):
                            delete_mode = True
                            delete_list = np.append(delete_list, t)
                    else :
                        if delete_mode :
                            delete_list = np.delete(delete_list, -1)
                            delete_mode = False 
                
                self.event_list[key][i]['timestamp']    = np.delete(self.event_list[key][i]['timestamp'],   delete_list.astype(int))
                self.event_list[key][i]['value']        = np.delete(self.event_list[key][i]['value'],       delete_list.astype(int))
                self.event_list[key][i]['device']       = np.delete(self.event_list[key][i]['device'],      delete_list.astype(int))
   
    
    def print_events(self, device_id=0, event_id=0) :

        if (device_id > 0):
            #reduce lists to specified event
            device_list_local = {device_id: [self.device_list[device_id]['name']]}
            if (event_id > 0):
                event_list_local  = {device_id: [self.event_list[device_id][event_id-1]]}
            else:
                event_list_local  = self.event_list
        else:
            #keep full lists to print all
            device_list_local = self.device_list
            event_list_local  = self.event_list

        for key in device_list_local:
            
            for i in range(len(event_list_local[key])) :
                
                x = pd.to_datetime(self.event_list[key][i]['timestamp'], utc=True, unit='ms')
                y = np.array(self.event_list[key][i]['value'])
                
                for ii in range(1) :
                    
                    f = plt.figure()
                    a = f.add_subplot()
                    
                    date_locator = mdates.AutoDateLocator()
                    date_form = mdates.AutoDateFormatter(date_locator)
                    #alternative: date_form = DateFormatter("%d.%m.%y %H:%M:%S")
                    a.xaxis.set_major_formatter(date_form)
                    a.xaxis.set_major_locator(date_locator)
                    a.set_title("Plot " + str(i + 1) + "/" + str(len(self.event_list[key])) + " for " + self.device_list[key]['name'] + " from " + x[0].strftime("%d.%m.%Y"))
                    plt.gcf().autofmt_xdate()
                    
                    a.plot(x, y, marker = 'o')
                    
                    #zoom in
                    x = x[20:40]
                    y = y[20:40]


    def draw_event_fft(self, device_id, event_id):
        #https://pythonnumericalmethods.studentorg.berkeley.edu/notebooks/chapter24.04-FFT-in-Python.html    
        # sampling rate
        sr = 1.0
        # sampling interval
        ts = 1.0/sr        
        x = self.event_list[device_id][event_id-1]['value']
        #x = np.append(x, np.flip(x)) #adding flipped event should reduce high amplitudes at edges 
        X = fft(x)
        #cut = 500
        #X[-cut:cut] = 0
        N = len(X)
        n = np.arange(N)
        T = N/sr
        freq = n/T 
        t = np.arange(0, len(x), ts)

        plt.figure(figsize = (12, 6))
        plt.subplot(121)

        plt.stem(freq, np.abs(X), 'b', \
                markerfmt=" ", basefmt="-b")
        plt.xlabel('Freq (Hz)')
        plt.ylabel('FFT Amplitude |X(freq)|')
        plt.xlim(0, 1)

        plt.subplot(122)
        plt.plot(t, ifft(X), 'r')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.tight_layout()
        plt.show()
    

    def generate_training_data_from_events(self, window_length_max, event_ratio) :
        #class vars
        self.window_length = 0
        self.train_ids = {'device_id' : np.array([]), 'event_id' : np.array([])}
        self.test_ids = {'device_id' : np.array([]), 'event_id' : np.array([])}

        #local vars
        train_x = np.array([])
        train_y = np.array([])
        test_x  = np.array([])
        test_y  = np.array([])
        
        #determine window length
        min_event_length = window_length_max
        for key in self.event_list:
            for i in range(len(self.event_list[key])):
                min_event_length = min(min_event_length, len(self.event_list[key][i]['timestamp']))
        self.window_length = min(min_event_length, window_length_max)
        print(f'chosen window_length: {self.window_length}')

        #determine minimal amount of datapoints for all devices to balance
        self.min_device_dps = 1000000000
        for key in self.device_list :
            sum_dps = 0
            for i in range(len(self.event_list[key])) :
                sum_dps = sum_dps + len(self.event_list[key][i]['timestamp'])
            print(f"available data points for {key} ({self.device_list[key]['name']}): {sum_dps}")
            self.min_device_dps = min(self.min_device_dps, sum_dps)
        print(f'self.min_device_dps: {self.min_device_dps}')

        #get sorted device ids to determine target pos to set to 1
        device_ids = np.array([])
        for key in self.device_list :
            device_ids = np.append(device_ids, int(key))
        self.device_ids_order = np.sort(device_ids)
        print(f'self.device_ids_order: {self.device_ids_order}')

        for key in self.device_list :
            # storage for values for current active device
            train_events_values = np.array([])
            test_events_values  = np.array([])
            
            num_train_events = round(event_ratio * len(self.event_list[key])) 
            num_test_events  = len(self.event_list[key]) - num_train_events

            for i in range(len(self.event_list[key])) :
                if i <= num_train_events :        
                    train_events_values = np.append(train_events_values, np.array(self.event_list[key][i]['value']).astype(float))
                    self.train_ids['device_id'] = np.append(self.train_ids['device_id'], int(key))
                    self.train_ids['event_id'] = np.append(self.train_ids['event_id'], i)
                else :
                    test_events_values  = np.append(test_events_values,  np.array(self.event_list[key][i]['value']).astype(float))
                    self.test_ids['device_id'] = np.append(self.test_ids['device_id'], int(key))
                    self.test_ids['event_id'] = np.append(self.test_ids['event_id'], i)
                    
            ### TRAIN values and target list
            
            # init batch targets for this device
            batch_target_values = np.zeros(len(self.device_list))
            batch_target_values[np.argwhere(self.device_ids_order == int(key))] = 1.

            # generate batches with values and targets
            i = 0 + self.window_length
            while i < train_events_values.size and i < self.min_device_dps * event_ratio:
                train_x = np.append(train_x, train_events_values[i - self.window_length : i])
                train_y = np.append(train_y, batch_target_values)
                i = i + self.window_length
            train_x = train_x.reshape((train_x.size // self.window_length, self.window_length))
            train_y = train_y.reshape((train_y.size // len(self.device_list), len(self.device_list)))
            
            ### TEST values and target list
            
            # generate batches with values and targets
            i = 0 + self.window_length
            while i < test_events_values.size and i < self.min_device_dps * (1 - event_ratio):
                test_x = np.append(test_x, test_events_values[i - self.window_length : i])
                test_y = np.append(test_y, batch_target_values)
                i = i + window_length
            test_x = test_x.reshape((test_x.size // self.window_length, self.window_length))
            test_y = test_y.reshape((test_y.size // len(self.device_list), len(self.device_list)))

        # Calculate the mean and standard deviation of the array
        mean_val = np.mean(train_x)
        std_val = np.std(train_x)
        # Perform z-score normalization
        train_x = (train_x - mean_val) / std_val
        print(f'z-normalize power values with mean = {mean_val} and std = {std_val}')
        
        # randomization of batches
        idx = np.random.permutation(len(train_x))
        train_x = train_x[idx]
        train_y = train_y[idx]
        print(f'randomized batches')
        print(f'train_x (batch num, window length): {train_x.shape}\ntrain_y (batch num, device num): {train_y.shape}\ntest_x  (batch num, window length): {test_x.shape}\ntest_y  (batch num, device num): {test_y.shape}')
        return train_x, train_y

    
    def generate_test_data_from_events(self, event_ratio) :
        test_x  = np.array([])
        test_y  = np.array([])

        for key in self.device_list :
            for i in range(len(self.event_list[key])) :
                if i >= len(self.event_list[key]) * (1 - event_ratio) :        
                    test_x = np.append(test_x, np.array(self.event_list[key][i]['value']).astype(float))
                    test_y = np.append(test_y, np.array(self.event_list[key][i]['device']).astype(float))
        
        return test_x, test_y, test_t

    
    def compare_algo_results(self, test_x, y) :
        #compare results
        correct = np.zeros(12) 
        wrong = np.zeros(12) 
        nothing = 0
        total = len(test_x['value']) 
        for i in range(total): 
            device_pos = int(np.log2(test_x['device'][i])) 
            if test_x['device'][i] == y[i]: 
                correct[device_pos] = correct[device_pos] + 1 
            else: 
                wrong[device_pos] = wrong[device_pos] + 1 
        
        #print table
        result_table = PrettyTable(['device name', 'total', 'correct', 'wrong', 'percent'], align='r') 
        for i in range(len(self.device_list)) : 
            result_table.add_row([ 
            self.device_list[np.power(2, i)]['name'], 
            int(wrong[i] + correct[i]), 
            int(correct[i]), 
            int(wrong[i]), 
            str(round(100 * correct[i] / (correct[i] + wrong[i]))) + "%" if correct[i] + wrong[i] != 0 else ""])
        
        print(result_table)


    def compare_with_testdata(self, predict_y, test_x, test_y) :
        self.cnt_wrong   = np.zeros(len(self.device_list))
        self.cnt_correct = np.zeros(len(self.device_list))
        self.test_x_wrong = dict(zip([1,2,4,8,16,32,64,128,256,512,1024,2048], [[],[],[],[],[],[],[],[],[], [],[],[]]))
        self.test_x_correct = dict(zip([1,2,4,8,16,32,64,128,256,512,1024,2048], [[],[],[],[],[],[],[],[],[], [],[],[]]))

        #calculate percentage of correct classifications
        for i in range(test_x.shape[0]) :
            predicted_pos = np.argmax(predict_y[i])
            test_device_array_pos = np.argmax(test_y[i])
            
            if predicted_pos != test_device_array_pos :
                self.cnt_wrong[test_device_array_pos] = self.cnt_wrong[test_device_array_pos] + 1
                #self.test_x_wrong[test_device] = np.append(self.test_x_wrong[test_device], test_x[i])
            else :
                self.cnt_correct[test_device_array_pos] = self.cnt_correct[test_device_array_pos] + 1
                #self.test_x_correct[test_device] = np.append(self.test_x_correct[test_device], test_x[i])
                
        result_table = PrettyTable(['device name', 'total', 'correct', 'wrong', 'percent'], align='r')
    
        for i in range(len(self.device_list)) :
            result_table.add_row([
                self.device_list[self.device_ids_order[i]]['name'], 
                int(self.cnt_wrong[i] + self.cnt_correct[i]),
                int(self.cnt_correct[i]),
                int(self.cnt_wrong[i]),
                str(round(100 * self.cnt_correct[i] / (self.cnt_correct[i] + self.cnt_wrong[i]))) + "%" if self.cnt_correct[i] + self.cnt_wrong[i] != 0 else ""])

        print(result_table)

        #cross comparision of classification results
        counts = dict([(key, 0) for key in self.device_list.keys()])
        for key in counts:
            counts[key] = dict([(key, 0) for key in self.device_list.keys()])

        for i in range(predict_y.shape[0]):
            counts[self.device_ids_order[np.argmax(test_y[i])]][self.device_ids_order[np.argmax(predict_y[i])]] = counts[self.device_ids_order[np.argmax(test_y[i])]][self.device_ids_order[np.argmax(predict_y[i])]] + 1

            result_table = PrettyTable(range(len(self.device_list)+1), align='r')
            
        for key in self.device_list :
            result_table.add_row(
                np.append(np.array([key]), list(counts[key].values()))
            )
        print(result_table)
