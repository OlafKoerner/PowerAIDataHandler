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
        self.device_list = {
            1: {'name' : 'espresso-machine', 'minpow' : 800},
            2: {'name' : 'washing-machine', 'minpow' : 500},
            4: {'name' : 'dish-washer', 'minpow' : 500},
            #8: {'name' : 'induction-cooker', 'minpow' : 800},
            16: {'name': 'irrigation-system', 'minpow': 400},
            32: {'name': 'oven', 'minpow': 800},
            #64: {'name': 'microwave', 'minpow': 800},
            #128: {'name': 'kitchen-light', 'minpow': 200},
            #256: {'name': 'living-room-light', 'minpow': 200},
            512: {'name': 'dining-room-light', 'minpow': 200},
            #1024: {'name': 'ground-floor-light', 'minpow': 200},
            #2048: {'name': 'upper-floor-light', 'minpow': 200},
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
                    if self.event_list[key][i]['value'][t] < self.device_list[key]['minpow'] :
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
    

    def generate_training_data_from_events(self, window_length, event_ratio) :
        #class vars
        self.train_ids = {'device_id' : np.array([]), 'event_id' : np.array([])}
        self.test_ids = {'device_id' : np.array([]), 'event_id' : np.array([])}

        #local vars
        train_x = np.array([])
        train_y = np.array([])
        test_x  = np.array([])
        test_y  = np.array([])
        
        #determine minimal amount of datapoints for all devices to balance
        min_device_dps = 10000000000
        for key in self.device_list :
            sum_dps = 0
            for i in range(len(self.event_list[key])) :
                sum_dps = sum_dps + len(self.event_list[key][i]['timestamp'])
            min_device_dps = min(min_device_dps, sum_dps)

        #get sorted device ids to determine target pos to set to 1
        device_ids = np.array([])
        for key in self.device_list :
            device_ids = np.append(device_ids, int(key))
        self.device_ids_order = np.sort(device_ids)

        for key in self.device_list :
            # storage for values for current active device
            train_events_values = np.array([])
            test_events_values  = np.array([])
            
            num_train_events = round(event_ratio * len(self.event_list[key])) 
            num_test_events  = len(self.event_list[key]) - num_train_events

            #print(f"len event_list[{key}]): {len(self.event_list[key])}")
            #print(" num_train_events:", num_train_events)
            #print(" num_test_events:", num_test_events)

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
            i = 0 + window_length
            while i < train_events_values.size :
                train_x = np.append(train_x, train_events_values[i - window_length : i])
                train_y = np.append(train_y, batch_target_values)
                i = i + window_length
            train_x = train_x.reshape((train_x.size // window_length, window_length))
            train_y = train_y.reshape((train_y.size // len(self.device_list), len(self.device_list)))
            
            #https://www.tutorialspoint.com/how-to-normalize-a-numpy-array-so-the-values-range-exactly-between-0-and-1
            for i in range(train_x.shape[0]):
                
                # Calculate the mean and standard deviation of the array
                mean_val = np.mean(train_x[i])
                std_val = np.std(train_x[i])
    
                # Perform z-score normalization
                train_x[i] = (train_x[i] - mean_val) / std_val
            

            #print("train_y.size", train_y.size)
            #print("len(self.device_list)", len(self.device_list))
            
            ### TEST values and target list
            
            # generate batches with values and targets
            i = 0 + window_length
            while i < test_events_values.size :
                test_x = np.append(test_x, test_events_values[i - window_length : i])
                test_y = np.append(test_y, batch_target_values)
                i = i + window_length
            test_x = test_x.reshape((test_x.size // window_length, window_length))
            test_y = test_y.reshape((test_y.size // len(self.device_list), len(self.device_list)))
            
            #https://www.tutorialspoint.com/how-to-normalize-a-numpy-array-so-the-values-range-exactly-between-0-and-1
            for i in range(test_x.shape[0]):
                
                # Calculate the mean and standard deviation of the array
                mean_val = np.mean(test_x[i])
                std_val = np.std(test_x[i])
    
                # Perform z-score normalization
                test_x[i] = (test_x[i] - mean_val) / std_val

        print(f'train_x (batch num, window length): {train_x.shape}\ntrain_y (batch num, device num): {train_y.shape}\ntest_x  (batch num, window length): {test_x.shape}\ntest_y  (batch num, device num): {test_y.shape}')
        return train_x, train_y, test_x, test_y

    
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

        for i in range(test_x.shape[0]) :
            predicted_pos = np.argmax(predict_y[i])
            predicted_device = self.device_ids_order(predicted_pos))
            
            test_device_array_pos = np.argwhere(test_y[i] > 0)
            test_device = int(test_y[i][test_device_array_pos[0]])
            
            #print(f'pred device: {predicted_device}, real device: {test_device}')
            
            if predicted_device != test_device :  #OKO: fixed potential bug in line 218 ????
                self.cnt_wrong[test_device_array_pos] = self.cnt_wrong[test_device_array_pos] + 1
                self.test_x_wrong[test_device] = np.append(self.test_x_wrong[test_device], test_x[i])
            else :
                self.cnt_correct[test_device_array_pos] = self.cnt_correct[test_device_array_pos] + 1
                self.test_x_correct[test_device] = np.append(self.test_x_correct[test_device], test_x[i])
                
    
        result_table = PrettyTable(['device name', 'total', 'correct', 'wrong', 'percent'], align='r')
    
        for i in range(len(self.device_list)) :
            result_table.add_row([
                self.device_list[self.device_ids_order[i]]['name'], 
                int(self.cnt_wrong[i] + self.cnt_correct[i]),
                int(self.cnt_correct[i]),
                int(self.cnt_wrong[i]),
                str(round(100 * self.cnt_correct[i] / (self.cnt_correct[i] + self.cnt_wrong[i]))) + "%" if self.cnt_correct[i] + self.cnt_wrong[i] != 0 else ""])

        print(result_table)
