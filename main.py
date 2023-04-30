
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from scipy.stats import poisson

df = pd.read_excel(r'/content/drive/MyDrive/Control_Chart_Limits.xlsx')
j0 = 0

def I_chart(control_chart_data,data,sample_size):
  control_chart_data = np.append(control_chart_data,data[-1])
  return control_chart_data

def MR_chart(control_chart_data,data,sample_size):
  control_chart_data = np.append(control_chart_data,0 if len(data)  == 1 else abs(data[-1] - data[-2]))
  return control_chart_data

def X_chart(control_chart_data,data,n):
  
    
  if (len(data)-1)%n == 0:

    control_chart_data= np.append(control_chart_data,data[-1])
    
    
  else :
    
    end_sample = data[-1 - len(data[:-1])%n :]
    control_chart_data = control_chart_data[:-1]
    control_chart_data = np.append(control_chart_data,np.mean(end_sample))
    
  
  return control_chart_data

def R_chart(control_chart_data,data,n):
  
  if (len(data)-1)%n == 0:
    control_chart_data= np.append(control_chart_data,0)
  else:
    end_sample = data[-1 - len(data[:-1])%n :]
    control_chart_data = control_chart_data[:-1]
    control_chart_data = np.append(control_chart_data,np.ptp(end_sample))
    
  return control_chart_data

def S_chart(control_chart_data,data,n):
  if (len(data)-1)%n == 0:
    control_chart_data= np.append(control_chart_data,0)
  else:
    end_sample = data[-1 - len(data[:-1])%n :]
    control_chart_data = control_chart_data[:-1]
    control_chart_data = np.append(control_chart_data,np.std(end_sample)/np.sqrt(len(end_sample)*np.sqrt(len(end_sample)-1)))
  return control_chart_data

def np_chart(control_chart_data,data,n, mean,sigma):
  if (len(data)-1)%n == 0:
    if data[-1] <= mean + sigma and data[-1] >= mean -sigma:
      control_chart_data = np.append(control_chart_data,0)
    else:
      control_chart_data = np.append(control_chart_data,1)
  else:
    #end_sample = data[-1 - len(data[:-1])%n :]
    #control_chart_data = control_chart_data[:-1]
    
    #num_defective = 0
    #for i in end_sample:
      #if not(i <= mean + sigma and i >= mean - sigma):
        #num_defective += 1
    
    #control_chart_data = np.append(control_chart_data,num_defective)
    return control_chart_data
    

  return control_chart_data  
    
        
def p_chart(control_chart_data,data,mean,sigma):

  if len(control_chart_data) == 0:

    if data[-1] <= mean + sigma and data[-1] >= mean -sigma:
      return [[0,1]]
    else:
      return [[1,1]]

  r = np.random.choice([0,1])
  if r == 1:
    
    n = int(control_chart_data[-1][1])
    
    end_sample = data[-1-n:]
    
    control_chart_data = control_chart_data[:-1]
    num_defective = 0
    for i in end_sample:
      if not(i <= mean + sigma and i >= mean - sigma):

        num_defective += 1
  
    if len(control_chart_data) == 0:
      control_chart_data = [[num_defective/(n+1),n+1]]
    else:
      control_chart_data = np.append(control_chart_data,[[num_defective/(n+1),n+1]], axis = 0)
    
  else:
    
    if data[-1] <= mean + sigma and data[-1] >= mean -sigma:
      control_chart_data = np.append(control_chart_data,[[0,1]], axis = 0)
    else:
      control_chart_data = np.append(control_chart_data,[[1,1]], axis = 0)
  return control_chart_data

def C_chart(control_chart_data,data,n):
  
  if  (len(data)%n) == 0:
    sum = 0
    for i in range(n):
      sum += data[-1-i]
    control_chart_data= np.append(control_chart_data,sum)
  else:
    #end_sample = data[-1 - len(data[:-1])%n :]
    #control_chart_data = control_chart_data[:-1]
    #control_chart_data = np.append(control_chart_data,np.sum(end_sample))
    return control_chart_data
  return control_chart_data

def U_chart(control_chart_data,data):
  if len(control_chart_data) == 0:
    return [[data[-1],1]]
  r = np.random.choice([0,1])
  if r == 1:
    n = int(control_chart_data[-1][1])
    end_sample = data[-1-n:]
    control_chart_data = control_chart_data[:-1]
    if len(control_chart_data) == 0:
      control_chart_data = [[np.sum(end_sample)/(n+1),n+1]]
    else:
      control_chart_data = np.append(control_chart_data,[[np.sum(end_sample)/(n+1),n+1]], axis = 0)
  else:
    control_chart_data = np.append(control_chart_data,[[data[-1],1]], axis = 0)
  return control_chart_data


def control_check(arr,control_limits):
  #print(arr)
  #print("Data",arr)
  #print("CL",control_limits)
  #print("P",arr,control_limits[0])
  """l = len(arr)
  x_axis = np.array([])
  for i in range(1,l+1):
    x_axis = np.append(x_axis,i)
  fig, ax = plt.subplots()
  ax.plot(x_axis,arr)
  ax.set_xlabel("Sr no")
  ax.set_ylabel("Data_points")
  ax.set_title("Control_Chart")
  plt.savefig("myplot.jpg") """
  
  
  if len(arr)>=1:
    if arr[len(arr)-1] < control_limits[len(arr)-1][0]:
      #print("A", arr[len(arr)-1],control_limits[len(arr)-1][0])
      return False
      
    elif arr[len(arr)-1] > control_limits[len(arr)-1][2]:
      #print("B",arr[len(arr)-1],control_limits[len(arr)-1][2])
      return False
  """if len(arr) >= 3:
    if ((arr[len(arr)-1] > control_limits[len(arr)-1][1] + 2*control_limits[len(arr)-1][3]) and (arr[len(arr)-2] > control_limits[len(arr)-2][1] + 2*control_limits[len(arr)-2][3])
        or (arr[len(arr)-1] > control_limits[len(arr)-1][1] + 2*control_limits[len(arr)-1][3] and arr[len(arr)-3] > control_limits[len(arr)-3][1] + 2*control_limits[len(arr)-3][3])
        or (arr[len(arr)-2] > control_limits[len(arr)-2][1] + 2*control_limits[len(arr)-2][3] and arr[len(arr)-3] > control_limits[len(arr)-3][1] + 2*control_limits[len(arr)-3][3])):
      print("C")
      return False
    elif ((arr[len(arr)-1] < control_limits[len(arr)-1][1] - 2*control_limits[len(arr)-1][3] and arr[len(arr)-2] < control_limits[len(arr)-2][1] - 2*control_limits[len(arr)-2][3])
        or (arr[len(arr)-1] < control_limits[len(arr)-1][1] - 2*control_limits[len(arr)-1][3] and arr[len(arr)-3] < control_limits[len(arr)-3][1] - 2*control_limits[len(arr)-3][3])
        or (arr[len(arr)-2] < control_limits[len(arr)-2][1] - 2*control_limits[len(arr)-2][3] and arr[len(arr)-3] < control_limits[len(arr)-3][1] - 2*control_limits[len(arr)-3][3])):
      print("D")
      return False """
  """if len(arr) >= 5:
    if ((arr[len(arr)-1] > control_limits[len(arr)-1][1] + control_limits[len(arr)-1][3] and arr[len(arr)-2] > control_limits[len(arr)-2][1] + control_limits[len(arr)-2][3] and arr[len(arr)-3] > control_limits[len(arr)-3][1] + control_limits[len(arr)-3][3] and arr[len(arr)-4] > control_limits[len(arr)-4][1] + control_limits[len(arr)-4][3])
        or(arr[len(arr)-1] > control_limits[len(arr)-1][1] + control_limits[len(arr)-1][3] and arr[len(arr)-2] > control_limits[len(arr)-2][1] + control_limits[len(arr)-2][3] and arr[len(arr)-3] > control_limits[len(arr)-3][1] + control_limits[len(arr)-3][3] and arr[len(arr)-5] > control_limits[len(arr)-5][1] + control_limits[len(arr)-5][3])
        or(arr[len(arr)-1] > control_limits[len(arr)-1][1] + control_limits[len(arr)-1][3] and arr[len(arr)-2] > control_limits[len(arr)-2][1] + control_limits[len(arr)-2][3] and arr[len(arr)-4] > control_limits[len(arr)-4][1] + control_limits[len(arr)-4][3] and arr[len(arr)-5] > control_limits[len(arr)-5][1] + control_limits[len(arr)-5][3])
        or(arr[len(arr)-1] > control_limits[len(arr)-1][1] + control_limits[len(arr)-1][3] and arr[len(arr)-3] > control_limits[len(arr)-3][1] + control_limits[len(arr)-3][3] and arr[len(arr)-4] > control_limits[len(arr)-4][1] + control_limits[len(arr)-4][3] and arr[len(arr)-5] > control_limits[len(arr)-5][1] + control_limits[len(arr)-5][3])
        or(arr[len(arr)-2] > control_limits[len(arr)-2][1] + control_limits[len(arr)-2][3] and arr[len(arr)-3] > control_limits[len(arr)-3][1] + control_limits[len(arr)-3][3] and arr[len(arr)-4] > control_limits[len(arr)-4][1] + control_limits[len(arr)-4][3] and arr[len(arr)-5] > control_limits[len(arr)-5][1] + control_limits[len(arr)-5][3])):
      print("E")
      return False 
    elif ((arr[len(arr)-1] < control_limits[len(arr)-1][1] - control_limits[len(arr)-1][3] and arr[len(arr)-2] < control_limits[len(arr)-2][1] - control_limits[len(arr)-2][3] and arr[len(arr)-3] < control_limits[len(arr)-3][1] - control_limits[len(arr)-3][3] and arr[len(arr)-4] < control_limits[len(arr)-4][1] - control_limits[len(arr)-4][3])
        or(arr[len(arr)-1] < control_limits[len(arr)-1][1] - control_limits[len(arr)-1][3] and arr[len(arr)-2] < control_limits[len(arr)-2][1] - control_limits[len(arr)-2][3] and arr[len(arr)-3] < control_limits[len(arr)-3][1] - control_limits[len(arr)-3][3] and arr[len(arr)-5] < control_limits[len(arr)-5][1] - control_limits[len(arr)-5][3])
        or(arr[len(arr)-1] < control_limits[len(arr)-1][1] - control_limits[len(arr)-1][3] and arr[len(arr)-2] < control_limits[len(arr)-2][1] - control_limits[len(arr)-2][3] and arr[len(arr)-4] < control_limits[len(arr)-4][1] - control_limits[len(arr)-4][3] and arr[len(arr)-5] < control_limits[len(arr)-5][1] - control_limits[len(arr)-5][3])
        or(arr[len(arr)-1] < control_limits[len(arr)-1][1] - control_limits[len(arr)-1][3] and arr[len(arr)-3] < control_limits[len(arr)-3][1] - control_limits[len(arr)-3][3] and arr[len(arr)-4] < control_limits[len(arr)-4][1] - control_limits[len(arr)-4][3] and arr[len(arr)-5] < control_limits[len(arr)-5][1] - control_limits[len(arr)-5][3])
        or(arr[len(arr)-2] < control_limits[len(arr)-2][1] - control_limits[len(arr)-2][3] and arr[len(arr)-3] < control_limits[len(arr)-3][1] - control_limits[len(arr)-3][3] and arr[len(arr)-4] < control_limits[len(arr)-4][1] - control_limits[len(arr)-4][3] and arr[len(arr)-5] < control_limits[len(arr)-5][1] - control_limits[len(arr)-5][3])):
      print("F")
      return False  """
  """if len(arr) >= 7:
    if (arr[len(arr)-1] > control_limits[len(arr)-1][1]
        and arr[len(arr)-2] > control_limits[len(arr)-2][1]
        and arr[len(arr)-3] > control_limits[len(arr)-3][1]
        and arr[len(arr)-4] > control_limits[len(arr)-4][1]
        and arr[len(arr)-5] > control_limits[len(arr)-5][1] 
        and arr[len(arr)-6] > control_limits[len(arr)-6][1] 
        and arr[len(arr)-7] > control_limits[len(arr)-7][1] ):
      print("G")
      return False
    elif (arr[len(arr)-1] < control_limits[len(arr)-1][1] 
        and arr[len(arr)-2] < control_limits[len(arr)-2][1]
        and arr[len(arr)-3] < control_limits[len(arr)-3][1] 
        and arr[len(arr)-4] < control_limits[len(arr)-4][1]
        and arr[len(arr)-5] < control_limits[len(arr)-5][1]
        and arr[len(arr)-6] < control_limits[len(arr)-6][1]
        and arr[len(arr)-7] < control_limits[len(arr)-7][1] ):
      print("H")
      return False """
  """if len(arr) >=7:
    if (arr[len(arr)-1] > arr[len(arr)-2]
        and arr[len(arr)-2] > arr[len(arr)-3]
        and arr[len(arr)-3] > arr[len(arr)-4]
        and arr[len(arr)-4] > arr[len(arr)-5]
        and arr[len(arr)-5] > arr[len(arr)-6]
        and arr[len(arr)-6] > arr[len(arr)-7]):
      print("G")
      return False
    elif (arr[len(arr)-1] < arr[len(arr)-2]
        and arr[len(arr)-2] < arr[len(arr)-3]
        and arr[len(arr)-3] < arr[len(arr)-4]
        and arr[len(arr)-4] < arr[len(arr)-5]
        and arr[len(arr)-5] < arr[len(arr)-6]
        and arr[len(arr)-6] < arr[len(arr)-7]):
      print("H")
      return False
  if len(arr) >= 8:
    if ( arr[len(arr)-1] > control_limits[len(arr)-1][1] + control_limits[len(arr)-1][3]
        and arr[len(arr)-2] > control_limits[len(arr)-2][1] + control_limits[len(arr)-2][3]
        and arr[len(arr)-3] > control_limits[len(arr)-3][1] + control_limits[len(arr)-3][3]
        and arr[len(arr)-4] > control_limits[len(arr)-4][1] + control_limits[len(arr)-4][3]
        and arr[len(arr)-5] > control_limits[len(arr)-5][1] + control_limits[len(arr)-5][3]
        and arr[len(arr)-6] > control_limits[len(arr)-6][1] + control_limits[len(arr)-6][3]
        and arr[len(arr)-7] > control_limits[len(arr)-7][1] + control_limits[len(arr)-7][3]
        and arr[len(arr)-8] > control_limits[len(arr)-8][1] + control_limits[len(arr)-8][3]
        ):
      print("I")
      return False
    elif ( arr[len(arr)-1] < control_limits[len(arr)-1][1] - control_limits[len(arr)-1][3]
        and arr[len(arr)-2] < control_limits[len(arr)-2][1] - control_limits[len(arr)-2][3]
        and arr[len(arr)-3] < control_limits[len(arr)-3][1] - control_limits[len(arr)-3][3]
        and arr[len(arr)-4] < control_limits[len(arr)-4][1] - control_limits[len(arr)-4][3]
        and arr[len(arr)-5] < control_limits[len(arr)-5][1] - control_limits[len(arr)-5][3]
        and arr[len(arr)-6] < control_limits[len(arr)-6][1] - control_limits[len(arr)-6][3]
        and arr[len(arr)-7] < control_limits[len(arr)-7][1] - control_limits[len(arr)-7][3]
        and arr[len(arr)-8] < control_limits[len(arr)-8][1] - control_limits[len(arr)-8][3]
        ):
      print("J")
      return False """

  if len(arr) >= 15:
    if ((arr[len(arr)-1] > control_limits[len(arr)-1][1]  and arr[len(arr)-1] < control_limits[len(arr)-1][1] + control_limits[len(arr)-1][3] )
        and (arr[len(arr)-2] > control_limits[len(arr)-2][1] and arr[len(arr)-2] < control_limits[len(arr)-2][1] + control_limits[len(arr)-2][3])
        and (arr[len(arr)-3] > control_limits[len(arr)-3][1] and arr[len(arr)-3] < control_limits[len(arr)-3][1] + control_limits[len(arr)-3][3])
        and (arr[len(arr)-4] > control_limits[len(arr)-4][1] and arr[len(arr)-4] < control_limits[len(arr)-4][1] + control_limits[len(arr)-4][3])
        and (arr[len(arr)-5] > control_limits[len(arr)-5][1] and arr[len(arr)-5] < control_limits[len(arr)-5][1] + control_limits[len(arr)-5][3])
        and (arr[len(arr)-6] > control_limits[len(arr)-6][1] and arr[len(arr)-6] < control_limits[len(arr)-6][1] + control_limits[len(arr)-6][3])
        and (arr[len(arr)-7] > control_limits[len(arr)-7][1] and arr[len(arr)-7] < control_limits[len(arr)-7][1] + control_limits[len(arr)-7][3])
        and (arr[len(arr)-8] > control_limits[len(arr)-8][1] and arr[len(arr)-8] < control_limits[len(arr)-8][1] + control_limits[len(arr)-8][3])
        and (arr[len(arr)-9] > control_limits[len(arr)-9][1] and arr[len(arr)-9] < control_limits[len(arr)-9][1] + control_limits[len(arr)-9][3])
        and (arr[len(arr)-10] > control_limits[len(arr)-10][1] and arr[len(arr)-10] < control_limits[len(arr)-10][1] + control_limits[len(arr)-10][3])
        and (arr[len(arr)-11] > control_limits[len(arr)-11][1] and arr[len(arr)-11] < control_limits[len(arr)-11][1] + control_limits[len(arr)-11][3])
        and (arr[len(arr)-12] > control_limits[len(arr)-12][1] and arr[len(arr)-12] < control_limits[len(arr)-12][1] + control_limits[len(arr)-12][3])
        and (arr[len(arr)-13] > control_limits[len(arr)-13][1] and arr[len(arr)-13] < control_limits[len(arr)-13][1] + control_limits[len(arr)-13][3])
        and (arr[len(arr)-14] > control_limits[len(arr)-14][1] and arr[len(arr)-14] < control_limits[len(arr)-14][1] + control_limits[len(arr)-14][3])
        and (arr[len(arr)-15] > control_limits[len(arr)-15][1] and arr[len(arr)-15] < control_limits[len(arr)-15][1] + control_limits[len(arr)-15][3])):
      print("K")
      return False
    elif ((arr[len(arr)-1] < control_limits[len(arr)-1][1]  and arr[len(arr)-1] > control_limits[len(arr)-1][1] - control_limits[len(arr)-1][3] )
        and (arr[len(arr)-2] < control_limits[len(arr)-2][1] and arr[len(arr)-2] > control_limits[len(arr)-2][1] - control_limits[len(arr)-2][3])
        and (arr[len(arr)-3] < control_limits[len(arr)-3][1] and arr[len(arr)-3] > control_limits[len(arr)-3][1] - control_limits[len(arr)-3][3])
        and (arr[len(arr)-4] < control_limits[len(arr)-4][1] and arr[len(arr)-4] > control_limits[len(arr)-4][1] - control_limits[len(arr)-4][3])
        and (arr[len(arr)-5] < control_limits[len(arr)-5][1] and arr[len(arr)-5] > control_limits[len(arr)-5][1] - control_limits[len(arr)-5][3])
        and (arr[len(arr)-6] < control_limits[len(arr)-6][1] and arr[len(arr)-6] > control_limits[len(arr)-6][1] - control_limits[len(arr)-6][3])
        and (arr[len(arr)-7] < control_limits[len(arr)-7][1] and arr[len(arr)-7] > control_limits[len(arr)-7][1] - control_limits[len(arr)-7][3])
        and (arr[len(arr)-8] < control_limits[len(arr)-8][1] and arr[len(arr)-8] > control_limits[len(arr)-8][1] - control_limits[len(arr)-8][3])
        and (arr[len(arr)-9] < control_limits[len(arr)-9][1] and arr[len(arr)-9] > control_limits[len(arr)-9][1] - control_limits[len(arr)-9][3])
        and (arr[len(arr)-10] < control_limits[len(arr)-10][1] and arr[len(arr)-10] > control_limits[len(arr)-10][1] - control_limits[len(arr)-10][3])
        and (arr[len(arr)-11] < control_limits[len(arr)-11][1] and arr[len(arr)-11] > control_limits[len(arr)-11][1] - control_limits[len(arr)-11][3])
        and (arr[len(arr)-12] < control_limits[len(arr)-12][1] and arr[len(arr)-12] > control_limits[len(arr)-12][1] - control_limits[len(arr)-12][3])
        and (arr[len(arr)-13] < control_limits[len(arr)-13][1] and arr[len(arr)-13] > control_limits[len(arr)-13][1] - control_limits[len(arr)-13][3])
        and (arr[len(arr)-14] < control_limits[len(arr)-14][1] and arr[len(arr)-14] > control_limits[len(arr)-14][1] - control_limits[len(arr)-14][3])
        and (arr[len(arr)-15] < control_limits[len(arr)-15][1] and arr[len(arr)-15] > control_limits[len(arr)-15][1] - control_limits[len(arr)-15][3])):
      #print("L")
      return False
  if len(arr) >= 14:
    if ((arr[len(arr)-1] > arr[len(arr)-2])
        and (arr[len(arr)-2] < arr[len(arr)-3])
        and (arr[len(arr)-3] > arr[len(arr)-4])
        and (arr[len(arr)-4] < arr[len(arr)-5])
        and (arr[len(arr)-5] > arr[len(arr)-6])
        and (arr[len(arr)-6] < arr[len(arr)-7])
        and (arr[len(arr)-7] > arr[len(arr)-8])
        and (arr[len(arr)-8] < arr[len(arr)-9])
        and (arr[len(arr)-9] > arr[len(arr)-10])
        and (arr[len(arr)-10] < arr[len(arr)-11])
        and (arr[len(arr)-11] > arr[len(arr)-12])
        and (arr[len(arr)-12] < arr[len(arr)-13])
        and (arr[len(arr)-13] > arr[len(arr)-14]) ):
      #print("M")
      return False
    elif ((arr[len(arr)-1] < arr[len(arr)-2])
        and (arr[len(arr)-2] > arr[len(arr)-3])
        and (arr[len(arr)-3] < arr[len(arr)-4])
        and (arr[len(arr)-4] > arr[len(arr)-5])
        and (arr[len(arr)-5] < arr[len(arr)-6])
        and (arr[len(arr)-6] > arr[len(arr)-7])
        and (arr[len(arr)-7] < arr[len(arr)-8])
        and (arr[len(arr)-8] > arr[len(arr)-9])
        and (arr[len(arr)-9] < arr[len(arr)-10])
        and (arr[len(arr)-10] > arr[len(arr)-11])
        and (arr[len(arr)-11] < arr[len(arr)-12])
        and (arr[len(arr)-12] > arr[len(arr)-13])
        and (arr[len(arr)-13] < arr[len(arr)-14]) ):
      #print("N")
      return False
  return True
  



def simulator(sample_size = 1, chart1 = I_chart, chart2 = MR_chart, mean = 10, sigma = 0.5, mean_poisson = 10, mode = 'continuous', chart = 'double', chart_name = 'I_chart', const_sample_size = True, type1 = 0.01, type2 = 0.92):
  data = np.array([])
  if mode == "continuous":
    data = np.random.normal(loc = mean, scale = sigma, size = 50)
  else:
    data = np.random.poisson(lam = mean_poisson, size = 51 )

  
  false_alarms = 0
  response_time = 0
  mean0 = mean
  sigma0 = sigma
  mean_poisson0 = mean_poisson
  control_chart_data1 = np.array([])
  control_chart_data2 = np.array([])
  error_points = np.random.exponential(0.5,size = 4)
  mean_error_point = np.mean(error_points)
  scaling_factor = 300/mean_error_point
  error_points = error_points*scaling_factor
  error_points = np.sort(error_points.astype(np.int32))
  error_points += 50
  #print("Y",error_points)
  error_activations = np.array([0,0,0,0])
  error_resolutions = np.array([0,0,0,0])
  n_iterations = error_points[-1] + 400
  sample_sizes =  np.array([])
  LCL1 = 0
  center_line1 = 0
  UCL1 = 0
  sigma1 = 0
  LCL2 = 0
  center_line2 = 0
  UCL2 = 0
  sigma2 = 0
  p_bar = 0
  u_bar = 0
  errors = np.array([])
  t_false = 4
  t_rect = 10
  P_R = 1
  S_R = 0.5
  profit = 10
  false_alarm_investigation_cost = 5
  c_rect = 6
  c_sample = 0.5
  CTU_array = []
  response_time_array = []
  j0 = 0
  img_array = []
  

  if chart == 'double':

    if sample_size == 1:
      x = np.array([])
      mr = np.array([])
      for i in range(1,len(data)+1):
        x = I_chart(x,data[:i],sample_size)
        mr = MR_chart(mr, data[:i],sample_size)
      x_bar = np.mean(x)
      mr_bar = np.mean(mr)
      LCL1 = x_bar - 2.66*mr_bar
      center_line1 = x_bar
      UCL1 = x_bar + 2.66*mr_bar
      sigma1 = mr_bar/df['d2'][0]
      LCL2 = 0
      center_line2 = mr_bar
      UCL2 = 3.27*mr_bar
      sigma2 = (mr_bar/df['d2'][0])*df['d3'][0]

    elif sample_size > 1 and sample_size < 11:
      x_bar = np.array([])
      r = np.array([])
      for i in range(1,len(data)+1):
        x_bar = X_chart(x_bar,data[:i],sample_size)
        r = R_chart(r, data[:i],sample_size)
      
      x_bar_bar = np.mean(x_bar)
      r_bar = np.mean(r)


      LCL1 = x_bar_bar - df['A2'][sample_size-1]*r_bar
      UCL1 = x_bar_bar + df['A2'][sample_size-1]*r_bar
      center_line1 = x_bar_bar
      sigma1 = (r_bar/df['d2'][sample_size-1])*df['d3'][sample_size-1]
      LCL2 = r_bar*df['D3'][sample_size-1]
      UCL2 = r_bar*df['D4'][sample_size-1]
      center_line2 = r_bar
      sigma2 = (r_bar/df['d2'][sample_size-1])*df['d3'][sample_size-1]
      
      

    else:
      x_bar = np.array([])
      s = np.array([])
      for i in range(1,len(data)+1):
        x_bar = X_chart(x_bar,data[:i],sample_size)
        s = S_chart(s, data[:i],sample_size)

      x_bar_bar = np.mean(x_bar)
      s_bar = np.mean(s)
      LCL1 = x_bar_bar - df['A3'][sample_size-1]*s_bar
      UCL1 = x_bar_bar + df['A3'][sample_size-1]*s_bar
      center_line1 = x_bar_bar
      sigma1 = s_bar/df['d2'][sample_size-1]
      LCL2 = s_bar*df['B3'][sample_size-1]
      UCL2 = s_bar*df['B4'][sample_size-1]
      center_line2 = s_bar
      sigma2 = (s_bar/df['d2'][0])*df['d3'][sample_size-1]
  elif const_sample_size:
    if chart_name == 'np_chart':
      p = np.array([])
      for i in range(1,len(data) + 1):
        p = np_chart(p,data[:i],sample_size, mean0, sigma0)
      p_bar = np.mean(p)
      LCL1 = sample_size*p_bar - 2.7*np.sqrt(sample_size*p_bar*(1 - p_bar))
      UCL1 = sample_size*p_bar + 2.7*np.sqrt(sample_size*p_bar*(1 - p_bar))
      center_line1 = sample_size*p_bar
      sigma1 = np.sqrt(sample_size*p_bar*(1 - p_bar))
    else:
      c = np.array([])
      for i in range(1,len(data) + 1):
        c = C_chart(c,data[:i],sample_size)
      c_bar = np.mean(c)
      #print("C",c_bar)
      new_process_mean = c_bar + np.sqrt(c_bar)
      rv1 = poisson(c_bar)
      rv2 = poisson(new_process_mean)
      z_score = rv1.ppf(1 - type1/2)
      k = (z_score - c_bar)/np.sqrt(c_bar)
      #print("R",c_bar,new_process_mean)
      a1 = 1 - (rv2.cdf(c_bar + k*np.sqrt(c_bar)))
      a2 = rv2.cdf(c_bar - k*np.sqrt(c_bar))
      a = 1 - (a1 + a2)
      #print("A",a)
      if a > type2:

        print("L")
        while a > type2:

          k -= 0.01
          a1 = 1 - (rv2.cdf(c_bar + k*np.sqrt(c_bar)))
          a2 = rv2.cdf(c_bar - k*np.sqrt(c_bar))
          a = 1 - (a1 + a2)
      print("A",a,k)

          



      UCL1 = c_bar + k*np.sqrt(c_bar)
      LCL1 = c_bar - k*np.sqrt(c_bar)
      center_line1 =  c_bar
      sigma1 = np.sqrt(c_bar)
  else:
    if chart_name == "p_chart":
      p = np.array([])

      for i in range(1,len(data) + 1):
        p = p_chart(p,data[:i],mean0, sigma0)
      sum = 0
      
      for i,k in p:
        sum += i
      p_bar = sum/len(p)
      #print("P-bar",p_bar)

        



    else:
      u = np.array([])
      for i in range(1,len(data) + 1):
        u = U_chart(u,data[:i])
      sum = 0
      
      for i,k in u:
        sum += i
      u_bar = sum/len(u)
      #print("u-bar",u_bar)



  if mode == 'continuous':
    data = np.append(data,np.random.normal(loc = mean, scale = sigma, size = 1))
    
    
    j = 51
    while j < n_iterations:
      if chart == 'double':
        control_chart_data1 = chart1(control_chart_data1,data[50:],sample_size)
        
        control_chart_data2 = chart2(control_chart_data1,data[50:],sample_size)
        
        control_chart_input1 = np.array([])
        if len(control_chart_data1) <= 15:
          control_chart_input1 = control_chart_data1
        else:
          control_chart_input1 = control_chart_data1[-15:]

        control_chart_input2 = np.array([])
        if len(control_chart_data2) <= 15:
          control_chart_input2 = control_chart_data2
        else:
          control_chart_input2 = control_chart_data2[-15:]
       
        control_limits1 = np.tile(np.array([LCL1,center_line1,UCL1,sigma1]),(len(control_chart_input1),1))
        control_limits2 =  np.tile(np.array([LCL2,center_line2,UCL2,sigma2]),(len(control_chart_input2),1))
        #if j < 150:
          #print("CCI",control_chart_input1, len(control_chart_input1))
          #print("CCL",control_limits1, len(control_limits1))
        check1 = control_check(control_chart_input1,control_limits1)
        check2 = control_check(control_chart_input2,control_limits2)
        x_axis = np.array([])
        for i in range(1,len(control_chart_input1) + 1):


          x_axis = np.append(x_axis,i)
        if len(control_limits1) != 0 and len(data[50:])%sample_size == 0:
                plt.figure(figsize=(2, 2))
                plt.plot(x_axis, control_chart_input1)
                plt.plot(x_axis,control_limits1[:,0])
                plt.plot(x_axis,control_limits1[:,1])
                plt.plot(x_axis,control_limits1[:,2])

                #plt.text(0.5,-1,"Process Shift")
                plt.xlabel('Samples')
                plt.ylabel('Control_chart_data_points')
                plt.title('I Chart')
                #plt.savefig(f"Plott{j0}.jpg")
                #img = cv2.imread(f"Plott{j0}.jpg")
                #img_array.append(img)
                j0 += 1
        if not check1:

          #print("ERROR DETECTED AT " + str(j))
  
          if len(errors) != 0:
            #print("Process Shift Detected at " + str(j))
            #print("Process Shift introduced at " + str(int(error_points[int(errors[0])])))
            #print("ER",errors)
            for i in errors:
              error_resolutions[int(i)] = 1

            
            #print("J",j)
            #errors = np.array([])
            mean = mean0
            sigma = sigma0
            #print("M",mean)
            #print("S",sigma)
            response_time = j - int(error_points[int(errors[0])])

            j = int(error_points[int(errors[0])])
            #print("J",j)
            #errors = np.array([])
            #print("P1",mean_poisson)
            mean_poisson = mean_poisson0
            #print("P",mean_poisson)
            if errors[0] == 0:
              t_in = false_alarms*t_false + error_points[0] - 50
              c_in = c_sample*(error_points[0] - 50) + false_alarms*t_false*P_R*profit + false_alarms*false_alarm_investigation_cost
            else:
              t_in = false_alarms*t_false + error_points[int(errors[0])] - error_points[int(errors[0])-1]
              c_in = c_sample*(error_points[int(errors[0])] - error_points[int(errors[0])-1]) + false_alarms*t_false*P_R*profit + false_alarms*false_alarm_investigation_cost
            t_out = response_time
            c_out = c_sample*response_time
            t_cycle = t_in + t_out + t_rect
            c_cycle = c_in + c_out + c_rect
            false_alarms = 0
            errors = np.array([])
            CTU_array.append(c_cycle/t_cycle)
            response_time_array.append(response_time)
            if len(control_limits1) != 0 and len(data[50:])%sample_size == 0:
                plt.figure(figsize=(2, 2))
                plt.plot(x_axis, control_chart_input1)
                plt.plot(x_axis,control_limits1[:,0])
                plt.plot(x_axis,control_limits1[:,1])
                plt.plot(x_axis,control_limits1[:,2])

                #plt.text(0.5,-1,"Process Shift")
                plt.xlabel('Samples')
                plt.ylabel('Control_chart_data_points')
                plt.title('Process Shift')
                #plt.savefig(f"Plott{j0}.jpg")
                #img = cv2.imread(f"Plott{j0}.jpg")
                #img_array.append(img)
                j0 += 1
            #print("CF",CTU_array,response_time_array)

          else:
            #print("False Alarm Detected at " + str(j))
            false_alarms += 1
            if len(control_limits1) != 0 and len(data[50:])%sample_size == 0:
                plt.figure(figsize=(2, 2))
                plt.plot(x_axis, control_chart_input1)
                plt.plot(x_axis,control_limits1[:,0])
                plt.plot(x_axis,control_limits1[:,1])
                plt.plot(x_axis,control_limits1[:,2])

                #plt.text(1,0.5,"False Alarm")
                plt.xlabel('Samples')
                plt.ylabel('Control_chart_data_points')
                plt.title('False Alarm')
                #plt.savefig(f"Plottt{j0}.jpg")
                
                #img = cv2.imread(f"Plottt{j0}.jpg")
                #img_array.append(img)
                j0 += 1 
            #print("F",false_alarms)

     


      elif const_sample_size:
        control_chart_data1 = chart1(control_chart_data1,data[50:],sample_size,mean0,sigma0)
        control_chart_input1 = np.array([])
        if j == 51:
          control_chart_input0 = np.array([])
        if len(control_chart_data1) <= 15:
          control_chart_input1 = control_chart_data1
        else:
          control_chart_input1 = control_chart_data1[-15:]
        control_limits1 = np.tile(np.array([LCL1,center_line1,UCL1,sigma1]),(len(control_chart_input1),1))
        check1 = control_check(control_chart_input1,control_limits1)
        if not check1:
          #print("ERROR",j)
          if len(errors) != 0:

            #print("Process Shift")
            #print("ER",errors)
            for i in errors:
              error_resolutions[int(i)] = 1

            j = int(error_points[int(errors[0])])
            #print("J",j)
            errors = np.array([])
            #print("M1",mean)
            #print("S1",sigma)
            mean = mean0
            sigma = sigma0
            #print("M",mean)
            #print("S",sigma)
          else:
            #print("False Alarm",j)
            pass
          


      else:

        control_chart_data1 = chart1(control_chart_data1,data[50:],mean0,sigma0)
        control_chart_input1 = np.array([])

        if len(control_chart_data1) <= 15:
          control_chart_input1 = control_chart_data1
          
        else:
          control_chart_input1 = control_chart_data1[-15:]
        control_chart_input = np.array([])
        control_limits1 = np.empty((0, 4))
        
        for i,k in control_chart_input1:

          LCL1 = p_bar - 2*np.sqrt(p_bar*(1 - p_bar)/k)
          center_line1 = p_bar
          UCL1 = p_bar + 2*np.sqrt(p_bar*(1 - p_bar)/k)
          sigma1 = np.sqrt(p_bar*(1 - p_bar)/k)
          control_chart_input = np.append(control_chart_input,i)
          control_limits1 = np.append(control_limits1,[[LCL1,center_line1,UCL1,sigma1]], axis = 0)
        #print("CCI",control_chart_input, len(control_chart_input1))
        #print("CCL",control_limits1, len(control_limits1))

        #check1 = control_check(control_chart_input,control_limits1)
        #if not check1:
          #print("ERROR",j)
          #for k in range(len(error_activations)):
           # if error_activations[k] == 1:
              #error_activations[k] = 0
              #j = error_points[k]

       
        

        


        





      if j in error_points:
        #print("E")
        
        if error_resolutions[np.where(error_points == j)[0][0]] == 0:
          #print("E1")
          error_index = np.where(error_points == j)[0][0]
          errors = np.append(errors,error_index)
          if error_index%2 == 0:
            r = np.random.choice([0,1])
            if r == 0:
              mean -= np.random.rand() + 1
            else:
              mean += np.random.rand() + 1
          else:
            r = np.random.choice([0,1])
            if r == 0:
              sigma -= 0.05 + np.random.rand()/10
            else:
              sigma +=  0.05 + np.random.rand()/10
            
          data = np.append(data,np.random.normal(loc = mean, scale = sigma, size = 1))
          j += 1
        else:
          data = np.append(data,np.random.normal(loc = mean, scale = sigma, size = 1))
          j += 1

        
      else:
        data = np.append(data,np.random.normal(loc = mean, scale = sigma, size = 1))
        j += 1
  else :
    j = 51
    while j < n_iterations:
      if const_sample_size:
        
        control_chart_data1 = chart1(control_chart_data1,data[50:],sample_size)
        
        control_chart_input1 = np.array([])
        
        
        if len(control_chart_data1) <= 15:
          control_chart_input1 = control_chart_data1
        else:
          control_chart_input1 = control_chart_data1[-15:]
        if j == 51:
          control_chart_input0 = control_chart_input1
        control_limits1 = np.tile(np.array([LCL1,center_line1,UCL1,sigma1]),(len(control_chart_input1),1))
        check1 = control_check(control_chart_input1,control_limits1)
        #print("CCI",control_chart_input1)
        #print("CCL",control_limits1)
        x_axis = np.array([])
        for i in range(1,len(control_chart_input1) + 1):

          x_axis = np.append(x_axis,i)
        
        if len(control_limits1) != 0 and len(data[50:])%sample_size == 0:
          plt.figure(figsize=(2, 2))
          plt.plot(x_axis, control_chart_input1)
          plt.plot(x_axis,control_limits1[:,0])
          plt.plot(x_axis,control_limits1[:,1])
          plt.plot(x_axis,control_limits1[:,2])

        
          plt.xlabel('Samples')
          plt.ylabel('Control_chart_data_points')
          plt.title('C Chart')
          plt.savefig(f"Plot{j0}.jpg")
          img = cv2.imread(f"Plot{j0}.jpg")
          img_array.append(img)
          j0 += 1 
          

# Show the plot
         # plt.show()
        if len(control_chart_input1) != 0 and len(control_chart_input0) != 0:
          if not check1 and control_chart_input1[-1] != control_chart_input0[-1]:
          
            #print("ERROR",j)
            if len(errors) != 0:

              print("Process Shift",j,int(error_points[int(errors[0])]))
              #print("ER",errors)
              x_axis = np.array([])
              for i in range(1,len(control_chart_input1) + 1):

                 x_axis = np.append(x_axis,i)
              
              if len(control_limits1) != 0 and len(data[50:])%sample_size == 0:
                plt.figure(figsize=(2, 2))
                plt.plot(x_axis, control_chart_input1)
                plt.plot(x_axis,control_limits1[:,0])
                plt.plot(x_axis,control_limits1[:,1])
                plt.plot(x_axis,control_limits1[:,2])

                #plt.text(0.5,-1,"Process Shift")
                plt.xlabel('Samples')
                plt.ylabel('Control_chart_data_points')
                plt.title('Process Shift')
                plt.savefig(f"Plott{j0}.jpg")
                img = cv2.imread(f"Plott{j0}.jpg")
                img_array.append(img)
                j0 += 1
                #plt.show() """
              for i in errors:
                error_resolutions[int(i)] = 1
              response_time = j - int(error_points[int(errors[0])])

              j = int(error_points[int(errors[0])])
              #print("J",j)
              #errors = np.array([])
              #print("P1",mean_poisson)
              mean_poisson = mean_poisson0
              #print("P",mean_poisson)
              if errors[0] == 0:
                t_in = false_alarms*t_false + error_points[0] - 50
                c_in = c_sample*(error_points[0] - 50) + false_alarms*t_false*P_R*profit + false_alarms*false_alarm_investigation_cost
              else:
                t_in = false_alarms*t_false + error_points[int(errors[0])] - error_points[int(errors[0])-1]
                c_in = c_sample*(error_points[int(errors[0])] - error_points[int(errors[0])-1]) + false_alarms*t_false*P_R*profit + false_alarms*false_alarm_investigation_cost
              t_out = response_time
              c_out = c_sample*response_time
              t_cycle = t_in + t_out + t_rect
              c_cycle = c_in + c_out + c_rect
              false_alarms = 0
              errors = np.array([])
              CTU_array.append(c_cycle/t_cycle)
              response_time_array.append(response_time)
              #print("CF",CTU_array,response_time_array)
              








              

       
            
            
            else:
              print("False Alarm",j)
              false_alarms += 1
              #print("F",false_alarms)

              if len(control_limits1) != 0 and len(data[50:])%sample_size == 0:
                plt.figure(figsize=(2, 2))
                plt.plot(x_axis, control_chart_input1)
                plt.plot(x_axis,control_limits1[:,0])
                plt.plot(x_axis,control_limits1[:,1])
                plt.plot(x_axis,control_limits1[:,2])

                #plt.text(1,0.5,"False Alarm")
                plt.xlabel('Samples')
                plt.ylabel('Control_chart_data_points')
                plt.title('False Alarm')
                plt.savefig(f"Plottt{j0}.jpg")
                
                img = cv2.imread(f"Plottt{j0}.jpg")
                img_array.append(img)
                j0 += 1 
                #plt.show()
        

        control_chart_input0 = control_chart_input1
          

        
        
   
        
        

        



      else:
        #U chart
        control_chart_data1 = chart1(control_chart_data1,data[50:])
        control_chart_input1 = np.array([])
        if len(control_chart_data1) <= 15:
          control_chart_input1 = control_chart_data1
          
        else:
          control_chart_input1 = control_chart_data1[-15:]
        control_chart_input = np.array([])
        control_limits1 = np.empty((0, 4))
        new_process_mean = u_bar + np.sqrt(u_bar)
        rv1 = poisson(u_bar)
        rv2 = poisson(new_process_mean)
        z_score = rv1.ppf(1 - type1/2)
        k = (z_score - u_bar)/np.sqrt(u_bar)
        #print("R",u_bar,new_process_mean)
        a1 = 1 - (rv2.cdf(u_bar + k*np.sqrt(u_bar)))
        a2 = rv2.cdf(u_bar - k*np.sqrt(u_bar))
        a = 1 - (a1 + a2)
        #print("A",a)
        if a > type2:

          #print("L")
          while a > type2:

            k -= 0.1
            a1 = 1 - (rv2.cdf(u_bar + k*np.sqrt(u_bar)))
            a2 = rv2.cdf(u_bar - k*np.sqrt(u_bar))
            a = 1 - (a1 + a2)
        #print(a,k)

          



      
        for i,l in control_chart_input1:

          LCL1 = u_bar - k*np.sqrt(u_bar/l)
          center_line1 = u_bar
          UCL1 = u_bar + k*np.sqrt(u_bar/l)
          sigma1 = np.sqrt(u_bar/l)
          control_chart_input = np.append(control_chart_input,i)
          control_limits1 = np.append(control_limits1,[[LCL1,center_line1,UCL1,sigma1]], axis = 0)
        #print("CCI",control_chart_input, len(control_chart_input1))
        #print("CCL",control_limits1, len(control_limits1))
        control_chart_input = np.array([])
        for i in control_chart_input1:
          control_chart_input = np.append(control_chart_input,i[0])
        check1 = control_check(control_chart_input,control_limits1)
        #print("CCI",control_chart_input1)
        #print("CCL",control_limits1)
        if not check1:
          print("ERROR DETECTED")
          if len(errors) != 0:

            print("Process Shift detected at: " + str(j))
            print("Process shift introduced at: " + str(int(error_points[int(errors[0])])))
            
            for i in errors:
              error_resolutions[int(i)] = 1
            response_time = j - int(error_points[int(errors[0])])
            j = int(error_points[int(errors[0])])
            #print("J",j)
            #errors = np.array([])
            #print("P1",mean_poisson)
            mean_poisson = mean_poisson0
            #print("P",mean_poisson)
            if errors[0] == 0:
              t_in = false_alarms*t_false + error_points[0] - 50
              c_in = c_sample*(error_points[0] - 50) + false_alarms*t_false*P_R*profit + false_alarms*false_alarm_investigation_cost
            else:
              t_in = false_alarms*t_false + error_points[int(errors[0])] - error_points[int(errors[0])-1]
              c_in = c_sample*(error_points[int(errors[0])] - error_points[int(errors[0])-1]) + false_alarms*t_false*P_R*profit + false_alarms*false_alarm_investigation_cost
            t_out = response_time
            c_out = c_sample*response_time
            t_cycle = t_in + t_out + t_rect
            c_cycle = c_in + c_out + c_rect
            false_alarms = 0
            errors = np.array([])
            CTU_array.append(c_cycle/t_cycle)
            response_time_array.append(response_time)
            #print("CF",CTU_array,response_time_array)

          else:
            print("False Alarm detected at: " + str(j))
            false_alarms += 1
            #print("F",false_alarms)

        



      if j in error_points:
        #print("E")
        
        if error_resolutions[np.where(error_points == j)[0][0]] == 0:
          #print("E1")
          error_index = np.where(error_points == j)[0][0]
          errors = np.append(errors,error_index)
          if error_index%2 == 0:
            r = np.random.choice([0,1])
            if r == 0:
              mean_poisson += np.random.poisson(lam = 1)
            else:
              mean_poisson -= np.random.poisson(lam = 1)
          else:
            r = np.random.choice([0,1])
            if r == 0:
              mean_poisson += np.random.poisson(lam = 1)
            else:
              mean_poisson -= np.random.poisson(lam = 1)
            
          data = np.append(data,np.random.poisson(lam = mean_poisson, size = 1))
          j += 1
        else:
          data = np.append(data,np.random.poisson(lam = mean_poisson, size = 1))
          j += 1


      else:
        data = np.append(data,np.random.poisson(lam = mean_poisson, size = 1))
        j += 1
  """height, width, layers = img_array[0].shape
  size = (width, height)
  out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, size) """
  return [[CTU_array,response_time_array],np.sum(error_resolutions)]
 
  
  
  
  
  


        

#print(simulator(sample_size = 2 ,chart1 = C_chart,chart_name = "C_chart",chart = "single", mode = 'discrete', const_sample_size = True))
#print(simulator(sample_size = 5 ,chart1 = C_chart,chart_name = "C_chart",chart = "single", mode = 'discrete', const_sample_size = True))
#print(simulator(sample_size = 1 ,chart1 = X_chart,chart2 = R_chart,chart_name = "X_chart",chart = "double", mode = 'continuous', const_sample_size = True))
result = simulator(chart1 = U_chart,chart_name = "U_chart",chart = "single", mode = 'discrete', const_sample_size = False)
if result[1] == 4:
  print("All process shifts were detected")
  print("CTU array: " + str(result[0][0]))
  print("Response Time array: " + str(result[0][1]))

"""results = simulator()
if results[1] == 4:
  print("The control chart detected all the process shifts!")
  cost_sum = np.sum(results[0][0])
  time_sum = np.sum(results[0][1])
  print("Array of CTUs : " + str(results[0][0]))
  print("Array of Response Times: " + str(results[0][1]))
  print("Objective value: " + str(1000*cost_sum + time_sum))
else:
  print("The control chart failed to detect all the process shifts")"""



"""optimal_sample_sizes = []
for _ in range(20):"""
"""optimal_sample_size = 0
min_objective = 1000000
for i in range(2,15):

  sol = [i,simulator(sample_size = i ,chart1 = X_chart,chart2 = R_chart,chart_name = "X_chart",chart = "double", mode = 'continuous', const_sample_size = True)]

  if sol[1][1] == 4:
    cost_sum = np.sum(sol[1][0][0])
    time_sum = np.sum(sol[1][0][1])
  print("CTU_Array for sample size " + str(i) + ":" + str(sol[1][0][0]))
  print("Response Time array for sample size " + str(i) + ":" + str(sol[1][0][1]))
  print("Total Cost : " + str(cost_sum))
  print("Total Response Time : " + str(time_sum))
  print("Objective : " + str(cost_sum*1000 + time_sum))
  min_objective = min(min_objective,cost_sum*1000 + time_sum)
  if cost_sum*1000 + time_sum == min_objective:
    optimal_sample_size = i
    

  #print("I",i,cost_sum,time_sum, cost_sum*1000 + time_sum)
  #print("OVER")
#sol = simulator(chart1 = U_chart,chart_name = "U_chart",chart = "single", mode = 'discrete', const_sample_size = False)
#if sol[1][2] == 4:
print("Optimal Sample Size :" + str(optimal_sample_size))
optimal_sample_sizes.append(optimal_sample_size)
  print("P",optimal_sample_sizes)"""
"""counts = np.bincount(np.array(optimal_sample_sizes).astype(int))
max_count_index = np.argmax(counts)
print(max_count_index) """


#cost_sum = np.sum(sol[1][1][0])
#time_sum = np.sum(sol[1][1][1])
#print("I",cost_sum,time_sum, cost_sum*1000 + time_sum)











      
      

	





	
 

	  




        
   
    



   
  
  

  

  
  
  

