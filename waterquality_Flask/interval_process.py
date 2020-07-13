import numpy as np

from sklearn import neighbors
from sklearn import linear_model
from sklearn import ensemble
from sklearn.ensemble import BaggingRegressor

from sklearn.linear_model import BayesianRidge


import matplotlib.pyplot as plt
plt.switch_backend('agg')
import time
from sklearn.externals import joblib
import os



os.environ["CUDA_VISIBLE_DEVICES"]="0"

from sklearn.model_selection import KFold
from keras.models import load_model
from keras import backend as K
import keras

import classic_regression as cr


BIN_SIZE = 10
BOUND = 300
LARGE_SIZE = 50

def send_to_bin(predicts = [[]], y_test = []):
    bins = [[] for i in range(0,31)]
    bins_true = [[] for i in range(0,31)]

    for predict in predicts:
        for index, value in enumerate(predict):
            x = 0
            while x<300:
                if x < value <=x+10:
                    bins[x//10].append(value)
                    bins_true[x//10].append(y_test[index])
                x += 10

            if value>=300:
                bins[30].append(value)
                bins_true[30].append(y_test[index])

    return bins, bins_true

def send_true_to_bin(predicts = [[]], y_test = []):
    bins = [[] for i in range(0,31)]
    bins_true = [[] for i in range(0,31)]

    for predict in predicts:
        for index, value in enumerate(y_test):
            x = 0
            while x<300:
                if x < value <=x+10:
                    bins[x//10].append(predict[index])
                    bins_true[x//10].append(value)
                x += 10

            if value>=300:
                bins[30].append(predict[index])
                bins_true[30].append(value)


    return bins, bins_true

def bin_process(bins = [[]], bins_true=[[]]):


    # print (bins)
    # bins = [x for x in bins if x != []]
    # bins_true = [x for x in bins_true if x != []]

    upper = [[] for i in range(0, 31)]
    lower = [[] for i in range(0, 31)]

    for bin_index, mybin in enumerate(bins):

        for index, value in enumerate(mybin):

            # print(bin_index)
            # print(index,value)

            if value >= bins_true[bin_index][index]:
                upper[bin_index].append(value - bins_true[bin_index][index])
            else:
                lower[bin_index].append(bins_true[bin_index][index] - value)
    # upper = [x for x in upper if x != []]
    # lower = [x for x in lower if x != []]


    final_upper = []
    final_lower = []
    for item in upper:
        if item!=[]:
            # final_upper.append(np.percentile(item,95))
            final_upper.append(np.percentile(item,100))
        else:
            final_upper.append(0)
    for item in lower:
        if item!=[]:
            # final_lower.append(np.percentile(item,95))
            final_lower.append(np.percentile(item,100))
        else:
            final_lower.append(0)

    return final_upper,final_lower


def get_upper(upper_model, x):

    return upper_model.predict([[x]])[0]

def get_lower(lower_model, x):

    return lower_model.predict([[x]])[0]


def draw_test(result=[], y_test=[], upper=[],lower=[], smooth_bin = True,upper_model=-1,lower_model=-1):
    plt.figure(figsize=(40,10))

    for item in result:
        if item<0: item=0

    print(upper)
    print(lower)
    np.save(dirs + "/" + "upper" + ".npy", upper)
    np.save(dirs + "/" + "lower" + ".npy", lower)



    u = []
    l = []

    if not smooth_bin:
        for value in result:
            if value > 300:
                u.append(value + upper[30])
                l.append(value - lower[30])
            else:
                u.append(value + upper[int(value//10)])
                l.append(value - lower[int(value // 10)])

    else:
        for value in result:
            if value > 300:
                u.append(value + get_upper(upper_model,30))
                l.append(value - get_lower(lower_model,30))
            else:
                u.append(value + upper[int(value//10)])
                l.append(value - lower[int(value // 10)])

    list_all = sorted(list(zip(l,y_test,u)),key = lambda x:x[1])

    nl=[]
    ny_test = []
    nu=[]

    targeted = 0

    targeted_x=[]
    targeted_y=[]

    untargeted_x=[]
    untargeted_y=[]

    for i,item in enumerate(list_all):
        nl.append(item[0])
        ny_test.append(item[1])
        nu.append(item[2])
        if item[0]<=item[1]<=item[2]:
            targeted+=1
            targeted_x.append(i)
            targeted_y.append(item[1])
        else:
            untargeted_x.append(i)
            untargeted_y.append(item[1])




    targeted_rate = targeted/len(nl)
    #plt.fill_between(np.arange(len(u)), u, l, facecolor="green")
    # plt.scatter(np.arange(len(result)), ny_test, c='r',label = "True y")
    plt.scatter(targeted_x,targeted_y,c='r',label="Targeted y")
    plt.scatter(untargeted_x,untargeted_y,c='blue',label="Untargeted y")
    plt.scatter(np.arange(len(nu)), nu, c='black',marker='_')
    plt.scatter(np.arange(len(nl)), nl, c='black', marker='_')
    plt.plot(np.arange(len(nu)), nu, c='black')
    plt.plot(np.arange(len(nl)), nl, c='black')
    plt.title("Targeted rate: %f" % targeted_rate)

    print("Targeted rate: %f" % targeted_rate)

    for x in np.arange(len(nu)):
        plt.fill_between(np.linspace(x-0.5,x+0.5), nu[x], nl[x], facecolor="green",alpha = 0.5)

    plt.legend()
    plt.savefig(dirs + "/" + "interval" + ".png", dpi=300)

    plt.figure(figsize=(50,10))

    interval_width = []
    for i in range(len(upper)):
        interval_width.append(upper[i] + lower[i])

    bin_name = []
    for i, item in enumerate(interval_width):
        bin_name.append(str(i*10))
    bin_name[-1]="300+"
    plt.plot(bin_name, interval_width)
    plt.savefig(dirs + "/" + "interval_width" + ".png", dpi=300)

    plt.figure(figsize=(50,10))

    tu=[]
    for value in result:
            if value > 300:
                tu.append(value + upper[30])
            else:
                tu.append(value + upper[int(value//10)])

    plt.plot(np.arange(len(tu)),tu)
    plt.plot(np.arange(len(u)),u)

    error = 0
    for i,item in enumerate(tu):
        error += (item - u[i])*(item - u[i])

    print ("ERROR:%f" % (error/len(tu)))

    plt.savefig(dirs + "/" + "interval_smooth" + ".png", dpi=300)


def transform_y(y_pred, scaler_test):

    y_pred = y_pred.reshape(-1, 1)

    y_pred = scaler_test.inverse_transform(y_pred)

    y_pred = y_pred.flatten()
    return y_pred

def reinitLayers(model):
    session = K.get_session()
    for layer in model.layers:
        if isinstance(layer, keras.engine.network.Network):
            reinitLayers(layer)
            continue
        print("LAYER::", layer.name)
        for v in layer.__dict__:
            v_arg = getattr(layer,v)
            if hasattr(v_arg,'initializer'):
                initializer_method = getattr(v_arg, 'initializer')
                initializer_method.run(session=session)
                print('reinitializing layer {}.{}'.format(layer.name, v))

def train_smoother(upper,lower):
    upper_model = BayesianRidge()
    lower_model = BayesianRidge()

    bin_x = []
    for i, item in enumerate(upper):
        bin_x.append([i*10])

    upper_model.fit(bin_x,upper)
    lower_model.fit(bin_x,lower)

    joblib.dump(upper_model, dirs+'/upper_model.pkl')
    joblib.dump(lower_model, dirs+'/lower_model.pkl')

    return upper_model,lower_model

def load_smoother(path):

    u_model = joblib.load(path+'/upper_model.pkl')
    l_model = joblib.load(path+'/lower_model.pkl')

    return u_model,l_model


def new_model(model=-1, empty_model=-1, mode = "NN",inversed=False, X_train=-1, y_train=-1, X_test=-1, y_test=-1):

    kf = KFold(n_splits=5)

    scaler_val = joblib.load('MaxAbsScaler.pkl')

    predicts = []
    y_interval = []

    model.fit(X_train,y_train)
    result = model.predict(X_test)

    if mode == "NN":
        for train_index, test_index in kf.split(X_train):
            X_train_interval, X_test_interval = X_train[train_index], X_train[test_index]
            y_train_interval, y_test_interval = y_train[train_index], y_train[test_index]

            reinitLayers(model)
            model.fit(X_train_interval,y_train_interval, batch_size = 4, epochs=200)
            predict = model.predict(X_test_interval)
            for item in predict:
                predicts.append(item)
            for item in y_test_interval:
                y_interval.append(item)
    else:
        X_train = np.reshape(X_train, (X_train.shape[0], -1))
        X_test = np.reshape(X_test, (X_test.shape[0], -1))

        for train_index, test_index in kf.split(X_train):
            X_train_interval, X_test_interval = X_train[train_index], X_train[test_index]
            y_train_interval, y_test_interval = y_train[train_index], y_train[test_index]
            model = empty_model
            model.fit(X_train_interval,y_train_interval)
            predict = model.predict(X_test_interval)
            for item in predict:
                predicts.append(item)
            for item in y_test_interval:
                y_interval.append(item)

    np.save(dirs + "/" + "predicts" + ".npy", predicts)
    np.save(dirs + "/" + "y_interval" + ".npy", y_interval)
    np.save(dirs + "/" + "result" + ".npy", result)

    if inversed:
        predicts = transform_y(predicts, scaler_val)
        result = transform_y(result, scaler_val)

    return result,predicts,y_interval

def load_result(path):

    predicts = np.load(path + "/" + "predicts" + ".npy")
    y_interval = np.load(path + "/" + "y_interval" + ".npy")
    result = np.load(path + "/" + "result.npy")

    upper = np.load(path + "/" + "upper.npy")
    lower = np.load(path + "/" + "lower.npy")

    return result,predicts,y_interval,upper,lower

def aug(x,rate):
    for i,item in enumerate(x):
        x[i] = rate*x[i]

    return x    

MODEL_NAME= "MobileNet"

MODE = "NN"
dirs = 'interval_result/retrain_' + MODEL_NAME

# LOAD_PATH = "interval_result/1571012974.0192485RandomForestRegressor"
LOAD_PATH = "interval_result/1570997508.460764MobileNet"

if not os.path.exists(dirs):
    os.makedirs(dirs)


if __name__ == "__main__":



    #Select Model(Classic or NN)
    #model = linear_model.LinearRegression()
    #model = ensemble.RandomForestRegressor(n_estimators=20)
    #model = load_model("weights_mobileNet_without_pre1570856691.343963_color.h5")
    #empty_model = linear_model.LinearRegression()

    
    #######If it is a new model, use the following method to generate.#######

    #X_train, y_train, X_test, y_test = cr.load_data()
    #result, predicts, y_interval = new_model(model, empty_model, mode = MODE,inversed=False,
                                             # X_train= X_train, y_train= y_train, X_test= X_test, y_test= y_test)
    
    #######If it is not a new model, load the old results.

    #result, predicts, y_interval, final_upper, final_lower = load_result(LOAD_PATH)

    #print(result)
    

    #
    # predicts= np.asarray(predicts).reshape(1,-1)[0]
    #
    # scaler_val = joblib.load('MaxAbsScaler.pkl')
    # predicts = transform_y(predicts, scaler_val)
    # result = transform_y(result, scaler_val)

    # b = bins, saving the predicts in this bin, eg. b[0] includes predicts like 1,2.1,3,4; b[1] includes 12,13,14,etc.
    # bt = bins_true saving the true value in this bin.
    
    #b,bt = send_true_to_bin([predicts], y_interval)
    
    #If it is a new model, run bin_process.
    # final_upper, final_lower = bin_process(b,bt)

    # Parameters for check the target rate error.
#     final_lower = aug(final_lower,1.6)
#     final_upper = aug(final_upper,1.6)

    #If it is a new model, train the smoother for this model and save them. u_model = Upper bound smoother, l_model = lower bound smoother

    # u_model,l_model = train_smoother(final_upper,final_lower)
    
    #If it is not a new model, just load the smoother
    u_model,l_model = load_smoother(r'/home/fei/Desktop/interval_result/1573514984.6583798MobileNet')
    #here is the predict result
    prediction = 10.0
    # load range model and predict upper and lower offsets
    upper_offset = u_model.predict(np.array(prediction, np.float).reshape((-1, 1)))[0]
    lower_offset = l_model.predict(np.array(prediction, np.float).reshape((-1, 1)))[0]

    print('predict range is [{:f}, {:f}]'.format(prediction - lower_offset, prediction + upper_offset) )

    #draw_test(result, y_test,final_upper,final_lower,True,u_model,l_model)






















#     X_train = np.reshape(X_train, (X_train.shape[0], -1))
#     X_test = np.reshape(X_test, (X_test.shape[0], -1))

#     kf = KFold(n_splits=5)

#     predicts = []
#     y_interval = []
#     #model = neighbors.KNeighborsRegressor()
#    # model = linear_model.LinearRegression()

#     model = ensemble.RandomForestRegressor(n_estimators=20)
#     # model = load_model("water/water/modelWights/weights_mobileNet_without_pre1570856691.343963_color.h5")

#     model.fit(X_train,y_train)
#     result = model.predict(X_test)

#     ####################################For NN####################################################
#     # for train_index, test_index in kf.split(X_train):
#     #     X_train_interval, X_test_interval = X_train[train_index], X_train[test_index]
#     #     y_train_interval, y_test_interval = y_train[train_index], y_train[test_index]

#     #     reinitLayers(model)
#     #     model.fit(X_train_interval,y_train_interval, batch_size = 4, epochs=200)
#     #     predict = model.predict(X_test_interval)
#     #     for item in predict:
#     #         predicts.append(item)
#     #     for item in y_test_interval:
#     #         y_interval.append(item)
#     ####################################For classic###################################################
#     for train_index, test_index in kf.split(X_train):
#         X_train_interval, X_test_interval = X_train[train_index], X_train[test_index]
#         y_train_interval, y_test_interval = y_train[train_index], y_train[test_index]
#         model = ensemble.RandomForestRegressor(n_estimators=20)
#         model.fit(X_train_interval,y_train_interval)
#         predict = model.predict(X_test_interval)
#         for item in predict:
#             predicts.append(item)
#         for item in y_test_interval:
#             y_interval.append(item)
#     # #
#     # # joblib.dump(model, 'model.pkl')
#     # model = joblib.load('model.pkl')





#     # print(result)

#     np.save(dirs + "/" + "predicts" + ".npy", predicts)
#     np.save(dirs + "/" + "y_interval" + ".npy", y_interval)
#     np.save(dirs + "/" + "result" + ".npy", result)

#     # predicts = np.load(dirs + "/" + "predicts" + ".npy")
#     # y_interval = np.load(dirs + "/" + "y_interval" + ".npy")
#     # result = np.load(dirs + "/" + "result.npy")

    # scaler_val = joblib.load('MaxAbsScaler.pkl')

    # print(predicts)

    # #predicts = transform_y(predicts, scaler_val)
    # #result = transform_y(result, scaler_val)

    # print(predicts)



    # print(b)

    # print(bt)

    # np.save(dirs + "/" + "bins" + ".npy", b)
    # np.save(dirs + "/" + "bins_true" + ".npy", bt)


    # b = np.load("bins.npy")
    # bt = np.load("bins_true.npy")
    # result = np.load("result.npy")





