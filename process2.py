import numpy as np
from os import listdir
from os.path import isfile, join

def MD(act, num_label):
    X = []
    lenFile = []
    files = [f for f in listdir('MobiAct_Dataset/{}/'.format(act)) if isfile(join('MobiAct_Dataset/{}/'.format(act), f))]
    for i in range(len(files)):
        if 'acc' in files[i]:
            temp = np.loadtxt('MobiAct_Dataset/{}/{}'.format(act,files[i]), skiprows=16, delimiter=',', dtype=str)
            temp = temp[:,1:]
            lenFile.append(len(temp))
            temp = temp[:582, :]
            temp = temp.astype('float32')
            temp = np.transpose(temp)

            X.append(temp)
    # print(min(lenFile))
    X = np.array(X)
    Y = np.full((X.shape[0],1),num_label)
    return X, Y

BSC_X, BSC_Y = MD('BSC', 1)
print(BSC_X.shape, BSC_Y.shape)
FKL_X, FKL_Y = MD('FKL', 1)
print(FKL_X.shape, FKL_Y.shape)
FOL_X, FOL_Y = MD('FOL', 1)
print(FOL_X.shape, FOL_Y.shape)
SDL_X, SDL_Y = MD('SDL', 1)
print(SDL_X.shape, SDL_Y.shape)

STD_X, STD_Y = MD('STD', 0)
print(STD_X.shape, STD_Y.shape)
WAL_X, WAL_Y = MD('WAL', 0)
print(WAL_X.shape, WAL_Y.shape)
JOG_X, JOG_Y = MD('JOG', 0)
print(JOG_X.shape, JOG_Y.shape)
JUM_X, JUM_Y = MD('JUM', 0)
print(JUM_X.shape, JUM_Y.shape)
STU_X, STU_Y = MD('STU', 0)
print(STU_X.shape, STU_Y.shape)
STN_X, STN_Y = MD('STN', 0)
print(STN_X.shape, STN_Y.shape)

ADL_Fall_X = np.concatenate((BSC_X,
                             FKL_X,
                             FOL_X,
                             SDL_X,
                             STD_X,
                             WAL_X,
                             JOG_X,
                             JUM_X,
                             STU_X,
                             STN_X), axis=0)
ADL_Fall_Y = np.concatenate((BSC_Y,
                             FKL_Y,
                             FOL_Y,
                             SDL_Y,
                             STD_Y,
                             WAL_Y,
                             JOG_Y,
                             JUM_Y,
                             STU_Y,
                             STN_Y), axis=0)
print(ADL_Fall_X.shape, ADL_Fall_Y.shape)
np.save('ADL_Fall_X', ADL_Fall_X)
np.save('ADL_Fall_Y', ADL_Fall_Y)
print('Done!')

BSC_X, BSC_Y = MD('BSC', 6)
print(BSC_X.shape, BSC_Y.shape)
FKL_X, FKL_Y = MD('FKL', 7)
print(FKL_X.shape, FKL_Y.shape)
FOL_X, FOL_Y = MD('FOL', 8)
print(FOL_X.shape, FOL_Y.shape)
SDL_X, SDL_Y = MD('SDL', 9)
print(SDL_X.shape, SDL_Y.shape)

STD_X, STD_Y = MD('STD', 0)
print(STD_X.shape, STD_Y.shape)
WAL_X, WAL_Y = MD('WAL', 1)
print(WAL_X.shape, WAL_Y.shape)
JOG_X, JOG_Y = MD('JOG', 2)
print(JOG_X.shape, JOG_Y.shape)
JUM_X, JUM_Y = MD('JUM', 3)
print(JUM_X.shape, JUM_Y.shape)
STU_X, STU_Y = MD('STU', 4)
print(STU_X.shape, STU_Y.shape)
STN_X, STN_Y = MD('STN', 5)
print(STN_X.shape, STN_Y.shape)

Ten_Move_X = np.concatenate((BSC_X,
                             FKL_X,
                             FOL_X,
                             SDL_X,
                             STD_X,
                             WAL_X,
                             JOG_X,
                             JUM_X,
                             STU_X,
                             STN_X), axis=0)
Ten_Move_Y = np.concatenate((BSC_Y,
                             FKL_Y,
                             FOL_Y,
                             SDL_Y,
                             STD_Y,
                             WAL_Y,
                             JOG_Y,
                             JUM_Y,
                             STU_Y,
                             STN_Y), axis=0)
print(Ten_Move_X.shape, Ten_Move_Y.shape)
np.save('Ten_Move_X', Ten_Move_X)
np.save('Ten_Move_Y', Ten_Move_Y)
print('Done!')