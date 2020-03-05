import scipy.io as sio
import  tensorflow as tf
import numpy as np
from os import listdir
from os.path import isfile, join

########################################### Subject_13 has wrong value

def MD(movement, num_label):
    X = []
    for i in range(len(files)):
        if movement in files[i]:
            temp = np.loadtxt('UMAFall_Dataset/' + files[i], skiprows=41, delimiter=';', dtype=str)
            index = temp[:, 6] == '0'
            temp = temp[index, 2:5]
            temp = temp[:2910, :]
            temp = temp.astype('float32')
            temp = np.transpose(temp)

            X.append(temp)
    X = np.array(X)
    Y = np.full((X.shape[0],1),num_label)
    return X, Y

files = [f for f in listdir('UMAFall_Dataset/') if isfile(join('UMAFall_Dataset/', f))]
# print(files)
label = ['Aplausing', 'HandsUP', 'MakingACall', 'OpeningDoor', 'Sitting_GettingUpOnAChair', 'Walking', 'Bending',
         'Hopping', 'Jogging', 'LyingDown_OnABed', 'GoUpstairs', 'GoDownstairs', 'backwardFall', 'forwardFall', 'lateralFall' ]
print(len(label))
'''
'Aplausing' = 0
'HandsUP' = 1
'MakingACall' = 2
'OpeningDoor' = 3
'Sitting_GettingUpOnAChair' = 4
'Walking' =5
'Bending' = 6
'Hopping' = 7
'Jogging' = 8
'LyingDown_OnABed' = 9
'GoUpstairs' = 10
'GoDownstairs' = 11
'backwardFall' = 12
'forwardFall' = 13
'lateralFall' = 14
'''

Aplausing_X, Aplausing_Y = MD('Aplausing', 0)
print(Aplausing_X.shape, Aplausing_Y.shape)

HandsUP_X, HandsUP_Y = MD('HandsUp', 1)
print(HandsUP_X.shape, HandsUP_Y.shape)

MakingACall_X, MakingACall_Y = MD('MakingACall', 2)
print(MakingACall_X.shape, MakingACall_Y.shape)

OpeningDoor_X, OpeningDoor_Y = MD('OpeningDoor', 3)
print(OpeningDoor_X.shape, OpeningDoor_Y.shape)

Sitting_GettingUpOnAChair_X, Sitting_GettingUpOnAChair_Y = MD('Sitting_GettingUpOnAChair', 4)
print(Sitting_GettingUpOnAChair_X.shape, Sitting_GettingUpOnAChair_Y.shape)

Walking_X, Walking_Y = MD('Walking', 5)
print(Walking_X.shape, Walking_Y.shape)

Bending_X, Bending_Y = MD('Bending', 6)
print(Bending_X.shape, Bending_Y.shape)

Hopping_X, Hopping_Y = MD('Hopping', 7)
print(Hopping_X.shape, Hopping_Y.shape)

Jogging_X, Jogging_Y = MD('Jogging', 8)
print(Jogging_X.shape, Jogging_Y.shape)

LyingDown_OnABed_X, LyingDown_OnABed_Y = MD('LyingDown_OnABed', 9)
print(LyingDown_OnABed_X.shape, LyingDown_OnABed_Y.shape)

GoUpstairs_X, GoUpstairs_Y = MD('GoUpstairs', 10)
print(GoUpstairs_X.shape, GoUpstairs_Y.shape)

GoDownstairs_X, GoDownstairs_Y = MD('GoDownstairs', 11)
print(GoDownstairs_X.shape, GoDownstairs_Y.shape)

backwardFall_X, backwardFall_Y = MD('backwardFall', 12)
print(backwardFall_X.shape, backwardFall_Y.shape)

forwardFall_X, forwardFall_Y = MD('forwardFall', 13)
print(forwardFall_X.shape, forwardFall_Y.shape)

lateralFall_X, lateralFall_Y = MD('lateralFall', 14)
print(lateralFall_X.shape, lateralFall_Y.shape)

move_15_X = np.concatenate((Aplausing_X,
                            HandsUP_X,
                            MakingACall_X,
                            OpeningDoor_X,
                            Sitting_GettingUpOnAChair_X,
                            Walking_X,
                            Bending_X,
                            Hopping_X,
                            Jogging_X,
                            LyingDown_OnABed_X,
                            GoUpstairs_X,
                            GoDownstairs_X,
                            backwardFall_X,
                            forwardFall_X,
                            lateralFall_X), axis=0)
move_15_Y = np.concatenate((Aplausing_Y,
                            HandsUP_Y,
                            MakingACall_Y,
                            OpeningDoor_Y,
                            Sitting_GettingUpOnAChair_Y,
                            Walking_Y,
                            Bending_Y,
                            Hopping_Y,
                            Jogging_Y,
                            LyingDown_OnABed_Y,
                            GoUpstairs_Y,
                            GoDownstairs_Y,
                            backwardFall_Y,
                            forwardFall_Y,
                            lateralFall_Y), axis=0)
move_15_X = move_15_X[:, :, ::2]
np.save('move_15_X', move_15_X)
np.save('move_15_Y', move_15_Y)
print(move_15_X.shape)
print(move_15_Y.shape)

ADL_X, ADL_Y = MD('_ADL_', 0)
print(ADL_X.shape, ADL_Y.shape)

Fall_X, Fall_Y = MD('_Fall_', 1)
print(Fall_X.shape, Fall_Y.shape)

Fall_ADL_X =  np.concatenate((ADL_X, Fall_X), axis=0)
Fall_ADL_Y =  np.concatenate((ADL_Y, Fall_Y), axis=0)
Fall_ADL_X = Fall_ADL_X[:, :, ::2]
print(Fall_ADL_X.shape, Fall_ADL_Y.shape)
np.save('Fall_ADL_X', Fall_ADL_X)
np.save('Fall_ADL_Y', Fall_ADL_Y)

print("Done!")