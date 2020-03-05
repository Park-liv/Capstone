===========================MobiActDataset===========================


Filename format:
<ADL OR FALL_CODE>_<SENSOR_CODE>_<SUBJECT_ID>_<TRIAL_NO>.txt


examples:
1 -->	WAL_acc_5_1.txt
2 -->	STD_ori_9_5.txt
3 -->	FKL_gyro_3_2.txt


Subjects:
+------+---------+-----------+-------+----------+----------+----------+
|  ID  |  Name   |  Surname  |  Age  |  Height  |  Weight  |  Gender  |
+------+---------+-----------+-------+----------+----------+----------+
|    1 | sub1    | sub1      |    32 |      180 |       85 |  M       |   
|    2 | sub2    | sub2      |    26 |      169 |       64 |  M       |
|    3 | sub3    | sub3      |    26 |      164 |       55 |  F       |  
|    4 | sub4    | sub4      |    32 |      186 |       93 |  M       |   
|    5 | sub5    | sub5      |    36 |      160 |       50 |  F       |    
|    6 | sub6    | sub6      |    22 |      172 |       62 |  F       |    
|    7 | sub7    | sub7      |    25 |      189 |       80 |  M       |   
|    8 | sub8    | sub8      |    22 |      183 |       93 |  M       |    
|    9 | sub9    | sub9      |    30 |      177 |      102 |  M       |   
|   10 | sub10   | sub10     |    26 |      170 |       90 |  F       |  
|   11 | sub11   | sub11     |    26 |      168 |       80 |  F       |  
|   12 | sub12   | sub12     |    29 |      178 |       83 |  M       |   
|   13 | sub13   | sub13     |    24 |      177 |       62 |  M       | 
|   14 | sub14   | sub14     |    24 |      178 |       85 |  M       |   
|   15 | sub15   | sub15     |    25 |      173 |       82 |  M       | 
|   16 | sub16   | sub16     |    27 |      172 |       56 |  F       |   
|   17 | sub17   | sub17     |    25 |      173 |       67 |  M       |   
|   18 | sub18   | sub18     |    25 |      176 |       73 |  M       |   
|   19 | sub19   | sub19     |    25 |      161 |       63 |  F       |   
|   20 | sub20   | sub20     |    26 |      178 |       71 |  M       |   
|   21 | sub21   | sub21     |    25 |      180 |       70 |  M       |   
|   22 | sub22   | sub22     |    22 |      187 |       90 |  M       |   
|   23 | sub23   | sub23     |    23 |      171 |       64 |  M       |   
|   24 | sub24   | sub24     |    21 |      193 |      112 |  M       |   
|   25 | sub25   | sub25     |    22 |      170 |       62 |  F       |    
|   26 | sub26   | sub26     |    25 |      163 |       60 |  F       |   
|   27 | sub27   | sub27     |    25 |      180 |       82 |  M       |  
|   28 | sub28   | sub28     |    23 |      178 |       70 |  Ì       |   
|   29 | sub29   | sub29     |    27 |      186 |      103 |  M       |    
|   30 | sub30   | sub30     |    47 |      172 |       90 |  M       |   
|   31 | sub31   | sub31     |    27 |      170 |       75 |  M       |   
|   32 | sub32   | sub32     |    25 |      190 |       77 |  M       |   
|   33 | sub33   | sub33     |    27 |      171 |       70 |  M       |   
|   34 | sub34   | sub34     |    24 |      175 |       85 |  Ì       |    
|   35 | sub35   | sub35     |    23 |      181 |       76 |  M       |    
|   36 | sub36   | sub36     |    22 |      164 |       62 |  F       |   
|   37 | sub37   | sub37     |    25 |      172 |       63 |  M       |  
|   38 | sub38   | sub38     |    21 |      170 |       88 |  F       |    
|   39 | sub39   | sub39     |    26 |      174 |       79 |  M       |    
|   40 | sub40   | sub40     |    23 |      178 |       95 |  M       |    
|   41 | sub41   | sub41     |    20 |      172 |       67 |  F       |  
|   42 | sub42   | sub42     |    22 |      173 |       73 |  M       |   
|   43 | sub43   | sub43     |    24 |      179 |       80 |  M       |   
|   44 | sub44   | sub44     |    25 |      176 |       80 |  M       |  
|   45 | sub45   | sub45     |    26 |      175 |       92 |  M       |   
|   46 | sub46   | sub46     |    23 |      175 |       68 |  F       |   
|   47 | sub47   | sub47     |    21 |      180 |       85 |  M       |  
|   48 | sub48   | sub48     |    22 |      180 |       80 |  M       |   
|   49 | sub49   | sub49     |    23 |      178 |       75 |  M       |    
|   50 | sub50   | sub50     |    23 |      165 |       50 |  F       |   
|   51 | sub51   | sub51     |    23 |      171 |       70 |  M       |    
|   52 | sub52   | sub52     |    20 |      179 |       79 |  M       |  
|   53 | sub53   | sub53     |    27 |      186 |      120 |  M       |   
|   54 | sub54   | sub54     |    27 |      164 |       55 |  F       |    
|   55 | sub55   | sub55     |    28 |      178 |       78 |  M       |    
|   56 | sub56   | sub56     |    29 |      170 |       75 |  M       |  
|   57 | sub57   | sub57     |    21 |      187 |       70 |  Ì       |
+------+---------+-----------+-------+----------+----------+----------+


Activities of Daily Living:
+----+------+--------------+--------+----------+--------------------------------+
| id | Code | Activity     | Trials | Duration | Description                    |
+----+------+--------------+--------+----------+--------------------------------+
| 1  | STD  | Standing     | 1      | 5m       | Standing with subtle movements |
| 2  | WAL  | Walking      | 1      | 5m       | Normal walking                 |
| 3  | JOG  | Jogging      | 3      | 30s      | Jogging                        |
| 4  | JUM  | Jumping      | 3      | 30s      | Continuous jumping             |
| 5  | STU  | Stairs up    | 6      | 10s      | Stairs up (10 stairs)          |
| 6  | STN  | Stairs down  | 6      | 10s      | Stairs down (10 stairs)        |
| 7  | SCH  | Sit chair    | 6      | 6s       | Sitting on a chair             |
| 8  | CSI  | Car-step in  | 6      | 6s       | Step in a car                  |
| 9  | CSO  | Car-step out | 6      | 6s       | Step out a car                 |
+----+------+--------------+--------+----------+--------------------------------+


Falls:
+----+------+--------------------+--------+----------+---------------------------------------------------------+
| id | Code | Activity           | Trials | Duration | Description                                             |
+----+------+--------------------+--------+----------+---------------------------------------------------------+
| 10 | FOL  | Forward-lying      | 3      | 10s      | Fall Forward from standing, use of hands to dampen fall |
| 11 | FKL  | Front-knees-lying  | 3      | 10s      | Fall forward from standing, first impact on knees       |
| 12 | BSC  | Back-sitting-chair | 3      | 10s      | Fall backward while trying to sit on a chair            |
| 13 | SDL  | Sideward-lying     | 3      | 10s      | Fall sidewards from standing, bending legs              |
+----+------+--------------------+--------+----------+---------------------------------------------------------+


Sensors:
+------+---------------+----------------------------------------------------+--------------------------------------------------------------+
| Code | Type          | Values                                             | Description                                                  |
+------+---------------+----------------------------------------------------+--------------------------------------------------------------+
| acc  | accelerometer | timestamp(ns),x,y,z(m/s^2)                         | Acceleration force along the x y z axes (including gravity). |
| gyro | gyroscope     | timestamp(ns),x,y,z(rad/s)                         | Rate of rotation around the x y z axes (Angular velocity).   |
| ori  | orientation   | timestamp(ns),Azimuth,Pitch,Roll(degrees)          | Angle around the z x y axes.                                 |
+------+---------------+----------------------------------------------------+--------------------------------------------------------------+