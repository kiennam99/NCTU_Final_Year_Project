
from __future__ import division 
import numpy as np

def filterResult(r):
    '''
    Select cars and pedestrian
    '''
    idx = np.isin(r['class_ids'],[1,3])
    rois = r['rois'][idx]
    masks = []
    for i in range(r['masks'].shape[0]):
        temp = []
        for j in range(r['masks'].shape[1]):
            temp.append(r['masks'][i][j][idx].tolist())
        masks.append(temp)
    masks = np.array(masks)            
    classid = r['class_ids'][idx]
    scores = r['scores'][idx]
    return rois,masks,classid,scores

def find_3D_Dist(p1,p2):
    '''
    @ Return 3d distance
    '''
    assert len(p1) == len(p2)
    x1,y1,z1 = p1
    x2,y2,z2 = p2
    return np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)

def find_2D_Dist(p1,p2):
    '''
    @ Return 2D distance
    '''
    assert len(p1) == len(p2)
    x1,y1 = p1
    x2,y2 = p2
    return np.sqrt((x2-x1)**2 + (y2-y1)**2 )

def getSpeed(point1,point2,interval):
    ''' 
    speed=displacement/interval
    '''
    return find_3D_Dist(point1,point2)/interval

def getTarPoint(m):
    '''
    @ return : list of targets' point
    @ given mask -> find average points
    '''
    # assert len(filter_result) == 4
    ret = []
    n = m.shape[2] # instances
    for i in range(n):
        temp = m[:,:,i]
        idx = np.argwhere(temp==True)
        mid = [int(idx[:,0].mean()), int(idx[:,1].mean())]
        ret.append(mid)
    return ret


def find_point_pair(tar1, tar2):
    '''
    find overlap
    @ tar1 : frame t-1 target sets (targets_num, maskpoint)
    @ tar2 : frame t target sets (targets_num, maskpoint) 
    @ return valid = 0 if point from next point just exist
    '''
    
    pointpair = {} #  i : j
    visited = {}
    alldist = {}
    for i in range(len(tar1)):
        visited[i]=-1

    for i,t2 in enumerate(tar2):
        mindist = 99999
        pointpair[i] = -1
        for j,t1 in enumerate(tar1):
            dist = find_2D_Dist(t1,t2)
            if dist < mindist:
                mindist = dist
                # if not used
                if visited[j]==-1:
                    pointpair[i] = j
                    visited[j] = 1
                    alldist[j] = [i,mindist]
                # if used
                else:
                    # if cur point closer
                    if mindist < alldist[j][1]:
                        pointpair[alldist[j][0]]=-1
                        alldist[j][1] = mindist
                        alldist[j][0] = i
                        pointpair[i]=j
            

    return pointpair





                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   

def getInterval(path):
    '''
    Get Time Interval
    '''
    interval = []
    with open(path,"r") as f:
        cnt = 0
        prev_time = 0
        for line in f.readlines():
            li = line.split(' ')[1].split(':')
            li[2] = li[2].split('\n')[0]
            
            if cnt == 0:
                prev_time = float(li[0])*60*60 + float(li[1])*60 + float(li[2])
                cnt+=1
            else:
                cur_time = float(li[0])*60*60 + float(li[1])*60 + float(li[2])
                interval.append(cur_time - prev_time)
                prev_time = cur_time
    return interval

def CalibratePoses(pcd,poses):
    '''
    Multiply poses to point clouds
    '''
    ret = [pcd[0]]
    pose = np.eye(4)
    for i in range(1,len(pcd)-1):
        pose = pose @ poses[i-1].T # 4*4
        pt = np.ones((pcd[i].shape[0],4))
        pt[:,:3] = pcd[i] # 49150 * 4
        pt = pt @ pose.T # 49150*4 * 4x4
        ret.append(pt[:,:3])
    return ret

def getAllSpeed(targets,point_pair,pcd_calib,interval,imgshp):
    '''
    Calculate speed
    ---------------------------------
    Target size = n
    Frame Start from 1
    ''' 
    speed = []  # (frame_size = n-1, speed)
    h,w = imgshp
    for i in range(1,len(targets)-1):
        temp = []
        for j,pt in enumerate(targets[i]):
            # For each masks in target[i]
            # get cur 3D points
            pt3d = pcd_calib[i][pt[0]*w + pt[1]]
            
            # Previous frame
            idx = point_pair[i-1][j]
            if idx != -1:
                prev_pt2d = targets[i-1][idx] 
                prev_pt3d = pcd_calib[i-1][prev_pt2d[0]*w + prev_pt2d[1]]    
                temp.append(getSpeed(prev_pt3d,pt3d,interval[i]*60*60))
            else:
                temp.append(0)

        speed.append(temp)
    return speed

