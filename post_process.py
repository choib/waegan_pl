import cv2 as cv
import json
import argparse
import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

#color_code={"T_O":(255,64,0),"T_C":(64,0,255), "T_H":(0,255,64)} # RGB
#color_code=[(0,64,255),(255,0,64), (64,255,0)] # BGR
#color_code=[(255,64,0),(64,0,255), (0,255,64)] # RGB
# color_code=[(255,0,0),(0,0,255), (0,255,0)] # RGB
# class_clr = ["#ff4000","#4000ff","#00ff40"]
# color_code= [(0,255,255), (20,255,255), (40,255,255),
#             (60,255,255), (80,255,255), (100,255,255),
#             (120,255,255),(140,255,255), (160,255,255)] 
color_code= [(52,255,255), (90,255,255), (105,255,255),
             (127,255,255), (30,255,255), (0,255,255),
             (150,255,255)] 
class_clr = ["#ff0000","#ffaa00","#aaff00",
            "#00ff00","#00ffaa","#00aaff",
            "#0000ff","#aa00ff","#ff00aa"]
# low_HSV = {"T_O":[(0, 100, 100),(170, 100, 100)],
#             "T_C":[(110, 100, 100)],
#             "T_H":[(90, 100, 100),(130, 100, 100)],
#             "full":[(0, 100, 100)],
#             "uncertain":[(0, 100, 100)]}
# high_HSV = {"T_O":[(10, 255, 255),(180, 255, 255)],
#             "T_C":[(130, 255, 255)],
#             "T_H":[(110, 255, 255),(150, 255, 255)],
#             "full":[(180, 255, 255)],
#             "uncertain":[(180, 225, 225)]}
low_hsv =  [[(47, 100, 100)],
    [(85, 100, 100)],
    [(100, 100, 100)],
    [(122, 100, 100)],
    [(25, 100, 100)],
    [(0, 100, 100),(175, 100, 100)],
    [(145, 100, 100)],
    [(0, 50, 50)],    
    [(0, 50, 50)]]
high_hsv = [[(57, 255, 255)],
    [(95, 255, 255)],
    [(110, 255, 255)],
    [(132, 255, 255)],
    [(35, 255, 255)],
    [(5,255,255), (180, 255, 255)],
    [(155, 255, 255)],
    [(180, 255, 255)],
    [(180, 225, 253)]]
# low_hsv = [[(0, 100, 100),(175, 100, 100)], #_0
#         [(15, 100, 100)], #_1
#         [(35, 100, 100)], #_2
#         [(55, 100, 100)], #_3
#         [(70, 100, 100)], #_4
#         [(90, 100, 100)], #_5
#         [(115, 100, 100)], #_6
#         [(130, 100, 100)], #_7
#         [(25, 100, 100)], #_8
#         [(0, 50, 50)], #_full
#         [(0, 50, 50)]] #_uncertaim
# high_hsv = [[(5,255,255), (180, 255, 255)], #_0
#         [(25, 255, 255)], #_1
#         [(45, 255, 255)], #_2 
#         [(65, 255, 255)], #_3
#         [(90, 255, 255)], #_4
#         [(110, 255, 255)], #_5
#         [(125, 255, 255)], #_6
#         [(150, 255, 255)], #_7
#         [(35, 255, 255)], #_8
#         [(180, 255, 255)], #_full
#         [(180, 253, 253)]] #_uncertain

iter = 3
detail = 0.002
no_class = 7
call_help = 7
img_h, img_w, img_d = 256, 384, 3


def cnts_extract(im_HSV,low_HSV,high_HSV, erode=False, dilate=False):
    im_threshold = cv.inRange(im_HSV, low_HSV, high_HSV)
    if dilate:
        kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
        im_threshold = cv.dilate(im_threshold, kernel, anchor=(0, 0), iterations=iter)
    if erode:
        kernel = cv.getStructuringElement(cv.MORPH_CROSS, (5, 5))
        im_threshold = cv.erode(im_threshold, kernel, anchor=(0, 0), iterations=iter)
    gray = cv.bitwise_and(im_HSV,im_HSV,mask=im_threshold)
    gray = cv.cvtColor(gray, cv.COLOR_RGB2GRAY)
    ret, binary = cv.threshold(gray, 5, 255, cv.THRESH_BINARY)
    _, contours, hierarchy = cv.findContours(binary, cv.RETR_CCOMP,cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, reverse=True, key=lambda x: cv.contourArea(x))
    area = 0
    if len(contours) > 0:
        for cnt in contours:
            #cnt0 = contours[0]
            area += cv.contourArea(cnt)
    else:
        contours = None
        #area = 0
    return contours, area

def save_contour(cnts, ratio, key, out, args):
    class_id = key
    polygons = []
    data={}
    data['id'] = class_id+1
    data['clr'] = class_clr[key]
    data['uncertainty'] = ratio
    #input_shape = (args.n_channel, args.img_height, args.img_width)
    data['size']= {'width':args.img_width,'height':args.img_height,'depth':args.n_channel} 
    data['feature'] = {}
    sdata = data['feature']
    for i,cnt in enumerate(cnts):
        #
        # data = data['feature'].append({'index':'{}'.format(i)})
        #data = data['feature']
        sdata[i]={}
        sdata[i]['type']='poly'
        epsilon = cv.arcLength(cnt, True) * detail
        approx_poly = cv.approxPolyDP(cnt, epsilon, True)
        sdata[i]['points'] = []
    
        for approx in approx_poly:
            pointX, pointY = tuple(approx[0])
            sdata[i]['points'].append({'x': int(pointX)})
            sdata[i]['points'].append({'y': int(pointY)})
        polygons.append(approx_poly)
        if args.val_target == 'update' :
            break
    with open(out, 'wt') as outfile:
        json.dump(data, outfile, indent=1)
    
    return polygons

def draw_pic(key,polygons,args):
    color = color_code[key]
    shape = (args.img_height, args.img_width,  args.n_channel)
    #print("shape: {}, color code: {}\n".format(shape, color))
    src = np.zeros(shape, dtype = np.uint8)
    for polygon in polygons:
        cv.fillPoly(src, [polygon], color, cv.LINE_AA)
    return src

def save_pic(src,picpath,args):
    bgrimg = cv.cvtColor(src, cv.COLOR_HSV2BGR)
    #src = cv.cvtColor(bgrimg, cv.COLOR_BGRA2RGBA)
    cv.imwrite(picpath, bgrimg)#cv.cvtColor(bgrimg, cv.COLOR_BGR2RGB))

def critic_segmentation(im_seg):
    critic=[]
    im_seg = cv.cvtColor(im_seg, cv.COLOR_RGB2BGR)
    im_HSV = cv.cvtColor(im_seg, cv.COLOR_BGR2HSV)
    full_cnt, full_area = cnts_extract(im_HSV, low_hsv[-2][0], high_hsv[-2][0])
    _, uncertain_area = cnts_extract(im_HSV, low_hsv[-1][0], high_hsv[-1][0], dilate=True, erode=True)
    full_area = 1e-3 if full_area < 1e-3 else full_area
    uncertainty = uncertain_area/full_area
    
    for id in range(no_class):
        cl_area = 0.0
        for j in range(len(low_hsv[id])):
            _, c_area = cnts_extract(im_HSV, low_hsv[id][j], high_hsv[id][j], dilate=True, erode=True)
            cl_area += c_area
        critic.append(cl_area/full_area)
    
    max_value = max(critic)
    max_index = critic.index(max_value)

    return max_index, uncertainty, max_value, full_cnt

def critic_segmentation_by_class(id, im_seg, im_gt, args):
    DEBUG = False
    critic=[]
    im_seg = cv.cvtColor(im_seg, cv.COLOR_RGB2BGR)
    im_gt = cv.cvtColor(im_gt, cv.COLOR_RGB2BGR)
    seg_HSV = cv.cvtColor(im_seg, cv.COLOR_BGR2HSV)
    gt_HSV = cv.cvtColor(im_gt, cv.COLOR_BGR2HSV)
    
    if DEBUG:
        pic1path = os.path.relpath(f"./tmp/seg_{id}.jpg")
        pic2path = os.path.relpath(f"./tmp/gt_{id}.jpg")
        save_pic(seg_HSV,pic1path,args)
        save_pic(gt_HSV,pic2path,args)

    seg_area = 0.0
    uncertain_area = 0.0
    seg_c = []
    u_seg_c = []
    gt_c = []
    predicted_mask = np.zeros([img_h, img_w, img_d],dtype=np.uint8)
    real_mask = np.zeros([img_h, img_w, img_d],dtype=np.uint8)
    predicted_bb = np.zeros([img_h, img_w, img_d],dtype=np.uint8)
    real_bb = np.zeros([img_h, img_w, img_d],dtype=np.uint8)
    for j in range(len(low_hsv[id])):
        seg_cnts, s_area = cnts_extract(seg_HSV, low_hsv[id][j], high_hsv[id][j], dilate=True, erode=True)
        seg_area += s_area
        seg_c.append(seg_cnts)
        #u_h_hsv = high_hsv[id][j]
        if seg_cnts is None:
            continue
        else:
            for i,cnt in enumerate(seg_cnts):
                epsilon = cv.arcLength(cnt, True) * detail
                approx_poly = cv.approxPolyDP(cnt, epsilon, True)
                cv.fillPoly(predicted_mask, pts =[approx_poly], color=(255,255,255))
                (x,y,w,h) = cv.boundingRect(cnt)
                cv.rectangle(predicted_bb, (x, y), (w, h), (255, 255, 255), -1)
                #predicted_mask = cv.add(predicted_mask,draw_pic(id, approx_poly, args))
        (h,s,v) = high_hsv[id][j]
        u_h_hsv = (h,253,253)
        u_seg_cnts, u_seg_area = cnts_extract(seg_HSV, low_hsv[id][j], u_h_hsv, dilate=True, erode=True)
        uncertain_area += u_seg_area
        if DEBUG:
            pic1path = os.path.relpath(f"./tmp/seg_{id}_{j}.jpg")
            save_pic(predicted_mask,pic1path,args)
            
    gt_area = 0.0
    for j in range(len(low_hsv[id])):
        gt_cnts, g_area = cnts_extract(gt_HSV, low_hsv[id][j], high_hsv[id][j], dilate=True, erode=True)
        gt_c.append(gt_cnts)
        gt_area += g_area
        if gt_cnts is None:
            continue
        else:
            for i,cnt in enumerate(gt_cnts):
                epsilon = cv.arcLength(cnt, True) * detail
                approx_poly = cv.approxPolyDP(cnt, epsilon, True)
                cv.fillPoly(real_mask, pts =[approx_poly], color=(255,255,255))
                (x,y,w,h) = cv.boundingRect(cnt)
                cv.rectangle(real_bb, (x, y), (w, h), (255, 255, 255), -1)
           #real_mask = cv.add(real_mask, draw_pic(id, approx_poly, args))
        if DEBUG:
            pic2path = os.path.relpath(f"./tmp/gt_{id}_{j}.jpg")
            save_pic(real_mask,pic2path,args)    
    # critic.append(cl_area/full_area)
    if DEBUG:
        pic1path = os.path.relpath(f"./tmp/predict_{id}.jpg")
        pic2path = os.path.relpath(f"./tmp/real_{id}.jpg")
        save_pic(predicted_mask,pic1path,args)
        save_pic(real_mask,pic2path,args)

    predicted_mask = cv.cvtColor(predicted_mask, cv.COLOR_RGB2GRAY)
    ret, predicted_mask = cv.threshold(predicted_mask, 5, 255, cv.THRESH_BINARY)
    real_mask = cv.cvtColor(real_mask, cv.COLOR_RGB2GRAY)
    ret, real_mask = cv.threshold(real_mask, 5, 255, cv.THRESH_BINARY)
    full_mask = np.ones(real_mask.shape,dtype=np.uint8)
    intersection = cv.bitwise_and(real_mask, predicted_mask, mask=full_mask)
    _, contours, hierarchy = cv.findContours(intersection, cv.RETR_CCOMP,cv.CHAIN_APPROX_SIMPLE)
    int_area = 0
    if len(contours) > 0:
        for cnt in contours:
            #cnt0 = contours[0]
            int_area += cv.contourArea(cnt)
    else:
        contours = None
    union = cv.bitwise_or(real_mask, predicted_mask, mask=full_mask)
    if DEBUG:
        pic1path = os.path.relpath(f"./tmp/int_{id}.jpg")
        pic2path = os.path.relpath(f"./tmp/uni_{id}.jpg")
        save_pic(cv.cvtColor(intersection,cv.COLOR_GRAY2RGB),pic1path,args)
        save_pic(cv.cvtColor(union,cv.COLOR_GRAY2RGB),pic2path,args)
    
    s_int = np.sum(intersection)
    s_uni = np.sum(union)
    
    iou = s_int / s_uni if s_uni > 1e-3 else None 
    uncertainty = uncertain_area/seg_area if seg_area > 1e-3 else None
    dice = 2.0 * int_area / (seg_area + gt_area) if int_area > 1e-3 else None
    r_area = seg_area/gt_area if gt_area > 1e-3 else None

    predicted_bb = cv.cvtColor(predicted_bb, cv.COLOR_RGB2GRAY)
    ret, predicted_bb = cv.threshold(predicted_bb, 5, 255, cv.THRESH_BINARY)
    real_bb = cv.cvtColor(real_bb, cv.COLOR_RGB2GRAY)
    ret, real_bb = cv.threshold(real_bb, 5, 255, cv.THRESH_BINARY)
    full_mask = np.ones(real_bb.shape,dtype=np.uint8)
    intersection = cv.bitwise_and(real_bb, predicted_bb, mask=full_mask)
    union = cv.bitwise_or(real_bb, predicted_bb, mask=full_mask)
    s_int = np.sum(intersection)
    s_uni = np.sum(union)
    iou_bb = s_int / s_uni if s_uni > 1e-3 else None 
    #
    return iou, iou_bb, dice, uncertainty, r_area
    
def update_segmentation(im_seg):
    # critic=[]
    im_seg = cv.cvtColor(im_seg, cv.COLOR_RGB2BGR)
    im_HSV = cv.cvtColor(im_seg, cv.COLOR_BGR2HSV)
    full_cnt, full_area = cnts_extract(im_HSV, low_hsv[-2][0], high_hsv[-2][0], dilate=True, erode=True)
    #_, uncertain_area = cnts_extract(im_HSV, low_hsv[-1][0], high_hsv[-1][0], erode=True)
    full_area = 1e-3 if full_area < 1e-3 else full_area
    #uncertainty = uncertain_area/full_area
    
    # for id in range(no_class):
    #     cl_area = 0.0
    #     for j in range(len(low_hsv[id])):
    #         _, c_area = cnts_extract(im_HSV, low_hsv[id][j], high_hsv[id][j], erode=True)
    #         cl_area += c_area
    #     critic.append(cl_area/full_area)
    
    # max_value = max(critic)
    # max_index = critic.index(max_value)

    #return max_index, uncertainty, max_value, full_cnt
    return full_area, full_cnt

def main():
    parser = argparse.ArgumentParser(description='Code for autolabeling of 7 classes')
    parser.add_argument('--seg', help='Input file path', default="./images/1009_seg_laryngoscope/1031_8.png", type=str)
    parser.add_argument('--gt', help='Input file path', default="./images/1009_gt_laryngoscope/1031_8.png", type=str)
    parser.add_argument('--split', help='number of image stack', default=8, type=int)
    parser.add_argument("--img_width", dest="img_width", default=384, type=int, help="width of image in pixels")
    parser.add_argument("--img_height", dest="img_height", default=256, type=int, help="height of image in pixels")
    parser.add_argument("--n_channel", dest="n_channel", default=3, type=int, help="channels in the input data")
    parser.add_argument("--val_target", dest="val_target", default='test', choices=['train','test','user','update'], type= str, help="target dataset to validation")
        
    args = parser.parse_args()
    
    seg = os.path.relpath("".join(args.seg))
    gt = os.path.relpath("".join(args.gt))
    base = os.path.splitext(args.seg)[0]

    img_seg = Image.open(seg)
    img_gt = Image.open(gt)
    img_seg.load()
    img_gt.load()
    img_seg = np.asarray(img_seg, dtype='uint8')
    img_gt = np.asarray(img_gt, dtype='uint8')
    sp_seg = np.vsplit(img_seg,args.split) #TODO: auto parse of split number
    sp_gt = np.vsplit(img_gt,args.split)
    for i, (im_seg, im_gt) in enumerate(zip(sp_seg,sp_gt)):
        
        # max_index, uncertainty, full_area, full_cnt = critic_segmentation(sp)
        # jsonpath = os.path.relpath(base+"_{}.json".format(i))
        # picpath = os.path.relpath(base+"_{}.jpg".format(i))
        for id in range(no_class):
            iou, iou_bb, dice, uncertainty, r_area = critic_segmentation_by_class(id, im_seg, im_gt, args)
            print(f"class {id}:, iou {iou}, bb iou {iou_bb}, dice {dice}, uncertainty {uncertainty}, area ratio {r_area}")
        # if full_area > 1e-3:
        #     #uncertainty = uncertain_area/full_area
        #     polygon = save_contour(full_cnt,uncertainty,max_index,jsonpath,args)
        #     key = max_index if uncertainty < 0.5 else call_help
        #     print("{} uncertainty: {}\tclass: {}\t{}".format(i,uncertainty,key,max_index))
        #     src = draw_pic(key,polygon,args)
        #     save_pic(src,picpath,args)

if __name__ == "__main__":
    main()