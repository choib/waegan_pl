import cv2 as cv
import json
import argparse
import os
import numpy as np
from PIL import Image

#color_code={"T_O":(255,64,0),"T_C":(64,0,255), "T_H":(0,255,64)} # RGB
#color_code=[(0,64,255),(255,0,64), (64,255,0)] # BGR
#color_code=[(255,64,0),(64,0,255), (0,255,64)] # RGB
# color_code=[(255,0,0),(0,0,255), (0,255,0)] # RGB
# class_clr = ["#ff4000","#4000ff","#00ff40"]
color_code= [(0,255,255), (20,255,255), (40,255,255),
            (60,255,255), (80,255,255), (100,255,255),
            (120,255,255),(140,255,255), (160,255,255)] 
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
# low_hsv =  [[(47, 100, 100)],
#     [(85, 100, 100)],
#     [(100, 100, 100)],
#     [(122, 100, 100)],
#     [(25, 100, 100)],
#     [(0, 100, 100),(175, 100, 100)],
#     [(145, 100, 100)],
#     [(0, 50, 50)],    
#     [(0, 50, 50)]]
# high_hsv = [[(57, 255, 255)],
#     [(95, 255, 255)],
#     [(110, 255, 255)],
#     [(132, 255, 255)],
#     [(35, 255, 255)],
#     [(5,255,255), (180, 255, 255)],
#     [(155, 255, 255)],
#     [(180, 255, 255)],
#     [(180, 225, 225)]]
low_hsv = [[(0, 100, 100),(175, 100, 100)], #_0
        [(15, 100, 100)], #_1
        [(35, 100, 100)], #_2
        [(55, 100, 100)], #_3
        [(75, 100, 100)], #_4
        [(95, 100, 100)], #_5
        [(115, 100, 100)], #_6
        [(135, 100, 100)], #_7
        [(155, 100, 100)], #_8
        [(0, 50, 50)], #_full
        [(0, 50, 50)]] #_uncertaim
high_hsv = [[(5,255,255), (180, 255, 255)], #_0
        [(25, 255, 255)], #_1
        [(45, 255, 255)], #_2 
        [(65, 255, 255)], #_3
        [(85, 255, 255)], #_4
        [(105, 255, 255)], #_5
        [(125, 255, 255)], #_6
        [(145, 255, 255)], #_7
        [(165, 255, 255)], #_8
        [(180, 255, 255)], #_full
        [(180, 225, 225)]] #_uncertain

iter = 1
detail = 0.002
no_class = 9
call_help = 9
img_h, img_w, img_d = 256, 384, 3


def cnts_extract(im_HSV,low_HSV,high_HSV, erode=False, dilate=False):
    im_threshold = cv.inRange(im_HSV, low_HSV, high_HSV)
    if dilate:
        kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
        im_threshold = cv.dilate(im_threshold, kernel, anchor=(0, 0), iterations=iter)
    if erode:
        kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
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
    src = cv.cvtColor(bgrimg, cv.COLOR_BGRA2RGBA)
    cv.imwrite(picpath, cv.cvtColor(src, cv.COLOR_BGR2RGB))

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
    parser = argparse.ArgumentParser(description='Code for autolabeling of 9 classes')
    parser.add_argument('--inp', help='Input file path', default="./images/1007_seg_SNUH_ICT4/140_8.png", type=str)
    parser.add_argument('--split', help='number of image stack', default=8, type=int)
    parser.add_argument("--img_width", dest="img_width", default=384, type=int, help="width of image in pixels")
    parser.add_argument("--img_height", dest="img_height", default=256, type=int, help="height of image in pixels")
    parser.add_argument("--n_channel", dest="n_channel", default=3, type=int, help="channels in the input data")
    parser.add_argument("--val_target", dest="val_target", default='test', choices=['train','test','user','update'], type= str, help="target dataset to validation")
        
    args = parser.parse_args()
    
    inp = os.path.relpath("".join(args.inp))
    base = os.path.splitext(args.inp)[0]

    img_org = Image.open(inp)
    img_org.load()
    img_org = np.asarray(img_org, dtype='uint8')
    splitted = np.vsplit(img_org,args.split) #TODO: auto parse of split number
    for i, sp in enumerate(splitted):
        
        max_index, uncertainty, full_area, full_cnt = critic_segmentation(sp)
        jsonpath = os.path.relpath(base+"_{}.json".format(i))
        picpath = os.path.relpath(base+"_{}.jpg".format(i))

        if full_area > 1e-3:
            #uncertainty = uncertain_area/full_area
            polygon = save_contour(full_cnt,uncertainty,max_index,jsonpath,args)
            key = max_index if uncertainty < 0.5 else call_help
            print("{} uncertainty: {}\tclass: {}\t{}".format(i,uncertainty,key,max_index))
            src = draw_pic(key,polygon,args)
            save_pic(src,picpath,args)

if __name__ == "__main__":
    main()