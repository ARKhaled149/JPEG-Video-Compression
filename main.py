import matplotlib.pyplot as plt
import numpy as np
from cv2 import vconcat
from numpy import asarray
import cv2
import image_slicer
from image_slicer import join
from PIL import ImageDraw, ImageFont
from PIL import Image
from sklearn.metrics import mean_squared_error
import glob

def YCBCR():
    capture = cv2.VideoCapture(r'D:\User Documents\Downloads\Video and Audio Technology\Practical\demo44.mp4')
    success, image = capture.read()
    count = 1
    while success:
        cv2.imwrite(r'D:\User Documents\Downloads\Video and Audio Technology\Practical\Project\OriginalFrames\frame%d.jpg' % count,image)
        image1 = cv2.imread(r'D:\User Documents\Downloads\Video and Audio Technology\Practical\Project\OriginalFrames\frame%d.jpg' % count)
        image2 = cv2.cvtColor(image1, cv2.COLOR_RGB2YCR_CB)
        y, cb, cr = cv2.split(image2)
        cv2.imwrite(r'D:\User Documents\Downloads\Video and Audio Technology\Practical\Project\YFrames\y%d.jpg' % count,y)
        cv2.imwrite( r'D:\User Documents\Downloads\Video and Audio Technology\Practical\Project\CBFrames\cb%d.jpg' % count, cb)
        cv2.imwrite(r'D:\User Documents\Downloads\Video and Audio Technology\Practical\Project\CRFrames\cr%d.jpg' % count, cr)
        success, image = capture.read()
        print('Read a new frame: ', success)
        count += 1
# YCBCR()

def YBlocks():
    count = 1
    while count < 70:
        tiles = image_slicer.slice(
            r'D:\User Documents\Downloads\Video and Audio Technology\Practical\Project\YFrames\y%d.jpg' % count,
            64, save=False)
        image_slicer.save_tiles(tiles,
                                directory=r'D:\User Documents\Downloads\Video and Audio Technology\Practical\Project\YBlocks', \
                                prefix='y%d' % count, format='jpeg')
        count += 1
# YBlocks()

def ImageToArray(path):
    img = Image.open(path)
    img.load()
    data = asarray(img, dtype="int32")
    return data

def ArrayToImage(array):
    img = Image.fromarray(array)
    return img

def Encoding():
    m = []
    m2 = []
    mv = []
    BestMatched = []
    i = 0
    w = 1
    j = 1
    k = 1
    x = 1
    y = 1
    TotalLoopCounter = 1
    u = 1
    while (w<69):
        while (j<9):
            while(k<9):
                matrix1 = ImageToArray(r'D:\User Documents\Downloads\Video and Audio Technology\Practical\Project\YBlocks\y%d_0%d_0%d.jpg' % (i,j,k))
                while (x<9):
                    while(y<9):
                        matrix2 = ImageToArray(r'D:\User Documents\Downloads\Video and Audio Technology\Practical\Project\YBlocks\y%d_0%d_0%d.jpg'%(w,x,y))
                        mse = mean_squared_error(matrix1, matrix2)
                        m.append(mse)
                        y += 1
                        TotalLoopCounter += 1
                    y = 1
                    x += 1
                MinimumValue = min(np.float64(m))
                m2.append(MinimumValue)
                z = 0
                xindex = 1
                yindex = 1
                while(z<64):
                    if m[z] == MinimumValue:
                        xindex = z//8
                        if(xindex == 0):
                            xindex = 1
                        yindex = z % 8
                        if (yindex == 0):
                            yindex = 1
                        break
                    z += 1
                NewXindex = j - xindex
                NewYindex = k - yindex
                a = [NewXindex,NewYindex]
                mv.append(a)
                MatrixB1 = ImageToArray(r'D:\User Documents\Downloads\Video and Audio Technology\Practical\Project\YBlocks\y%d_0%d_0%d.jpg'%(w,xindex,yindex))
                MatrixB2 = ImageToArray(r'D:\User Documents\Downloads\Video and Audio Technology\Practical\Project\YBlocks\y%d_0%d_0%d.jpg'%(i,j,k))
                BestMatchedBlock = MatrixB1 - MatrixB2
                BestMatched.append(BestMatchedBlock)
                FinalImage = ArrayToImage((BestMatchedBlock))
                cv2.imwrite(r'D:\User Documents\Downloads\Video and Audio Technology\Practical\Project\YFinalBlocks\FinalBlock%d.jpg'%u,np.float64(FinalImage))
                u += 1
                z = 0
                m = []
                x = 1
                k += 1
            j += 1
            k = 1
        i += 1
        w += 1
        j = 1
        # print("i = ",i)
    # print('Motion Vector for all frames appended 4352 MVs ',mv)
    # print('Motion Vectors 5th:6th ',mv[320:384])
    # print('BestMatched Blocks 5th:6th ',BestMatched[320:384])
    print(("BestMatched Matrices are "),BestMatched)
    # print("Total No of Loops = ",TotalLoopCounter)
    # print("Total No of Blocks + 1 = ",u)
# Encoding()

def CreateYFF():
    a = 1
    b = 9
    c = 17
    d = 25
    e = 33
    f = 41
    g = 49
    h = 57
    j = 1
    k = 1
    ff = 0
    tmp = []
    while(ff<68):
        while(j<9):
            image1 = cv2.imread(r'D:\User Documents\Downloads\Video and Audio Technology\Practical\Project\YFinalBlocks\FinalBlock%d.jpg'%a)
            image2 = cv2.imread(r'D:\User Documents\Downloads\Video and Audio Technology\Practical\Project\YFinalBlocks\FinalBlock%d.jpg'%b)
            image3 = cv2.imread(r'D:\User Documents\Downloads\Video and Audio Technology\Practical\Project\YFinalBlocks\FinalBlock%d.jpg'%c)
            image4 = cv2.imread(r'D:\User Documents\Downloads\Video and Audio Technology\Practical\Project\YFinalBlocks\FinalBlock%d.jpg'%d)
            image5 = cv2.imread(r'D:\User Documents\Downloads\Video and Audio Technology\Practical\Project\YFinalBlocks\FinalBlock%d.jpg'%e)
            image6 = cv2.imread(r'D:\User Documents\Downloads\Video and Audio Technology\Practical\Project\YFinalBlocks\FinalBlock%d.jpg'%f)
            image7 = cv2.imread(r'D:\User Documents\Downloads\Video and Audio Technology\Practical\Project\YFinalBlocks\FinalBlock%d.jpg'%g)
            image8 = cv2.imread(r'D:\User Documents\Downloads\Video and Audio Technology\Practical\Project\YFinalBlocks\FinalBlock%d.jpg'%h)
            img = cv2.vconcat([image1,image2,image3,image4,image5,image6,image7,image8])
            tmp.append(img)
            a += 1
            b += 1
            c += 1
            d += 1
            e += 1
            f += 1
            g += 1
            h += 1
            j += 1
        image2 = np.concatenate([tmp[0],tmp[1],tmp[2],tmp[3],tmp[4],tmp[5],tmp[6],tmp[7]],axis=1)
        cv2.imwrite(r'D:\User Documents\Downloads\Video and Audio Technology\Practical\Project\YFinalFrames\YFinalFrame%d.jpg'%ff,image2)
        tmp = []
        j = 1
        a += 56
        b += 56
        c += 56
        d += 56
        e += 56
        f += 56
        g += 56
        h += 56
        if(ff==68):
            break
        ff += 1
# CreateYFF()

def Merging():
    i = 0
    while(i<68):
        y = cv2.imread(r'D:\User Documents\Downloads\Video and Audio Technology\Practical\Project\YFinalFrames\YFinalFrame%d.jpg'%i)
        cb = cv2.imread(r'D:\User Documents\Downloads\Video and Audio Technology\Practical\Project\CBFrames\cb%d.jpg'%i)
        cr = cv2.imread(r'D:\User Documents\Downloads\Video and Audio Technology\Practical\Project\CRFrames\cr%d.jpg'%i)
        img = cv2.merge([y[:,:,0],cb[:,:,0],cr[:,:,0]])
        cv2.imwrite(r'D:\User Documents\Downloads\Video and Audio Technology\Practical\Project\AllFrames\Frame%i.jpg'%i,img)
        i += 1
# Merging()

def YCBCRtoRGB():
    i = 0
    while(i<68):
        allf = cv2.imread(r'D:\User Documents\Downloads\Video and Audio Technology\Practical\Project\AllFrames\Frame%d.jpg'%i)
        img = cv2.cvtColor(allf, cv2.COLOR_YCR_CB2BGR)
        cv2.imwrite(r'D:\User Documents\Downloads\Video and Audio Technology\Practical\Project\FinalFramesF\FFrame%d.jpg'%i,img)
        i += 1
# YCBCRtoRGB()

def CreateVideo():
    img_array = []
    for filename in glob.glob(r'D:\User Documents\Downloads\Video and Audio Technology\Practical\Project\FinalFramesF\*.jpg'):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)
    out = cv2.VideoWriter(
        r'D:\User Documents\Downloads\Video and Audio Technology\Practical\Project\ProjectVideo.avi',
        cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
# CreateVideo()
