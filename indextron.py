#%%=====================================================================================================================================================
# File conversion
# pyuic5 -x ui_main_window.ui -o ui_main_window.py    # .ui --> .py

#%%=====================================================================================================================================================
## %%writefile neural_cortex_app.py

# Neural cortex application

from time import time
from PyQt5 import QtCore, QtGui, QtWidgets
from ui_main_window import *
import sys
import cv2


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        self.ui.takePhotoPushButton.clicked.connect(self.takePhoto)
        self.ui.selectImagePushButton.clicked.connect(self.selectImage)
        
        
    def takePhoto(self):
        video = cv2.VideoCapture(0)
        video.set(3, 800)
        video.set(4, 600)
        ret, frame = video.read()
        video.release()
        
        if ret:
            pixmap = self.convert_opencv_to_pixmap(frame)
            self.ui.imageLabel.setPixmap(pixmap)
            
            cv2.imwrite('img.jpg', frame)
        
        
    def convert_opencv_to_pixmap(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qt_image)
        return pixmap
        
        
    def selectImage(self):
        image_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Select an Image', '', 'Image Files (*.png *.jpg *.jpeg)')
        pixmap = QtGui.QPixmap(image_path)
        pixmap = pixmap.scaled(self.ui.imageLabel.width(), self.ui.imageLabel.height(), QtCore.Qt.KeepAspectRatio)
        self.ui.imageLabel.setPixmap(pixmap)
        
        
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

#%%=====================================================================================================================================================
# Image resizing

from PIL import Image


image1 = Image.open('images/items/1.jpg').resize((800, 600))
image2 = Image.open('images/items/2.jpg').resize((800, 600))
image3 = Image.open('images/items/3.jpg').resize((800, 600))

# image1.save('images/items/1_0.jpg')
# image2.save('images/items/1_1.jpg')
# image3.save('images/items/1_2.jpg')

#%%=====================================================================================================================================================
# Image concatenation

from PIL import Image


def get_concat_h(img1, img2):
    dst = Image.new('RGB', (img1.width + img2.width, img1.height))
    dst.paste(img1, (0, 0))
    dst.paste(img2, (img1.width, 0))
    return dst

def get_concat_v(img1, img2):
    dst = Image.new('RGB', (img1.width, img1.height + img2.height))
    dst.paste(img1, (0, 0))
    dst.paste(img2, (0, img1.height))
    return dst


first_object1 = Image.open('images/teana/teana1.jpg')
first_object2 = Image.open('images/teana/teana2.jpg')
second_object1 = Image.open('images/raf/raf1.jpg')
second_object2 = Image.open('images/raf/raf2.jpg')
second_object3 = Image.open('images/raf/raf3.jpg')

first_object1_2 = get_concat_v(first_object1, first_object2)
second_object1_2 = get_concat_v(second_object1, second_object2)
second_object1_3 = get_concat_v(second_object1, second_object3)
first_object1_second_object1 = get_concat_v(first_object1, second_object1)

# first_object1_2.save('teana1-2.jpg')
# second_object1_2.save('raf1-2.jpg')
# second_object1_3.save('raf1-3.jpg')
# first_object1_second_object1.save('teana1-raf1.jpg')


sh_first_object1_2 = Image.open('plots_and_tables/signature_histograms/teana-raf/sh_teana1-2.png')
sh_second_object1_2 = Image.open('plots_and_tables/signature_histograms/teana-raf/sh_raf1-2.png')
sh_second_object1_3 = Image.open('plots_and_tables/signature_histograms/teana-raf/sh_raf1-3.png')
sh_first_object1_second_object1 = Image.open('plots_and_tables/signature_histograms/teana-raf/sh_teana1-raf1.png')

sh_first_object1_2.thumbnail((sh_first_object1_2.width, first_object1_2.height))
sh_second_object1_2.thumbnail((sh_second_object1_2.width, second_object1_2.height))
sh_second_object1_3.thumbnail((sh_second_object1_3.width, second_object1_3.height))
sh_first_object1_second_object1.thumbnail((sh_first_object1_second_object1.width, first_object1_second_object1.height))

first_object1_2_comb = get_concat_h(sh_first_object1_2, first_object1_2)
second_object1_2_comb = get_concat_h(sh_second_object1_2, second_object1_2)
second_object1_3_comb = get_concat_h(sh_second_object1_3, second_object1_3)
first_object1_second_object1_comb = get_concat_h(sh_first_object1_second_object1, first_object1_second_object1)

# first_object1_2_comb.save('teana1-2_comb.jpg')
# second_object1_2_comb.save('raf1-2_comb.jpg')
# second_object1_3_comb.save('raf1-3_comb.jpg')
# first_object1_second_object1_comb.save('teana1-raf1_comb.jpg')

#%%=====================================================================================================================================================
%%time

# Image subtraction

from numba import njit
from PIL import Image
import numpy as np


# image0 = Image.open('images/items/items1_0.jpg')
# image1 = Image.open('images/items/items1_1.jpg')

# image0 = Image.open('images/teana/teana0.jpg')
# image1 = Image.open('images/teana/teana1.jpg')
# image0 = Image.open('images/raf/raf0.jpg')
# image1 = Image.open('images/raf/raf1.jpg')

image0 = Image.open('images/rose/rose0.jpg')
image1 = Image.open('images/rose/rose1.jpg')
# image0 = Image.open('images/cat/cat0.jpg')
# image1 = Image.open('images/cat/cat1.jpg')

image_array0 = np.array(image0.resize((400, 300)))
image_array1 = np.array(image1.resize((400, 300)))


@njit
def GetSignature(image_array0, image_array1, st=16, thr=32):
    h, w = image_array1.shape[:2]
    sig_ltH = 0

    obj_sig = np.zeros((h*w, 3), dtype=np.uint8)
    picture_array = np.copy(image_array1)

    image_array0 = image_array0.astype('int16')
    image_array1 = image_array1.astype('int16')

    for y in range(st, h-st):
        for x in range(st, w-st):

            channels1 = image_array1[y, x]
            mn = 999999

            for m in range(-st, st+1):
                for n in range(-st, st+1):

                    channels0 = image_array0[y+m, x+n]

                    dd = (abs(channels1[0] - channels0[0])
                    + abs(channels1[1] - channels0[1])
                    + abs(channels1[2] - channels0[2]))

                    if dd < mn:
                        mn = dd

            if mn >= thr:
                obj_sig[sig_ltH] = image_array1[y, x]
                sig_ltH += 1
                picture_array[y, x] = [255, 0 ,0]

    obj_sig = obj_sig[:sig_ltH]
    return obj_sig, picture_array


obj_sig, picture_array = GetSignature(image_array0, image_array1, st=16, thr=32)
# np.save('object_signatures/obj_sig.npy', obj_sig)

picture = Image.fromarray(picture_array)
# picture.save('00_1.jpg')
picture.show()

#%%=====================================================================================================================================================
%%time

# Indextron

from numba import njit #, prange
from PIL import Image
import numpy as np


@njit
def Read_Set_114(cv_N1, root_A1, L_114,
                 N_114_maX, reg_sz114,
                 pixel, lth,
                 rad, h_dist_114):

    scc = np.zeros(N_114_maX+1, dtype='int64')
    lstT1 = np.zeros(N_114_maX, dtype='int64')
    lngtH1, wll = 0, 0

    for reg in range(lth):

        for n in range(-rad, rad+1):
            add = pixel[reg] + n
        
            if 0 <= add < reg_sz114:
                nam = root_A1[add, reg]

                while nam > 0:

                    if reg - scc[nam] <= h_dist_114:
                        scc[nam] += 1

                        if scc[nam] == lth:
                            lstT1[lngtH1] = nam
                            lngtH1 += 1

                    nam = cv_N1[nam, reg]

    for n in range(lngtH1):
        nam = lstT1[n]

        if lth - scc[nam] <= h_dist_114 and abs(lth - L_114) <= h_dist_114:
            lstT1[wll] = nam
            wll += 1

    return lstT1, wll


@njit
def Write_Set_114(cv_N1, root_A1, root_T1,
                  pixel, lth, N_114):

    for reg in range(lth):
        add = pixel[reg]

        if root_A1[add, reg] == 0:
            root_A1[add, reg] = N_114
        else:
            cv_N1[root_T1[add, reg], reg] = N_114

        root_T1[add, reg] = N_114
    
    return cv_N1, root_A1, root_T1


@njit
def Read_Set_115(cv_N2, root_A2,
                 N_115_maX, ft, lth):

    scc = np.zeros(N_115_maX+1, dtype='int64')
    lstT2 = np.zeros(N_115_maX, dtype='int64')
    lngtH2, win, mx, SSS = 0, 0, 0, 0

    for reg in range(lth):
        add = 0
        nam = root_A2[add, ft[reg]]

        while nam > 0:
            scc[nam] += 1

            if scc[nam] == 1:
                lstT2[lngtH2] = nam
                lngtH2 += 1
                
            nam = cv_N2[nam, ft[reg]]

    for n in range(lngtH2):
        nam = lstT2[n]

        if lth - scc[nam] <= lth:

            if scc[nam] > mx:
                mx = scc[nam]
                win = nam
                SSS = scc[nam]

    return win, SSS


@njit
def Write_Set_115(cv_N2, root_A2, root_T2,
                  ft, lth, N_115):

    for reg in range(lth):
        add = 0

        if root_A2[add, ft[reg]] == 0:
            root_A2[add, ft[reg]] = N_115
        else:
            cv_N2[root_T2[add, ft[reg]], ft[reg]] = N_115

        root_T2[add, ft[reg]] = N_115

    return cv_N2, root_A2, root_T2


@njit
def UpdateClassHistogramAndList(lstT, lngtH, ftT, lthH, SH):

    for n in range(lngtH):

        if SH[lstT[n]-1] == 0:
            ftT[lthH] = lstT[n]
            lthH += 1

        SH[lstT[n]-1] += 1
    
    return ftT, lthH, SH


@njit
def thr_filter(ftT, lthH, SH, SHM, T1, sig_num, learn):
    ft = np.zeros(lthH, dtype='int64')
    lth = 0

    for n in range(lthH):

        if SH[ftT[n]-1] >= T1:
            ft[lth] = ftT[n]
            lth += 1

            if learn:
                SHM[sig_num, ftT[n]-1] = SH[ftT[n]-1]
    
    return ft, lth, SHM


@njit
def threshold_filter(ftT, lthH, SH, SHM, T1, sig_num=0, learn=False):
    T2 = T1

    ft, lth, SHM = thr_filter(ftT, lthH, SH, SHM, T1, sig_num, learn)

    if not learn and lth <= 7:
        T2 = T1 // 2
        ft, lth, SHM = thr_filter(ftT, lthH, SH, SHM, T2, sig_num, learn)

    return ft, lth, SHM, T2


@njit
def inhibitor_filter(ftT, lthH, SH, SHM, T1, N_115):

    for m in range(1, N_115+1):
        for n in range(lthH):

            if SH[ftT[n]-1] >= T1:
                SHM[m, ftT[n]-1] = 0
                SH[ftT[n]-1] = 0

    return SHM


@njit
def BasicWave(mmap, y, x, wind_raD, N_114_maX, image_array):
    yx_lst1 = np.zeros((N_114_maX, 2), dtype='int64')
    yx_lst2 = np.zeros((N_114_maX, 2), dtype='int64')

    im_hghT, im_wdtH = image_array.shape[:2]
    yx_lth1, yx_lth2 = 0, 1

    if mmap[y, x] == 1:
        yx_lst1[yx_lth1] = y, x
        yx_lth1 = 1
        mmap[y, x] = 0

        while True:

            for yv in range(y-wind_raD, y+wind_raD+1):
                        
                if 0 <= yv < im_hghT:
        
                    for xv in range(x-wind_raD, x+wind_raD+1):

                        if 0 <= xv < im_wdtH:

                            if mmap[yv, xv] == 1:
                                yx_lst1[yx_lth1] = yv, xv
                                yx_lst2[yx_lth2-1] = yv, xv
                                yx_lth1 += 1
                                yx_lth2 += 1
                                mmap[yv, xv] = 0

                            if yx_lth1 >= N_114_maX:
                                return yx_lst1, yx_lth1

            yx_lth2 -= 1
            if yx_lth2 > 0:
                y, x = yx_lst2[yx_lth2-1]
            else:
                return yx_lst1, yx_lth1

    else:
        return yx_lst1, yx_lth1


@njit
def FindSignatureInPicture(cv_N1, root_A1, L_114,
                           N_114_maX, reg_sz114, N_115,
                           lth, SHM, T1, rad, h_dist_114,
                           image_array):

    im_hghT, im_wdtH = image_array.shape[:2]

    mmap = np.zeros((im_hghT, im_wdtH), dtype=np.uint8)
    colored_image_array = np.full_like(image_array, 255)

    for y in range(im_hghT):
        for x in range(im_wdtH):

            pixel = image_array[y, x]

            lstT, lngtH = Read_Set_114(cv_N1, root_A1, L_114,
                                       N_114_maX, reg_sz114,
                                       pixel, lth,
                                       rad, h_dist_114)

            if lngtH == 0:
                ewin = 0
                continue

            mx = 0

            for m in range(1, N_115+1):
                for n in range(lngtH):

                    if SHM[m, lstT[n]-1] > mx:
                        mx = SHM[m, lstT[n]-1]

            if mx >= T1:
                ewin = 2
                mmap[y, x] = 1
            else:
                ewin = 1

            if ewin == 2:
                colored_image_array[y, x] = [255, 0, 0]
            elif ewin == 1:
                colored_image_array[y, x] = [0, 255, 0]

    return colored_image_array, mmap


@njit
def inhibitor(cv_N1, root_A1, L_114, root_T2,
              N_114_maX, reg_sz114,
              lth, SHM, T1, rad, h_dist_114,
              wind_raD, sg_thr, image_array0):

    N_115 = max(root_T2.flatten())

    colored_image_array0, mmap0 = FindSignatureInPicture(cv_N1, root_A1, L_114,
                                                         N_114_maX, reg_sz114, N_115,
                                                         lth, SHM, T1, rad, h_dist_114,
                                                         image_array0)

    for y1 in range(len(mmap0)):
        for x1 in range(len(mmap0[0])):

            yx_lst, yx_lth = BasicWave(mmap0, y1, x1, wind_raD, N_114_maX, image_array0)

            if yx_lth > sg_thr:
                SH0 = np.zeros(N_114_maX, dtype='int64')
                ftT0 = np.zeros(N_114_maX, dtype='int64')
                lthH0 = 0
                
                for y2, x2 in yx_lst[:yx_lth]:
                    pixel = image_array0[y2, x2]

                    lstT0, lngtH0 = Read_Set_114(cv_N1, root_A1, L_114,
                                                 N_114_maX, reg_sz114,
                                                 pixel, lth,
                                                 rad, h_dist_114)

                    ftT0, lthH0, SH0 = UpdateClassHistogramAndList(lstT0, lngtH0, ftT0, lthH0, SH0)

                SHM = inhibitor_filter(ftT0, lthH0, SH0, SHM, T1, N_115)

    return SHM


@njit
def learn_signature(cv_N1, root_A1, root_T1, L_114, N_114_maX, reg_sz114,
                    cv_N2, root_A2, root_T2,
                    sig, lth, SHM, T1, rad, h_dist_114):

    SH1 = np.zeros(N_114_maX, dtype='int64')
    ftT1 = np.zeros(N_114_maX, dtype='int64')
    lthH1 = 0

    N_114 = max(cv_N1.flatten())
    N_115 = max(root_T2.flatten())+1

    for pixel in sig:
        lstT1, lngtH1 = Read_Set_114(cv_N1, root_A1, L_114,
                                     N_114_maX, reg_sz114,
                                     pixel, lth,
                                     rad, h_dist_114)

        if lngtH1 == 0:
            N_114 += 1
            lngtH1 = 1
            lstT1[0] = N_114

            cv_N1, root_A1, root_T1 = Write_Set_114(cv_N1, root_A1, root_T1,
                                                    pixel, lth, N_114)

        ftT1, lthH1, SH1 = UpdateClassHistogramAndList(lstT1, lngtH1, ftT1, lthH1, SH1)

    ft1, lth1, SHM, T2 = threshold_filter(ftT1, lthH1, SH1, SHM, T1, sig_num=N_115, learn=True)

    cv_N2, root_A2, root_T2 = Write_Set_115(cv_N2, root_A2, root_T2,
                                            ft1, lth1, N_115)

    return (cv_N1, root_A1, root_T1,
            cv_N2, root_A2, root_T2,
            SHM, ft1, lth1, ftT1, lthH1, SH1)


@njit
def recognize_sig(cv_N1, root_A1, L_114, N_114_maX, reg_sz114,
                  cv_N2, root_A2, N_115_maX,
                  lth, SHM, T1, rad, h_dist_114,
                  mmap, wind_raD, sg_thr, image_array):

    yx_lst_max = np.zeros((0, 0), dtype='int64')
    ft_max = np.zeros(0, dtype='int64')
    ftT_max = np.zeros(0, dtype='int64')
    SH_max = np.zeros(0, dtype='int64')
    WIN2, SSS_max, lth_max, lthH_max, T2_max, yx_lth_max = 0, 0, 0, 0, 0, 0

    for y1 in range(len(mmap)):
        for x1 in range(len(mmap[0])):

            yx_lst, yx_lth = BasicWave(mmap, y1, x1, wind_raD, N_114_maX, image_array)

            if yx_lth > sg_thr:
                SH2 = np.zeros(N_114_maX, dtype='int64')
                ftT2 = np.zeros(N_114_maX, dtype='int64')
                lthH2 = 0
                
                for y2, x2 in yx_lst[:yx_lth]:
                    pixel = image_array[y2, x2]

                    lstT2, lngtH2 = Read_Set_114(cv_N1, root_A1, L_114,
                                                 N_114_maX, reg_sz114,
                                                 pixel, lth,
                                                 rad, h_dist_114)

                    ftT2, lthH2, SH2 = UpdateClassHistogramAndList(lstT2, lngtH2, ftT2, lthH2, SH2)

                ft2, lth2, SHM, T2 = threshold_filter(ftT2, lthH2, SH2, SHM, T1, learn=False)

                ewin, SSS = Read_Set_115(cv_N2, root_A2,
                                         N_115_maX, ft2, lth2)

                if SSS > SSS_max:
                    SSS_max = SSS
                    ft_max = ft2
                    lth_max = lth2
                    ftT_max = ftT2
                    lthH_max = lthH2
                    SH_max = SH2
                    T2_max = T2
                    yx_lst_max = yx_lst
                    yx_lth_max = yx_lth

                    if SSS >= 3:
                        WIN2 = ewin

                if SSS >= 3:
                    print(ewin, SSS, yx_lth)

    return (WIN2, SSS_max, ft_max, lth_max,
            ftT_max, lthH_max, SH_max, T2_max,
            yx_lst_max, yx_lth_max)


def recognize_signature(cv_N1, root_A1, L_114, N_114_maX, reg_sz114,
                        cv_N2, root_A2, root_T2, N_115_maX,
                        lth, SHM, T1, rad, h_dist_114,
                        wind_raD, sg_thr, image_array):

    N_115 = max(root_T2.flatten())

    colored_image_array, mmap = FindSignatureInPicture(cv_N1, root_A1, L_114,
                                                       N_114_maX, reg_sz114, N_115,
                                                       lth, SHM, T1, rad, h_dist_114,
                                                       image_array)

    (WIN2, SSS_max, ft_max, lth_max,
     ftT_max, lthH_max, SH_max, T2_max,
     yx_lst_max, yx_lth_max) = recognize_sig(cv_N1, root_A1, L_114, N_114_maX, reg_sz114,
                                             cv_N2, root_A2, N_115_maX,
                                             lth, SHM, T1, rad, h_dist_114,
                                             mmap, wind_raD, sg_thr, image_array)

    for y, x in yx_lst_max[:yx_lth_max]:

        for yy in range(2):
            for xx in range(2):

                if y+yy < len(image_array) and x+xx < len(image_array[0]):
                    colored_image_array[y+yy, x+xx] = [255, 0, 255]

    colored_image = Image.fromarray(colored_image_array)
    # colored_image.save('0_0.jpg')
    colored_image.show()

    return (WIN2, SSS_max, ft_max, lth_max,
            ftT_max, lthH_max, SH_max, T2_max)




# Learning

# items
# image0 = Image.open('images/items/items1_0.jpg')
# image1 = Image.open('images/items/items1_1.jpg')
# image1 = Image.open('images/items/items1_2.jpg')

# teana
# image0 = Image.open('images/teana/teana0.jpg')
# image1 = Image.open('images/teana/teana1.jpg')
# image1 = Image.open('images/teana/teana2.jpg')
# raf
# image0 = Image.open('images/raf/raf0.jpg')
# image1 = Image.open('images/raf/raf1.jpg')
# image1 = Image.open('images/raf/raf2.jpg')
# image1 = Image.open('images/raf/raf3.jpg')

# rose
image0 = Image.open('images/rose/rose0.jpg')
image1 = Image.open('images/rose/rose1.jpg')
# image1 = Image.open('images/rose/rose2.jpg')
# cat
# image0 = Image.open('images/cat/cat0.jpg')
# image1 = Image.open('images/cat/cat1.jpg')
# image1 = Image.open('images/cat/cat2.jpg')
# image1 = Image.open('images/cat/cat3.jpg')

image_array0 = np.array(image0.resize((400, 300)))
image_array1 = np.array(image1.resize((400, 300)))
# image_array2 = np.array(image2.resize((400, 300)))

# signatures
# sig1 = np.load('object_signatures/obj_sig.npy')
# sig1 = np.load('object_signatures/eraser_sig.npy')
# sig1 = np.load('object_signatures/teana_sig.npy')
# sig2 = np.load('object_signatures/raf_sig.npy')
sig1 = np.load('object_signatures/rose_sig.npy')
sig2 = np.load('object_signatures/cat_sig.npy')

lth = len(sig1[0])

rad = 4
h_dist_114 = 0

N_114_maX = 10000
nof_reg114 = 3
reg_sz114 = 256

N_115_maX = 10
nof_reg115 = 10000
reg_sz115 = 1

T1 = 32
wind_raD = 12
sg_thr = 200

load_progress = False
save_progress = False

SHM = np.zeros((N_115_maX, nof_reg115), dtype='int64')

cv_N1 = np.zeros((N_114_maX+1, nof_reg114), dtype='int64')
root_A1 = np.zeros((reg_sz114, nof_reg114), dtype='int64')
root_T1 = np.zeros((reg_sz114, nof_reg114), dtype='int64')
L_114 = nof_reg114

cv_N2 = np.zeros((N_115_maX+1, nof_reg115), dtype='int64')
root_A2 = np.zeros((reg_sz115, nof_reg115), dtype='int64')
root_T2 = np.zeros((reg_sz115, nof_reg115), dtype='int64')

if load_progress:
    SHM = np.load('saved_progress/SHM.npy')
    cv_N1 = np.load('saved_progress/cv_N1.npy')
    root_A1 = np.load('saved_progress/root_A1.npy')
    root_T1 = np.load('saved_progress/root_T1.npy')

    cv_N2 = np.load('saved_progress/cv_N2.npy')
    root_A2 = np.load('saved_progress/root_A2.npy')
    root_T2 = np.load('saved_progress/root_T2.npy')
    print('(Progress Loaded)\n')

print('Learning Signature 1...\n')
(cv_N1, root_A1, root_T1,
 cv_N2, root_A2, root_T2,
 SHM, ft00, lth00,
 ftT00, lthH00, SH00) = learn_signature(cv_N1, root_A1, root_T1, L_114, N_114_maX, reg_sz114,
                                        cv_N2, root_A2, root_T2,
                                        sig1, lth, SHM, T1, rad, h_dist_114)

print('Inhibiting...\n')
SHM = inhibitor(cv_N1, root_A1, L_114, root_T2,
                N_114_maX, reg_sz114,
                lth, SHM, T1, rad, h_dist_114,
                wind_raD, sg_thr, image_array0)

print('Learning Signature 2...\n')
(cv_N1, root_A1, root_T1,
 cv_N2, root_A2, root_T2,
 SHM, ft00, lth00,
 ftT00, lthH00, SH00) = learn_signature(cv_N1, root_A1, root_T1, L_114, N_114_maX, reg_sz114,
                                        cv_N2, root_A2, root_T2,
                                        sig2, lth, SHM, T1, rad, h_dist_114)

if save_progress:
    np.save('saved_progress/SHM.npy', SHM)
    np.save('saved_progress/cv_N1.npy', cv_N1)
    np.save('saved_progress/root_A1.npy', root_A1)
    np.save('saved_progress/root_T1.npy', root_T1)

    np.save('saved_progress/cv_N2.npy', cv_N2)
    np.save('saved_progress/root_A2.npy', root_A2)
    np.save('saved_progress/root_T2.npy', root_T2)
    print('(Progress Saved)\n')




# Recognition

print('Recognizing Image...\n')
(WIN2, SSS1, ft01, lth01,
 ftT01, lthH01, SH01, T21) = recognize_signature(cv_N1, root_A1, L_114, N_114_maX, reg_sz114,
                                                 cv_N2, root_A2, root_T2, N_115_maX,
                                                 lth, SHM, T1, rad, h_dist_114,
                                                 wind_raD, sg_thr, image_array1)
print('Winner:', WIN2, ' Score:', SSS1, '\n')

# print('Recognizing Image 2...\n')
# (WIN2, SSS2, ft02, lth02,
#  ftT02, lthH02, SH02, T22) = recognize_signature(cv_N1, root_A1, L_114, N_114_maX, reg_sz114,
#                                                  cv_N2, root_A2, root_T2, N_115_maX,
#                                                  lth, SHM, T1, rad, h_dist_114,
#                                                  wind_raD, sg_thr, image_array2)
# print('Winner2:', WIN2, ' Score2:', SSS2, '\n')

#%%=====================================================================================================================================================
# Shows signature histogram

import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.rcParams['figure.dpi'] = 1000


# plt.axhline(T21, lw=1, color='tab:orange')
# plt.axhline(-T22, lw=1, color='tab:orange')

# plt.plot(ft1[:lth1], [T21]*lth1, marker='.', ms=4, mec='tab:red', mfc='tab:red', lw=1, color='tab:orange', label='Threshold')
# plt.plot(ft2[:lth2], [-T22]*lth2, marker='.', ms=4, mec='tab:red', mfc='tab:red', lw=1, color='tab:orange')

# plt.bar(ftT1[:lthH1], SH1[:lthH1], color='tab:blue', label='Learning')
# plt.bar(np.sort(ftT2[:lthH2]), SH2[SH2!=0]*-1, color='tab:green', label='Recognition')


plt.axhline(T21, lw=0.5, color='black', label='Пороговый уровень')
plt.axhline(0, lw=0.5, color='black')
plt.axhline(-T22, lw=0.5, color='black')

plot0 = np.array([i for i in ft01[:lth01] if i in ft02[:lth02]])
plot0 = list(map(int, plot0/2))
plot1 = list(map(int, ft01[:lth01]/2))
plot2 = list(map(int, ft02[:lth02]/2))

plt.plot(plot1, [T21]*lth01, marker='o', ms=1, mec='red', mfc='red', lw=0.5, color='black', label='Найденный признак')
plt.plot(plot0, [0]*len(plot0), marker='o', ms=3, mec='red', mfc='red', lw=0.5, color='black')
plt.plot(plot2, [-T22]*lth02, marker='o', ms=1, mec='red', mfc='red', lw=0.5, color='black')

bar1 = list(map(int, np.sort(ftT01[:lthH01])/2))
bar2 = list(map(int, np.sort(ftT02[:lthH02])/2))

plt.bar(bar1, SH01[SH01!=0], width=1, color='darkblue', label=f'Teana 1')
plt.bar(bar2, SH02[SH02!=0]*-1, width=1, color='darkgreen', label=f'Teana 2')

plt.ylim(-100, 100)

locs, labels = plt.yticks()
pos_locs = list(map(int, abs(locs)))
plt.yticks(ticks=locs, labels=pos_locs)


# plt.title('Signature Histograms (Eraser)')
# plt.title(f'Выделение общих признаков Teana 1-2 при пороге {T1} ({int(T1/2)})')
# plt.title(f'Выделение общих признаков Raf 1-2 при пороге {T1} ({int(T1/2)})')
# plt.title(f'Выделение общих признаков Teana 1 - Raf 1 при пороге {T1} ({int(T1/2)})')
plt.xlabel('Класс')
plt.ylabel('Частота')
plt.legend()


# plt.savefig('sh_eraser.png', bbox_inches='tight', facecolor='w')

# plt.savefig(f'sh_teana1-2.png', bbox_inches='tight', facecolor='w')
# plt.savefig(f'sh_raf1-2.png', bbox_inches='tight', facecolor='w')
# plt.savefig(f'sh_teana1-raf1.png', bbox_inches='tight', facecolor='w')

# plt.savefig(f'sh_rose1-2.png', bbox_inches='tight', facecolor='w')
# plt.savefig(f'sh_cat1-2.png', bbox_inches='tight', facecolor='w')
# plt.savefig(f'sh_rose1-cat1.png', bbox_inches='tight', facecolor='w')
plt.show()

#%%=====================================================================================================================================================
# Gene expression data preparation for NC

import pandas as pd
import numpy as np


dataset = pd.read_csv('datasets/gene_expression_cancer_RNA-seq_801x20531/data.csv', index_col=0)
labels = pd.read_csv('datasets/gene_expression_cancer_RNA-seq_801x20531/labels.csv', index_col=0)

classes = {'PRAD':1, 'LUAD':2, 'BRCA':3, 'KIRC':4, 'COAD':5}


norm_dataset = 255 * (dataset - dataset.min()) / (dataset.max() - dataset.min())
norm_dataset = norm_dataset.fillna(0)
norm_dataset = norm_dataset.astype('uint8')

# norm_dataset.to_csv('genes801x20531.txt', header=False, index=False)
# labels.to_csv('labels801x1.txt', header=False, index=False)


np_norm_dataset = norm_dataset.to_numpy()
# np.save('genes801x20531.npy', np_norm_dataset)


np_labels = np.squeeze(labels.to_numpy())

for cclass in classes:
    np_labels[np_labels == cclass] = classes[cclass]

np_labels = np_labels.astype('uint8')

# np.save('labels801.npy', np_labels)
# np.savetxt('labels801.txt', np_labels, fmt='%i', delimiter=',')

#%%=====================================================================================================================================================
# Covertype data preparation for NC

import numpy as np


dataset = np.loadtxt('datasets/covertype_581012x54/covtype.data', delimiter=',')

norm_dataset = np.zeros((dataset.shape[0],12), dtype='uint8')
labels = np.zeros(dataset.shape[0], dtype='uint8')


labels[:] = dataset[:,-1]
norm_dataset[:,6:9] = dataset[:,6:9]

for col in range(10):

    if col not in [6,7,8]:
        colon = dataset[:,col]
        norm_colon = 255 * (colon - colon.min()) / (colon.max() - colon.min())
        norm_dataset[:,col] = norm_colon

norm_dataset[:,10] = np.argmax(dataset[:,10:14], axis=1)
norm_dataset[:,11] = np.argmax(dataset[:,14:-1], axis=1)


# np.save('covertype581012x12.npy', norm_dataset)
# np.save('labels581012.npy', labels)

print(norm_dataset.shape, norm_dataset.dtype, norm_dataset)
print(labels.shape, labels.dtype, labels)

#%%=====================================================================================================================================================
# Neural cortex

from numba import njit, prange, cuda
import numpy as np
# import math


# def signature_histogram(winners, SHM, signature, T1, learn=True):
#     classes, SH = np.unique(winners, return_counts=True)
#     ft2 = classes[SH > T1].astype('int64')
#     lth2 = len(ft2)

#     if learn:
#         SHM[signature, :len(SH)] = SH * (SH > T1)
    
#     return ft2, lth2, SHM


# #                                                   Step 3  level 2
# @njit
# def Read_Set_115(cv_N2, root_A2, L_115,
#                  ft2, lth2, N_115_maX, reg_sz115):

#     scc = np.zeros(N_115_maX+1, dtype='int64')
#     lstT2 = np.zeros(N_115_maX, dtype='int64')
#     lngtH2 = 0

#     win, mx = 0, 0
#     LLL, SSS, TTT = 0, 0, 0

#     for reg in range(lth2):
#         add = 0

#         nam = root_A2[add, reg]

#         while nam > 0:
#             scc[nam] += 1

#             if scc[nam] == 1:
#                 lstT2[lngtH2] = nam
#                 lngtH2 += 1
                
#             nam = cv_N2[nam, reg]

#     for n in range(lngtH2):
#         nam = lstT2[n]

#         if lth2 - scc[nam] <= lth2 * 2 / 3:

#             if scc[nam] > mx:
#                 mx = scc[nam]
#                 win = nam
#                 LLL = lth2
#                 SSS = scc[nam]
#                 TTT = L_115[nam]

#     return lstT2, win, LLL, SSS, TTT


# #                                                   Step 4  level 2
# @njit
# def Write_Set_115(cv_N2, root_A2, root_T2,
#                   ft2, lth2, win):

#     for reg in range(lth2):
#         add = 0

#         if root_A2[add, reg] == 0:
#             root_A2[add, reg] = win
#         else:
#             cv_N2[root_T2[add, reg], reg] = win

#         root_T2[add, reg] = win

#     return cv_N2, root_A2, root_T2


# #                                                   Step 3
# @njit#(parallel=True)
# def Test1_Read_Vector_111(cv_N, cv_R, reg_sz111,
#                           ft, lth, radius, hamming_dist_111,
#                           sc, lstT, lngtH, loop_counter):

#     for n in range(lngtH):
#         sc[lstT[n]] = 0

#     lngtH, cnd = 0, 0

#     for reg in range(lth):
        
#         for rad in range(-radius, radius+1):
#             add = ft[reg] + rad

#             if 0 <= add < reg_sz111:
#                 add += reg * reg_sz111
#                 nam = cv_R[add]

#                 while nam > 0:
#                     loop_counter += 1        # loop counter

#                     if sc[nam] >= reg - hamming_dist_111:

#                         if sc[nam] == 0:
#                             lstT[lngtH] = nam
#                             lngtH += 1

#                         sc[nam] += 1
#                         cnd = nam

#                     nam = cv_N[nam, reg]

#         if sc[cnd] <= reg - hamming_dist_111:
#             win = 0
#             return sc, lstT, lngtH, win, loop_counter

#     win = cnd
#     return sc, lstT, lngtH, win, loop_counter


# #                                                   Step 4
# @njit#(parallel=True)
# def Write_Vector_111(cv_N, cv_R, cv_T, reg_sz111,
#                      ft, lth, win):

#     for reg in range(lth):
#         add = ft[reg] + reg * reg_sz111

#         if cv_R[add] == 0:
#             cv_R[add] = win
#         else:
#             cv_N[cv_T[add], reg] = win

#         cv_T[add] = win

#     return cv_N, cv_R, cv_T


# #                                                   Step 2    alpha cortex
# @njit#(parallel=True)
# def VectorCortex109(cv_R, cv_L, cv_N, cv_T, cv_A, classes, labels,
#                     reg_sz111, ft, lth, radius, hamming_dist_109, freeze,
#                     sc, lstT, lngtH, N_111, vec,
#                     right_guesses, wrong_guesses, unrec_vectors, loop_counter,
#                     class_vectors):

#     #    --> step 3
#     sc, lstT, lngtH, win, loop_counter = Test1_Read_Vector_111(cv_N, cv_R, reg_sz111,
#                                                                ft, lth, radius, hamming_dist_109,
#                                                                sc, lstT, lngtH, loop_counter)

#     if classes[win] == labels[vec]:
#         right_guesses += 1
#     elif win == 0:
#         unrec_vectors += 1
#     else:
#         wrong_guesses += 1

#     if win == 0 and not freeze:
#         N_111 += 1
#         win = N_111

#         classes[N_111] = labels[vec]
#         class_vectors[N_111-1] = ft

#         #    --> step 4
#         cv_N, cv_R, cv_T = Write_Vector_111(cv_N, cv_R, cv_T, reg_sz111,
#                                             ft, lth, win)

#     return (cv_R, cv_L, cv_N, cv_T, cv_A, classes,
#             sc, lstT, lngtH, N_111, win,
#             right_guesses, wrong_guesses, unrec_vectors,
#             loop_counter, class_vectors)


# @njit
# def create_percent_array(number, step=1, mmin=0, mmax=100):
#     length = (mmax - mmin) // step + 1
#     percent_array = np.zeros(length, dtype='int64')

#     for i, percent in enumerate(range(mmin, mmax + step, step)):
#         percent_array[i] = number * percent // 100
    
#     return percent_array


# @njit
# def FeatureRegularizer(i_rangE):
#     rF = np.zeros(i_rangE+1, dtype='int64')
#     o_rang = (-3 + math.sqrt(9 + 8 * i_rangE)) // 2 + 2
#     m = 0

#     for n in range(1, o_rang):
#         m += n

#         for k in range(m, m+n+1):
            
#             if k <= i_rangE:
#                 rF[k] = n

#     return rF


# @njit
# def Statistics_111C(cv_N, cv_R, nof_reg111, reg_sz111):
#     col, area = 0, 0

#     for reg in range(nof_reg111):

#         for n in range(reg_sz111):
#             add = n + reg * reg_sz111

#             if cv_R[add] > 0:
#                 col += 1
#                 nam = cv_R[add]

#                 while nam > 0:
#                     area += 1
#                     nam = cv_N[nam, reg]

#     return col, area


def calc_num_blocks(array_size, threads_per_block):
    arg1_type = type(array_size)
    arg2_type = type(threads_per_block)

    if arg1_type == arg2_type == int:
        return (array_size + (threads_per_block - 1)) // threads_per_block

    elif arg1_type == arg2_type == tuple and len(array_size) == len(threads_per_block):
        num_blocks = [0] * len(array_size)

        for i, (size, threads) in enumerate(zip(array_size, threads_per_block)):
            num_blocks[i] = (size + (threads - 1)) // threads
        return tuple(num_blocks)

    else:
        print('Arguments should be either of type int or type tuple with the same shape.')


@cuda.jit
def gpu_write(cv_R, cv_L, cv_N, cv_T, cv_A, ft,
              nof_reg111, reg_sz111, nam):

    reg = cuda.grid(1)

    if reg < nof_reg111:
        add = ft[reg] + reg * reg_sz111
        p = cv_T[add]
        cv_A_reg = cv_A + reg
        cv_N[cv_A_reg] = nam
        cv_T[add] = cv_A_reg

        if p == 0:
            cv_R[add] = cv_A_reg
        else:
            cv_L[p] = cv_A_reg


@njit(parallel=True)
def cpu_write(cv_R, cv_L, cv_N, cv_T, cv_A, ft,
              nof_reg111, reg_sz111, nam):

    for reg in prange(nof_reg111):
        add = ft[reg] + reg * reg_sz111
        p = cv_T[add]
        cv_A_reg = cv_A + reg
        cv_N[cv_A_reg] = nam
        cv_T[add] = cv_A_reg

        if p == 0:
            cv_R[add] = cv_A_reg
        else:
            cv_L[p] = cv_A_reg

    return cv_R, cv_L, cv_N, cv_T


#                                                   New step 4
def T_Write_Vector_111(cv_R, cv_L, cv_N, cv_T, cv_A, ft,
                       nof_reg111, reg_sz111, nam,
                       use_gpu, gpu_settings):

    if use_gpu:
        gpu_write[gpu_settings[0]](cv_R, cv_L, cv_N, cv_T, cv_A, ft,
                                   nof_reg111, reg_sz111, nam)
    else:
        cv_R, cv_L, cv_N, cv_T = cpu_write(cv_R, cv_L, cv_N, cv_T, cv_A, ft,
                                           nof_reg111, reg_sz111, nam)

    cv_A += nof_reg111

    return cv_R, cv_L, cv_N, cv_T, cv_A


@cuda.jit
def gpu_read(cv_R, cv_L, cv_N, ft, sc_map,
             nof_reg111, reg_sz111, radius):

    reg, rad = cuda.grid(2)

    if reg < nof_reg111 and rad <= radius * 2:

        if reg < 10 or (reg >= 10 and rad - radius == 0):            # additional condition
            add = ft[reg] + rad - radius

            if 0 <= add < reg_sz111:
                add += reg * reg_sz111
                p = cv_R[add]

                while p > 0:
                    nam = cv_N[p]
                    sc_map[nam, reg] = 1
                    p = cv_L[p]


@cuda.jit
def gpu_sum(sc_map, sc, nof_cla111, nof_reg111):
    cla = cuda.grid(1)

    if cla <= nof_cla111:

        for reg in range(nof_reg111):

            if sc_map[cla, reg]:
                sc[cla] += 1
                sc_map[cla, reg] = 0


@njit(parallel=True)
def cpu_read(cv_R, cv_L, cv_N, ft, sc_map, sc,
             nof_cla111, nof_reg111, reg_sz111, radius):

    for reg in prange(nof_reg111):

        for rad in range(-radius, radius+1):

            if reg < 10 or (reg >= 10 and rad == 0):            # additional condition
                add = ft[reg] + rad

                if 0 <= add < reg_sz111:
                    add += reg * reg_sz111
                    p = cv_R[add]

                    while p > 0:
                        nam = cv_N[p]
                        sc_map[nam, reg] = 1
                        p = cv_L[p]

    for cla in prange(nof_cla111+1):
        sc[cla] = np.sum(sc_map[cla])

    return sc


#                                                   New step 3
def Read_Vector_Win_111(cv_R, cv_L, cv_N, ft,
                        nof_cla111, nof_reg111, reg_sz111,
                        threshold, radius,
                        use_gpu, gpu_settings):

    sc_map = np.zeros((nof_cla111+1, nof_reg111), dtype='uint8')
    sc = np.zeros(nof_cla111+1, dtype='int64')
    mx, win1 = 0, 0

    if use_gpu:
        sc_map = cuda.to_device(sc_map)
        sc = cuda.to_device(sc)

        gpu_read[gpu_settings[1]](cv_R, cv_L, cv_N, ft, sc_map,
                                  nof_reg111, reg_sz111, radius)

        gpu_sum[gpu_settings[2]](sc_map, sc, nof_cla111, nof_reg111)

        sc = sc.copy_to_host()
    else:
        sc = cpu_read(cv_R, cv_L, cv_N, ft, sc_map, sc,
                      nof_cla111, nof_reg111, reg_sz111, radius)

    win1 = np.argmax(sc)
    mx = sc[win1]

    if mx < threshold:
        win1 = 0

    return mx, win1


#                                                   New step 2
def VectorCortex111(cv_R, cv_L, cv_N, cv_T, cv_A, classes, labels, ft,
                    nof_cla111, nof_reg111, reg_sz111, threshold,
                    radius, N_111, vec, freeze, use_gpu, gpu_settings,
                    right_guesses, wrong_guesses, unrec_vectors):

    if freeze:
        mx, win = Read_Vector_Win_111(cv_R, cv_L, cv_N, ft,
                                      nof_cla111, nof_reg111, reg_sz111,
                                      threshold, radius,
                                      use_gpu, gpu_settings)

        if win > 0:

            if classes[win] == labels[vec]:
                right_guesses += 1
            else:
                wrong_guesses += 1
        else:
            unrec_vectors += 1
    else:
        N_111 += 1
        classes[N_111] = labels[vec]

        cv_R, cv_L, cv_N, cv_T, cv_A = T_Write_Vector_111(cv_R, cv_L, cv_N, cv_T, cv_A, ft,
                                                          nof_reg111, reg_sz111, N_111,
                                                          use_gpu, gpu_settings)

    return (cv_R, cv_L, cv_N, cv_T, cv_A, classes, N_111,
            right_guesses, wrong_guesses, unrec_vectors)


@cuda.jit
def gpu_write_vectors(cv_R, cv_L, cv_N, cv_T, cv_A, features,
                      nof_vectors, nof_reg111, reg_sz111, N_111,
                      gpu_settings):

    blk, reg = cuda.grid(2)
    vec_per_block = gpu_settings[3][1][0]

    for vec in range(vec_per_block * blk, vec_per_block * (blk+1)):

        if vec < nof_vectors and reg < nof_reg111:
            add = features[vec, reg] + reg * reg_sz111
            p = cv_T[blk, add]
            cv_A_vec_reg = cv_A + nof_reg111 * (vec % vec_per_block) + reg
            cv_N[blk, cv_A_vec_reg] = N_111 + vec
            cv_T[blk, add] = cv_A_vec_reg

            if p == 0:
                cv_R[blk, add] = cv_A_vec_reg
            else:
                cv_L[blk, p] = cv_A_vec_reg


@cuda.jit
def gpu_read_vec(cv_R, cv_L, cv_N, ft, sc_map,
                 nof_reg111, reg_sz111, radius,
                 gpu_settings):

    blk, reg, rad = cuda.grid(3)
    blocks = gpu_settings[3][0][0]

    if blk < blocks and reg < nof_reg111 and rad <= radius * 2:

        if reg < 10 or (reg >= 10 and rad - radius == 0):            # additional condition
            add = ft[reg] + rad - radius

            if 0 <= add < reg_sz111:
                add += reg * reg_sz111
                p = cv_R[blk, add]

                while p > 0:
                    nam = cv_N[blk, p]
                    sc_map[nam, reg] = 1
                    p = cv_L[blk, p]


def gpu_read_vectors(cv_R, cv_L, cv_N, classes, labels, features,
                     nof_vectors, nof_cla111, nof_reg111, reg_sz111,
                     threshold, radius, div, gpu_settings,
                     right_guesses, wrong_guesses, unrec_vectors):

    sc_map = np.zeros((nof_cla111+1, nof_reg111), dtype='uint8')
    sc_map = cuda.to_device(sc_map)

    print('Vectors processed:')

    for vec in range(nof_vectors):
        sc = np.zeros(nof_cla111+1, dtype='int64')
        ft = features[vec]

        sc = cuda.to_device(sc)

        gpu_read_vec[gpu_settings[4]](cv_R, cv_L, cv_N, ft, sc_map,
                                      nof_reg111, reg_sz111, radius,
                                      gpu_settings)

        gpu_sum[gpu_settings[2]](sc_map, sc, nof_cla111, nof_reg111)

        sc = sc.copy_to_host()

        win = np.argmax(sc)
        mx = sc[win]

        if mx < threshold:
            win = 0

        if win > 0:

            if classes[win] == labels[vec]:
                right_guesses += 1
            else:
                wrong_guesses += 1
        else:
            unrec_vectors += 1

        if (vec+1) % div == 0 or vec+1 == nof_vectors:
            print(vec+1, '/', nof_vectors)

    return right_guesses, wrong_guesses, unrec_vectors


@njit(parallel=True)
def cpu_read_vectors(cv_R, cv_L, cv_N, classes, labels, features,
                     nof_vectors, nof_cla111, nof_reg111, reg_sz111,
                     threshold, radius, div, gpu_settings,
                     right_guesses, wrong_guesses, unrec_vectors):

    blocks = gpu_settings[3][0][0]

    print('Vectors processed:')

    for vec in range(nof_vectors):
        sc_map = np.zeros((nof_cla111+1, nof_reg111), dtype='uint8')
        sc = np.zeros(nof_cla111+1, dtype='int64')

        for blk in prange(blocks):

            for reg in range(nof_reg111):

                for rad in range(-radius, radius+1):

                    if reg < 10 or (reg >= 10 and rad == 0):            # additional condition
                        add = features[vec, reg] + rad

                        if 0 <= add < reg_sz111:
                            add += reg * reg_sz111
                            p = cv_R[blk, add]

                            while p > 0:
                                nam = cv_N[blk, p]
                                sc_map[nam, reg] = 1
                                p = cv_L[blk, p]

        for cla in prange(nof_cla111+1):
            sc[cla] = np.sum(sc_map[cla])

        win = np.argmax(sc)
        mx = sc[win]

        if mx < threshold:
            win = 0

        if win > 0:

            if classes[win] == labels[vec]:
                right_guesses += 1
            else:
                wrong_guesses += 1
        else:
            unrec_vectors += 1

        if (vec+1) % div == 0 or vec+1 == nof_vectors:
            print(vec+1, '/', nof_vectors)

    return right_guesses, wrong_guesses, unrec_vectors


def gpu_cortex(cv_R, cv_L, cv_N, cv_T, cv_A, classes, labels, features,
               nof_vectors, nof_cla111, nof_reg111, reg_sz111, N_111,
               threshold, radius, div, gpu_settings):

    sc_map = np.zeros((nof_cla111+1, nof_reg111), dtype='uint8')
    sc_map = cuda.to_device(sc_map)

    print('Vectors processed:')

    for vec in range(nof_vectors):
        sc = np.zeros(nof_cla111+1, dtype='int64')
        ft = features[vec]

        sc = cuda.to_device(sc)

        gpu_read[gpu_settings[1]](cv_R, cv_L, cv_N, ft, sc_map,
                                  nof_reg111, reg_sz111, radius)

        gpu_sum[gpu_settings[2]](sc_map, sc, nof_cla111, nof_reg111)

        sc = sc.copy_to_host()

        win = np.argmax(sc)
        mx = sc[win]

        if mx < threshold:
            win = 0

        if win == 0:
            N_111 += 1
            classes[N_111] = labels[vec]

            gpu_write[gpu_settings[0]](cv_R, cv_L, cv_N, cv_T, cv_A, ft,
                                       nof_reg111, reg_sz111, N_111)

            cv_A += nof_reg111

        if (vec+1) % div == 0 or vec+1 == nof_vectors:
            print(vec+1, '/', nof_vectors)

        if N_111 >= nof_cla111:
            print(vec+1, '/', nof_vectors)
            break

    return cv_R, cv_L, cv_N, cv_T, cv_A, classes, N_111


#                                                   Step 1
# @njit#(parallel=True)
def cortex(cv_R, cv_L, cv_N, cv_T, cv_A, classes, features, labels,
           nof_cla111, nof_reg111, reg_sz111, pv111_reT, threshold,
           radius, hamming_dist_109, freeze, regularize,
           use_gpu, gpu_settings,
           highest_acc=0, highest_acc_rad=0, highest_acc_hd=0):

    right_guesses, wrong_guesses, unrec_vectors = 0, 0, 0
    N_111 = 0

    nof_vectors = features.shape[0]
    div = 10**(len(str(nof_vectors))-2)

    # lngtH = 0
    # sc = np.zeros(nof_cla111+1, dtype='int64')
    # lstT = np.zeros(nof_cla111, dtype='int64')

    winners = np.zeros(nof_vectors, dtype='int16')
    class_vectors = np.zeros((nof_cla111, nof_reg111), dtype='int64')

    # rF = FeatureRegularizer(reg_sz111)

    if use_gpu and not freeze:
        N_111 += 1

        gpu_write_vectors[gpu_settings[3]](cv_R, cv_L, cv_N, cv_T, cv_A, features,
                                           nof_vectors, nof_reg111, reg_sz111, N_111,
                                           gpu_settings)

        classes[1:] = labels[:]
        cv_A += nof_reg111 * nof_vectors

        # cv_R, cv_L, cv_N, cv_T, cv_A, classes, N_111 = gpu_cortex(cv_R, cv_L, cv_N, cv_T, cv_A, classes, labels, features,
        #                                                           nof_vectors, nof_cla111, nof_reg111, reg_sz111, N_111,
        #                                                           threshold, radius, div, gpu_settings)
    elif use_gpu and freeze:
        right_guesses, wrong_guesses, unrec_vectors = gpu_read_vectors(cv_R, cv_L, cv_N, classes, labels, features,
                                                                       nof_vectors, nof_cla111, nof_reg111, reg_sz111,
                                                                       threshold, radius, div, gpu_settings,
                                                                       right_guesses, wrong_guesses, unrec_vectors)
    elif not use_gpu and freeze:
        right_guesses, wrong_guesses, unrec_vectors = cpu_read_vectors(cv_R, cv_L, cv_N, classes, labels, features,
                                                                       nof_vectors, nof_cla111, nof_reg111, reg_sz111,
                                                                       threshold, radius, div, gpu_settings,
                                                                       right_guesses, wrong_guesses, unrec_vectors)
    else:
        print('Vectors processed:')
        for vec in range(nof_vectors):
            ft = features[vec]

            # if regularize:
            #     for i in range(nof_reg111):
            #         ft[i] = rF[ft[i]]

            #    --> step 2
            (cv_R, cv_L, cv_N, cv_T, cv_A, classes, N_111,
             right_guesses, wrong_guesses, unrec_vectors) = VectorCortex111(cv_R, cv_L, cv_N, cv_T, cv_A, classes, labels, ft,
                                                                            nof_cla111, nof_reg111, reg_sz111, threshold,
                                                                            radius, N_111, vec, freeze, use_gpu, gpu_settings,
                                                                            right_guesses, wrong_guesses, unrec_vectors)

            # (cv_R, cv_L, cv_N, cv_T, cv_A, classes,
            #  sc, lstT, lngtH, N_111, win,
            #  right_guesses, wrong_guesses, unrec_vectors,
            #  class_vectors) = VectorCortex109(cv_R, cv_L, cv_N, cv_T, cv_A, classes, labels,
            #                                   reg_sz111, ft, nof_reg111, radius, hamming_dist_109, freeze,
            #                                   sc, lstT, lngtH, N_111, vec,
            #                                   right_guesses, wrong_guesses, unrec_vectors,
            #                                   class_vectors)

            # winners[vec] = win

            if (vec+1) % div == 0 or vec+1 == nof_vectors:
                print(vec+1, '/', nof_vectors)

            if N_111 >= nof_cla111:
                print(vec+1, '/', nof_vectors)
                break
    print()

    # class_vectors = class_vectors[:N_111]

    # col, area = Statistics_111C(cv_N, cv_R, nof_reg111, reg_sz111)
    # av_col_height = area / col

    # print('columns:', col,
    #       '  area:', area,
    #       '  average column height:', av_col_height)

    print('classes:', np.count_nonzero(classes))

    if freeze:
        acc = right_guesses / nof_vectors * 100
        error = wrong_guesses / nof_vectors * 100
        unrec = unrec_vectors / nof_vectors * 100

        if highest_acc < acc:
            highest_acc = acc
            highest_acc_rad = radius
            highest_acc_hd = hamming_dist_109

        print('right guesses:', right_guesses,
              '  wrong guesses:', wrong_guesses,
              '  unrec vectors:', unrec_vectors)

        print('accuracy:', acc, '%',
              '  error:', error, '%',
              '  unrecognized:', unrec, '%')
    print()

    return (cv_R, cv_L, cv_N, cv_T, cv_A, classes,
            winners, class_vectors,
            highest_acc, highest_acc_rad, highest_acc_hd)


# #                                                   Divides classes into two groups
# def divide_classes(winners, class_vectors):
#     numbers = np.unique(winners)
#     skipped_indices = []
#     is_success = False

#     for v1, vector1 in enumerate(class_vectors):

#         if v1 in skipped_indices:
#             continue

#         euclid_distances = {}
#         skipped_indices.append(v1)

#         for v2, vector2 in enumerate(class_vectors):

#             if v2 not in skipped_indices:
#                 euclid_dist = np.linalg.norm(vector1 - vector2)
#                 euclid_distances[v2] = euclid_dist

#         if euclid_distances:
#             shortest_dist_index = min(euclid_distances)
#             winners[winners == numbers[shortest_dist_index]] = numbers[v1]
#             skipped_indices.append(shortest_dist_index)

#     numbers, quantity = np.unique(winners, return_counts=True)

#     if numbers.shape[0] == 2:
#         winners[winners == numbers[np.argmax(quantity)]] = 255
#         winners[winners == numbers[np.argmin(quantity)]] = 0
#         is_success = True

#         print('number of classes = 2 (success)\n')
#         return winners, is_success
#     else:
#         print('number of classes =', numbers.shape[0], '(fail)\n')
#         return winners, is_success


# #                                                   Test of parameters
# def parameters_test(nof_cla111, nof_reg111, reg_sz111, pv111_reT, threshold, use_gpu, gpu_settings,
#                     rad=1, hd=0, freeze=True, regularize=False, divide=False):

#     highest_acc, highest_acc_rad, highest_acc_hd = 0, 0, 0

#     rad_array = create_percent_array(reg_sz111, step=5, mmin=0, mmax=100)
#     # hd_array = create_percent_array(nof_reg111, step=5, mmin=0, mmax=100)

#     for rad in rad_array:
#     # for hd in hd_array:

#         # vgg16
#         # features, labels = np.load('feature_maps/cifar10_vgg16/train_fm_cifar10_vgg16.npy'), np.load('feature_maps/cifar10_vgg16/train_labels_cifar10.npy')        # training set
#         # features, labels = np.load('feature_maps/fer2013_vgg16/train_fm_fer2013_vgg16.npy'), np.load('feature_maps/fer2013_vgg16/train_labels_fer2013.npy')        # training set

#         # image array
#         # features, labels = np.load('feature_maps/image_array_600x800x3_flatten.npy'), np.ones(features.shape[0], dtype=np.uint8)

#         # gene expression
#         # features, labels = np.load('feature_maps/gene_expression/genes801x20531.npy')[::2], np.load('feature_maps/gene_expression/labels801.npy')[::2]                     # training set (odd)
#         features, labels = np.load('feature_maps/gene_expression/genes801x20531.npy')[1::2], np.load('feature_maps/gene_expression/labels801.npy')[1::2]                   # test set (even)

#         cv_N = np.load('saved_progress/cv_N.npy')
#         cv_R = np.load('saved_progress/cv_R.npy')
#         cv_T = np.load('saved_progress/cv_T.npy')
#         classes = np.load('saved_progress/classes.npy')

#         #    --> step 1
#         (_, _, _, _, _, _,
#          winners, class_vectors,
#          highest_acc, highest_acc_rad, highest_acc_hd) = cortex(cv_R, cv_L, cv_N, cv_T, cv_A, classes, features, labels,
#                                                                 nof_cla111, nof_reg111, reg_sz111, pv111_reT, threshold,
#                                                                 rad, rad*2, hd, freeze, regularize, use_gpu, gpu_settings,
#                                                                 highest_acc=highest_acc,
#                                                                 highest_acc_rad=highest_acc_rad,
#                                                                 highest_acc_hd=highest_acc_hd)

#         # if divide:
#         #     winners, success = divide_classes(winners, class_vectors)
            
#         #     if success:
#         #         return highest_acc, highest_acc_rad, highest_acc_hd, winners

#     print('highest accuracy:', highest_acc,
#           '  radius of highest accuracy:', highest_acc_rad,
#           '  hamming distance of highest accuracy:', highest_acc_hd, '\n')

#     return highest_acc, highest_acc_rad, highest_acc_hd, winners




# cortex level 1                                    Step 0

# fer2013 vgg16
# features, labels = np.load('feature_maps/fer2013_vgg16/train_fm_fer2013_vgg16.npy'), np.load('feature_maps/fer2013_vgg16/train_labels_fer2013.npy')              # training set
# features, labels = np.load('feature_maps/fer2013_vgg16/validation_fm_fer2013_vgg16.npy'), np.load('feature_maps/fer2013_vgg16/validation_labels_fer2013.npy')    # validation set
# features, labels = np.load('feature_maps/fer2013_vgg16/test_fm_fer2013_vgg16.npy'), np.load('feature_maps/fer2013_vgg16/test_labels_fer2013.npy')                # test set

# cifar10 vgg16
# features, labels = np.load('feature_maps/cifar10_vgg16/train_fm_cifar10_vgg16.npy'), np.load('feature_maps/cifar10_vgg16/train_labels_cifar10.npy')              # training set
# features, labels = np.load('feature_maps/cifar10_vgg16/test_fm_cifar10_vgg16.npy'), np.load('feature_maps/cifar10_vgg16/test_labels_cifar10.npy')                # test set

# image array
# features, labels = np.load('feature_maps/image_array_600x800x3_flatten.npy'), np.ones(features.shape[0], dtype='uint8')

# object array
# features, labels = np.load('object_signatures/eraser_signature.npy'), np.ones(features.shape[0], dtype='uint8')

# gene expression
# features, labels = np.load('feature_maps/gene_expression/genes801x20531.npy')[::2], np.load('feature_maps/gene_expression/labels801.npy')[::2]                     # training set (odd)
# features, labels = np.load('feature_maps/gene_expression/genes801x20531.npy')[1::2], np.load('feature_maps/gene_expression/labels801.npy')[1::2]                   # test set (even)

# covertype
features, labels = np.load('feature_maps/covertype/covertype581012x12.npy')[:11340], np.load('feature_maps/covertype/labels581012.npy')[:11340]                     # training set
# features, labels = np.load('feature_maps/covertype/covertype581012x12.npy')[11340:11340+3780], np.load('feature_maps/covertype/labels581012.npy')[11340:11340+3780]                   # test set

freeze, load_progress, save_progress = False, False, True     # train and save
# freeze, load_progress, save_progress = True, True, False      # load and test
regularize = False
test_params = False
use_gpu = True

radius = 6                                                    # 0 .. reg_sz111  85 / 6
hamming_dist_109 = 0                                          # 0 .. nof_reg111  0

nof_cla111 = 11340                                            # N max classes  801 / 290506
nof_reg111 = 12                                               # K vector size  20531 / 12
reg_sz111 = 256                                               # X value range  256

print('max classes:', nof_cla111,
      '  vector size:', nof_reg111,
      '  value range:', reg_sz111)

print('radius:', radius,
      '  hamming distance:', hamming_dist_109, '\n')

threads_per_block = (32,                                      # 128..512 (*32)  128 / 32
                     (4, 8),                                  # (32, 8)=256 / (4, 8)=32
                     128,                                     # 32 / 128
                     (128, 2),                                # - / (128, 2)=256
                     (32, 2, 4))                              # - / (32, 2, 4)=256

blocks_per_grid = (calc_num_blocks(nof_reg111, threads_per_block[0]),
                   calc_num_blocks((nof_reg111, radius*2+1), threads_per_block[1]),
                   calc_num_blocks(nof_cla111, threads_per_block[2]),
                   calc_num_blocks((nof_cla111, nof_reg111), threads_per_block[3]),
                   calc_num_blocks((nof_cla111, nof_reg111, radius*2+1), threads_per_block[4]))

gpu_settings = [[]] * len(threads_per_block)

for i, _ in enumerate(threads_per_block):
    gpu_settings[i] = (blocks_per_grid[i], threads_per_block[i])

gpu_settings = tuple(gpu_settings)

blocks = gpu_settings[3][0][0]
threshold = int(nof_reg111 * 0.75)
pv111_reT = reg_sz111 * nof_reg111
pv111_linkS = int(pv111_reT * (nof_cla111 / (reg_sz111 * (blocks-1))))       # pv111_reT * 5

if load_progress:
    cv_A = np.load('saved_progress/cv_A.npy')
    cv_R = np.load('saved_progress/cv_R.npy')
    cv_L = np.load('saved_progress/cv_L.npy')
    cv_N = np.load('saved_progress/cv_N.npy')
    cv_T = np.load('saved_progress/cv_T.npy')
    classes = np.load('saved_progress/classes.npy')
    print('(Progress 1 Loaded)\n')
else:
    cv_A = 1
    cv_R = np.zeros((blocks, pv111_reT+1), dtype='int64')
    cv_L = np.zeros((blocks, pv111_linkS+1), dtype='int64')
    cv_N = np.zeros((blocks, pv111_linkS+1), dtype='int64')
    cv_T = np.zeros((blocks, pv111_reT+1), dtype='int64')
    classes = np.zeros(nof_cla111+1, dtype='int64')

if use_gpu:
    features = np.ascontiguousarray(features)
    features = cuda.to_device(features)
    cv_R = cuda.to_device(cv_R)
    cv_L = cuda.to_device(cv_L)
    cv_N = cuda.to_device(cv_N)
    cv_T = cuda.to_device(cv_T)

# if test_params:
# #    --> test of parameters
#     highest_acc, highest_acc_rad, highest_acc_hd, _ = parameters_test(nof_cla111, nof_reg111, reg_sz111, pv111_reT, threshold, use_gpu, gpu_settings)
# else:
#    --> step 1
%time cv_R, cv_L, cv_N, cv_T, cv_A, classes, _, _, highest_acc, highest_acc_rad, highest_acc_hd = cortex(cv_R, cv_L, cv_N, cv_T, cv_A, classes, features, labels, nof_cla111, nof_reg111, reg_sz111, pv111_reT, threshold, radius, hamming_dist_109, freeze, regularize, use_gpu, gpu_settings)

if save_progress and not test_params:

    if use_gpu:
        cv_R = cv_R.copy_to_host()
        cv_L = cv_L.copy_to_host()
        cv_N = cv_N.copy_to_host()
        cv_T = cv_T.copy_to_host()

    np.save('saved_progress/cv_A.npy', cv_A)
    np.save('saved_progress/cv_R.npy', cv_R)
    np.save('saved_progress/cv_L.npy', cv_L)
    np.save('saved_progress/cv_N.npy', cv_N)
    np.save('saved_progress/cv_T.npy', cv_T)
    np.save('saved_progress/classes.npy', classes)
    # np.save('saved_progress/winners.npy', winners)
    print('\n(Progress 1 Saved)\n')




# # cortex level 2
# signature = 0
# N_115_maX = 10
# nof_reg115 = 10000
# reg_sz115 = 1
# T1 = 32

# load_progress2 = False
# save_progress2 = True

# SHM = np.zeros((N_115_maX, nof_reg115), dtype='int64')
# cv_N2 = np.zeros((N_115_maX+1, nof_reg115), dtype='int64')
# root_A2 = np.zeros((reg_sz115, nof_reg115), dtype='int64')
# root_T2 = np.zeros((reg_sz115, nof_reg115), dtype='int64')
# L_115 = np.zeros(N_115_maX, dtype='int64')

# if load_progress2:
#     SHM = np.load('saved_progress/SHM.npy')
#     cv_N2 = np.load('saved_progress/cv_N2.npy')
#     root_A2 = np.load('saved_progress/root_A2.npy')
#     root_T2 = np.load('saved_progress/root_T2.npy')
#     print('(Progress 2 Loaded)\n')

# ft2, lth2, SHM = signature_histogram(winners, SHM, signature, T1, learn=True)
# L_115[signature] = lth2

# lstT2, win, LLL, SSS, TTT = Read_Set_115(cv_N2, root_A2, L_115,
#                                          ft2, lth2, N_115_maX, reg_sz115)
# print(win, LLL, SSS, TTT, lstT2)

# if win == 0:
#     win = 1

# cv_N2, root_A2, root_T2 = Write_Set_115(cv_N2, root_A2, root_T2,
#                                         ft2, lth2, win)

# lstT2, win, LLL, SSS, TTT = Read_Set_115(cv_N2, root_A2, L_115,
#                                          ft2, lth2, N_115_maX, reg_sz115)
# print(win, LLL, SSS, TTT, lstT2)

# if save_progress2:
#     np.save('saved_progress/SHM.npy', SHM)
#     np.save('saved_progress/cv_N2.npy', cv_N2)
#     np.save('saved_progress/root_A2.npy', root_A2)
#     np.save('saved_progress/root_T2.npy', root_T2)
#     print('(Progress 2 Saved)\n')

#%%=====================================================================================================================================================
# Simple cortex

from numba import njit, prange
import numpy as np


@njit(parallel=True)
def simple_cortex(feature_vecs, labels, memory, classes, radius, learn):

    if learn:
        class_counter = np.count_nonzero(classes)

        for vec in prange(feature_vecs.shape[0]):

            for ft in prange(feature_vecs.shape[1]):
                memory[ft, feature_vecs[vec, ft], class_counter + vec] = 1
                classes[class_counter + vec] = labels[vec]

        print('number of vectors:', feature_vecs.shape[0])
    else:
        right_guesses, wrong_guesses, unrec_vectors = 0, 0, 0

        for vec in prange(feature_vecs.shape[0]):
            score = np.zeros(classes.shape[0], dtype='int64')
            mx, win = 0, -1

            for ft in prange(feature_vecs.shape[1]):

                for rad in prange(-radius, radius+1):
                    add = feature_vecs[vec, ft] + rad

                    if add < 0 or add >= memory.shape[1]:
                        continue

                    for cla in prange(memory.shape[2]):

                        if memory[ft, add, cla]:
                            score[cla] += 1

            win = np.argmax(score)

            if win != -1:

                if classes[win] == labels[vec]:
                    right_guesses += 1
                else:
                    wrong_guesses += 1
            else:
                unrec_vectors += 1

        num_vectors = feature_vecs.shape[0]
        acc = right_guesses / num_vectors * 100
        error = wrong_guesses / num_vectors * 100
        unrec = unrec_vectors / num_vectors * 100

        print('number of vectors:', num_vectors,
              '  radius:', radius)

        print('right guesses:', right_guesses,
              '  wrong guesses:', wrong_guesses,
              '  unrec vectors:', unrec_vectors)
        
        print('accuracy:', acc, '%',
              '  error:', error, '%',
              '  unrecognized:', unrec, '%')

    return memory, classes


feature_vecs1, labels1 = np.load('feature_maps/gene_expression/genes801x20531.npy')[::2], np.load('feature_maps/gene_expression/labels801.npy')[::2]
feature_vecs2, labels2 = np.load('feature_maps/gene_expression/genes801x20531.npy')[1::2], np.load('feature_maps/gene_expression/labels801.npy')[1::2]

K_groups = 20531
X_colons = 256
N_classes = 801

radius = 85

# learn, load, save = True, False, True
# learn, load, save = False, True, False

memory = np.zeros((K_groups, X_colons, N_classes), dtype='uint8')
classes = np.zeros(N_classes, dtype='int64')

# if load:
#     memory = np.load('memory.npy')
#     classes = np.load('classes.npy')

print('(Learning...)')
%time memory, classes = simple_cortex(feature_vecs1, labels1, memory, classes, radius, learn=True)
# print(memory.shape, memory.dtype, memory)
# print(classes.shape, classes.dtype, classes)
print('(Testing...)')
%time memory, classes = simple_cortex(feature_vecs2, labels2, memory, classes, radius, learn=False)

# if save:
#     np.save('memory.npy', memory)
#     np.save('classes.npy', classes)

#%%=====================================================================================================================================================
# Gene expression data preparation for NN

from tensorflow.keras.utils import to_categorical
import pandas as pd
import numpy as np


dataset = pd.read_csv('datasets/gene_expression_cancer_RNA-seq_801x20531/data.csv', index_col=0)
labels = pd.read_csv('datasets/gene_expression_cancer_RNA-seq_801x20531/labels.csv', index_col=0)

classes = {'PRAD':0, 'LUAD':1, 'BRCA':2, 'KIRC':3, 'COAD':4}


norm_dataset = (dataset - dataset.min()) / (dataset.max() - dataset.min())
norm_dataset = norm_dataset.fillna(0)
norm_dataset = norm_dataset.astype('float32')
np_norm_dataset = norm_dataset.to_numpy()
# np_norm_dataset = np.around(np_norm_dataset, 3)


np_labels = np.squeeze(labels.to_numpy())

for cclass in classes:
    np_labels[np_labels == cclass] = classes[cclass]

np_labels = to_categorical(np_labels)
np_labels = np_labels.astype('uint8')


train_features = np_norm_dataset[::2]
test_features = np_norm_dataset[1::2]

train_labels = np_labels[::2]
test_labels = np_labels[1::2]

print(train_features.shape, train_features.dtype, test_features.shape, test_features.dtype)
print(train_labels.shape, train_labels.dtype, test_labels.shape, test_labels.dtype)

#%%=====================================================================================================================================================
%%time
# Gene expression NN

from tensorflow.keras import models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf


# with tf.device('cpu:0'):
model = models.Sequential()
model.add(Dense(256, activation='relu', input_shape=(20531,)))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(5, activation='softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers.Adam(),
    metrics=['accuracy'])

checkpoint = ModelCheckpoint(
    'gene_expression_model.h5',
    monitor='val_loss',
    verbose=1,
    save_best_only=True)

history = model.fit(
    train_features,
    train_labels,
    validation_data=(test_features, test_labels),
    batch_size=64,
    epochs=20,
    callbacks=[checkpoint])

print(model.summary())

model = models.load_model('gene_expression_model.h5')

test_loss, test_acc = model.evaluate(test_features, test_labels, batch_size=64)
print('test acc:', test_acc)
print('test loss:', test_loss)

#%%=====================================================================================================================================================
# Вектора признаков

#%%=====================================================================================================================================================
# Создание изображения из обработанного вектора признаков

# %matplotlib inline
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import numpy as np


image_array = np.load('saved_progress/output.npy')
image_array = image_array.reshape(600, 800)

numbers, quantity = np.unique(image_array, return_counts=True)
# threshold = (np.max(quantity) - np.amin(quantity)) / 3

# for i, q in enumerate(quantity):

#     if q > threshold:
#         image_array[image_array == numbers[i]] = 255
#     else:
#         image_array[image_array == numbers[i]] = 0

# print(image_array.shape, image_array.dtype, image_array)

plt.bar(numbers, quantity)
plt.xticks(numbers)
plt.title('R = 35%')
plt.xlabel('Классы')
plt.ylabel('Мощность')
# plt.savefig('bar.png', bbox_inches='tight')
plt.figure()

plt.axis('off')
plt.imshow(image_array, cmap='gray')
# plt.savefig('sign.png', bbox_inches='tight', pad_inches=0)
plt.show()

img = image.load_img('SVA_sign_dark.jpeg', target_size = (600, 800))
image_array = image.img_to_array(img) / 255

plt.axis('off')
plt.imshow(image_array)
# plt.savefig('sign0.png', bbox_inches='tight', pad_inches=0)
plt.show()

#%%=====================================================================================================================================================
# Перевод изображения в вектор признаков

# %matplotlib inline
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import numpy as np


img = image.load_img('SVA_sign_dark.jpeg', target_size = (600, 800)) #, color_mode = 'grayscale')
image_array = image.img_to_array(img)
image_array = image_array.astype(np.uint8)
image_array = image_array.reshape(-1, image_array.shape[-1])
# image_array = np.tile(image_array, 3)

# np.save('image_array_600x800x3_flatten.npy', image_array)

print(image_array.shape, image_array.dtype, image_array)

# plt.imshow(image_array)
# plt.show()

#%%=====================================================================================================================================================
# Нейронные сети

#%%=====================================================================================================================================================
# CIFAR-10

# Настройка конфигурации GPU

from tensorflow.compat.v1 import ConfigProto, InteractiveSession


config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)




# Перевод данных из датасета в массивы

from tensorflow.keras.datasets import cifar10
import numpy as np


# 0=airplane, 1=automobile, 2=bird, 3=cat, 4=deer, 5=dog, 6=frog, 7=horse, 8=ship, 9=truck
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

print(train_images.shape, train_images.dtype, test_images.shape, test_images.dtype)
print(train_labels.shape, train_labels.dtype, test_labels.shape, test_labels.dtype)




# Подготовка данных к передаче в нейросеть

from tensorflow.keras.utils import to_categorical


train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# train_labels += 1
# test_labels += 1

train_labels = np.squeeze(train_labels)
test_labels = np.squeeze(test_labels)

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

train_labels = train_labels.astype(np.uint8)
test_labels = test_labels.astype(np.uint8)

# np.save('train_images_cifar10.npy', train_images)
# np.save('test_images_cifar10.npy', test_images)

# np.save('train_labels_cifar10.npy', train_labels)
# np.save('test_labels_cifar10.npy', test_labels)

print(train_images.shape, train_images.dtype, test_images.shape, test_images.dtype)
print(train_labels.shape, train_labels.dtype, test_labels.shape, test_labels.dtype)




# Создание генераторов

from tensorflow.keras.preprocessing.image import ImageDataGenerator


batch_size = 64

# train_gen = ImageDataGenerator(
#     rotation_range=40,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest')

train_gen = ImageDataGenerator()
test_gen = ImageDataGenerator()

train_gen = train_gen.flow(
    train_images,
    train_labels,
    batch_size = batch_size,
    shuffle=False)

test_gen = test_gen.flow(
    test_images,
    test_labels,
    batch_size = batch_size,
    shuffle=False)

#%%=====================================================================================================================================================
# CIFAR-10

# Пропуск всех изображений через свёрточную основу нейросети

from tensorflow.keras import models


conv_base = models.load_model('cifar10_conv_base_cnn.h5')

# conv_base = models.load_model('cifar10_conv_base_vgg16.h5')

train_fm = conv_base.predict_generator(train_gen)
test_fm = conv_base.predict_generator(test_gen)

train_fm *= 100
test_fm *= 100

# train_fm[train_fm > 1000] = 1000
# test_fm[test_fm > 1000] = 1000

train_fm = train_fm.astype('int16')
test_fm = test_fm.astype('int16')

# np.save('train_fm_cifar10_cnn.npy', train_fm)
# np.save('test_fm_cifar10_cnn.npy', test_fm)

# np.save('train_fm_cifar10_vgg16.npy', train_fm)
# np.save('test_fm_cifar10_vgg16.npy', test_fm)

print(np.max(train_fm), np.max(test_fm))
print(train_fm.shape, train_fm.dtype, train_fm)
print(test_fm.shape, test_fm.dtype, test_fm)

#%%=====================================================================================================================================================
# CIFAR-10

# CNN модель

from tensorflow.keras import models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout


nof_epochs = 30

def lr_schedule(epoch):
    lrate = 0.0005
    if epoch > 20:
        lrate = 0.0001
    return lrate

model = models.Sequential()
model.add(Conv2D(64, (3,3), activation='relu', padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.0005), metrics=['accuracy'])

checkpoint = ModelCheckpoint('cifar10_cnn_model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

history = model.fit_generator(
    train_gen,
    epochs=nof_epochs,
    validation_data=test_gen,
    callbacks=[checkpoint, LearningRateScheduler(lr_schedule)])

model = models.load_model('cifar10_cnn_model.h5')

test_loss, test_acc = model.evaluate_generator(test_gen)
print('test_acc:', test_acc)
print('test_loss:', test_loss)

#%%=====================================================================================================================================================
# CIFAR-10

# Замороженная VGG16

from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import VGG16


nof_epochs = 10

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

for layer in vgg16.layers:
    layer.trainable = False

model_fr = models.Sequential()
model_fr.add(vgg16)
model_fr.add(layers.Flatten())
model_fr.add(layers.Dropout(0.5))
model_fr.add(layers.Dense(256, activation='relu'))
model_fr.add(layers.Dense(10, activation='softmax'))

model_fr.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.0005), metrics=['accuracy'])

checkpoint = ModelCheckpoint('cifar10_cnn_vgg16_fr.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

history = model_fr.fit_generator(
    train_gen,
    epochs=nof_epochs,
    validation_data=test_gen,
    callbacks=[checkpoint])

model_fr = models.load_model('cifar10_cnn_vgg16_fr.h5')

test_loss, test_acc = model_fr.evaluate_generator(test_gen)
print('test_acc:', test_acc)
print('test_loss:', test_loss)

#%%=====================================================================================================================================================
# CIFAR-10

# Размороженная VGG16

nof_epochs = 10

set_trainable = False
for layer in vgg16.layers:
    if layer.name == 'block2_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

model_ufr = models.Sequential()
model_ufr.add(vgg16)
model_ufr.add(model_fr.layers[1])
model_ufr.add(model_fr.layers[2])
model_ufr.add(model_fr.layers[3])
model_ufr.add(model_fr.layers[4])

model_ufr.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.0001), metrics=['accuracy'])

checkpoint = ModelCheckpoint('cifar10_cnn_vgg16.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

history = model_ufr.fit_generator(
    train_gen,
    epochs=nof_epochs,
    validation_data=test_gen,
    callbacks=[checkpoint])

model_ufr = models.load_model('cifar10_cnn_vgg16.h5')

test_loss, test_acc = model_ufr.evaluate_generator(test_gen)
print('test_acc:', test_acc)
print('test_loss:', test_loss)

#%%=====================================================================================================================================================
#%%=====================================================================================================================================================
# FER2013

# Настройка конфигурации GPU

from tensorflow.compat.v1 import ConfigProto, InteractiveSession


config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)




# Перевод данных из датасета в массивы

import pandas as pd
import numpy as np


h, w = 48, 48

fer2013_ds = pd.read_csv('datasets/fer2013.csv')        # 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral

nof_train_samples = len(fer2013_ds[fer2013_ds['Usage'] == 'Training'])
nof_val_samples = len(fer2013_ds[fer2013_ds['Usage'] == 'PublicTest'])
nof_test_samples = len(fer2013_ds[fer2013_ds['Usage'] == 'PrivateTest'])

train_images = np.zeros((nof_train_samples, h, w))
validation_images = np.zeros((nof_val_samples, h, w))
test_images = np.zeros((nof_test_samples, h, w))

train_labels = np.zeros(nof_train_samples)
validation_labels = np.zeros(nof_val_samples)
test_labels = np.zeros(nof_test_samples)

i, j, k = 0, 0, 0
for index, row in fer2013_ds.iterrows():
    
    if row['Usage'] == 'Training':
        train_images[i] = np.fromstring(row['pixels'], dtype=np.uint8, sep=' ').reshape(h, w)
        train_labels[i] = row['emotion']
        i += 1

    elif row['Usage'] == 'PublicTest':
        validation_images[j] = np.fromstring(row['pixels'], dtype=np.uint8, sep=' ').reshape(h, w)
        validation_labels[j] = row['emotion']
        j += 1

    elif row['Usage'] == 'PrivateTest':
        test_images[k] = np.fromstring(row['pixels'], dtype=np.uint8, sep=' ').reshape(h, w)
        test_labels[k] = row['emotion']
        k += 1

print(train_images.shape, train_images.dtype, validation_images.shape, validation_images.dtype, test_images.shape, test_images.dtype)
print(train_labels.shape, train_labels.dtype, validation_labels.shape, validation_labels.dtype, test_labels.shape, test_labels.dtype)




# Подготовка данных к передаче в нейросеть

from tensorflow.keras.utils import to_categorical

stacks = 1
train_images = np.stack((train_images,) * stacks, -1)
validation_images = np.stack((validation_images,) * stacks, -1)
test_images = np.stack((test_images,) * stacks, -1)

train_images = train_images.astype('float32') / 255
validation_images = validation_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# train_labels += 1
# validation_labels += 1
# test_labels += 1

train_labels = to_categorical(train_labels)
validation_labels = to_categorical(validation_labels)
test_labels = to_categorical(test_labels)

train_labels = train_labels.astype(np.uint8)
validation_labels = validation_labels.astype(np.uint8)
test_labels = test_labels.astype(np.uint8)

# np.save('train_images_fer2013.npy', train_images)
# np.save('validation_images_fer2013.npy', validation_images)
# np.save('test_images_fer2013.npy', test_images)

# np.save('train_labels_fer2013.npy', train_labels)
# np.save('validation_labels_fer2013.npy', validation_labels)
# np.save('test_labels_fer2013.npy', test_labels)

print(train_images.shape, train_images.dtype, validation_images.shape, validation_images.dtype, test_images.shape, test_images.dtype)
print(train_labels.shape, train_labels.dtype, validation_labels.shape, validation_labels.dtype, test_labels.shape, test_labels.dtype)




# Создание генераторов

from tensorflow.keras.preprocessing.image import ImageDataGenerator


batch_size = 64

# train_gen = ImageDataGenerator(
#     rotation_range=40,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest')

train_gen = ImageDataGenerator()
validation_gen = ImageDataGenerator()
test_gen = ImageDataGenerator()

train_gen = train_gen.flow(
    train_images,
    train_labels,
    batch_size = batch_size,
    shuffle=False)

validation_gen = validation_gen.flow(
    validation_images,
    validation_labels,
    batch_size = batch_size,
    shuffle=False)

test_gen = test_gen.flow(
    test_images,
    test_labels,
    batch_size = batch_size,
    shuffle=False)

#%%=====================================================================================================================================================
# FER2013

# Пропуск всех изображений через свёрточную основу нейросети

from tensorflow.keras import models


conv_base = models.load_model('fer2013_conv_base_cnn.h5')

# conv_base = models.load_model('fer2013_conv_base_vgg16.h5')

train_fm = conv_base.predict_generator(train_gen)
validation_fm = conv_base.predict_generator(validation_gen)
test_fm = conv_base.predict_generator(test_gen)

train_fm *= 100
validation_fm *= 100
test_fm *= 100

# train_fm[train_fm > 1000] = 1000
# validation_fm[validation_fm > 1000] = 1000
# test_fm[test_fm > 1000] = 1000

train_fm = train_fm.astype('int16')
validation_fm = validation_fm.astype('int16')
test_fm = test_fm.astype('int16')

# np.save('train_fm_fer2013_cnn.npy', train_fm)
# np.save('validation_fm_fer2013_cnn.npy', validation_fm)
# np.save('test_fm_fer2013_cnn.npy', test_fm)

# np.save('train_fm_fer2013_vgg16.npy', train_fm)
# np.save('validation_fm_fer2013_vgg16.npy', validation_fm)
# np.save('test_fm_fer2013_vgg16.npy', test_fm)

print(np.max(train_fm), np.max(validation_fm), np.max(test_fm))
print(train_fm.shape, train_fm.dtype, train_fm)
print(validation_fm.shape, validation_fm.dtype, validation_fm)
print(test_fm.shape, test_fm.dtype, test_fm)

#%%=====================================================================================================================================================
# FER2013

# CNN модель

from tensorflow.keras import models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout


nof_epochs = 50

def lr_schedule(epoch):
    lrate = 0.0005
    if epoch > 20:
        lrate = 0.0001
    return lrate

model = models.Sequential()
model.add(Conv2D(64, (3,3), activation='relu', padding='same', input_shape=(48, 48, 1)))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.0005), metrics=['accuracy'])

checkpoint = ModelCheckpoint('fer2013_cnn_model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

history = model.fit_generator(
    train_gen,
    epochs=nof_epochs,
    validation_data=validation_gen,
    callbacks=[checkpoint, LearningRateScheduler(lr_schedule)])

model = models.load_model('fer2013_cnn_model.h5')

test_loss, test_acc = model.evaluate_generator(test_gen)
print('test_acc:', test_acc)
print('test_loss:', test_loss)

#%%=====================================================================================================================================================
# FER2013

# Замороженная VGG16

from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import VGG16


nof_epochs = 10

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(48, 48, 3))

for layer in vgg16.layers:
    layer.trainable = False

model_fr = models.Sequential()
model_fr.add(vgg16)
model_fr.add(layers.Flatten())
model_fr.add(layers.Dropout(0.5))
model_fr.add(layers.Dense(256, activation='relu'))
model_fr.add(layers.Dense(7, activation='softmax'))

model_fr.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.0005), metrics=['accuracy'])

checkpoint = ModelCheckpoint('fer2013_cnn_vgg16_fr.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

history = model_fr.fit_generator(
    train_gen,
    epochs=nof_epochs,
    validation_data=validation_gen,
    callbacks=[checkpoint])

model_fr = models.load_model('fer2013_cnn_vgg16_fr.h5')

test_loss, test_acc = model_fr.evaluate_generator(test_gen)
print('test_acc:', test_acc)
print('test_loss:', test_loss)

#%%=====================================================================================================================================================
# FER2013

# Размороженная VGG16

nof_epochs = 10

set_trainable = False
for layer in vgg16.layers:
    if layer.name == 'block2_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

model_ufr = models.Sequential()
model_ufr.add(vgg16)
model_ufr.add(model_fr.layers[1])
model_ufr.add(model_fr.layers[2])
model_ufr.add(model_fr.layers[3])
model_ufr.add(model_fr.layers[4])

model_ufr.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.0001), metrics=['accuracy'])

checkpoint = ModelCheckpoint('fer2013_cnn_vgg16.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

history = model_ufr.fit_generator(
    train_gen,
    epochs=nof_epochs,
    validation_data=validation_gen,
    callbacks=[checkpoint])

model_ufr = models.load_model('fer2013_cnn_vgg16.h5')

test_loss, test_acc = model_ufr.evaluate_generator(test_gen)
print('test_acc:', test_acc)
print('test_loss:', test_loss)

#%%=====================================================================================================================================================
#%%=====================================================================================================================================================
# Выводит графики точности и потерь

# %matplotlib inline
import matplotlib.pyplot as plt


train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_acc)+1)

plt.plot(epochs, train_acc, label='Training accuracy')
plt.plot(epochs, val_acc, label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# plt.savefig('accuracy.png', bbox_inches='tight')
# plt.savefig('accuracy.pdf', bbox_inches='tight')
plt.figure()

plt.plot(epochs, train_loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# plt.savefig('loss.png', bbox_inches='tight')
# plt.savefig('loss.pdf', bbox_inches='tight')
plt.show()

#%%=====================================================================================================================================================
# Создание новой модели нейросети из свёрточной основы старой модели

from tensorflow.keras import models, optimizers


model = models.load_model('cifar10_cnn_model.h5')
# model = models.load_model('fer2013_cnn_model.h5')

# model = models.load_model('cifar10_cnn_vgg16.h5')
# model = models.load_model('fer2013_cnn_vgg16.h5')

new_model = models.Sequential()

for layer in model.layers[:16]:
    new_model.add(layer)

new_model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.0005), metrics=['accuracy'])

# new_model.save('cifar10_conv_base_cnn.h5')
# new_model.save('fer2013_conv_base_cnn.h5')

# new_model.save('cifar10_conv_base_vgg16.h5')
# new_model.save('fer2013_conv_base_vgg16.h5')

new_model.summary()

#%%=====================================================================================================================================================
# Код для вывода графиков

#%%=====================================================================================================================================================
# Выводит графики количества чисел

# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np


predictions = np.load('train_fm_fer2013.npy')

numbers, quantity = np.unique(predictions, return_counts=True)
# numbers, quantity = np.unique(train_fm, return_counts=True)

print('number of zeros:', quantity[0])

plt.bar(numbers[1:201], quantity[1:201])
plt.title('Quantity of numbers in training set (part 1)')
plt.xlabel('Numbers')
plt.ylabel('Quantity')

# plt.savefig('train_qof_numbers_part1.png', bbox_inches='tight')
# plt.savefig('train_qof_numbers_part1.pdf', bbox_inches='tight')
plt.figure()

plt.bar(numbers[201:], quantity[201:])
plt.title('Quantity of numbers in training set (part 2)')
plt.xlabel('Numbers')
plt.ylabel('Quantity')

# plt.savefig('train_qof_numbers_part2.png', bbox_inches='tight')
# plt.savefig('train_qof_numbers_part2.pdf', bbox_inches='tight')
plt.show()

#%%=====================================================================================================================================================
# Выводит сглаженные графики точности и потерь

def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

plt.plot(epochs, smooth_curve(train_acc), label='Smoothed training accuracy')
plt.plot(epochs, smooth_curve(val_acc), label='Smoothed validation accuracy')
plt.title('Smoothed training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# plt.savefig('smoothed_accuracy.png', bbox_inches='tight')
# plt.savefig('smoothed_accuracy.pdf', bbox_inches='tight')
plt.figure()

plt.plot(epochs, smooth_curve(train_loss), label='Smoothed training loss')
plt.plot(epochs, smooth_curve(val_loss), label='Smoothed validation loss')
plt.title('Smoothed training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# plt.savefig('smoothed_loss.png', bbox_inches='tight')
# plt.savefig('smoothed_loss.pdf', bbox_inches='tight')
plt.show()

#%%=====================================================================================================================================================
# Код для проверки одного изображения

#%%=====================================================================================================================================================
# Пропуск изображения через свёрточную основу нейросети

from tensorflow.keras.preprocessing import image
from tensorflow.keras import models
import numpy as np


conv_base = models.load_model('fer2013_conv_base_vgg16.h5')

img = image.load_img('pew.jpg', target_size = (48, 48))
image_array = image.img_to_array(img)
image_array = np.expand_dims(image_array, axis = 0) / 255

predictions = conv_base.predict(image_array)

# print(predictions.shape, predictions.dtype, predictions)


# img = image.load_img('pew.jpg', target_size = (48, 48), color_mode = 'grayscale')
# image_array = image.img_to_array(img)
# image_array = np.tile(image_array, 3)
# image_array = np.expand_dims(image_array, axis = 0) / 255

# predictions = conv_base.predict(image_array)

# print(predictions.shape, predictions.dtype, predictions)


predictions[predictions > 1000] = 1000
predictions = predictions.astype('int16')

print(predictions.shape, predictions.dtype, predictions)

#%%=====================================================================================================================================================
# Обработка и вывод цветной и чёрно-белой версий изображения

# %matplotlib inline
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import numpy as np


img = image.load_img('pew.jpg', target_size = (48, 48))
image_array = image.img_to_array(img) / 255

plt.imshow(image_array)
plt.show()

img = image.load_img('pew.jpg', target_size = (48, 48), color_mode = 'grayscale')
image_array = image.img_to_array(img)
image_array = np.tile(image_array, 3) / 255

plt.imshow(image_array)
plt.show()
