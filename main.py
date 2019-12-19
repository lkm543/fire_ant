import os
import wx
import skimage.segmentation as seg
from skimage import io
import random
import cv2
import numpy as np


# http://www.blog.pythonlibrary.org/2010/03/26/creating-a-simple-photo-viewer-with-wxpython/
# https://github.com/h4k1m0u/wx-imageprocessing/blob/master/imageprocessing.py
class Frame(wx.App):
    def __init__(self, redirect=False, filename=None):
        wx.App.__init__(self, redirect, filename)
        self.block_size = 10 # %
        self.threshold = 30
        self.frame = wx.Frame(None, title='Photo Control')
 
        self.panel = wx.Panel(self.frame)
 
        self.PhotoMaxSize = 640
 
        self.createwidgets()
        self.frame.Show()
 
    def createwidgets(self):
        instructions = 'Browse for an image'
        img = wx.Image(640,640)
        self.imageCtrl = wx.StaticBitmap(self.panel, wx.ID_ANY, 
                                         wx.Bitmap(img))
 
        instructLbl = wx.StaticText(self.panel, label=instructions)
        self.photoTxt = wx.TextCtrl(self.panel, size=(200,-1))
        browseBtn = wx.Button(self.panel, label='Browse')
        browseBtn.Bind(wx.EVT_BUTTON, self.on_browse)
 
        self.mainSizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.mainSizer.Add(wx.StaticLine(self.panel, wx.ID_ANY),
                           0, wx.ALL|wx.EXPAND, 5)
        self.mainSizer.Add(instructLbl, 0, wx.ALL, 5)
        self.mainSizer.Add(self.imageCtrl, 0, wx.ALL, 5)
        self.sizer.Add(self.photoTxt, 0, wx.ALL, 5)
        self.sizer.Add(browseBtn, 0, wx.ALL, 5)        
        self.mainSizer.Add(self.sizer, 0, wx.ALL, 5)
 
        self.panel.SetSizer(self.mainSizer)
        self.mainSizer.Fit(self.frame)

        self.panel.Layout()

    def on_browse(self, event):
        """ 
        Browse for file
        """
        wildcard = "JPEG files (*.jpg)|*.JPG"
        dialog = wx.FileDialog(None, "Choose a file",
                               wildcard=wildcard,
                               style=wx.FD_OPEN)
        if dialog.ShowModal() == wx.ID_OK:
            self.photoTxt.SetValue(dialog.GetPath())
            self.on_view()
        dialog.Destroy()

    def on_view(self):
        img = self.load_image()
        # scale the image, preserving the aspect ratio
        W = img.GetWidth()
        H = img.GetHeight()
        if W > H:
            NewW = self.PhotoMaxSize
            NewH = self.PhotoMaxSize * H / W
        else:
            NewW = self.PhotoMaxSize * W / H
            NewH = self.PhotoMaxSize
        img = img.Scale(NewW,NewH)

        self.imageCtrl.SetBitmap(wx.Bitmap(img))
        self.panel.Refresh()

    def load_image(self):
        filepath = self.photoTxt.GetValue()
        image = io.imread(filepath)
        image = self.segmentation(image)
        height, width, nrgb = image.shape
        wximg = wx.ImageFromBuffer(width, height, image)
        save_filename = self.photoTxt.GetValue().split('\\')[-1]
        self.save_image(wx.Bitmap(wximg), save_filename)
        return wximg

    def segmentation(self, image_ndarray):
        '''
        # Graph Based Segmentation 
        segmentator = cv2.ximgproc.segmentation.createGraphSegmentation(sigma=1, k=30, min_size=500)
        segment = segmentator.processImage(image_ndarray)
        seg_image = np.zeros(image_ndarray.shape, np.uint8)
        for i in range(np.max(segment)):
            # 將第 i 個分割的座標取出
            y, x = np.where(segment == i)

        # 隨機產生顏色
        color = [random.randint(0, 255), random.randint(0, 255),random.randint(0, 255)]

        # 設定第 i 個分割區的顏色
        for xi, yi in zip(x, y):
            seg_image[yi, xi] = color

        # 將原始圖片與分割區顏色合併
        result = cv2.addWeighted(image_ndarray, 0.3, seg_image, 0.7, 0)
        '''

        # Convert to Gray and threshold, erode and dilate
        gray = cv2.cvtColor(image_ndarray, cv2.COLOR_BGR2GRAY)
        cv2.normalize(gray, gray, 0, 255, cv2.NORM_MINMAX)
        ret, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        binary = cv2.erode(otsu_mask, None, iterations=45)
        binary = cv2.dilate(binary, None, iterations=120)
        binary = cv2.erode(binary, None, iterations=75)
        # binary = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,111,21)
        result = image_ndarray.copy()
        masked = cv2.bitwise_and(image_ndarray, image_ndarray, mask=binary)
        result = cv2.addWeighted(image_ndarray, 0.35, masked, 0.65, 0)

        '''
        hsv = cv2.cvtColor(image_ndarray, cv2.COLOR_BGR2HSV)
        # print(hsv)
        center_hue, center_saturation, center_value = self.get_center_color(image_ndarray)
        gray = cv2.cvtColor(image_ndarray, cv2.COLOR_RGB2GRAY)
        kernel_size = 3
        gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)
        mask = cv2.Canny(gray, 30, 255)
        # cv2.imshow('Edges', edges)
        # print(f'Select Hue: {center_hue}')
        '''

        '''
        row, col, channel = image_ndarray.shape
        ret, otsu_mask = cv2.threshold(hsv[:,:,2], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        mask = np.zeros((row, col, 1), np.uint8)
        '''
        '''
        for i in range(row):
            for j in range(col):
                if center_value-self.threshold < hsv[i,j,2] < center_value+self.threshold:
                    mask[i,j] = 1
        '''
        # masked = cv2.bitwise_and(image_ndarray, image_ndarray, mask=mask)
        # result = cv2.addWeighted(image_ndarray, 0.2, masked, 0.8, 0)

        '''
        # Canny and find contour
        gray = cv2.cvtColor(image_ndarray, cv2.COLOR_BGR2GRAY)
        cv2.normalize(gray, gray, 0, 255, cv2.NORM_MINMAX)
        canny = cv2.Canny(gray, 30, 150)
        cnts = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = image_ndarray.copy()
        cv2.drawContours(result, cnts, -1, (0, 255, 0), 2)
        '''

        return result

    def get_center_color(self, image_ndarray):
        row, col, channel = image_ndarray.shape
        width = int(min(row, col) * self.block_size / 100 / 2)
        center_row = int(row/2)
        center_col = int(col/2)
        crop_image = image_ndarray[
            center_row-width:center_row+width,
            center_col-width:center_col+width,
            :
        ]
        # cv2.imshow('crop_image', crop_image)
        hsv = cv2.cvtColor(crop_image, cv2.COLOR_BGR2HSV)
        hue = np.median(hsv[:, :, 0])
        saturation = np.median(hsv[:,:,1])
        value = np.median(hsv[:,:,2])
        # print(hue,saturation,value)
        return hue, saturation, value

    def save_image(self, image, filename):
        image.SaveFile(filename, wx.BITMAP_TYPE_JPEG)

if __name__ == '__main__':
    app = Frame()
    app.MainLoop()
