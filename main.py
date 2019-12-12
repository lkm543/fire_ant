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
        self.frame = wx.Frame(None, title='Photo Control')
 
        self.panel = wx.Panel(self.frame)
 
        self.PhotoMaxSize = 640
 
        self.createWidgets()
        self.frame.Show()
 
    def createWidgets(self):
        instructions = 'Browse for an image'
        img = wx.Image(640,480)
        self.imageCtrl = wx.StaticBitmap(self.panel, wx.ID_ANY, 
                                         wx.Bitmap(img))
 
        instructLbl = wx.StaticText(self.panel, label=instructions)
        self.photoTxt = wx.TextCtrl(self.panel, size=(200,-1))
        browseBtn = wx.Button(self.panel, label='Browse')
        browseBtn.Bind(wx.EVT_BUTTON, self.onBrowse)
 
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

    def onBrowse(self, event):
        """ 
        Browse for file
        """
        wildcard = "JPEG files (*.jpg)|*.JPG"
        dialog = wx.FileDialog(None, "Choose a file",
                               wildcard=wildcard,
                               style=wx.FD_OPEN)
        if dialog.ShowModal() == wx.ID_OK:
            self.photoTxt.SetValue(dialog.GetPath())
            self.onView()
        dialog.Destroy()

    def onView(self):
        img = self.loadImage()
        # scale the image, preserving the aspect ratio
        W = img.GetWidth()
        H = img.GetHeight()
        if W > H:
            NewW = self.PhotoMaxSize
            NewH = self.PhotoMaxSize * H / W
        else:
            NewH = self.PhotoMaxSize
            NewW = self.PhotoMaxSize * W / H
        img = img.Scale(NewW,NewH)

        self.imageCtrl.SetBitmap(wx.Bitmap(img))
        self.panel.Refresh()

    def loadImage(self):
        filepath = self.photoTxt.GetValue()
        image = io.imread(filepath)
        image = self.segmentation(image)
        height, width, nrgb = image.shape
        wximg = wx.ImageFromBuffer(width, height, image)
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
        ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        erode = cv2.erode(binary, None, iterations=30)
        dilate = cv2.dilate(erode, None, iterations=30)
        dilate = cv2.dilate(erode, None, iterations=50)
        erode = cv2.erode(binary, None, iterations=50)
        # binary = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,111,21)
        result = cv2.cvtColor(dilate, cv2.COLOR_GRAY2RGB)

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

if __name__ == '__main__':
    app = Frame()
    app.MainLoop()
