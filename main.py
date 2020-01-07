import cv2
import numpy as np
import wx
from skimage import io

from segmentation import image_process


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
        self.imageCtrl_l = wx.StaticBitmap(self.panel, wx.ID_ANY, 
                                         wx.Bitmap(img))
        self.imageCtrl_2 = wx.StaticBitmap(self.panel, wx.ID_ANY, 
                                         wx.Bitmap(img))
 
        instructLbl = wx.StaticText(self.panel, label=instructions)
        self.photoTxt = wx.TextCtrl(self.panel, size=(200,-1))
        browseBtn = wx.Button(self.panel, label='Browse')
        browseBtn.Bind(wx.EVT_BUTTON, self.on_browse)
 
        self.mainSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.sizer = wx.BoxSizer(wx.VERTICAL)

        self.mainSizer.Add(wx.StaticLine(self.panel, wx.ID_ANY),
                           0, wx.ALL|wx.EXPAND, 5)
        self.sizer.Add(instructLbl, 0, wx.ALL, 5)
        self.sizer.Add(self.photoTxt, 0, wx.ALL, 5)
        self.sizer.Add(browseBtn, 0, wx.ALL, 5)
        self.mainSizer.Add(self.sizer, 0, wx.ALL, 5)
        self.mainSizer.Add(self.imageCtrl_l, 0, wx.ALL, 5)
        self.mainSizer.Add(self.imageCtrl_2, 0, wx.ALL, 5)
 
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
            origin_img, processed_img = self.load_image()
            self.on_view(self.imageCtrl_l, origin_img)
            self.on_view(self.imageCtrl_2, processed_img)
        dialog.Destroy()

    def on_view(self, imageCtrl, img):
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

        imageCtrl.SetBitmap(wx.Bitmap(img))
        self.panel.Refresh()

    def load_image(self):
        filepath = self.photoTxt.GetValue()
        image = io.imread(filepath)
        height, width, nrgb = image.shape
        origin_image = wx.ImageFromBuffer(width, height, image)
        image = image_process.segmentation(image)
        height, width, nrgb = image.shape
        wximg = wx.ImageFromBuffer(width, height, np.array(image))
        save_filename = self.photoTxt.GetValue().split('\\')[-1]
        # image.SaveFile(filename, wx.BITMAP_TYPE_JPEG)
        return origin_image, wximg

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

if __name__ == '__main__':
    app = Frame()
    app.MainLoop()
