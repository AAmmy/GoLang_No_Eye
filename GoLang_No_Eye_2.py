#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, time, cv2, numpy

def gen_tk(tk_size=100):
    return [cv2.resize(tk, (tk_size, tk_size)) for tk in tk_f]

def gen_cr(cr_id, cr_h=100):
    cr_w = credit[cr_id][0].shape[1] / (credit[cr_id][0].shape[0] / float(cr_h))
    return [cv2.resize(cr, (int(cr_w), cr_h)) for cr in credit[cr_id]]
    
def set_tk(img, x=100, y=100, w=200, h=50):
    tk = gen_tk(min([h, img.shape[0], img.shape[1]]))
    y0, y1, x0, x1 = y, y + h, x, x + h
    overlay_bw(img, tk[0:2], x0, x1, y0, y1)
    y0, y1, x0, x1 = y, y + h, x + w, x + w + h
    overlay_bw(img, tk[2:4], x0, x1, y0, y1)

def set_cr(img, cr, cr_x=100, cr_y=100):
    y0, y1, x0, x1 = cr_y, cr_y + cr[0].shape[0], cr_x, cr_x + cr[0].shape[1]
    overlay_bw(img, cr[0:2], x0, x1, y0, y1)

def overlay_bw(img, mask, x0, x1, y0, y1):
    overlay_b(img, mask[1], x0, x1, y0, y1)
    overlay_b(img, mask[0], x0, x1, y0, y1)
    overlay_w(img, mask[0], x0, x1, y0, y1)

def overlay_b(img, mask, x0, x1, y0, y1):
    img[y0:y1, x0:x1] -= img[y0:y1, x0:x1] * (mask / 200)

def overlay_w(img, mask, x0, x1, y0, y1):
    img[y0:y1, x0:x1] += mask

def showText(frame, x, y, ex, ey, ew, eh):
    ofset = 0.3 * eh
    if not ([ex, ey, ew, eh] == numpy.zeros(4)).any():
        tk_x, tk_y, tk_size, tk_len =x+ex+ofset, y+ey, eh, ew-eh-ofset
        if tk_x + tk_len + tk_size < frame.shape[1] and tk_y + tk_size < frame.shape[0]:
            set_tk(frame, x=tk_x, y=tk_y, w=tk_len, h=tk_size)
            
    p0.update(len(credit))
    cr_id = p0.cr_id
    cr_size = int(eh * 2 * p0.cr_size)
    cr = gen_cr(cr_id, min([cr_size, frame.shape[0], frame.shape[1]]))
    cr_x = x + ex + ew / 2 - cr[0].shape[1] / 2
    cr_y = y + ey + 50 + p0.cr_y
    
    try:
        set_cr(frame, cr, cr_x=cr_x, cr_y=cr_y)
    except Exception, e:
        print e
        
class timer0:
    def __init__(self):
        self.start_time = time.time()
    def time(self, t=5):
        elapsed_time = time.time() - self.start_time
        if elapsed_time > t:
            self.start_time = time.time()
            return True
        return False

class params:
    def __init__(self):
        self.t0 = timer0()
        self.cr_id = 0
        self.cr_size = 1
        self.cr_y = 0
    def update(self, id_len):
        if self.t0.time(2):
            oldid = self.cr_id
            newid = self.cr_id
            while oldid == newid: newid = numpy.random.randint(0, id_len)
            self.cr_id = newid
            self.cr_size = max(numpy.random.rand() + 1, 1)
            self.cr_y = numpy.random.randint(0, 20)

cv_cascade_path = '/usr/share/opencv/haarcascades/'
face_cascade = cv2.CascadeClassifier(cv_cascade_path + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv_cascade_path + 'haarcascade_mcs_eyepair_small.xml')

credit_path = 'credit/'
credit = [[cv2.imread(credit_path + fname + bw) for bw in ['/w.png', '/b.png']] for fname in os.listdir(credit_path)]
tk_f = [cv2.imread('teikyou/' + fname) for fname in ['t_s.png', 't_l.png', 'k_s.png', 'k_l.png']]

p0 = params()
h_n, h_c = 3, 0
hist = list(numpy.zeros((h_n, 6)))

cap = cv2.VideoCapture(-1)
while(True):
    _, frame = cap.read()
    if not frame == None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if not faces == ():
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                eye = eye_cascade.detectMultiScale(roi_gray)
                if not eye == ():
                    hist[h_c] = [x, y] + list(eye[0])
                    h_c = (h_c + 1) % h_n
        mx, my, mex, mey, mew, meh = [int(max(numpy.mean(ar), 1)) for ar in numpy.array(hist).transpose()]
        showText(frame, mx, my, mex, mey, mew, meh)
        cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
cap.release()
cv2.destroyAllWindows()

