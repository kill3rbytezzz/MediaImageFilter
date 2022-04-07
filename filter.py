import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt

img = cv2.imread("download.jpeg")

def bgtogray(imgs):

    R = np.array(imgs[:,:,2])
    G = np.array(imgs[:,:,1])
    B = np.array(imgs[:,:,0])
    
    R = (R *.3333)
    G = (G *.3333)
    B = (B *.3333)
    
    Avg = (R+G+B)
    
    grayImage = copy.copy(imgs)
    
    for i in range(3):
        grayImage[:,:,i] = Avg
    
    return grayImage


def im2bw(gr,thd):
    br,kl,d = gr.shape
    bw = np.array(gr[:,:,0])

    for i in range(br):
        for j in range (kl):
            if (gr[i,j,0]<thd):
                bw[i,j]=0
            else:
                bw[i,j] = 255

    bwimgs = copy.copy(gr)

    for i in range(3):
        bwimgs[:,:,i] = bw

    return bwimgs
    
def blur(gr):
 
    m, n,d = gr.shape
    b = np.array(gr[:,:,0])
     
    for i in range(1, m-1):
        for j in range(1, n-1):
            temp = [gr[i-1][j-1][0],
                   gr[i-1][j][0],
                   gr[i-1][j + 1][0],
                   gr[i][j-1][0],
                   gr[i][j][0],
                   gr[i][j + 1][0],
                   gr[i + 1][j-1][0],
                   gr[i + 1][j][0],
                   gr[i + 1][j + 1][0]]
             
            print(temp)
            temp = sorted(temp)
            b[i, j]= temp[4]
    
    bimgs = copy.copy(gr)

    for i in range(3):
        bimgs[:,:,i] = b
    return bimgs


gray = bgtogray(img)
bw = im2bw(gray,100)
b = blur(gray)

print (gray.shape)
print ("Nilai piksel B,G,R pada Citra bewarna pada baris 1000 sampai 105 dan kolom 100 sampai 105 :\n")
print ("B: \n", img[100:105, 100:105,0], "\n\nG: \n", img[100:105, 100:105,1], "\n\nR: \n", img[100:105, 100:105,2])
print ("\n Nilai piksel B,G,R pada Citra bewarna pada baris 1000 sampai 105 dan kolom 100 sampai 105 :\n")
print ("B: \n", gray[100:105, 100:105,0], "\n\nG: \n", gray[100:105, 100:105,1], "\n\nR: \n", gray[100:105, 100:105,2])

plt.subplot(221)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Citra Asli")
plt.xticks([]), plt.yticks([])

plt.subplot(222)
plt.imshow(gray)
plt.title("Grayscale")
plt.xticks([]), plt.yticks([])

plt.subplot(223)
plt.imshow(bw)
plt.title("B&W")
plt.xticks([]), plt.yticks([])

plt.subplot(224)
plt.imshow(b)
plt.title("Median")
plt.xticks([]), plt.yticks([])

plt.show()
