import numpy as np
import cv2 as cv
from multiprocessing.pool import ThreadPool
import time
import scipy.misc
import matplotlib.pyplot as plt

#Set up detector, matcher
def init_feature():

    detector = cv.BRISK_create(thresh=60, octaves=4)

    # flann_params= dict(algorithm = 6,  #Parameter configuration
    #                        table_number = 12,
    #                        key_size = 20,
    #                        multi_probe_level = 2)
    # searchParams = dict(checks=10)
    # matcher = cv.FlannBasedMatcher(flann_params, searchParams)
    matcher = cv.BFMatcher(cv.NORM_HAMMING)
    return detector, matcher

#KNN was used to screen the characteristics
def filter_matches(kp1, kp2, matches, ratio = 0.7):
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append(kp1[m.queryIdx])
            mkp2.append(kp2[m.trainIdx])
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, list(kp_pairs)

#The homography matrix is used to find the target position coordinates and the feature points are connected
def explore_match(img1, img2, kp_pairs, status = None, H = None):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    vis = cv.cvtColor(vis, cv.COLOR_GRAY2BGR)

    #Find the four angles of registration according to the homography matrix
    if H is not None:
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32( cv.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0) )
        cv.polylines(vis, [corners], True, (255, 255, 255))

    if status is None:
        status = np.ones(len(kp_pairs), np.bool_)

    #Feature points are extracted and wired
    p1, p2 = [], []
    for kpp in kp_pairs:
        p1.append(np.int32(kpp[0].pt))
        p2.append(np.int32(np.array(kpp[1].pt) + [w1, 0]))

    green = (0, 255, 0)
    red = (0, 0, 255)

    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            col = green
            cv.circle(vis, (x1, y1), 2, col, -1)
            cv.circle(vis, (x2, y2), 2, col, -1)
        else:
            col = red
            r = 2
            thickness = 3
            cv.line(vis, (x1-r, y1-r), (x1+r, y1+r), col, thickness)
            cv.line(vis, (x1-r, y1+r), (x1+r, y1-r), col, thickness)
            cv.line(vis, (x2-r, y2-r), (x2+r, y2+r), col, thickness)
            cv.line(vis, (x2-r, y2+r), (x2+r, y2-r), col, thickness)

    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            cv.line(vis, (x1, y1), (x2, y2), green)

    scipy.misc.imsave('./4.jpg', vis)
    # vis.save('./1.jpg')
    # plt.imshow(vis)
    # plt.show()

#Affine transformation is performed on the registration image
def affine_skew(tilt, phi, img, mask=None):
    h, w = img.shape[:2]
    if mask is None:
        mask = np.zeros((h, w), np.uint8)
        mask[:] = 255
    A = np.float32([[1, 0, 0], [0, 1, 0]])
    if phi != 0.0:
        phi = np.deg2rad(phi)
        s, c = np.sin(phi), np.cos(phi)
        A = np.float32([[c,-s], [ s, c]])
        corners = [[0, 0], [w, 0], [w, h], [0, h]]
        tcorners = np.int32( np.dot(corners, A.T) )
        x, y, w, h = cv.boundingRect(tcorners.reshape(1,-1,2))
        A = np.hstack([A, [[-x], [-y]]])
        img = cv.warpAffine(img, A, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
    if tilt != 1.0:
        s = 0.8*np.sqrt(tilt*tilt-1)
        img = cv.GaussianBlur(img, (0, 0), sigmaX=s, sigmaY=0.01)
        img = cv.resize(img, (0, 0), fx=1.0/tilt, fy=1.0, interpolation=cv.INTER_NEAREST)
        A[0] /= tilt
    if phi != 0.0 or tilt != 1.0:
        h, w = img.shape[:2]
        mask = cv.warpAffine(mask, A, (w, h), flags=cv.INTER_NEAREST)
    Ai = cv.invertAffineTransform(A)
    return img, mask, Ai

#Multiple threads perform affine transformations
def affine_detect(detector, img, mask=None, pool=None):
    params = [(1.0, 0.0)]
    for t in 2**(0.5*np.arange(1,6)):
        for phi in np.arange(0, 180, 72.0 / t):
            params.append((t, phi))

    def f(p):
        t, phi = p
        timg, tmask, Ai = affine_skew(t, phi, img)
        keypoints, descrs = detector.detectAndCompute(timg, tmask)
        for kp in keypoints:
            x, y = kp.pt
            kp.pt = tuple( np.dot(Ai, (x, y, 1)) )
        if descrs is None:
            descrs = []
        return keypoints, descrs

    keypoints, descrs = [], []

    ires = pool.imap(f, params)

    for i, (k, d) in enumerate(ires):
        keypoints.extend(k)
        descrs.extend(d)

    return keypoints, np.array(descrs)

def gamma_trans(img, gamma):  # Gamma processing
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv.LUT(img, gamma_table)


def custom_blur_demo(image):  # sharpening

  kernel = np.array([[-1, -1, -1, -1, -1],[-1, 2, 2, 2, -1],
                     [-1, 2, 8, 2, -1], [-1, 2, 2, 2, -1], [-1, -1, -1, -1, -1]]) / 8.0

  dst = cv.filter2D(image, -1, kernel=kernel)
  return dst

def img_bla(image):  # Limit the adaptive threshold equalization of contrast
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    dst = clahe.apply(image)
    return dst

def test(fn1, fn2):

    img1 = cv.imread(fn1, 0)
    img2 = cv.imread(fn2, 0)

    #Optional part
    # img2 = gamma_trans(img2, 0.8)  #Choose this function to lighten the image
    # img2 = custom_blur_demo(img2)  #Image blur selection sharpens this function
    # roi1 = cv.selectROI(windowName="roi1", img=img1, showCrosshair=True, fromCenter=False)  # Mouse select ROI area
    # x1, y1, w1, h1 = roi1
    # img1 = img1[y1:y1 + h1, x1:x1 + w1]
    # img1 = img_bla(img1)

    start = time.clock()
    detector, matcher = init_feature()  # Initialize detector, matcher

    pool = ThreadPool(processes=cv.getNumberOfCPUs())
    kp1, desc1 = affine_detect(detector, img1, pool=pool)
    kp2, desc2 = affine_detect(detector, img2, pool=pool)
    pool.close()

    print('img1 - %d features, img2 - %d features' % (len(kp1), len(kp2)))

    len_desc1 = len(desc1)
    len_desc2 = len(desc2)
    len_kp1 = len(kp1)
    len_kp2 = len(kp2)

    def matchs(n):
        desc1_1 = desc1[int(len_desc1 * 0.1* n):int(len_desc1 * 0.1* (n + 1)), :]
        desc2_1 = desc2[int(len_desc2 * 0.1 * n):int(len_desc2 * 0.1* (n + 1)), :]
        raw_matches_1 = matcher.knnMatch(desc1_1, trainDescriptors=desc2_1, k=2)
        kp1_1 = kp1[int(len_kp1 * 0.1 * n):int(len_kp1 * 0.1* (n + 1))]
        kp2_1 = kp2[int(len_kp2 * 0.1 * n):int(len_kp2 * 0.1 * (n + 1))]
        p1_1, p2_1, kp_pairs1 = filter_matches(kp1_1, kp2_1, raw_matches_1)
        return p1_1, p2_1, kp_pairs1

    pool2 = ThreadPool(processes=cv.getNumberOfCPUs())
    results = []
    d = range(0, 10, 1)
    for i in d:
        results.append(pool2.apply_async(matchs, (i,)))
    pool2.close()
    pool2.join()


    p1 = []
    p2 = []
    kp_pairs=[]
    for res in results:
        data_1 = (res.get())[0]
        data_2 = (res.get())[1]
        data_3 = (res.get())[2]
        p1 = p1+list(data_1)
        p2 = p2+list(data_2)
        kp_pairs = kp_pairs+data_3
    p1 = np.array(p1)
    p2 = np.array(p2)

    #The monotonic change matrix H was obtained by the feature points, and the feature points were further screened by RANSAC
    if len(p1) >= 4:
        H, status = cv.findHomography(p1, p2, cv.RANSAC, 5.0)
        kp_pairs = [kpp for kpp, flag in zip(kp_pairs, status) if flag]


        #If you choose the ROI region,need complete the cut part of the ROI area picture
        # for i in range(len(p1)):
        #     p1[i][0] = p1[i][0] + x1
        #     p1[i][1] = p1[i][1] + y1


        elapsed = (time.clock() - start)
        print("Time used:", elapsed)
        print('%d / %d  inliers/matched' % (np.sum(status), len(status)))

    else:
        H, status = None, None
    explore_match(img1, img2, kp_pairs, None, H)
    return H
if __name__ == "__main__":
    fn1 = './2_1 (1).jpg'
    fn2 = './2_1 (2).jpg'
    H = test(fn1, fn2)