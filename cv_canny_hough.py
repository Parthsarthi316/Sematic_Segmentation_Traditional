import numpy as np
import math
import cv2
from matplotlib import pyplot as plt

nearest_circle = []
curr_dist = 0

img1 = cv2.imread(r"E:\Downloads3\base_1.jpeg") # query Image
img2 = cv2.imread(r"E:\Downloads3\leprecon-20211210T234703Z-001\leprecon\mag5.jpg") # target Image 8 11 5

# img2 = cv2.imread(r"E:\Downloads3\test_2.jpeg") # query Image


img2=cv2.resize(img2,(768,1024))
img_final = img2
img1=cv2.GaussianBlur(img1,(11,11),cv2.BORDER_DEFAULT)
img2=cv2.GaussianBlur(img2,(11,11),cv2.BORDER_DEFAULT)

# Initiate ORB detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# Descriptor shape
# print(des1.shape)
# print(des2.shape)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
# rev_matches=[]
# rev_matches[::-2]
good_matches = matches[:50]

img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,None, flags=2)

# Show all the matches
plt.imshow(img3),plt.show()

src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
matchesMask = mask.ravel().tolist()
h,w = img1.shape[:2]
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

dst = cv2.perspectiveTransform(pts,M)
dst += (w, 0)  # adding offset

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
               singlePointColor = None,
               # matchesMask = matchesMask, # draw only inliers
               flags = 2)
t1=np.reshape(dst_pts,(len(dst_pts),2))
point=np.median(t1,axis=0)
k,l=int(point[0]),int(point[1])
# Draw keypoints matches
img3 = cv2.drawMatches(img1,kp1,img2,kp2,good_matches, None,**draw_params)

# Draw bounding box on target image and show along with train image 
img3 = cv2.polylines(img3, [np.int32(dst)], True, (0,0,255),3, cv2.LINE_AA)
img3=cv2.resize(img3,(854,480))

# Print location of median
print(f"median: {k}, {l}") 

# or another option for display output
# plt.imshow(img3, 'result'), plt.show()

# Draw bounding box on target image
# side=330
# top=100
# bottom=600
# start_point=(l-side,k-top)
# end_point=(l+side,k+bottom)
# img2=cv2.resize(img2,(854,480))
# image = cv2.rectangle(img2, start_point, end_point, color=(255,255,255), thickness=5)
# image = cv2.resize(img2, (480,754))


# Applying canny
gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 5)
center=(0,0)
radius=0
gray=cv2.GaussianBlur(gray,(25,25),cv2.BORDER_DEFAULT)
gray=cv2.Canny(gray,20,30) #20,30
# cv2.imshow("g",gray)
rows = gray.shape[0]

# Applying hough 
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8,
                           param1=100, param2=30,
                           minRadius=10, maxRadius=250)

if circles is not None:
    circles = np.uint16(np.around(circles))
    ctr = 0
    for i in circles[0, :]:
        ctr+=1
        # Finding nearest circle to the median 
        print(f"circle {ctr} points: " + str(i), end="")
        dist = abs(math.sqrt((k-i[0])**2 + (l-i[1])**2))
        print(" dist: " + str(dist))
        if dist<curr_dist or curr_dist == 0:
            nearest_circle = i
            curr_dist = dist
            dist = 0
        # numbering circle
        cv2.putText(gray, str(ctr), (i[0],i[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (246,255,12), 3)    
        # draw circle 
        radius = i[2]
        cv2.circle(gray, (i[0],i[1]), radius, (255, 0, 255), 3)

print(f"nearest circle: {nearest_circle}")            

c1,c2=nearest_circle[0], nearest_circle[1]    
side=(nearest_circle[2])*2.5
side=int(side)


# draw median on final image
cv2.circle(img_final, (k,l), 1, (255, 0, 255), 10)
# draw circle on final image
cv2.circle(img_final, (c1,c2), nearest_circle[2], (255, 0, 255), 3)
# dst=[()]
# cv2.polylines((img_final, [np.int32(dst)], True, (0,0,255),3, cv2.LINE_AA)

p1=(c1-side,c2-side)
p2=(c1+side,c2-side)
p3=(c1+side,c2+side)
p4=(c1-side,c2+side)




# box=cv2.boxPoints(start)
# cv2.rectangle(img_final,start,end,5,thickness=5)



















a,b=gray.shape
l=[]
for i in range(a):
    for j in range(b):
        if gray[i][j]==255 and i>(c1-side) and i<(c1+side) and j>(c2-side) and j<(c2+side):
            l.append([i,j])

cv2.imshow("matches", img3)
cv2.imshow("detected circles", gray)

# cv2.waitKey(0)




import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
import math

# X, y = make_classification(n_samples=1000)
X=np.array(l)
n_samples = X.shape[0]

pca = PCA()
X_transformed = pca.fit_transform(X)

# We center the data and compute the sample covariance matrix.
X_centered = X - np.mean(X, axis=0)
cov_matrix = np.dot(X_centered.T, X_centered) / n_samples
eigenvalues = pca.explained_variance_
eigenvectors=[]
for eigenvalue, eigenvector in zip(eigenvalues, pca.components_):    
    # print(np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)))
    # print(eigenvalue)
    eigenvectors.append(eigenvector)

print(eigenvectors)
tampp = 45
angle2=math.atan2(eigenvectors[1][1],eigenvectors[1][0])*180/math.pi+tampp
angle1=math.atan2(eigenvectors[0][1],eigenvectors[0][0])*180/math.pi+tampp
print(angle1)
print(angle2)
angle_deg=max(angle1,angle2)
print(angle_deg)

rect=(c1,c2),(side*2,side*2),angle1
box=cv2.boxPoints(rect)
box=np.int0(box)
cv2.drawContours(img_final,[box],0,(0,0,255),4)


cv2.imshow("detected circles", gray)
cv2.imshow("result", img_final)
cv2.waitKey(0)
cv2.destroyAllWindows()

