# Classification-of-stars-and-nebulae
Classify bright point-like objects as stars.  Classify extended, diffuse regions as nebulae.
pip install opencv-python numpy scikit-learn matplotlib

import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load astronomical image
img = cv2.imread('astronomy.jpg')
if img is None:
    raise Exception("Image not found!")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Preprocessing
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

features = []
positions = []

# Extract features: area and circularity
for cnt in contours:
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0 or area < 10:
        continue
    circularity = 4 * np.pi * area / (perimeter ** 2)
    M = cv2.moments(cnt)
    if M['m00'] == 0: continue
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    features.append([area, circularity])
    positions.append((cx, cy))

# Clustering (KMeans) to classify as star or nebula
features = np.array(features)
kmeans = KMeans(n_clusters=2, random_state=0).fit(features)
labels = kmeans.labels_

# Draw results
output = img.copy()
for i, (cx, cy) in enumerate(positions):
    label = labels[i]
    color = (0, 255, 0) if label == 0 else (0, 0, 255)
    cv2.circle(output, (cx, cy), 5, color, 2)

# Show result
cv2.imshow('Classification: Green=Class0, Red=Class1', output)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optional: Plot features to see clustering
plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='coolwarm')
plt.xlabel("Area")
plt.ylabel("Circularity")
plt.title("KMeans Feature Space")
plt.show()
