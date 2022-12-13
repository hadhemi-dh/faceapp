import cv2

# Load the cascade file for face detection
face_cascade = cv2.CascadeClassifier('path/to/face/cascade/file.xml')

# Load the input image
img = cv2.imread('hadhemi dhaouadi.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Print the number of faces detected
print(f'{len(faces)} faces detected')

# Loop through the faces and draw a rectangle around them
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Save the output image
cv2.imwrite('path/to/output/image.jpg', img)
