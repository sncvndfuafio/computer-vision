import cv2

# Load an image
img = cv2.imread(r'C:\Users\PAKISTAN\Desktop\open cv\ahmed.jpg')

# Display the image
cv2.imshow('Original Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

resized = cv2.resize(img, (400, 300))
cv2.imshow('Resized Image', resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Draw a rectangle
cv2.rectangle(img, (50, 50), (200, 200), (0, 255, 0), 2)

# Draw a circle
cv2.circle(img, (300, 300), 50, (255, 0, 0), 3)

# Add text
cv2.putText(img, "OpenCV Demo", (50, 400),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

cv2.imshow('Drawings', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

blur = cv2.GaussianBlur(img, (15, 15), 0)
cv2.imshow('Blurred Image', blur)
cv2.waitKey(0)
cv2.destroyAllWindows()

edges = cv2.Canny(img, 100, 200)
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.imshow('Live Feed', frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
