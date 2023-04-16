from keras.models import load_model
from keras.applications import InceptionResNetV2
from preprocess_data import *


model = load_model('ucf101model.h5')
model.evaluate(test_ds)

random_path = test_paths[np.random.choice(len(test_paths))]

x = np.array([get_frames(random_path, general_options[0], general_options[1], general_options[2], general_options[3], general_options[4])])
y = np.argmax(model(x))

label = label_reverse[y]

cap = cv2.VideoCapture(random_path)
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        # Add predicted label as text to the video frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2
        cv2.putText(frame, str(label), org, font, fontScale, color, thickness, cv2.LINE_AA)

        # Display the video frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

# Release the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()