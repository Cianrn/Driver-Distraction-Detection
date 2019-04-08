from driver_detection_train import *
import random

input_shape = (120, 140)
num_classes = 10
cnn = CNN(num_classes=num_classes, learning_rate=LEARNING_RATE, input_shape=input_shape, model='simple')
cnn.load_model()
test_imgs = load_test_data(num_examples=20)
for img in test_imgs:
	pred, lab = cnn.predict_single(img)
	state = label_dict[lab[0]]
	cv2.rectangle(img, (40, 52), (320, 8), (255,255,255), cv2.FILLED)
	cv2.rectangle(img, (40, 52), (320, 8), (0,0,0), 3)
	cv2.putText(img, str(state), (42, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (0, 0, 255), 2)
	plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
	plt.show()


