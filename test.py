# Test Images
new_size=224
test_images=[]
test_labels=[]
for i in os.listdir(testpath):# entering to the test folder
  print("Entering to the folder name:",i)
  files=gb.glob(pathname=str(testpath +'/' + i + '/*.jpg'))# pointing to all the .jpg extension image folder
  print("Number of images in the folder is",len(files))
  class_cancer={'benign':0,'malignant':1}
  for j in files:
      image_raw=cv2.imread(j)
      image=cv2.cvtColor(image_raw,cv2.COLOR_BGR2RGB)
      resize_image=cv2.resize(image,(new_size,new_size))
      test_images.append(list(resize_image))
      test_labels.append(class_cancer[i])

def list_to_array_test(test_images,test_labels):
  return np.array(test_images),np.array(test_labels)

X_test,y_test=list_to_array_test(test_images,test_labels)
print(X_test.shape)
print(y_test.shape)