#First read the data paths 
trainpath = r'#Please give the path where you are saving the train folder.'
testpath = r'#please give the path where you are saving the test folder.'

# Train Images
new_size=224
train_images=[]
train_labels=[]
for i in os.listdir(trainpath):
  print("Entering to the folder name:",i)
  files=gb.glob(pathname=str(trainpath+'/' + i + '/*.jpg'))
  print("Number of images in the folder is",len(files))
  for j in files:
      class_cancer={'benign':0,'malignant':1}
      image_raw=cv2.imread(j)
      image=cv2.cvtColor(image_raw,cv2.COLOR_BGR2RGB)
      resize_image=cv2.resize(image,(new_size,new_size))
      train_images.append(list(resize_image))
      train_labels.append(class_cancer[i])


# Visualizing Train Images
w=40
h=30
fig=plt.figure(figsize=(12, 8))
columns = 5
rows = 4

for i in range(1, columns*rows +1):
    ax = fig.add_subplot(rows, columns, i)
    if train_labels[i] == 0:
        ax.title.set_text('benign')
    elif train_labels[i] == 1:
        ax.title.set_text('malignant')
    plt.imshow(train_images[i], interpolation='nearest')
plt.show()

X_train,y_train=list_to_array_train(train_images,train_labels)
print(X_train.shape)
print(y_train.shape)