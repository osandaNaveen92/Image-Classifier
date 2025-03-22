def keras_to_categorical(y_train,y_test):
  return to_categorical(y_train),to_categorical(y_test)

y_train,y_test=keras_to_categorical(y_train,y_test)
y_train.shape,y_test.shape

#Use a pretrained VGG-16 model on Imagenet dataset by removing the top fully 
#connected layers and adding three dense layers having 64, 32 and 2 neurons with 
#relu, sigmoid and softmax activation functions respectively for classifying the two 
#class Skin_Cancer RGB dataset

def model_vgg16():
  VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224, 3))
  #Make loaded layers as non-trainable. This is important as we want to work with pre-trained weights
  for layer in VGG_model.layers:
    layer.trainable = False #True for actual transfer learning
  feature=keras.layers.GlobalAveragePooling2D()(VGG_model.output)
  d1=Dense(units=64,kernel_initializer="glorot_uniform", activation='relu')(feature)
  d2=Dense(units=32,kernel_initializer="glorot_uniform", activation='sigmoid')(d1)
  d3=Dense(units=2,kernel_initializer="glorot_uniform", activation='softmax')(d2)
  output = Model(inputs =VGG_model.input, outputs =d3)
  
  return output

model16=model_vgg16()
model16.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'],run_eagerly=True)
history = model16.fit(X_train, y_train, validation_split=0.2,epochs= 10, batch_size= 1, verbose=1,validation_data=(X_test,y_test))