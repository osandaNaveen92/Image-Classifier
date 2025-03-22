import pickle
model16.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'],run_eagerly=True)
pickle.dump(model16, open('model16.pkl','wb'))
train_feature_16=model16.predict(X_train)
test_feature_16=model16.predict(X_test)

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf=rf.fit(train_feature_16,y_train)
train_pred=rf.predict(train_feature_16)
test_pred=rf.predict(test_feature_16)
print("Train Accuracy Score",accuracy_score(train_pred,y_train))
print("Test Accuracy Score",accuracy_score(test_pred,y_test))
fig = plt.figure(1)
plt.figure(figsize=(1,1))
plt.title("Confusion Matrix")
cm = confusion_matrix(y_test.argmax(axis=1),test_pred.argmax(axis=1))
sns.heatmap(cm,cmap="Blues",cbar=True, annot=True,annot_kws={"size": 12})