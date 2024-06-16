from keras.models import load_model

print('Trying to load model')
m = load_model('media/model.h5')
print(m)
print('model loaded')
cl = m.predict([[[2.035,-4.45,-5.06]]])
print(cl)