import mreza
import pripremazamrezu
from keras.datasets import mnist
#from matplotlib import pyplot as plt

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#print('Showing image example.')
#first_image = train_images[0]
#first_image = pripremazamrezu.zameni_belu_i_crnu(first_image)
#plt.imshow(first_image, cmap='gray')
#plt.show()
#print('')

train = []
print('Priprema podataka za treniranje.')
for i in range(len(train_images)):
    train.append(pripremazamrezu.zameni_belu_i_crnu(train_images[i]))

test = []
print('Priprema podataka za testiranje.')
for i in range(len(test_images)):
    test.append(pripremazamrezu.zameni_belu_i_crnu(test_images[i]))


print('Pripremanje ulaza za treniranje.')
inputs_train = pripremazamrezu.priprema_za_mrezu(train)

print('Pripremanje ulaza za testiranje')
inputs_test = pripremazamrezu.priprema_za_mrezu(test)

print('Pravljenje mreze.')
net = mreza.napravi_mrezu()

print('Treniranje mreze.')
net = mreza.treniraj_mrezu(net, inputs_train, train_labels, inputs_test, test_labels)

print('\n \nSacuvavanje mreze.')
net.save('my_model.h5')
print('Mreza je sacuvana uspesno.')