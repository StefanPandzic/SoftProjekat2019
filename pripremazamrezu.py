import cv2
import numpy as np



def konverzija_u_bin(slika):
    return slika / 255

def matrix_to_vector(data):
    return data.flatten()

def zameni_belu_i_crnu(slika):
    zameni = cv2.subtract(255, slika)

    kernel = np.ones((3, 3))
    img = cv2.erode(zameni, kernel, iterations=1)
    img = cv2.dilate(img, kernel, iterations=1)

    ret, thresh = cv2.threshold(zameni, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return thresh

def get_broj(slika, x, y, h, w):

    isecena = slika[y:y + h, x:x + w]
    promenavelicine = cv2.resize(isecena, (28, 28), interpolation=cv2.INTER_NEAREST)

    return promenavelicine

def priprema_za_mrezu(numbers):
    priprema = []

    for number in numbers:
        scaled = konverzija_u_bin(number)
        priprema.append(matrix_to_vector(scaled))
        priprema = np.array(priprema, np.float32)

    return priprema

def convert_output(alphabet):
    nn_outputs = []

    for index in range(len(alphabet)):
        output = np.zeros(len(alphabet))
        output[index] = 1
        nn_outputs.append(output)

    return np.array(nn_outputs)