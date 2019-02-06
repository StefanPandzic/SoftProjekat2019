import cv2
import os
import numpy as np
import math
import pripremazamrezu

# keras
# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.models import load_model

dataset = load_model('my_model.h5')
idx = 0
isecenibr = []
sumaniz = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def izoliraj_brojeve(slika):
    gornja_granica = np.array([199, 199, 199], dtype="uint8")
    donja_granica = np.array([255, 255, 255], dtype="uint8")
    mask = cv2.inRange(slika, gornja_granica, donja_granica)

    brojevi = cv2.bitwise_and(slika, slika, mask=mask)
    siva = prebaci_u_sivu(brojevi)

    blur = cv2.GaussianBlur(siva, (5, 5), 0)
    return blur


def prikazi_sumu(image, sum_string):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (20, 450)
    fontScale = 2
    fontColor = (0, 255, 0)
    lineType = 1

    cv2.putText(image, sum_string, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)


#geometrija
def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def altitude(a, b, c):
    s = (a + b + c) / 2
    return 2 * math.sqrt(s * (s - a) * (s - b) * (s - c)) / a

def na_liniji(x1, y1, x2, y2, x, y):
    if x2 > x and x > x1 and y2 < y and y < y1:
        return True
    return False

#konture
def konture_brojeva(slika):
    konture, proba = cv2.findContours(slika.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rezultat = []
    for kontura in konture:
        (x, y, w, h) = cv2.boundingRect(kontura)
        area = cv2.contourArea(kontura)
        if h > 12 and area > 30 and area < 1000:
            kordinate = (x, y, w, h)
            rezultat.append(kordinate)
    return rezultat

def konture_linije(slika):
    lines = cv2.HoughLinesP(slika, 1, np.pi / 180, 100, 100, 10)

    kordinate = []
    maksimalna_udaljenost = 0

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                distan = distance(x1, y1, x2, y2)
                if distan > maksimalna_udaljenost:
                    kordinate = []
                    maksimalna_udaljenost = distan
                    kordinate.append((x1, y1))
                    kordinate.append((x2, y2))

    return kordinate

def prebaci_u_sivu(slika):
    return cv2.cvtColor(slika, cv2.COLOR_BGR2GRAY)

#Proba
def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret,image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin
def invert(image):
    return 255-image
def resize_region(region):
    return cv2.resize(region,(30,30), interpolation = cv2.INTER_NEAREST)
def select_roi(image_orig, image_bin):
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = [] # lista sortiranih regiona po x osi (sa leva na desno)
    regions_array = []
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour) #koordinate i velicina granicnog pravougaonika
        area = cv2.contourArea(contour)
        if area > 30 and h < 40 and h > 15 and w > 10:
            # kopirati [y:y+h+1, x:x+w+1] sa binarne slike i smestiti u novu sliku
            # oznaciti region pravougaonikom na originalnoj slici (image_orig) sa rectangle funkcijom
            #kontura2 = cv2.drawContours( blank.copy(), contour, 1, 1 )
            region = image_bin[y:y+h+1,x:x+w+1]
            regions_array.append([resize_region(region), (x,y,w,h)])
            cv2.line(image_orig,(x,y+h),(x+w,y),(0,255,0),2)
            #rec = tuple(x, y,)
            #xv = presek(kontura2)
            #if xv:
            #    cv2.rectangle(image_orig,(x,y),(x+w,y+h),(0,0,255),2)

            #x,y,w,h = current
            #cv2.rectangle(image_orig, (x,y), (x+w,y+h), 255,2)
            #ret, current = cv2.meanShift(img, (x,y,w,h), term_crit)
    regions_array = sorted(regions_array, key=lambda item: item[1][0])
    sorted_regions = sorted_regions = [region[0] for region in regions_array]
    sorted_rectangles = [region[1] for region in regions_array]
    region_distances = []
    # Izdvojiti sortirane parametre opisujucih pravougaonika
    # Izracunati rastojanja izmedju svih susednih regiona po x osi i dodati ih u region_distances niz
    for index in range(0, len(sorted_rectangles)-1):
        current = sorted_rectangles[index]
        next_rect = sorted_rectangles[index+1]
        distance = next_rect[0] - (current[0]+current[2]) #X_next - (X_current + W_current)
        region_distances.append(distance)
    return image_orig, sorted_regions, sorted_rectangles, region_distances
#

def pracenje_brojeva(x, y, brojevi_prosli, frejm):

    prosli = False

    if len(brojevi_prosli) > 0:
        for broj_prosao in brojevi_prosli:
            if x == broj_prosao[0] or y == broj_prosao[1]:
                prosli = True

            if prosli == False:
                if distance(x, y, broj_prosao[0], broj_prosao[1]) < 18:
                    if (frejm - broj_prosao[2]) < 20:
                        prosli = True

    if prosli == False:
        brojevi_prosli.append((x, y, frejm))
        return True
    return False

def predvidi_brojeve(slika, dataset):
    niz = []
    niz.append(slika)
    niz = pripremazamrezu.priprema_za_mrezu(niz)
    predvidi = np.array(niz, np.float32)
    predvidi_rezultat = dataset.predict_classes(predvidi)

    return predvidi_rezultat[0]

for idx in range(10):

    cap = cv2.VideoCapture('Video/video-' + str(idx) + '.avi')
    prvi_frejm = cap.read() [1]

    siva = prebaci_u_sivu(prvi_frejm)
    lista_kordinata = konture_linije(siva)

    print('Loading video-' + str(idx) + '.avi.')

    x1 = lista_kordinata[0][0]
    y1 = lista_kordinata[0][1]
    x2 = lista_kordinata[1][0]
    y2 = lista_kordinata[1][1]

    brojevi_prosli = []
    broj_frejmova = 0

    while cap.isOpened():
        ret, frejm = cap.read()
        broj_frejmova += 1
        cv2.line(frejm, (x1, y1), (x2, y2), (0, 0, 255), 2)

        if not ret:
            break
        izolirano = izoliraj_brojeve(frejm)
        lista_kordinata = konture_brojeva(izolirano)

        for kordinate in lista_kordinata:
            (x, y, w, h) = kordinate
            cv2.rectangle(frejm, (x, y), (x + w, y + h), (0, 255, 0), 1)

            get = pripremazamrezu.get_broj(izolirano, x, y, h, w)
            zamenjeno = pripremazamrezu.zameni_belu_i_crnu(get)

            a = distance(x1, y1, x2, y2)
            b = distance(x1, y1, x, y)
            c = distance(x2, y2, x, y)
            h = altitude(a, b, c)

            if na_liniji(x1, y1, x2, y2, x, y):

                if h < 3:
                    if pracenje_brojeva(x, y, brojevi_prosli, broj_frejmova):
                        broj = predvidi_brojeve(zamenjeno, dataset)
                        sumaniz[idx] = sumaniz[idx] + broj

        prikazi_sumu(frejm, str(sumaniz[idx]))
        cv2.imshow("video-" + str(idx), frejm)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()


print('\n \nSacuvavanje rezultata u fajl.')
os.remove("out.txt")
file = open('out.txt', 'w+')
file.write('RA 48/2013 Stefan Pandzic \n')
file.write('suma')
for index in range(10):
    file.write('\n' + 'video-' + str(index) + '.avi' + '\t' + str(sumaniz[index]))
file.close()
print('Uspesno sacuvano u fajl.')