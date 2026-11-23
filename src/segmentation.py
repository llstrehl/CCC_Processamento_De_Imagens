import cv2
import numpy as np

def segment_grains(image, dark_background=True, dist_factor=0.52, min_area=50):

    #Suaviza regiões de cor parecida, preservando as bordas (reduz ruído)
    shifted = cv2.pyrMeanShiftFiltering(image, 5, 50)

    #cinza + normalização
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    normalized_image = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    #thresholding, dejo os graos brancos
    modo = cv2.THRESH_BINARY if dark_background else cv2.THRESH_BINARY_INV
    ret, thresh = cv2.threshold(normalized_image, 0, 255, modo | cv2.THRESH_OTSU)

    
    kernel = np.ones((3,3),np.uint8)
    #elimina manchas de ruido branco
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)

    # Dilata os grãos: o que ficar fora disso é fundo com certeza
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Distância de cada pixel ao fundo; fica só o centro de cada grão (longe das bordas)    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, dist_factor*dist_transform.max(), 255, 0)

    #Franja onde nao se sabe se e fundo ou grao
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    #Enumera os marcadores, 0 som os que se desconocem
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers+1
    markers[unknown==255] = 0
    #Separa os graos pegados com -1
    markers = cv2.watershed(image,markers)

    #Para cada marcador, cria uma mascara e encontra o contorno do grao
    contours = []
    for label in np.unique(markers):
        if label <= 1:      # -1 bordes, 0 sin segmentar, 1 fondo
            continue
        mask = np.uint8(markers == label) * 255
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cs, key=cv2.contourArea)
        if cv2.contourArea(c) >= min_area:
            contours.append(c)
    return contours, markers

if __name__ == "__main__":
    img = cv2.imread("images/sementes_teste.jpg")
    cnts, _ = segment_grains(img)
    out = img.copy()
    cv2.drawContours(out, cnts, -1, (0, 255, 0), 2)
    print(len(cnts), "granos")
    cv2.imshow("resultado", out)
    cv2.waitKey(0)

