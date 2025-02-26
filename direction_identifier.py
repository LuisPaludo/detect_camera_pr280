import cv2
import imutils
from imutils.video import VideoStream

# URL do stream HLS
url = "https://camera1.pr280.com.br/index.m3u8"

# Inicializa o vídeo stream
vs = VideoStream(url).start()

# Inicializa o subtrador de fundo
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    frame = vs.read()
    if frame is None:
        break

    # Redimensiona o frame para melhor performance
    frame = imutils.resize(frame, width=500)

    # Aplica a subtração de fundo
    fgmask = fgbg.apply(frame)

    # Aplica uma limiarização para remover ruídos
    thresh = cv2.threshold(fgmask, 25, 255, cv2.THRESH_BINARY)[1]

    # Encontra contornos na imagem limiarizada
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Define as regiões de interesse (ROIs)
    height, width = frame.shape[:2]
    roi1 = (0, 0, width, height // 2)  # Sentido 1
    roi2 = (0, height // 2, width, height // 2)  # Sentido 2

    # Desenha os contornos detectados e verifica o sentido
    for contour in contours:
        if cv2.contourArea(contour) < 500:  # Filtra contornos pequenos
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        if y < height // 2:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Sentido 1
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Sentido 2

    # Exibe o frame com os retângulos desenhados
    cv2.imshow("Frame", frame)
    cv2.imshow("FG Mask", fgmask)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.stop()
cv2.destroyAllWindows()