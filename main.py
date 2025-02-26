import cv2

# URL do stream HLS
url = "https://camera1.pr280.com.br/index.m3u8"

# Captura do vídeo
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Erro ao abrir o stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar frame.")
        break

    # Exibe o frame
    cv2.imshow('Frame', frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()