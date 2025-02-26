import cv2
import numpy as np

# Carrega o modelo YOLO pré-treinado
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
# net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Define as classes de veículos que queremos detectar
vehicle_classes = ["car", "truck", "bus", "motorcycle", "bicycle"]

# Obtém as camadas de saída
layer_names = net.getLayerNames()

# Corrige o acesso às camadas de saída
output_layers = []
for i in net.getUnconnectedOutLayers():
    if isinstance(i, np.ndarray):  # Verifica se é um array numpy (OpenCV >= 4.0)
        output_layers.append(layer_names[i[0] - 1])
    else:  # Caso contrário, é um número inteiro (OpenCV < 4.0)
        output_layers.append(layer_names[i - 1])

# Captura do vídeo
url = "https://camera1.pr280.com.br/index.m3u8"
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Erro ao abrir o stream.")
    exit()

# Define cores para cada tipo de veículo
colors = {
    "car": (0, 255, 0),       # Verde para carros
    "truck": (255, 0, 0),      # Azul para caminhões
    "bus": (0, 0, 255),        # Vermelho para ônibus
    "motorcycle": (255, 255, 0), # Amarelo para motos
    "bicycle": (0, 255, 255)   # Ciano para bicicletas
}

frame_count = 0
skip_frames = 2  # Processa 1 frame a cada 3 (ajuste conforme necessário)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 360))  # Reduz a resolução para 640x360
    if not ret:
        print("Erro ao capturar frame.")
        break

    frame_count += 1
    if frame_count % skip_frames != 0:
        continue  # Pula o frame

    # Redimensiona o frame para o tamanho esperado pelo YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Processa as detecções
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2 and classes[class_id] in vehicle_classes:  # Filtra apenas veículos
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Aplica a supressão de não-máximos para remover detecções redundantes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Desenha as caixas delimitadoras
    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            x, y, w, h = box
            class_name = classes[class_ids[i]]
            color = colors.get(class_name, (0, 0, 0))  # Usa a cor correspondente ao tipo de veículo
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{class_name} {confidences[i]:.2f}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Exibe o frame
    cv2.imshow('Frame', frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()