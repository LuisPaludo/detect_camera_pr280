import cv2
import numpy as np
import time
from scipy.spatial import distance

class Config:
    def __init__(self):
        self.net = self.load_yolo_model()
        self.classes = self.load_classes()
        self.vehicle_indices = self.define_vehicle_indices()
        self.output_layers = self.get_output_layers()
        self.colors = self.define_colors()

    @staticmethod
    def load_yolo_model():
        return cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

    @staticmethod
    def load_classes():
        with open("coco.names", "r") as file:
            return file.read().strip().split("\n")

    @staticmethod
    def define_colors():
        return {
            "car": (0, 255, 0),        # Green
            "truck": (255, 0, 0),      # Blue
            "bus": (0, 0, 255),        # Red
        }

    def define_vehicle_indices(self):
        vehicle_classes = ["car", "truck", "bus"]
        return [self.classes.index(vehicle_class) for vehicle_class in vehicle_classes if vehicle_class in self.classes]

    def get_output_layers(self):
        layer_names = self.net.getLayerNames()
        unconnected_layers = self.net.getUnconnectedOutLayers()
        output_layers = []
        for i in unconnected_layers:
            output_layers.append(layer_names[i - 1])
        return output_layers


class VehicleTracker:
    def __init__(self, max_disappeared=30, max_distance=50):
        # Contador de objetos e dicionário para rastrear objetos
        self.next_object_id = 1
        self.objects = {}  # ID: (centroide, classe_id, confiança, contador_desaparecimentos, caixa)
        self.disappeared = {}  # ID: contador_desaparecimentos

        # Número máximo de frames consecutivos em que um objeto pode estar
        # marcado como desaparecido antes de ser excluído
        self.max_disappeared = max_disappeared

        # Distância máxima entre centroides para considerar o mesmo objeto
        self.max_distance = max_distance

    def register(self, centroid, class_id, confidence, box):
        # Registra um novo objeto com o próximo ID disponível
        self.objects[self.next_object_id] = (centroid, class_id, confidence, 0, box)
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        # Remove um objeto do rastreamento
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, detections):
        # Atualize o rastreador com novas detecções
        # detections é uma lista de tuplas (caixa, confiança, class_id)

        # Se não houver detecções, marque todos os objetos como desaparecidos
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1

                # Se excedeu o número máximo de frames desaparecidos, remova-o
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            # Não há nada mais a fazer
            return self.objects

        # Inicializa uma matriz de centroides atuais
        input_centroids = []
        input_info = []  # (class_id, confidence, box)

        # Loop sobre as detecções
        for box, confidence, class_id in detections:
            # Extrai as coordenadas da caixa
            x, y, w, h = box

            # Calcula o centroide
            cx = int(x + w / 2.0)
            cy = int(y + h / 2.0)
            centroid = (cx, cy)

            # Adiciona à lista de centroides e informações
            input_centroids.append(centroid)
            input_info.append((class_id, confidence, box))

        # Se não temos objetos rastreados, registre todos
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], *input_info[i])

        # Caso contrário, tente associar os centroides de entrada aos objetos existentes
        else:
            # Obtém os IDs e centroides dos objetos rastreados
            object_ids = list(self.objects.keys())
            object_centroids = [self.objects[id][0] for id in object_ids]

            # Calcula a distância entre cada par de centroides de objeto e de entrada
            D = distance.cdist(np.array(object_centroids), np.array(input_centroids))

            # Para encontrar a menor distância para cada linha, ordene as linhas
            rows = D.min(axis=1).argsort()

            # Para encontrar a menor distância para cada coluna, ordene as colunas
            cols = D.argmin(axis=1)[rows]

            # Controla quais linhas e colunas já examinamos
            used_rows = set()
            used_cols = set()

            # Loop sobre as combinações (linha, coluna) de índices
            for (row, col) in zip(rows, cols):
                # Se já examinamos esta linha ou coluna, ignore
                if row in used_rows or col in used_cols:
                    continue

                # Se a distância for maior que o máximo, ignore
                if D[row, col] > self.max_distance:
                    continue

                # Obtém o ID do objeto
                object_id = object_ids[row]

                # Atualiza o centroide, classe e confiança
                centroid = input_centroids[col]
                class_id, confidence, box = input_info[col]

                # Atualiza o objeto e reseta a contagem de desaparecimentos
                self.objects[object_id] = (centroid, class_id, confidence, 0, box)
                self.disappeared[object_id] = 0

                # Indica que examinamos esta linha e coluna
                used_rows.add(row)
                used_cols.add(col)

            # Calcula as linhas e colunas que ainda não foram examinadas
            unused_rows = set(range(D.shape[0])) - used_rows
            unused_cols = set(range(D.shape[1])) - used_cols

            # Se o número de objetos rastreados >= número de objetos de entrada
            # verifique quais deles desapareceram
            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    # Obtém o ID do objeto
                    object_id = object_ids[row]

                    # Incrementa a contagem de desaparecimentos
                    self.disappeared[object_id] += 1

                    # Se excedeu o número máximo de frames desaparecidos, remova-o
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)

            # Caso contrário, registre novos objetos para as detecções não utilizadas
            else:
                for col in unused_cols:
                    self.register(input_centroids[col], *input_info[col])

        # Retorna o conjunto de objetos rastreados
        return self.objects

def process_frame(frame, net, output_layers, vehicle_indices, conf_threshold=0.5, nms_threshold=0.4):
    """Processa um único frame para detecção de veículos"""
    # Redimensionamento para entrada do modelo (reduzindo para 320x320 para maior velocidade)
    height, width = frame.shape[:2]
    model_input_size = (320, 320)  # Reduzido de 416x416 para melhorar performance

    # Pré-processamento mais eficiente
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, model_input_size, swapRB=True, crop=False)
    net.setInput(blob)

    # Forward pass
    outs = net.forward(output_layers)

    # Inicializa listas para resultados
    class_ids = []
    confidences = []
    boxes = []

    # Processamento de detecções com filtragem prévia por classe e confiança
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filtra apenas veículos com confiança significativa
            if confidence > conf_threshold and class_id in vehicle_indices:
                # Converte coordenadas da caixa para escala da imagem atual
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = max(0, int(center_x - w / 2))
                y = max(0, int(center_y - h / 2))

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Aplicação de NMS (Non-Maximum Suppression)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    results = []
    if len(indices) > 0:
        indices = indices.flatten()
        for i in indices:
            results.append((boxes[i], confidences[i], class_ids[i]))

    return results

def main():
    # Captura de vídeo
    url = "https://camera1.pr280.com.br/index.m3u8"
    cap = cv2.VideoCapture(url)
    config = Config()

    if not cap.isOpened():
        print("Erro ao abrir o stream.")
        return

    # Configura o FPS do vídeo original (se disponível)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0 or original_fps > 60:  # Valores inválidos ou irreais
        original_fps = 30.0  # Valor padrão para streaming

    print(f"FPS original do vídeo: {original_fps}")

    # Configurações para otimização
    process_width = 512  # Reduzido para processamento
    skip_frames = 4      # Processa 1 frame a cada 3 (reduzido para melhor rastreamento)

    # Parâmetros de detecção
    conf_threshold = 0.3  # Aumentado para reduzir falsos positivos
    nms_threshold = 0.4

    # Variáveis para medição de desempenho
    frame_count = 0
    start_time = time.time()
    fps_update_interval = 30  # Atualiza FPS a cada 30 frames processados
    processed_frames = 0

    # Inicializa o rastreador de veículos
    tracker = VehicleTracker(max_disappeared=10, max_distance=80)

    # Tempo de espera fixo para cada frame (em ms)
    wait_time = 1  # 1ms de espera

    # Gera cores aleatórias para IDs
    np.random.seed(42)  # Para reprodutibilidade
    id_colors = {}

    print("Iniciando processamento de vídeo...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Erro ao capturar frame.")
                break

            frame_count += 1

            # Cria uma cópia do frame para desenho
            display_frame = frame.copy()

            # Processa apenas 1 a cada skip_frames
            if frame_count % skip_frames == 0:
                process_frame_resized = cv2.resize(
                    frame, (
                        process_width, int(frame.shape[0] * process_width / frame.shape[1])
                    )
                )

                # Faz a detecção
                detections = process_frame(
                    process_frame_resized,
                    config.net,
                    config.output_layers,
                    config.vehicle_indices,
                    conf_threshold,
                    nms_threshold
                )

                # Ajusta as coordenadas das caixas para o tamanho original
                scale_x = frame.shape[1] / process_frame_resized.shape[1]
                scale_y = frame.shape[0] / process_frame_resized.shape[0]

                scaled_detections = []
                for box, confidence, class_id in detections:
                    x, y, w, h = box
                    x = int(x * scale_x)
                    y = int(y * scale_y)
                    w = int(w * scale_x)
                    h = int(h * scale_y)
                    scaled_detections.append(([x, y, w, h], confidence, class_id))

                # Atualiza o rastreador com as novas detecções
                objects = tracker.update(scaled_detections)
                processed_frames += 1
            else:
                # Usa os objetos já rastreados (sem atualização)
                objects = tracker.objects

            # Desenha os objetos rastreados
            for object_id, (centroid, class_id, confidence, disappear_count, box) in objects.items():
                # Obtém a classe e a cor
                class_name = config.classes[class_id]
                class_color = config.colors.get(class_name, (0, 0, 0))

                # Gera uma cor única para este ID (se ainda não existir)
                if object_id not in id_colors:
                    id_colors[object_id] = (
                        np.random.randint(0, 255),
                        np.random.randint(0, 255),
                        np.random.randint(0, 255)
                    )
                id_color = id_colors[object_id]

                # Extrai as coordenadas da caixa
                x, y, w, h = box

                # Desenha a caixa delimitadora
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), class_color, 2)

                # Cria um rótulo com ID, tipo e confiança
                label = f"ID:{object_id} {class_name} {confidence:.2f}"

                # Adiciona um fundo ao texto para melhor visibilidade
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(display_frame, (x, y - 25), (x + label_size[0], y), id_color, -1)

                # Desenha o texto
                cv2.putText(display_frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Desenha o centroide
                cv2.circle(display_frame, centroid, 4, id_color, -1)

            # Adiciona contador de FPS
            if processed_frames > 0:
                elapsed_time = time.time() - start_time
                fps = processed_frames / elapsed_time
                display_fps = frame_count / elapsed_time

                cv2.putText(display_frame, f"Proc FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(display_frame, f"Disp FPS: {display_fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Veículos: {len(objects)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

            # Exibe o frame
            cv2.imshow('Detecção e Rastreamento de Veículos', display_frame)

            # Espere um tempo fixo curto (1ms) para permitir que o vídeo flua naturalmente
            if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                break

    finally:
        # Libera recursos
        cap.release()
        cv2.destroyAllWindows()
        print(f"Processamento finalizado:")
        print(f"- Veículos únicos detectados: {tracker.next_object_id - 1}")
        print(f"- Frames processados: {processed_frames}")
        print(f"- Frames totais: {frame_count}")
        print(f"- Tempo total: {time.time() - start_time:.2f} segundos")
        if processed_frames > 0:
            print(f"- FPS processamento: {processed_frames / (time.time() - start_time):.2f}")
        print(f"- FPS exibição: {frame_count / (time.time() - start_time):.2f}")

if __name__ == "__main__":
    main()