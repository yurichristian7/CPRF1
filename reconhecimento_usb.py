import os
import cv2
import numpy as np
from datetime import datetime, time
import mysql.connector
from mysql.connector import Error

# ================= CONFIGURAÇÕES DO BANCO DE DADOS =================
DB_CONFIG = {
    'host': 'localhost',  # ou o IP do seu servidor MySQL
    'user': 'seu_usuario',
    'password': 'sua_senha',
    'database': 'cprf'
}

# ================= CONFIGURAÇÕES =================
# Cascades
face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
eye_cascade_path = cv2.data.haarcascades + "haarcascade_eye.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

# Diretórios
log_dir = "logs"
faces_dir = "rostos_salvos"
dataset_dir = "dataset"  # Alterado para dataset_dir 
os.makedirs(log_dir, exist_ok=True)
os.makedirs(faces_dir, exist_ok=True)
os.makedirs(dataset_dir, exist_ok=True)

already_logged_today = set()
FACE_SIZE = (100, 100)
CONFIDENCE_THRESHOLD = 80  # Ajuste inicial
start_time = time(7, 0, 0)
end_time = time(12, 0, 0)

# Redimensionamento para processamento mais rápido
PROCESSING_WIDTH = 640

# Variáveis globais
recognizer = None
label_map = {}

# ================= FUNÇÕES DO BANCO DE DADOS =================
def create_connection():
    """Cria conexão com o banco de dados"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            return connection
    except Error as e:
        print(f"Erro ao conectar ao MySQL: {e}")
    return None

def create_tables():
    """Cria as tabelas necessárias no banco de dados"""
    connection = create_connection()
    if connection is None:
        return False
        
    try:
        cursor = connection.cursor()
        
        # Tabela de administradores (conforme especificada)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS `administradores` (
                `id_admin` int(11) NOT NULL,
                `usuario` varchar(50) NOT NULL,
                `senha` varchar(255) NOT NULL,
                `ultimo_login` datetime DEFAULT NULL,
                PRIMARY KEY (`id_admin`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci
        """)
        
        connection.commit()
        print("Tabela de administradores verificada/criada com sucesso!")
        return True
        
    except Error as e:
        print(f"Erro ao criar tabela: {e}")
        return False
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def get_person_id(nome):
    """Obtém o ID da pessoa no banco de dados"""
    connection = create_connection()
    if connection is None:
        return None
        
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT id_admin FROM administradores WHERE usuario = %s", (nome,))
        result = cursor.fetchone()
        return result[0] if result else None
    except Error as e:
        print(f"Erro ao buscar administrador: {e}")
        return None
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def insert_person(nome):
    """Insere uma nova pessoa no banco de dados"""
    connection = create_connection()
    if connection is None:
        return None
        
    try:
        cursor = connection.cursor()
        # Para a tabela administradores, precisamos definir um id_admin
        # Vamos usar o próximo ID disponível
        cursor.execute("SELECT MAX(id_admin) FROM administradores")
        result = cursor.fetchone()
        next_id = 1 if result[0] is None else result[0] + 1
        
        cursor.execute("INSERT INTO administradores (id_admin, usuario, senha) VALUES (%s, %s, %s)", 
                      (next_id, nome, 'senha_temp'))
        connection.commit()
        return next_id
    except Error as e:
        print(f"Erro ao inserir administrador: {e}")
        return None
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def save_recognition_to_db(nome, confianca, coordenada_x, coordenada_y, arquivo_rosto=None):
    """Salva o registro de reconhecimento no banco de dados"""
    # Para a tabela administradores, não temos uma tabela de registros de reconhecimento
    # Esta função será adaptada para trabalhar com a estrutura existente
    connection = create_connection()
    if connection is None:
        return False
        
    try:
        cursor = connection.cursor()
        
        # Para a tabela administradores, vamos apenas atualizar o último login
        # se for um administrador conhecido
        if nome.lower() != "desconhecido":
            pessoa_id = get_person_id(nome)
            if pessoa_id is not None:
                cursor.execute("""
                    UPDATE administradores 
                    SET ultimo_login = %s 
                    WHERE id_admin = %s
                """, (datetime.now(), pessoa_id))
                connection.commit()
                return True
        
        return False
        
    except Error as e:
        print(f"Erro ao salvar reconhecimento: {e}")
        return False
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

# ================= FUNÇÕES EXISTENTES (MODIFICADAS) =================
def log_recognition(name, confidence, face_img, x, y):
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    current_time = now.time()
    if not (start_time <= current_time <= end_time):
        return

    unique_key = f"{today_str}{name.lower()}{x}{y}{now.strftime('%H-%M-%S')}"
    if unique_key in already_logged_today:
        return
    already_logged_today.add(unique_key)

    if name.lower() == "desconhecido":
        filename = f"desconhecidos_{today_str}.txt"
    else:
        filename = f"log_{today_str}.txt"

    log_path = os.path.join(log_dir, filename)
    with open(log_path, "a", encoding='utf-8') as f:
        f.write(f"{now.strftime('%H:%M:%S')} - {name} (Confiança: {int(confidence)})\n")

    person_dir = os.path.join(faces_dir, name.lower())
    os.makedirs(person_dir, exist_ok=True)
    resized_face = cv2.resize(face_img, FACE_SIZE)
    face_filename = f"{today_str}{now.strftime('%H-%M-%S')}{name.lower()}{x}{y}.png"
    face_path = os.path.join(person_dir, face_filename)
    cv2.imwrite(face_path, resized_face)
    
    # ============ NOVO: Salvar no banco de dados ============
    save_recognition_to_db(name, confidence, x, y, face_path)

def get_images_and_labels():
    labels, faces = [], []
    label_map = {}
    next_id = 0

    if not os.path.exists(dataset_dir):
        print(f"AVISO: Diretório '{dataset_dir}' não encontrado.")
        return faces, labels, label_map

    # Lista de pessoas conhecidas (Yuri, Thalles, Daniel, Arthur)
    pessoas_conhecidas = ["Yuri", "Thalles", "Daniel", "Arthur"]
    
    # ============ NOVO: Sincronizar com o banco de dados ============
    for person_name in pessoas_conhecidas:
        # Verificar se a pessoa existe no banco, se não, inserir
        if get_person_id(person_name) is None:
            insert_person(person_name)
            print(f"Pessoa '{person_name}' adicionada ao banco de dados")
    
    for person_name in pessoas_conhecidas:
        person_dir = os.path.join(dataset_dir, person_name)
        if not os.path.exists(person_dir):
            print(f"AVISO: Diretório '{person_name}' não encontrado em '{dataset_dir}'")
            continue
            
        label_map[next_id] = person_name
        image_files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        processed_count = 0

        for filename in image_files:
            img_path = os.path.join(person_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
                
            # As imagens do dataset já devem ser rostos, então apenas redimensionamos
            img = cv2.equalizeHist(img)
            face_resized = cv2.resize(img, FACE_SIZE)
            faces.append(face_resized)
            labels.append(next_id)
            processed_count += 1

        print(f"{person_name}: {processed_count} rostos processados")
        next_id += 1

    print(f"Total de rostos: {len(faces)} | Total de pessoas: {len(label_map)}")
    return faces, labels, label_map

def train_recognizer():
    print("Treinando LBPH...")
    faces, labels, label_map = get_images_and_labels()
    if len(faces) == 0:
        print("AVISO: Nenhuma face para treinar! Use a opção 'A' para adicionar pessoas.")
        return None, {}
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    print("Treinamento concluído!")
    return recognizer, label_map

def add_new_person():
    person_name = input("Digite o nome da nova pessoa: ").strip()
    if not person_name:
        print("Nome inválido!")
        return
    
    person_dir = os.path.join(dataset_dir, person_name)  # Alterado para pessoas_dir
    os.makedirs(person_dir, exist_ok=True)
    
    # ============ NOVO: Adicionar pessoa ao banco de dados ============
    person_id = insert_person(person_name)
    if person_id:
        print(f"Pessoa '{person_name}' adicionada ao banco de dados com ID: {person_id}")
    
    print(f"Coletando imagens para {person_name}. Posicione-se frente à câmera.")
    print("Pressione ESPAÇO para capturar ou 'q' para sair")
    print("Dica: Fique a ~50cm da câmera, com boa iluminação no rosto")
    
    cam = cv2.VideoCapture(1)  # Alterado para câmera USB (índice 1)
    count = 0
    max_count = 20  # Número de imagens a capturar
    
    while count < max_count:
        ret, frame = cam.read()
        if not ret:
            break
            
        # Redimensionar para processamento mais rápido
        height, width = frame.shape[:2]
        new_width = PROCESSING_WIDTH
        new_height = int((new_width / width) * height)
        frame_resized = cv2.resize(frame, (new_width, new_height))
        
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        
        # Ajustar parâmetros para câmera de notebook (mais sensível)
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(60, 60),  # Tamanho menor para câmera próxima
            maxSize=(300, 300)  # Limitar tamanho máximo
        )
        
        face_detected = False
        for (x, y, w, h) in faces:
            # Calcular o tamanho do rosto em relação à imagem
            face_ratio = w / new_width
            
            # Verificar se o rosto está em uma posição e tamanho adequados
            if 0.15 <= face_ratio <= 0.4:  # Rosto nem muito grande nem muito pequeno
                cv2.rectangle(frame_resized, (x, y), (x+w, y+h), (0, 255, 0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                
                # Verificar se há olhos para garantir que é um rosto
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
                if len(eyes) >= 1:
                    face_detected = True
                    # Mostrar instrução para capturar
                    cv2.putText(frame_resized, "ROSTO DETECTADO - ESPACO para capturar", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Mostrar dica de distância
                    if face_ratio < 0.2:
                        cv2.putText(frame_resized, "CHEGUE MAIS PERTO", (x, y-40), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                    elif face_ratio > 0.3:
                        cv2.putText(frame_resized, "DISTANCIA BOA", (x, y-40), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    cv2.putText(frame_resized, "OLHOS NAO DETECTADOS", (x, y-40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                # Rosto muito grande ou muito pequeno
                cv2.rectangle(frame_resized, (x, y), (x+w, y+h), (0, 165, 255), 2)
                if face_ratio < 0.15:
                    cv2.putText(frame_resized, "CHEGUE MAIS PERTO", (x, y-40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                else:
                    cv2.putText(frame_resized, "AFASTE-SE UM POUCO", (x, y-40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        
        cv2.putText(frame_resized, f"Capturadas: {count}/{max_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if not face_detected:
            cv2.putText(frame_resized, "Posicione seu rosto na camera", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame_resized, "Distancia ideal: ~50cm", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        cv2.imshow("Coletando amostras", frame_resized)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Espaço para capturar
            if face_detected:
                for (x, y, w, h) in faces:
                    roi_gray = gray[y:y+h, x:x+w]
                    eyes = eye_cascade.detectMultiScale(roi_gray)
                    if len(eyes) >= 1:
                        # Salvar apenas a região do rosto
                        face_img = cv2.resize(roi_gray, FACE_SIZE)
                        img_path = os.path.join(person_dir, f"{person_name}_{count}.jpg")
                        cv2.imwrite(img_path, face_img)
                        count += 1
                        print(f"Imagem {count}/{max_count} capturada")
                        # Feedback visual
                        cv2.putText(frame_resized, "IMAGEM CAPTURADA!", (x, y-60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.imshow("Coletando amostras", frame_resized)
                        cv2.waitKey(300)  # Pequena pausa para feedback visual
                        break
            else:
                print("Nenhum rosto adequado detectado para capturar!")
        elif key == ord('q'):
            break
    
    cam.release()
    cv2.destroyAllWindows()
    print(f"Coleta concluída para {person_name}. {count} imagens salvas.")
    
    # Retreinar o reconhecedor com a nova pessoa
    global recognizer, label_map
    recognizer, label_map = train_recognizer()
    if recognizer is not None:
        print("Novo modelo treinado com sucesso!")

def recognize_faces(frame, gray):
    """Função para reconhecer rostos - só é chamada se o recognizer estiver treinado"""
    if recognizer is None:
        # Se não há recognizer, apenas detecta rostos sem reconhecer
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(60, 60),
            maxSize=(300, 300)
        )
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, "Desconhecido (Nao treinado)", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return len(faces)
    
    # Se há recognizer, faz o reconhecimento completo
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(60, 60),
        maxSize=(300, 300)
    )

    recognized_faces = 0
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        
        # Verificar se o rosto tem tamanho adequado
        face_ratio = w / frame.shape[1]
        if face_ratio < 0.15 or face_ratio > 0.4:
            continue
            
        if len(eyes) < 1:
            continue

        label_id, confidence = recognizer.predict(roi_gray)
        if confidence < CONFIDENCE_THRESHOLD:
            name = label_map.get(label_id, "Desconhecido")
            color = (0, 255, 0)
            recognized_faces += 1
        else:
            name = "Desconhecido"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        text = f"{name} - {int(confidence)}"
        if debug_mode:
            text += " (DEBUG)"
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        log_recognition(name, confidence, roi_gray, x, y)
    
    return recognized_faces

# ================= INICIALIZAÇÃO DO BANCO DE DADOS =================
print("Inicializando conexão com o banco de dados...")
if create_tables():
    print("Banco de dados configurado com sucesso!")
else:
    print("AVISO: Não foi possível configurar o banco de dados. O sistema funcionará sem salvar no MySQL.")

# ================= TREINAMENTO INICIAL =================
recognizer, label_map = train_recognizer()

# ================= WEBCAM =================
camera_index = 1  # Alterado para câmera USB (índice 1)
cam = cv2.VideoCapture(camera_index)
if not cam.isOpened():
    print(f"Erro: Não foi possível acessar câmera USB {camera_index}")
    # Tentar câmera alternativa (índice 2) se a primeira USB não funcionar
    camera_index = 2
    cam = cv2.VideoCapture(camera_index)
    if not cam.isOpened():
        print(f"Erro: Não foi possível acessar câmera USB {camera_index}")
        exit()

debug_mode = False
print("Pressione 'q' para sair | 'd' para debug ON/OFF | 'a' para adicionar nova pessoa")
print("Sistema configurado para reconhecer MÚLTIPLOS ROSTOS simultaneamente")
print(f"Usando câmera USB (índice {camera_index})")

while True:
    ret, frame = cam.read()
    if not ret:
        break
        
    # Redimensionar para processamento mais rápido
    height, width = frame.shape[:2]
    new_width = PROCESSING_WIDTH
    new_height = int((new_width / width) * height)
    frame_resized = cv2.resize(frame, (new_width, new_height))
    
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    # Reconhecer rostos (ou apenas detectar se não há modelo treinado)
    recognized_count = recognize_faces(frame_resized, gray)
    
    # Status
    if recognizer is None:
        status_text = "MODELO NAO TREINADO - Use 'A' para adicionar pessoas"
        color = (0, 0, 255)
    else:
        status_text = f"Pessoas treinadas: {len(label_map)} | Reconhecidos: {recognized_count}"
        color = (255, 255, 255)
    
    cv2.putText(frame_resized, status_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame_resized, f"Debug: {'ON' if debug_mode else 'OFF'}", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame_resized, "Q: Sair | D: Debug | A: Adicionar pessoa", 
                (10, frame_resized.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Reconhecimento Facial - Multiplos Rostos", frame_resized)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('d'):
        debug_mode = not debug_mode
        print(f"Debug {'ativado' if debug_mode else 'desativado'}")
    elif key == ord('a'):
        # Liberar a câmera antes de adicionar nova pessoa
        cam.release()
        cv2.destroyAllWindows()
        add_new_person()
        # Reabrir a câmera após adicionar a pessoa
        cam = cv2.VideoCapture(camera_index)
        if not cam.isOpened():
            print(f"Erro: Não foi possível reabrir a câmera USB {camera_index}")
            break

cam.release()
cv2.destroyAllWindows()
print("Sistema encerrado.")