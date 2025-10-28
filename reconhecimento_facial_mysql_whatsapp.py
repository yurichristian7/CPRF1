import os
import cv2
import numpy as np
from datetime import datetime, time
import mysql.connector
import pywhatkit
import time as time_module

# ================= CONFIGURAÇÕES =================
# CONFIGURAÇÕES MySQL
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "reconhecimento_facial2"
}

# CONFIGURAÇÕES WhatsApp
WHATSAPP_CONFIG = {
    "phone_number": "+5531996393313",  # Número do responsável no formato internacional
    "group_id": None  # Ou ID do grupo se quiser enviar para grupo
}

# CONEXÃO COM BANCO
try:
    db = mysql.connector.connect(**db_config)
    cursor = db.cursor()
    print("Conexão com MySQL estabelecida com sucesso!")
except mysql.connector.Error as err:
    print(f"Erro ao conectar no MySQL: {err}")
    exit(1)

# Cascades
face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
eye_cascade_path = cv2.data.haarcascades + "haarcascade_eye.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

# Diretórios
log_dir = "logs"
faces_dir = "rostos_salvos"
dataset_dir = "dataset"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(faces_dir, exist_ok=True)
os.makedirs(dataset_dir, exist_ok=True)

# Dicionário para controlar o último registro de cada pessoa por hora
last_recognition_time = {}
FACE_SIZE = (100, 100)
CONFIDENCE_THRESHOLD = 80  # Ajuste inicial
start_time = time(7, 0, 0)
end_time = time(23, 0, 0)

# Redimensionamento para processamento mais rápido
PROCESSING_WIDTH = 640

# Variáveis globais
recognizer = None
label_map = {}

# ================= FUNÇÕES WHATSAPP =================
def enviar_mensagem_whatsapp(nome_pessoa, confianca):
    """
    Envia mensagem via WhatsApp quando uma pessoa é reconhecida
    usando pywhatkit
    """
    try:
        mensagem = f"🔍 *CPRF*\n\n👤 *Pessoa reconhecida:* {nome_pessoa}\n\n⏰ *Horário:* {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\nSistema funcionando normalmente!"
        
        # Obter hora atual para agendamento imediato
        agora = datetime.now()
        hora = agora.hour
        minuto = agora.minute + 1  # Enviar daqui a 1 minuto
        
        # Ajustar se passar de 59 minutos
        if minuto >= 60:
            minuto = minuto - 60
            hora = hora + 1 if hora < 23 else 0
        
        print(f"📱 Agendando mensagem WhatsApp para {hora}:{minuto:02d}")
        
        # Enviar mensagem via pywhatkit
        pywhatkit.sendwhatmsg(
            WHATSAPP_CONFIG["phone_number"],
            mensagem,
            hora,
            minuto,
            wait_time=15,
            tab_close=True
        )
        
        print(f"Mensagem WhatsApp enviada para {nome_pessoa}")
        return True
        
    except Exception as e:
        print(f"Erro ao enviar WhatsApp: {str(e)}")
        return False

def enviar_mensagem_whatsapp_instantanea(nome_pessoa, confianca):
    """
    Versão que envia mensagem e fecha o WhatsApp
    """
    try:
        mensagem = f"🔍 *CPRF*\n\n👤 *Pessoa reconhecida:* {nome_pessoa}\n\n⏰ *Horário:* {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\nSistema funcionando normalmente!"
        
        print(f"📱 Preparando envio do WhatsApp para {nome_pessoa}")
        
        # Enviar mensagem instantânea
        pywhatkit.sendwhatmsg_instantly(
            WHATSAPP_CONFIG["phone_number"],
            mensagem,
            wait_time=15,
            tab_close=True  # Fecha após enviar
        )
        
        print(f"Mensagem WhatsApp enviada para {nome_pessoa}")
        return True
        
    except Exception as e:
        print(f"Erro ao enviar WhatsApp: {str(e)}")
        return False

def enviar_alerta_desconhecido_instantaneo():
    """
    Versão que envia alerta e fecha o WhatsApp
    """
    try:
        mensagem = f"🚨 *ALERTA - Pessoa Desconhecida*\n\n⚠️ Foi detectada uma pessoa não cadastrada no sistema!\n⏰ *Horário:* {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n📍 *Local:* Câmera de Reconhecimento\n\n🔍 *Ação necessária:* Verificar imediatamente!"
        
        print("📱 Preparando envio de ALERTA WhatsApp")
        
        pywhatkit.sendwhatmsg_instantly(
            WHATSAPP_CONFIG["phone_number"],
            mensagem,
            wait_time=15,
            tab_close=True  # Fecha após enviar
        )
        
        print("Alerta WhatsApp enviado para pessoa desconhecida")
        return True
        
    except Exception as e:
        print(f"Erro ao enviar alerta WhatsApp: {str(e)}")
        return False

# ================= FUNÇÕES EXISTENTES =================
def salvar_registro_mysql(nome, confianca, data_hora, caminho_img):
    try:
        sql = ("INSERT INTO registros_reconhecimento "
               "(nome, confianca, data_hora, caminho_imagem) "
               "VALUES (%s, %s, %s, %s)")
        valores = (nome, confianca, data_hora, caminho_img)
        cursor.execute(sql, valores)
        db.commit()
        print(f"Registro salvo no MySQL: {nome} - {confianca}%")
        return True
    except mysql.connector.Error as err:
        print(f"Erro ao inserir no MySQL: {err}")
        return False

def pode_salvar_no_banco(nome):
    """Verifica se pode salvar no banco (apenas uma vez por hora por pessoa)"""
    agora = datetime.now()
    chave = f"{nome}_{agora.strftime('%Y-%m-%d %H')}"  # Nome + data + hora
    
    if chave in last_recognition_time:
        # Verifica se já passou pelo menos 1 hora desde o último envio
        tempo_passado = (agora - last_recognition_time[chave]).total_seconds()
        if tempo_passado < 3600:  # 1 hora em segundos
            return False
    
    # Marcar que já salvou para esta pessoa nesta hora
    last_recognition_time[chave] = agora
    return True

def limpar_registros_antigos():
    """Limpa registros antigos do dicionário de controle"""
    agora = datetime.now()
    chaves_para_remover = []
    
    for chave, timestamp in last_recognition_time.items():
        # Se o registro tem mais de 24 horas, remover
        if (agora - timestamp).total_seconds() > 86400:  # 24 horas em segundos
            chaves_para_remover.append(chave)
    
    for chave in chaves_para_remover:
        del last_recognition_time[chave]

def list_available_cameras():
    """Lista todas as câmeras disponíveis no sistema"""
    available_cameras = []
    print("Procurando câmeras disponíveis...")
    
    # Testar as primeiras 2 câmeras
    for i in range(2):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available_cameras.append(i)
                print(f"Câmera {i} encontrada - Resolução: {frame.shape[1]}x{frame.shape[0]}")
            cap.release()
        else:
            print(f"Câmera {i} não disponível")
    
    return available_cameras

def select_camera():
    """Permite ao usuário selecionar qual câmera usar"""
    cameras = list_available_cameras()
    
    if not cameras:
        print("Nenhuma câmera encontrada! Verifique as conexões.")
        return None
    
    print("\nCâmeras disponíveis:")
    for i, cam_idx in enumerate(cameras):
        print(f"{i}: Câmera {cam_idx}")
    
    while True:
        try:
            choice = input(f"Escolha uma câmera (0-{len(cameras)-1}) ou pressione Enter para padrão (0): ").strip()
            if choice == "":
                selected_camera = cameras[0]
                print(f"Usando câmera padrão: {selected_camera}")
                return selected_camera
            
            choice_idx = int(choice)
            if 0 <= choice_idx < len(cameras):
                selected_camera = cameras[choice_idx]
                print(f"Usando câmera: {selected_camera}")
                return selected_camera
            else:
                print(f"Escolha inválida. Digite um número entre 0 e {len(cameras)-1}")
        except ValueError:
            print("Por favor, digite um número válido.")

def log_recognition(name, confidence, face_img, x, y):
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    current_time = now.time()
    
    # Verificar se está no horário permitido
    if not (start_time <= current_time <= end_time):
        print(f"Fora do horário permitido ({start_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')})")
        return

    # Criar diretório para a pessoa
    person_dir = os.path.join(faces_dir, name.lower())
    os.makedirs(person_dir, exist_ok=True)
    
    # Salvar imagem do rosto
    resized_face = cv2.resize(face_img, FACE_SIZE)
    face_filename = f"{today_str}_{now.strftime('%H-%M-%S')}_{name.lower()}_{x}_{y}.png"
    face_path = os.path.join(person_dir, face_filename)
    cv2.imwrite(face_path, resized_face)
    
    # Salvar no MySQL apenas se for rosto conhecido e apenas uma vez por hora
    salvar_no_mysql = False
    enviar_whatsapp = False
    
    if name.lower() != "desconhecido" and pode_salvar_no_banco(name):
        if salvar_registro_mysql(name, int(confidence), now.strftime('%Y-%m-%d %H:%M:%S'), face_path):
            salvar_no_mysql = True
            # Enviar mensagem WhatsApp apenas quando salvar no MySQL (1x por hora)
            # E apenas se não enviou nos últimos 5 minutos
            chave_whatsapp = f"whatsapp_{name.lower()}_{now.strftime('%Y-%m-%d %H')}"
            if chave_whatsapp not in last_recognition_time:
                enviar_whatsapp = enviar_mensagem_whatsapp_instantanea(name, confidence)
                last_recognition_time[chave_whatsapp] = now
    elif name.lower() == "desconhecido":
        # Enviar alerta para pessoa desconhecida (com controle para não enviar muitas mensagens)
        chave_desconhecido = f"desconhecido_{now.strftime('%Y-%m-%d %H')}"
        if chave_desconhecido not in last_recognition_time:
            enviar_alerta_desconhecido_instantaneo()
            last_recognition_time[chave_desconhecido] = now
    
    # Salvar também no arquivo de log local
    if name.lower() == "desconhecido":
        filename = f"desconhecidos_{today_str}.txt"
    else:
        filename = f"log_{today_str}.txt"

    log_path = os.path.join(log_dir, filename)
    with open(log_path, "a", encoding='utf-8') as f:
        mysql_info = " [SALVO NO MYSQL]" if salvar_no_mysql else ""
        whatsapp_info = " [WHATSAPP ENVIADO]" if enviar_whatsapp else ""
        f.write(f"{now.strftime('%H:%M:%S')} - {name} (Confiança: {int(confidence)}){mysql_info}{whatsapp_info}\n")

def get_images_and_labels():
    labels, faces = [], []
    label_map = {}
    next_id = 0

    if not os.path.exists(dataset_dir):
        print(f"AVISO: Diretório '{dataset_dir}' não encontrado.")
        return faces, labels, label_map

    people = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    print(f"Pessoas encontradas no dataset: {people}")

    for person_name in people:
        person_dir = os.path.join(dataset_dir, person_name)
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

def add_new_person(camera_index):
    person_name = input("Digite o nome da nova pessoa: ").strip()
    if not person_name:
        print("Nome inválido!")
        return
    
    person_dir = os.path.join(dataset_dir, person_name)
    os.makedirs(person_dir, exist_ok=True)
    
    print(f"Coletando imagens para {person_name}. Posicione-se frente à câmera.")
    print("Pressione ESPAÇO para capturar ou 'q' para sair")
    print("Dica: Fique a ~50cm da câmera, com boa iluminação no rosto")
    
    cam = cv2.VideoCapture(camera_index)
    if not cam.isOpened():
        print(f"Erro: Não foi possível acessar câmera {camera_index}")
        return
    
    count = 0
    max_count = 10  # Número de imagens a capturar
    
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
        
        # Ajustar parâmetros para câmera
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(60, 60),
            maxSize=(300, 300)
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
        cv2.putText(frame_resized, f"Camera: {camera_index}", (10, frame_resized.shape[0]-40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
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

        # Limpar registros antigos periodicamente
        if len(last_recognition_time) > 100:  # Limitar tamanho do dicionário
            limpar_registros_antigos()
            
        log_recognition(name, confidence, roi_gray, x, y)
    
    return recognized_faces

# ================= PROGRAMA PRINCIPAL =================
if __name__ == "__main__":
    # Selecionar câmera
    camera_index = select_camera()
    if camera_index is None:
        print("Nenhuma câmera disponível. Encerrando...")
        exit()
    
    # Treinamento inicial
    recognizer, label_map = train_recognizer()
    
    # Inicializar webcam
    cam = cv2.VideoCapture(camera_index)
    if not cam.isOpened():
        print(f"Erro: Não foi possível acessar câmera {camera_index}")
        exit()

    debug_mode = False
    print("\nPressione 'q' para sair | 'd' para debug ON/OFF | 'a' para adicionar nova pessoa")
    print("Sistema configurado para reconhecer MÚLTIPLOS ROSTOS simultaneamente")
    print(f"Usando câmera: {camera_index}")
    print(f"Horário de funcionamento: {start_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')}")
    print("MySQL: Rostos conhecidos salvos apenas 1x por hora | Desconhecidos não salvos no MySQL")
    print("WhatsApp: Mensagens enviadas para rostos conhecidos e alertas para desconhecidos")
    print(f"Número WhatsApp: {WHATSAPP_CONFIG['phone_number']}")

    # Contador para limpeza periódica
    frame_count = 0
    
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
        
        # Limpar registros antigos a cada 100 frames
        frame_count += 1
        if frame_count >= 100:
            limpar_registros_antigos()
            frame_count = 0
        
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
        cv2.putText(frame_resized, f"Camera: {camera_index}", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
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
            add_new_person(camera_index)
            # Reabrir a câmera após adicionar a pessoa
            cam = cv2.VideoCapture(camera_index)
            if not cam.isOpened():
                print(f"Erro: Não foi possível reabrir a câmera {camera_index}")
                break

    cam.release()
    cv2.destroyAllWindows()
    
    # Fechar conexão com o MySQL
    try:
        cursor.close()
        db.close()
        print("Conexão com MySQL fechada.")
    except:
        pass
        
    print("Sistema encerrado.")