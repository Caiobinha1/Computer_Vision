#neste codigo estamos tirando foto, passando pela inferencia, se nao identifica rosto 1 ele roda uma inferencia separados para cada rosto
#conexoes: camera, botao gpio2, servo gpio18 (pwmo), led gpio8
from picamera2 import Picamera2
from ultralytics import YOLO
import RPi.GPIO as GPIO
import cv2
import time
from gpiozero import Button, AngularServo
import sys
import select

# ConfiguraÃ§Ã£o dos pinos GPIO
GPIO.setmode(GPIO.BOARD)
GPIO.setup(8, GPIO.OUT, initial=GPIO.LOW)
button = Button(2)
servo = AngularServo(18, min_pulse_width=0.0006, max_pulse_width=0.0023)

# Inicializar Picamera2 e modelo YOLO
picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration())
model = YOLO('eu_novo.pt')

def capture_image():
    """Captura uma imagem da camera e retorna como um array RGB."""
    picam2.start()
    img = picam2.capture_array()
    picam2.stop()
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def detect_objects(img):
    """Detecta objetos na imagem usando YOLO e retorna os resultados."""
    results = model(img)  # InferÃªncia com YOLO
    annotated_img = results[0].plot()
    boxes = results[0].boxes
    return annotated_img, boxes

def check_for_quit():
    """Verifica se a tecla 'q' foi pressionada para encerrar o programa."""
    if select.select([sys.stdin], [], [], 0)[0]:
        key = sys.stdin.read(1)
        if key.lower() == 'q':
            print("Encerrando o programa...")
            GPIO.cleanup()  # Limpar GPIO antes de sair
            sys.exit(0)

def crop_faces(img, boxes):
    """Recorta as Ã¡reas de detecÃ§Ã£o (rostos) da imagem original."""
    face_images = []
    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box.xyxy[0])  # Coordenadas da ca>
        face_images.append(img[y_min:y_max, x_min:x_max])
    return face_images

if __name__ == "__main__":
    print("Pressione o botÃ£o para capturar uma imagem e detectar objetos.")
    print("Pressione 'q' para encerrar o programa.")

    # Configurar terminal para leitura nÃ£o bloqueante
    sys.stdin = open('/dev/stdin', 'r')

    while True:
        servo.angle = 85  # PosiÃ§Ã£o inicial (trancado)
        GPIO.output(8, GPIO.LOW)  #simular servo trancado

        # Verificar se "q" foi pressionado
        check_for_quit()

        # Aguardar botÃ£o ser pressionado
        if button.is_pressed:
            print("BotÃ£o pressionado!")

            # Acender o LED e capturar imagem
            #GPIO.output(8, GPIO.HIGH)
            start_time = time.time()
            img = capture_image()
            capture_time = time.time() - start_time
            #GPIO.output(8, GPIO.LOW)

            print(f"Tempo para capturar imagem: {capture_time:.4f} segundos")

            # Detectar objetos na imagem
            annotated_img, boxes = detect_objects(img)

            # Exibir a imagem anotada inicial
            print("Exibindo resultado inicial...")
            img_resize = cv2.resize(annotated_img, (640, 480))
            cv2.imshow("Resultado Inicial", img_resize)
            cv2.waitKey(0)

            # Verificar se a classe 1 foi detectada
            detected_classes = [int(box.cls) for box in boxes]
            if 1 in detected_classes:
                print("Classe 1 detectada: Acesso Liberado!")
                servo.angle = 5  # Abrir (liberar)
                GPIO.output(8, GPIO.HIGH)
                time.sleep(3)  # Manter por 3 segundos
            else:
                print("Nenhuma classe 1 detectada: Acesso Negado.")
                # Recortar os rostos detectados
                faces = crop_faces(img, boxes)
                print(f"{len(faces)} rostos detectados. Realizando inferÃªncias individuais.")


                for i, face in enumerate(faces):
                    print(f"Rodando inferÃªncia para o rosto {i + 1}...")
                    resize_face = cv2.resize(face,(640,640),interpolation=cv2.INTER_AREA)

                    face_results = model(resize_face)  # Nova inferÃªncia
                    annotated_face = face_results[0].plot()
                    face_classes = [int(box.cls) for box in face_results[0].boxes]

                    # Exibir o rosto com resultados anotados
                    face_resize = cv2.resize(annotated_face, (320, 240))
                    cv2.imshow(f"Resultado Rosto {i + 1}", face_resize)
                    cv2.waitKey(0)

                    if 1 in face_classes:
                        print(f"Classe 1 detectada no rosto {i + 1}: Acesso Liberado!")
                        servo.angle = 5
                        GPIO.output(8, GPIO.HIGH)
                        time.sleep(3)
                        break
                else:
                    print("Nenhuma classe 1 detectada apÃ³s nova verificaÃ§Ã£o.")

            cv2.destroyAllWindows()



    
