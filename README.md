# multimodal-classifier
A multimodal AI application capable of performing real-time classification using images, video, and audio. It includes computer vision models based on transfer learning and a CTC-based speech recognition system. The project integrates all components into a unified, fully functional deep learning application.

## Parte B – Speech to Text con CTC

La segunda parte del proyecto se centra en reconocimiento de voz con Connectionist Temporal Classification (CTC). Se añadieron componentes completos para cubrir las responsabilidades mostradas en la consigna:

### 1. Recolección de datos
- **Estructura de carpetas:**
	- `data/raw/audio/` contiene los audios originales divididos opcionalmente en `train/`, `val/` y `test`.
	- `data/processed/audio/` queda disponible para features precomputadas si deseas optimizar tiempos.
- **Metadata etiquetada:** todos los clips registrados deben aparecer en `data/raw/audio/metadata.csv` con columnas `path,text,split`. El `path` es relativo a `data/raw/audio/`.
- **Volumen mínimo:** el pipeline verifica que existan al menos 100 audios etiquetados (35 recolectados + 35 propios + audios adicionales descargados) antes de entrenar.
- **Script de captura:**
	```bash
	python scripts/capture_audio_dataset.py --duration 3 --split train
	```
	El script pide una frase, graba el audio y actualiza automáticamente el CSV. Puedes pasar `--phrases "hola mundo" "necesito ayuda"` para registrar lotes.

### 2. Modelo speech-to-text (CTC)
- **Preprocesamiento:** `src/data/preprocessing/audio_preprocessor.py` convierte cada clip a log-mel + MFCC (normalizados) con SpecAugment opcional. Incluye configuración centralizada (`AudioPreprocessConfig`).
- **Tokenizer:** `src/data/preprocessing/text_tokenizer.py` gestiona el vocabulario caracter a caracter (incluye tildes y ñ) y produce las secuencias para CTC.
- **DataLoader:** `src/data/loaders/audio_loader.py` arma `DataLoader`s con padding dinámico y collation adaptado a CTC. Divide train/val/test automáticamente si el CSV no trae `split`.
- **Modelo:** `src/models/audio_recognizer/ctc_speech_recognizer.py` implementa un encoder CNN + BiLSTM + capa lineal con `CTCLoss`, métricas de pérdida y WER, y métodos de guardado/carga.
- **Entrenamiento:**
	```bash
	python src/training/audio/train_ctc_speech.py \
			--metadata data/raw/audio/metadata.csv \
			--audio-root data/raw/audio \
			--epochs 30 --batch-size 8
	```
	El script guarda el mejor checkpoint en `models/checkpoints/ctc_speech_<timestamp>.pth` junto con `ctc_speech_info.json` (tokenizer, preprocessing y métricas).

### 3. Transcripción en tiempo real
- `src/app/realtime/audio_transcriber.py` captura audio del micrófono (usando `sounddevice`), procesa ventanas de 3 s y muestra el texto en vivo.
	```bash
	python -m src.app.realtime.audio_transcriber \
			--model models/checkpoints/ctc_speech_*.pth \
			--info models/checkpoints/ctc_speech_info.json
	```
- Puedes ajustar `--window` y `--chunk` para optimizar latencia/respuesta según el hardware.

Con estos componentes puedes completar todo el flujo de la Parte 2: reunir ≥100 audios etiquetados, entrenar un modelo CTC y ejecutar transcripción en tiempo real optimizada para respuestas rápidas.
