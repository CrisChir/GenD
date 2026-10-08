# GenD – Container Docker (Gradio App)

Acest folder conține tot ce este necesar pentru a rula aplicația Gradio GenD (`app/run.py`) într-un container Docker, inclusiv pe **macOS** (Intel și Apple Silicon), pentru verificarea functionalității end-to-end.

## Structură

```
docker/
├── Dockerfile            # imaginea (python:3.12-slim + toate dependențele)
├── docker-compose.yml    # pornire cu un singură comandă
├── dockerignore           # se copiază în rădăcină ca .dockerignore
└── README.md              # acest fișier
```

## 1. Cerințe (macOS)

- **Docker Desktop pentru Mac** – https://www.docker.com/products/docker-desktop/
  - Pe Apple Silicon (M1/M2/M3/M4) activați opțiunea **"Use Rosetta for x86_64/amd64 emulation"** doar dacă construiți imagini amd64; imaginea nativă arm64 este recomandată (mai rapidă).
- Minim ~6 GB spațiu liber (imaginile PyTorch sunt mari).
- (Opțional) token Hugging Face pentru modelul gated `yermandy/GenD_DINOv3_L`.

## 2. Pregătire (o singură dată)

```bash
cd GenD                      # rădăcina repo-ului
cp docker/dockerignore .dockerignore
```

## 3. Construire și pornire

### Varianta A – Docker Compose (recomandată)

```bash
# din rădăcina repo-ului:
export HF_TOKEN="hf_vÎrful_tău"        # opțional; necesar doar pentru DINOv3-L
docker compose -f docker/docker-compose.yml up --build
```

### Varianta B – Docker clasic

```bash
docker build -f docker/Dockerfile -t gend-app .
docker run --rm -p 7860:7860 \
  -e HF_TOKEN="$HF_TOKEN" \
  -v gend-hf-cache:/cache/huggingface \
  gend-app
```

## 4. Verificarea funcționării

1. Deschideți în browser: **http://localhost:7860**
2. În interfața Gradio:
   - **Model source**: alegeți *Hugging Face*
   - **Model ID**: `yermandy/GenD_CLIP_L_14` (funcționează fără token)
3. Încărcați o imagine sau un videoclip și apăsați butonul de analiză.
4. Verificați în log-uri că modelul rulează pe CPU:

```bash
docker logs -f gend-app
docker exec -it gend-app python -c "import torch; print('cuda:', torch.cuda.is_available())"
```

Rezultatul așteptat pe macOS: `cuda: False` — aplicația folosește automat CPU cu `float32` (codul din `app/run.py` detectează dispozitivul).

## 5. Note pentru macOS / Apple Silicon

| Aspect | Detaliu |
|---|---|
| Arhitectură | Imaginea se construiește nativ `linux/arm64` — fără emulare, pornire rapidă |
| onnxruntime | `requirements.txt` cere `onnxruntime-gpu`, care nu există pe arm64; `Dockerfile` îl înlocuiește automat cu `onnxruntime` (CPU) pe arm64 |
| GPU | Docker pe macOS nu expune GPU-ul Metal către containere — inferența rulează pe CPU; pentru performanță GPU folosiți serverul cu Tesla P40 |
| Precision | `app/run.py` forțează `float32` pe CPU — identic cu comportamentul pe P40 |

## 6. Rulare pe server cu GPU (Linux + Tesla P40)

Imaginea de bază poate fi schimbată într-una CUDA. Adăugați în `Dockerfile`:

```dockerfile
FROM nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04
# ... apoi instalați python3.12 + pip (apt) și continuați identic
```

și decomentați blocul `deploy.resources.reservations.devices` din `docker-compose.yml`. Pornire:

```bash
docker compose -f docker/docker-compose.yml up --build
```

Notă: pe P40 (arhitectura Pascal) aplicația forțează `float32`; nu este nevoie de `onnxruntime-gpu` decât dacă doriți detecția RetinaFace pe GPU.

## 7. Utilizare cu checkpoint local

Pentru a testa un checkpoint local (`.ckpt`) în loc de modelele Hugging Face:

```bash
docker run --rm -p 7860:7860 \
  -v /calea/către/runs:/app/runs \
  gend-app
```

Alegeți apoi *Local checkpoint* în interfața Gradio și introduceți calea `runs/.../checkpoints/best_mAP.ckpt`.

## 8. Depanare rapidă

| Simptom | Cauză probabilită | Soluție |
|---|---|---|
| `onnxruntime-gpu` nu se instalează | build pe arm64 | deja rezolvat automat de Dockerfile (folosește `onnxruntime`) |
| Port 7860 ocupat | alt proces Gradio | schimbați mappingul: `-p 7861:7860` sau `docker rm -f gend-app` |
| Timeout la descărcarea modelului HF | prima descărcare este ~1-2 GB | lăsați containerul să ruleze; cache-ul persistă în volumul `hf-cache` |
| Eroare 401 la `GenD_DINOv3_L` | model gated | cereți acces pe pagina Hugging Face și setați `HF_TOKEN` |
| `no such file or directory: app/run.py` | build din folder greșit | construiți mereu din rădăcina repo-ului cu `-f docker/Dockerfile` |
| Gradio "external IP request" warning | `GRADIO_SERVER_NAME` | deja setat la `0.0.0.0` în imagine |

## 9. Curățare

```bash
docker compose -f docker/docker-compose.yml down          # oprește containerul
docker compose -f docker/docker-compose.yml down -v        # oprește + șterge volumele (modelele descărcate)
docker image rm gend-app                                    # șterge imaginea
```
