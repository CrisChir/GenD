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

## 10. API REST (integrare în aplicații mai mari)

Începând cu această versiune, containerul rulează implicit `app/api.py`, un serviciu FastAPI
care expune **în același proces și același port (7860)** atât API-ul REST, cât și interfața
Gradio (montată la `/ui`).

| Endpoint | Metodă | Descriere |
|---|---|---|
| `/api/v1/health` | GET | starea serviciului, dispozitivul (CPU/CUDA), modelele încărcate |
| `/api/v1/models` | GET | lista modelelor Hugging Face, checkpoint-ul local implicit, stilurile de grafic |
| `/api/v1/analyze` | POST | analizează o imagine sau un videoclip încărcat; returnează scorurile, verdictul și graficul |
| `/api/v1/files/...` | GET | fișiere generate (grafice, media adnotată) |
| `/ui` | GET | interfața Gradio completă |
| `/docs` | GET | documentația OpenAPI interactivă (Swagger UI) |

### Exemplu de apel (imagine)

```bash
curl -X POST http://localhost:7860/api/v1/analyze \
  -F "file=@poza.jpg" \
  -F "graph_style=Organic" \
  -F "legend=true"
```

### Exemplu de apel (videoclip, cu subeșantionare)

```bash
curl -X POST http://localhost:7860/api/v1/analyze \
  -F "file=@clip.mp4" \
  -F "hf_model=yermandy/GenD_CLIP_L_14" \
  -F "stride=5" \
  -F "max_frames=50" \
  -F "graph_style=Radar 12 sectors" \
  -F "legend=true"
```

### Răspuns (JSON)

```json
{
  "job_id": "a1b2c3d4e5f6",
  "filename": "poza.jpg",
  "media_type": "image",
  "num_frames": 1,
  "num_faces": 2,
  "avg_p_fake": 0.8713,
  "median_p_fake": 0.8690,
  "p_fake_scores": [0.871, 0.872],
  "verdict": {"label": "FAKE", "p_fake": 0.8713, "threshold": 0.5},
  "graph": {
    "style": "Organic",
    "legend": true,
    "filename": "input_score_organic.png",
    "url": "/api/v1/files/api/a1b2c3d4e5f6/input_score_organic.png",
    "base64_png": "data:image/png;base64,..."
  },
  "annotated_media": {"filename": "input_annot.png", "url": "/api/v1/files/api/a1b2c3d4e5f6/input_annot.png"},
  "processing_seconds": 12.4
}
```

Câmpul `graph.base64_png` conține imaginea graficului (cu legendă) încorporată direct în
răspuns, astfel încât aplicația consumatoare nu mai trebuie să facă un al doilea request.
Alternativ, `graph.url` poate fi descărcat separat.

### Parametri request (`/api/v1/analyze`, toți opționali în afară de `file`)

| Parametru | Implicit | Descriere |
|---|---|---|
| `model_source` | `Hugging Face` | `Hugging Face` sau `Local Checkpoint` |
| `hf_model` | `yermandy/GenD_DINOv3_L` | identificatorul modelului HF |
| `local_ckpt` | vezi `DEFAULT_CKPT` | calea către checkpoint local |
| `face_thresh` | `0.5` | pragul detectorului de fețe |
| `scale` | `1.3` | scalarea alinierii feței |
| `target_size` | `-1` | dimensiunea feței în px (`-1` = original) |
| `max_faces` | `-1` | numărul maxim de fețe pe cadru (`-1` = toate) |
| `stride` | `1` | procesează 1 din N cadre (doar video) |
| `max_frames` | `-1` | limita de cadre procesate (`-1` = fără limită) |
| `graph_style` | `Organic` | `Organic`, `Radar 12 sectors` sau `None` |
| `legend` | `true` | afișează legenda pe grafic |
| `verdict_threshold` | `0.5` | pragul peste care verdictul este `FAKE` |
| `include_base64` | `false` | include și media adnotată ca base64 (doar imagini) |

### CORS

Pentru apeluri din browser din alte origini, setați variabila de mediu:

```yaml
environment:
  API_CORS_ORIGINS: "https://app-exemplu.ro"
```

Valoarea implicită este `*` (permisivă). Inferența este serializată intern (un singur
request de analiză rulează la un moment dat); pentru volume mari, scalați pe orizontală
cu mai multe replici ale containerului.

### Revenire la modul doar UI

```bash
docker run --rm -p 7860:7860 gend-app python app/run.py
```
