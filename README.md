# 🌿 PlantGuard AI — Plant Disease Detection

AI-система диагностики болезней растений по фото листьев.  
**MobileNetV2 + Test-Time Augmentation + Mistral-7B LLM рекомендации.**

---

## ⚡ Быстрый старт

### 1. Клонировать репозиторий
```bash
git clone https://github.com/mara-werils/plant-disease-detection.git
cd plant-disease-detection
```

### 2. Запустить Backend (Python API)
```bash
cd api
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

Создай файл `.env` в папке `api/`:
```
HF_TOKEN=hf_ваш_токен_huggingface
```
> Токен можно получить бесплатно на [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

Запусти сервер:
```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```
> При первом запуске модель скачается автоматически (~15 МБ).

### 3. Запустить Frontend (Next.js)
```bash
cd plant-disease-web
npm install
npm run dev
```

### 4. Открыть в браузере
Перейди на **http://localhost:3000** — загрузи фото листа и получи диагноз!

---

## 🛠 Технологии

| Компонент | Технология |
|-----------|-----------|
| ML модель | MobileNetV2 (PyTorch, HuggingFace) |
| XAI | Saliency Maps (градиентный анализ) |
| LLM | Mistral-7B через HuggingFace API |
| Backend | FastAPI + Uvicorn |
| Frontend | Next.js + React |

---

## 📋 Требования

- **Python** 3.10+
- **Node.js** 18+
- **HuggingFace токен** (бесплатный)
