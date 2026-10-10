# SmartSemsar - PropTech-AI-Platform
AI-Driven Property Matching & 3D Virtual Staging Engine

## Furniture: notebooks and code

| Notebook | Runs on | What it does |
|---|---|---|
| `smartsemsar_demo.ipynb` | Colab (CPU is enough) | The app (`app.py`): one listing, your wishes, one photo of your own piece |
| `SmartSemsar_multi_photo.ipynb` | Colab (CPU is enough) | Several photos, the whole apartment furnished with them |
| `cv_service/furniture_server_kaggle.ipynb` | Kaggle, GPU T4 x2 | Image-to-3D server (Hunyuan3D-2 mini, coloured). Run All twice; put the address it prints in `COLAB_URL` |
| `cv_service/colab_server.ipynb` | Colab, T4 GPU | The same server with TripoSR (shape and colour from the photo, faster, rougher) |

The free Kaggle and Colab sessions stop after a while without use; run them again and use the new link.

In the code:
- `backend/agents/graph.py`: `staging_3d_node` builds the 3D apartment (`model_state`), then `furniture_node`
  furnishes it with the wishes and own pieces in `furniture_state` (`backend/schema/state.py`).
- `backend/services/multi_furnish.py`: own pieces -> catalog -> rules or LLM -> validator -> walkthrough.
- `backend/services/model_check.py`: Gemini and Groq retire models often; `python -m backend.services.model_check`
  shows which ones answer and points the app at them.
