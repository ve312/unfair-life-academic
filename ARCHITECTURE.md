# Unfair Life - Master Architecture

## Core Stack
- **Frontend Game Client:** Phaser 3 + HTML/CSS UI Overlay + Howler.js (Audio)
- **Content CMS:** Strapi (Headless CMS)
- **Backend-For-Frontend (BFF):** Node.js + NestJS
- **AI Microservice:** Python + FastAPI (Hexagonal Architecture) + Scikit-Learn

## 1. Frontend Game Client (`/frontend`)
**Paradigm:** Hybrid Reactive UI + Game Canvas
- Renders the automated story and timed choices.
- Communicates exclusively with the NestJS BFF.
- Relies on Howler.js for dynamic, cross-fading audio.

## 2. Backend-For-Frontend (`/bff-api`)
**Paradigm:** Modular Domain-Driven Architecture
- The traffic controller. Never handles direct UI or ML logic.
- Fetches the authored story from Strapi.
- Sends the character profile to FastAPI to get the mobility prediction.
- Combines the story and AI prediction into a single JSON payload for Phaser.

## 3. Narrative & Content Hub (`/cms`)
**Paradigm:** Relational Node-Graph Architecture (Strapi)
- The creative dashboard. No hardcoded stories in the frontend.
- Houses the Story, Scene, Decision, and AudioTrack schemas.
- Contains the actual text, image URLs, and MP3 URLs.

## 4. AI Prediction Microservice (`/ai-service`)
**Paradigm:** Hexagonal (Clean) Architecture & Offline Pipeline
- **`/training-pipeline`**: Offline data science environment. Cleans Income/SocialLatino CSVs, balances with SMOTE, trains KNN/RF/SVM/LogReg, and exports `best_model.joblib`.
- **`/inference-api`**: FastAPI server. Loads `.joblib` into memory on startup. Exposes `/predict` endpoint to the BFF.