# Free Photoreal Render Sources — Activation Guide

The render engine tries sources in this order and falls back automatically.
Add ANY of the keys below to `.env` and restart the studio
(`start_studio_background.bat`) — no other changes needed.

```
mcp-tools + official FLUX spaces (your HF token quota)
→ Cloudflare Workers AI (10,000 free Neurons/DAY)      ← best new free tier
→ Stability AI ($0.05/render, best quality)
→ Gemini / OpenRouter / OpenAI (paid per image)
→ Local preview model (always available)
```

---

## 1. HuggingFace token — ✅ already configured

Your token (`MOMSTEALERIII`) is stored in `.env`. Free accounts get a small
daily ZeroGPU quota; it resets at UTC midnight (05:30 IST). When it runs out
you'll see "Photoreal engine busy — auto-retrying…" and the app retries for
you automatically.

## 2. Cloudflare Workers AI — FREE, ~20-30 images EVERY day, no card ⭐

The best new free tier: 10,000 free "Neurons" per day, permanently.

1. Sign up free: **dash.cloudflare.com/sign-up** (email only, no card)
2. Get your **Account ID**: dashboard home → right sidebar → copy "Account ID"
3. Create an **API Token**: My Profile → API Tokens → *Create Token* → use the
   **"Workers AI" template** (or Custom token with Workers AI:Edit permission)
4. Paste both into `.env`:
   ```
   CLOUDFLARE_API_TOKEN=your_token
   CLOUDFLARE_ACCOUNT_ID=your_account_id
   ```
5. Restart the studio.

Note: this tier runs SD1.5-img2img — solid quality, slightly below Kontext.
It sits **after** the HF spaces in the chain, so it takes over exactly when
your daily HF quota runs out. Two free tiers = full-day coverage.

## 3. ModelScope — free Qwen-Image-Edit (Alibaba, no card)

1. Sign up: **modelscope.cn** (GitHub or phone, no card)
2. Access token: modelscope.cn → Account → *Access Tokens* → create + copy
3. Paste into `.env`: `MODELSCOPE_API_TOKEN=ms-...`
4. Restart the studio.

## 4. OpenRouter — $1 ≈ 30 renders (already wired)

- **openrouter.ai/credits** → add $1–5 → nothing else to configure
- Your key is already in `.env`; the engine uses
  `google/gemini-2.5-flash-image` automatically

## 5. Stability AI — best quality, $0.05/render (already wired)

- **platform.stability.ai** → sign up → *API Keys* → create key
- `.env`: `STABILITY_API_KEY=sk-...` → restart
- Purpose-built `control/sketch` endpoint — ideal for this exact app

---

## After adding any key

Close the studio server window (or `taskkill` the python process on port
8000) and double-click `start_studio_background.bat`. The engine detects the
new keys at startup and the chain updates automatically.

**Check what's active anytime:** open the terminal and run
`.venv\Scripts\python.exe scripts\probe_cloud_providers.py`
— it prints which providers are ready, cooling down, or missing keys.
