import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { GoogleGenerativeAI } from "@google/generative-ai";
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load Scientific Knowledge Base
const knowledgePath = path.resolve(__dirname, 'knowledge_base.json');
const knowledgeBase = JSON.parse(fs.readFileSync(knowledgePath, 'utf-8'));

const app = express();
const port = process.env.PORT || 3004;

app.use(cors());
app.use(express.json({ limit: '50mb' }));

const apiKey = process.env.GEMINI_API_KEY;
if (!apiKey) {
    console.error('❌ GEMINI_API_KEY is not set!');
}
const genAI = new GoogleGenerativeAI(apiKey || "");

// Each model has SEPARATE free-tier quota, so fallback actually helps
const MODELS = ["gemini-2.0-flash", "gemini-2.0-flash-lite"];

function getModel(modelName: string, jsonMode = false) {
    const config: any = {};
    if (jsonMode) config.responseMimeType = "application/json";
    return genAI.getGenerativeModel({ model: modelName, generationConfig: config });
}

// Retry across multiple models with backoff
async function generateWithFallback(
    buildRequest: (modelName: string) => Promise<any>,
    maxRetriesPerModel = 2
): Promise<any> {
    let lastError: any;
    for (const modelName of MODELS) {
        for (let attempt = 0; attempt <= maxRetriesPerModel; attempt++) {
            try {
                console.log(`🔄 ${modelName} attempt ${attempt + 1}`);
                return await buildRequest(modelName);
            } catch (error: any) {
                lastError = error;
                if (error?.status === 429) {
                    if (attempt < maxRetriesPerModel) {
                        const wait = (attempt + 1) * 2000 + Math.random() * 1000;
                        console.log(`⏳ Rate limited on ${modelName}, waiting ${(wait/1000).toFixed(1)}s...`);
                        await new Promise(r => setTimeout(r, wait));
                        continue;
                    }
                    console.log(`⚠️ ${modelName} quota exhausted, trying next model...`);
                    break; // try next model
                }
                throw error; // non-429 error, throw immediately
            }
        }
    }
    throw lastError || new Error('All models exhausted');
}

function safeParseJSON(text: string) {
    try {
        return JSON.parse(text);
    } catch {
        const match = text.match(/\{[\s\S]*\}/);
        if (match) return JSON.parse(match[0]);
        return null;
    }
}

// ==================== ANALYZE ====================
app.post('/api/analyze', async (req, res) => {
    try {
        const { images, language = 'English' } = req.body;
        if (!images || !Array.isArray(images) || images.length === 0) {
            return res.status(400).json({ error: 'No images provided.' });
        }
        console.log(`📸 Analysis: ${images.length} image(s), lang: ${language}`);

        const prompt = `You are an expert agricultural scientist and plant pathologist. Analyze the provided plant/crop/vegetable/fruit images carefully.

Identify:
1. The plant species
2. Any disease, pest damage, nutrient deficiency, or health issue visible
3. Provide actionable treatment solutions
4. Estimate soil fertility parameters
5. Calculate approximate fertilizer requirements and costs in Indian Rupees (INR)
6. Recommend the best next crop for soil recovery

Reference data: ${JSON.stringify(knowledgeBase.fertilizer_logic)}

Return valid JSON:
{
    "diseaseResult": "Name of disease or 'Healthy'",
    "solution": "Detailed treatment",
    "preventiveMeasures": ["measure 1", "measure 2", "measure 3"],
    "soilFertility": { "pH": "value", "nitrogen": "Low/Medium/High", "phosphorus": "Low/Medium/High", "potassium": "Low/Medium/High", "soilType": "type" },
    "fertilizerCost": { "urea": "cost INR", "dap": "cost INR", "mop": "cost INR", "totalCost": "total INR" },
    "nextCropRecommendation": "crop with reason"
}
Respond in ${language}.`;

        const imageParts = images.map((img: any) => ({
            inlineData: { data: img.data, mimeType: img.mimeType || 'image/jpeg' }
        }));

        const result = await generateWithFallback(async (modelName) => {
            const model = getModel(modelName, true);
            return await model.generateContent([prompt, ...imageParts]);
        });

        const parsed = safeParseJSON(result.response.text());
        if (parsed) {
            console.log('✅ Analysis complete');
            res.json(parsed);
        } else {
            res.status(500).json({ error: 'AI returned invalid format. Please try again.' });
        }
    } catch (error: any) {
        console.error('❌ Analysis error:', error.message);
        res.status(error?.status === 429 ? 429 : 500).json({
            error: error?.status === 429
                ? 'API rate limit reached. Please wait 30 seconds and try again.'
                : (error.message || 'Analysis failed.')
        });
    }
});

// ==================== DOCTOR AI CHAT ====================
app.post('/api/chat', async (req, res) => {
    try {
        const { message, history = [], language = 'English' } = req.body;
        if (!message?.trim()) {
            return res.status(400).json({ error: 'Message is empty.' });
        }

        const safeHistory = history
            .filter((h: any) => h?.content && h?.role)
            .map((h: any) => ({
                role: h.role === 'user' ? 'user' : 'model',
                parts: [{ text: h.content }]
            }));

        const systemPrompt = `You are "Doctor AI", a professional agricultural expert. Provide helpful, concise, practical farming advice. Respond in ${language}.`;

        const result = await generateWithFallback(async (modelName) => {
            const model = genAI.getGenerativeModel({ model: modelName });
            const chat = model.startChat({
                history: safeHistory,
                generationConfig: { maxOutputTokens: 1500 }
            });
            return await chat.sendMessage(systemPrompt + "\n\nUser: " + message);
        });

        res.json({ response: result.response.text() });
    } catch (error: any) {
        console.error('❌ Chat error:', error.message);
        res.status(error?.status === 429 ? 429 : 500).json({
            error: error?.status === 429
                ? 'API rate limit reached. Please wait and try again.'
                : (error.message || 'Chat failed.')
        });
    }
});

// ==================== ENCYCLOPEDIA ====================
app.post('/api/encyclopedia', async (req, res) => {
    try {
        const { query, language = 'English' } = req.body;
        if (!query?.trim()) {
            return res.status(400).json({ error: 'Search query is empty.' });
        }

        const prompt = `You are an agricultural encyclopedia. Provide detailed information about "${query}".
Return valid JSON:
{
    "cropName": "common name",
    "scientificName": "scientific name",
    "description": "2-3 sentence description",
    "growthCycle": "growth stages and duration",
    "commonDiseases": ["disease 1", "disease 2", "disease 3"],
    "idealSoil": "ideal soil conditions",
    "optimalHarvest": "best harvest time"
}
Respond in ${language}.`;

        const result = await generateWithFallback(async (modelName) => {
            const model = getModel(modelName, true);
            return await model.generateContent(prompt);
        });

        const parsed = safeParseJSON(result.response.text());
        if (parsed) {
            res.json(parsed);
        } else {
            res.status(500).json({ error: 'Failed to parse encyclopedia data.' });
        }
    } catch (error: any) {
        console.error('❌ Encyclopedia error:', error.message);
        res.status(error?.status === 429 ? 429 : 500).json({
            error: error?.status === 429
                ? 'API rate limit reached. Please wait and try again.'
                : (error.message || 'Encyclopedia search failed.')
        });
    }
});

// ==================== REGIONAL ALERTS ====================
app.post('/api/alerts', async (req, res) => {
    try {
        const { latitude, longitude, language = 'English' } = req.body;

        const prompt = `Based on coordinates lat:${latitude}, lon:${longitude}, provide agricultural alerts and weather.
Return valid JSON:
{
    "region": "region name",
    "alerts": ["alert 1", "alert 2"],
    "weather": { "temp": "temperature", "humidity": "humidity%", "condition": "condition" }
}
Respond in ${language}.`;

        const result = await generateWithFallback(async (modelName) => {
            const model = getModel(modelName, true);
            return await model.generateContent(prompt);
        });

        const parsed = safeParseJSON(result.response.text());
        if (parsed) {
            res.json(parsed);
        } else {
            res.status(500).json({ error: 'Failed to parse alerts.' });
        }
    } catch (error: any) {
        console.error('❌ Alerts error:', error.message);
        res.status(error?.status === 429 ? 429 : 500).json({
            error: error?.status === 429 ? 'API rate limit reached.' : (error.message || 'Alerts failed.')
        });
    }
});

// ==================== HEALTH CHECK ====================
app.get('/api/health', (_req, res) => {
    res.json({ status: 'ok', apiKeySet: !!apiKey, models: MODELS, timestamp: new Date().toISOString() });
});

app.listen(port, () => {
    console.log(`🌱 Botanica Backend running on port ${port}`);
    console.log(`   API Key: ${apiKey ? '✅ Set' : '❌ MISSING'}`);
    console.log(`   Models: ${MODELS.join(' → ')}`);
});
