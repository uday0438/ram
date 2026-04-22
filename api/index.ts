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

import knowledgeBase from '../knowledge_base.json';

const app = express();

app.use(cors());
app.use(express.json({ limit: '10mb' }));

const apiKey = process.env.GEMINI_API_KEY;
const genAI = new GoogleGenerativeAI(apiKey || "");

app.post('/api/analyze', async (req, res) => {
    try {
        const { images, language = 'English' } = req.body;
        const model = genAI.getGenerativeModel({ 
            model: "gemini-1.5-flash",
            generationConfig: { 
                responseMimeType: "application/json"
            }
        });

        const prompt = `
            Analyze these crop images. Identify plant and disease. 
            Ground with: ${JSON.stringify(knowledgeBase.fertilizer_logic)}
            Return JSON matching:
            {
                "diseaseResult": "string",
                "solution": "string",
                "preventiveMeasures": ["string"],
                "soilFertility": { "pH": "string", "nitrogen": "string", "phosphorus": "string", "potassium": "string", "soilType": "string" },
                "fertilizerCost": { "urea": "string", "dap": "string", "mop": "string", "totalCost": "string" },
                "nextCropRecommendation": "string"
            }
            Language: ${language}.
        `;

        const imageParts = images.map((img: any) => ({
            inlineData: { data: img.data, mimeType: img.mimeType }
        }));

        const result = await model.generateContent([prompt, ...imageParts]);
        res.json(JSON.parse(result.response.text()));
    } catch (error: any) {
        console.error('Analysis error:', error);
        res.status(500).json({ error: error.message });
    }
});

app.post('/api/chat', async (req, res) => {
    try {
        const { message, history = [], language = 'English' } = req.body;
        const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });
        
        const chat = model.startChat({
            history: history.map((h: any) => ({
                role: h.role === 'user' ? 'user' : 'model',
                parts: [{ text: h.content }]
            })),
            generationConfig: {
                maxOutputTokens: 1000,
            }
        });

        const systemPrompt = `You are "Doctor AI", a professional agricultural expert. Respond in ${language}.`;
        const result = await chat.sendMessage(message + "\n\n" + systemPrompt);
        res.json({ response: result.response.text() });
    } catch (error: any) {
        console.error('Chat error:', error);
        res.status(500).json({ error: error.message });
    }
});

app.post('/api/encyclopedia', async (req, res) => {
    try {
        const { query, language = 'English' } = req.body;
        const model = genAI.getGenerativeModel({ 
            model: "gemini-1.5-flash",
            generationConfig: { responseMimeType: "application/json" }
        });

        const prompt = `Provide encyclopedia info for ${query} in JSON. Language: ${language}. Fields: cropName, scientificName, description, growthCycle, commonDiseases[], idealSoil, optimalHarvest.`;
        const result = await model.generateContent(prompt);
        res.json(JSON.parse(result.response.text()));
    } catch (error: any) {
        res.status(500).json({ error: error.message });
    }
});

app.post('/api/alerts', async (req, res) => {
    try {
        const { latitude, longitude, language = 'English' } = req.body;
        const model = genAI.getGenerativeModel({ 
            model: "gemini-1.5-flash",
            generationConfig: { responseMimeType: "application/json" }
        });

        const prompt = `Estimate regional ag alerts and weather for lat:${latitude}, lon:${longitude} in JSON. Language: ${language}. Fields: region, alerts[], weather:{temp, humidity, condition}.`;
        const result = await model.generateContent(prompt);
        res.json(JSON.parse(result.response.text()));
    } catch (error: any) {
        res.status(500).json({ error: error.message });
    }
});

// For local testing
if (process.env.NODE_ENV !== 'production') {
    const port = process.env.PORT || 3004;
    app.listen(port, () => {
        console.log(`Backend running on port ${port}`);
    });
}

export default app;
