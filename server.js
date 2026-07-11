import express from 'express';
import cors from 'cors';
import multer from 'multer';
import dotenv from 'dotenv';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { execFileSync } from 'child_process';
import { fileURLToPath } from 'url';

dotenv.config();

const app = express();
const port = Number(process.env.PORT) || 3001;
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const pythonExecutable = process.env.PYTHON_PATH || 'python';
const westernOverrideClasses = new Set([
  'burger',
  'hamburg',
  'pizza',
  'french fries',
  'pasta',
  'noodles',
  'sandwich',
  'cake',
  'ice cream'
]);

const startServer = (portToUse) => {
  const server = app.listen(portToUse, () => {
    console.log(`YOLO food analyzer API running on http://localhost:${portToUse}`);
  });

  server.on('error', (error) => {
    if (error.code === 'EADDRINUSE') {
      console.error(`Port ${portToUse} is already in use. Stop the old API server and restart this app.`);
      process.exit(1);
    }

    console.error(error);
    process.exit(1);
  });
};

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024 }
});

app.use(cors());
app.use(express.json());

function analyzeImageFile(imagePath, modelPath) {
  const scriptPath = path.join(__dirname, 'yolo_infer.py');
  const args = modelPath ? [scriptPath, imagePath, modelPath] : [scriptPath, imagePath];
  const stdout = execFileSync(pythonExecutable, args, {
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe']
  });

  return JSON.parse(stdout);
}

function hasWesternOverride(analysis) {
  return analysis.foods?.some((food) => westernOverrideClasses.has(food.name.toLowerCase()));
}

function chooseBestAnalysis(primary, secondary) {
  if (!secondary) {
    return primary;
  }

  if (!primary.foods?.length) {
    return {
      ...secondary,
      analysis_source: 'local_yolo_secondary'
    };
  }

  if (hasWesternOverride(secondary)) {
    return {
      ...secondary,
      analysis_source: 'local_yolo_secondary',
      summary: `${secondary.summary} Secondary general food model was selected for non-Indian food coverage.`
    };
  }

  return primary;
}

function analyzeWithLocalYolo(imageBuffer) {
  const tempFilePath = path.join(os.tmpdir(), `food-analyzer-${Date.now()}.jpg`);
  fs.writeFileSync(tempFilePath, imageBuffer);

  try {
    const primaryModelPath = process.env.FOOD_MODEL_PATH || '';
    const secondaryModelPath = process.env.SECONDARY_FOOD_MODEL_PATH || '';
    const primary = analyzeImageFile(tempFilePath, primaryModelPath);
    let secondary = null;

    if (secondaryModelPath) {
      try {
        secondary = analyzeImageFile(tempFilePath, secondaryModelPath);
      } catch (secondaryError) {
        console.warn('Secondary YOLO model failed:', secondaryError.message);
      }
    }

    const selected = chooseBestAnalysis(
      { ...primary, analysis_source: 'local_yolo_primary' },
      secondary ? { ...secondary, analysis_source: 'local_yolo_secondary' } : null
    );

    return selected;
  } catch (error) {
    const stderr = error.stderr?.toString?.() || '';
    const stdout = error.stdout?.toString?.() || '';
    const message = stderr || stdout || error.message;
    throw new Error(`Local YOLO inference failed: ${message}`);
  } finally {
    try {
      if (fs.existsSync(tempFilePath)) {
        fs.unlinkSync(tempFilePath);
      }
    } catch {
      // Ignore temp cleanup failures.
    }
  }
}

app.get('/', (req, res) => {
  res.json({ message: 'Local YOLO Food Nutrition Analyzer API is running.' });
});

app.post('/analyze', upload.single('image'), (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'Please upload an image.' });
    }

    return res.json(analyzeWithLocalYolo(req.file.buffer));
  } catch (error) {
    console.error('YOLO analysis failed:', error);
    return res.status(500).json({
      error: 'Unable to analyze the image with the local YOLO model.',
      details: error.message
    });
  }
});

startServer(port);
