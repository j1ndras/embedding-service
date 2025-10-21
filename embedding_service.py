#!/usr/bin/env python3
"""
Self-Hosted Embedding Service
Model: all-MiniLM-L6-v2 (384 dimensions)
Fast, lightweight, runs locally!
"""

from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load model (runs on first request, cached after)
logger.info("Loading model: all-MiniLM-L6-v2...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
logger.info("Model loaded successfully! Ready to serve requests.")

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model': 'all-MiniLM-L6-v2',
        'dimensions': 384
    })

@app.route('/embed', methods=['POST'])
def embed():
    """
    Generate embedding for single text
    
    Request:
        {
            "text": "donat cokelat"
        }
    
    Response:
        {
            "embedding": [0.123, 0.456, ...],
            "dimensions": 384
        }
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing "text" field'}), 400
        
        text = data['text']
        
        if not text or not text.strip():
            return jsonify({'error': 'Text cannot be empty'}), 400
        
        # Generate embedding
        embedding = model.encode(text, convert_to_numpy=True)
        
        # Convert to list for JSON serialization
        embedding_list = embedding.tolist()
        
        return jsonify({
            'embedding': embedding_list,
            'dimensions': len(embedding_list),
            'text': text
        })
        
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/embed/batch', methods=['POST'])
def embed_batch():
    """
    Generate embeddings for multiple texts
    
    Request:
        {
            "texts": ["donat cokelat", "spiku besar"]
        }
    
    Response:
        {
            "embeddings": [[0.123, ...], [0.456, ...]],
            "count": 2
        }
    """
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({'error': 'Missing "texts" field'}), 400
        
        texts = data['texts']
        
        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({'error': 'texts must be a non-empty array'}), 400
        
        # Filter empty texts
        texts = [t for t in texts if t and t.strip()]
        
        if len(texts) == 0:
            return jsonify({'error': 'All texts are empty'}), 400
        
        # Generate embeddings (batched for efficiency)
        embeddings = model.encode(texts, convert_to_numpy=True)
        
        # Convert to list
        embeddings_list = embeddings.tolist()
        
        return jsonify({
            'embeddings': embeddings_list,
            'count': len(embeddings_list),
            'dimensions': 384
        })
        
    except Exception as e:
        logger.error(f"Error generating batch embeddings: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/similarity', methods=['POST'])
def similarity():
    """
    Calculate cosine similarity between two texts
    
    Request:
        {
            "text1": "donat cokelat",
            "text2": "donat coklat"
        }
    
    Response:
        {
            "similarity": 0.95
        }
    """
    try:
        data = request.get_json()
        
        if not data or 'text1' not in data or 'text2' not in data:
            return jsonify({'error': 'Missing text1 or text2'}), 400
        
        text1 = data['text1']
        text2 = data['text2']
        
        # Generate embeddings
        embeddings = model.encode([text1, text2], convert_to_numpy=True)
        
        # Calculate cosine similarity
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        
        return jsonify({
            'similarity': float(similarity),
            'text1': text1,
            'text2': text2
        })
        
    except Exception as e:
        logger.error(f"Error calculating similarity: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import os
    
    # Configuration
    host = os.getenv('EMBED_HOST', '0.0.0.0')
    port = int(os.getenv('EMBED_PORT', 5000))
    
    logger.info(f"Starting embedding service on {host}:{port}")
    logger.info("Model: all-MiniLM-L6-v2 (384 dimensions)")
    logger.info("Ready to generate embeddings!")
    
    # Run Flask app
    app.run(host=host, port=port, threaded=True)
