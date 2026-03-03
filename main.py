from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. INITIALIZE API AND LOAD AI IN THE BACKGROUND ---
app = FastAPI(title="Sustainable Materials AI", version="2.0")

print("Loading dataset and AI Model... This takes a few seconds.")
# Load full dataset so we can sort dynamically
df = pd.read_csv('product_sustainability_dataset.csv')
unique_products = df['Product_Input'].unique()

# Load Model and compute vectors ONCE at startup to make the API blazing fast
model = SentenceTransformer('all-MiniLM-L6-v2')
product_embeddings = model.encode(unique_products)
print("✅ Server Ready!")

# --- 2. DEFINE THE INPUT FORMAT ---
class UserQuery(BaseModel):
    query: str
    preference: str = "cheapest"  # Options: "cheapest", "sustainable", "balanced"

# --- 3. THE API ENDPOINT ---
@app.post("/recommend")
def get_recommendation(request: UserQuery):
    # 1. Understand what the user means
    user_vector = model.encode([request.query])
    sims = cosine_similarity(user_vector, product_embeddings).flatten()
    
    best_match_idx = np.argmax(sims)
    confidence = sims[best_match_idx]
    
    if confidence < 0.40:
        raise HTTPException(status_code=404, detail="Product not found in database.")
        
    matched_product = unique_products[best_match_idx]
    
    # 2. Extract ALL materials available for this specific product
    product_data = df[df['Product_Input'] == matched_product].copy()
    
    # 3. Apply the Expanded Logic based on User Preference
    request.preference = request.preference.lower()
    
    if request.preference == "cheapest":
        # Sort by Price FIRST. If there's a tie, break the tie by picking the lowest EISc score.
        best_row = product_data.sort_values(
            by=['Alternative_Price_USD', 'EISc_Alternative_Score'], 
            ascending=[True, True]
        ).iloc[0]
        
    elif request.preference == "sustainable":
        # Sort by EISc FIRST. If there's a tie, break the tie by picking the cheapest price.
        best_row = product_data.sort_values(
            by=['EISc_Alternative_Score', 'Alternative_Price_USD'], 
            ascending=[True, True]
        ).iloc[0]
        
    elif request.preference == "balanced":
        # TRUE BALANCED AI LOGIC: Min-Max Normalization (0 to 1 scale)
        # This prevents a massive price gap from being ignored by a tiny EISc gap.
        min_price = product_data['Alternative_Price_USD'].min()
        max_price = product_data['Alternative_Price_USD'].max()
        
        min_eisc = product_data['EISc_Alternative_Score'].min()
        max_eisc = product_data['EISc_Alternative_Score'].max()
        
        # Avoid dividing by zero if all items have the exact same price/score
        price_range = (max_price - min_price) if (max_price - min_price) != 0 else 1
        eisc_range = (max_eisc - min_eisc) if (max_eisc - min_eisc) != 0 else 1
        
        product_data['price_norm'] = (product_data['Alternative_Price_USD'] - min_price) / price_range
        product_data['eisc_norm'] = (product_data['EISc_Alternative_Score'] - min_eisc) / eisc_range
        
        # Add the normalized scores together (Lower is better)
        product_data['combined_score'] = product_data['price_norm'] + product_data['eisc_norm']
        
        # Sort by the combined score, breaking any ties by checking the raw price
        best_row = product_data.sort_values(
            by=['combined_score', 'Alternative_Price_USD'], 
            ascending=[True, True]
        ).iloc[0]
        
    else:
        raise HTTPException(status_code=400, detail="Invalid preference. Use cheapest, sustainable, or balanced.")

    # 4. Return clean, structured JSON
    return {
        "metadata": {
            "matched_product": matched_product,
            "confidence_score": round(float(confidence) * 100, 2),
            "optimization_strategy": request.preference
        },
        "original_material": {
            "name": best_row['Cheapest_Original_Material'],
            "price_usd": float(best_row['Original_Price_USD']),
            "eisc_score": int(best_row['EISc_Original_Score'])
        },
        "recommended_alternative": {
            "name": best_row['Sustainable_Alternative'],
            "price_usd": float(best_row['Alternative_Price_USD']),
            "eisc_score": int(best_row['EISc_Alternative_Score'])
        }
    }