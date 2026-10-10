import os
from pathlib import Path
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

def populate_chroma_db():
    # 1. تحديد الـ Project Root وتحديد مسار الـ CSV و ChromaDB
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    CSV_PATH = PROJECT_ROOT / "data" / "final_merged_dataset.csv"
    CHROMA_PATH = PROJECT_ROOT / "data" / "chroma_db"

    print(f"📂 reading dataset from: {CSV_PATH}")
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"❌ لم يتم العثور على ملف الـ CSV في المسار: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    # 2. إعداد ChromaDB وموديل الـ Embedding متعدد اللغات
    print("🚀 Initializing ChromaDB client and SentenceTransformer model...")
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    
    # تنظيف الـ Collection إذا كانت موجودة مسبقاً للبدء من جديد
    try:
        client.delete_collection(name="properties_semantic")
        print("🧹 Collection قديمة تم مسحها لإعادة البناء...")
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name="properties_semantic",
        embedding_function=ef
    )

    # 3. إعداد البيانات للنص الدلالي والـ Metadata
    ids = []
    documents = []
    metadatas = []

    print("⚙️ Preparing text representations for Location Embeddings...")
    for idx, row in df.iterrows():
        # تأكيد معرف العقار
        prop_id = str(row.get('property_id', idx))
        
        # تجميع أجزاء الموقع والعنوان لبناء نص غني للـ Vector Search
        
        location = str(row.get('location', '')) if pd.notna(row.get('location')) else ''
        
        # دمجم الأماكن بفاصل واضح
        text_content = f"{location}".strip(" |")
        if not text_content:
            text_content = "عقار بدون عنوان تفصيلي"

        ids.append(prop_id)
        documents.append(text_content)
        
        # تخزين البيانات الأساسية كـ Metadata لسرعة الـ Filter داخل ChromaDB
        metadatas.append({
            "property_id": prop_id,
            "location": location,
            "price": float(row['price']) if pd.notna(row.get('price')) and str(row.get('price')).replace('.','',1).isdigit() else 0.0
        })

    # 4. الحفظ في ChromaDB على دفعات (Batching) لتفادي استهلاك الذاكرة
    batch_size = 250
    total_records = len(ids)
    print(f"📦 Indexing {total_records} properties into ChromaDB...")

    for i in range(0, total_records, batch_size):
        end_idx = min(i + batch_size, total_records)
        collection.add(
            ids=ids[i:end_idx],
            documents=documents[i:end_idx],
            metadatas=metadatas[i:end_idx]
        )
        print(f"  ✓ Processed [{i} -> {end_idx}] / {total_records}")

    print(f"\n✅ Finished! ChromaDB populated successfully at: {CHROMA_PATH}")

# if __name__ == "__main__":
#     populate_chroma_db()