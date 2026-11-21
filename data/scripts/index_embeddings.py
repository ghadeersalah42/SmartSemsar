# import os
# from pathlib import Path
# import pandas as pd
# import chromadb
# from chromadb.utils import embedding_functions

# def populate_chroma_db():
#     # 1. تحديد الـ Project Root وتحديد مسار الـ CSV و ChromaDB
#     PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
#     CSV_PATH = PROJECT_ROOT / "data" / "final_merged_dataset.csv"
#     CHROMA_PATH = PROJECT_ROOT / "data" / "chroma_db"

#     print(f"📂 reading dataset from: {CSV_PATH}")
#     if not CSV_PATH.exists():
#         raise FileNotFoundError(f"❌ لم يتم العثور على ملف الـ CSV في المسار: {CSV_PATH}")

#     df = pd.read_csv(CSV_PATH)

#     # 2. إعداد ChromaDB وموديل الـ Embedding متعدد اللغات
#     print("🚀 Initializing ChromaDB client and SentenceTransformer model...")
#     ef = embedding_functions.SentenceTransformerEmbeddingFunction(
#         model_name="paraphrase-multilingual-MiniLM-L12-v2"
#     )
    
#     client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    
#     # تنظيف الـ Collection إذا كانت موجودة مسبقاً للبدء من جديد
#     try:
#         client.delete_collection(name="properties_semantic")
#         print("🧹 Collection قديمة تم مسحها لإعادة البناء...")
#     except Exception:
#         pass

#     collection = client.get_or_create_collection(
#         name="properties_semantic",
#         embedding_function=ef
#     )

#     # 3. إعداد البيانات للنص الدلالي والـ Metadata
#     ids = []
#     documents = []
#     metadatas = []

#     print("⚙️ Preparing text representations for Location Embeddings...")
#     for idx, row in df.iterrows():
#         # تأكيد معرف العقار
#         prop_id = str(row.get('property_id', idx))
        
#         # تجميع أجزاء الموقع والعنوان لبناء نص غني للـ Vector Search
        
#         location = str(row.get('location', '')) if pd.notna(row.get('location')) else ''
        
#         # دمجم الأماكن بفاصل واضح
#         text_content = f"{location}".strip(" |")
#         if not text_content:
#             text_content = "عقار بدون عنوان تفصيلي"

#         ids.append(prop_id)
#         documents.append(text_content)
        
#         # تخزين البيانات الأساسية كـ Metadata لسرعة الـ Filter داخل ChromaDB
#         metadatas.append({
#             "property_id": prop_id,
#             "location": location,
#             "price": float(row['price']) if pd.notna(row.get('price')) and str(row.get('price')).replace('.','',1).isdigit() else 0.0
#         })

#     # 4. الحفظ في ChromaDB على دفعات (Batching) لتفادي استهلاك الذاكرة
#     batch_size = 250
#     total_records = len(ids)
#     print(f"📦 Indexing {total_records} properties into ChromaDB...")

#     for i in range(0, total_records, batch_size):
#         end_idx = min(i + batch_size, total_records)
#         collection.add(
#             ids=ids[i:end_idx],
#             documents=documents[i:end_idx],
#             metadatas=metadatas[i:end_idx]
#         )
#         print(f"  ✓ Processed [{i} -> {end_idx}] / {total_records}")

#     print(f"\n✅ Finished! ChromaDB populated successfully at: {CHROMA_PATH}")

# if __name__ == "__main__":
#     populate_chroma_db()



######################################################################################################

# from pathlib import Path
# import pandas as pd
# import chromadb
# from chromadb.utils import embedding_functions

# PROPERTY_TYPE_WORDS = {
#     "studio": ["studio"],
#     "villa": ["villa", "stand alone", "standalone"],
#     "duplex": ["duplex"],
#     "townhouse": ["townhouse", "town house"],
# }


# def extract_property_type_from_title(title: str) -> str:
#     """استخراج نوع العقار من عنوان الإعلان نفسه (والافتراضي شقة / apartment)"""
#     t = str(title).lower() if pd.notna(title) else ""
#     for prop_type, words in PROPERTY_TYPE_WORDS.items():
#         if any(w in t for w in words):
#             return prop_type
#     return "apartment"


# def build_property_text(row, enrich: bool = True) -> str:
#     """بناء نص الـ embedding من بيانات العقار في الداتا بيز"""
#     location = str(row.get("location", "")).strip(" |") if pd.notna(row.get("location")) else ""

#     if not enrich:
#         return location or "عقار بدون عنوان تفصيلي"

#     parts = []
#     if location:
#         parts.append(location)

#     prop_type = extract_property_type_from_title(row.get("title", ""))
#     parts.append(prop_type)

#     rooms_val = row.get("pf_bedrooms")
#     if pd.notna(rooms_val) and str(rooms_val).strip() != "":
#         try:
#             rooms_num = int(float(rooms_val))
#             parts.append(f"{rooms_num} rooms")
#         except (ValueError, TypeError):
#             pass

#     price_val = row.get("price")
#     if pd.notna(price_val) and str(price_val).strip() != "":
#         try:
#             price_num = float(price_val)
#             if price_num > 0:
#                 parts.append(f"price {price_num:,.0f}")
#         except (ValueError, TypeError):
#             pass

#     return " | ".join(parts).strip() or "عقار بدون عنوان تفصيلي"


# def populate_chroma_db(enrich: bool = True):
#     PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
#     CSV_PATH = PROJECT_ROOT / "data" / "final_merged_dataset.csv"
#     CHROMA_PATH = PROJECT_ROOT / "data" / "chroma_db"

#     print(f"📂 reading dataset from: {CSV_PATH}")
#     if not CSV_PATH.exists():
#         raise FileNotFoundError(f"❌ لم يتم العثور على ملف الـ CSV في المسار: {CSV_PATH}")

#     df = pd.read_csv(CSV_PATH)

#     print("🚀 Initializing ChromaDB client and SentenceTransformer model...")
#     ef = embedding_functions.SentenceTransformerEmbeddingFunction(
#         model_name="paraphrase-multilingual-MiniLM-L12-v2"
#     )

#     client = chromadb.PersistentClient(path=str(CHROMA_PATH))

#     try:
#         client.delete_collection(name="properties_semantic")
#         print("🧹 Collection قديمة تم مسحها لإعادة البناء...")
#     except Exception:
#         pass

#     collection = client.get_or_create_collection(
#         name="properties_semantic",
#         embedding_function=ef
#     )

#     ids, documents, metadatas = [], [], []
#     mode_label = "Enriched (location + price + rooms + property type)" if enrich else "Location-only"
#     print(f"⚙️ Preparing embeddings [{mode_label}]...")

#     for idx, row in df.iterrows():
#         prop_id = str(row.get("property_id", idx))
#         location = str(row.get("location", "")) if pd.notna(row.get("location")) else ""
#         prop_type = extract_property_type_from_title(row.get("title", ""))

#         rooms_clean = 0
#         if pd.notna(row.get("pf_bedrooms")):
#             try:
#                 rooms_clean = int(float(row.get("pf_bedrooms")))
#             except (ValueError, TypeError):
#                 rooms_clean = 0

#         price_clean = 0.0
#         if pd.notna(row.get("price")):
#             try:
#                 price_clean = float(row.get("price"))
#             except (ValueError, TypeError):
#                 price_clean = 0.0

#         ids.append(prop_id)
#         documents.append(build_property_text(row, enrich=enrich))
#         metadatas.append({
#             "property_id": prop_id,
#             "location": location,
#             "price": price_clean,
#             "rooms": rooms_clean,
#             "property_type": prop_type,
#         })

#     batch_size = 250
#     total_records = len(ids)
#     print(f"📦 Indexing {total_records} properties into ChromaDB...")

#     for i in range(0, total_records, batch_size):
#         end_idx = min(i + batch_size, total_records)
#         collection.add(
#             ids=ids[i:end_idx],
#             documents=documents[i:end_idx],
#             metadatas=metadatas[i:end_idx]
#         )
#         print(f"  ✓ Processed [{i} -> {end_idx}] / {total_records}")

#     print(f"\n✅ Finished! ChromaDB populated successfully at: {CHROMA_PATH}")


# if __name__ == "__main__":
#     populate_chroma_db(enrich=True)

##################################

from pathlib import Path
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

PROPERTY_TYPE_WORDS = {
    "studio": ["studio"],
    "villa": ["villa", "stand alone", "standalone"],
    "duplex": ["duplex"],
    "townhouse": ["townhouse", "town house"],
}

MODEL_NAME = "intfloat/multilingual-e5-small"
COLLECTION_NAME = "properties_semantic"


def extract_property_type_from_title(title: str) -> str:
    t = str(title).lower() if pd.notna(title) else ""
    for prop_type, words in PROPERTY_TYPE_WORDS.items():
        if any(w in t for w in words):
            return prop_type
    return "apartment"


def build_property_text(row, enrich: bool = True) -> str:
    location = str(row.get("location", "")).strip(" |") if pd.notna(row.get("location")) else ""
    if not enrich:
        return location or "عقار بدون عنوان تفصيلي"

    parts = []
    if location:
        parts.append(location)

    prop_type = extract_property_type_from_title(row.get("title", ""))
    parts.append(prop_type)

    rooms_val = row.get("pf_bedrooms")
    if pd.notna(rooms_val) and str(rooms_val).strip() != "":
        try:
            rooms_num = int(float(rooms_val))
            parts.append(f"{rooms_num} rooms")
        except (ValueError, TypeError):
            pass

    price_val = row.get("price")
    if pd.notna(price_val) and str(price_val).strip() != "":
        try:
            price_num = float(price_val)
            if price_num > 0:
                parts.append(f"price {price_num:,.0f}")
        except (ValueError, TypeError):
            pass

    return " | ".join(parts).strip() or "عقار بدون عنوان تفصيلي"


def populate_chroma_db(enrich: bool = True):
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    CSV_PATH = PROJECT_ROOT / "data" / "final_merged_dataset.csv"
    CHROMA_PATH = PROJECT_ROOT / "data" / "chroma_db"

    print(f"📂 reading dataset from: {CSV_PATH}")
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"❌ لم يتم العثور على ملف الـ CSV في المسار: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    print(f"🚀 Initializing ChromaDB client and {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    try:
        client.delete_collection(name=COLLECTION_NAME)
        print("🧹 Collection قديمة تم مسحها لإعادة البناء...")
    except Exception:
        pass

    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    ids, documents, metadatas = [], [], []
    mode_label = "Enriched (location + price + rooms + property type)" if enrich else "Location-only"
    print(f"⚙️ Preparing embeddings [{mode_label}]...")

    for idx, row in df.iterrows():
        prop_id = str(row.get("property_id", idx))
        location = str(row.get("location", "")) if pd.notna(row.get("location")) else ""
        prop_type = extract_property_type_from_title(row.get("title", ""))

        rooms_clean = 0
        if pd.notna(row.get("pf_bedrooms")):
            try:
                rooms_clean = int(float(row.get("pf_bedrooms")))
            except (ValueError, TypeError):
                rooms_clean = 0

        price_clean = 0.0
        if pd.notna(row.get("price")):
            try:
                price_clean = float(row.get("price"))
            except (ValueError, TypeError):
                price_clean = 0.0

        ids.append(prop_id)
        documents.append(build_property_text(row, enrich=enrich))
        metadatas.append({
            "property_id": prop_id,
            "location": location,
            "price": price_clean,
            "rooms": rooms_clean,
            "property_type": prop_type,
        })

    batch_size = 250
    total_records = len(ids)
    print(f"📦 Indexing {total_records} properties into ChromaDB...")

    for i in range(0, total_records, batch_size):
        end_idx = min(i + batch_size, total_records)
        batch_docs = documents[i:end_idx]
        batch_embs = model.encode(
            batch_docs,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        ).tolist()

        collection.add(
            ids=ids[i:end_idx],
            documents=batch_docs,
            embeddings=batch_embs,
            metadatas=metadatas[i:end_idx],
        )
        print(f"  ✓ Processed [{i} -> {end_idx}] / {total_records}")

    print(f"\n✅ Finished! ChromaDB populated successfully at: {CHROMA_PATH}")


if __name__ == "__main__":
    populate_chroma_db(enrich=True)