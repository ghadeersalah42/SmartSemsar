import os
import zipfile
import pandas as pd
from dotenv import load_dotenv

# 1. تحميل مفاتيح البيئة من .env
load_dotenv()

# ضبط متغيرات Kaggle للـ Authentication تلقائياً
os.environ['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USERNAME', '')
os.environ['KAGGLE_KEY'] = os.getenv('KAGGLE_KEY', '')

# المسارات الأساسية
DATA_DIR = "data_pipeline"
FLOORPLANS_DIR = os.path.join(DATA_DIR, "floorplans")
os.makedirs(FLOORPLANS_DIR, exist_ok=True)

# 📌 استبدلي هذا باسم الداتاسيت المطلوبة من كاجل (موجودة في رابط الداتاسيت على Kaggle)
# مثال: https://www.kaggle.com/datasets/mrdwivedi/property-finder-egypt-data
# يكون الـ Slug هو: "mrdwivedi/property-finder-egypt-data"
KAGGLE_DATASET_SLUG = "fikrykandil/egypt-property-finder-cleaned"
OUTPUT_CSV = os.path.join(DATA_DIR, "cleaned_all_egypt.csv")


def format_location(row):
    """
    دمج أعمدة الموقع في نص موحد ونظيف مع التعامل مع القيم الناقصة
    """
    parts = []
    
    # 1. المجمع السكني (Compound) إن وجد
    if pd.notna(row.get('compound')) and str(row.get('compound')).strip():
        parts.append(str(row['compound']).strip())
        
    # 2. الحي / المنطقة (District) إن وجد
    if pd.notna(row.get('district')) and str(row.get('district')).strip():
        parts.append(str(row['district']).strip())
        
    # 3. المدينة (City)
    if pd.notna(row.get('city')) and str(row.get('city')).strip():
        parts.append(str(row['city']).strip())
        
    # 4. المحافظة (region / Governorate)
    region_val = row.get('region') if pd.notna(row.get('region')) else row.get('Governorate')
    if pd.notna(region_val) and str(region_val).strip():
        parts.append(str(region_val).strip())
        
    return ", ".join(parts) if parts else "القاهرة"


def download_and_prepare_dataset():
    if not os.environ['KAGGLE_USERNAME'] or not os.environ['KAGGLE_KEY']:
        raise ValueError("⚠️ يرجى ضبط KAGGLE_USERNAME و KAGGLE_KEY داخل ملف .env أولاً!")

    print(f"📥 جاري سحب الداتاسيت مباشرة من Kaggle ({KAGGLE_DATASET_SLUG})...")
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(KAGGLE_DATASET_SLUG, path=DATA_DIR, unzip=True)
        print("✅ تم سحب وفك ضغط الداتاسيت بنجاح!")
    except Exception as e:
        print(f"❌ حدث خطأ أثناء السحب من Kaggle: {e}")
        return

    # قراءة ملف CSV المحمل
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv') and f != "properties.csv"]
    if not csv_files:
        print("⚠️ لم يتم العثور على ملفات CSV محملة داخل المجلد!")
        return
    
    raw_csv_path = os.path.join(DATA_DIR, csv_files[0])
    print(f"🔄 جاري معالجة وتنظيف الموقع والأسعار من: {raw_csv_path}")

    df = pd.read_csv(raw_csv_path)

    # 1. بناء عمود location موحد من أعمدة التقسيم الجغرافي
    df['location'] = df.apply(format_location, axis=1)

    # 2. توحيد بقية أسماء الأعمدة الأساسية
    column_mapping = {
        'Title': 'title', 'title_ar': 'title', 'name': 'title',
        'Price': 'price', 'price_egp': 'price', 'Price (EGP)': 'price',
        'Bedrooms': 'bedrooms', 'beds': 'bedrooms',
        'Bathrooms': 'bathrooms', 'baths': 'bathrooms',
        'Area': 'area_sqm', 'size': 'area_sqm', 'area': 'area_sqm'
    }
    df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns}, inplace=True)

    # 3. تنظيف الأسعار والمساحات
    if 'price' in df.columns:
        df['price'] = df['price'].astype(str).str.replace(r'[^\d.]', '', regex=True)
        df['price'] = pd.to_numeric(df['price'], errors='coerce')

    # 4. إضافة ID فريد لكل عقار
    df['property_id'] = [f"PROP_{i+1001}" for i in range(len(df))]

    # 5. استخراج البيانات المنظفة والنهائية
    required_cols = ['property_id', 'title', 'price', 'location', 'bedrooms', 'bathrooms', 'area_sqm']
    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    df_final = df[required_cols].dropna(subset=['price']).head(100).copy()

    # حفظ الملف
    df_final.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"🎉 تم تجهيز داتاسيت العقارات بنجاح مع تركيب العناوين الجغرافية في: {OUTPUT_CSV}")

# if __name__ == "__main__":
#     download_and_prepare_dataset()