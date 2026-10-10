import pandas as pd
import numpy as np
import os

def merge_datasets():
    print("📖 جاري تحميل بيانات Property Finder و CubiCasa...")
    
    pf_path = "./data_pipeline/data/cleaned_all_egypt.csv"
    cc_path = "./data_pipeline/data/cubicasa_parsed.csv"
    
    if not os.path.exists(pf_path) or not os.path.exists(cc_path):
        print("❌ خطأ: تأكدي من وجود الملفين داخل فولدر ./data/")
        return

    df_pf = pd.read_csv(pf_path)
    df_cc = pd.read_csv(cc_path)
    
    merged_rows = []
    print("🔗 جاري المطابقة بناءً على عدد الغرف، الحمامات، وأقرب مساحة...")
    
    for _, prop in df_pf.iterrows():
        p_beds = int(prop['bedrooms'])
        p_baths = int(prop['bathrooms'])
        p_area = float(prop['area_sqm'])
        
        # 1. التصفية حسب عدد الغرف والحمامات
        candidates = df_cc[
            (df_cc['svg_bedrooms'] == p_beds) & 
            (df_cc['svg_bathrooms'] == p_baths)
        ]
        
        if candidates.empty:
            candidates = df_cc[df_cc['svg_bedrooms'] == p_beds]
            
        if candidates.empty:
            candidates = df_cc
            
        # 2. المطابقة بأقرب مساحة
        size_diffs = np.abs(candidates['svg_area_sqm'] - p_area)
        best_match = candidates.loc[size_diffs.idxmin()]
        
        image_name = str(best_match.get('local_image_name', f"{best_match['cubicasa_id']}.png"))
        
        merged_rows.append({
            'property_id': prop['property_id'],
            'title': prop['title'],
            'price': prop['price'],
            'location': prop['location'],
            'pf_bedrooms': p_beds,
            'pf_bathrooms': p_baths,
            'pf_area_sqm': p_area,
            
            # البيانات المستخرجة والمطابقة من CubiCasa
            'cubicasa_id': best_match['cubicasa_id'],
            'svg_bedrooms': best_match['svg_bedrooms'],
            'svg_bathrooms': best_match['svg_bathrooms'],
            'svg_area_sqm': best_match['svg_area_sqm'],
            'area_diff_sqm': round(abs(best_match['svg_area_sqm'] - p_area), 2),
            'image_path': f"./data_pipeline/data/matched_images/{image_name}"
        })
        
    df_merged = pd.DataFrame(merged_rows)
    output_path = "./data_pipeline/data/final_merged_dataset.csv"
    df_merged.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"🎉 تم الدمج بنجاح! الملف النهائي جاهز في: {output_path}")

# if __name__ == "__main__":
#     merge_datasets()