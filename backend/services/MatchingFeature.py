import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import os
from backend.schema.state import CustomerRequirements
import os
import pandas as pd
from typing import Optional, Literal, List, Dict, Any
from pydantic import BaseModel, Field


class PropertyMatchingService:
    def __init__(self, csv_path: str, chroma_path: str):
        print("---enter---")
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        print("---after csv---")
        
        # 1. إعداد ChromaDB والـ Multilingual Embedding Function
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.client = chromadb.PersistentClient(path=chroma_path)
        
        self.collection = self.client.get_or_create_collection(
            name="properties_semantic", 
            embedding_function=self.ef
        )
        print("---final---")

    def _filter_dataframe( self, max_price: float = None, bedrooms: int = None, area: float = None,ignore_rooms_and_area: bool = False) -> list:
        filtered = self.df.copy()
        print("---filter---")

        # 1. الحفاظ على شرط السعر دائمًا (Budget Strict Filter)
        if max_price and max_price > 0:
            filtered = filtered[filtered['price'] <= (max_price+max_price*0.10)]

        # 2. تطبيق شروط الغرف والمساحة فقط في المحاولة الأولى
        if not ignore_rooms_and_area:
            if bedrooms and 'pf_bedrooms' in filtered.columns:
                filtered = filtered[filtered['pf_bedrooms'] == bedrooms]

            if area and 'pf_area_sqm' in filtered.columns:
                filtered = filtered[(filtered['pf_area_sqm'] >= (area-(area*0.15)) )& (filtered['pf_area_sqm'] <=  (area+(area*0.15)))]

        # إرجاع قائمة الـ IDs المطابقة للـ Hard Constraints
        return filtered['property_id'].astype(str).tolist()
    

    def search_properties(self, reqs: CustomerRequirements={},Purpose: str="", min_results_threshold: int = 3) -> dict:
            """
            البحث الذكي ثنائي المرحلة مع آلية التنازل التدريجي (Fallback)
            """
            if Purpose=="general_browse":
                return {
                "total_found": len(self.df),
                "used_fallback": False,
                "properties": self.df.head(15).copy().to_dict(orient='records')
            }

            print(f"\n📥 [Input Req] Location: '{reqs.location}' | Max Budget: {reqs.budget_max} | Rooms: {reqs.bedrooms} | Area: {reqs.area_sqm}")
            
            # user_location_query = user_requirements.get("location", "")
            # max_price = user_requirements.get("budget_max")
            # bedrooms = user_requirements.get("bedrooms")
            # area_sqm = user_requirements.get("area_sqm")

            # # إعداد حدود المساحة بـ Margin ±15%
            # min_area = area_sqm * 0.85 if area_sqm else None
            # max_area = area_sqm * 1.15 if area_sqm else None

            # -------------------------------------------------------------------
            # 🟢 المحاولة الأولى: الفلترة الصارمة (Strict Search)
            # -------------------------------------------------------------------
            print("🔍 [Attempt 1] جاري الفلترة الصارمة (السعر + الغرف + المساحة)...")
            candidate_ids = self._filter_dataframe(
                max_price=reqs.budget_max,
                bedrooms=reqs.bedrooms,
                area=reqs.area_sqm,
                ignore_rooms_and_area=False
            )

            results = self._rank_by_location_embeddings(reqs.location, candidate_ids)

            # -------------------------------------------------------------------
            # 🟡 المحاولة الثانية: Fallbacks (الحفاظ على السعر والمكان مع إلغاء باقي الشروط)
            # -------------------------------------------------------------------
            used_fallback = False
            if len(results) < min_results_threshold:
                used_fallback = True
                print(f"⚠️ النتائج غير كافية ({len(results)} عقارات). جاري تطبيق الـ Fallback Strategy...")
                print("🔄 [Attempt 2] الحفاظ على السعر والمكان، مع التنازل عن الغرف والمساحة...")

                fallback_candidate_ids = self._filter_dataframe(
                    max_price=reqs.budget_max,
                    ignore_rooms_and_area=True  # إلغاء قيود الغرف والمساحة
                )

                results = self._rank_by_location_embeddings(reqs.location, fallback_candidate_ids)

            return {
                "total_found": len(results),
                "used_fallback": used_fallback,
                "properties": results
            }

    def _rank_by_location_embeddings(self, location_query: str, candidate_ids: list, top_k: int = 5) -> list:
        """ترتيب قائمة العقارات المرشحة باستخدام Vector Similarity في ChromaDB"""
        if not candidate_ids:
            return []

        # إذا لم يحدد المستخدم مكانًا، نرجع نتائج الفلترة المباشرة
        if not location_query:
            return self.df[self.df['property_id'].astype(str).isin(candidate_ids[:top_k])].to_dict(orient='records')

        # استعلام ChromaDB بالـ Embeddings مع تحديد الـ IDs المفلترة مسبقاً (Where Metadata)
        chroma_results = self.collection.query(
            query_texts=[location_query], # يعمل سواء النص عربي أو إنجليزي
            where={"property_id": {"$in": candidate_ids}},
            n_results=min(top_k, len(candidate_ids))
        )

        matched_ids = chroma_results['ids'][0] if chroma_results['ids'] else []

        # جلب تفاصيل العقارات الكاملة من الـ DataFrame مرتبة حسب تشابه المكان
        matched_df = self.df[self.df['property_id'].astype(str).isin(matched_ids)]
        return matched_df.to_dict(orient='records')
    
    def get_property_by_id(self, property_id: str) -> dict:
        matched = self.df[self.df['property_id'].astype(str) == str(property_id)]
        if not matched.empty:
            return matched.iloc[0].to_dict()
        return None


# if __name__ == "__main__":
#     BASE_DIR = os.path.dirname(__file__)
#     CSV_PATH = os.path.abspath(os.path.join(BASE_DIR, "../data/final_merged_dataset.csv"))
#     CHROMA_PATH = os.path.abspath(os.path.join(BASE_DIR, "../data/chroma_db"))

#     matcher = PropertyMatchingService(csv_path=CSV_PATH, chroma_path=CHROMA_PATH)

#     # برومبت يطلب شقة في التجمع الخامس بسعر محدد
#     sample_requirements = {
#         "location_query": "شقة للبيع في التجمع الخامس", # عربي عادي
#         "max_price": 5000000,                            # 5 مليون
#         "bedrooms": 3,
#         "area_sqm": 150
#     }

#     final_properties = matcher.search_properties(sample_requirements)
#     print(f"\n✅ إجمالي العقارات المسترجعة: {len(final_properties)}")
#     for p in final_properties:
#         print(f"📌 [{p.get('property_id')}] -{p.get('title')} | السعر: {p.get('price')} | الموقع: {p.get('location')}")



# if __name__ == "__main__":
#     print("---lll---")
#     CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
#     print("---lll---")
#     # 2. الرجوع خطوتين للوصول لـ Root المشروع (PropTech-AI-Platform)
#     PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
#     print("---lll---")

#     # 3. تحديد مسارات الـ data و chroma الصح
#     CSV_PATH = os.path.join(PROJECT_ROOT, "data", "final_merged_dataset.csv")
#     print("---lll---")
#     CHROMA_PATH = os.path.join(PROJECT_ROOT, "data", "chroma_db")
#     print("---lll---")
#     # 1. تقومين بتهيئة الـ Service
#     matcher = PropertyMatchingService(csv_path=CSV_PATH, chroma_path=CHROMA_PATH)
#     print("---lll---")

#     # 2. تجهيز حالات تجريبية كـ CustomerRequirements Objects مباشرة
#     test_case_1 = CustomerRequirements(
#         location="التجمع الخامس",
#         budget_max=6000000.0,
#         bedrooms=3,
#         area_sqm=160.0,
#         property_type="apartment"
#     )
#     print("---lll---")

#     test_case_impossible_rooms = CustomerRequirements(
#         location="الشيخ زايد",
#         budget_max=3000000.0,
#         bedrooms=10, # شرط تعجيزي لتجربة الـ Fallback
#         area_sqm=500.0
#     )
#     print("---lll---")

#     # 3. تشغيل الفانكشن مباشرة بدون أي Nodes!
#     print("----------------------------------------")
#     print("🧪 Test 1: Normal Search")
#     res1 = matcher.search_properties(test_case_1)
#     print(f"✅ Total Found: {res1['total_found']} | Fallback Used: {res1['used_fallback']}")
#     print("🏠 [نتائج Test 1]:")
#     for idx, p in enumerate(res1['properties'], 1):
#         print(f"  {idx}. ID: {p.get('property_id')} | Title: {p.get('title')}")
#         print(f"     💰 السعر: {p.get('price'):,} ج.م | 🛏️ الغرف: {p.get('pf_bedrooms')} | 📐 المساحة: {p.get('pf_area_sqm')} م² | 📍 الموقع: {p.get('location')}\n")

#     print("\n----------------------------------------")
#     print("🧪 Test 2: Fallback Trigger Search")
#     res2 = matcher.search_properties(test_case_impossible_rooms)
#     print(f"✅ Total Found: {res2['total_found']} | Fallback Used: {res2['used_fallback']}")
#     print("🏠 [نتائج Test 2 (Fallback)]:")
#     for idx, p in enumerate(res2['properties'], 1):
#         print(f"  {idx}. ID: {p.get('property_id')} | Title: {p.get('title')}")
#         print(f"     💰 السعر: {p.get('price'):,} ج.م | 🛏️ الغرف: {p.get('pf_bedrooms')} | 📐 المساحة: {p.get('pf_area_sqm')} م² | 📍 الموقع: {p.get('location')}\n")