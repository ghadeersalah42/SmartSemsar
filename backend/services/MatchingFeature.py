# import pandas as pd
# import chromadb
# from chromadb.utils import embedding_functions
# import os
# from backend.schema.state import CustomerRequirements
# import os
# import pandas as pd
# from typing import Optional, Literal, List, Dict, Any
# from pydantic import BaseModel, Field


# LOCATION_ALIASES = {
#     "التجمع الخامس": ["The 5th Settlement"], "التجمع الاول": ["The 1st Settlement"], "التجمع": ["Settlement", "New Cairo"],
#     "القاهرة الجديدة": ["New Cairo"], "نيو كايرو": ["New Cairo"], "القطامية": ["Katameya"], "الرحاب": ["Rehab"],
#     "مدينتي": ["Madinaty"], "اللوتس": ["Lotus"], "النرجس": ["Narges"], "الاندلس": ["Andalus"], "هايد بارك": ["Hyde Park"],
#     "زايد": ["Sheikh Zayed"], "اكتوبر": ["6 October"], "المعادي": ["Maadi"], "دجلة": ["Degla"], "الزمالك": ["Zamalek"],
#     "جاردن سيتي": ["Garden City"], "مدينة نصر": ["Nasr City"], "الهرم": ["Haram"], "المقطم": ["Mokattam"],
#     "اسكندرية": ["Alexandria"], "الاسكندرية": ["Alexandria"], "سموحة": ["Smouha"], "الشيخ زايد": ["Sheikh Zayed"],
# }


# # المساحة اللي العميل قالها ± 10% (180 متر -> من 162 لـ 198)
# AREA_TOLERANCE = 0.10

# # نوع العقار من العنوان (الداتا مفيهاش عمود للنوع)
# PROPERTY_TYPE_WORDS = {
#     "studio": ["studio"],
#     "villa": ["villa", "stand alone", "standalone"],
#     "duplex": ["duplex"],
#     "townhouse": ["townhouse", "town house"],
# }

# def location_keywords(location_query: str) -> list:
#     q = str(location_query).replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
#     for arabic in sorted(LOCATION_ALIASES, key=len, reverse=True):
#         if arabic in q:
#             return LOCATION_ALIASES[arabic]
#     return []


# class PropertyMatchingService:
#     def __init__(self, csv_path: str, chroma_path: str):
#         print("---enter---")
#         self.csv_path = csv_path
#         self.df = pd.read_csv(csv_path)
#         print("---after csv---")
        
#         # 1. إعداد ChromaDB والـ Multilingual Embedding Function
#         self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
#             model_name="paraphrase-multilingual-MiniLM-L12-v2"
#         )
#         self.client = chromadb.PersistentClient(path=chroma_path)
        
#         self.collection = self.client.get_or_create_collection(
#             name="properties_semantic", 
#             embedding_function=self.ef
#         )
#         # الـ index مش بيترفع على git: لو فاضي (أول تشغيل على جهاز جديد)، نبنيه من الـ CSV،
#         # وإلا ترتيب المكان بيرجع فاضي وكل بحث بيطلع 0 نتيجة
#         if self.collection.count() == 0:
#             self._build_location_index()
#         print("---final---")

#     def _build_location_index(self):
       
#         ids, documents, metadatas = [], [], []

#         for idx, row in self.df.iterrows():
#             prop_id = str(row.get('property_id', idx))
#             location = str(row['location']) if pd.notna(row.get('location')) else ''
#             ids.append(prop_id)
#             documents.append(location.strip(" |") or "")
#             metadatas.append({"property_id": prop_id, "location": location})

#         for i in range(0, len(ids), 100):
#             self.collection.add(ids=ids[i:i + 100], documents=documents[i:i + 100],
#                                 metadatas=metadatas[i:i + 100])

#     def _filter_dataframe(self, reqs: CustomerRequirements, ignore_rooms_and_area: bool = False) -> list:

#         filtered = self.df.copy()

#         if reqs.budget_max and reqs.budget_max > 0:
#             filtered = filtered[filtered['price'] <= (reqs.budget_max*1.10)]
#         if reqs.budget_min and reqs.budget_min > 0:
#             filtered = filtered[filtered['price'] >= (reqs.budget_min*0.9)]

#         if not ignore_rooms_and_area:
#             if reqs.bedrooms and 'pf_bedrooms' in filtered.columns:
#                 filtered = filtered[filtered['pf_bedrooms'] == reqs.bedrooms]

#             if reqs.area_sqm and 'pf_area_sqm' in filtered.columns:
#                 filtered = filtered[
#                     (filtered['pf_area_sqm'] >= (reqs.area_sqm * (1 - AREA_TOLERANCE))) & 
#                     (filtered['pf_area_sqm'] <= (reqs.area_sqm * (1 + AREA_TOLERANCE)))
#                 ]
 
#         return filtered['property_id'].astype(str).tolist()

#     def search_properties(self, reqs: CustomerRequirements = None, Purpose: str = "", min_results_threshold: int = 1,top_k: Optional[int] = None) -> dict:
#         """الشقق اللي بتحقق كل طلبات العميل، الأقرب للمساحة المطلوبة الأول. لو مفيش، بترجع فاضية"""
#         if Purpose == "general_browse":
#             return {
#                 "total_found": len(self.df),
#                 "used_fallback": False,
#                 "properties": self.df.copy().to_dict(orient='records')
#             }

#         print(f"\n📥 [Input Req] Location: '{reqs.location}' | Budget: {reqs.budget_min}-{reqs.budget_max} | Rooms: {reqs.bedrooms} | Area: {reqs.area_sqm} | Type: {reqs.property_type}")
#         print("🔍 [Attempt 1] جاري الفلترة الصارمة...")

#         # -------------------------------------------------------------------
#         # 🟢 المحاولة الأولى: الفلترة الصارمة (Strict Search)
#         # -------------------------------------------------------------------
        
#         candidate_ids = self._filter_dataframe(reqs, ignore_rooms_and_area=False)
#         results = self._rank_by_location_and_area(reqs.location, candidate_ids, reqs.area_sqm, top_k)

#         # -------------------------------------------------------------------
#         # 🟡 المحاولة الثانية: Fallback Strategy (الحفاظ على السعر والمكان وتجاهل الغرف والمساحة)
#         # -------------------------------------------------------------------
#         used_fallback = False
#         if len(results) < min_results_threshold:
#             used_fallback = True
#             print(f"⚠️ النتائج غير كافية ({len(results)} عقارات). جاري تطبيق الـ Fallback Strategy...")
#             print("🔄 [Attempt 2] الحفاظ على السعر والمكان، مع التنازل عن الغرف والمساحة...")

#             fallback_candidate_ids = self._filter_dataframe(reqs, ignore_rooms_and_area=True)
#             results = self._rank_by_location_and_area(reqs.location, fallback_candidate_ids, target_area=None, top_k=top_k)
#         print(f"✅ {len(results)} matching properties (Fallback Used: {used_fallback})")
#         return {"total_found": len(results), "used_fallback": used_fallback, "properties": results}

#     def _rank_by_location_and_area(self, location_query: str, candidate_ids: list, target_area: float = None, top_k: Optional[int] = None) -> list:
#         """ترتيب النتائج باستخدام الـ Aliases اللغوية أولاً، ثم ChromaDB Embeddings، ثم الأقرب للمساحة المطلوبة"""
#         if not candidate_ids:
#             return []

#         candidates_df = self.df[self.df['property_id'].astype(str).isin(candidate_ids)]
#         if candidates_df.empty:
#             return []

#         if location_query:
#             keywords = location_keywords(location_query)
#             if keywords:
#                 in_area = candidates_df['location'].fillna('').str.contains('|'.join(keywords), case=False, regex=True)
#                 candidates_df = candidates_df[in_area]
#             elif self.collection.count():
#                 ids = candidates_df['property_id'].astype(str).tolist()
#                 chroma_results = self.collection.query(
#                     query_texts=[location_query], 
#                     where={"property_id": {"$in": ids}},
#                     n_results=min(top_k, len(ids))
#                 )
#                 matched_ids = chroma_results['ids'][0] if chroma_results['ids'] else []
#                 candidates_df = candidates_df[candidates_df['property_id'].astype(str).isin(matched_ids)]

#         if candidates_df.empty:
#             return []

#         if target_area:
#             candidates_df = candidates_df.assign(_diff=(candidates_df['pf_area_sqm'] - target_area).abs())
#             candidates_df = candidates_df.sort_values(['_diff', 'price']).drop(columns='_diff')
#         else:
#             candidates_df = candidates_df.sort_values('price')

#         if top_k is not None:
#             candidates_df = candidates_df.head(top_k)

#         return candidates_df.head(top_k).to_dict(orient='records')
    
#     def get_property_by_id(self, property_id: str) -> dict:
#         matched = self.df[self.df['property_id'].astype(str) == str(property_id)]
#         if not matched.empty:
#             return matched.iloc[0].to_dict()
#         return None



from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from backend.schema.state import CustomerRequirements
from backend.ai_config import model, threshold

LOCATION_ALIASES = {
    "التجمع الخامس": ["The 5th Settlement"], "التجمع الاول": ["The 1st Settlement"], "التجمع": ["Settlement", "New Cairo"],
    "القاهرة الجديدة": ["New Cairo"], "نيو كايرو": ["New Cairo"], "القطامية": ["Katameya"], "الرحاب": ["Rehab"],
    "مدينتي": ["Madinaty"], "اللوتس": ["Lotus"], "النرجس": ["Narges"], "الاندلس": ["Andalus"], "هايد بارك": ["Hyde Park"],
    "زايد": ["Sheikh Zayed"], "اكتوبر": ["6 October"], "المعادي": ["Maadi"], "دجلة": ["Degla"], "الزمالك": ["Zamalek"],
    "جاردن سيتي": ["Garden City"], "مدينة نصر": ["Nasr City"], "الهرم": ["Haram"], "المقطم": ["Mokattam"],
    "اسكندرية": ["Alexandria"], "الاسكندرية": ["Alexandria"], "سموحة": ["Smouha"], "الشيخ زايد": ["Sheikh Zayed"],
}

AREA_TOLERANCE = threshold("matching.area_tolerance")

PROPERTY_TYPE_WORDS = {
    "studio": ["studio"],
    "villa": ["villa", "stand alone", "standalone"],
    "duplex": ["duplex"],
    "townhouse": ["townhouse", "town house"],
}

E5_MODEL_NAME = model("embedding")
RERANKER_MODEL_NAME = model("reranker")


def extract_property_type(title: Any) -> str:
    t = str(title).lower() if pd.notna(title) else ""
    for ptype, words in PROPERTY_TYPE_WORDS.items():
        if any(w in t for w in words):
            return ptype
    return "apartment"


def build_enriched_text(row: pd.Series) -> str:
    loc = str(row.get("location", "")).strip(" |") if pd.notna(row.get("location")) else ""
    ptype = extract_property_type(row.get("title", ""))

    rooms_val = row.get("pf_bedrooms")
    rooms_str = ""
    if pd.notna(rooms_val) and str(rooms_val).strip() != "":
        try:
            rooms_str = f"{int(float(rooms_val))} rooms"
        except (ValueError, TypeError):
            rooms_str = ""

    price_val = row.get("price")
    price_str = ""
    if pd.notna(price_val) and str(price_val).strip() != "":
        try:
            price_num = float(price_val)
            if price_num > 0:
                price_str = f"price {price_num:,.0f}"
        except (ValueError, TypeError):
            price_str = ""

    parts = [p for p in [loc, ptype, rooms_str, price_str] if p]
    return " | ".join(parts) or "عقار"


def location_keywords(location_query: str) -> list:
    q = str(location_query).replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    for arabic in sorted(LOCATION_ALIASES, key=len, reverse=True):
        if arabic in q:
            return LOCATION_ALIASES[arabic]
    return []


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    s_min, s_max = float(scores.min()), float(scores.max())
    if s_max > s_min:
        return (scores - s_min) / (s_max - s_min)
    return np.zeros_like(scores, dtype=float)


class PropertyMatchingService:
    def __init__(self, csv_path: str, chroma_path: str):
        print("---enter---")
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.df["_str_id"] = self.df["property_id"].astype(str)
        self._id_to_pos = {pid: idx for idx, pid in enumerate(self.df["_str_id"].tolist())}
        self._enriched_docs: List[str] = [build_enriched_text(row) for _, row in self.df.iterrows()]
        print("---after csv---")

        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_or_create_collection(name="properties_semantic")

        self._bm25: Optional[BM25Okapi] = None
        self._e5_model: Optional[SentenceTransformer] = None
        self._reranker: Optional[CrossEncoder] = None
        print("---final---")

    def available_deal_types(self) -> List[str]:
        """إيجار ولا تمليك حسب الداتا نفسها: عمود deal_type (rent / buy) لو موجود.
        الداتا الحالية مفيهاش العمود ده، والسعر فيها إيجار شهري، فكلها إيجار."""
        if "deal_type" in self.df.columns:
            found = {str(v).strip().lower() for v in self.df["deal_type"].dropna()} & {"rent", "buy"}
            if found:
                return sorted(found)
        return ["rent"]

    def _ensure_bm25(self):
        if self._bm25 is None:
            tokenized_docs = [doc.lower().split() for doc in self._enriched_docs]
            self._bm25 = BM25Okapi(tokenized_docs)

    def _ensure_e5(self):
        if self._e5_model is None:
            self._e5_model = SentenceTransformer(E5_MODEL_NAME, device="cpu")

    def _ensure_reranker(self):
        if self._reranker is None:
            self._reranker = CrossEncoder(RERANKER_MODEL_NAME, device="cpu")

    def _filter_dataframe(self, reqs: CustomerRequirements, ignore_rooms_and_area: bool = False) -> list:
        filtered = self.df.copy()

        if reqs.budget_max and reqs.budget_max > 0:
            filtered = filtered[filtered["price"] <= (reqs.budget_max * threshold("matching.budget_max_slack"))]
        if reqs.budget_min and reqs.budget_min > 0:
            filtered = filtered[filtered["price"] >= (reqs.budget_min * threshold("matching.budget_min_slack"))]

        if not ignore_rooms_and_area:
            if reqs.bedrooms and "pf_bedrooms" in filtered.columns:
                filtered = filtered[filtered["pf_bedrooms"] == reqs.bedrooms]

            if reqs.area_sqm and "pf_area_sqm" in filtered.columns:
                filtered = filtered[
                    (filtered["pf_area_sqm"] >= (reqs.area_sqm * (1 - AREA_TOLERANCE)))
                    & (filtered["pf_area_sqm"] <= (reqs.area_sqm * (1 + AREA_TOLERANCE)))
                ]

        return filtered["_str_id"].tolist()

    def search_properties(self,reqs: CustomerRequirements = None, Purpose: str = "", min_results_threshold: int = 1,top_k: Optional[int] = None ) -> dict:
        if Purpose == "general_browse":
            clean_df = self.df.drop(columns=["_str_id"], errors="ignore")
            return {
                "total_found": len(clean_df),
                "used_fallback": False,
                "properties": clean_df.copy().to_dict(orient="records"),
            }

        print(
            f"\n📥 [Input Req] Location: '{reqs.location}' | Budget: {reqs.budget_min}-{reqs.budget_max} "
            f"| Rooms: {reqs.bedrooms} | Area: {reqs.area_sqm} | Type: {reqs.property_type}"
        )
        print("🔍 [Attempt 1] جاري الفلترة الصارمة...")

        candidate_ids = self._filter_dataframe(reqs, ignore_rooms_and_area=False)
        results = self._rank_by_location_and_area(reqs, candidate_ids, reqs.area_sqm, top_k)

        used_fallback = False
        if len(results) < min_results_threshold:
            used_fallback = True
            print(f"⚠️ النتائج غير كافية ({len(results)} عقارات). جاري تطبيق الـ Fallback Strategy...")
            print("🔄 [Attempt 2] الحفاظ على السعر والمكان، مع التنازل عن الغرف والمساحة...")

            fallback_candidate_ids = self._filter_dataframe(reqs, ignore_rooms_and_area=True)
            results = self._rank_by_location_and_area(reqs, fallback_candidate_ids, target_area=None, top_k=top_k)

        print(f"✅ {len(results)} matching properties (Fallback Used: {used_fallback})")
        return {"total_found": len(results), "used_fallback": used_fallback, "properties": results}

    def _build_query_text(self, reqs: CustomerRequirements) -> str:
        parts = []
        if reqs.location:
            parts.append(str(reqs.location).strip())
        if reqs.property_type:
            parts.append(str(reqs.property_type).strip())
        if reqs.bedrooms:
            parts.append(f"{reqs.bedrooms} rooms")
        if reqs.budget_max:
            parts.append(f"price {float(reqs.budget_max):,.0f}")
        return " | ".join(parts)

    def _get_chroma_embs_for_ids(self, ids: List[str]) -> Dict[str, np.ndarray]:
        if not ids or self.collection.count() == 0:
            return {}
        res = self.collection.get(ids=ids, include=["embeddings"])
        out: Dict[str, np.ndarray] = {}
        got_ids = res.get("ids") or []
        got_embs = res.get("embeddings")
        if got_embs is not None:
            for pid, emb in zip(got_ids, got_embs):
                if emb is not None:
                    arr = np.asarray(emb, dtype=np.float32)
                    norm = np.linalg.norm(arr)
                    if norm > 0:
                        arr = arr / norm
                    out[str(pid)] = arr
        return out

    def _hybrid_rerank_ids(self, query_text: str, candidate_ids: List[str], top_n: int = 15) -> List[str]:
        valid_ids = [pid for pid in candidate_ids if pid in self._id_to_pos]
        if not valid_ids or not query_text.strip():
            return candidate_ids

        
        if len(valid_ids) == 1:
            return valid_ids

        self._ensure_e5()
        self._ensure_bm25()
        self._ensure_reranker()

        q_emb = self._e5_model.encode([query_text], normalize_embeddings=True)[0].astype(np.float32)
        emb_map = self._get_chroma_embs_for_ids(valid_ids)

        dense_scores = np.zeros(len(valid_ids), dtype=float)
        for i, pid in enumerate(valid_ids):
            if pid in emb_map:
                dense_scores[i] = float(np.dot(emb_map[pid], q_emb))

        all_bm25 = self._bm25.get_scores(query_text.lower().split())
        bm25_scores = np.array([float(all_bm25[self._id_to_pos[pid]]) for pid in valid_ids], dtype=float)

        hybrid_scores = (threshold("matching.dense_weight") * _normalize_scores(dense_scores)
                         + threshold("matching.bm25_weight") * _normalize_scores(bm25_scores))
        order = np.argsort(-hybrid_scores)[: min(top_n, len(valid_ids))]
        cand_ids = [valid_ids[i] for i in order]

        pairs = [[query_text, self._enriched_docs[self._id_to_pos[pid]]] for pid in cand_ids]
        rerank_scores = self._reranker.predict(pairs, batch_size=64, show_progress_bar=False)

        reranked_ids = [pid for pid, _ in sorted(zip(cand_ids, rerank_scores), key=lambda x: x[1], reverse=True)]
        reranked_set = set(reranked_ids)
        remaining = [pid for pid in valid_ids if pid not in reranked_set]
        return reranked_ids + remaining

    def _rank_by_location_and_area(self,reqs: CustomerRequirements,candidate_ids: list,target_area: float = None,top_k: Optional[int] = None) -> list:
        if not candidate_ids:
            return []

        candidates_df = self.df[self.df["_str_id"].isin(candidate_ids)].copy()
        if candidates_df.empty:
            return []

        location_query = reqs.location
        if location_query:
            keywords = location_keywords(location_query)
            if keywords:
                in_area = candidates_df["location"].fillna("").str.contains("|".join(keywords), case=False, regex=True)
                filtered_loc_df = candidates_df[in_area]
                if not filtered_loc_df.empty:
                    candidates_df = filtered_loc_df

        query_text = self._build_query_text(reqs)
        if query_text:
            ranked_ids = self._hybrid_rerank_ids(query_text, candidates_df["_str_id"].tolist(), top_n=15)
            rank_map = {pid: idx for idx, pid in enumerate(ranked_ids)}
            candidates_df["_rank_score"] = candidates_df["_str_id"].map(lambda x: rank_map.get(x, 10**9))
        else:
            candidates_df["_rank_score"] = 0

        if target_area:
            candidates_df["_diff"] = (candidates_df["pf_area_sqm"] - target_area).abs()
            candidates_df = candidates_df.sort_values(["_rank_score", "_diff", "price"]).drop(
                columns=["_rank_score", "_diff"]
            )
        else:
            candidates_df = candidates_df.sort_values(["_rank_score", "price"]).drop(columns=["_rank_score"])

        candidates_df = candidates_df.drop(columns=["_str_id"], errors="ignore")
        if top_k is not None:
            candidates_df = candidates_df.head(top_k)

        return candidates_df.to_dict(orient="records")

    def get_property_by_id(self, property_id: str) -> dict:
        matched = self.df[self.df["property_id"].astype(str) == str(property_id)]
        if not matched.empty:
            row_dict = matched.iloc[0].to_dict()
            row_dict.pop("_str_id", None)
            return row_dict
        return None