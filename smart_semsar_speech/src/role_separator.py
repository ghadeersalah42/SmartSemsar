"""
Broker vs. Client Role Separation Engine
Accurately separates and classifies speakers in Egyptian real estate calls into
Broker (سمسار / وكيل عقاري) and Client (عميل / مشتري).
"""

import re
from typing import List, Dict, Any, Tuple, Optional

try:
    from config import BROKER_INDICATORS, CLIENT_INDICATORS
except ImportError:
    try:
        from ..config import BROKER_INDICATORS, CLIENT_INDICATORS
    except ImportError:
        # Fallback default Egyptian indicators
        BROKER_INDICATORS = [
            "معاك", "مع حضرتك", "شركة", "المستشار العقاري", "مستشارك العقاري",
            "سمسار", "الوكيل العقاري", "أهلاً بحضرتك", "أقدر أساعدك", "كمبوند",
            "المطور", "سعر المتر", "أنظمة السداد", "المقدم", "أقساط", "استلام فوري"
        ]
        CLIENT_INDICATORS = [
            "أنا بدور على", "كنت بسأل عن", "عايز شقة", "عايز فيلا", "محتاج",
            "ميزانيتي", "البادجت", "مليون", "3 غرف", "قريبة من المدارس", "مش عايز"
        ]

try:
    from .text_normalization import normalize_arabic_text
except ImportError:
    from text_normalization import normalize_arabic_text


class RoleSeparator:
    """Classifies speaker clusters into Broker and Client roles."""

    def __init__(self, broker_keywords: Optional[List[str]] = None, client_keywords: Optional[List[str]] = None):
        self.broker_keywords = [
            normalize_arabic_text(k) for k in (broker_keywords or BROKER_INDICATORS)
        ]
        self.client_keywords = [
            normalize_arabic_text(k) for k in (client_keywords or CLIENT_INDICATORS)
        ]

    def score_text(self, text: str) -> Dict[str, Any]:
        """Calculates broker and client keyword scores for a given text."""
        norm_text = normalize_arabic_text(text)
        
        broker_hits = []
        broker_score = 0
        for kw in self.broker_keywords:
            matches = len(re.findall(re.escape(kw), norm_text))
            if matches > 0:
                broker_hits.append(kw)
                weight = 2.0 if any(intro in kw for intro in ["معاك", "شركة", "مستشار", "اهلا بحضرتك", "اقدر اساعدك"]) else 1.0
                broker_score += matches * weight
                
        client_hits = []
        client_score = 0
        for kw in self.client_keywords:
            matches = len(re.findall(re.escape(kw), norm_text))
            if matches > 0:
                client_hits.append(kw)
                weight = 2.0 if any(need in kw for need in ["انا بدور", "ميزانيتي", "البادجت", "عايز", "عاوز", "محتاج"]) else 1.0
                client_score += matches * weight
                
        return {
            "broker_score": broker_score,
            "client_score": client_score,
            "broker_hits": broker_hits,
            "client_hits": client_hits
        }

    def separate_roles(
        self,
        conversation: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Separates speakers into 'broker' and 'client' across all conversation turns.
        """
        speaker_texts: Dict[str, List[str]] = {}
        for turn in conversation:
            spk = turn.get("speaker")
            text = turn.get("text", "")
            if spk is not None:
                speaker_texts.setdefault(spk, []).append(text)
                
        speakers = list(speaker_texts.keys())
        if not speakers:
            return {
                "role_mapping": {},
                "confidence": 0.0,
                "speaker_scores": {},
                "assigned_turns": conversation
            }
            
        if len(speakers) == 1:
            spk = speakers[0]
            score_res = self.score_text(" ".join(speaker_texts[spk]))
            role = "broker" if score_res["broker_score"] >= score_res["client_score"] else "client"
            return {
                "role_mapping": {spk: role},
                "confidence": 0.6,
                "speaker_scores": {spk: score_res},
                "assigned_turns": [
                    {**turn, "role": role} for turn in conversation
                ]
            }
            
        speaker_scores = {}
        for spk in speakers:
            combined_text = " ".join(speaker_texts[spk])
            speaker_scores[spk] = self.score_text(combined_text)
            
        spk_a, spk_b = speakers[0], speakers[1]
        
        # Hypothesis 1: spk_a is Broker, spk_b is Client
        h1_score = speaker_scores[spk_a]["broker_score"] + speaker_scores[spk_b]["client_score"]
        
        # Hypothesis 2: spk_a is Client, spk_b is Broker
        h2_score = speaker_scores[spk_a]["client_score"] + speaker_scores[spk_b]["broker_score"]
        
        if h1_score >= h2_score:
            broker_speaker = spk_a
            client_speaker = spk_b
            best_diff = h1_score - h2_score
            total_score = max(1.0, h1_score + h2_score)
        else:
            broker_speaker = spk_b
            client_speaker = spk_a
            best_diff = h2_score - h1_score
            total_score = max(1.0, h1_score + h2_score)
            
        role_mapping = {
            broker_speaker: "broker",
            client_speaker: "client"
        }
        
        confidence = min(1.0, 0.5 + (best_diff / (2.0 * total_score)) * 0.5)
        
        assigned_turns = []
        for turn in conversation:
            spk = turn.get("speaker")
            role = role_mapping.get(spk, "unknown")
            assigned_turns.append({
                **turn,
                "role": role
            })
            
        return {
            "role_mapping": role_mapping,
            "broker_speaker": broker_speaker,
            "client_speaker": client_speaker,
            "confidence": round(confidence, 3),
            "speaker_scores": speaker_scores,
            "assigned_turns": assigned_turns
        }
