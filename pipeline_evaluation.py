import os
# from backend.services.property_service import search_properties
from backend.services.google_calendar_service import is_google_calendar_authenticated
from backend.services.MatchingFeature import PropertyMatchingService, CustomerRequirements
from backend.services.user_input_text import extract_requirements

def run_evaluation():
    print("="*50)
    print("🚀 SmartSemsar Evaluation Pipeline Report")
    print("="*50)
    matcher = PropertyMatchingService(csv_path="data/final_merged_dataset.csv",chroma_path="data/chroma_db")
    
    test_cases = [
        "عايزة شقة في التجمع الخامس مساحتها 180 متر،3 غرف، ميزانيتي 70 الف جنيه",
        "عايزة شقة في الشيخ زايد 10 غرف، مساحتها 400 متر، ميزانيتي 5 مليون جنيه"
    ]
    
    successful_searches = 0
    for i, reqs in enumerate(test_cases,1):
        print(f"\n--- Test Case {i} ---")
        print(f"💬 Prompt: \"{reqs}\"")
        resumed_text = reqs.strip() if isinstance(reqs, str) else None
        
        req = extract_requirements(call_transcript=None, raw_text=resumed_text)
        res = matcher.search_properties(req, "search_by_reqs")
        
        if isinstance(res, dict):
            found = res.get("total_found", 0)
            fallback = res.get("used_fallback", False)
        elif isinstance(res, list):
            found = len(res)
            fallback = False
        else:
            found = 0
            fallback = False

        print(f"Test Case {i}: Found {found} properties | Fallback Triggered: {fallback}")
        if found > 0:
            successful_searches += 1
            
    search_score = (successful_searches / len(test_cases)) * 100
    print(f"\n📊 Property Matching Success Rate: {search_score}%")
    
    # 2. تقييم الجاهزية والتكامل (Integration & Auth Test)
    cal_auth = is_google_calendar_authenticated()
    print(f"📅 Google Calendar OAuth Integration: {'✅ Passed' if cal_auth else '❌ Failed'}")
    
    print("="*50)
    print("✨ Evaluation Completed Successfully!")

if __name__ == "__main__":
    run_evaluation()