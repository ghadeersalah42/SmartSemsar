import streamlit as st
from langgraph.types import Command
import os
from dotenv import load_dotenv, find_dotenv

import sys
import uuid
import time

LOADED_AT = time.time()     # app.py بيقارنه بتاريخ الملف عشان يعيد تحميله لو اتعدل

# استيراد خدمات تليجرام والتقسيط والتقويم: موجودة في smart_semsar_ui/services
# (بتستورد بعضها كـ services.xxx، فلازم فولدر smart_semsar_ui يبقى في الـ sys.path)
_UI_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "smart_semsar_ui"))
if _UI_DIR not in sys.path:
    sys.path.append(_UI_DIR)
from services.telegram_service import (send_property_report, send_telegram_message,
                                       find_chat_by_username, is_telegram_configured)
from services.property_service import calculate_installments
from services.google_calendar_service import create_viewing_event

import datetime

load_dotenv(find_dotenv())

# --- صفحة الفرش بعفش المستخدم (Furniture_multi_photo_edit.py) ---
import socket
import subprocess
import time
from pathlib import Path

FURNISH_SCRIPT = Path(__file__).resolve().parents[2] / "Furniture_multi_photo_edit.py"
FURNISH_PORT = int(os.environ.get("SMARTSEMSAR_FURNISH_PORT", "7861"))
FURNISH_URL = f"http://127.0.0.1:{FURNISH_PORT}"


def prop_field(prop, *names, default=None):
    """قيمة من بيانات الشقة، سواء كانت dict أو PropertyItem (أول اسم ليه قيمة)"""
    for name in names:
        value = prop.get(name) if isinstance(prop, dict) else getattr(prop, name, None)
        if value not in (None, "") and value == value:      # value == value: مش NaN
            return value
    return default


# --- الجولات الـ 3D بتتعرض من سيرفر صغير، فالصفحة (لحد 6 ميجا) مش بتتبعت للمتصفح مع كل ضغطة ---
import functools
import http.server
import threading
from urllib.parse import quote

MODELS_DIR = Path(__file__).resolve().parents[2] / "data" / "models"
MODELS_PORT = int(os.environ.get("SMARTSEMSAR_MODELS_PORT", "7862"))
VOICE_DIR = Path(__file__).resolve().parents[2] / "data" / "raw_calls"     # تسجيلات المستخدم من الشات
DOWN_PAYMENT_PERCENT = 15      # مقدم التمليك (من سعر الشقة)؛ الباقي بيتقسط على المدة اللي العميل يختارها
DEPOSIT_MONTHS = 2             # تأمين الإيجار: إيجار شهرين مقدم، بيرجع آخر مدة العقد لو مفيش تلفيات
VOICE_DIR.mkdir(parents=True, exist_ok=True)


@st.cache_resource
def start_models_server():
    """بيشغل السيرفر مرة واحدة بس طول ما Streamlit شغال"""
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass
    handler = functools.partial(Quiet, directory=str(MODELS_DIR))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", MODELS_PORT), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def walkthrough_url(path):
    """لينك الجولة؛ ?t= بيتغير لما الملف يتبني من جديد، فالمتصفح بيجيب النسخة الجديدة.
    None لو الملف مش في data/models (السيرفر مش بيشوف غيره)"""
    path = Path(path).resolve()
    if path.parent != MODELS_DIR:
        return None
    start_models_server()
    return f"http://127.0.0.1:{MODELS_PORT}/{quote(path.name)}?t={int(path.stat().st_mtime)}"


def furnish_app_ready():
    """صفحة الفرش شغالة؟ (حد بيرد على البورت بتاعها)"""
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", FURNISH_PORT)) == 0


def start_furnish_app():
    """بيشغل صفحة الفرش في الخلفية مرة واحدة بس، واللوج بيتكتب في out/_furnish_app.log"""
    if furnish_app_ready():
        return
    proc = st.session_state.get("furnish_proc")
    if proc is not None and proc.poll() is None:      # لسه بتقوم
        return
    log = FURNISH_SCRIPT.parent / "out" / "_furnish_app.log"
    log.parent.mkdir(exist_ok=True)
    st.session_state["furnish_proc"] = subprocess.Popen(
        [sys.executable, str(FURNISH_SCRIPT), "--from-app"], cwd=FURNISH_SCRIPT.parent,
        stdout=open(log, "w", encoding="utf-8"), stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"})

def open_property_3d(orchestrator, config, property_id):
    """زرار "معاينة ثلاثية الأبعاد": لو الجراف واقف مستني اختيار (select_property) نكمله بالـ id،
    ولو البحث خلص عند match_properties (مفيش حد مستني) نسجل الاختيار كأنه طالع من select_property
    فالجراف يكمل على staging_3d على طول"""
    state = orchestrator.get_state(config)
    if state.tasks and any(task.interrupts for task in state.tasks):
        payload = Command(resume=property_id)
    else:
        from backend.agents.matching_node import property_item_by_id
        orchestrator.update_state(config, {
            "selected_property_id": str(property_id),
            "selected_property": property_item_by_id(property_id),
            "current_stage": "property_selected",
        }, as_node="select_property")
        payload = None
    for _ in orchestrator.stream(payload, config=config):
        pass


def results_reply(values):
    """رد البوت بعد البحث بدل "تم تنفيذ الطلب بنجاح": عدد النتايج، أو ليه مفيش"""
    if values.get("macro_intent") == "general_browse":
        return "دي كل الشقق المتاحة عندنا 👇 دوس «معاينة ثلاثية الأبعاد» على اللي تعجبك."
    found = values.get("top_properties") or []
    if found:
        return f"لقيتلك {len(found)} شقق مطابقة لطلبك 👇 دوس «معاينة ثلاثية الأبعاد» على اللي تعجبك."
    reqs = values.get("customer_reqs")
    if getattr(reqs, "deal_type", None) == "buy":
        return "حالياً كل الشقق المتاحة عندنا إيجار بس، مفيش شقق تمليك. تحب أدورلك على إيجار بنفس المواصفات؟"
    asked = []
    if getattr(reqs, "location", None):
        asked.append(reqs.location)
    if getattr(reqs, "bedrooms", None):
        asked.append(f"{reqs.bedrooms} غرف")
    if getattr(reqs, "area_sqm", None):
        asked.append(f"{reqs.area_sqm:,.0f} متر")
    if getattr(reqs, "budget_min", None):
        asked.append(f"من {reqs.budget_min:,.0f} جنيه")
    if getattr(reqs, "budget_max", None):
        asked.append(f"لحد {reqs.budget_max:,.0f} جنيه")
    wanted = f" ({'، '.join(asked)})" if asked else ""
    return (f"للأسف مفيش شقة بتحقق كل طلباتك{wanted}. "
            "جرّب تغيّر حاجة واحدة، زي المساحة أو الميزانية أو المنطقة.")


def render_telegram_link():
    """صندوق ربط تليجرام: كود مؤقت -> المستخدم يفتح البوت ويدوس Start -> نتأكد من تليجرام ونحفظ الـ chat id"""
    if st.session_state.get("telegram_chat_id"):
        st.success("✅ تليجرام متربط، التقرير هيوصلك هناك.")
        return

    configured, _ = is_telegram_configured()
    if not configured:
        st.warning("⚠️ بوت تليجرام مش متظبط: حطي TELEGRAM_BOT_TOKEN في ملف .env وشغلي الأبلكيشن تاني.")
        return

    with st.container(border=True):
        st.markdown("**📲 اكتب يوزر التليجرام بتاعك عشان يوصلك تقرير الحجز**")
        with st.form("telegram_user_form", border=False):
            col_user, col_link = st.columns([3, 1], vertical_alignment="bottom")
            with col_user:
                username = st.text_input("يوزر التليجرام", placeholder="@username", label_visibility="collapsed")
            with col_link:
                linked = st.form_submit_button("ربط", use_container_width=True)
        if not linked:
            return
        res = find_chat_by_username(username)
        if res.get("success"):
            st.session_state["telegram_chat_id"] = res["chat_id"]
            send_telegram_message(res["chat_id"], "✅ تم ربط حسابك بـ Smart Semsar، تقرير الحجز هيوصلك هنا.")
            st.rerun()
        elif res.get("needs_start"):
            # قيد من تليجرام: البوت مايقدرش يكلم حد غير لما يفتحه مرة واحدة بس
            bot = res.get("bot_username")
            st.warning(f"البوت لسه مايعرفش الحساب ده. افتح @{bot} في تليجرام ودوس **Start** مرة واحدة بس، "
                       "وبعدين ارجع دوس «ربط» تاني.")
            st.link_button("افتح البوت في تليجرام", f"https://t.me/{bot}")
        else:
            st.error(f"❌ {res.get('error')}")


def render_payment_details(price, is_rent, prop_info):
    """إيجار: الإيجار الشهري، التأمين (DEPOSIT_MONTHS شهور من الإيجار)، أول دفعة وإجمالي السنة.
    تمليك: المقدم DOWN_PAYMENT_PERCENT من السعر، والعميل بيختار يقسط الباقي على كام شهر"""
    if is_rent:
        st.markdown("### 🔑 تفاصيل الإيجار")
        months = DEPOSIT_MONTHS
        st.markdown(f"""
        - **الإيجار الشهري:** EGP {price:,.0f}
        - **التأمين (إيجار شهرين مقدم):** EGP {price * months:,.0f}، ويُسترد هذا المبلغ في نهاية مدة العقد في حال عدم وجود أي تلفيات.
        - **أول دفعة عند الاستلام (التأمين فقط):** EGP {price * months:,.0f}
        - **إجمالي الإيجار في السنة:** EGP {price * 12:,.0f}
        """)
        st.caption("الأرقام تقديرية؛ التأمين ومواعيد الدفع بتتحدد نهائي في العقد مع المالك.")
        return

    st.markdown("### 🧮 خطة التقسيط")
    down_pct = DOWN_PAYMENT_PERCENT
    st.markdown(f"**المقدم {down_pct}% من سعر الشقة:** EGP {price * down_pct / 100:,.0f}")
    months = st.selectbox("هتقسط الباقي على كام شهر؟", [12, 24, 36, 48, 60, 72, 84, 96, 108, 120], index=4,
                          format_func=lambda m: f"{m} شهر ({m // 12} سنين)" if m > 12 else "12 شهر (سنة)")
    plan = calculate_installments(price, down_payment_percent=down_pct, years=months / 12, annual_interest_rate=0.0)
    st.session_state["installment_plan"] = {**plan, "down_payment_percent": down_pct,
                                            "years": months // 12, "annual_interest_rate": 0.0}
    st.markdown(f"""
    - **المبلغ المتبقي بعد المقدم:** EGP {plan['remaining']:,.0f}
    - **القسط الشهري (على {months} شهر):** EGP {plan['monthly_installment']:,.0f} / شهرياً
    """)
    st.caption("الحساب من غير فوايد؛ خطط التقسيط الفعلية بتختلف من مطور للتاني.")



def deliver_booking_report(prop_info):
    """التقرير بيتبعت بعد الحجز بس ومرة واحدة. لو تليجرام مكانش متربط وقت الحجز، بيتبعت أول ما يتربط"""
    booking = st.session_state.get("booking_data")
    if not booking or booking.get("property_id") != str(prop_info["id"]) or st.session_state.get("report_sent"):
        return
    if not st.session_state.get("telegram_chat_id"):
        return          # صندوق اليوزر فوق، وأول ما يتربط التقرير بيتبعت
    ok, msg = send_report_to_telegram(prop_info)
    st.session_state["report_sent"] = ok
    (st.success if ok else st.warning)(msg)


def send_report_to_telegram(prop_info):
    """بيبعت تقرير الشقة (ومعاه ميعاد المعاينة لو اتحجز) لتليجرام العميل. بيرجع (نجح؟, رسالة للعرض)"""
    chat_id = st.session_state.get("telegram_chat_id")
    if not chat_id:
        return False, "⚠️ اربط حسابك على تليجرام الأول (فوق) عشان التقرير يتبعتلك."
    res = send_property_report(
        chat_id=chat_id, property_info=prop_info,
        customer_name=st.session_state.get("customer_name") or "Valued Customer",
        installment_plan=st.session_state.get("installment_plan") if prop_info.get("deal_type") == "buy" else None,
        booking_data=st.session_state.get("booking_data"))
    if res.get("success"):
        return True, "✅ تم إرسال التقرير لتليجرام بنجاح!"
    return False, f"❌ تعذر إرسال التقرير: {res.get('error')}"


def render_chat_flow():
    # --- تخصيص الـ CSS لترتيب السايدبار وجمال الواجهة بستايل ذهبي فخم ---
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;900&display=swap');

            /* 1. اتجاه الصفحة والخط العام */
            html, body, [data-testid="stAppViewContainer"] {
                direction: rtl;
                text-align: right;
                font-family: 'Tajawal', sans-serif !important;
            }

            /* خلفية التطبيق العامة بدرجة دافئة هادئة */
            [data-testid="stAppViewContainer"] {
                background: linear-gradient(135deg, #fbf9f5 0%, #f4ede2 100%);
            }
            /* الكلام غامق على الخلفية الفاتحة حتى لو المتصفح على الـ dark mode */
            [data-testid="stMain"], [data-testid="stMain"] h1, [data-testid="stMain"] h2,
            [data-testid="stMain"] h3, [data-testid="stMain"] p, [data-testid="stMain"] li,
            [data-testid="stMain"] label {
                color: #2c2c2c;
            }

            /* 2. تصميم السايدبار الشيك والراقي */
            section[data-testid="stSidebar"] {
                background-color: #ffffff !important;
                border-left: 1px solid #e8dec8 !important;
            }

            section[data-testid="stSidebar"] > div:first-child {
                display: flex;
                flex-direction: column;
                height: 100vh;
                position: relative;
                padding-top: 3.5rem !important;
                padding-bottom: 0px !important;
                overflow: hidden !important;
            }

            /* اسم البرنامج فوق، وخانة الكتابة والتسجيل تحت: خلفية ذهبي فاتح عشان يبانوا منفصلين عن الشات */
            .st-key-sb_header, .st-key-sb_footer {
                background: linear-gradient(135deg, #f8eed8 0%, #efdcb2 100%);
                border: 1px solid #e2cb98;
                border-radius: 16px;
                padding: 10px 12px;
                box-shadow: 0 2px 10px rgba(170, 119, 28, 0.10);
            }
            .st-key-sb_footer [data-testid="stForm"] {
                background: #fffdf8;
                border-color: #e2cb98;
            }
            .st-key-sb_footer [data-testid="stAudioInput"] > div {
                background: #fffdf8;
                border-radius: 12px;
            }

            /* 3. ضبط مساحة الصفحة الرئيسية على الشمال */
            div[data-testid="stAppViewContainer"] > section:nth-child(2) {
                margin-right: 21rem !important;
                margin-left: 0 !important;
            }

            /* 4. تصميم رسائل العميل بلمسة أنيقة */
            div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
                background-color: #fffdf9 !important;
                border: 1px solid #e8dec8 !important;
                border-radius: 16px 16px 4px 16px !important;
                padding: 12px;
                margin-bottom: 8px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02);
            }

            /* 5. تصميم رسائل المساعد */
            div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) {
                background-color: #f7f3eb !important;
                border: 1px solid #dfd2bc !important;
                border-radius: 16px 16px 16px 4px !important;
                padding: 12px;
                margin-bottom: 8px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02);
            }

            /* تنسيق أزرار المعاينة الذهبية */
            .stButton > button {
                background: linear-gradient(135deg, #d4af37 0%, #aa771c 100%) !important;
                color: #ffffff !important;
                font-weight: 700 !important;
                border-radius: 10px !important;
                border: none !important;
                padding: 8px 16px !important;
                box-shadow: 0 4px 15px rgba(212, 175, 55, 0.3) !important;
                transition: all 0.3s ease !important;
            }
            .stButton > button:hover {
                background: linear-gradient(135deg, #e6c547, #bc8420) !important;
                box-shadow: 0 6px 20px rgba(212, 175, 55, 0.5) !important;
                transform: translateY(-2px);
            }
        </style>
    """, unsafe_allow_html=True)

    # 1. تحميل الجراف
    @st.cache_resource
    def load_orchestrator():
        from backend.agents.semsar_graph import the_Orchestrator
        return the_Orchestrator

    try:
        the_Orchestrator = load_orchestrator()
    except Exception as e:
        st.error(f"❌ خطأ في تحميل الجراف أو موديلات الباك إند: {e}")
        st.stop()

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "browse_page" not in st.session_state:
        st.session_state.browse_page = 0

    if "show_checkout_page" not in st.session_state:
        st.session_state.show_checkout_page = False

    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    # ==================== 📱 الـ SIDEBAR ====================
    with st.sidebar:
        with st.container(key="sb_header"):      # اسم البرنامج و➕: لون مختلف عن الشات
            col_title, col_btn = st.columns([3, 1], vertical_alignment="center")
            with col_title:
                st.markdown("<h3 style='color: #aa771c; margin:0; font-weight: 700;'>Smart Semsar</h3>", unsafe_allow_html=True)
            with col_btn:
                st.markdown("""
                    <style>
                    div[data-testid="stSidebar"] button[kind="primary"] {
                        background: linear-gradient(135deg, #d4af37 0%, #aa771c 100%) !important;
                        color: white !important;
                        border: 1px solid rgba(255, 255, 255, 0.4) !important;
                        border-radius: 12px !important;
                        box-shadow: 0 4px 15px rgba(212, 175, 55, 0.3) !important;
                        font-weight: bold !important;
                    }
                    div[data-testid="stSidebar"] button[kind="primary"]:hover {
                        background: linear-gradient(135deg, #e6c547, #bc8420) !important;
                        transform: translateY(-2px);
                        box-shadow: 0 6px 20px rgba(212, 175, 55, 0.5) !important;
                    }
                    </style>
                """, unsafe_allow_html=True)
            
                if st.button("➕", help="بحث عقاري جديد", type="primary"):
                    st.session_state.messages = []
                    st.session_state.thread_id = str(uuid.uuid4())
                    st.session_state.browse_page = 0
                    st.session_state.show_checkout_page = False
                    st.session_state.pop("selected_property_id", None)
                    st.session_state.pop("booking_data", None)
                    st.session_state.pop("report_sent", None)
                    st.session_state.pop("show_furnish_page", None)
                    st.session_state.show_installment_plan = False
                    st.rerun()

        # عرض الرسائل بانتظام
        chat_container = st.container(height=400)      # سايب مكان لزرار التسجيل تحت
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        with st.container(key="sb_footer"):      # خانة الكتابة والتسجيل: لون مختلف عن الشات
            # صندوق إدخال الرسائل
            with st.form(key="chat_form", clear_on_submit=True):
                col_input, col_btn_send = st.columns([4, 1])
                with col_input:
                    user_query = st.text_input("اكتب طلبك", placeholder="اكتب طلبك هنا...", label_visibility="collapsed")
                with col_btn_send:
                    submit_button = st.form_submit_button("إرسال", use_container_width=True)

            # أو سجّل طلبك بصوتك: التسجيل بيتحول لنص ويمشي زي الكلام المكتوب بالظبط
            # (المفتاح بيتغير بعد كل تسجيل، فالتسجيل القديم مش بيتبعت تاني مع كل rerun)
            st.session_state.setdefault("voice_key", 0)
            voice_note = st.audio_input("سجّل طلبك بصوتك", key=f"voice_{st.session_state.voice_key}",
                                        label_visibility="collapsed")

    if voice_note is not None:
        st.session_state.voice_key += 1
        voice_path = VOICE_DIR / f"voice_{uuid.uuid4().hex[:8]}.wav"
        voice_path.write_bytes(voice_note.getvalue())
        with st.spinner("جاري تحويل التسجيل لنص..."):
            try:
                from backend.services.user_input_text import transcribe_voice_note
                voice_text = transcribe_voice_note(str(voice_path))
            except Exception as ex:
                voice_text = ""
                st.error(f"❌ تعذر تحويل التسجيل لنص: {ex}")
        if voice_text:
            user_query, submit_button = voice_text, True
        else:
            st.warning("⚠️ مسمعناش كلام واضح في التسجيل، جرّب تاني.")

    # ==================== 🖥️ MAIN AREA ====================
    current_state = the_Orchestrator.get_state(config)
    graph_values = current_state.values if hasattr(current_state, "values") else {}
    macro_intent = graph_values.get("macro_intent", None)

    if submit_button and user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.session_state.show_checkout_page = False
        with st.spinner("جاري المعالجة والبحث..."):
            try:
                if current_state.tasks and any(task.interrupts for task in current_state.tasks):
                    execution_input = Command(resume=user_query)
                else:
                    execution_input = {
                        "session_id": st.session_state.thread_id,
                        "user_input_type": "text",
                        "user_query": user_query,
                        "user_text": user_query,
                    }

                from langchain_core.tracers.context import collect_runs

                with collect_runs() as cb:
                    for event in the_Orchestrator.stream(execution_input, config=config):
                        pass
                    
                    # طباعة رابط الـ Trace في التيرمنال فوراً
                    if cb.traced_runs:
                        for run in cb.traced_runs:
                            # استخدام الرابط العام المباشر لمنصة LangSmith مع الـ Run ID
                            trace_url = f"https://smith.langchain.com/runs/{run.id}"
                            print(f"🔗 LangSmith Trace Link: {trace_url}")
                updated_state = the_Orchestrator.get_state(config)
                if updated_state.tasks and any(task.interrupts for task in updated_state.tasks):
                    interrupt_payload = updated_state.tasks[0].interrupts[0].value
                    if not (isinstance(interrupt_payload, dict) and interrupt_payload.get("type") in ["property_selection", "general_browse"]):
                        interrupt_msg = str(interrupt_payload) if not isinstance(interrupt_payload, dict) else interrupt_payload.get("message", "")
                        st.session_state.messages.append({"role": "assistant", "content": interrupt_msg})
                else:
                    st.session_state.messages.append({"role": "assistant", "content": results_reply(updated_state.values)})
                
                st.rerun()
            except Exception as ex:
                st.error(f"❌ حدث خطأ: {ex}")

    def extract_all_properties():
        if current_state.tasks and any(task.interrupts for task in current_state.tasks):
            int_val = current_state.tasks[0].interrupts[0].value
            if isinstance(int_val, dict) and int_val.get("properties"):
                return int_val.get("properties")

        from_graph = graph_values.get("top_properties") or graph_values.get("properties") or graph_values.get("filtered_properties")
        if from_graph:
            return from_graph

        # كل الشقق بتتعرض في التصفح العام بس؛ البحث اللي ملوش نتايج بيفضل فاضي (مش بيعرض كل الداتا)
        if macro_intent != "general_browse":
            selected = graph_values.get("selected_property")
            return [selected] if selected else []

        try:
            from backend.agents.matching_node import Property_matching_service
            return Property_matching_service.search_properties(Purpose="general_browse")["properties"]
        except Exception:
            return []

    all_props = extract_all_properties()
    has_interrupt = current_state.tasks and any(task.interrupts for task in current_state.tasks)
    # البوت سأل سؤال (المواصفات، إيجار ولا تمليك) ومستني الرد: الشقق متظهرش لحد ما يرد
    interrupt_value = current_state.tasks[0].interrupts[0].value if has_interrupt else None
    waiting_for_answer = has_interrupt and not (isinstance(interrupt_value, dict) and interrupt_value.get("properties"))
    model_state = graph_values.get("model_state")
    
    propID = (
        graph_values.get("selected_property_id") or 
        graph_values.get("property_id") or 
        graph_values.get("current_property_id") or 
        st.session_state.get("selected_property_id")
    )

    # ----------------------------------------------------
    # صفحة إكمال العملية (Checkout / Action Hub)
    # ----------------------------------------------------
    if st.session_state.get("show_checkout_page", False):
        st.markdown("## 📋 إكمال عملية الحجز والمعاينة")
        st.markdown("---")
        
        # استخراج بيانات الشقة الحالية لعرضها
        current_prop = None
        for p in all_props:
            p_id = prop_field(p, 'property_id', 'id')
            if str(p_id) == str(propID):
                current_prop = p
                break
        
        p_title = prop_field(current_prop, 'title', default='عقار فاخر')
        p_price = prop_field(current_prop, 'price', default=2500000)
        p_location = prop_field(current_prop, 'location', default='القاهرة الجديدة')
        p_bedrooms = prop_field(current_prop, 'pf_bedrooms', 'bedrooms', default=3)
        p_area = prop_field(current_prop, 'pf_area_sqm', 'area_sqm', default=180)

        # إيجار ولا تمليك: زي ما العميل قال؛ ولو مقالش، إيجار (كل الشقق اللي في الداتا إيجار)
        deal_type = getattr(graph_values.get("customer_reqs"), "deal_type", None)
        is_rent = deal_type != "buy"
        price_label = "الإيجار الشهري" if is_rent else "السعر الإجمالي"

        # صندوق بيانات الشقة
        st.markdown(f"""
            <div style="
                background: #ffffff;
                border: 1px solid #e8dec8;
                border-radius: 16px;
                padding: 24px;
                margin-bottom: 24px;
                box-shadow: 0 4px 15px rgba(180, 150, 100, 0.05);
            ">
                <h3 style="color: #aa771c; margin-top:0;">{p_title} (#{propID})</h3>
                <p><b>الموقع:</b> {p_location} &nbsp;|&nbsp; <b>المساحة:</b> {p_area} متر مربع &nbsp;|&nbsp; <b>عدد الغرف:</b> {p_bedrooms}</p>
                <h4 style="color: #2c2c2c;">{price_label}: EGP {p_price:,.0f}</h4>
            </div>
        """, unsafe_allow_html=True)

        prop_info = {"id": str(propID), "title": p_title, "price": p_price, "location": p_location,
                     "area_sqm": p_area, "bedrooms": p_bedrooms, "deal_type": "rent" if is_rent else "buy",
                     "deposit_months": DEPOSIT_MONTHS}

        # ---- ربط تليجرام: المستخدم بيفتح البوت ويدوس Start، وإحنا بناخد الـ chat id من تليجرام نفسه ----
        render_telegram_link()
        deliver_booking_report(prop_info)      # لو حجز قبل ما يربط، التقرير بيتبعت أول ما يربط

        st.markdown("### اختر الإجراء المطلوب:")


        col_btn1, col_btn2 = st.columns(2)

        with col_btn1:
            if st.button("1- تفاصيل الإيجار والتأمين" if is_rent else "1- خطة التقسيط", use_container_width=True):
                st.session_state["show_installment_plan"] = True

        with col_btn2:
            if st.button("2- حجز موعد معاينة", use_container_width=True):
                st.session_state["show_booking_form"] = not st.session_state.get("show_booking_form", False)

        if st.session_state.get("show_booking_form", False):
            st.markdown("---")
            st.markdown("### 📅 تحديد موعد المعاينة على Google Calendar")
            
            col_date, col_time = st.columns(2)
            with col_date:
                booking_date = st.date_input("اختر تاريخ المعاينة", min_value=datetime.date.today())
            with col_time:
                booking_time = st.time_input("اختر وقت المعاينة", value=datetime.time(14, 0))
                
            cust_name = st.text_input("اسم العميل", value=st.session_state.get("customer_name", ""))
            cust_phone = st.text_input("رقم الهاتف", value="01000000000")
            booking_notes = st.text_area("ملاحظات إضافية للمعاينة", value="معاينة الشقة مع السمسار")

            if st.button("تأكيد الحجز والإضافة للتقويم 🚀", type="primary"):
                if cust_name.strip():
                    st.session_state["customer_name"] = cust_name.strip()
                with st.spinner("جاري إنشاء الموعد على Google Calendar..."):
                    # استدعاء دالة الكاليندر الحقيقية
                    res = create_viewing_event(
                        property_info=prop_info,
                        customer_name=cust_name,
                        customer_phone=cust_phone,
                        appointment_date=booking_date.isoformat(),
                        appointment_time=booking_time.strftime("%H:%M"),
                        notes=booking_notes
                    )

                    if res.get("success"):
                        st.success("✅ تم حجز الموعد بنجاح وإضافته إلى Google Calendar الخاص بك!")
                        if res.get("html_link"):
                            st.markdown(f"[🔗 عرض الموعد في تقويم جوجل]({res.get('html_link')})")
                        st.session_state["show_booking_form"] = False

                        # العميل خلص: التقرير (ومعاه ميعاد المعاينة) بيتبعت لتليجرام لوحده
                        st.session_state["booking_data"] = {
                            "property_id": str(propID),
                            "date": booking_date.isoformat(),
                            "time": booking_time.strftime("%H:%M"),
                            "status": "confirmed",
                            "calendar_event_id": res.get("event_id"),
                            "calendar_link": res.get("html_link"),
                            "notes": booking_notes,
                        }
                        st.session_state["report_sent"] = False
                        deliver_booking_report(prop_info)
                    else:
                        st.error(f"❌ حدث خطأ أثناء الحجز: {res.get('error')}")
        # الإيجار: الإيجار الشهري والتأمين وأول دفعة. التمليك: مقدم ثابت والعميل بيختار مدة التقسيط
        if st.session_state.get("show_installment_plan", False):
            st.markdown("---")
            render_payment_details(p_price, is_rent, prop_info)

        st.markdown("---")
        if st.button("العودة إلى المعاينة ثلاثية الأبعاد 3D", type="secondary"):
            st.session_state.show_checkout_page = False
            st.rerun()

    # ----------------------------------------------------
    # صفحة الفرش بعفش العميل (Furniture_multi_photo_edit.py) جوه نفس الصفحة، وزرار يرجع للمعاينة والحجز
    # ----------------------------------------------------
    elif st.session_state.get("show_furnish_page") and model_state and getattr(model_state, "status", None) == "ready":
        # أدوات الفرش محتاجة مساحة أعرض من باقي الصفحات
        st.markdown("<style>.block-container, [data-testid='stMainBlockContainer'] {max-width: 1200px;}</style>",
                    unsafe_allow_html=True)

        def furnish_nav(where):
            """زرار الرجوع لصفحة المعاينة (فيها باقي الزراير) وزرار الحجز على طول"""
            col_back, col_book = st.columns(2)
            with col_back:
                if st.button("⬅️ رجوع للمعاينة والحجز", key=f"furnish_back_{where}", use_container_width=True):
                    st.session_state["show_furnish_page"] = False
                    st.rerun()
            with col_book:
                if st.button("اكمال العملية", key=f"furnish_checkout_{where}", use_container_width=True, type="primary"):
                    st.session_state["show_furnish_page"] = False
                    st.session_state.show_checkout_page = True
                    st.rerun()

        st.markdown("### 🛋️ فرش الشقة بعفشك")
        furnish_nav("top")

        start_furnish_app()
        if not furnish_app_ready():
            with st.spinner("صفحة الفرش بتجهز..."):
                for _ in range(120):
                    if furnish_app_ready():
                        break
                    time.sleep(1)
        if furnish_app_ready():
            # visit = محادثة البحث دي: صفحة الفرش بتبدأ فاضية لكل زيارة جديدة، ولو رجع ودخل تاني شغله بيفضل
            st.components.v1.iframe(f"{FURNISH_URL}/?property_id={propID}&visit={st.session_state.thread_id}",
                                    height=1400, scrolling=True)
        else:
            st.error("صفحة الفرش مقامتش، شوفي out/_furnish_app.log")

        st.markdown("---")
        furnish_nav("bottom")

    # ----------------------------------------------------
    # 1. الأولوية القصوى: عرض الـ 3D Staging لو الجولة جاهزة
    # ----------------------------------------------------
    elif model_state and getattr(model_state, "status", None) == "ready":
        walkthrough_path = getattr(model_state, "walkthrough_path", None)
        glb_path = getattr(model_state, "glb_path", None)
        
        if walkthrough_path and os.path.exists(walkthrough_path):
            try:
                url = walkthrough_url(walkthrough_path)
                if url:
                    st.components.v1.iframe(url, height=600, scrolling=True)
                else:
                    with open(walkthrough_path, "r", encoding="utf-8") as f:
                        st.components.v1.html(f.read(), height=600, scrolling=True)
            except Exception as e:
                st.error(f"❌ تعذر العرض: {e}")
                
        if glb_path and os.path.exists(glb_path):
            with open(glb_path, "rb") as file_to_download:
                st.download_button(
                    label="تحميل نموذج الـ 3D (GLB)",
                    data=file_to_download,
                    file_name=os.path.basename(glb_path),
                    mime="model/gltf-binary"
                )

        st.markdown("---")
        
        # الأزرار الأربعة تحت الـ 3D Staging
        col_rep, col_furnish, col_checkout, col_back_prop = st.columns(4)
        
        with col_rep:
            if st.button("عرض التقرير", use_container_width=True, type="primary"):
                st.session_state["current_page"] = "Report"
                st.info("صفحة التقرير قيد التجهيز، ترقبها قريباً!")
                
        with col_furnish:
            start_furnish_app()      # بتبدأ تقوم في الخلفية أول ما الـ 3D يظهر
            if st.button("فرش ومعاينة", use_container_width=True, type="primary"):
                st.session_state["show_furnish_page"] = True      # بتفتح جوه نفس الصفحة مش في تاب جديد
                st.rerun()

        with col_checkout:
            if st.button("اكمال العملية", use_container_width=True, type="primary"):
                st.session_state.show_checkout_page = True
                st.rerun()
        
        with col_back_prop:
            if st.button("اختيار شقة أخرى", use_container_width=True):
                # 1. إجبار الواجهة على الخروج من وضع الـ 3D والـ Checkout
                st.session_state.pop("selected_property_id", None)
                st.session_state.pop("show_furnish_page", None)
                st.session_state.show_checkout_page = False
                st.session_state.show_installment_plan = False
                st.session_state.show_booking_form = False
                st.session_state["force_property_selection"] = True
                try:
                    the_Orchestrator.update_state(config, {
                        "model_state": None,
                        "selected_property_id": None,
                        "micro_intent": "property_selection"
                    })
                except Exception:
                    pass
                
                st.rerun()

    # ----------------------------------------------------
    # 2. حالة الـ Selection أو الـ Interrupt (اختيار شقة معينة)
    # ----------------------------------------------------
    elif not waiting_for_answer and (has_interrupt or macro_intent in ["property_selection", "recommendation", "search_by_reqs"]):
        st.markdown("### العقارات المقترحة والمطابقة لطلبك")

        if not all_props:
            st.info(results_reply(graph_values))
        if all_props:
            for idx, prop in enumerate(all_props):
                p_id = prop_field(prop, 'property_id', 'id', default=idx)
                p_title = prop_field(prop, 'title', default='عقار فاخر')
                p_price = prop_field(prop, 'price', default=0)
                p_location = prop_field(prop, 'location', default='مصر')
                p_bedrooms = prop_field(prop, 'pf_bedrooms', 'bedrooms', default=3)
                p_area = prop_field(prop, 'pf_area_sqm', 'area_sqm', default=180)

                st.markdown(f"""
                    <div style="
                        background: #ffffff;
                        border: 1px solid #e8dec8;
                        border-radius: 16px;
                        padding: 20px 24px;
                        margin-bottom: 12px;
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        box-shadow: 0 6px 20px rgba(180, 150, 100, 0.06);
                    ">
                        <div>
                            <div style="font-size: 0.75rem; color: #8c7b65; margin-bottom: 4px; font-weight: 600;">رقم المعرف: #{p_id}</div>
                            <div style="font-size: 1.2rem; font-weight: 700; color: #2c2c2c; margin-bottom: 6px;">{p_title}</div>
                            <div style="color: #6c5d47; font-size: 0.9rem; margin-bottom: 8px;">الموقع: {p_location}</div>
                            <div style="font-size: 0.85rem; color: #6b7280; font-weight: 500;">
                                عدد الغرف: {p_bedrooms} &nbsp;&nbsp;|&nbsp;&nbsp; المساحة: {p_area} متر مربع
                            </div>
                        </div>
                        <div style="text-align: left;">
                            <div style="font-size: 1.25rem; font-weight: 800; color: #aa771c; margin-bottom: 10px;">EGP {p_price:,.0f}</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                col_space, col_btn = st.columns([4, 1])
                with col_btn:
                    if st.button("معاينة ثلاثية الأبعاد", key=f"sel_prop_{p_id}_{idx}", use_container_width=True):
                        st.session_state["selected_property_id"] = p_id
                        
                        with st.spinner("جاري معالجة الاختيار وتوليد نموذج الـ 3D..."):
                            try:
                                open_property_3d(the_Orchestrator, config, p_id)
                                st.rerun()
                            except Exception as ex:
                                st.error(f"❌ حدث خطأ: {ex}")

    # ----------------------------------------------------
    # 3. حالة الـ General Browse
    # ----------------------------------------------------
    elif macro_intent == "general_browse":
        st.markdown("### كافة العقارات المتاحة للتصفح")
        if all_props:
            ITEMS_PER_PAGE = 6
            total_props = len(all_props)
            total_pages = (total_props + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE if total_props > 0 else 1

            start_idx = st.session_state.browse_page * ITEMS_PER_PAGE
            end_idx = min(start_idx + ITEMS_PER_PAGE, total_props)
            current_page_props = all_props[start_idx:end_idx]

            for idx, prop in enumerate(current_page_props):
                global_idx = start_idx + idx
                p_id = prop_field(prop, 'property_id', 'id', default=global_idx)
                p_title = prop_field(prop, 'title', default='عقار فاخر')
                p_price = prop_field(prop, 'price', default=0)
                p_location = prop_field(prop, 'location', default='مصر')
                p_bedrooms = prop_field(prop, 'pf_bedrooms', 'bedrooms', default=3)
                p_area = prop_field(prop, 'pf_area_sqm', 'area_sqm', default=180)

                st.markdown(f"""
                    <div style="
                        background: #ffffff;
                        border: 1px solid #e8dec8;
                        border-radius: 16px;
                        padding: 20px 24px;
                        margin-bottom: 12px;
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        box-shadow: 0 6px 20px rgba(180, 150, 100, 0.06);
                    ">
                        <div>
                            <div style="font-size: 0.75rem; color: #8c7b65; margin-bottom: 4px; font-weight: 600;">رقم المعرف: #{p_id}</div>
                            <div style="font-size: 1.2rem; font-weight: 700; color: #2c2c2c; margin-bottom: 6px;">{p_title}</div>
                            <div style="color: #6c5d47; font-size: 0.9rem; margin-bottom: 8px;">الموقع: {p_location}</div>
                            <div style="font-size: 0.85rem; color: #6b7280; font-weight: 500;">
                                عدد الغرف: {p_bedrooms} &nbsp;&nbsp;|&nbsp;&nbsp; المساحة: {p_area} متر مربع
                            </div>
                        </div>
                        <div style="text-align: left;">
                            <div style="font-size: 1.25rem; font-weight: 800; color: #aa771c; margin-bottom: 10px;">EGP {p_price:,.0f}</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                col_space, col_btn = st.columns([4, 1])
                with col_btn:
                    if st.button("معاينة ثلاثية الأبعاد", key=f"gb_p_{global_idx}", use_container_width=True):
                        with st.spinner("جاري تجهيز جولة المعاينة..."):
                            try:
                                st.session_state["selected_property_id"] = p_id
                                open_property_3d(the_Orchestrator, config, p_id)
                                st.rerun()
                            except Exception as ex:
                                st.error(f"❌ حدث خطأ: {ex}")

            if total_pages > 1:
                st.markdown("---")
                col_prev, col_info, col_next = st.columns([1, 2, 1], vertical_alignment="center")
                with col_prev:
                    if st.session_state.browse_page > 0:
                        if st.button("الصفحة السابقة", use_container_width=True):
                            st.session_state.browse_page -= 1
                            st.rerun()
                with col_info:
                    st.markdown(f"<p style='text-align: center; margin:0;'>صفحة <b>{st.session_state.browse_page + 1}</b> من <b>{total_pages}</b></p>", unsafe_allow_html=True)
                with col_next:
                    if st.session_state.browse_page < total_pages - 1:
                        if st.button("الصفحة التالية", use_container_width=True):
                            st.session_state.browse_page += 1
                            st.rerun()

    # ----------------------------------------------------
    # 4. الحالة الافتراضية (Home)
    # ----------------------------------------------------
    else:
        col1, col2 = st.columns([1, 6])
        with col1:
            st.markdown("#")
        with col2:
            st.title("Smart Semsar AI")
        
        st.markdown("""
        ### مرحبًا بك في منصة سمسار الذكية 🌟
        منصتك الذكية للبحث عن العقار المناسب ومعاينته تفاعلياً بتقنيات الـ 3D. 
        اكتب طلبك في القائمة الجانبية بالأسفل للبدء في رحلة البحث والعرض بكل سهولة واحترافية.
        """)
        st.markdown("---")